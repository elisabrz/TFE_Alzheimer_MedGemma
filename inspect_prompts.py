"""
inspect_prompts.py — Vérifie le contenu textuel des prompts construits par TfeDataset.

Permet de vérifier visuellement :
  1. Quelles features apparaissent dans le prompt (et leur formattage)
  2. Quelles features sont OMISES parce qu'imputées (signal_in_prompt: false)
  3. Comment les valeurs imputées sont marquées (estimated) (signal_in_prompt: true)
  4. La structure complète du prompt (system + user + chat template MedGemma)

NE PAS afficher les images (~256 tokens chacune × 4 = ~1000 tokens illisibles
de bruit). Uniquement la partie textuelle du prompt + les métadonnées
d'imputation par sample.

Usage :
    cd /home/elisa/TFE_final
    python inspect_prompts.py                                     # 5 samples train
    python inspect_prompts.py --split val --n 10                  # 10 samples val
    python inspect_prompts.py --cohort NACC --n 3                 # 3 samples NACC
    python inspect_prompts.py --label AD --n 5                    # 5 sujets AD
    python inspect_prompts.py --subject-id 002_S_0413              # un sujet précis
    python inspect_prompts.py --output prompts_dump.txt           # sauver dans fichier
    python inspect_prompts.py --prompt-mode image_centric          # autre mode prompt
    python inspect_prompts.py --features-only                      # juste la section
                                                                  #  clinical_info, sans le wrapper

Sortie :
    Pour chaque sample sélectionné :
      ════════════════════════════════════════════════════════════
        Sample N — subject_id | source | label | n_real | n_imputed
      ════════════════════════════════════════════════════════════

      1. RAW VALUES IN CSV
         (chaque feature : valeur, flag imputed/real, présence dans le prompt)

      2. FORMATTED CLINICAL INFO
         (le bloc tel qu'inséré dans le prompt)

      3. FULL PROMPT (textuel uniquement, images symbolisées par [IMG])

Vérifications automatiques :
    - Compte les features qui apparaissent vs présentes dans le DataFrame
    - Vérifie cohérence entre flag _imputed et valeur attendue
    - Détecte les features qui auraient dû être omises mais sont présentes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR
sys.path.insert(0, str(PROJECT_ROOT))

# Imports tardifs pour permettre --help sans charger le processor
def _lazy_imports():
    from transformers import AutoProcessor
    from utils import load_config
    from dataset import TfeDataset, FEATURE_LABELS
    return AutoProcessor, load_config, TfeDataset, FEATURE_LABELS


# ═══════════════════════════════════════════════════════════════════════════
# Détection imputation par feature
# ═══════════════════════════════════════════════════════════════════════════

def detect_imputation_status(
    row: pd.Series, features: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Pour chaque feature, détecte si la valeur est réelle ou imputée.

    Logique en cascade :
      1. Si <feature>_imputed existe → flag direct (1 = imputé, 0 = réel)
      2. Si feature == 'mmse_score' et has_real_measures existe → flag indirect
      3. Sinon → "unknown" (pas de moyen de distinguer)

    Retourne dict[feature] -> {'value', 'is_imputed', 'is_nan', 'detection_method'}
    """
    status: Dict[str, Dict[str, Any]] = {}
    for feat in features:
        raw_value = row.get(feat, np.nan) if feat in row.index else np.nan
        is_nan = pd.isna(raw_value)

        imputed_col = f"{feat}_imputed"
        if imputed_col in row.index:
            flag = row[imputed_col]
            is_imputed = bool(int(flag) == 1) if not pd.isna(flag) else False
            method = f"col {imputed_col}={int(flag) if not pd.isna(flag) else 'NaN'}"
        elif feat == "mmse_score" and "has_real_measures" in row.index:
            has_real = row["has_real_measures"]
            is_imputed = bool(int(has_real) == 0) if not pd.isna(has_real) else True
            method = f"has_real_measures={int(has_real) if not pd.isna(has_real) else 'NaN'}"
        else:
            is_imputed = False  # unknown — assume not imputed
            method = "unknown (no flag column)"

        status[feat] = {
            "value":            raw_value,
            "is_imputed":       is_imputed,
            "is_nan":           is_nan,
            "detection_method": method,
        }
    return status


# ═══════════════════════════════════════════════════════════════════════════
# Extraction du prompt textuel sans tokens d'image
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_prompt(ds, idx: int) -> Tuple[str, str]:
    """
    Extrait le prompt textuel construit par TfeDataset pour l'échantillon `idx`.

    Méthode robuste : appelle ds[idx] (la VRAIE méthode du dataset) pour obtenir
    les input_ids tokenisés, puis décode en remplaçant les tokens d'image par
    des markers lisibles. Cette approche fonctionne quelle que soit la structure
    interne de dataset.py — pas de dépendance à _get_prompt_template ou autres
    méthodes privées.

    Retourne (clinical_info, full_prompt_text) :
      - clinical_info : juste la section "- AGE: 75 years\n..." (via _format_clinical_info)
      - full_prompt_text : le prompt complet décodé, avec [IMAGE×N] à la place
                           des patches d'image
    """
    row = ds.df.iloc[idx]

    # 1. clinical_info brut (méthode publique, toujours dispo)
    clinical_info = ds._format_clinical_info(row)

    # 2. Prompt complet via décodage des input_ids
    # On appelle la vraie méthode __getitem__ → on a EXACTEMENT ce que voit
    # le modèle pendant le training/inférence.
    try:
        sample = ds[idx]
        input_ids = sample["input_ids"]
        full_text = _decode_with_image_markers(ds.processor, input_ids)
    except Exception as e:
        full_text = (
            f"[Erreur lors de l'appel ds[{idx}]: {type(e).__name__}: {e}]\n\n"
            f"Section clinical_info disponible :\n{clinical_info}"
        )

    return clinical_info, full_text


def _decode_with_image_markers(processor, input_ids) -> str:
    """
    Décode input_ids en texte humainement lisible, en collapsant les longues
    séquences de tokens d'image (ils représentent les patches IRM et polluent
    la sortie sur ~1000 tokens).

    Stratégie :
      1. Décode token par token (skip_special_tokens=False pour garder
         <start_of_turn>, etc.)
      2. Identifie les tokens dont le nom contient 'image' / 'soft_token' /
         'patch' → les remplace par un compteur [IMAGE PATCHES × N]
      3. Reconstruit le texte
    """
    import torch
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.tolist()
    else:
        ids = list(input_ids)

    tokenizer = processor.tokenizer

    # Identifie les token IDs qui correspondent à des patches d'image.
    # MedGemma utilise <image_soft_token> répété pour chaque patch.
    # On les détecte par leur nom dans le vocab.
    image_token_ids = set()
    for tok_str, tok_id in tokenizer.get_vocab().items():
        ts_lower = tok_str.lower()
        if any(kw in ts_lower for kw in ["image_soft", "image_patch", "img_patch"]):
            image_token_ids.add(tok_id)
        # Pattern Gemma : <image_soft_token> est un token unique répété
        if tok_str == "<image_soft_token>":
            image_token_ids.add(tok_id)

    # Fallback : si on n'a pas identifié de token "image", on cherche
    # une séquence de tokens identiques très longue (heuristique).
    if not image_token_ids:
        # Compte les runs de tokens consécutifs identiques
        from collections import Counter
        runs: List[Tuple[int, int]] = []  # (token_id, count)
        prev = None
        count = 0
        for tid in ids:
            if tid == prev:
                count += 1
            else:
                if prev is not None and count >= 50:
                    runs.append((prev, count))
                prev = tid
                count = 1
        if prev is not None and count >= 50:
            runs.append((prev, count))
        # Tokens qui apparaissent en runs >= 50 → probablement des image patches
        for tid, _ in runs:
            image_token_ids.add(tid)

    # Construit le texte par segments, en remplaçant les runs de tokens-image
    output: List[str] = []
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] in image_token_ids:
            # Trouve la fin du run d'image tokens
            j = i
            while j < n and ids[j] in image_token_ids:
                j += 1
            n_patches = j - i
            output.append(f"[IMAGE PATCHES × {n_patches}]")
            i = j
        else:
            # Décode le segment textuel jusqu'au prochain token-image
            j = i
            while j < n and ids[j] not in image_token_ids:
                j += 1
            segment_ids = ids[i:j]
            text_segment = tokenizer.decode(segment_ids, skip_special_tokens=False)
            output.append(text_segment)
            i = j

    return "".join(output)


# ═══════════════════════════════════════════════════════════════════════════
# Affichage formaté
# ═══════════════════════════════════════════════════════════════════════════

def format_sample_report(
    sample_num: int, total: int,
    row: pd.Series, status: Dict[str, Dict[str, Any]],
    clinical_info: str, full_prompt: str,
    features_only: bool = False,
) -> str:
    """Construit le rapport texte pour un sample."""
    lines: List[str] = []

    # ── En-tête ──────────────────────────────────────────────────────────
    sid = str(row.get("subject_id", "?"))
    src = str(row.get("source", "?"))
    label = int(row["label"]) if "label" in row.index else -1
    label_str = "AD" if label == 1 else ("CN" if label == 0 else "?")

    n_features = len(status)
    n_real = sum(1 for s in status.values() if not s["is_imputed"] and not s["is_nan"])
    n_imputed = sum(1 for s in status.values() if s["is_imputed"])
    n_nan = sum(1 for s in status.values() if s["is_nan"])

    lines.append("═" * 78)
    lines.append(f" Sample {sample_num}/{total}")
    lines.append(f"   subject_id : {sid}")
    lines.append(f"   source     : {src}")
    lines.append(f"   label      : {label_str} ({label})")
    lines.append(f"   features   : {n_real} réelles, {n_imputed} imputées, "
                 f"{n_nan} NaN  (sur {n_features})")
    lines.append("═" * 78)
    lines.append("")

    # ── 1. Statut par feature ──────────────────────────────────────────
    lines.append("1. STATUT DES FEATURES (CSV brut)")
    lines.append("─" * 78)
    lines.append(f"   {'Feature':<18}{'Valeur':<22}{'Statut':<14}{'Détection':<22}")
    lines.append(f"   {'-'*18}{'-'*22}{'-'*14}{'-'*22}")
    for feat, s in status.items():
        val = s["value"]
        if s["is_nan"]:
            val_str = "NaN"
            status_str = "❌ NaN"
        elif s["is_imputed"]:
            val_str = f"{val}" if not isinstance(val, float) else f"{val:.2f}"
            status_str = "🔧 IMPUTÉ"
        else:
            val_str = f"{val}" if not isinstance(val, float) else f"{val:.2f}"
            status_str = "✓ RÉEL"
        # Truncate
        if len(val_str) > 20:
            val_str = val_str[:17] + "..."
        method = s["detection_method"]
        if len(method) > 20:
            method = method[:17] + "..."
        lines.append(f"   {feat:<18}{val_str:<22}{status_str:<14}{method:<22}")
    lines.append("")

    # ── 2. Section clinical_info insérée dans le prompt ────────────────
    lines.append("2. CLINICAL_INFO INSÉRÉ DANS LE PROMPT")
    lines.append("─" * 78)
    if not clinical_info.strip():
        lines.append("   <vide>")
    else:
        for ln in clinical_info.split("\n"):
            lines.append(f"   {ln}")
    lines.append("")

    # ── Vérifications de cohérence ─────────────────────────────────────
    lines.append("3. VÉRIFICATIONS DE COHÉRENCE")
    lines.append("─" * 78)
    warnings: List[str] = []

    # Pour chaque feature imputée, vérifier si elle apparaît dans clinical_info
    for feat, s in status.items():
        if s["is_imputed"]:
            # On regarde si le label de la feature ou son nom apparaît
            # Rule of thumb : si on voit le nom dans clinical_info, c'est qu'elle
            # est PRÉSENTE dans le prompt (avec ou sans marker estimated)
            present_in_prompt = (feat in clinical_info) or any(
                lab in clinical_info
                for lab in [feat.replace("_", " ")]
            )
            if present_in_prompt:
                marker_estimated = "(estimated)" in clinical_info
                # Note : on ne peut pas savoir SI cette feature précise est marquée
                # estimated, juste si le marker apparaît globalement
                if not marker_estimated:
                    warnings.append(
                        f"   ⚠ '{feat}' est imputée ET présente sans marker "
                        f"(estimated) — cohérent avec signal_in_prompt: false "
                        f"(omission attendue)"
                    )
                else:
                    warnings.append(
                        f"   ℹ '{feat}' imputée et présente avec marker "
                        f"(estimated) attendu — cohérent avec signal_in_prompt: true"
                    )

    # Compte direct des occurrences de "(estimated)" dans clinical_info
    n_estimated = clinical_info.count("(estimated)")
    if n_estimated > 0:
        lines.append(f"   ℹ {n_estimated} occurrence(s) de '(estimated)' "
                     f"dans clinical_info → mode 'signal_in_prompt: true'")
    else:
        if n_imputed > 0 and n_real == 0:
            # Toutes les features sont imputées mais aucun marker → omission totale
            lines.append(f"   ✓ {n_imputed} features imputées et 0 marker '(estimated)'")
            lines.append(f"     → mode 'signal_in_prompt: false' (omission complète)")
        elif n_imputed > 0:
            lines.append(f"   ℹ {n_imputed} features imputées et 0 marker '(estimated)'")
            lines.append(f"     → mode 'signal_in_prompt: false' OU pipeline ancien")

    # Compte features visibles dans clinical_info (heuristique)
    bullets_in_prompt = clinical_info.count("\n- ") + (
        1 if clinical_info.startswith("- ") else 0
    )
    lines.append(f"   • {bullets_in_prompt} bullets dans clinical_info")
    lines.append(f"   • Attendu si signal_in_prompt=false : ~{n_real} bullets")
    lines.append(f"   • Attendu si signal_in_prompt=true  : ~{n_real + n_imputed} bullets")
    if bullets_in_prompt == n_real:
        lines.append(f"   ✓ Cohérent avec OMISSION des imputés (signal_in_prompt: false)")
    elif bullets_in_prompt == (n_real + n_imputed):
        lines.append(f"   ✓ Cohérent avec INCLUSION des imputés (signal_in_prompt: true)")
    else:
        lines.append(f"   ⚠ Nombre de bullets ne correspond ni à {n_real} ni à "
                     f"{n_real + n_imputed} — vérification manuelle recommandée")

    if warnings:
        lines.extend(warnings)
    lines.append("")

    # ── 4. Prompt complet ──────────────────────────────────────────────
    if not features_only:
        lines.append("4. PROMPT COMPLET (chat template appliqué, sans pixel data)")
        lines.append("─" * 78)
        for ln in full_prompt.split("\n"):
            lines.append(f"   {ln}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Sélection des samples
# ═══════════════════════════════════════════════════════════════════════════

def select_samples(
    df: pd.DataFrame,
    n: int,
    cohort: Optional[str] = None,
    label: Optional[str] = None,
    subject_id: Optional[str] = None,
    seed: int = 42,
) -> List[int]:
    """Sélectionne `n` indices selon les filtres."""
    if subject_id is not None:
        matches = df.index[df["subject_id"].astype(str) == subject_id].tolist()
        if not matches:
            raise ValueError(f"subject_id '{subject_id}' introuvable")
        return matches[:n]

    sub = df
    if cohort is not None:
        if "source" not in sub.columns:
            raise ValueError("Colonne 'source' absente du DataFrame")
        sub = sub[sub["source"].astype(str) == cohort]
        if len(sub) == 0:
            raise ValueError(f"Aucun sujet de cohorte '{cohort}'")

    if label is not None:
        label_int = 1 if label.upper() == "AD" else 0
        sub = sub[sub["label"] == label_int]
        if len(sub) == 0:
            raise ValueError(f"Aucun sujet de label '{label}'")

    rng = np.random.RandomState(seed)
    indices = sub.index.tolist()
    if len(indices) <= n:
        return indices
    chosen = rng.choice(indices, size=n, replace=False)
    return sorted(chosen.tolist())


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspecte les prompts construits par TfeDataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str,
                   default=str(THIS_DIR / "config" / "config_base.yaml"),
                   help="Config YAML (défaut: config/config_base.yaml)")
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "val", "test"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n", type=int, default=5,
                   help="Nombre de samples à inspecter (défaut: 5)")
    p.add_argument("--cohort", type=str, default=None,
                   choices=["ADNI", "NACC", "OASIS"],
                   help="Filtrer une cohorte")
    p.add_argument("--label", type=str, default=None,
                   choices=["CN", "AD"],
                   help="Filtrer une classe")
    p.add_argument("--subject-id", type=str, default=None,
                   help="Inspecter un sujet précis (override autres filtres)")
    p.add_argument("--prompt-mode", type=str, default=None,
                   choices=["full", "ablation", "minimal", "image_centric", "image_focused"],
                   help="Override config.prompt.mode")
    p.add_argument("--prompt-variant", type=str, default=None,
                   choices=["v1", "v2", "v3"])
    p.add_argument("--features-only", action="store_true",
                   help="N'affiche que la section clinical_info, pas le prompt complet")
    p.add_argument("--output", type=str, default=None,
                   help="Sauve le rapport dans un fichier (sinon stdout)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    AutoProcessor, load_config, TfeDataset, _ = _lazy_imports()

    # ── Chargement config ────────────────────────────────────────────────
    config = load_config(args.config)

    # Override prompt si demandé
    if args.prompt_mode is not None:
        config.setdefault("prompt", {})
        config["prompt"]["mode"] = args.prompt_mode
        if args.prompt_variant is not None:
            config["prompt"]["variant"] = args.prompt_variant
        elif args.prompt_mode != "image_focused":
            config["prompt"].pop("variant", None)

    # ── Résolution chemin CSV ────────────────────────────────────────────
    splits_dir = Path(config["data"]["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = (PROJECT_ROOT / splits_dir).resolve()
    csv_path = splits_dir / f"fold_{args.fold}" / f"{args.split}.csv"
    if not csv_path.exists():
        print(f"[!] CSV introuvable : {csv_path}")
        return 1

    # ── Chargement processor ─────────────────────────────────────────────
    print(f"[*] Chargement processor MedGemma...", flush=True)
    processor = AutoProcessor.from_pretrained(config["model"]["name"])

    # ── Construction du dataset ─────────────────────────────────────────
    # is_training=True pour inclure le label "CN"/"AD" dans les messages
    print(f"[*] Construction TfeDataset depuis {csv_path}...", flush=True)
    ds = TfeDataset(str(csv_path), processor, config, is_training=True)
    print(f"[*] Dataset {args.split} chargé : {len(ds)} samples")
    print(f"[*] Mode prompt configuré : {getattr(ds, 'prompt_mode', 'inconnu')}")

    # ── Sélection des samples ───────────────────────────────────────────
    try:
        indices = select_samples(
            ds.df, n=args.n,
            cohort=args.cohort,
            label=args.label,
            subject_id=args.subject_id,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"[!] {e}")
        return 1

    print(f"[*] Inspection de {len(indices)} samples : {indices}\n")

    # ── Détection feature list active ───────────────────────────────────
    features_active = list(getattr(ds, "tabular_features", []))
    if not features_active:
        # Fallback : toutes les colonnes connues du config
        features_active = list(config.get("data", {}).get("tabular_features", []))

    # ── Génération du rapport ───────────────────────────────────────────
    output_lines: List[str] = []

    # En-tête global
    output_lines.append("=" * 78)
    output_lines.append(" INSPECTION DES PROMPTS — TfeDataset")
    output_lines.append("=" * 78)
    output_lines.append(f" CSV         : {csv_path}")
    output_lines.append(f" Split       : {args.split} (fold {args.fold})")
    output_lines.append(f" N samples   : {len(indices)} / {len(ds)}")
    output_lines.append(f" Filtres     : cohort={args.cohort or 'all'}, "
                        f"label={args.label or 'all'}, "
                        f"subject_id={args.subject_id or 'auto'}")
    output_lines.append(f" Prompt mode : {getattr(ds, 'prompt_mode', '?')}")
    if hasattr(ds, "imputation_signal_in_prompt"):
        sig = ds.imputation_signal_in_prompt
        sig_str = "AFFICHÉES (estimated)" if sig else "OMISES du prompt"
        output_lines.append(f" Imputation  : signal_in_prompt={sig} → valeurs imputées {sig_str}")
    output_lines.append(f" Features    : {features_active}")
    output_lines.append("=" * 78)
    output_lines.append("")

    for i, idx in enumerate(indices, 1):
        row = ds.df.iloc[idx]
        status = detect_imputation_status(row, features_active)

        try:
            clinical_info, full_prompt = extract_text_prompt(ds, idx)
        except Exception as e:
            print(f"[!] extract_text_prompt failed for idx={idx}: {e}")
            clinical_info = ds._format_clinical_info(row)
            full_prompt = f"[Erreur d'extraction : {e}]"

        report = format_sample_report(
            sample_num=i, total=len(indices),
            row=row, status=status,
            clinical_info=clinical_info, full_prompt=full_prompt,
            features_only=args.features_only,
        )
        output_lines.append(report)

    # ── Statistiques globales ───────────────────────────────────────────
    output_lines.append("")
    output_lines.append("=" * 78)
    output_lines.append(" STATISTIQUES GLOBALES SUR LES SAMPLES INSPECTÉS")
    output_lines.append("=" * 78)
    total_real = total_imputed = total_nan = 0
    for idx in indices:
        row = ds.df.iloc[idx]
        s = detect_imputation_status(row, features_active)
        total_real    += sum(1 for v in s.values() if not v["is_imputed"] and not v["is_nan"])
        total_imputed += sum(1 for v in s.values() if v["is_imputed"])
        total_nan     += sum(1 for v in s.values() if v["is_nan"])
    total = total_real + total_imputed + total_nan
    if total > 0:
        output_lines.append(f"  Sur {len(indices)} samples × {len(features_active)} features = "
                            f"{total} cellules :")
        output_lines.append(f"   - Réelles  : {total_real:5d} ({total_real/total*100:.1f}%)")
        output_lines.append(f"   - Imputées : {total_imputed:5d} ({total_imputed/total*100:.1f}%)")
        output_lines.append(f"   - NaN      : {total_nan:5d} ({total_nan/total*100:.1f}%)")
    output_lines.append("=" * 78)

    # ── Sortie ──────────────────────────────────────────────────────────
    full_output = "\n".join(output_lines)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_output, encoding="utf-8")
        print(f"\n[✓] Rapport écrit dans {out_path}  ({len(full_output)} caractères)")
    else:
        print(full_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())