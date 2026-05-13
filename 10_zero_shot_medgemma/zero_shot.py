"""
zero_shot.py — Tâche 10 : Baseline MedGemma vanilla (zero-shot).

Évalue MedGemma 1.5 4B SANS fine-tuning pour la classification CN/AD.
Sert de référence absolue : "Sans entraînement spécifique, quel est le score ?"

Stratégie d'inférence :
    1. Charge MedGemma quantifié 4-bit (pas de LoRA)
    2. Pour chaque patient du test set du fold demandé :
        - Prépare le prompt full ou minimal selon config
        - Génère via model.generate (max 5 tokens, temperature=0)
        - Parse la sortie : cherche "CN" ou "AD" dans les premiers tokens
        - Calcul probabilités via softmax sur logits CN vs AD
        au position de prédiction
    3. Calcule métriques (AUC, F1, sens, spec)

Lancement :
    cd 10_zero_shot_medgemma/
    python zero_shot.py
    python zero_shot.py --split val
    python zero_shot.py --max-samples 100         # quick test
    python zero_shot.py --prompt-mode minimal     # IRM seules

Outputs (dans results/10_zero_shot/) :
    metrics_test.json, predictions_test.csv,
    cohort_metrics_test.csv, roc_curve_test.png, confusion_matrix_test.png

Résultat attendu :
    - Mode full      : AUC ~0.65-0.80 (MedGemma sait que les neuropsych
                                      faibles correspondent à AD)
    - Mode minimal   : AUC ~0.50-0.55 (proche du hasard sans contexte clinique)
    Le gap entre zero-shot et fine-tuned mesure le gain réel du fine-tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from tqdm import tqdm

# ── Imports locaux ─────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    load_config, setup_env, check_vram, release_gpu,
    set_token_ids, save_metrics_json,
    plot_roc_curve, plot_confusion_matrix,
)
from dataset import TfeDataset, tfe_collate_fn


# ═══════════════════════════════════════════════════════════════════════════
# 1. INFÉRENCE ZERO-SHOT
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_zero_shot(
    model,
    processor,
    dataset: TfeDataset,
    cn_id: int,
    ad_id: int,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Inférence zero-shot sur le dataset complet.

    ⚠️ IMPORTANT : le dataset DOIT être construit avec is_training=True afin
    que le label assistant ("CN"/"AD") soit présent dans labels — c'est la
    convention "verbalizer with primer" pour zero-shot. find_answer_position
    repère alors le token CN/AD comme à l'entraînement et calcule le logit
    au logit_pos correct (ans_pos - 1).

    Cela évite le bug subtil où, sans label assistant, le prochain token prédit
    après le prompt n'est PAS un verbalizer mais un mot de phrase libre
    (ex: "The", "Based"), rendant softmax(CN, AD) au mauvais endroit.

    Retourne dict avec arrays alignés (preds, probs, true_cls, subject_ids,
    sources). Les samples qui échouent (exception, verbalizer mismatch) ont
    prob=NaN — à filtrer avant de calculer les métriques.
    """
    model.eval()

    # On utilise le DataLoader + tfe_collate_fn (cohérent avec evaluate_dataset)
    # → garantit que pixel_values, pixel_attention_mask, token_type_ids sont
    # construits exactement comme à l'entraînement.
    from utils import find_answer_position
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=tfe_collate_fn, num_workers=0,
    )

    preds: List[float] = []
    probs: List[float] = []
    true_cls: List[int] = []
    subject_ids: List[str] = []
    sources: List[str] = []
    n_skipped = 0
    sample_idx = 0

    for batch in tqdm(loader, desc="Zero-shot inference"):
        # Préfetch metadata avant tout (au cas où le forward crash)
        df_row = dataset.df.iloc[sample_idx]
        true_lab_csv = int(df_row["label"]) if "label" in df_row.index else None
        sid = str(df_row.get("subject_id", f"unknown_{sample_idx}"))
        src = str(df_row.get("source", "?"))

        try:
            # Pop MMSE keys (présents si multitâche, ignorés en zero-shot)
            batch.pop("mmse_score", None)
            batch.pop("regression_weight", None)
            labels = batch.get("labels")

            # Reshape pixel_values 5D → 4D (n_views collapsé sur batch)
            if "pixel_values" in batch and batch["pixel_values"].ndim == 5:
                b, n, c, h, w = batch["pixel_values"].shape
                batch["pixel_values"] = batch["pixel_values"].view(b * n, c, h, w)
            if "pixel_attention_mask" in batch and batch["pixel_attention_mask"].ndim == 4:
                b, n, h, w = batch["pixel_attention_mask"].shape
                batch["pixel_attention_mask"] = batch["pixel_attention_mask"].view(b * n, h, w)

            # token_type_ids requis par MedGemma
            if "token_type_ids" not in batch and "input_ids" in batch:
                batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

            batch_gpu = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            outputs = model(**batch_gpu, use_cache=False)

            # Cherche le verbalizer dans les labels (présent car is_training=True)
            if labels is None:
                raise RuntimeError(
                    "labels=None — le dataset doit être en is_training=True "
                    "pour que le verbalizer soit présent dans la séquence"
                )
            logit_pos, true_lab = find_answer_position(labels[0], cn_id, ad_id)

            if logit_pos is None or true_lab is None:
                # Verbalizer mismatch → skip (préserve l'alignement via NaN)
                n_skipped += 1
                preds.append(np.nan)
                probs.append(np.nan)
                true_cls.append(true_lab_csv if true_lab_csv is not None else -1)
                subject_ids.append(sid)
                sources.append(src)
                sample_idx += 1
                continue

            cn_logit = float(outputs.logits[0, logit_pos, cn_id].cpu())
            ad_logit = float(outputs.logits[0, logit_pos, ad_id].cpu())
            exp_cn = np.exp(cn_logit)
            exp_ad = np.exp(ad_logit)
            prob_ad = exp_ad / (exp_cn + exp_ad)
            pred = 1 if prob_ad > 0.5 else 0

            preds.append(pred)
            probs.append(prob_ad)
            true_cls.append(int(true_lab))   # label depuis le verbalizer (cohérent training)
            subject_ids.append(sid)
            sources.append(src)

        except Exception as e:
            print(f"\n[!] Erreur sur sample {sample_idx} ({sid}) : {type(e).__name__}: {e}")
            n_skipped += 1
            # NaN au lieu de fabriquer 0/0/0.5 (qui biaisait toutes les métriques)
            preds.append(np.nan)
            probs.append(np.nan)
            true_cls.append(true_lab_csv if true_lab_csv is not None else -1)
            subject_ids.append(sid)
            sources.append(src)

        sample_idx += 1

    if n_skipped > 0:
        print(f"\n[!] {n_skipped}/{len(dataset)} samples skippés "
              f"(NaN dans probs — filtrés avant métriques)")

    return {
        "preds":       np.asarray(preds, dtype=float),
        "probs":       np.asarray(probs, dtype=float),
        "true_cls":    np.asarray(true_cls),
        "subject_ids": subject_ids,
        "sources":     sources,
        "n_skipped":   n_skipped,
    }


def compute_zero_shot_metrics(
    preds: np.ndarray, probs: np.ndarray, true_cls: np.ndarray,
) -> Dict[str, float]:
    """
    Métriques classification standard.

    ⚠️ Filtre les NaN avant calcul (samples skippés en inférence).
    ⚠️ Le seuil optimal Youden est calculé sur le TEST (pas de val en zero-shot)
       → biais optimiste, à mentionner explicitement dans la thèse.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score,
        roc_auc_score, roc_curve,
    )

    # Filtre NaN
    mask = ~np.isnan(probs)
    preds = preds[mask]
    probs = probs[mask]
    true_cls = np.asarray(true_cls)[mask]
    n_used = mask.sum()

    out: Dict[str, float] = {"n_used": int(n_used)}
    if n_used == 0:
        for k in ("accuracy", "f1", "sensitivity", "specificity", "auc"):
            out[k] = float("nan")
        return out

    # Cast preds en int après filtrage (NaN n'est plus présent)
    preds_int = preds.astype(int)

    out["accuracy"]    = float(accuracy_score(true_cls, preds_int))
    out["f1"]          = float(f1_score(true_cls, preds_int, zero_division=0))
    out["sensitivity"] = float(recall_score(true_cls, preds_int, pos_label=1, zero_division=0))
    out["specificity"] = float(recall_score(true_cls, preds_int, pos_label=0, zero_division=0))

    if len(np.unique(true_cls)) >= 2:
        out["auc"] = float(roc_auc_score(true_cls, probs))
        # Seuil optimal Youden — calculé sur test (pas de val en zero-shot)
        fpr, tpr, thr = roc_curve(true_cls, probs)
        opt_idx = int(np.argmax(tpr - fpr))
        opt_thr = float(thr[opt_idx])
        preds_opt = (probs >= opt_thr).astype(int)
        out["optimal_threshold"] = opt_thr
        out["f1_calibrated"] = float(f1_score(true_cls, preds_opt, zero_division=0))
        out["sensitivity_calibrated"] = float(
            recall_score(true_cls, preds_opt, pos_label=1, zero_division=0)
        )
        out["specificity_calibrated"] = float(
            recall_score(true_cls, preds_opt, pos_label=0, zero_division=0)
        )
    else:
        out["auc"] = 0.5
    return out


def compute_cohort_breakdown(
    preds: np.ndarray, probs: np.ndarray, true_cls: np.ndarray,
    sources: List[str],
) -> pd.DataFrame:
    """Métriques par cohorte (filtre NaN avant calcul)."""
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, roc_auc_score,
    )

    sources_arr = np.asarray(sources)
    nan_mask = np.isnan(probs)
    if nan_mask.any():
        # Filtre NaN globalement avant cohort breakdown
        keep = ~nan_mask
        preds = preds[keep]
        probs = probs[keep]
        true_cls = np.asarray(true_cls)[keep]
        sources_arr = sources_arr[keep]

    rows: List[Dict[str, Any]] = []
    cohorts = ["GLOBAL"] + sorted(set(sources_arr.tolist()))

    for cohort in cohorts:
        if cohort == "GLOBAL":
            mask = np.ones(len(true_cls), dtype=bool)
        else:
            mask = (sources_arr == cohort)

        n = int(mask.sum())
        if n == 0:
            continue

        t = np.asarray(true_cls)[mask]
        p = preds[mask].astype(int)
        pr = probs[mask]

        row: Dict[str, Any] = {
            "cohort": cohort, "n": n,
            "n_cn":   int((t == 0).sum()),
            "n_ad":   int((t == 1).sum()),
        }
        if n >= 2 and len(set(t)) >= 2:
            row["accuracy"]    = float(accuracy_score(t, p))
            row["auc"]         = float(roc_auc_score(t, pr))
            row["f1"]          = float(f1_score(t, p, zero_division=0))
            row["sensitivity"] = float(recall_score(t, p, pos_label=1, zero_division=0))
            row["specificity"] = float(recall_score(t, p, pos_label=0, zero_division=0))
        else:
            for k in ("accuracy", "auc", "f1", "sensitivity", "specificity"):
                row[k] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 2. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T10 — Zero-shot MedGemma (sans fine-tuning)"
    )
    p.add_argument("--config", type=str, default=str(THIS_DIR / "config.yaml"))
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limite le nombre de sujets (debug). Ex: 100")
    p.add_argument("--prompt-mode", type=str, default=None,
                   choices=["full", "ablation", "minimal", "image_centric"],
                   help="Override du mode prompt depuis le config")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    # Override prompt mode si demandé
    if args.prompt_mode:
        config.setdefault("prompt", {})["mode"] = args.prompt_mode

    setup_env(seed=config["training"].get("seed", 42))

    # Résolution chemins
    splits_dir = Path(config["data"]["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = (PROJECT_ROOT / splits_dir).resolve()
    csv_path = splits_dir / f"fold_{args.fold}" / f"{args.split}.csv"
    if not csv_path.exists():
        print(f"[✗] Split introuvable : {csv_path}")
        return 1

    out_dir = args.output or config["training"]["output_dir"]
    if not Path(out_dir).is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_mode_used = (
        args.prompt_mode
        or config.get("prompt", {}).get("mode")
        or "auto"
    )
    sig_in_prompt = config.get("imputation", {}).get("signal_in_prompt", True)

    print(f"\n{'='*70}")
    print(f"  ZERO-SHOT MEDGEMMA")
    print(f"{'='*70}")
    print(f"  Modèle              : {config['model']['name']} (vanilla, no LoRA)")
    print(f"  Split               : {args.split} (fold_{args.fold})")
    print(f"  Prompt mode (config): {prompt_mode_used}")
    print(f"  Imputation policy   : "
          f"{'valeurs imputées AFFICHÉES (estimated)' if sig_in_prompt else 'valeurs imputées OMISES du prompt'}")
    if args.max_samples:
        print(f"  Max samples         : {args.max_samples}")
    print(f"  Output dir          : {out_dir}")
    print(f"{'='*70}\n")

    model = None
    try:
        check_vram(min_gb=8.0)

        # ── Quantization ─────────────────────────────────────────────────
        q = config["quantization"]
        bnb = BitsAndBytesConfig(
            load_in_4bit=q["load_in_4bit"],
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if q["bnb_4bit_compute_dtype"] == "bfloat16"
                else torch.float16
            ),
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        )

        # ── Modèle vanilla ───────────────────────────────────────────────
        print(f"[*] Chargement MedGemma vanilla...")
        model = AutoModelForImageTextToText.from_pretrained(
            config["model"]["name"],
            quantization_config=bnb,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(config["model"]["name"])

        cn_id = processor.tokenizer.encode("CN", add_special_tokens=False)[0]
        ad_id = processor.tokenizer.encode("AD", add_special_tokens=False)[0]
        print(f"[*] Token IDs → CN: {cn_id}, AD: {ad_id}")
        set_token_ids(cn_id, ad_id)

        # ── Dataset ──────────────────────────────────────────────────────
        # is_training=True : pour que le verbalizer "CN"/"AD" soit présent
        # dans labels et que find_answer_position puisse repérer le logit_pos
        # correct (sinon on lit les logits là où le modèle ne prédit pas un
        # verbalizer mais un mot de phrase libre comme "The"/"Based").
        ds = TfeDataset(
            str(csv_path), processor, config,
            is_training=True,
            max_samples=args.max_samples,
        )
        print(f"[*] Dataset {args.split} : {len(ds)} échantillons")

        # Lit le mode prompt RÉEL utilisé par le dataset (post auto-détection)
        # — important quand prompt.mode n'est pas explicite dans la config.
        actual_prompt_mode = getattr(ds, "prompt_mode", prompt_mode_used)
        actual_sig = getattr(ds, "imputation_signal_in_prompt", sig_in_prompt)
        if actual_prompt_mode != prompt_mode_used or actual_sig != sig_in_prompt:
            print(f"[*] Mode prompt effectif : {actual_prompt_mode}")
            print(f"[*] Imputation effective : "
                  f"{'AFFICHÉES' if actual_sig else 'OMISES du prompt'}")
        prompt_mode_used = actual_prompt_mode
        sig_in_prompt = actual_sig

        # Print du premier prompt formaté (debug + reproductibilité TFE)
        try:
            sample0 = ds[0]
            decoded = processor.tokenizer.decode(sample0["input_ids"])
            print(f"\n[*] Exemple de prompt (sample 0, {len(sample0['input_ids'])} tokens) :")
            print(f"{'─'*70}")
            print(decoded[:1500] + ("..." if len(decoded) > 1500 else ""))
            print(f"{'─'*70}\n")
        except Exception as e:
            print(f"[!] Impossible d'afficher l'exemple : {e}")

        # ── Inférence zero-shot ──────────────────────────────────────────
        print(f"\n[*] Inférence zero-shot...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = predict_zero_shot(model, processor, ds, cn_id, ad_id, device=device)

        # ── Métriques ────────────────────────────────────────────────────
        metrics = compute_zero_shot_metrics(
            results["preds"], results["probs"], results["true_cls"]
        )

        print(f"\n{'─'*70}")
        print(f"  RÉSULTATS ZERO-SHOT")
        print(f"{'─'*70}")
        print(f"  Accuracy    : {metrics['accuracy']:.4f}")
        print(f"  AUC         : {metrics['auc']:.4f}")
        print(f"  F1          : {metrics['f1']:.4f}")
        print(f"  Sensibilité : {metrics['sensitivity']:.4f}")
        print(f"  Spécificité : {metrics['specificity']:.4f}")
        if "optimal_threshold" in metrics:
            print(f"\n  Seuil optimal Youden = {metrics['optimal_threshold']:.4f}")
            print(f"    F1          : {metrics['f1_calibrated']:.4f}")
            print(f"    Sensibilité : {metrics['sensitivity_calibrated']:.4f}")
            print(f"    Spécificité : {metrics['specificity_calibrated']:.4f}")
        print(f"{'─'*70}\n")

        # ── Métriques par cohorte ────────────────────────────────────────
        cohort_df = compute_cohort_breakdown(
            results["preds"], results["probs"], results["true_cls"],
            results["sources"],
        )
        print(f"  PAR COHORTE")
        print(f"{'─'*70}")
        print(cohort_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"{'─'*70}\n")

        # ── Sauvegarde ───────────────────────────────────────────────────
        save_metrics_json({
            "task_name":           config.get("task_name", "10_zero_shot"),
            "split":               args.split,
            "fold":                args.fold,
            "n_samples":           len(ds),
            "prompt_mode":         prompt_mode_used,
            "signal_in_prompt":    sig_in_prompt,
            "metrics":             metrics,
        }, out_dir / f"metrics_{args.split}.json")
        print(f"  💾 metrics_{args.split}.json")

        pred_df = pd.DataFrame({
            "subject_id":  results["subject_ids"],
            "source":      results["sources"],
            "true_label":  results["true_cls"],
            "pred_label":  results["preds"],
            "prob_AD":     results["probs"],
        })
        pred_df.to_csv(out_dir / f"predictions_{args.split}.csv", index=False)
        print(f"  💾 predictions_{args.split}.csv ({len(pred_df)} lignes)")

        cohort_df.to_csv(out_dir / f"cohort_metrics_{args.split}.csv", index=False)
        print(f"  💾 cohort_metrics_{args.split}.csv")

        # Filtre NaN pour les plots (samples skippés)
        valid_mask = ~np.isnan(results["probs"])
        true_valid = np.asarray(results["true_cls"])[valid_mask]
        probs_valid = results["probs"][valid_mask]
        preds_valid = results["preds"][valid_mask].astype(int)

        plot_roc_curve(
            true_valid, probs_valid,
            out_dir / f"roc_curve_{args.split}.png",
            title=f"ROC zero-shot — MedGemma vanilla ({args.split})",
            extra_caption=f"AUC = {metrics['auc']:.4f} | n = {len(true_valid)} | mode = {prompt_mode_used}",
        )
        print(f"  💾 roc_curve_{args.split}.png")

        plot_confusion_matrix(
            true_valid, preds_valid,
            out_dir / f"confusion_matrix_{args.split}.png",
            title=f"Matrice de confusion zero-shot ({args.split})",
        )
        print(f"  💾 confusion_matrix_{args.split}.png")

        if "optimal_threshold" in metrics:
            preds_opt = (probs_valid >= metrics["optimal_threshold"]).astype(int)
            plot_confusion_matrix(
                true_valid, preds_opt,
                out_dir / f"confusion_matrix_{args.split}_optimal.png",
                title=f"Matrice de confusion zero-shot — seuil optimal "
                      f"{metrics['optimal_threshold']:.3f} (biaisé : calc sur test)",
            )
            print(f"  💾 confusion_matrix_{args.split}_optimal.png")

        print(f"\n[✓] Zero-shot terminé — résultats dans {out_dir}\n")
        return 0

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"\n[!] OOM : {e}")
        else:
            print(f"\n[!] RuntimeError : {e}")
        return 2

    except Exception as e:
        print(f"\n[!] Erreur : {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1

    finally:
        release_gpu(model=model)


if __name__ == "__main__":
    sys.exit(main())