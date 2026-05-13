"""
_common.py — Utilitaires partagés entre tous les scripts d'explainability.

Centralise :
    - Argparse commun (--checkpoint, --task, --n-patients, --strategy, etc.)
    - Résolution checkpoint + config (réutilise evaluate.py)
    - Chargement modèle (réutilise load_model_for_eval)
    - Sélection des patients selon stratégie (random / stratified / tp_fn_mix)
    - Helpers IRM (extraction coupe brute, superposition heatmap, mosaïque)
    - Détection automatique des couches cibles (vision encoder, last attention)

Importés par : gradcam_mri.py, occlusion_mri.py, feature_importance.py,
attention_rollout.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── Imports projet ──────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    load_config, setup_env, check_vram, release_gpu,
    is_valid_checkpoint, set_token_ids,
)
from dataset import TfeDataset, MultiViewSliceExtractor
from evaluate import (
    resolve_task_dir, find_best_model, load_model_for_eval,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ARGPARSE COMMUN
# ═══════════════════════════════════════════════════════════════════════════

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Args communs aux 4 scripts d'explainability."""
    parser.add_argument(
        "--task", type=str, default=None,
        help="Nom de la tâche (auto-détecte le best_model). "
             "Ex: 01_no_mmse, 02_with_mmse",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Chemin explicite vers un checkpoint (override --task)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML (sinon : <task_dir>/config.yaml)",
    )
    parser.add_argument(
        "--n-patients", type=int, default=20,
        help="Nombre de patients à analyser (défaut : 20)",
    )
    parser.add_argument(
        "--strategy", type=str, default="stratified",
        choices=["random", "stratified", "tp_fn_mix"],
        help="Stratégie de sélection des patients",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Split à analyser (défaut : test)",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Fold à analyser (défaut : 0)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override du dossier output (défaut : <ckpt>/explainability/<method>/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--predictions-csv", type=str, default=None,
        help="Chemin vers predictions_test.csv (généré par evaluate.py). "
             "Requis pour la stratégie tp_fn_mix.",
    )
    parser.add_argument(
        "--filter-real-only", action="store_true", default=False,
        help=(
            "Restreint la sélection aux patients avec données réelles (non imputées). "
            "Cascade de détection : colonnes <feature>_imputed → has_real_measures. "
            "Garde uniquement les patients où la majorité des features sont mesurées. "
            "Critique pour feature_importance (perturbation biaisée sur valeurs imputées) "
            "et Grad-CAM (validation clinique de l'interprétation IRM)."
        ),
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════
# 2. RÉSOLUTION CHECKPOINT + CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def resolve_checkpoint_and_config(args: argparse.Namespace) -> Tuple[Path, Path]:
    """
    Résout (checkpoint_path, config_path) à partir de --task ou --checkpoint.
    Retourne deux Path validés.
    """
    if args.task is None and args.checkpoint is None:
        raise ValueError("Spécifie --task ou --checkpoint")

    if args.task:
        task_dir = resolve_task_dir(args.task)
        config_path = (
            Path(args.config) if args.config else task_dir / "config.yaml"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Config introuvable : {config_path}")

        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint).resolve()
        else:
            config = load_config(str(config_path))
            checkpoint_path = find_best_model(task_dir, config)
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not args.config:
            raise ValueError(
                "--config requis quand on utilise --checkpoint sans --task"
            )
        config_path = Path(args.config)

    if not is_valid_checkpoint(str(checkpoint_path)):
        raise FileNotFoundError(
            f"Checkpoint invalide (pas d'adapter_config.json) : {checkpoint_path}"
        )

    return checkpoint_path, config_path


def resolve_output_dir(
    args: argparse.Namespace, checkpoint_path: Path, method_name: str,
) -> Path:
    """
    Détermine le dossier output. Par défaut : <ckpt>/explainability/<method>/.
    """
    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = checkpoint_path / "explainability" / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHARGEMENT MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_dataset(
    config_path: Path,
    checkpoint_path: Path,
    split: str = "test",
    fold: int = 0,
    is_training_for_dataset: bool = True,
) -> Tuple[Any, Any, Any, int, int, bool]:
    """
    Pipeline standard de chargement.

    Args:
        is_training_for_dataset: True pour avoir les labels assistant dans
            la séquence (utile pour Grad-CAM / occlusion / feature importance).
            False pour zero-shot inference.

    Retourne (processor, model, dataset, cn_id, ad_id, has_mmse_head).
    """
    config = load_config(str(config_path))
    setup_env(seed=config["training"].get("seed", 42))

    # Chargement modèle (réutilise evaluate.py)
    processor, model, has_mmse = load_model_for_eval(
        config, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Token IDs CN/AD
    cn_id = processor.tokenizer.encode("CN", add_special_tokens=False)[0]
    ad_id = processor.tokenizer.encode("AD", add_special_tokens=False)[0]
    set_token_ids(cn_id, ad_id)

    # Dataset
    splits_dir = Path(config["data"]["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = (PROJECT_ROOT / splits_dir).resolve()
    csv_path = splits_dir / f"fold_{fold}" / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split introuvable : {csv_path}")

    dataset = TfeDataset(
        str(csv_path), processor, config,
        is_training=is_training_for_dataset,
    )

    return processor, model, dataset, cn_id, ad_id, has_mmse


# ═══════════════════════════════════════════════════════════════════════════
# 4. SÉLECTION DES PATIENTS
# ═══════════════════════════════════════════════════════════════════════════

def get_real_mask(df: pd.DataFrame, min_real_ratio: float = 0.5) -> np.ndarray:
    """
    Retourne un masque booléen : True si le patient a suffisamment de features réelles.

    Cascade de détection (ordre de priorité) :
      1. Colonnes <feature>_imputed (1=imputé, 0=réel) — méthode préférentielle.
         Un patient est "réel" si ≤ (1 - min_real_ratio) × n_features sont imputées.
      2. has_real_measures (0/1) — colonne globale MMSE. Si présente et que les
         colonnes _imputed sont absentes, l'utilise comme proxy.
      3. Fallback : tous True (pas de moyen de filtrer → on garde tout).

    Args:
        min_real_ratio : fraction minimale de features non-imputées pour garder
                         un patient. Défaut 0.5 = au moins la moitié réelles.
    """
    imputed_cols = [c for c in df.columns if c.endswith("_imputed")]

    if imputed_cols:
        # Méthode 1 : colonnes _imputed explicites
        imputed_matrix = df[imputed_cols].fillna(1).astype(int)  # NaN → imputé
        n_features = len(imputed_cols)
        n_real_per_patient = (imputed_matrix == 0).sum(axis=1)
        real_ratio = n_real_per_patient / n_features
        mask = (real_ratio >= min_real_ratio).values
        n_kept = int(mask.sum())
        print(f"[*] Filtre données réelles (≥{min_real_ratio*100:.0f}% features réelles) : "
              f"{n_kept}/{len(df)} patients conservés")
        return mask

    if "has_real_measures" in df.columns:
        # Méthode 2 : proxy MMSE
        mask = (df["has_real_measures"].fillna(0).astype(int) == 1).values
        n_kept = int(mask.sum())
        print(f"[*] Filtre données réelles (has_real_measures=1) : "
              f"{n_kept}/{len(df)} patients conservés")
        return mask

    # Fallback
    print(f"[!] Pas de colonne _imputed ni has_real_measures — filtre désactivé")
    return np.ones(len(df), dtype=bool)


def select_patients(
    df: pd.DataFrame,
    n_patients: int,
    strategy: str = "stratified",
    predictions_csv: Optional[str] = None,
    seed: int = 42,
    filter_real_only: bool = False,
    min_real_ratio: float = 0.5,
) -> List[int]:
    """
    Sélectionne n_patients indices du df selon la stratégie.

    Stratégies :
        - 'random'      : tirage aléatoire stratifié CN/AD (50/50)
        - 'stratified'  : équilibre CN/AD × cohortes (proportionnel à n_patients)
        - 'tp_fn_mix'   : nécessite predictions_csv. Par défaut 4 buckets de
                          n_patients//4 :
                            * True Positive  (AD prédits AD, prob > 0.7)
                            * False Negative (AD prédits CN, prob < 0.3)
                            * True Negative  (CN prédits CN, prob < 0.3)
                            * Hard cases     (prob ∈ [0.4, 0.6])

    Args:
        filter_real_only : si True, restreint le pool aux patients avec données
                           réelles (non imputées). Voir get_real_mask().

    Retourne la liste des indices dans df (indices du df original, pas du sous-df).
    """
    rng = np.random.RandomState(seed)

    # ── Filtre données réelles ─────────────────────────────────────────────
    if filter_real_only:
        real_mask = get_real_mask(df, min_real_ratio=min_real_ratio)
        df_filtered = df[real_mask].copy()
        if len(df_filtered) == 0:
            print(f"[!] Aucun patient avec données réelles — filtre désactivé")
            df_filtered = df
        # Les fonctions internes retournent des indices du df_filtered.
        # On doit les remap vers les indices du df original.
        filtered_indices = np.where(real_mask)[0]  # positions dans df original
    else:
        df_filtered = df
        filtered_indices = np.arange(len(df))

    # ── Sélection dans le pool filtré ─────────────────────────────────────
    if strategy == "random":
        local_idx = _select_random(df_filtered.reset_index(drop=True), n_patients, rng)
    elif strategy == "stratified":
        local_idx = _select_stratified(df_filtered.reset_index(drop=True), n_patients, rng)
    elif strategy == "tp_fn_mix":
        if predictions_csv is None:
            raise ValueError(
                "Stratégie tp_fn_mix nécessite --predictions-csv "
                "(générer d'abord via evaluate.py)"
            )
        local_idx = _select_tp_fn_mix(
            df_filtered.reset_index(drop=True), predictions_csv, n_patients, rng
        )
    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")

    # Remap des indices locaux (dans df_filtered) vers indices globaux (dans df)
    return sorted([int(filtered_indices[i]) for i in local_idx])


def _select_random(
    df: pd.DataFrame, n: int, rng: np.random.RandomState,
) -> List[int]:
    """Tirage aléatoire 50/50 CN/AD."""
    n_cn, n_ad = n // 2, n - n // 2
    cn_idx = df[df["label"] == 0].index.tolist()
    ad_idx = df[df["label"] == 1].index.tolist()
    selected_cn = rng.choice(cn_idx, size=min(n_cn, len(cn_idx)), replace=False)
    selected_ad = rng.choice(ad_idx, size=min(n_ad, len(ad_idx)), replace=False)
    return sorted(selected_cn.tolist() + selected_ad.tolist())


def _select_stratified(
    df: pd.DataFrame, n: int, rng: np.random.RandomState,
) -> List[int]:
    """
    Stratification CN/AD × cohorte. Si pas de colonne 'source', fallback random.
    """
    if "source" not in df.columns:
        return _select_random(df, n, rng)

    cohorts = df["source"].unique()
    selected: List[int] = []

    # n total réparti proportionnellement aux cohortes
    cohort_sizes = df["source"].value_counts()
    total = cohort_sizes.sum()

    for cohort in cohorts:
        sub = df[df["source"] == cohort]
        n_cohort = max(2, int(round(n * len(sub) / total)))
        # 50/50 CN/AD dans la cohorte
        n_cn_c = n_cohort // 2
        n_ad_c = n_cohort - n_cn_c
        cn_pool = sub[sub["label"] == 0].index.tolist()
        ad_pool = sub[sub["label"] == 1].index.tolist()
        if cn_pool:
            selected.extend(
                rng.choice(cn_pool, size=min(n_cn_c, len(cn_pool)), replace=False).tolist()
            )
        if ad_pool:
            selected.extend(
                rng.choice(ad_pool, size=min(n_ad_c, len(ad_pool)), replace=False).tolist()
            )

    # Trim/pad pour matcher exactement n
    selected = sorted(set(selected))
    if len(selected) > n:
        selected = sorted(rng.choice(selected, size=n, replace=False).tolist())
    elif len(selected) < n:
        # Compléter au hasard
        remaining = list(set(df.index) - set(selected))
        if remaining:
            extra = rng.choice(
                remaining, size=min(n - len(selected), len(remaining)), replace=False
            ).tolist()
            selected = sorted(selected + extra)
    return selected


def _select_tp_fn_mix(
    df: pd.DataFrame, predictions_csv: str, n: int,
    rng: np.random.RandomState,
) -> List[int]:
    """
    4 buckets équilibrés : TP, FN, TN, hard cases.
    Nécessite predictions_test.csv (généré par evaluate.py).
    """
    pred_df = pd.read_csv(predictions_csv)
    n_per_bucket = max(1, n // 4)

    # Merge avec df sur subject_id pour obtenir les indices df
    if "subject_id" not in df.columns or "subject_id" not in pred_df.columns:
        raise ValueError(
            "Colonne 'subject_id' requise dans df et predictions_csv"
        )
    merged = df.reset_index().merge(
        pred_df[["subject_id", "true_label", "pred_label", "prob_AD"]],
        on="subject_id", how="inner",
    )

    # Buckets
    tp = merged[
        (merged["true_label"] == 1) & (merged["pred_label"] == 1)
        & (merged["prob_AD"] >= 0.7)
    ]
    fn = merged[
        (merged["true_label"] == 1) & (merged["pred_label"] == 0)
    ]
    tn = merged[
        (merged["true_label"] == 0) & (merged["pred_label"] == 0)
        & (merged["prob_AD"] <= 0.3)
    ]
    hard = merged[
        (merged["prob_AD"] >= 0.4) & (merged["prob_AD"] <= 0.6)
    ]

    selected: List[int] = []
    for bucket, name in [(tp, "TP"), (fn, "FN"), (tn, "TN"), (hard, "Hard")]:
        if len(bucket) == 0:
            print(f"  [!] Bucket {name} vide")
            continue
        take = min(n_per_bucket, len(bucket))
        chosen = rng.choice(bucket["index"].values, size=take, replace=False)
        selected.extend(chosen.tolist())
        print(f"  [✓] {name} : {take} patients (pool de {len(bucket)})")

    return sorted(selected)


# ═══════════════════════════════════════════════════════════════════════════
# 5. HELPERS IRM (visualisation)
# ═══════════════════════════════════════════════════════════════════════════

VIEW_NAMES = ["coronal_1", "coronal_2", "axial_1", "axial_2"]


def get_raw_slices(scan_path: str, output_size: int = 448) -> List[np.ndarray]:
    """
    Charge les 4 coupes IRM brutes (pas de prompt, juste les images).
    Retourne 4 ndarrays uint8 de shape (output_size, output_size, 3).
    """
    extractor = MultiViewSliceExtractor(
        n_coronal=2, n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),
        output_size=output_size,
    )
    pil_slices = extractor.extract_all(scan_path)
    return [np.array(img) for img in pil_slices]


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Superpose une heatmap sur une image grayscale/RGB.

    Args:
        image    : (H, W) ou (H, W, 3) uint8
        heatmap  : (H', W') float, sera rescalée [0, 1] et redimensionnée à H,W
        alpha    : opacité du heatmap (0=image seule, 1=heatmap seule)
        colormap : nom du colormap matplotlib

    Retourne (H, W, 3) uint8.
    """
    import matplotlib.cm as cm
    from PIL import Image as PIL_Image

    # Image en RGB uint8
    if image.ndim == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    else:
        img_rgb = image
    if img_rgb.dtype != np.uint8:
        img_rgb = (
            (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-9) * 255
        ).astype(np.uint8)

    H, W = img_rgb.shape[:2]

    # Resize heatmap si nécessaire
    if heatmap.shape != (H, W):
        hm_pil = PIL_Image.fromarray(
            (heatmap * 255 / (heatmap.max() + 1e-9)).astype(np.uint8)
        )
        hm_pil = hm_pil.resize((W, H), PIL_Image.Resampling.BILINEAR)
        heatmap = np.array(hm_pil).astype(np.float32) / 255.0
    else:
        # Normalisation
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

    # Application colormap
    cmap = cm.get_cmap(colormap)
    hm_colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)

    # Superposition
    overlay = (
        (1 - alpha) * img_rgb.astype(np.float32)
        + alpha * hm_colored.astype(np.float32)
    ).astype(np.uint8)
    return overlay


def save_mosaic(
    images: List[np.ndarray],
    out_path: Path,
    titles: Optional[List[str]] = None,
    main_title: Optional[str] = None,
) -> None:
    """
    Sauve une mosaïque horizontale des 4 vues (avec titres optionnels).
    """
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(
        axes, images, titles or [f"View {i+1}" for i in range(n)],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    if main_title:
        fig.suptitle(main_title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. DÉTECTION COUCHES CIBLES POUR HOOKS
# ═══════════════════════════════════════════════════════════════════════════

def find_vision_encoder_last_layer(model) -> Optional[torch.nn.Module]:
    """
    Cherche la dernière couche du vision encoder MedSigLIP.

    Cherche typiquement :
        model.model.vision_tower.vision_model.encoder.layers[-1]
        ou variantes (peft model wrap les noms).
    Retourne None si non trouvé.
    """
    candidates = [
        "model.vision_tower.vision_model.encoder.layers",
        "vision_tower.vision_model.encoder.layers",
        "base_model.model.vision_tower.vision_model.encoder.layers",
    ]
    # Naviguer dans le modèle réel (peft wrap)
    target = model
    if hasattr(model, "model"):  # TfeMedGemmaCls / TfeMedGemmaWithMMSE
        target = model.model
    if hasattr(target, "model"):  # PeftModel
        target = target.model

    # Recherche par named_modules (plus robuste)
    for name, module in target.named_modules():
        if (
            "vision_tower" in name
            and "encoder.layers" in name
            and isinstance(module, torch.nn.ModuleList)
        ):
            return module[-1]

    # Fallback : la dernière ModuleList de toute la pile
    last_modlist = None
    for _, module in target.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            last_modlist = module
    return last_modlist[-1] if last_modlist is not None else None


def find_text_decoder_last_layer(model) -> Optional[torch.nn.Module]:
    """
    Cherche la dernière couche du décodeur texte (Gemma).
    Utile pour MMSE hook et attention rollout.
    """
    target = model
    if hasattr(target, "model"):
        target = target.model
    if hasattr(target, "model"):
        target = target.model

    for name, module in target.named_modules():
        if (
            ("language_model" in name or "text_model" in name)
            and "layers" in name
            and isinstance(module, torch.nn.ModuleList)
        ):
            return module[-1]

    # Fallback : dernière ModuleList globale
    last = None
    for _, mod in target.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            last = mod
    return last[-1] if last is not None else None


# ═══════════════════════════════════════════════════════════════════════════
# 7. UTILITAIRES POUR FORWARD PASS EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════

def prepare_inputs_for_forward(
    item: Dict[str, torch.Tensor], device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prépare un sample TfeDataset pour un forward pass.
    Ajoute la dim batch + token_type_ids + transferts GPU.
    """
    inputs = {
        "input_ids":      item["input_ids"].unsqueeze(0).to(device),
        "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
        "pixel_values":   item["pixel_values"].to(device),
    }
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    return inputs


def find_logit_position_for_answer(
    labels: torch.Tensor, cn_id: int, ad_id: int,
) -> Optional[int]:
    """
    Trouve logit_pos pour la première occurrence de CN ou AD dans les labels.
    Convention autoregressive : logit_pos = ans_pos - 1.
    """
    for pos in range(labels.shape[0]):
        tok = int(labels[pos])
        if tok in (cn_id, ad_id):
            return max(0, pos - 1)
    # Fallback : dernier token non-(-100)
    valid = (labels != -100).nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        return None
    return max(0, int(valid[-1]) - 1)


def get_ad_prob_from_logits(
    logits: torch.Tensor, logit_pos: int, cn_id: int, ad_id: int,
) -> float:
    """Softmax sur (CN, AD) au logit_pos. Retourne P(AD)."""
    cn_logit = float(logits[0, logit_pos, cn_id])
    ad_logit = float(logits[0, logit_pos, ad_id])
    exp_cn = np.exp(cn_logit)
    exp_ad = np.exp(ad_logit)
    return exp_ad / (exp_cn + exp_ad)