"""
feature_importance.py — Importance des features tabulaires.

Implémente DEUX approches complémentaires :

1. PERTURBATION (analyse globale, agrégée sur le test set entier)
   Pour chaque feature, on remplace sa valeur par la médiane train-only
   pour TOUS les patients, puis on mesure ΔAUC = AUC_baseline - AUC_perturbé.
   → Importance positive = la feature est utile au modèle.
   → Coût : 17 features × N patients = ~17 × 1213 forward = 5h environ.
            Avec --n-patients 200 → ~50 min.

2. LOCO leave-one-out (analyse par patient, fine-grain)
   Pour chaque patient × feature, on retire la feature du prompt et on
   mesure |Δprob_AD|. Donne une carte d'importance par patient.
   → Coût : 16 features × n_patients forward = identique à perturbation,
            mais on aggrège différemment.
   → Sortie : 1 ligne/patient × 1 colonne/feature, plus utilisable
              cliniquement ("pour ce patient X, retirer AGE change la
              prédiction de 0.18, donc la décision dépend fortement de l'âge").

Outputs (dans <ckpt>/explainability/feature_importance/) :
    perturbation_global.csv      # 1 ligne/feature : ΔAUC + IC95% bootstrap
    perturbation_global.png      # bar chart ranking
    loco_per_patient.csv         # 1 ligne/patient × 16 colonnes
    loco_summary.csv             # moyenne + std de |Δprob| par feature
    loco_ranking.png             # bar chart ranking LOCO

Lancement :
    python feature_importance.py --task 02_with_mmse --n-patients 100 \\
        --strategy stratified
    python feature_importance.py --task 01_no_mmse --features AGE BMI MMSE
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from _common import (
    add_common_args, resolve_checkpoint_and_config, resolve_output_dir,
    load_model_and_dataset, select_patients,
    prepare_inputs_for_forward, find_logit_position_for_answer,
    get_ad_prob_from_logits, PROJECT_ROOT,
)
from utils import release_gpu, save_metrics_json, load_config


# ═══════════════════════════════════════════════════════════════════════════
# 1. INFÉRENCE OPTIMISÉE
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_prob_ad(
    model, dataset, patient_idx: int, cn_id: int, ad_id: int,
    device: str = "cuda",
) -> Optional[float]:
    """Forward + softmax, retourne P(AD)."""
    item = dataset[patient_idx]
    inputs = prepare_inputs_for_forward(item, device=device)
    labels = item["labels"]
    logit_pos = find_logit_position_for_answer(labels, cn_id, ad_id)
    if logit_pos is None:
        return None
    out = model(**inputs, use_cache=False)
    return get_ad_prob_from_logits(out.logits, logit_pos, cn_id, ad_id)


# ═══════════════════════════════════════════════════════════════════════════
# 2. PERTURBATION (zero-out / médiane)
# ═══════════════════════════════════════════════════════════════════════════

def compute_train_medians(
    splits_dir: Path, fold: int, features: List[str],
) -> Dict[str, float]:
    """Médiane train-only de chaque feature (utilisée pour l'imputation)."""
    train_csv = splits_dir / f"fold_{fold}" / "train.csv"
    train_df = pd.read_csv(train_csv)
    medians: Dict[str, float] = {}
    for feat in features:
        if feat in train_df.columns:
            medians[feat] = float(train_df[feat].median())
    return medians


def predict_with_perturbed_feature(
    model, dataset, patient_indices: List[int],
    feature: str, replacement_value: float,
    cn_id: int, ad_id: int,
    device: str = "cuda",
) -> List[float]:
    """
    Pour chaque patient, remplace `feature` par `replacement_value` puis prédit.
    Retourne liste des P(AD).
    """
    probs: List[float] = []
    original_values = dataset.df[feature].copy()
    try:
        # Modification globale temporaire
        dataset.df[feature] = replacement_value
        for idx in patient_indices:
            p = predict_prob_ad(model, dataset, idx, cn_id, ad_id, device)
            probs.append(p if p is not None else 0.5)
    finally:
        dataset.df[feature] = original_values
    return probs


def run_perturbation_analysis(
    model, dataset, patient_indices: List[int],
    features: List[str], train_medians: Dict[str, float],
    cn_id: int, ad_id: int,
    output_dir: Path, n_bootstrap: int = 100,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Analyse globale par perturbation : ΔAUC = AUC_orig - AUC_perturbé.
    """
    print(f"\n[*] Perturbation : baseline...")
    baseline_probs: List[float] = []
    true_labels: List[int] = []
    for idx in tqdm(patient_indices, desc="Baseline"):
        p = predict_prob_ad(model, dataset, idx, cn_id, ad_id, device)
        baseline_probs.append(p if p is not None else 0.5)
        true_labels.append(int(dataset.df.iloc[idx]["label"]))

    baseline_probs_arr = np.asarray(baseline_probs)
    true_arr = np.asarray(true_labels)

    if len(set(true_labels)) < 2:
        raise ValueError("Pas de variation des labels — AUC indéfinie")
    baseline_auc = float(roc_auc_score(true_arr, baseline_probs_arr))
    print(f"    Baseline AUC : {baseline_auc:.4f}")

    # Pour chaque feature
    rng = np.random.RandomState(42)
    rows: List[Dict[str, Any]] = []
    for feat in tqdm(features, desc="Features"):
        if feat not in train_medians:
            print(f"  [!] {feat} : médiane train absente, skip")
            continue
        median_val = train_medians[feat]
        perturbed_probs = predict_with_perturbed_feature(
            model, dataset, patient_indices,
            feature=feat, replacement_value=median_val,
            cn_id=cn_id, ad_id=ad_id, device=device,
        )
        perturbed_arr = np.asarray(perturbed_probs)
        try:
            auc_perturbed = float(roc_auc_score(true_arr, perturbed_arr))
        except ValueError:
            auc_perturbed = float("nan")
        delta = baseline_auc - auc_perturbed

        # Bootstrap IC95% sur ΔAUC
        boot_deltas = []
        n_total = len(true_arr)
        for _ in range(n_bootstrap):
            boot_idx = rng.choice(n_total, size=n_total, replace=True)
            try:
                a_b = roc_auc_score(true_arr[boot_idx], baseline_probs_arr[boot_idx])
                a_p = roc_auc_score(true_arr[boot_idx], perturbed_arr[boot_idx])
                boot_deltas.append(a_b - a_p)
            except ValueError:
                continue
        if boot_deltas:
            lb = float(np.percentile(boot_deltas, 2.5))
            ub = float(np.percentile(boot_deltas, 97.5))
        else:
            lb, ub = float("nan"), float("nan")

        rows.append({
            "feature":          feat,
            "baseline_auc":     baseline_auc,
            "perturbed_auc":    auc_perturbed,
            "delta_auc":        delta,
            "delta_lb95":       lb,
            "delta_ub95":       ub,
            "replacement_val":  median_val,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("delta_auc", ascending=False).reset_index(drop=True)
    df.to_csv(output_dir / "perturbation_global.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.35)))
    y = np.arange(len(df))
    colors = ["#2C7BB6" if d > 0 else "#D7191C" for d in df["delta_auc"]]
    ax.barh(y, df["delta_auc"], xerr=[
        df["delta_auc"] - df["delta_lb95"],
        df["delta_ub95"] - df["delta_auc"],
    ], color=colors, ecolor="black", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("ΔAUC = AUC_baseline − AUC_perturbed")
    ax.set_title(
        f"Perturbation feature importance (baseline AUC = {baseline_auc:.4f})",
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "perturbation_global.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. LOCO (Leave-One-Feature-Out)
# ═══════════════════════════════════════════════════════════════════════════

def predict_without_feature(
    model, dataset, patient_idx: int, feature: str,
    cn_id: int, ad_id: int, device: str = "cuda",
) -> Optional[float]:
    """
    Retire `feature` de la liste tabular_features du dataset, prédit, restore.
    Le prompt sera reconstruit sans cette feature.
    """
    original_features = dataset.tabular_features.copy()
    try:
        dataset.tabular_features = [f for f in original_features if f != feature]
        prob = predict_prob_ad(model, dataset, patient_idx, cn_id, ad_id, device)
    finally:
        dataset.tabular_features = original_features
    return prob


def run_loco_analysis(
    model, dataset, patient_indices: List[int],
    features: List[str], cn_id: int, ad_id: int,
    output_dir: Path, device: str = "cuda",
) -> pd.DataFrame:
    """
    LOCO : pour chaque (patient, feature), retire la feature et mesure Δprob.
    """
    print(f"\n[*] LOCO : baselines par patient...")
    baselines: Dict[int, float] = {}
    for idx in tqdm(patient_indices, desc="Baselines"):
        p = predict_prob_ad(model, dataset, idx, cn_id, ad_id, device)
        baselines[idx] = p if p is not None else 0.5

    print(f"\n[*] LOCO : impact par feature...")
    rows: List[Dict[str, Any]] = []
    for idx in tqdm(patient_indices, desc="LOCO patients"):
        row_data: Dict[str, Any] = {
            "subject_id":  str(dataset.df.iloc[idx]["subject_id"]),
            "source":      str(dataset.df.iloc[idx].get("source", "?")),
            "true_label":  int(dataset.df.iloc[idx]["label"]),
            "baseline_prob": baselines[idx],
        }
        for feat in features:
            if feat not in dataset.tabular_features:
                row_data[f"delta_{feat}"] = float("nan")
                continue
            p_loco = predict_without_feature(
                model, dataset, idx, feat, cn_id, ad_id, device,
            )
            row_data[f"delta_{feat}"] = (
                (p_loco - baselines[idx]) if p_loco is not None else float("nan")
            )
        rows.append(row_data)

    loco_df = pd.DataFrame(rows)
    loco_df.to_csv(output_dir / "loco_per_patient.csv", index=False)

    # Résumé : moyenne, std, et |Δ| moyen par feature
    delta_cols = [c for c in loco_df.columns if c.startswith("delta_")]
    summary_rows = []
    for col in delta_cols:
        feat = col.replace("delta_", "")
        vals = loco_df[col].dropna().values
        if len(vals) == 0:
            continue
        summary_rows.append({
            "feature":       feat,
            "n":             len(vals),
            "mean_delta":    float(np.mean(vals)),
            "std_delta":     float(np.std(vals)),
            "mean_abs_delta": float(np.mean(np.abs(vals))),
            "max_abs_delta": float(np.max(np.abs(vals))),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        "mean_abs_delta", ascending=False
    ).reset_index(drop=True)
    summary_df.to_csv(output_dir / "loco_summary.csv", index=False)

    # Plot ranking
    fig, ax = plt.subplots(figsize=(9, max(4, len(summary_df) * 0.35)))
    y = np.arange(len(summary_df))
    ax.barh(y, summary_df["mean_abs_delta"],
            xerr=summary_df["std_delta"], color="#FDAE61",
            ecolor="black", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(summary_df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Δ P(AD)| (effet de retirer la feature, par patient)")
    ax.set_title(
        f"LOCO feature importance (n_patients = {len(loco_df)})",
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "loco_ranking.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Importance des features tabulaires (Perturbation + LOCO)"
    )
    add_common_args(parser)
    parser.add_argument("--features", type=str, nargs="+", default=None,
                        help="Subset de features (défaut : toutes celles "
                             "actives dans la config)")
    parser.add_argument("--n-bootstrap", type=int, default=100,
                        help="Bootstraps IC95% sur ΔAUC")
    parser.add_argument("--skip-perturbation", action="store_true")
    parser.add_argument("--skip-loco", action="store_true")
    args = parser.parse_args()

    try:
        checkpoint_path, config_path = resolve_checkpoint_and_config(args)
        output_dir = resolve_output_dir(args, checkpoint_path, "feature_importance")

        print(f"\n{'='*70}")
        print(f"  FEATURE IMPORTANCE (Perturbation + LOCO)")
        print(f"{'='*70}")
        print(f"  Checkpoint : {checkpoint_path}")
        print(f"  Strategy   : {args.strategy} (n={args.n_patients})")
        print(f"  Output     : {output_dir}")
        print(f"{'='*70}\n")

        processor, model, dataset, cn_id, ad_id, _ = load_model_and_dataset(
            config_path, checkpoint_path,
            split=args.split, fold=args.fold,
            is_training_for_dataset=True,
        )
        model.eval()

        config = load_config(str(config_path))
        features = args.features or list(dataset.tabular_features)
        if not features:
            print("[!] Aucune feature à analyser (mode minimal ?)")
            return 1
        print(f"[*] {len(features)} features : {features[:5]}{'...' if len(features) > 5 else ''}")

        selected_idx = select_patients(
            dataset.df, n_patients=args.n_patients,
            strategy=args.strategy,
            predictions_csv=args.predictions_csv, seed=args.seed,
            filter_real_only=args.filter_real_only,
        )
        print(f"[*] Patients sélectionnés : {len(selected_idx)}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        results_summary: Dict[str, Any] = {
            "n_patients": len(selected_idx),
            "n_features": len(features),
            "checkpoint": str(checkpoint_path),
        }

        # ── Perturbation globale ─────────────────────────────────────────
        if not args.skip_perturbation:
            splits_dir = Path(config["data"]["splits_dir"])
            if not splits_dir.is_absolute():
                splits_dir = (PROJECT_ROOT / splits_dir).resolve()
            train_medians = compute_train_medians(splits_dir, args.fold, features)
            print(f"\n[*] Médianes train-only calculées pour {len(train_medians)} features")

            pert_df = run_perturbation_analysis(
                model, dataset, selected_idx,
                features=features, train_medians=train_medians,
                cn_id=cn_id, ad_id=ad_id,
                output_dir=output_dir,
                n_bootstrap=args.n_bootstrap, device=device,
            )
            results_summary["perturbation_top5"] = pert_df.head(5).to_dict("records")
            print(f"\n[✓] Perturbation : perturbation_global.csv + .png")

        # ── LOCO ─────────────────────────────────────────────────────────
        if not args.skip_loco:
            loco_summary = run_loco_analysis(
                model, dataset, selected_idx,
                features=features, cn_id=cn_id, ad_id=ad_id,
                output_dir=output_dir, device=device,
            )
            results_summary["loco_top5"] = loco_summary.head(5).to_dict("records")
            print(f"\n[✓] LOCO : loco_per_patient.csv + loco_summary.csv + ranking.png")

        save_metrics_json(results_summary, output_dir / "summary.json")
        print(f"\n[✓] Tous les résultats dans {output_dir}")
        return 0

    except Exception as e:
        print(f"\n[!] Erreur : {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1
    finally:
        try:
            release_gpu()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())