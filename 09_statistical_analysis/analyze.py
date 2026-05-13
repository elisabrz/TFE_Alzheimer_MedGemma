"""
analyze.py — Tâche 9 : Analyse statistique des features cliniques.

Trois capacités principales :
  1. Filtre des données IMPUTÉES : seules les valeurs réellement mesurées sont
     prises en compte dans tous les tests statistiques (Mann-Whitney, Chi²,
     Kruskal-Wallis, Spearman, Shapiro). Les valeurs imputées par médiane train
     sont des constantes qui biaisent les statistiques (effet artificiellement
     amplifié, corrélations dégradées). Détection via colonnes `<feature>_imputed`
     (générées par prepare_splits.py).

  2. Multi-fold : par défaut tourne sur les 5 folds (0-4) et agrège la variance
     des métriques (effet, p-value, decay) à travers les folds. Permet d'avoir
     une estimation robuste vs un single-fold biaisé.

  3. Détection de biais (paradoxe de Simpson) :
       decay = |effet_global| - mean_pondéré(|effet_intra-cohorte|)
       decay > 0.12  → biais fortuit (effet global plus fort que somme des intras)
       decay > 0.04  → biais ambigu

Pour CHAQUE feature des 16 (ou sous-ensemble configuré) :
    1. Test global CN vs AD (Mann-Whitney pour continues, Chi² pour catégorielles)
       sur valeurs RÉELLES uniquement
    2. Tests intra-cohorte (ADNI/NACC/OASIS) sur valeurs réelles
    3. Test inter-cohorte (Kruskal-Wallis : la cohorte influence-t-elle la feature ?)
    4. Taille d'effet rank-biserial avec IC95% bootstrap
    5. Correction Holm-Bonferroni sur l'ensemble des p-values

Lancement (depuis 09_statistical_analysis/) :
    python analyze.py                         # défaut : 5 folds, split=train
    python analyze.py --folds 0               # single fold (rapide)
    python analyze.py --folds 0 1 2           # subset de folds
    python analyze.py --features AGE BMI      # subset de features
    python analyze.py --split all             # train+val+test mergés (pas de variance)
    python analyze.py --n-bootstrap 1000      # bootstrap plus précis (lent)

Outputs (dans results/09_statistical_analysis/) :
    multifold_summary.csv         : 1 ligne/feature, mean±std des métriques sur 5 folds
    multifold_decay_overview.pdf  : barplot decay moyen avec error bars
    per_fold/fold_X/              : artefacts complets de chaque fold
        ├── summary_all_features.csv
        ├── decay_table.csv
        ├── holm_corrected_pvalues.csv
        ├── descriptive_table.csv
        ├── shapiro_normality.csv
        ├── summary_report.json
        └── figures/
    report.md                     : rapport multi-fold synthétisé pour le TFE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Imports locaux ─────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config, save_metrics_json


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIG FEATURES
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_META: Dict[str, Dict[str, Any]] = {
    # Démographie (4)
    "AGE":       {"type": "continuous",  "label": "Age (years)"},
    "PTGENDER":  {"type": "binary",      "label": "Gender (1=M, 2=F)"},
    "PTEDUCAT":  {"type": "continuous",  "label": "Education (years)"},
    "PTMARRY":   {"type": "categorical", "label": "Marital status"},
    # Neuropsych (6)
    "CATANIMSC": {"type": "continuous",  "label": "Animal Fluency (raw)"},
    "TRAASCOR":  {"type": "continuous",  "label": "Trail Making A (sec)"},
    "TRABSCOR":  {"type": "continuous",  "label": "Trail Making B (sec)"},
    "DSPANFOR":  {"type": "continuous",  "label": "Digit Span Forward"},
    "DSPANBAC":  {"type": "continuous",  "label": "Digit Span Backward"},
    "BNTTOTAL":  {"type": "continuous",  "label": "Boston Naming"},
    # Anthropométrie (2)
    "BMI":       {"type": "continuous",  "label": "BMI"},
    "VSWEIGHT":  {"type": "continuous",  "label": "Weight (kg)"},
    # Antécédents médicaux (4)
    "MH14ALCH":  {"type": "binary",      "label": "Alcohol use"},
    "MH16SMOK":  {"type": "binary",      "label": "Smoking"},
    "MH4CARD":   {"type": "binary",      "label": "Cardiovascular hist."},
    "MH2NEURL":  {"type": "binary",      "label": "Neurological hist."},
    # MMSE (cible — pas d'analyse mais affichée pour référence)
    "mmse_score": {"type": "continuous", "label": "MMSE score"},
    # Hippocampus volume (présent dans CSV, ADNI uniquement)
    "HIPPOCAMPUS_VOL": {"type": "continuous", "label": "Hippocampus volume (mm³)"},
}

THRESHOLDS = {
    "decay_fortuit":  0.12,
    "decay_ambigu":   0.04,
    "min_group_size": 20,
    "n_bootstrap":    500,
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. FILTRE DONNÉES RÉELLES (NON IMPUTÉES)
# ═══════════════════════════════════════════════════════════════════════════

def _get_real_mask(df: pd.DataFrame, feature: str) -> np.ndarray:
    """
    Retourne un masque booléen identifiant les lignes où la valeur de `feature`
    est RÉELLEMENT mesurée (par opposition à imputée par médiane train).

    Détection (en cascade) :
      1. Si colonne `<feature>_imputed` existe (pipeline T0 nouveau) :
         mask = (imputed == 0) ∧ notna(feature)
      2. Cas spécial mmse_score avec colonne `has_real_measures` (rétro-compat) :
         mask = (has_real_measures == 1) ∧ notna(feature)
      3. Fallback : mask = notna(feature) seul (compat anciens CSV)

    Pourquoi : les valeurs imputées sont des constantes (médiane train) qui
    sur-représentent une seule valeur. Inclure ces valeurs dans Mann-Whitney,
    Chi², ou Shapiro biaise massivement les statistiques.
    """
    if feature not in df.columns:
        return np.zeros(len(df), dtype=bool)

    base_mask = df[feature].notna().values

    imputed_col = f"{feature}_imputed"
    if imputed_col in df.columns:
        not_imputed = (df[imputed_col].fillna(0).astype(int) == 0).values
        return base_mask & not_imputed

    if feature == "mmse_score" and "has_real_measures" in df.columns:
        has_real = (df["has_real_measures"].fillna(0).astype(int) == 1).values
        return base_mask & has_real

    return base_mask


def _make_real_only_df(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Retourne une copie du DataFrame où les valeurs imputées sont remplacées par
    NaN. Utilisé pour Spearman correlation (gestion native NaN pairwise) et
    Shapiro (drop NaN par feature).

    Ne modifie PAS les lignes (n_lignes inchangé) — juste les valeurs.
    Les colonnes 'label' et 'source' sont préservées pour la stratification.
    """
    df_real = df.copy()
    for feat in features:
        if feat not in df_real.columns:
            continue
        real_mask = _get_real_mask(df, feat)
        # Inverse : imputed_or_nan = ~real_mask
        df_real.loc[~real_mask, feat] = np.nan
    return df_real


# ═══════════════════════════════════════════════════════════════════════════
# 3. STATS FONDAMENTALES (inchangées)
# ═══════════════════════════════════════════════════════════════════════════

def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Correction Holm-Bonferroni — contrôle FWER, plus puissante que Bonferroni."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    prev_corrected = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        cf = n - rank
        cp = min(p * cf, 1.0)
        cp = max(cp, prev_corrected)
        corrected[orig_idx] = cp
        prev_corrected = cp
    return corrected


def rank_biserial_ci(
    x: np.ndarray, y: np.ndarray,
    n_bootstrap: int = 500, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Rank-biserial r de Mann-Whitney + IC95% bootstrap."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return 0.0, -1.0, 1.0, 1.0
    nx, ny = len(x), len(y)
    try:
        u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        return 0.0, -1.0, 1.0, 1.0
    rb = 1.0 - (2.0 * u_stat) / (nx * ny)
    rng = np.random.RandomState(seed)
    rb_boot = []
    for _ in range(n_bootstrap):
        xb = rng.choice(x, size=nx, replace=True)
        yb = rng.choice(y, size=ny, replace=True)
        try:
            ub_stat, _ = mannwhitneyu(xb, yb, alternative="two-sided")
            rb_boot.append(1.0 - (2.0 * ub_stat) / (nx * ny))
        except Exception:
            rb_boot.append(rb)
    lb = float(np.percentile(rb_boot, 100 * alpha / 2))
    ub = float(np.percentile(rb_boot, 100 * (1 - alpha / 2)))
    return float(rb), lb, ub, float(p_val)


def chi2_test_binary(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Chi² Pearson features binaires CN vs AD. Retourne (cramers_v, p_value)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return 0.0, 1.0
    labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    values = np.concatenate([x, y])
    try:
        mask = ~np.isnan(values)
        labels, values = labels[mask], values[mask]
        if len(labels) < 4:
            return 0.0, 1.0
        contingency = pd.crosstab(labels.astype(int), values.astype(int))
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return 0.0, 1.0
        chi2, p, dof, _ = chi2_contingency(contingency)
        n = len(labels)
        cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))
        return cramers_v, float(p)
    except Exception:
        return 0.0, 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. ANALYSE PAR FEATURE (filtre données réelles)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_feature(
    df: pd.DataFrame, feature: str,
    n_bootstrap: int = 500, seed: int = 42,
) -> Dict[str, Any]:
    """
    Pour une feature donnée, calcule les statistiques uniquement sur les
    valeurs RÉELLEMENT MESURÉES (excluant les imputations).

    Retourne dict avec :
      - n_real_cn / n_real_ad : nombre de valeurs réelles utilisées
      - n_imputed : nombre de valeurs imputées exclues
      - effect_global, lb_global, ub_global, p_global
      - cohorts : dict ADNI/NACC/OASIS avec stats intra
      - decay : signal de paradoxe Simpson
    """
    if feature not in df.columns:
        return {"feature": feature, "error": "Feature absente du DataFrame"}

    meta = FEATURE_META.get(feature, {"type": "continuous", "label": feature})
    is_binary = (meta["type"] == "binary")

    # ── Filtre données réelles ────────────────────────────────────────────
    real_mask = _get_real_mask(df, feature)
    n_total = len(df)
    n_real = int(real_mask.sum())
    n_imputed = int(df[feature].notna().sum() - n_real)  # imputés = notna mais pas réels
    df_real = df[real_mask].copy()

    cn_global = df_real[df_real["label"] == 0][feature].values
    ad_global = df_real[df_real["label"] == 1][feature].values

    if len(cn_global) < 5 or len(ad_global) < 5:
        return {
            "feature": feature, "label": meta["label"],
            "type": meta["type"],
            "n_real_cn": int(len(cn_global)),
            "n_real_ad": int(len(ad_global)),
            "n_imputed_excluded": n_imputed,
            "error": (
                f"Trop peu de valeurs réelles : CN={len(cn_global)}, "
                f"AD={len(ad_global)} ({n_imputed} imputées exclues)"
            ),
        }

    # ── Test global ───────────────────────────────────────────────────────
    if is_binary:
        effect_global, p_global = chi2_test_binary(cn_global, ad_global)
        lb_global, ub_global = float("nan"), float("nan")
    else:
        effect_global, lb_global, ub_global, p_global = rank_biserial_ci(
            cn_global, ad_global, n_bootstrap=n_bootstrap, seed=seed
        )

    # ── Tests intra-cohorte ───────────────────────────────────────────────
    cohort_results: Dict[str, Dict[str, Any]] = {}
    cohorts_in_df = (
        sorted(df_real["source"].dropna().unique())
        if "source" in df_real.columns else []
    )
    for cohort in cohorts_in_df:
        sub = df_real[df_real["source"] == cohort]
        cn_sub = sub[sub["label"] == 0][feature].values
        ad_sub = sub[sub["label"] == 1][feature].values

        if (len(cn_sub) < THRESHOLDS["min_group_size"]
                or len(ad_sub) < THRESHOLDS["min_group_size"]):
            cohort_results[cohort] = {
                "n_cn": int(len(cn_sub)), "n_ad": int(len(ad_sub)),
                "skipped": True,
                "skip_reason": f"insufficient real values (need ≥{THRESHOLDS['min_group_size']})",
            }
            continue

        if is_binary:
            eff, p = chi2_test_binary(cn_sub, ad_sub)
            lb, ub = float("nan"), float("nan")
        else:
            eff, lb, ub, p = rank_biserial_ci(
                cn_sub, ad_sub, n_bootstrap=min(n_bootstrap, 200), seed=seed
            )
        cohort_results[cohort] = {
            "n_cn":   int(len(cn_sub)),
            "n_ad":   int(len(ad_sub)),
            "effect": float(eff),
            "lb":     float(lb) if not np.isnan(lb) else None,
            "ub":     float(ub) if not np.isnan(ub) else None,
            "p":      float(p),
            "skipped": False,
        }

    # ── Test inter-cohorte (Kruskal-Wallis sur la feature, valeurs réelles) ──
    inter_p = float("nan")
    if len(cohorts_in_df) >= 2 and not is_binary:
        groups = [
            df_real[df_real["source"] == c][feature].values
            for c in cohorts_in_df
        ]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) >= 2:
            try:
                _, inter_p = kruskal(*groups)
                inter_p = float(inter_p)
            except Exception:
                pass

    # ── DECAY (paradoxe de Simpson) ───────────────────────────────────────
    valid_cohorts = [c for c, r in cohort_results.items() if not r.get("skipped")]
    if valid_cohorts:
        weights = np.array([
            cohort_results[c]["n_cn"] + cohort_results[c]["n_ad"]
            for c in valid_cohorts
        ], dtype=float)
        intra_effects = np.array([
            abs(cohort_results[c]["effect"]) for c in valid_cohorts
        ])
        weights /= weights.sum()
        weighted_intra = float(np.sum(weights * intra_effects))
        decay = abs(effect_global) - weighted_intra
    else:
        weighted_intra = float("nan")
        decay = float("nan")

    if np.isnan(decay):
        decay_class = "indeterminé"
    elif abs(decay) > THRESHOLDS["decay_fortuit"]:
        decay_class = "FORTUIT (Simpson)"
    elif abs(decay) > THRESHOLDS["decay_ambigu"]:
        decay_class = "ambigu"
    else:
        decay_class = "intrinsèque"

    return {
        "feature":            feature,
        "label":              meta["label"],
        "type":               meta["type"],
        "n_real_cn":          int(len(cn_global)),
        "n_real_ad":          int(len(ad_global)),
        "n_imputed_excluded": n_imputed,
        "real_pct":           float(n_real / n_total * 100) if n_total > 0 else 0.0,
        "effect_global":      float(effect_global),
        "lb_global":          float(lb_global) if not np.isnan(lb_global) else None,
        "ub_global":          float(ub_global) if not np.isnan(ub_global) else None,
        "p_global":           float(p_global),
        "weighted_intra":     weighted_intra,
        "decay":              float(decay) if not np.isnan(decay) else None,
        "decay_class":        decay_class,
        "p_inter_cohorte":    inter_p,
        "cohorts":            cohort_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_distribution(
    df: pd.DataFrame, feature: str, out_path: Path, label: str = None,
) -> Path:
    """
    Distribution par classe CN/AD sur valeurs RÉELLES uniquement.
    Continue : raincloud (violin + box + strip)
    Binaire  : barplot proportions par cohorte
    """
    meta = FEATURE_META.get(feature, {"type": "continuous"})
    is_binary = (meta["type"] == "binary")
    label = label or feature
    out_path.parent.mkdir(parents=True, exist_ok=True)

    real_mask = _get_real_mask(df, feature)
    df_plot = df[real_mask].copy()
    if len(df_plot) == 0:
        return out_path
    df_plot["label_str"] = df_plot["label"].map({0: "CN", 1: "AD"})

    sns.set_theme(style="white")
    palette = {"CN": "#4472C4", "AD": "#ED7D31"}

    if is_binary:
        fig, ax = plt.subplots(figsize=(7, 5))
        if "source" in df_plot.columns:
            ct = (
                df_plot.groupby(["source", "label_str"])[feature]
                .apply(lambda x: x.dropna().mean())
                .reset_index()
            )
            sns.barplot(
                data=ct, x="source", y=feature, hue="label_str",
                palette=palette, ax=ax,
            )
            ax.set_xlabel("Cohorte")
        else:
            ct = df_plot.groupby("label_str")[feature].apply(
                lambda x: x.dropna().mean()
            ).reset_index()
            sns.barplot(data=ct, x="label_str", y=feature, palette=palette, ax=ax)
        ax.set_ylabel(f"Proportion ({label} = 1)")
        ax.set_title(
            f"{label} — Proportion par classe et cohorte (valeurs réelles, n={len(df_plot)})",
            fontweight="bold",
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.violinplot(
            x="label_str", y=feature, data=df_plot, palette=palette,
            inner=None, ax=ax, alpha=0.4,
        )
        sns.boxplot(
            x="label_str", y=feature, data=df_plot, width=0.15,
            color="white", ax=ax, linewidth=1.5, showfliers=False,
        )
        sns.stripplot(
            x="label_str", y=feature, data=df_plot, size=2,
            palette=palette, alpha=0.15, jitter=True, ax=ax,
        )
        ax.set_xlabel("")
        ax.set_xticklabels(["Contrôles (CN)", "Alzheimer (AD)"])
        ax.set_ylabel(label)
        ax.set_title(
            f"{label} — Distribution par classe (valeurs réelles, n={len(df_plot)})",
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_decay_overview(results: List[Dict[str, Any]], out_path: Path) -> Path:
    """Barplot du decay pour toutes les features (single-fold)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if r.get("decay") is not None and "error" not in r]
    if not valid:
        return out_path
    valid.sort(key=lambda r: abs(r["decay"]), reverse=True)
    feats = [r["feature"] for r in valid]
    decays = [r["decay"] for r in valid]
    classes = [r["decay_class"] for r in valid]
    color_map = {
        "FORTUIT (Simpson)": "#D7191C",
        "ambigu":            "#FDAE61",
        "intrinsèque":       "#2C7BB6",
        "indeterminé":       "#888888",
    }
    colors = [color_map.get(c, "#888888") for c in classes]

    fig, ax = plt.subplots(figsize=(10, max(4, len(feats) * 0.4)))
    y_pos = np.arange(len(feats))
    ax.barh(y_pos, decays, color=colors)
    ax.axvline(x=THRESHOLDS["decay_fortuit"], color="red", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=-THRESHOLDS["decay_fortuit"], color="red", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=THRESHOLDS["decay_ambigu"], color="orange", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=-THRESHOLDS["decay_ambigu"], color="orange", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=0, color="black", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Decay = |effet global| - moyenne pondérée(|effet intra|)")
    ax.set_title("Détection des biais de Simpson — Decay par feature", fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color, label=label)
        for label, color in color_map.items()
        if label in set(classes)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_decay_with_variance(agg_df: pd.DataFrame, out_path: Path) -> Path:
    """
    Barplot du decay MOYEN sur N folds avec error bars (±std).
    Utilisé en mode multi-fold pour visualiser la stabilité de la détection
    de biais de Simpson à travers les splits.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "decay_mean" not in agg_df.columns:
        return out_path

    valid = agg_df[agg_df["decay_mean"].notna()].copy()
    if len(valid) == 0:
        return out_path
    valid["decay_abs"] = valid["decay_mean"].abs()
    valid = valid.sort_values("decay_abs", ascending=False)

    feats = valid["feature"].tolist()
    means = valid["decay_mean"].values
    stds  = valid["decay_std"].fillna(0).values
    classes = valid["decay_class_mode"].tolist()
    n_folds = valid["n_folds"].iloc[0] if len(valid) else 0

    color_map = {
        "FORTUIT (Simpson)": "#D7191C",
        "ambigu":            "#FDAE61",
        "intrinsèque":       "#2C7BB6",
        "indeterminé":       "#888888",
    }
    colors = [color_map.get(c, "#888888") for c in classes]

    fig, ax = plt.subplots(figsize=(11, max(4, len(feats) * 0.4)))
    y_pos = np.arange(len(feats))
    ax.barh(
        y_pos, means, xerr=stds,
        color=colors, ecolor="black", capsize=3,
        error_kw={"linewidth": 1.0, "alpha": 0.7},
    )
    ax.axvline(x=THRESHOLDS["decay_fortuit"], color="red", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=-THRESHOLDS["decay_fortuit"], color="red", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=THRESHOLDS["decay_ambigu"], color="orange", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=-THRESHOLDS["decay_ambigu"], color="orange", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(x=0, color="black", lw=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.set_xlabel(
        f"Decay = |effet global| - moyenne pondérée(|effet intra|)\n"
        f"Mean ± std sur {int(n_folds)} folds"
    )
    ax.set_title(
        f"Détection des biais de Simpson — Variance sur {int(n_folds)} folds",
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color, label=label)
        for label, color in color_map.items()
        if label in set(classes)
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=9,
        title="Classification dominante",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 6. ANALYSES GLOBALES (sur valeurs réelles uniquement)
# ═══════════════════════════════════════════════════════════════════════════

def run_shapiro_tests(
    df: pd.DataFrame, features: List[str],
) -> pd.DataFrame:
    """Test de normalité Shapiro-Wilk sur valeurs RÉELLES de chaque feature continue."""
    rows: List[Dict[str, Any]] = []
    for feat in features:
        if feat not in df.columns:
            continue
        meta = FEATURE_META.get(feat, {"type": "continuous"})
        if meta["type"] != "continuous":
            continue
        real_mask = _get_real_mask(df, feat)
        vals = df.loc[real_mask, feat].values
        if len(vals) < 3:
            continue
        if len(vals) > 5000:
            rng = np.random.RandomState(42)
            vals = rng.choice(vals, size=5000, replace=False)
        try:
            W, p = shapiro(vals)
            rows.append({
                "feature":      feat,
                "label":        meta.get("label", feat),
                "n_real":       int(len(vals)),
                "shapiro_W":    float(W),
                "shapiro_p":    float(p),
                "is_normal_05": bool(p > 0.05),
            })
        except Exception as e:
            rows.append({"feature": feat, "error": str(e)})
    return pd.DataFrame(rows)


def plot_correlation_matrix(
    df: pd.DataFrame, features: List[str], out_path: Path,
) -> Path:
    """
    Heatmap corrélation Spearman des features continues sur valeurs RÉELLES.
    Utilise pairwise complete observations (NaN auto-géré par pandas).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cont_features = [
        f for f in features
        if f in df.columns
        and FEATURE_META.get(f, {}).get("type") == "continuous"
    ]
    if len(cont_features) < 2:
        return out_path

    # Remplace les valeurs imputées par NaN pour que .corr() utilise pairwise complete obs
    df_real = _make_real_only_df(df, cont_features)
    corr = df_real[cont_features].corr(method="spearman")

    label_map = {f: FEATURE_META.get(f, {}).get("label", f) for f in cont_features}
    corr_labeled = corr.rename(index=label_map, columns=label_map)

    n = len(cont_features)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.85), max(8, n * 0.78)))
    sns.heatmap(
        corr_labeled, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.4, cbar_kws={"shrink": 0.7},
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title(
        "Corrélation Spearman des features continues (valeurs réelles, pairwise complete)",
        fontweight="bold", pad=10,
    )
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(str(out_path).replace(".png", ".csv"))
    return out_path


def plot_pca_features(
    df: pd.DataFrame, features: List[str], out_path: Path,
) -> Path:
    """
    PCA 2D des features cliniques avec coloration CN/AD.

    ⚠️ IMPUTATION POUR VISUALISATION : la PCA ne tolère pas les NaN. Utiliser
    uniquement des valeurs réelles ferait perdre ~99% des sujets NACC/OASIS.
    On impute donc par médiane train de la feature pour cette visualisation —
    à ne PAS interpréter comme un test statistique.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid_features = [
        f for f in features
        if f in df.columns
        and FEATURE_META.get(f, {}).get("type") in ("continuous", "binary")
    ]
    if len(valid_features) < 3:
        return out_path

    # Impute pour viz (NaN-safe)
    df_viz = df[valid_features + ["label"]].copy()
    df_viz[valid_features] = df_viz[valid_features].fillna(df_viz[valid_features].median())
    sub = df_viz.dropna()
    if len(sub) < 10:
        return out_path

    X = sub[valid_features].values
    y = sub["label"].astype(int).values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    cn_mask = (y == 0)
    ad_mask = (y == 1)
    ax.scatter(X_pca[cn_mask, 0], X_pca[cn_mask, 1], s=12, alpha=0.4,
               c="#2C7BB6", label=f"CN (n={cn_mask.sum()})")
    ax.scatter(X_pca[ad_mask, 0], X_pca[ad_mask, 1], s=12, alpha=0.4,
               c="#D7191C", label=f"AD (n={ad_mask.sum()})")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title(
        "PCA des features cliniques — projection CN vs AD\n"
        "(imputation médiane pour viz uniquement)",
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    loadings = pd.DataFrame(
        pca.components_.T, columns=["PC1", "PC2"], index=valid_features,
    )
    loadings["PC1_abs"] = loadings["PC1"].abs()
    loadings = loadings.sort_values("PC1_abs", ascending=False).drop(columns=["PC1_abs"])
    loadings.to_csv(str(out_path).replace(".png", "_loadings.csv"))
    return out_path


def plot_tsne_features(
    df: pd.DataFrame, features: List[str], out_path: Path,
    perplexity: int = 30, seed: int = 42,
) -> Path:
    """
    t-SNE 2D des features cliniques. Comme PCA, impute pour viz uniquement.
    """
    from sklearn.manifold import TSNE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid_features = [
        f for f in features
        if f in df.columns
        and FEATURE_META.get(f, {}).get("type") in ("continuous", "binary")
    ]
    if len(valid_features) < 3:
        return out_path

    has_source = "source" in df.columns
    cols = valid_features + ["label"] + (["source"] if has_source else [])
    df_viz = df[cols].copy()
    df_viz[valid_features] = df_viz[valid_features].fillna(df_viz[valid_features].median())
    sub = df_viz.dropna()
    if len(sub) < 20:
        return out_path

    X = sub[valid_features].values
    y = sub["label"].astype(int).values
    X_scaled = StandardScaler().fit_transform(X)

    print(f"  [t-SNE] Calcul sur {len(sub)} patients (perplexity={perplexity})...",
          flush=True)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_jobs=1)
    X_tsne = tsne.fit_transform(X_scaled)

    n_panels = 2 if has_source else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    cn_m = (y == 0); ad_m = (y == 1)
    ax.scatter(X_tsne[cn_m, 0], X_tsne[cn_m, 1], s=5, alpha=0.3,
               c="#2C7BB6", label=f"CN (n={cn_m.sum()})", rasterized=True)
    ax.scatter(X_tsne[ad_m, 0], X_tsne[ad_m, 1], s=5, alpha=0.3,
               c="#D7191C", label=f"AD (n={ad_m.sum()})", rasterized=True)
    ax.set_title("t-SNE — CN vs AD", fontweight="bold")
    ax.legend(loc="best", markerscale=3)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    if has_source:
        ax2 = axes[1]
        cohort_colors = {"ADNI": "#2ecc71", "NACC": "#e67e22", "OASIS": "#9b59b6"}
        sources = sub["source"].values
        for cohort, color in cohort_colors.items():
            m = (sources == cohort)
            if m.sum() == 0:
                continue
            ax2.scatter(X_tsne[m, 0], X_tsne[m, 1], s=5, alpha=0.3,
                        c=color, label=f"{cohort} (n={m.sum()})", rasterized=True)
        ax2.set_title("t-SNE — par cohorte", fontweight="bold")
        ax2.legend(loc="best", markerscale=3)
        ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.suptitle(
        f"t-SNE des features cliniques (imputation médiane pour viz, "
        f"{len(valid_features)} features, perplexity={perplexity})",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_descriptive_table(
    df: pd.DataFrame, features: List[str],
) -> pd.DataFrame:
    """
    Table 1 du TFE : statistiques descriptives par cohorte (valeurs RÉELLES uniquement).
    Continues : médiane [Q1, Q3] (n_real)
    Binaires  : n_pos (% n_real)
    """
    rows: List[Dict[str, Any]] = []
    cohorts_to_iterate = ["GLOBAL"]
    if "source" in df.columns:
        cohorts_to_iterate += sorted(df["source"].dropna().unique())

    for cohort in cohorts_to_iterate:
        sub = df.copy() if cohort == "GLOBAL" else df[df["source"] == cohort].copy()
        if len(sub) == 0:
            continue

        n = len(sub)
        n_cn = int((sub["label"] == 0).sum()) if "label" in sub.columns else 0
        n_ad = int((sub["label"] == 1).sum()) if "label" in sub.columns else 0
        row: Dict[str, Any] = {
            "cohort":  cohort,
            "n":       n,
            "n_CN":    n_cn,
            "n_AD":    n_ad,
            "pct_AD":  f"{n_ad / n * 100:.1f}%" if n > 0 else "NA",
        }

        for feat in features:
            if feat not in sub.columns:
                continue
            real_mask = _get_real_mask(sub, feat)
            vals = sub.loc[real_mask, feat]
            n_real_feat = len(vals)
            if n_real_feat == 0:
                row[feat] = "NA"
                continue

            meta = FEATURE_META.get(feat, {"type": "continuous"})
            if meta["type"] == "continuous":
                med = float(vals.median())
                q1 = float(vals.quantile(0.25))
                q3 = float(vals.quantile(0.75))
                row[feat] = f"{med:.1f} [{q1:.1f}, {q3:.1f}] (n={n_real_feat})"
            elif meta["type"] == "binary":
                n_pos = int((vals == 1).sum())
                row[feat] = f"{n_pos}/{n_real_feat} ({n_pos / n_real_feat * 100:.1f}%)"
            else:
                mode = vals.mode().iloc[0] if len(vals) > 0 else "NA"
                row[feat] = f"mode={mode} (n={n_real_feat})"
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 7. AGRÉGATION MULTI-FOLD
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_fold_results(
    fold_results: Dict[int, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Agrège les résultats par feature à travers les folds.
    Pour chaque feature : mean ± std de effect, decay, p_global, n_real ;
    et compteur des classifications de decay (FORTUIT / ambigu / intrinsèque).
    """
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    for fold, results in fold_results.items():
        for r in results:
            if "error" in r:
                continue
            feat = r["feature"]
            by_feature.setdefault(feat, []).append({**r, "_fold": fold})

    rows = []
    for feat, fold_list in by_feature.items():
        effects = np.array([r["effect_global"] for r in fold_list])
        ps = np.array([r["p_global"] for r in fold_list])
        n_cns = np.array([r["n_real_cn"] for r in fold_list])
        n_ads = np.array([r["n_real_ad"] for r in fold_list])
        decays = np.array([
            r["decay"] for r in fold_list if r.get("decay") is not None
        ])
        decay_classes = [r["decay_class"] for r in fold_list]
        class_counts = Counter(decay_classes)

        n_folds = len(fold_list)
        std_eff = float(np.std(effects, ddof=1)) if n_folds > 1 else 0.0
        std_dec = float(np.std(decays, ddof=1)) if len(decays) > 1 else 0.0

        rows.append({
            "feature":             feat,
            "label":               fold_list[0].get("label", feat),
            "type":                fold_list[0].get("type", "?"),
            "n_folds":             n_folds,
            "n_real_cn_mean":      float(n_cns.mean()),
            "n_real_ad_mean":      float(n_ads.mean()),
            "effect_mean":         float(effects.mean()),
            "effect_std":          std_eff,
            "effect_min":          float(effects.min()),
            "effect_max":          float(effects.max()),
            "p_mean":              float(ps.mean()),
            "p_max":               float(ps.max()),
            "n_folds_sig_p_raw":   int((ps < 0.05).sum()),
            "decay_mean":          float(decays.mean()) if len(decays) > 0 else None,
            "decay_std":           std_dec,
            "decay_min":           float(decays.min()) if len(decays) > 0 else None,
            "decay_max":           float(decays.max()) if len(decays) > 0 else None,
            "n_folds_FORTUIT":     class_counts.get("FORTUIT (Simpson)", 0),
            "n_folds_ambigu":      class_counts.get("ambigu", 0),
            "n_folds_intrinseque": class_counts.get("intrinsèque", 0),
            "decay_class_mode":    class_counts.most_common(1)[0][0] if class_counts else "indeterminé",
            "decay_class_stable":  class_counts.most_common(1)[0][1] == n_folds if class_counts else False,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 8. RAPPORTS MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def write_markdown_report_singlefold(
    out_path: Path,
    df: pd.DataFrame,
    features: List[str],
    summary_results: List[Dict[str, Any]],
    descriptive_df: pd.DataFrame,
    shapiro_df: pd.DataFrame,
    n_sig_raw: int,
    n_sig_corr: int,
    args: argparse.Namespace,
    fold: int,
) -> Path:
    """Rapport Markdown pour un fold unique."""
    lines: List[str] = []
    lines.append(f"# Rapport d'analyse statistique — fold {fold}\n")
    lines.append(f"**Split** : {args.split} (fold {fold})  ")
    lines.append(f"**N sujets** : {len(df)}  ")
    if "label" in df.columns:
        n_cn = int((df["label"] == 0).sum())
        n_ad = int((df["label"] == 1).sum())
        lines.append(f"**CN** : {n_cn} ({n_cn/len(df)*100:.1f}%)  ")
        lines.append(f"**AD** : {n_ad} ({n_ad/len(df)*100:.1f}%)  ")
    lines.append(f"**N features** : {len(features)}  ")
    lines.append(f"**N bootstrap** : {args.n_bootstrap}\n")
    lines.append("> ⚠️ Toutes les statistiques sont calculées sur les valeurs ")
    lines.append("> **réellement mesurées** uniquement. Les valeurs imputées ")
    lines.append("> par médiane train (T0) sont exclues.\n")

    lines.append("## 1. Table descriptive\n")
    lines.append("Continues : médiane [Q1, Q3] (n_real). Binaires : n positifs / n_real.\n")
    lines.append(descriptive_df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## 2. Séparation CN vs AD\n")
    lines.append(f"**Significatifs (raw p<0.05)** : {n_sig_raw}  ")
    lines.append(f"**Significatifs (Holm p<0.05)** : {n_sig_corr}\n")

    valid = [r for r in summary_results
             if "error" not in r and r.get("effect_global") is not None]
    valid.sort(key=lambda r: abs(r["effect_global"]), reverse=True)

    lines.append("### Top features par taille d'effet\n")
    lines.append("| Feature | n_CN_real | n_AD_real | Effect | p | Decay | Class |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in valid[:15]:
        eff = r.get("effect_global", 0)
        decay = r.get("decay")
        decay_str = f"{decay:+.3f}" if decay is not None else "—"
        lines.append(
            f"| `{r['feature']}` | {r.get('n_real_cn', '?')} | {r.get('n_real_ad', '?')} "
            f"| {eff:+.3f} | {r.get('p_global', 1):.3g} "
            f"| {decay_str} | {r.get('decay_class', '?')} |"
        )
    lines.append("")

    lines.append("## 3. Détection biais Simpson\n")
    fortuit = [r for r in valid if r.get("decay_class") == "FORTUIT (Simpson)"]
    ambigu = [r for r in valid if r.get("decay_class") == "ambigu"]
    intrins = [r for r in valid if r.get("decay_class") == "intrinsèque"]
    lines.append(f"- **FORTUIT** (decay > 0.12) : {len(fortuit)} features")
    for r in fortuit:
        lines.append(
            f"  - `{r['feature']}` : effet global = {r['effect_global']:+.3f}, "
            f"intra pondéré = {r.get('weighted_intra', 0):+.3f}, "
            f"decay = {r['decay']:+.3f}"
        )
    lines.append(f"- **Ambigu** : {len(ambigu)} features")
    lines.append(f"- **Intrinsèque** : {len(intrins)} features\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_markdown_report_multifold(
    out_path: Path,
    folds: List[int],
    args: argparse.Namespace,
    agg_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    n_subjects: int,
    n_cn: int,
    n_ad: int,
) -> Path:
    """
    Rapport Markdown synthétisant l'analyse multi-fold.
    """
    lines: List[str] = []
    lines.append(f"# Analyse statistique multi-fold — {len(folds)} folds\n")
    lines.append(f"**Folds analysés** : {folds}  ")
    lines.append(f"**Split** : {args.split}  ")
    lines.append(f"**N sujets (par fold)** : {n_subjects} ")
    lines.append(f"(CN={n_cn}, AD={n_ad}, %AD={n_ad/n_subjects*100:.1f})  ")
    lines.append(f"**N bootstrap** : {args.n_bootstrap}\n")

    lines.append("> ⚠️ Toutes les statistiques sont calculées sur les valeurs ")
    lines.append("> **réellement mesurées** uniquement. Les valeurs imputées ")
    lines.append("> (médiane train, T0) sont exclues.\n")
    lines.append("> ⚠️ Un fold = un partitionnement train/val/test différent du même ")
    lines.append("> ensemble de 6065 sujets. La variance observée reflète la ")
    lines.append("> sensibilité de l'analyse au choix du split.\n")

    # ── Section 1 : Table descriptive (sur fold_0 — identique à tout autre fold pour split=all) ──
    lines.append("## 1. Table descriptive (fold de référence)\n")
    lines.append("Continues : médiane [Q1, Q3] (n_real). Binaires : n positifs / n_real.\n")
    lines.append(descriptive_df.to_markdown(index=False))
    lines.append("\n")

    # ── Section 2 : Stabilité des effets ──
    lines.append("## 2. Stabilité des tailles d'effet sur les folds\n")
    lines.append(f"Pour chaque feature : mean ± std de l'effet de Mann-Whitney ")
    lines.append(f"(rank-biserial) à travers les {len(folds)} folds.\n")

    valid = agg_df[agg_df["effect_mean"].notna()].copy()
    valid["effect_abs"] = valid["effect_mean"].abs()
    valid = valid.sort_values("effect_abs", ascending=False)

    lines.append("| Feature | n_CN_real (μ) | n_AD_real (μ) | Effect (μ ± σ) | p_max | Sig. folds | Decay (μ ± σ) | Decay class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in valid.head(20).iterrows():
        eff_str = f"{r['effect_mean']:+.3f} ± {r['effect_std']:.3f}"
        if r.get("decay_mean") is not None:
            decay_str = f"{r['decay_mean']:+.3f} ± {r['decay_std']:.3f}"
        else:
            decay_str = "—"
        stable = "✓" if r["decay_class_stable"] else "~"
        lines.append(
            f"| `{r['feature']}` | {r['n_real_cn_mean']:.0f} | {r['n_real_ad_mean']:.0f} "
            f"| {eff_str} | {r['p_max']:.3g} | {r['n_folds_sig_p_raw']}/{int(r['n_folds'])} "
            f"| {decay_str} | {stable} {r['decay_class_mode']} |"
        )
    lines.append("")

    # ── Section 3 : Détection biais Simpson stable / instable ──
    lines.append("## 3. Détection biais Simpson — stabilité multi-fold\n")
    fortuit_stable = valid[
        (valid["decay_class_mode"] == "FORTUIT (Simpson)")
        & valid["decay_class_stable"]
    ]
    fortuit_unstable = valid[
        (valid["decay_class_mode"] == "FORTUIT (Simpson)")
        & ~valid["decay_class_stable"]
    ]
    lines.append(f"- **FORTUIT stable** (toutes folds = FORTUIT) : {len(fortuit_stable)} features")
    for _, r in fortuit_stable.iterrows():
        lines.append(
            f"  - `{r['feature']}` : decay = {r['decay_mean']:+.3f} ± {r['decay_std']:.3f}"
        )
    lines.append(f"- **FORTUIT instable** (FORTUIT majoritaire mais pas tous) : {len(fortuit_unstable)} features")
    for _, r in fortuit_unstable.iterrows():
        lines.append(
            f"  - `{r['feature']}` : FORTUIT={int(r['n_folds_FORTUIT'])}/"
            f"{int(r['n_folds'])} folds, decay = {r['decay_mean']:+.3f} ± {r['decay_std']:.3f}"
        )
    lines.append("")

    lines.append("## 4. Figures\n")
    lines.append("- `figures/multifold_decay_overview.pdf` — barplot decay moyen ± std sur folds")
    lines.append("- `figures/correlation_matrix.png` — corrélations Spearman (valeurs réelles, pairwise)")
    lines.append("- `figures/pca_features.png` — PCA 2D (imputation médiane pour viz uniquement)")
    lines.append("- `figures/tsne_features.png` — t-SNE 2D (imputation médiane pour viz uniquement)")
    lines.append("- `per_fold/fold_X/figures/` — distributions par feature pour chaque fold\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 9. PIPELINE D'UN FOLD
# ═══════════════════════════════════════════════════════════════════════════

def load_combined_data(splits_dir: Path, fold: int, split: str = "all") -> pd.DataFrame:
    """Charge le(s) split(s) demandé(s) du fold."""
    if split == "all":
        dfs = []
        for s in ["train", "val", "test"]:
            csv = splits_dir / f"fold_{fold}" / f"{s}.csv"
            if csv.exists():
                d = pd.read_csv(csv)
                d["__split"] = s
                dfs.append(d)
        if not dfs:
            raise FileNotFoundError(f"Aucun split trouvé dans {splits_dir}/fold_{fold}/")
        return pd.concat(dfs, ignore_index=True)
    else:
        csv = splits_dir / f"fold_{fold}" / f"{split}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Split introuvable : {csv}")
        return pd.read_csv(csv)


def run_single_fold(
    fold: int, splits_dir: Path, out_dir: Path,
    features_to_analyze: List[str], args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exécute l'analyse complète sur un fold donné. Retourne (results, summary_df,
    descriptive_df, shapiro_df) pour agrégation ultérieure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    df = load_combined_data(splits_dir, fold, args.split)
    n_subj = len(df)
    n_cn = int((df["label"] == 0).sum())
    n_ad = int((df["label"] == 1).sum())
    print(f"  fold {fold} : {n_subj} sujets (CN={n_cn}, AD={n_ad})")

    # ── Analyse par feature ──
    results: List[Dict[str, Any]] = []
    p_values: List[float] = []
    p_keys: List[Tuple[int, str]] = []

    for i, feat in enumerate(features_to_analyze, 1):
        result = analyze_feature(df, feat, n_bootstrap=args.n_bootstrap, seed=args.seed)
        results.append(result)
        if "error" not in result:
            plot_feature_distribution(
                df, feat, figures_dir / f"{feat}.pdf",
                label=result.get("label", feat),
            )
            p_values.append(result["p_global"])
            p_keys.append((i - 1, "p_global"))
            for cohort, cr in result["cohorts"].items():
                if not cr.get("skipped") and "p" in cr:
                    p_values.append(cr["p"])
                    p_keys.append((i - 1, f"cohort:{cohort}"))

    # ── Holm-Bonferroni ──
    corrected = holm_bonferroni_correction(p_values)
    holm_rows = []
    for (idx, key), p_raw, p_corr in zip(p_keys, p_values, corrected):
        feat = results[idx]["feature"]
        holm_rows.append({
            "feature":   feat,
            "test_key":  key,
            "p_raw":     p_raw,
            "p_holm":    p_corr,
            "sig_raw":   "*" if p_raw < 0.05 else " ",
            "sig_corr":  "*" if p_corr < 0.05 else " ",
        })
    holm_df = pd.DataFrame(holm_rows)
    holm_df.to_csv(out_dir / "holm_corrected_pvalues.csv", index=False)
    n_sig_raw = sum(1 for r in holm_rows if r["p_raw"] < 0.05)
    n_sig_corr = sum(1 for r in holm_rows if r["p_holm"] < 0.05)

    # ── Tableau résumé ──
    summary_rows = []
    for r in results:
        if "error" in r:
            summary_rows.append({"feature": r["feature"], "error": r["error"]})
            continue
        summary_rows.append({
            "feature":         r["feature"],
            "label":           r["label"],
            "type":            r["type"],
            "n_real_cn":       r["n_real_cn"],
            "n_real_ad":       r["n_real_ad"],
            "n_imputed_excluded": r["n_imputed_excluded"],
            "real_pct":        r["real_pct"],
            "effect_global":   r["effect_global"],
            "lb95":            r["lb_global"],
            "ub95":            r["ub_global"],
            "p_global":        r["p_global"],
            "weighted_intra":  r["weighted_intra"],
            "decay":           r["decay"],
            "decay_class":     r["decay_class"],
            "p_inter_cohorte": r["p_inter_cohorte"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_all_features.csv", index=False)

    # decay table
    decay_df = summary_df[
        ["feature", "effect_global", "weighted_intra", "decay", "decay_class"]
    ].copy()
    if "decay" in decay_df.columns:
        decay_df = decay_df.dropna(subset=["decay"])
        decay_df = decay_df.reindex(
            decay_df["decay"].abs().sort_values(ascending=False).index
        )
    decay_df.to_csv(out_dir / "decay_table.csv", index=False)

    # plot decay
    plot_decay_overview(results, figures_dir / "decay_overview.pdf")

    # Shapiro
    shapiro_df = run_shapiro_tests(df, features_to_analyze)
    if not shapiro_df.empty:
        shapiro_df.to_csv(out_dir / "shapiro_normality.csv", index=False)

    # Descriptive table
    descriptive_df = build_descriptive_table(df, features_to_analyze)
    descriptive_df.to_csv(out_dir / "descriptive_table.csv", index=False)

    # JSON
    save_metrics_json({
        "fold":          fold,
        "split":         args.split,
        "n_subjects":    n_subj,
        "n_cn":          n_cn,
        "n_ad":          n_ad,
        "n_features":    len(features_to_analyze),
        "n_bootstrap":   args.n_bootstrap,
        "thresholds":    THRESHOLDS,
        "results":       results,
        "n_sig_raw":     n_sig_raw,
        "n_sig_corr":    n_sig_corr,
    }, out_dir / "summary_report.json")

    # Markdown single-fold
    write_markdown_report_singlefold(
        out_dir / "report.md",
        df=df, features=features_to_analyze,
        summary_results=results,
        descriptive_df=descriptive_df,
        shapiro_df=shapiro_df,
        n_sig_raw=n_sig_raw, n_sig_corr=n_sig_corr,
        args=args, fold=fold,
    )

    return results, summary_df, descriptive_df, shapiro_df


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T9 — Analyse statistique des features cliniques (CN vs AD), "
                    "données réelles uniquement, multi-fold."
    )
    p.add_argument("--config", type=str, default=str(THIS_DIR / "config.yaml"))
    p.add_argument("--features", type=str, nargs="+", default=None,
                   help="Sous-ensemble de features (défaut: toutes)")
    p.add_argument("--split", type=str, default="train",
                   choices=["all", "train", "val", "test"],
                   help="Split à analyser. 'train' recommandé pour multi-fold "
                        "(variance significative). 'all' donne 0 variance "
                        "puisque train+val+test est identique entre folds. "
                        "Défaut : train.")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Liste des folds à analyser (défaut: 0 1 2 3 4). "
                        "Un seul fold pour single-fold mode.")
    p.add_argument("--n-bootstrap", type=int, default=THRESHOLDS["n_bootstrap"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Override output_dir de config.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    splits_dir = Path(config["data"]["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = (PROJECT_ROOT / splits_dir).resolve()

    out_dir = args.output or config["training"]["output_dir"]
    if not Path(out_dir).is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    folds = sorted(set(args.folds))
    is_multifold = len(folds) > 1

    print(f"\n{'='*70}")
    print(f"  ANALYSE STATISTIQUE — DONNÉES RÉELLES UNIQUEMENT")
    print(f"{'='*70}")
    print(f"  Mode         : {'multi-fold (' + str(len(folds)) + ' folds)' if is_multifold else 'single-fold'}")
    print(f"  Folds        : {folds}")
    print(f"  Split        : {args.split}")
    print(f"  Splits dir   : {splits_dir}")
    print(f"  Output dir   : {out_dir}")
    print(f"  N bootstrap  : {args.n_bootstrap}")
    if args.split == "all" and is_multifold:
        print(f"\n  ⚠️  --split=all + multi-fold : les folds partagent le même")
        print(f"      univers de sujets, la variance entre folds sera nulle.")
        print(f"      Recommandé : --split train")
    print(f"{'='*70}\n")

    # ── Liste des features ──
    if args.features:
        features_to_analyze = args.features
    else:
        # Charge fold 0 pour déterminer les features présentes
        df_probe = load_combined_data(splits_dir, folds[0], args.split)
        features_to_analyze = [f for f in FEATURE_META.keys() if f in df_probe.columns]

    print(f"[*] {len(features_to_analyze)} features à analyser")
    print(f"    {', '.join(features_to_analyze[:10])}"
          + (", ..." if len(features_to_analyze) > 10 else ""))

    # ── Boucle sur les folds ──
    fold_results: Dict[int, List[Dict[str, Any]]] = {}
    last_descriptive_df = None
    last_shapiro_df = None
    last_n_subj = last_n_cn = last_n_ad = 0

    for fold in folds:
        print(f"\n[*] Fold {fold} en cours...")
        if is_multifold:
            fold_out_dir = out_dir / "per_fold" / f"fold_{fold}"
        else:
            fold_out_dir = out_dir

        results, _, descriptive_df, shapiro_df = run_single_fold(
            fold=fold, splits_dir=splits_dir, out_dir=fold_out_dir,
            features_to_analyze=features_to_analyze, args=args,
        )
        fold_results[fold] = results
        last_descriptive_df = descriptive_df
        last_shapiro_df = shapiro_df

        # Compte sujets pour le rapport
        df_fold = load_combined_data(splits_dir, fold, args.split)
        last_n_subj = len(df_fold)
        last_n_cn = int((df_fold["label"] == 0).sum())
        last_n_ad = int((df_fold["label"] == 1).sum())

    # ── Mode single-fold : terminé ici ──
    if not is_multifold:
        print(f"\n[✓] Analyse single-fold terminée — résultats dans {out_dir}\n")
        return 0

    # ── Mode multi-fold : agrégation ──
    print(f"\n{'='*70}")
    print(f"  AGRÉGATION MULTI-FOLD")
    print(f"{'='*70}")

    agg_df = aggregate_fold_results(fold_results)
    agg_df.to_csv(out_dir / "multifold_summary.csv", index=False)
    print(f"  💾 multifold_summary.csv ({len(agg_df)} features)")

    # Plot decay avec error bars
    plot_decay_with_variance(agg_df, figures_dir / "multifold_decay_overview.pdf")
    print(f"  💾 figures/multifold_decay_overview.pdf")

    # Analyses globales (sur fold[0] — référence si split=all, ou sur la même structure pour split=train)
    df_ref = load_combined_data(splits_dir, folds[0], args.split)
    print(f"\n[*] Corrélation Spearman (valeurs réelles, pairwise complete)...")
    plot_correlation_matrix(df_ref, features_to_analyze, figures_dir / "correlation_matrix.png")
    print(f"  💾 figures/correlation_matrix.png + .csv")

    print(f"\n[*] PCA des features (imputation pour viz)...")
    plot_pca_features(df_ref, features_to_analyze, figures_dir / "pca_features.png")
    print(f"  💾 figures/pca_features.png")

    print(f"\n[*] t-SNE des features (imputation pour viz)...")
    plot_tsne_features(df_ref, features_to_analyze, figures_dir / "tsne_features.png",
                       perplexity=30, seed=args.seed)
    print(f"  💾 figures/tsne_features.png")

    # Rapport multi-fold
    print(f"\n[*] Rapport multi-fold...")
    write_markdown_report_multifold(
        out_path=out_dir / "report.md",
        folds=folds, args=args, agg_df=agg_df,
        descriptive_df=last_descriptive_df,
        n_subjects=last_n_subj, n_cn=last_n_cn, n_ad=last_n_ad,
    )
    print(f"  💾 report.md")

    # JSON synthétique
    save_metrics_json({
        "mode":           "multifold",
        "folds":          folds,
        "split":          args.split,
        "n_subjects_per_fold": last_n_subj,
        "n_features":     len(features_to_analyze),
        "n_bootstrap":    args.n_bootstrap,
        "thresholds":     THRESHOLDS,
        "fold_results":   {str(k): v for k, v in fold_results.items()},
    }, out_dir / "multifold_report.json")

    # Récap console
    print(f"\n{'='*70}")
    print(f"  RÉCAPITULATIF MULTI-FOLD")
    print(f"{'='*70}")
    fortuit_stable = agg_df[
        (agg_df["decay_class_mode"] == "FORTUIT (Simpson)")
        & agg_df["decay_class_stable"]
    ]
    fortuit_unstable = agg_df[
        (agg_df["decay_class_mode"] == "FORTUIT (Simpson)")
        & ~agg_df["decay_class_stable"]
    ]
    print(f"  Features FORTUIT stable (toutes folds)   : {len(fortuit_stable)}")
    for _, r in fortuit_stable.iterrows():
        print(f"    - {r['feature']:18s} decay = {r['decay_mean']:+.3f} ± {r['decay_std']:.3f}")
    print(f"  Features FORTUIT instable               : {len(fortuit_unstable)}")
    for _, r in fortuit_unstable.iterrows():
        print(f"    - {r['feature']:18s} {int(r['n_folds_FORTUIT'])}/{int(r['n_folds'])} folds, "
              f"decay = {r['decay_mean']:+.3f} ± {r['decay_std']:.3f}")
    print(f"{'='*70}\n")
    print(f"[✓] Tous les résultats dans : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())