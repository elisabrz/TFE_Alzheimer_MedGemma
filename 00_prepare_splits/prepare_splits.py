"""
prepare_splits.py — Génération des splits 5-fold stratifiés niveau sujet.

Pipeline autonome qui intègre :
    1. Le nettoyage des features cliniques (repris de prepare_tfe_dataset_v2.py)
    2. L'agrégation MMSE + volume hippocampique pour ADNI
    3. Le filtre "première visite par sujet" (référence : travail de Tanguy)
    4. La validation croisée 5-fold stratifiée niveau SUJET
    5. L'imputation médiane GLOBALE train-only, appliquée aux 3 splits
       Marquage `<col>_imputed` ajouté pour traçabilité dans le prompt LLM.
    6. La sauvegarde propre avec métadonnées JSON pour reproductibilité

Sorties (dans --output-dir, par défaut ../data/splits/) :
    fold_0/
        train.csv               # 4367 sujets (72%) — imputé + marqué
        val.csv                 #  485 sujets (8%)  — imputé + marqué
        test.csv                # 1213 sujets (20%) — imputé + marqué
        imputation_stats.json   # Médianes train-only (traçabilité)
    fold_1/ ... fold_4/         # Idem
    splits_metadata.json        # Stats globales + seeds + hash du CSV source

Usage :
    cd 00_prepare_splits/
    python prepare_splits.py
    # ou avec overrides CLI :
    python prepare_splits.py --config config.yaml \
                             --input /chemin/tfe_dataset_complet_v2.csv \
                             --output-dir ../data/splits \
                             --seed 42

Conception :
    - Si l'entrée est un CSV Tanguy brut (all.csv), on fait tout le pipeline.
    - Si c'est un CSV déjà nettoyé (détecté par la présence de `mmse_score` et
      `has_real_measures`), on skippe les étapes 1-2 et on passe directement
      au filtre + split + imputation.
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES (reprises de prepare_tfe_dataset_v2.py + sync_tfe_splits_v2.py)
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_MAP = {
    "adni":  "ADNI-skull",
    "nacc":  "NACC-skull",
    "oasis": "OASIS-skull",
}

NACC_MISSING = -4.0

# Plages cliniques pour clipping des valeurs aberrantes
CONTINUOUS_COLS: Dict[str, Tuple[float, float]] = {
    "AGE":       (18,  110),
    "PTEDUCAT":  (0,   30),
    "CATANIMSC": (0,   100),
    "TRAASCOR":  (0,   999),
    "TRABSCOR":  (0,   999),
    "DSPANFOR":  (0,   20),
    "DSPANBAC":  (0,   20),
    "BNTTOTAL":  (0,   60),
    "BMI":       (10,  60),
    "VSWEIGHT":  (30,  200),
}

# Colonnes binaires d'antécédents médicaux
MH_COLS = ["MH14ALCH", "MH16SMOK", "MH4CARD", "MH2NEURL"]

# Remapping PTMARRY (codes étendus NACC → codes standards ADNI)
PTMARRY_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 1}

# Colonnes à imputer par médiane train-only (médiane GLOBALE, pas par classe)
CATEGORICAL_IMPUTE = ["PTMARRY", "PTGENDER"]
MMSE_COLS          = ["mmse_score", "hippo_vol"]
ALL_IMPUTE_COLS    = (
    list(CONTINUOUS_COLS.keys()) + MH_COLS + CATEGORICAL_IMPUTE + MMSE_COLS
)


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1-2 : NETTOYAGE (repris de prepare_tfe_dataset_v2.py)
# ═══════════════════════════════════════════════════════════════════════════

def fix_mri_path(row: pd.Series, mri_root: str) -> str:
    """Reconstruit le chemin absolu vers le fichier NIfTI."""
    folder = SOURCE_MAP.get(str(row["source"]).lower())
    if folder:
        filename = os.path.basename(row["scan_path"])
        return os.path.join(mri_root, folder, str(row["subject_id"]), filename)
    return row["scan_path"]


def aggregate_adni_measures(
    df: pd.DataFrame,
    adni_mmse_csv: Optional[str] = None,
    adni_fs_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Ajoute mmse_score, hippo_vol, has_real_measures à partir des CSV ADNI."""
    if adni_mmse_csv is None or not os.path.exists(adni_mmse_csv):
        print(f"  [!] CSV MMSE ADNI absent ({adni_mmse_csv}) — mmse_score = NaN partout")
        df["mmse_score"] = np.nan
    else:
        df_mmse = pd.read_csv(adni_mmse_csv, low_memory=False)
        df_mmse["MMSCORE"] = pd.to_numeric(df_mmse["MMSCORE"], errors="coerce")
        mmse_map = df_mmse.groupby("subject_id")["MMSCORE"].median()
        df["mmse_score"] = df["subject_id"].map(mmse_map)

    if adni_fs_csv is None or not os.path.exists(adni_fs_csv):
        print(f"  [!] CSV FreeSurfer ADNI absent ({adni_fs_csv}) — hippo_vol = NaN partout")
        df["hippo_vol"] = np.nan
    else:
        df_fs = pd.read_csv(adni_fs_csv, low_memory=False)
        df_fs["hippo_total"] = (
            pd.to_numeric(df_fs["ST29SV"], errors="coerce")
            + pd.to_numeric(df_fs["ST88SV"], errors="coerce")
        )
        hippo_map = df_fs.groupby("PTID")["hippo_total"].median()
        df["hippo_vol"] = df["subject_id"].map(hippo_map)

    df["has_real_measures"] = df["mmse_score"].notna().astype(int)
    n_real = int(df["has_real_measures"].sum())
    print(f"  Patients ADNI avec mesures réelles MMSE : {n_real}")
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage déterministe (pas de leakage) :
        A. Sentinelles -4 (NACC "non collecté") → NaN
        B. Codes binaires MH* (2 → 1 'passé' = 'présent', 9 → NaN)
        C. PTMARRY remappé selon PTMARRY_MAP
        D. Clipping aux plages cliniques valides
    """
    df = df.copy()

    # A. Sentinelles -4 → NaN sur toutes les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        n_before = int((df[col] == NACC_MISSING).sum())
        if n_before > 0:
            df[col] = df[col].replace(NACC_MISSING, np.nan)
            print(f"    Sentinelle -4 → NaN : {col:15} ({n_before} valeurs)")

    # B. Binaires MH*
    for col in MH_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].replace(2.0, 1.0)
        df[col] = df[col].replace(9.0, np.nan)
        invalid = ~df[col].isin([0.0, 1.0]) & df[col].notna()
        if invalid.sum() > 0:
            print(f"    Valeurs invalides → NaN : {col:15} ({int(invalid.sum())})")
            df.loc[invalid, col] = np.nan

    # C. PTMARRY
    if "PTMARRY" in df.columns:
        original = df["PTMARRY"].copy()
        df["PTMARRY"] = df["PTMARRY"].map(PTMARRY_MAP)
        n_lost = int(original.notna().sum() - df["PTMARRY"].notna().sum())
        if n_lost > 0:
            print(f"    PTMARRY codes inconnus → NaN : {n_lost} valeurs")

    # D. Clipping clinique
    for col, (vmin, vmax) in CONTINUOUS_COLS.items():
        if col not in df.columns:
            continue
        oor = df[col].notna() & ((df[col] < vmin) | (df[col] > vmax))
        if oor.sum() > 0:
            print(
                f"    Hors plage [{vmin},{vmax}] → NaN : {col:15} "
                f"({int(oor.sum())} valeurs)"
            )
            df.loc[oor, col] = np.nan

    return df


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : FILTRE 1 VISITE PAR SUJET
# ═══════════════════════════════════════════════════════════════════════════

def filter_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit 1 ligne = 1 sujet. Référence : papier CBMS Tanguy
    ('Only the first visit per subject is used to avoid temporal leakage').

    - Si le DataFrame contient déjà 1 ligne par sujet : log et retourne tel quel.
    - Sinon : garde la première occurrence (deterministe avec tri préalable
      sur subject_id + scan_path pour reproductibilité).
    """
    n_rows = len(df)
    n_subjects = df["subject_id"].nunique()

    if n_rows == n_subjects:
        print(f"  [✓] Déjà 1 ligne par sujet ({n_subjects} sujets uniques)")
        return df.reset_index(drop=True)

    print(
        f"  [!] {n_rows} lignes pour {n_subjects} sujets — "
        f"filtrage à la 1re visite ({n_rows - n_subjects} lignes à retirer)"
    )

    # Tri déterministe pour que le "first" soit reproductible entre exécutions
    df_sorted = df.sort_values(["subject_id", "scan_path"], kind="stable")
    df_filtered = df_sorted.drop_duplicates(subset="subject_id", keep="first")
    df_filtered = df_filtered.reset_index(drop=True)

    assert len(df_filtered) == n_subjects, \
        f"Filtre 1-visite incohérent : {len(df_filtered)} ≠ {n_subjects}"

    print(f"  [✓] {len(df_filtered)} sujets retenus")
    return df_filtered


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : GÉNÉRATION DES 5 FOLDS STRATIFIÉS
# ═══════════════════════════════════════════════════════════════════════════

def create_fold_assignments(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    StratifiedKFold niveau sujet sur la colonne `label` (CN/AD).
    Retourne un array `fold_id` de longueur len(df), chaque entrée ∈ [0, n_folds-1].
    Le sujet en position i est en test dans le fold `fold_id[i]`.
    """
    assert len(df) == df["subject_id"].nunique(), \
        "create_fold_assignments attend 1 ligne par sujet"

    y = df["label"].values
    fold_id = np.full(len(df), -1, dtype=int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
        fold_id[test_idx] = fold_idx

    assert (fold_id >= 0).all(), "Tous les sujets doivent être assignés à un fold"
    return fold_id


def split_fold(
    df: pd.DataFrame,
    fold_id: np.ndarray,
    fold_k: int,
    val_size: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pour le fold k :
        - test = sujets dont fold_id == k           (~20%)
        - les autres 80% sont splittés 90/10 en train/val, stratifié sur label
    Retourne (train_df, val_df, test_df).
    """
    test_mask = fold_id == fold_k
    df_test = df[test_mask].copy().reset_index(drop=True)
    df_trainval = df[~test_mask].copy().reset_index(drop=True)

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size,
        stratify=df_trainval["label"],
        random_state=seed,
        shuffle=True,
    )
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 : IMPUTATION MÉDIANE GLOBALE TRAIN-ONLY
# ═══════════════════════════════════════════════════════════════════════════

def fit_imputation_stats(df_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calcule les médianes train-only pour chaque colonne à imputer.

    On stocke 3 valeurs par colonne :
        - cn_median, ad_median  : conservées pour traçabilité (non utilisées
                                   par défaut depuis la décision médiane globale)
        - global_median         : médiane train toutes classes confondues,
                                   appliquée aux 3 splits (cf. Tanguy CBMS).

    Si une médiane globale est NaN (colonne entièrement absente du train),
    la colonne est skippée — aucune imputation possible.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in ALL_IMPUTE_COLS:
        if col not in df_train.columns:
            continue

        cn_med = df_train[df_train["label"] == 0][col].median()
        ad_med = df_train[df_train["label"] == 1][col].median()
        g_med  = df_train[col].median()

        if pd.isna(g_med):
            print(f"    [!] {col} : médiane globale train = NaN, impossible à imputer")
            continue

        # Fallback CN/AD sur global si une classe est entièrement NaN
        if pd.isna(cn_med):
            cn_med = g_med
        if pd.isna(ad_med):
            ad_med = g_med

        stats[col] = {
            "cn_median":     float(cn_med),
            "ad_median":     float(ad_med),
            "global_median": float(g_med),
            "n_nan_train":   int(df_train[col].isna().sum()),
        }
    return stats


def apply_imputation(
    df: pd.DataFrame, stats: Dict[str, Dict[str, float]], split_name: str,
    use_class_label: bool = False,
    track_imputation: bool = True,
) -> pd.DataFrame:
    """
    Impute par médiane train (globale par défaut, classe si use_class_label=True).

    Si track_imputation=True, ajoute pour chaque colonne imputée une colonne
    booléenne `<col>_imputed` (1 = valeur imputée, 0 = valeur réelle).
    Cette colonne est ajoutée même si nan_mask.any() est False (cohérence
    schéma entre les 3 splits — toutes auront les mêmes colonnes).

    NOTE méthodologique : par défaut `use_class_label=False` partout. Cohérent
    avec Tanguy CBMS et évite tout leakage par-classe (pas besoin du label
    pour imputer val/test, qui serait inconnu en production).
    """
    df = df.copy()
    for col, s in stats.items():
        if col not in df.columns:
            continue

        nan_mask = df[col].isna()

        if track_imputation:
            df[f"{col}_imputed"] = nan_mask.astype(int)

        if not nan_mask.any():
            continue

        if use_class_label:
            df.loc[(df["label"] == 0) & nan_mask, col] = s["cn_median"]
            df.loc[(df["label"] == 1) & nan_mask, col] = s["ad_median"]
        else:
            df.loc[nan_mask, col] = s["global_median"]

    return df


def verify_no_nan(df: pd.DataFrame, split_name: str, cols: List[str]) -> bool:
    remaining = {
        c: int(df[c].isna().sum())
        for c in cols
        if c in df.columns and df[c].isna().any()
    }
    if remaining:
        print(f"    [!] {split_name} : NaN résiduels → {remaining}")
        return False
    print(f"    [✓] {split_name} : aucun NaN résiduel dans les colonnes critiques")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════

def md5sum(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def describe_split(df: pd.DataFrame, name: str) -> Dict:
    """Statistiques compactes d'un split pour log + metadata."""
    n = len(df)
    n_cn = int((df["label"] == 0).sum())
    n_ad = int((df["label"] == 1).sum())
    cohort_counts = (
        df["source"].value_counts().to_dict() if "source" in df.columns else {}
    )

    # Stats imputation : combien de samples ont au moins une feature imputée
    imp_cols = [c for c in df.columns if c.endswith("_imputed")]
    n_with_imputed = (
        int((df[imp_cols].sum(axis=1) > 0).sum()) if imp_cols else 0
    )

    return {
        "name":            name,
        "n":               n,
        "n_cn":            n_cn,
        "n_ad":            n_ad,
        "ad_pct":          round(100.0 * n_ad / max(n, 1), 2),
        "cohorts":         {k: int(v) for k, v in cohort_counts.items()},
        "n_with_imputed":  n_with_imputed,
        "imputed_pct":     round(100.0 * n_with_imputed / max(n, 1), 1),
    }


def log_split(stats: Dict) -> None:
    print(
        f"    {stats['name']:<6} n={stats['n']:<5} "
        f"CN={stats['n_cn']:<5} AD={stats['n_ad']:<4} "
        f"(AD={stats['ad_pct']:.1f}%) "
        f"imputés={stats['n_with_imputed']} ({stats['imputed_pct']:.1f}%) "
        f"cohortes={stats['cohorts']}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Génère les splits 5-fold stratifiés niveau sujet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=str(here / "config.yaml"),
        help="Fichier de configuration YAML."
    )
    p.add_argument(
        "--input", type=str, default=None,
        help="CSV d'entrée (override du config)."
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Dossier de sortie (override du config)."
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Seed (override du config)."
    )
    p.add_argument(
        "--n-folds", type=int, default=None,
        help="Nombre de folds (override du config)."
    )
    p.add_argument(
        "--skip-cleaning", action="store_true",
        help="Saute le nettoyage (utile si le CSV est déjà nettoyé)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Résolution des paramètres (CLI > config) ──────────────────────────
    input_csv  = args.input      or cfg["input_csv"]
    output_dir = args.output_dir or cfg["output_dir"]
    seed       = args.seed       or cfg.get("seed", 42)
    n_folds    = args.n_folds    or cfg.get("n_folds", 5)
    val_size   = 1.0 - cfg.get("train_val_split", 0.9)
    seeds_log  = cfg.get("seeds", [42, 123, 456])

    # Expansion des variables d'environnement (${DATA_ROOT}, etc.)
    input_csv  = os.path.expandvars(input_csv)
    output_dir = os.path.expandvars(output_dir)

    # Paramètres optionnels de nettoyage
    mri_root       = os.path.expandvars(cfg.get("mri_root", ""))
    adni_mmse_csv  = os.path.expandvars(cfg.get("adni_mmse_csv", "") or "")
    adni_fs_csv    = os.path.expandvars(cfg.get("adni_fs_csv", "")   or "")

    # ── Banner ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PREPARE_SPLITS — 5-fold stratifié niveau sujet")
    print("  Imputation : médiane GLOBALE train-only (3 splits) + tracking")
    print("=" * 70)
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_dir}")
    print(f"  Seed   : {seed} | n_folds = {n_folds} | val_size = {val_size:.2f}")

    # ── [1/6] Chargement ──────────────────────────────────────────────────
    print("\n[1/6] Chargement du CSV source...")
    if not os.path.exists(input_csv):
        print(f"\n❌ Fichier introuvable : {input_csv}")
        sys.exit(1)
    df = pd.read_csv(input_csv)
    csv_hash = md5sum(input_csv)
    print(f"  {len(df)} lignes, {df['subject_id'].nunique()} sujets uniques")
    print(f"  MD5 : {csv_hash}")

    # Détection : CSV déjà nettoyé ?
    already_cleaned = (
        "mmse_score" in df.columns
        and "has_real_measures" in df.columns
        and not args.skip_cleaning
    )
    if already_cleaned and not args.skip_cleaning:
        print("  → CSV déjà nettoyé (mmse_score + has_real_measures détectés)")
    elif args.skip_cleaning:
        print("  → --skip-cleaning : saute les étapes 2 et 3")

    # ── [2/6] Reconstruction chemins IRM + [3/6] agrégation ADNI ───────────────
    if not already_cleaned and not args.skip_cleaning:
        print("\n[2/6] Reconstruction des chemins IRM...")
        if mri_root:
            df["scan_path"] = df.apply(lambda r: fix_mri_path(r, mri_root), axis=1)
            print(f"  Chemins reconstruits depuis {mri_root}")
        else:
            print("  [!] mri_root non défini — chemins inchangés")

        print("\n[3/6] Agrégation MMSE + volume hippocampique (ADNI)...")
        df = aggregate_adni_measures(df, adni_mmse_csv, adni_fs_csv)
    else:
        print("\n[2-3/6] Skippés (déjà fait en amont)")

    # ── [4/6] Nettoyage des features cliniques ───────────────────────────
    if not already_cleaned and not args.skip_cleaning:
        print("\n[4/6] Nettoyage déterministe des features cliniques...")
        n_nan_before = int(df[ALL_IMPUTE_COLS].isna().sum().sum())
        df = clean_features(df)
        n_nan_after = int(df[ALL_IMPUTE_COLS].isna().sum().sum())
        print(f"  NaN avant nettoyage : {n_nan_before}")
        print(f"  NaN après nettoyage : {n_nan_after} (à imputer train-only)")
    else:
        print("\n[4/6] Skippé (déjà nettoyé en amont)")

    # ── [5/6] Filtre 1 visite par sujet ──────────────────────────────────
    print("\n[5/6] Filtre « première visite par sujet »...")
    df = filter_first_visit(df)
    n_subjects = len(df)

    print(f"\n  Distribution des classes après filtre :")
    n_cn = int((df["label"] == 0).sum())
    n_ad = int((df["label"] == 1).sum())
    print(f"    CN  : {n_cn:5d} ({100*n_cn/n_subjects:.1f}%)")
    print(f"    AD  : {n_ad:5d} ({100*n_ad/n_subjects:.1f}%)")

    if "source" in df.columns:
        print(f"\n  Distribution des cohortes :")
        for src, count in df["source"].value_counts().items():
            print(f"    {src:<6} : {count:5d} ({100*count/n_subjects:.1f}%)")

    # ── [6/6] Génération des 5 folds ─────────────────────────────────────
    print(f"\n[6/6] Génération des {n_folds} folds stratifiés...")
    fold_id = create_fold_assignments(df, n_folds=n_folds, seed=seed)

    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "input_csv":        input_csv,
        "input_csv_md5":    csv_hash,
        "n_subjects_total": n_subjects,
        "n_folds":          n_folds,
        "seed":             seed,
        "seeds_for_repro":  seeds_log,
        "val_size":         val_size,
        "imputation":       {
            "strategy":         "median_global_train_only",
            "applied_to":       ["train", "val", "test"],
            "tracking_columns": "<col>_imputed (1 = imputé, 0 = réel)",
        },
        "folds":            {},
    }

    for k in range(n_folds):
        print(f"\n  ─── Fold {k} ───")
        df_train, df_val, df_test = split_fold(
            df, fold_id, fold_k=k, val_size=val_size, seed=seed
        )

        # Vérification anti-fuite
        train_subj = set(df_train["subject_id"])
        val_subj   = set(df_val["subject_id"])
        test_subj  = set(df_test["subject_id"])

        tv = train_subj & val_subj
        tt = train_subj & test_subj
        vt = val_subj & test_subj

        if tv or tt or vt:
            raise RuntimeError(
                f"Fuite sujet fold {k} : "
                f"train∩val={len(tv)} train∩test={len(tt)} val∩test={len(vt)}"
            )

        # Imputation train-only (médiane globale appliquée aux 3 splits)
        # Marquage `<col>_imputed` pour traçabilité dans le prompt LLM.
        print(f"    Fit imputation (train-only, médiane globale)...")
        impute_stats = fit_imputation_stats(df_train)
        df_train = apply_imputation(
            df_train, impute_stats, "TRAIN",
            use_class_label=False, track_imputation=True,
        )
        df_val = apply_imputation(
            df_val, impute_stats, "VAL",
            use_class_label=False, track_imputation=True,
        )
        df_test = apply_imputation(
            df_test, impute_stats, "TEST",
            use_class_label=False, track_imputation=True,
        )

        # Cohérence has_real_measures ↔ mmse_score_imputed (par split)
        for split_name, split_df in [
            ("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test),
        ]:
            if ("has_real_measures" in split_df.columns
                    and "mmse_score_imputed" in split_df.columns):
                inconsistent = (
                    ((split_df["has_real_measures"] == 1)
                     & (split_df["mmse_score_imputed"] == 1)).sum()
                    + ((split_df["has_real_measures"] == 0)
                       & (split_df["mmse_score_imputed"] == 0)).sum()
                )
                if inconsistent > 0:
                    print(
                        f"    [!] {split_name}: {inconsistent} sujets — "
                        f"has_real_measures incohérent avec mmse_score_imputed"
                    )

        # Vérification NaN résiduels (les 3 splits doivent être propres maintenant)
        critical = [c for c in ALL_IMPUTE_COLS if c in df_train.columns]
        verify_no_nan(df_train, "TRAIN", critical)
        verify_no_nan(df_val,   "VAL",   critical)
        verify_no_nan(df_test,  "TEST",  critical)

        # Stats par split (incluant n_with_imputed)
        stats_train = describe_split(df_train, "TRAIN")
        stats_val   = describe_split(df_val,   "VAL")
        stats_test  = describe_split(df_test,  "TEST")
        log_split(stats_train)
        log_split(stats_val)
        log_split(stats_test)

        # Sauvegarde
        fold_dir = os.path.join(output_dir, f"fold_{k}")
        os.makedirs(fold_dir, exist_ok=True)
        df_train.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        df_val.to_csv(os.path.join(fold_dir, "val.csv"),     index=False)
        df_test.to_csv(os.path.join(fold_dir, "test.csv"),   index=False)
        with open(os.path.join(fold_dir, "imputation_stats.json"), "w") as f:
            json.dump(impute_stats, f, indent=2)
        print(f"    💾 {fold_dir}/{{train,val,test}}.csv + imputation_stats.json")

        metadata["folds"][f"fold_{k}"] = {
            "train": stats_train, "val": stats_val, "test": stats_test,
            "n_imputation_cols": len(impute_stats),
        }

    # ── Métadonnées globales ─────────────────────────────────────────────
    meta_path = os.path.join(output_dir, "splits_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n💾 Métadonnées globales : {meta_path}")

    # ── Récapitulatif final ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✓ SPLITS GÉNÉRÉS AVEC SUCCÈS")
    print("=" * 70)
    print(f"  {n_folds} folds sauvegardés dans {output_dir}")
    print(f"  Imputation : médiane globale train-only sur les 3 splits")
    print(f"  Colonnes <col>_imputed disponibles pour le prompt LLM")
    print(f"  Fold 0 prêt pour les entraînements du TFE (seed={seed})")


if __name__ == "__main__":
    main()