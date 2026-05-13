"""
evaluate.py — Évaluation standalone d'un checkpoint sur le test set (ou val).

Charge un checkpoint MedGemma + LoRA (+ tête MMSE si présente), tourne
l'évaluation sur le split demandé du fold demandé, et sauvegarde :

    metrics_test.json          : toutes les métriques (global + par cohorte)
    predictions_test.csv       : 1 ligne par patient (subject_id, true, pred, prob, mmse_pred)
    cohort_metrics_test.csv    : métriques stratifiées ADNI / NACC / OASIS
    roc_curve_test.png         : courbe ROC test
    confusion_matrix_test.png  : matrice de confusion test (seuil 0.5 et seuil optimal)
    mmse_scatter_test.png      : nuage MMSE prédit vs vrai (si tête MMSE présente)

Usage :
    # Auto-detect : trouve le best_model dans results/01_no_mmse/ et évalue sur test
    python evaluate.py --task 01_no_mmse

    # Checkpoint explicite
    python evaluate.py --checkpoint results/01_no_mmse/best_model

    # Évaluer sur val au lieu de test
    python evaluate.py --task 01_no_mmse --split val

    # Output dir customisé
    python evaluate.py --task 01_no_mmse --output /tmp/test_eval/

    # Différent fold (par défaut fold_0)
    python evaluate.py --task 01_no_mmse --fold 1

Exit codes :
    0 : succès
    1 : checkpoint introuvable / split CSV introuvable
    2 : erreur GPU (OOM, CUDA)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ── Imports locaux (depuis la racine du projet) ─────────────────────────────
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from utils import (
    load_config, setup_env, check_vram, release_gpu,
    is_valid_checkpoint, load_mmse_head, set_token_ids,
    evaluate_dataset, compute_mmse_metrics,
    plot_roc_curve, plot_confusion_matrix, plot_mmse_scatter,
    save_metrics_json,
)
from dataset import TfeDataset, tfe_collate_fn
from trainers import TfeMedGemmaCls, TfeMedGemmaWithMMSE


# ═══════════════════════════════════════════════════════════════════════════
# 1. RÉSOLUTION CHECKPOINT / CONFIG / CHEMINS
# ═══════════════════════════════════════════════════════════════════════════

# Mapping task_name → dossier
TASK_DIRS: Dict[str, str] = {
    "01_no_mmse":         "01_train_no_mmse",
    "01_train_no_mmse":   "01_train_no_mmse",
    "02_with_mmse":       "02_train_with_mmse",
    "02_train_with_mmse": "02_train_with_mmse",
    "03_ablation_neuro":  "03_ablation_neuro",
    "04_ablation_demo":   "04_ablation_demo",
    "05_ablation_text":   "05_ablation_text",
    "06_reprompt_images": "06_reprompt_images",
}


def resolve_task_dir(task: str) -> Path:
    """Retourne le path du dossier de tâche."""
    if task in TASK_DIRS:
        d = THIS_DIR / TASK_DIRS[task]
        if d.exists():
            return d
    # Tente le nom direct
    direct = THIS_DIR / task
    if direct.exists():
        return direct
    raise FileNotFoundError(
        f"Tâche '{task}' introuvable. "
        f"Choix : {sorted(set(TASK_DIRS.keys()))}"
    )


def find_best_model(task_dir: Path, config: dict) -> Path:
    """
    Localise le best_model dans results/<task>/.
    Cherche 'best_model' (nom standard) et fallback sur best_model_name custom.
    """
    out_dir = config["training"]["output_dir"]
    if not Path(out_dir).is_absolute():
        out_dir = (THIS_DIR / out_dir).resolve()
    out_dir = Path(out_dir)

    candidates = [
        out_dir / config.get("training", {}).get("best_model_name", "best_model"),
        out_dir / "best_model",
        out_dir / "best_model_step2",
        out_dir / "best_model_step3",
        out_dir / "best_model_baseline",
    ]
    for c in candidates:
        if c.exists() and is_valid_checkpoint(str(c)):
            return c
    raise FileNotFoundError(
        f"Aucun best_model valide trouvé dans {out_dir}. "
        f"Tentatives : {[str(c.name) for c in candidates]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

def load_model_for_eval(
    config: dict,
    checkpoint_path: Path,
    device: str = "cuda",
):
    """
    Charge MedGemma + LoRA + (optionnel) tête MMSE depuis le checkpoint.

    Auto-détecte la présence de la tête MMSE via le fichier mmse_head.pt.
    Retourne (model, has_mmse_head).
    """
    print(f"[*] Chargement processor...")
    processor = AutoProcessor.from_pretrained(config["model"]["name"])

    # Quantization (même config qu'à l'entraînement)
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

    print(f"[*] Chargement base model + LoRA depuis {checkpoint_path}...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(
        base_model, str(checkpoint_path), is_trainable=False
    )

    # Auto-détection présence tête MMSE
    has_mmse_head = (checkpoint_path / "mmse_head.pt").exists()
    config_says_mmse = bool(config.get("mmse_head", {}).get("enabled", False))

    if has_mmse_head and config_says_mmse:
        hidden_size = base_model.config.text_config.hidden_size
        model = TfeMedGemmaWithMMSE(peft_model, hidden_size=hidden_size)
        model.mmse_head = load_mmse_head(str(checkpoint_path), hidden_size=hidden_size)
        model.mmse_head = model.mmse_head.to(device)
        print(f"[*] Mode : multitâche (tête MMSE chargée, hidden_size={hidden_size})")
    elif has_mmse_head and not config_says_mmse:
        print(f"[!] mmse_head.pt présent mais config.mmse_head.enabled=False — ignorée")
        model = TfeMedGemmaCls(peft_model)
    elif not has_mmse_head and config_says_mmse:
        print(f"[!] config.mmse_head.enabled=True mais pas de mmse_head.pt — mode cls")
        model = TfeMedGemmaCls(peft_model)
        config_says_mmse = False
    else:
        model = TfeMedGemmaCls(peft_model)
        print(f"[*] Mode : classification seule (pas de tête MMSE)")

    model.eval()
    return processor, model, config_says_mmse and has_mmse_head


# ═══════════════════════════════════════════════════════════════════════════
# 3. MÉTRIQUES STRATIFIÉES PAR COHORTE
# ═══════════════════════════════════════════════════════════════════════════

def compute_cohort_metrics(
    df: pd.DataFrame,
    probs: List[float],
    true_cls: List[int],
    thresholds: Dict[str, float],
    mmse_pred: Optional[List[float]] = None,
    mmse_true: Optional[List[float]] = None,
    has_real: Optional[List[bool]] = None,
) -> pd.DataFrame:
    """
    Décompose les métriques par cohorte (ADNI / NACC / OASIS / GLOBAL)
    pour CHAQUE seuil fourni dans `thresholds`.

    Args:
        thresholds : dict label → valeur, ex.
            {"0.5": 0.5, "youden_val": 0.469}
        Pour chaque (cohorte, seuil), une ligne est produite dans le DataFrame.

    Returns:
        DataFrame avec colonnes :
            cohort | threshold_label | threshold | n | n_cn | n_ad
            | accuracy | auc | f1 | sensitivity | specificity
            [+ mmse_mae_real | mmse_rmse_real | mmse_cc_real | mmse_r2_real
               | mmse_n_real | mmse_mae_all_including_imputed]

    Note: l'AUC est threshold-independent — répétée à l'identique sur les lignes
    du même cohort, normal et attendu.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, roc_auc_score,
    )

    rows: List[Dict[str, Any]] = []
    true_arr  = np.asarray(true_cls)
    probs_arr = np.asarray(probs)
    cohorts = df["source"].astype(str).values if "source" in df.columns else None

    cohorts_to_iterate = ["GLOBAL"]
    if cohorts is not None:
        cohorts_to_iterate += sorted(set(cohorts))

    for cohort in cohorts_to_iterate:
        mask = (np.ones(len(true_arr), dtype=bool) if cohort == "GLOBAL"
                else (cohorts == cohort))

        n = int(mask.sum())
        if n == 0:
            continue

        t  = true_arr[mask]
        pr = probs_arr[mask]
        n_cn = int((t == 0).sum())
        n_ad = int((t == 1).sum())

        # AUC : calculée une seule fois, dépend pas du seuil
        auc_value = float("nan")
        if n >= 2 and len(set(t)) >= 2:
            auc_value = float(roc_auc_score(t, pr))

        # MMSE : calculé une seule fois (indépendant du seuil aussi)
        mmse_metrics: Dict[str, Any] = {}
        if mmse_pred is not None and len(mmse_pred) == len(true_arr):
            mp = np.asarray(mmse_pred)[mask]
            mt = np.asarray(mmse_true)[mask]
            hr = np.asarray(has_real)[mask] if has_real is not None else None
            if hr is None or hr.sum() < 2:
                mmse_metrics["mmse_mae_all_including_imputed"] = (
                    float(np.mean(np.abs(mp - mt))) if n > 0 else float("nan")
                )
                mmse_metrics["mmse_mae_real"] = float("nan")
                mmse_metrics["mmse_n_real"]   = 0
            else:
                m = compute_mmse_metrics(mp, mt, hr)
                mmse_metrics["mmse_mae_real"]  = m["mae_real"]
                mmse_metrics["mmse_rmse_real"] = m["rmse_real"]
                mmse_metrics["mmse_cc_real"]   = m["cc_real"]
                mmse_metrics["mmse_r2_real"]   = m["r2_real"]
                mmse_metrics["mmse_n_real"]    = m["n_real"]
                mmse_metrics["mmse_mae_all_including_imputed"] = (
                    float(np.mean(np.abs(mp - mt))) if n > 0 else float("nan")
                )

        # ── Boucle sur les seuils : une ligne par (cohorte, seuil) ─────
        for thr_label, thr_value in thresholds.items():
            preds_thr = (probs_arr >= thr_value).astype(int)
            p = preds_thr[mask]

            row: Dict[str, Any] = {
                "cohort":          cohort,
                "threshold_label": thr_label,
                "threshold":       float(thr_value),
                "n":               n,
                "n_cn":            n_cn,
                "n_ad":            n_ad,
                "auc":             auc_value,
            }

            if n >= 2 and len(set(t)) >= 2:
                row["accuracy"]    = float(accuracy_score(t, p))
                row["f1"]          = float(f1_score(t, p, zero_division=0))
                row["sensitivity"] = float(
                    recall_score(t, p, pos_label=1, zero_division=0)
                )
                row["specificity"] = float(
                    recall_score(t, p, pos_label=0, zero_division=0)
                )
            else:
                for k in ("accuracy", "f1", "sensitivity", "specificity"):
                    row[k] = float("nan")

            # MMSE (identique pour les deux seuils — copié dans chaque ligne
            # pour faciliter l'analyse côté tableur)
            row.update(mmse_metrics)
            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ROUTINE D'ÉVALUATION COMPLÈTE
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    fold: int,
    output_dir: Path,
    ignore_cohort_filter: bool = True,
) -> int:
    """
    Pipeline complet :
        1. Charge config + modèle + tête MMSE
        2. Charge le dataset (test ou val) du fold demandé
        3. Évalue (evaluate_dataset)
        4. Calcule métriques globales + stratifiées par cohorte
        5. Génère plots (ROC, CM, scatter MMSE)
        6. Sauvegarde JSON + CSV + PNG dans output_dir

    Args:
        ignore_cohort_filter : si True (défaut), ignore config.data.cohort_filter
            au moment de l'évaluation. C'est ce qu'on veut pour T8 :
            entraînement filtré (ADNI seul) mais évaluation sur le test
            complet pour comparer entre configs sur la MÊME baseline.

    Retourne 0 si succès, code d'erreur sinon.
    """
    config = load_config(str(config_path))
    setup_env(seed=config["training"].get("seed", 42))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Si demandé, on retire le filtre de la config en mémoire
    if ignore_cohort_filter and config.get("data", {}).get("cohort_filter"):
        original_filter = config["data"].pop("cohort_filter", None)
        print(f"[*] Cohort filter désactivé pour évaluation : "
              f"original = {original_filter}")

    print(f"\n{'='*70}")
    print(f"  ÉVALUATION CHECKPOINT")
    print(f"{'='*70}")
    print(f"  Task        : {config.get('task_name', 'unknown')}")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Split       : {split} (fold_{fold})")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*70}\n")

    # ── Vérifie que le split CSV existe ──────────────────────────────────
    splits_dir = Path(config["data"]["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = (THIS_DIR / splits_dir).resolve()
    csv_path = splits_dir / f"fold_{fold}" / f"{split}.csv"
    if not csv_path.exists():
        print(f"[✗] Split introuvable : {csv_path}")
        return 1

    # ── VRAM check ───────────────────────────────────────────────────────
    try:
        check_vram(min_gb=8.0)  # éval moins gourmande qu'entraînement
    except RuntimeError as e:
        print(f"[!] {e}")
        return 2

    model = None
    try:
        # ── Modèle ───────────────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, model, has_mmse = load_model_for_eval(
            config, checkpoint_path, device=device
        )

        # Token IDs CN/AD — vérification cohérence avec entraînement
        cn_id = processor.tokenizer.encode("CN", add_special_tokens=False)[0]
        ad_id = processor.tokenizer.encode("AD", add_special_tokens=False)[0]
        print(f"[*] Token IDs (eval) → CN: {cn_id}, AD: {ad_id}")

        # Compare aux IDs sauvés au training (eval_metrics.json) si présents
        eval_metrics_path = checkpoint_path / "eval_metrics.json"
        if eval_metrics_path.exists():
            try:
                import json as _json
                with open(eval_metrics_path) as f:
                    saved = _json.load(f)
                saved_cn = saved.get("cn_token_id")
                saved_ad = saved.get("ad_token_id")
                if saved_cn is not None and saved_ad is not None:
                    if saved_cn != cn_id or saved_ad != ad_id:
                        print(f"[!] DRIFT TOKEN IDS détecté !")
                        print(f"    training : CN={saved_cn}, AD={saved_ad}")
                        print(f"    eval     : CN={cn_id}, AD={ad_id}")
                        print(f"    → utilisation des IDs sauvés au training")
                        cn_id, ad_id = saved_cn, saved_ad
                    else:
                        print(f"[✓] Token IDs cohérents avec l'entraînement")
            except Exception as e:
                print(f"[!] Lecture eval_metrics.json échouée : {e}")
        set_token_ids(cn_id, ad_id)

        # ── Dataset ──────────────────────────────────────────────────────
        # Force is_training=True pour avoir les labels (et le label assistant
        # dans la séquence pour que le verbalizer fonctionne).
        # Note : is_training contrôle UNIQUEMENT l'inclusion du label assistant,
        # pas d'augmentation random — donc l'éval reste déterministe.
        ds = TfeDataset(
            str(csv_path), processor, config, is_training=True
        )
        print(f"[*] Dataset {split} : {len(ds)} échantillons")

        # ── Évaluation ───────────────────────────────────────────────────
        print(f"\n[*] Inférence en cours (≈ {len(ds) * 2.5 / 60:.0f} min estimées)...")
        ev_out = evaluate_dataset(
            model, ds, tfe_collate_fn,
            cn_id=cn_id, ad_id=ad_id,
            batch_size=1, device=device,
            return_indices=True,
        )
        results, preds, probs, true_cls, mmse_pred, mmse_true, kept_indices = ev_out

        # Assert alignement
        assert len(preds) == len(probs) == len(true_cls) == len(kept_indices), (
            f"Désalignement evaluate_dataset : preds={len(preds)} "
            f"probs={len(probs)} true_cls={len(true_cls)} kept={len(kept_indices)}"
        )
        n_skipped = len(ds) - len(kept_indices)
        if n_skipped > 0:
            print(f"[!] {n_skipped} sample(s) skippé(s) par evaluate_dataset "
                  f"(verbalizer mismatch). Métriques calculées sur {len(kept_indices)} samples.")

        # ── Métriques globales ───────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  RÉSULTATS GLOBAUX")
        print(f"{'─'*70}")
        print(f"  Accuracy    : {results.get('accuracy', float('nan')):.4f}")
        print(f"  AUC         : {results.get('auc', float('nan')):.4f}")
        print(f"  F1          : {results.get('f1', float('nan')):.4f}")
        print(f"  Sensibilité : {results.get('sensitivity', float('nan')):.4f}")
        print(f"  Spécificité : {results.get('specificity', float('nan')):.4f}")
        if "f1_calibrated" in results:
            print(f"\n  ⚠️ Seuil optimal Youden = {results['optimal_threshold']:.4f} "
                  f"calculé SUR le test → biais optimiste")
            print(f"    F1          : {results['f1_calibrated']:.4f}")
            print(f"    Sensibilité : {results['sensitivity_calibrated']:.4f}")
            print(f"    Spécificité : {results['specificity_calibrated']:.4f}")
            print(f"    → Pour reporting honnête, utiliser le seuil val_optimal_threshold")
            print(f"      stocké dans best_model/eval_metrics.json")
        if has_mmse and not np.isnan(results.get("mae_real", float("nan"))):
            print(f"\n  MMSE (n_real={results.get('n_real', 0)}) :")
            print(f"    MAE  : {results['mae_real']:.2f} pts")
            print(f"    RMSE : {results['rmse_real']:.2f} pts")
            print(f"    CC   : {results['cc_real']:.4f}")
        print(f"{'─'*70}\n")

        # ── Subset df sur kept_indices (alignement strict) ───────────────
        ds_df_kept = ds.df.iloc[kept_indices].reset_index(drop=True)

        # has_real_measures par patient (aligné sur kept_indices)
        if has_mmse and "has_real_measures" in ds_df_kept.columns:
            has_real = ds_df_kept["has_real_measures"].astype(bool).tolist()
        else:
            has_real = None

        # ── Récupération du seuil Youden val (avant compute_cohort_metrics) ──
        # Préfère le seuil sauvé sur val (eval_metrics.json) au seuil recalculé
        # sur test (biaisé optimiste).
        val_optimal_threshold = None
        if eval_metrics_path.exists():
            try:
                import json as _json
                with open(eval_metrics_path) as f:
                    saved = _json.load(f)
                val_optimal_threshold = saved.get("optimal_threshold")
            except Exception:
                pass

        # Construction du dict des seuils à évaluer
        # On reporte SYSTÉMATIQUEMENT les deux : 0.5 (naïf) + Youden val (calibré)
        thresholds_to_eval: Dict[str, float] = {"0.5": 0.5}
        if val_optimal_threshold is not None:
            thresholds_to_eval["youden_val"] = float(val_optimal_threshold)
        else:
            # Pas de checkpoint val → fallback sur seuil optimal test (biaisé)
            test_thr = results.get("optimal_threshold")
            if test_thr is not None:
                thresholds_to_eval["youden_test"] = float(test_thr)
                print(f"[!] Pas de seuil val disponible — utilise seuil test "
                      f"(BIAISÉ, à éviter pour reporting) = {test_thr:.3f}")

        # ── Métriques par cohorte × seuil ────────────────────────────────
        cohort_df = compute_cohort_metrics(
            ds_df_kept,
            probs=probs,
            true_cls=true_cls,
            thresholds=thresholds_to_eval,
            mmse_pred=mmse_pred if has_mmse else None,
            mmse_true=mmse_true if has_mmse else None,
            has_real=has_real,
        )
        print(f"  MÉTRIQUES PAR COHORTE × SEUIL "
              f"({len(thresholds_to_eval)} seuils × "
              f"{cohort_df['cohort'].nunique()} cohortes = "
              f"{len(cohort_df)} lignes)")
        print(f"{'─'*70}")
        print(cohort_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"{'─'*70}\n")

        # ── Sauvegarde ───────────────────────────────────────────────────
        # JSON métriques globales — bloc supplémentaire dual-threshold extrait
        # depuis cohort_df (GLOBAL row) pour traçabilité.
        from sklearn.metrics import (
            accuracy_score, f1_score, recall_score
        )
        true_arr_full = np.asarray(true_cls)
        probs_arr_full = np.asarray(probs)
        per_threshold_global: Dict[str, Dict[str, float]] = {}
        for thr_label, thr_value in thresholds_to_eval.items():
            preds_thr = (probs_arr_full >= thr_value).astype(int)
            per_threshold_global[thr_label] = {
                "threshold":   float(thr_value),
                "accuracy":    float(accuracy_score(true_arr_full, preds_thr)),
                "f1":          float(f1_score(true_arr_full, preds_thr, zero_division=0)),
                "sensitivity": float(recall_score(true_arr_full, preds_thr,
                                                  pos_label=1, zero_division=0)),
                "specificity": float(recall_score(true_arr_full, preds_thr,
                                                  pos_label=0, zero_division=0)),
            }

        metrics_to_save = dict(results)
        metrics_to_save["task_name"]      = config.get("task_name", "unknown")
        metrics_to_save["checkpoint"]     = str(checkpoint_path)
        metrics_to_save["split"]          = split
        metrics_to_save["fold"]           = fold
        metrics_to_save["n_samples"]      = len(ds)
        metrics_to_save["has_mmse_head"]  = has_mmse
        metrics_to_save["thresholds_evaluated"] = per_threshold_global
        save_metrics_json(metrics_to_save, output_dir / f"metrics_{split}.json")
        print(f"  💾 metrics_{split}.json (avec dual-threshold)")

        # Predictions CSV (par patient — aligné via kept_indices)
        pred_df = pd.DataFrame({
            "subject_id": ds_df_kept["subject_id"].values,
            "source":     ds_df_kept["source"].values
                          if "source" in ds_df_kept.columns else "?",
            "true_label": true_cls,
            "pred_label": preds,
            "prob_AD":    probs,
        })
        if has_mmse:
            pred_df["mmse_true"]         = mmse_true
            pred_df["mmse_pred"]         = mmse_pred
            if "has_real_measures" in ds_df_kept.columns:
                pred_df["has_real_mmse"] = ds_df_kept["has_real_measures"].astype(int).values
        # Assert final
        assert len(pred_df) == len(true_cls), (
            f"predictions_csv désaligné : pred_df={len(pred_df)} vs true_cls={len(true_cls)}"
        )
        pred_df.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        print(f"  💾 predictions_{split}.csv ({len(pred_df)} lignes)")

        # Cohort metrics CSV
        cohort_df.to_csv(output_dir / f"cohort_metrics_{split}.csv", index=False)
        print(f"  💾 cohort_metrics_{split}.csv")

        # ROC curve PNG
        plot_roc_curve(
            true_cls, probs,
            output_dir / f"roc_curve_{split}.png",
            title=f"ROC — {config.get('task_name', 'unknown')} ({split} fold_{fold})",
            extra_caption=f"AUC = {results.get('auc', 0):.4f} | n = {len(true_cls)}",
        )
        print(f"  💾 roc_curve_{split}.png")

        # Confusion matrices : UNE PNG par cohorte × seuil
        # → ex: confusion_matrix_test_GLOBAL_thr05.png
        #       confusion_matrix_test_GLOBAL_thrYoudenVal.png
        #       confusion_matrix_test_ADNI_thr05.png, etc.
        # CSV de log : 1 ligne par CM générée pour traçabilité.
        cm_log_rows: List[Dict[str, Any]] = []
        ds_sources = (ds_df_kept["source"].astype(str).values
                      if "source" in ds_df_kept.columns else None)
        true_arr = np.asarray(true_cls)
        probs_arr = np.asarray(probs)

        cohorts_for_cm = ["GLOBAL"]
        if ds_sources is not None:
            cohorts_for_cm += sorted(set(ds_sources))

        for cohort_name in cohorts_for_cm:
            mask = (np.ones(len(true_arr), dtype=bool) if cohort_name == "GLOBAL"
                    else (ds_sources == cohort_name))
            if mask.sum() < 2:
                continue
            t_sub = true_arr[mask].tolist()
            p_sub_prob = probs_arr[mask]

            for thr_label, thr_value in thresholds_to_eval.items():
                # Slug fichier safe : 0.5 → "thr05", youden_val → "thrYoudenVal"
                thr_slug = (
                    "thr05" if thr_label == "0.5" else
                    f"thr{thr_label.replace('_', '').title().replace('.', '')}"
                )
                preds_sub = (p_sub_prob >= thr_value).astype(int).tolist()
                fname = f"confusion_matrix_{split}_{cohort_name}_{thr_slug}.png"
                title = (f"Matrice de confusion — {cohort_name} "
                         f"({split}, seuil = {thr_value:.3f})\n"
                         f"n={mask.sum()} | CN={sum(1 for x in t_sub if x==0)} "
                         f"AD={sum(1 for x in t_sub if x==1)}")
                plot_confusion_matrix(
                    t_sub, preds_sub, output_dir / fname, title=title,
                )
                cm_log_rows.append({
                    "cohort":          cohort_name,
                    "threshold_label": thr_label,
                    "threshold":       float(thr_value),
                    "n":               int(mask.sum()),
                    "filename":        fname,
                })

        pd.DataFrame(cm_log_rows).to_csv(
            output_dir / f"confusion_matrices_index_{split}.csv", index=False
        )
        print(f"  💾 {len(cm_log_rows)} confusion matrices "
              f"({len(cohorts_for_cm)} cohortes × {len(thresholds_to_eval)} seuils) "
              f"→ confusion_matrices_index_{split}.csv"
        )

        # MMSE scatter (si tête présente)
        if has_mmse and len(mmse_pred) > 0:
            real_mask = (
                ds_df_kept["has_real_measures"].astype(bool).values
                if "has_real_measures" in ds_df_kept.columns else None
            )
            plot_mmse_scatter(
                mmse_true, mmse_pred,
                output_dir / f"mmse_scatter_{split}.png",
                real_mask=real_mask,
                title=f"MMSE prédit vs vrai — {split}",
            )
            print(f"  💾 mmse_scatter_{split}.png")

        print(f"\n[✓] Évaluation terminée — résultats dans {output_dir}\n")
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


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Évaluation standalone d'un checkpoint sur test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemples :
  python evaluate.py --task 01_no_mmse
  python evaluate.py --task 02_with_mmse --split val
  python evaluate.py --checkpoint results/01_no_mmse/best_model
  python evaluate.py --task 03_ablation_neuro --output /tmp/eval_T3/
""",
    )
    p.add_argument("--task", type=str, default=None,
                   help="Nom de la tâche (auto-détecte le best_model). "
                        "Ex: 01_no_mmse, 02_with_mmse, 03_ablation_neuro")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Chemin explicite vers un checkpoint (override --task auto-detect)")
    p.add_argument("--config", type=str, default=None,
                   help="Chemin vers le config YAML (sinon : <task_dir>/config.yaml)")
    p.add_argument("--split", type=str, default="test",
                   choices=["val", "test"],
                   help="Split à évaluer (défaut : test)")
    p.add_argument("--fold", type=int, default=0,
                   help="Fold à évaluer (défaut : 0)")
    p.add_argument("--output", type=str, default=None,
                   help="Dossier de sortie (sinon : <checkpoint_dir>/test_results/)")
    p.add_argument("--apply-cohort-filter", action="store_true",
                   help="Applique le filtre cohorte de la config aussi au test "
                        "(par défaut : test sans filtre, pour comparaison T8)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.task is None and args.checkpoint is None:
        print("[!] Spécifie --task ou --checkpoint")
        return 1

    # ── Résolution config + checkpoint ───────────────────────────────────
    if args.task:
        task_dir = resolve_task_dir(args.task)
        config_path = Path(args.config) if args.config else task_dir / "config.yaml"
        if not config_path.exists():
            print(f"[!] Config introuvable : {config_path}")
            return 1

        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint).resolve()
        else:
            config = load_config(str(config_path))
            checkpoint_path = find_best_model(task_dir, config)
            print(f"[*] Best model auto-détecté : {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not args.config:
            print("[!] --config requis quand on utilise --checkpoint sans --task")
            return 1
        config_path = Path(args.config)

    if not is_valid_checkpoint(str(checkpoint_path)):
        print(f"[!] Checkpoint invalide (pas d'adapter_config.json) : {checkpoint_path}")
        return 1

    # ── Output dir ──────────────────────────────────────────────────────
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        # Par défaut : <checkpoint>/test_results/  (à côté du best_model)
        output_dir = checkpoint_path.parent / f"{args.split}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    return run_evaluation(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split=args.split,
        fold=args.fold,
        output_dir=output_dir,
        ignore_cohort_filter=not args.apply_cohort_filter,
    )


if __name__ == "__main__":
    sys.exit(main())