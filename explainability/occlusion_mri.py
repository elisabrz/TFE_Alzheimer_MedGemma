"""
occlusion_mri.py — Occlusion par patches sur les coupes IRM.

Méthode (Zeiler & Fergus 2014) :
    Pour chaque patient et chaque vue IRM, on parcourt l'image avec une fenêtre
    glissante (patch de zéros), et on mesure de combien la probabilité P(AD)
    chute. Plus la chute est grande, plus le patch occulté était critique.

Plus lent que Grad-CAM (196 forward passes par image au lieu de 1+1 backward),
mais plus interprétable cliniquement et model-agnostic.

Paramétrage par défaut :
    patch_size = 32, stride = 32 → grille 14×14 sur 448×448 (196 patches/vue)
    → ~5 min/patient sur RTX 4080 (4 vues × 196 patches)
    → 100 patients ≈ 8h. Réduisible avec --patch-size 64 (49 patches → 4× plus rapide)

Outputs (dans <ckpt>/explainability/occlusion/) :
    patient_<id>/
        coronal_1.png, ..., axial_2.png    # heatmaps superposées
        mosaic.png                         # 4 vues combinées
        sensitivity_map.npz                # raw heatmaps (4, 14, 14)
    summary.csv

Lancement :
    python occlusion_mri.py --task 02_with_mmse --n-patients 20 \\
        --strategy stratified
    python occlusion_mri.py --task 02_with_mmse --patch-size 64 \\
        --n-patients 100   # plus rapide mais grille moins fine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from _common import (
    add_common_args, resolve_checkpoint_and_config, resolve_output_dir,
    load_model_and_dataset, select_patients,
    get_raw_slices, overlay_heatmap, save_mosaic, VIEW_NAMES,
    prepare_inputs_for_forward, find_logit_position_for_answer,
    get_ad_prob_from_logits,
)
from utils import release_gpu, save_metrics_json


# ═══════════════════════════════════════════════════════════════════════════
# OCCLUSION CORE
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_baseline_prob(
    model, inputs: Dict[str, torch.Tensor],
    cn_id: int, ad_id: int, logit_pos: int,
) -> float:
    """Prob AD sans occlusion (référence)."""
    out = model(**inputs, use_cache=False, output_hidden_states=False)
    return get_ad_prob_from_logits(out.logits, logit_pos, cn_id, ad_id)


@torch.no_grad()
def compute_occlusion_map_for_view(
    model, inputs_template: Dict[str, torch.Tensor],
    view_idx: int, n_views: int,
    cn_id: int, ad_id: int, logit_pos: int,
    baseline_prob: float,
    patch_size: int = 32, stride: int = 32,
    image_size: int = 448,
) -> np.ndarray:
    """
    Génère la carte de sensibilité pour UNE vue d'IRM.

    inputs_template : pixel_values shape (n_views, C, H, W) = 4 vues empilées.
    On occluera uniquement la vue view_idx.

    Retourne ndarray shape (n_grid, n_grid) avec les Δprob = baseline - occluded.
    """
    grid = (image_size - patch_size) // stride + 1
    sensitivity = np.zeros((grid, grid), dtype=np.float32)

    pixel_orig = inputs_template["pixel_values"].clone()  # (n_views, C, H, W)

    # Valeur d'occlusion = pixel "neutre" du modèle = 0 (puisque normalisé [-1, 1])
    occlusion_value = 0.0

    for i in range(grid):
        for j in range(grid):
            y0 = i * stride
            x0 = j * stride
            y1 = y0 + patch_size
            x1 = x0 + patch_size

            # Modifier UNIQUEMENT la vue view_idx
            occluded = pixel_orig.clone()
            occluded[view_idx, :, y0:y1, x0:x1] = occlusion_value
            inputs_occluded = dict(inputs_template)
            inputs_occluded["pixel_values"] = occluded

            out = model(**inputs_occluded, use_cache=False)
            prob_occluded = get_ad_prob_from_logits(
                out.logits, logit_pos, cn_id, ad_id,
            )
            # Sensibilité = chute de proba (positive = patch important pour AD)
            sensitivity[i, j] = baseline_prob - prob_occluded

    # Normalisation : on garde le signe (positif = supporte AD)
    return sensitivity


def run_occlusion_for_patient(
    model, dataset, patient_idx: int,
    cn_id: int, ad_id: int,
    output_dir: Path,
    patch_size: int = 32, stride: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Pipeline complet occlusion pour un patient."""
    row = dataset.df.iloc[patient_idx]
    subject_id = str(row["subject_id"])
    patient_dir = output_dir / f"patient_{subject_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    raw_slices = get_raw_slices(row["scan_path"], output_size=448)

    item = dataset[patient_idx]
    inputs = prepare_inputs_for_forward(item, device=device)
    labels = item["labels"]
    logit_pos = find_logit_position_for_answer(labels, cn_id, ad_id)
    if logit_pos is None:
        return {"subject_id": subject_id, "error": "logit_pos introuvable"}

    # Baseline
    baseline_prob = get_baseline_prob(model, inputs, cn_id, ad_id, logit_pos)

    # Occlusion par vue
    n_views = inputs["pixel_values"].shape[0]
    overlays: List[np.ndarray] = []
    sensitivity_maps: List[np.ndarray] = []
    max_drops: Dict[str, float] = {}

    for v_idx, view_name in enumerate(VIEW_NAMES):
        if v_idx >= n_views:
            break
        sens_map = compute_occlusion_map_for_view(
            model, inputs, view_idx=v_idx, n_views=n_views,
            cn_id=cn_id, ad_id=ad_id, logit_pos=logit_pos,
            baseline_prob=baseline_prob,
            patch_size=patch_size, stride=stride,
        )
        sensitivity_maps.append(sens_map)
        max_drops[view_name] = float(sens_map.max())

        # Heatmap = valeurs positives uniquement (zones supportant AD)
        # On clipe à 0 et on normalise
        positive_sens = np.clip(sens_map, 0, None)
        raw_img = raw_slices[v_idx]
        overlay = overlay_heatmap(
            raw_img, positive_sens, alpha=0.5, colormap="hot",
        )
        overlays.append(overlay)
        Image.fromarray(overlay).save(patient_dir / f"{view_name}.png")

    # Sauvegarde maps brutes (pour analyses futures)
    np.savez_compressed(
        patient_dir / "sensitivity_map.npz",
        coronal_1=sensitivity_maps[0] if len(sensitivity_maps) > 0 else None,
        coronal_2=sensitivity_maps[1] if len(sensitivity_maps) > 1 else None,
        axial_1=sensitivity_maps[2] if len(sensitivity_maps) > 2 else None,
        axial_2=sensitivity_maps[3] if len(sensitivity_maps) > 3 else None,
        baseline_prob=baseline_prob,
        patch_size=patch_size,
        stride=stride,
    )

    # Mosaïques
    label_text = "AD" if int(row["label"]) == 1 else "CN"
    pred_text = "AD" if baseline_prob > 0.5 else "CN"
    title = (
        f"{subject_id} | True={label_text} | Pred={pred_text} | "
        f"P(AD)={baseline_prob:.3f} | patch={patch_size}px"
    )
    save_mosaic(
        overlays, patient_dir / "mosaic.png",
        titles=VIEW_NAMES, main_title=title,
    )
    save_mosaic(
        raw_slices, patient_dir / "mosaic_raw.png",
        titles=VIEW_NAMES, main_title=f"{subject_id} (raw)",
    )

    return {
        "subject_id":     subject_id,
        "source":         str(row.get("source", "?")),
        "true_label":     int(row["label"]),
        "pred_label":     1 if baseline_prob > 0.5 else 0,
        "baseline_prob":  baseline_prob,
        **{f"max_drop_{v}": max_drops.get(v, 0.0) for v in VIEW_NAMES},
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Occlusion IRM par patches")
    add_common_args(parser)
    parser.add_argument("--patch-size", type=int, default=32,
                        help="Taille patch occlusion (défaut 32px)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride (défaut = patch_size, pas de chevauchement)")
    args = parser.parse_args()
    if args.stride is None:
        args.stride = args.patch_size

    try:
        checkpoint_path, config_path = resolve_checkpoint_and_config(args)
        output_dir = resolve_output_dir(args, checkpoint_path, "occlusion")

        n_patches_per_view = ((448 - args.patch_size) // args.stride + 1) ** 2
        est_per_patient = n_patches_per_view * 4 * 0.3  # 0.3s/forward typique
        est_total_min = (est_per_patient * args.n_patients) / 60

        print(f"\n{'='*70}")
        print(f"  OCCLUSION IRM")
        print(f"{'='*70}")
        print(f"  Checkpoint : {checkpoint_path}")
        print(f"  Strategy   : {args.strategy} (n={args.n_patients})")
        print(f"  Patches    : {args.patch_size}px / stride {args.stride}px")
        print(f"  → {n_patches_per_view} patches/vue × 4 vues = "
              f"{n_patches_per_view * 4} forward/patient")
        print(f"  Estimation : ~{est_total_min:.0f} min total")
        print(f"  Output     : {output_dir}")
        print(f"{'='*70}\n")

        processor, model, dataset, cn_id, ad_id, _ = load_model_and_dataset(
            config_path, checkpoint_path,
            split=args.split, fold=args.fold,
            is_training_for_dataset=True,
        )
        model.eval()

        selected_idx = select_patients(
            dataset.df, n_patients=args.n_patients,
            strategy=args.strategy,
            predictions_csv=args.predictions_csv, seed=args.seed,
        )
        print(f"[*] Patients sélectionnés : {len(selected_idx)}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        results: List[Dict[str, Any]] = []
        errors = 0
        for patient_idx in tqdm(selected_idx, desc="Occlusion"):
            try:
                r = run_occlusion_for_patient(
                    model, dataset, patient_idx,
                    cn_id=cn_id, ad_id=ad_id,
                    output_dir=output_dir,
                    patch_size=args.patch_size, stride=args.stride,
                    device=device,
                )
                results.append(r)
                if "error" in r:
                    errors += 1
            except Exception as e:
                print(f"\n[!] Patient {patient_idx} : {type(e).__name__}: {e}")
                errors += 1

        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        save_metrics_json({
            "n_patients":   len(selected_idx),
            "n_errors":     errors,
            "patch_size":   args.patch_size,
            "stride":       args.stride,
            "checkpoint":   str(checkpoint_path),
            "results":      results,
        }, output_dir / "summary.json")

        print(f"\n[✓] Occlusion terminée : {len(results) - errors}/{len(results)} succès")
        print(f"    Outputs : {output_dir}")
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