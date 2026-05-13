"""
gradcam_mri.py — Grad-CAM sur le vision encoder MedSigLIP.

Méthode (Selvaraju et al. 2017) :
    1. Hook forward + backward sur la dernière couche conv/attn du vision encoder
    2. Forward pass sur (image, prompt complet)
    3. Backward du logit AD (au logit_pos) par rapport aux activations
    4. Pondération des feature maps par les gradients moyens
    5. ReLU + upscale → heatmap 448×448 superposée à l'IRM brute

Sélection patients :
    --strategy stratified  : équilibre CN/AD × cohortes (défaut)
    --strategy tp_fn_mix   : 4 buckets TP/FN/TN/Hard (nécessite predictions.csv)
    --strategy random      : aléatoire 50/50

Outputs (dans <ckpt>/explainability/gradcam/) :
    patient_<id>/
        coronal_1.png         # heatmap superposée vue par vue
        coronal_2.png
        axial_1.png
        axial_2.png
        mosaic.png            # 4 vues avec heatmaps en mosaïque
        mosaic_raw.png        # 4 vues brutes (référence)
    summary.csv               # 1 ligne/patient : true, pred, prob, max_attention_per_view
    summary.json              # même chose en JSON

Lancement :
    python gradcam_mri.py --task 01_no_mmse --n-patients 100 --strategy stratified
    python gradcam_mri.py --task 02_with_mmse --strategy tp_fn_mix \\
        --predictions-csv results/02_with_mmse/best_model/test_results/predictions_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from _common import (
    add_common_args, resolve_checkpoint_and_config, resolve_output_dir,
    load_model_and_dataset, select_patients,
    get_raw_slices, overlay_heatmap, save_mosaic, VIEW_NAMES,
    find_vision_encoder_last_layer,
    prepare_inputs_for_forward, find_logit_position_for_answer,
    get_ad_prob_from_logits,
)
from utils import release_gpu, save_metrics_json
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════
# GRAD-CAM CORE
# ═══════════════════════════════════════════════════════════════════════════

class GradCAMHook:
    """
    Capture activations + gradients sur une couche cible.
    Compatible avec ViT (la couche cible doit retourner un tenseur ou tuple
    dont le premier élément est le tenseur des activations).
    """

    def __init__(self, target_layer: torch.nn.Module):
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.requires_grad:
            out.retain_grad()
        self.activations = out  # référence GPU gardée pour le backward

    def _bwd_hook(self, module, grad_input, grad_output):
        g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        # Déplacer immédiatement sur CPU → libère la VRAM
        self.gradients = g.detach().cpu().float()
        if self.activations is not None:
            self.activations = self.activations.detach().cpu().float()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def compute_gradcam_heatmap(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    n_views: int = 4,
    grid_size: Optional[int] = None,
) -> np.ndarray:
    """
    Calcule heatmap Grad-CAM par vue.

    Args:
        activations  : (n_views, n_tokens, hidden) ou (1, n_views * n_tokens, hidden)
                       — selon comment SigLIP retourne ses tokens
        gradients    : même shape
        n_views      : nombre de vues IRM concaténées sur l'axe batch
        grid_size    : taille du grille spatiale (sqrt(n_tokens))

    Retourne ndarray shape (n_views, grid_size, grid_size) en [0, 1].
    """
    # Aplatissement standard ViT : (batch, seq, hidden)
    if activations.dim() == 3:
        # SigLIP-ViT : (n_views, n_tokens+cls, hidden) typiquement
        # On prend la moyenne pondérée pixel-wise

        # Si batch == n_views et n_tokens correspond à une grille carrée
        b, n_tok, _ = activations.shape
        # Skip le CLS token si présent (n_tok est carré-1 typiquement)
        if grid_size is None:
            # Auto-détection grid : ⌊sqrt(n_tok)⌋, skip 1 token CLS si reste
            sqrt_n = int(np.sqrt(n_tok))
            if sqrt_n * sqrt_n == n_tok:
                grid_size = sqrt_n
                tokens_act = activations
                tokens_grad = gradients
            elif sqrt_n * sqrt_n == n_tok - 1:
                # Skip CLS au position 0
                grid_size = sqrt_n
                tokens_act = activations[:, 1:, :]
                tokens_grad = gradients[:, 1:, :]
            else:
                # Fallback : prendre les sqrt²(n_tok) premiers
                grid_size = sqrt_n
                tokens_act = activations[:, : sqrt_n * sqrt_n, :]
                tokens_grad = gradients[:, : sqrt_n * sqrt_n, :]
        else:
            tokens_act = activations[:, : grid_size * grid_size, :]
            tokens_grad = gradients[:, : grid_size * grid_size, :]

        # Pondération : moyenne des gradients par feature map
        # weights shape : (b, hidden)
        weights = tokens_grad.mean(dim=1)  # global average pool sur les tokens
        # heatmap : (b, n_tokens) = somme pondérée
        cam = (tokens_act * weights.unsqueeze(1)).sum(dim=-1)  # (b, n_tokens)
        cam = F.relu(cam)  # ReLU standard Grad-CAM

        # Reshape en grilles 2D
        cam = cam.view(b, grid_size, grid_size)

        # Normalisation [0, 1] par image
        cam_min = cam.view(b, -1).min(dim=1, keepdim=True)[0].view(b, 1, 1)
        cam_max = cam.view(b, -1).max(dim=1, keepdim=True)[0].view(b, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-9)

        # Si on a plus de batch que n_views, prendre les n_views premiers
        if cam.shape[0] >= n_views:
            cam = cam[:n_views]
        return cam.detach().cpu().numpy()

    raise ValueError(
        f"Shape activations inattendue : {activations.shape}. "
        f"Attendu 3D (b, seq, hidden)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PAR PATIENT
# ═══════════════════════════════════════════════════════════════════════════

def _get_vision_tower(model) -> Optional[torch.nn.Module]:
    """Remonte jusqu'au module vision_tower quel que soit le wrapping PEFT."""
    target = model
    for _ in range(4):
        if hasattr(target, "vision_tower"):
            return target.vision_tower
        if hasattr(target, "model"):
            target = target.model
        else:
            break
    for name, module in model.named_modules():
        if name.endswith("vision_tower"):
            return module
    return None


def run_gradcam_for_patient(
    model, dataset, patient_idx: int,
    target_layer: torch.nn.Module,
    cn_id: int, ad_id: int,
    output_dir: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Génère heatmaps Grad-CAM pour un patient.

    Stratégie mémoire : backward UNIQUEMENT dans le vision encoder SigLIP.
      - Pass 1 : forward complet torch.no_grad() → prob_AD pour annotation
      - Pass 2 : forward vision_tower seul avec pixel_values.requires_grad=True
                 → backward de la norme du CLS token vers les activations SigLIP
      Mémoire : ~3-4 GB (vs ~15 GB pour un backward full-LLM).
    """
    row = dataset.df.iloc[patient_idx]
    subject_id = str(row["subject_id"])
    patient_dir = output_dir / f"patient_{subject_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    raw_slices = get_raw_slices(row["scan_path"], output_size=448)

    item = dataset[patient_idx]
    inputs = prepare_inputs_for_forward(item, device=device)
    labels = item["labels"]
    logit_pos = find_logit_position_for_answer(labels, cn_id, ad_id)

    # ── Pass 1 : forward complet no_grad → prob_AD ────────────────────
    prob_ad = 0.5
    with torch.no_grad():
        try:
            out_full = model(**inputs, use_cache=False,
                             output_hidden_states=False)
            if logit_pos is not None:
                ad_s = out_full.logits[0, logit_pos, ad_id].float()
                cn_s = out_full.logits[0, logit_pos, cn_id].float()
                prob_ad = float(
                    torch.softmax(torch.stack([cn_s, ad_s]), dim=0)[1].item()
                )
            del out_full
        except Exception:
            pass
    torch.cuda.empty_cache()

    # ── Pass 2 : forward vision_tower avec grad → backward SigLIP ────
    vision_tower = _get_vision_tower(model)
    if vision_tower is None:
        return {"subject_id": subject_id, "error": "vision_tower introuvable"}

    # RÈGLE GRAD-CAM : au moins l'entrée doit avoir requires_grad=True
    # pour que le graphe de calcul existe et que backward() fonctionne.
    # On convertit en float32 (les noyaux NF4 acceptent fp32 en entrée),
    # et on coupe du graphe du pass 1 via detach().
    # PAS d'autocast ici : peut interrompre le graphe avec bitsandbytes NF4.
    pv = inputs["pixel_values"].detach().float()  # (n_views, 3, H, W) fp32
    pv.requires_grad_(True)                        # ← construit le graphe

    cam_hook = GradCAMHook(target_layer)
    vision_tower.eval()

    try:
        # torch.enable_grad() explicite — sécurité si un contexte no_grad
        # externe était actif (ex: appelé depuis un loop d'éval)
        with torch.enable_grad():
            vision_out = vision_tower(
                pixel_values=pv,
                output_attentions=False,
                output_hidden_states=False,
            )
            # CLS token : (n_views, hidden_dim)
            cls_tokens = vision_out.last_hidden_state[:, 0, :]
            # Score proxy : norme L2 agrégée sur toutes les vues
            proxy_score = cls_tokens.norm(dim=-1).sum()
            proxy_score.backward()  # backward uniquement dans vision_tower

        if cam_hook.activations is None or cam_hook.gradients is None:
            return {"subject_id": subject_id,
                    "error": "Hook n'a pas capturé activations/gradients"}

        # activations et gradients déjà sur CPU (déplacés dans le hook)
        heatmaps = compute_gradcam_heatmap(
            cam_hook.activations, cam_hook.gradients, n_views=4,
        )

    finally:
        cam_hook.remove()
        vision_tower.zero_grad()
        del pv
        if "vision_out" in dir():
            del vision_out
        torch.cuda.empty_cache()

    # ── Sauvegarde ───────────────────────────────────────────────────
    overlays: List[np.ndarray] = []
    max_intensities: Dict[str, float] = {}
    for v_idx, view_name in enumerate(VIEW_NAMES):
        if v_idx >= heatmaps.shape[0]:
            break
        hm = heatmaps[v_idx]
        max_intensities[view_name] = float(hm.max())
        overlay = overlay_heatmap(raw_slices[v_idx], hm,
                                  alpha=0.45, colormap="jet")
        overlays.append(overlay)
        Image.fromarray(overlay).save(patient_dir / f"{view_name}.png")

    label_text = "AD" if int(row["label"]) == 1 else "CN"
    pred_text  = "AD" if prob_ad > 0.5 else "CN"
    save_mosaic(overlays, patient_dir / "mosaic.png",
                titles=VIEW_NAMES,
                main_title=(f"{subject_id} | True={label_text} "
                            f"| Pred={pred_text} | P(AD)={prob_ad:.3f}"))
    save_mosaic(raw_slices, patient_dir / "mosaic_raw.png",
                titles=VIEW_NAMES,
                main_title=f"{subject_id} (raw IRM)")

    return {
        "subject_id":  subject_id,
        "source":      str(row.get("source", "?")),
        "true_label":  int(row["label"]),
        "pred_label":  1 if prob_ad > 0.5 else 0,
        "prob_AD":     prob_ad,
        "logit_pos":   int(logit_pos) if logit_pos is not None else -1,
        **{f"max_{v}": max_intensities.get(v, 0.0) for v in VIEW_NAMES},
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM IRM sur MedSigLIP")
    add_common_args(parser)
    args = parser.parse_args()

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        # ── Résolution paths ─────────────────────────────────────────────
        checkpoint_path, config_path = resolve_checkpoint_and_config(args)
        output_dir = resolve_output_dir(args, checkpoint_path, "gradcam")

        print(f"\n{'='*70}")
        print(f"  GRAD-CAM MRI")
        print(f"{'='*70}")
        print(f"  Checkpoint : {checkpoint_path}")
        print(f"  Strategy   : {args.strategy} (n={args.n_patients})")
        print(f"  Output     : {output_dir}")
        print(f"{'='*70}\n")

        # ── Modèle + dataset ─────────────────────────────────────────────
        processor, model, dataset, cn_id, ad_id, _ = load_model_and_dataset(
            config_path, checkpoint_path,
            split=args.split, fold=args.fold,
            is_training_for_dataset=True,  # besoin labels assistant
        )
        print(f"[*] Dataset {args.split} : {len(dataset)} patients")

        # ── Sélection patients ───────────────────────────────────────────
        selected_idx = select_patients(
            dataset.df,
            n_patients=args.n_patients,
            strategy=args.strategy,
            predictions_csv=args.predictions_csv,
            seed=args.seed,
            filter_real_only=args.filter_real_only,
        )
        print(f"[*] Patients sélectionnés : {len(selected_idx)}")

        # ── Couche cible ─────────────────────────────────────────────────
        target_layer = find_vision_encoder_last_layer(model)
        if target_layer is None:
            print("[!] Couche cible vision encoder introuvable")
            return 1
        print(f"[*] Couche cible : {type(target_layer).__name__}")

        # ── Boucle Grad-CAM ──────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results: List[Dict[str, Any]] = []
        errors = 0
        for patient_idx in tqdm(selected_idx, desc="Grad-CAM"):
            try:
                r = run_gradcam_for_patient(
                    model, dataset, patient_idx,
                    target_layer=target_layer,
                    cn_id=cn_id, ad_id=ad_id,
                    output_dir=output_dir, device=device,
                )
                results.append(r)
                if "error" in r:
                    errors += 1
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n[!] OOM patient {patient_idx} — skip")
                torch.cuda.empty_cache()
                errors += 1
                results.append({
                    "subject_id": str(dataset.df.iloc[patient_idx].get("subject_id", "?")),
                    "error": f"OOM: {e}",
                })
            except Exception as e:
                print(f"\n[!] Patient {patient_idx} : {type(e).__name__}: {e}")
                errors += 1
                results.append({
                    "subject_id": str(dataset.df.iloc[patient_idx].get("subject_id", "?")),
                    "error":      f"{type(e).__name__}: {e}",
                })
            finally:
                torch.cuda.empty_cache()

        # ── Sauvegarde résumé ────────────────────────────────────────────
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        save_metrics_json({
            "n_patients":   len(selected_idx),
            "n_errors":     errors,
            "strategy":     args.strategy,
            "checkpoint":   str(checkpoint_path),
            "results":      results,
        }, output_dir / "summary.json")

        print(f"\n[✓] Grad-CAM terminé : {len(results) - errors}/{len(results)} succès")
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