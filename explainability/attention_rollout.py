"""
attention_rollout.py — Attention rollout (Abnar & Zuidema 2020).

Méthode :
    Pour les Transformers, l'attention au token "AD" final ne suffit pas car
    chaque couche redistribue. On calcule le PRODUIT CUMULÉ des matrices
    d'attention sur toutes les couches (avec skip connection prise en compte) :

        rollout = ∏_l 0.5 * (A_l + I)        # A_l = attention couche l avec heads moyennés

    Le résultat indique pour chaque token de sortie quelle proportion d'attention
    a "flowé" depuis chaque token d'entrée à travers tout le réseau.

    On extrait la ligne correspondant au logit_pos (token avant la réponse),
    et on l'analyse :
        - Tokens image (vision tokens) → contribution de chaque vue IRM
        - Tokens texte → contribution de chaque feature du prompt
        - Tokens spéciaux → "image_start", "BOS", etc.

Outputs (dans <ckpt>/explainability/attention_rollout/) :
    patient_<id>/
        rollout_image_tokens.png    # heatmap des contribs image (4 vues)
        rollout_text_tokens.png     # contribs des tokens texte (top 30)
        rollout_full.npz            # rollout matrix complète + tokens decoded
    summary.csv                     # contribution agrégée par bucket
                                    # (image_total, text_total, special_total)

Lancement :
    python attention_rollout.py --task 02_with_mmse --n-patients 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from _common import (
    add_common_args, resolve_checkpoint_and_config, resolve_output_dir,
    load_model_and_dataset, select_patients,
    prepare_inputs_for_forward, find_logit_position_for_answer,
)
from utils import release_gpu, save_metrics_json


# ═══════════════════════════════════════════════════════════════════════════
# 1. ATTENTION ROLLOUT
# ═══════════════════════════════════════════════════════════════════════════

def compute_attention_rollout(
    attentions: Tuple[torch.Tensor, ...],
    add_residual: bool = True,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """
    Calcule le rollout d'attention sur toutes les couches.

    Args:
        attentions   : tuple (n_layers,) de tenseurs (1, n_heads, seq, seq)
        add_residual : ajoute identité (skip connection) avant produit
        head_fusion  : 'mean' ou 'max' pour fusionner les heads

    Retourne (seq, seq) : la matrice de rollout finale.
    """
    if not attentions:
        raise ValueError("attentions vide")

    # Fusion heads par couche → (n_layers, seq, seq)
    fused = []
    for att in attentions:
        if att.dim() == 4:
            att = att[0]  # (n_heads, seq, seq)
        if head_fusion == "mean":
            f = att.mean(dim=0)
        elif head_fusion == "max":
            f = att.max(dim=0)[0]
        else:
            raise ValueError(f"head_fusion inconnu : {head_fusion}")
        fused.append(f.float())

    # Produit cumulatif avec skip
    seq = fused[0].shape[0]
    eye = torch.eye(seq, device=fused[0].device, dtype=fused[0].dtype)
    rollout = eye.clone()
    for f in fused:
        if add_residual:
            f = 0.5 * f + 0.5 * eye
            # Re-normalisation par ligne
            f = f / (f.sum(dim=-1, keepdim=True) + 1e-9)
        rollout = f @ rollout

    return rollout


def identify_token_buckets(
    input_ids: torch.Tensor, processor,
) -> Dict[str, List[int]]:
    """
    Catégorise les tokens en buckets : image / text / special.

    Heuristique :
        - Tokens spéciaux : ID dans tokenizer.all_special_ids
        - Tokens image : tokens d'image MedGemma (typiquement 256 tokens
          consécutifs par vue, ID typique ~262144 ou via vocab tokenizer)
        - Tokens texte : le reste

    Note : MedGemma utilise des tokens spéciaux <image_start>, <image>, etc.
    L'identification précise dépend du processor.
    """
    tokenizer = processor.tokenizer
    special_ids = set(tokenizer.all_special_ids)

    # Tentative de récupération des IDs image-related
    image_token_ids = set()
    for name in ["<image>", "<image_soft_token>", "<start_of_image>", "<end_of_image>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid is not None and tid != tokenizer.unk_token_id:
                image_token_ids.add(tid)
        except Exception:
            pass

    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    if isinstance(ids[0], list):
        ids = ids[0]

    image_indices: List[int] = []
    text_indices: List[int] = []
    special_indices: List[int] = []
    for i, tok in enumerate(ids):
        if tok in special_ids:
            special_indices.append(i)
        elif tok in image_token_ids or tok > 250000:
            # MedGemma utilise des IDs > 250000 pour les image tokens
            image_indices.append(i)
        else:
            text_indices.append(i)

    return {
        "image":   image_indices,
        "text":    text_indices,
        "special": special_indices,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. PIPELINE PAR PATIENT
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_rollout_for_patient(
    model, dataset, patient_idx: int,
    processor, cn_id: int, ad_id: int,
    output_dir: Path, device: str = "cuda",
) -> Dict[str, Any]:
    """Pipeline rollout pour un patient."""
    row = dataset.df.iloc[patient_idx]
    subject_id = str(row["subject_id"])
    patient_dir = output_dir / f"patient_{subject_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    item = dataset[patient_idx]
    inputs = prepare_inputs_for_forward(item, device=device)
    labels = item["labels"]
    logit_pos = find_logit_position_for_answer(labels, cn_id, ad_id)
    if logit_pos is None:
        return {"subject_id": subject_id, "error": "logit_pos introuvable"}

    # Forward avec output_attentions=True
    try:
        outputs = model(
            **inputs, use_cache=False,
            output_attentions=True, output_hidden_states=False,
        )
    except Exception as e:
        return {
            "subject_id": subject_id,
            "error":      f"forward avec output_attentions=True : {e}",
        }

    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        return {
            "subject_id": subject_id,
            "error":      "outputs.attentions est None (modèle ne supporte pas ?)",
        }

    # Rollout
    rollout = compute_attention_rollout(outputs.attentions, add_residual=True)

    # Distribution d'attention au logit_pos
    seq_len = rollout.shape[0]
    if logit_pos >= seq_len:
        logit_pos = seq_len - 1
    attn_at_logit = rollout[logit_pos].cpu().numpy()  # (seq,)

    # Probabilités CN / AD au logit_pos
    cn_logit = float(outputs.logits[0, logit_pos, cn_id].cpu())
    ad_logit = float(outputs.logits[0, logit_pos, ad_id].cpu())
    exp_cn = np.exp(cn_logit)
    exp_ad = np.exp(ad_logit)
    prob_ad = exp_ad / (exp_cn + exp_ad)

    # Bucketisation tokens
    buckets = identify_token_buckets(inputs["input_ids"][0], processor)

    # Agrégation par bucket
    image_total = float(np.sum(attn_at_logit[buckets["image"]])) if buckets["image"] else 0.0
    text_total = float(np.sum(attn_at_logit[buckets["text"]])) if buckets["text"] else 0.0
    special_total = (
        float(np.sum(attn_at_logit[buckets["special"]])) if buckets["special"] else 0.0
    )
    total = image_total + text_total + special_total
    if total > 0:
        image_pct = image_total / total
        text_pct = text_total / total
        special_pct = special_total / total
    else:
        image_pct = text_pct = special_pct = 0.0

    # ── Plot 1 : contribution par bucket ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    labels_pie = ["Image", "Text", "Special"]
    values = [image_pct, text_pct, special_pct]
    colors_pie = ["#2C7BB6", "#FDAE61", "#888888"]
    ax.barh(labels_pie, values, color=colors_pie)
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=10)
    ax.set_xlim([0, max(values) * 1.2 if max(values) > 0 else 1])
    ax.set_xlabel("Attention rollout (proportion)")
    ax.set_title(
        f"{subject_id} | True={'AD' if int(row['label'])==1 else 'CN'} | "
        f"P(AD)={prob_ad:.3f}",
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(patient_dir / "rollout_buckets.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2 : top tokens texte (les plus regardés par le logit AD) ────
    text_indices = buckets["text"]
    if text_indices:
        text_attn = attn_at_logit[text_indices]
        top_n = min(30, len(text_indices))
        top_idx = np.argsort(text_attn)[-top_n:][::-1]
        top_token_ids = [
            int(inputs["input_ids"][0, text_indices[i]].cpu()) for i in top_idx
        ]
        top_tokens = processor.tokenizer.convert_ids_to_tokens(top_token_ids)
        top_attns = text_attn[top_idx]

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
        y = np.arange(len(top_tokens))
        ax.barh(y, top_attns, color="#FDAE61")
        ax.set_yticks(y)
        ax.set_yticklabels(top_tokens, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Attention rollout")
        ax.set_title(
            f"Top {top_n} text tokens — {subject_id}", fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(patient_dir / "rollout_top_text_tokens.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── Sauvegarde brute ─────────────────────────────────────────────────
    np.savez_compressed(
        patient_dir / "rollout_full.npz",
        rollout=rollout.cpu().numpy(),
        attn_at_logit=attn_at_logit,
        image_indices=np.array(buckets["image"]),
        text_indices=np.array(buckets["text"]),
        special_indices=np.array(buckets["special"]),
        prob_ad=prob_ad,
    )

    return {
        "subject_id":      subject_id,
        "source":          str(row.get("source", "?")),
        "true_label":      int(row["label"]),
        "pred_label":      1 if prob_ad > 0.5 else 0,
        "prob_AD":         prob_ad,
        "image_pct":       image_pct,
        "text_pct":        text_pct,
        "special_pct":     special_pct,
        "n_image_tokens":  len(buckets["image"]),
        "n_text_tokens":   len(buckets["text"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Attention rollout")
    add_common_args(parser)
    args = parser.parse_args()

    try:
        checkpoint_path, config_path = resolve_checkpoint_and_config(args)
        output_dir = resolve_output_dir(args, checkpoint_path, "attention_rollout")

        print(f"\n{'='*70}")
        print(f"  ATTENTION ROLLOUT")
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

        selected_idx = select_patients(
            dataset.df, n_patients=args.n_patients,
            strategy=args.strategy,
            predictions_csv=args.predictions_csv, seed=args.seed,
        )
        print(f"[*] Patients sélectionnés : {len(selected_idx)}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        results: List[Dict[str, Any]] = []
        errors = 0
        for idx in tqdm(selected_idx, desc="Rollout"):
            try:
                r = run_rollout_for_patient(
                    model, dataset, idx,
                    processor=processor, cn_id=cn_id, ad_id=ad_id,
                    output_dir=output_dir, device=device,
                )
                results.append(r)
                if "error" in r:
                    errors += 1
            except Exception as e:
                print(f"\n[!] Patient {idx} : {type(e).__name__}: {e}")
                errors += 1

        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_dir / "summary.csv", index=False)

        # Stats globales
        if len(summary_df) > 0:
            print(f"\n  Contribution moyenne par bucket :")
            for bucket in ["image", "text", "special"]:
                col = f"{bucket}_pct"
                if col in summary_df.columns:
                    mean_val = summary_df[col].mean() * 100
                    print(f"    {bucket:8s} : {mean_val:.1f}%")

        save_metrics_json({
            "n_patients":   len(selected_idx),
            "n_errors":     errors,
            "checkpoint":   str(checkpoint_path),
            "results":      results,
        }, output_dir / "summary.json")

        print(f"\n[✓] Attention rollout : {len(results) - errors}/{len(results)} succès")
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