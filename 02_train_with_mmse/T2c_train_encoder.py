"""
T2c_train_encoder.py — Lance T2-c (extraction MMSE niveau encodeur).

Réutilise le pipeline de trainers.py (chargement modèle, dataset, trainer args)
mais remplace TfeMedGemmaWithMMSE + TfeMultitaskTrainer par les variantes
encoder de trainers_encoder.py.

Usage :
    python T2c_train_encoder.py --config 02_with_mmse/T2c_config_encoder.yaml

NE TOUCHE PAS aux fichiers existants. Les autres tâches (T1, T2, T6) restent
intactes et utilisent leurs entrées habituelles via 0X_*/train.py.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

# Project root sur PYTHONPATH
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

import torch
import wandb
from transformers import AutoProcessor

from utils import (
    setup_env, signal_handler, set_token_ids, register_signal_refs,
    check_vram, release_gpu, load_config, get_last_checkpoint,
    load_mmse_head,
)
from dataset import TfeDataset, tfe_collate_fn
from trainers import (
    _resolve_path, _build_quantization, _load_base_and_lora,
    _build_training_args, _verify_verbalizer_tokens,
    ClearCacheCallback,
)
from utils import EvalCallback
from trainers_encoder import (
    TfeMedGemmaWithMMSEEncoder,
    TfeMultitaskTrainerEncoder,
    get_vision_hidden_size,
)


def parse_args():
    p = argparse.ArgumentParser(description="T2-c : MMSE extraction encodeur")
    p.add_argument("--config", type=str, required=True,
                   help="Chemin config YAML T2-c")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint à reprendre (optionnel)")
    p.add_argument("--pool-strategy", type=str, default="cls",
                   choices=["cls", "mean"],
                   help="Pooling sur les tokens vision (cls = token 0, mean = moyenne)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[!] Config introuvable : {config_path}")
        return 1

    config = load_config(str(config_path))
    project_root = config_path.parent.parent

    if not config.get("mmse_head", {}).get("enabled", False):
        print("[!] T2-c nécessite mmse_head.enabled=True dans la config")
        return 1

    # ── Setup ────────────────────────────────────────────────────────────
    setup_env(seed=config["training"]["seed"])
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    out_dir = str(_resolve_path(config["training"]["output_dir"], project_root))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Output dir : {out_dir}")
    print(f"[*] Pool strategy : {args.pool_strategy}")

    splits_dir = _resolve_path(config["data"]["splits_dir"], project_root)
    fold = config["data"]["fold"]
    fold_dir = splits_dir / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Splits introuvables : {fold_dir}")

    last_checkpoint = args.resume if args.resume else get_last_checkpoint(out_dir)
    if last_checkpoint:
        print(f"[*] Reprise depuis : {last_checkpoint}")

    trainer = None
    model = None

    try:
        check_vram(min_gb=10.0)

        # ── WandB ────────────────────────────────────────────────────────
        log_cfg = config.get("logging", {})
        wandb.init(
            project=log_cfg.get("wandb_project", "tfe-alzheimer-medgemma"),
            name=log_cfg.get("run_name", "t2c_encoder_fold0_seed42"),
            config={**config, "extraction_level": "encoder",
                    "pool_strategy": args.pool_strategy},
            resume="allow",
        )

        # ── Processor + token IDs ────────────────────────────────────────
        processor = AutoProcessor.from_pretrained(config["model"]["name"])
        cn_id = processor.tokenizer.encode("CN", add_special_tokens=False)[0]
        ad_id = processor.tokenizer.encode("AD", add_special_tokens=False)[0]
        print(f"[*] Token IDs → CN: {cn_id}, AD: {ad_id}")
        set_token_ids(cn_id, ad_id)
        _verify_verbalizer_tokens(processor, cn_id, ad_id)

        # ── Modèle (base + LoRA + wrapper ENCODER) ───────────────────────
        base_model, peft_model = _load_base_and_lora(config, last_checkpoint)

        # Hidden size du vision encoder (≠ LM)
        vision_hidden_size = get_vision_hidden_size(peft_model)
        print(f"[*] Vision hidden size : {vision_hidden_size}")

        model = TfeMedGemmaWithMMSEEncoder(
            peft_model,
            vision_hidden_size=vision_hidden_size,
            pool_strategy=args.pool_strategy,
        )
        print(f"[*] Tête MMSE encoder : LayerNorm({vision_hidden_size}) + "
              f"Linear({vision_hidden_size}→1) fp32")

        if last_checkpoint:
            head_path = Path(last_checkpoint) / "mmse_head.pt"
            if head_path.exists():
                model.mmse_head = load_mmse_head(
                    last_checkpoint, hidden_size=vision_hidden_size,
                )
                model.mmse_head = model.mmse_head.to("cuda")

        register_signal_refs(model=model)

        # ── Datasets ─────────────────────────────────────────────────────
        cohort_filter = config.get("data", {}).get("cohort_filter")
        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"
        train_ds = TfeDataset(
            str(train_csv), processor, config,
            is_training=True, cohort_filter=cohort_filter,
        )
        val_ds = TfeDataset(
            str(val_csv), processor, config,
            is_training=True, cohort_filter=cohort_filter,
        )
        print(f"[*] Train : {len(train_ds)} | Val : {len(val_ds)}")

        # ── EvalCallback (réutilisation totale, hidden_state hook ignoré) ─
        es = config.get("early_stopping", {})
        eval_callback = EvalCallback(
            val_dataset=val_ds,
            collate_fn=tfe_collate_fn,
            cn_id=cn_id,
            ad_id=ad_id,
            output_dir=out_dir,
            best_name=config.get("training", {}).get("best_model_name", "best_model"),
            patience=es.get("patience", 2),
            min_delta=es.get("min_delta", 0.005),
            use_wandb=True,
            processor=processor,
            metric_name=es.get("metric", "auc"),
        )

        # ── TrainingArguments ────────────────────────────────────────────
        training_args = _build_training_args(config, out_dir)

        # ── Trainer ENCODER ──────────────────────────────────────────────
        loss_cfg = config.get("loss", {})
        mmse_cfg = config.get("mmse_head", {})
        trainer = TfeMultitaskTrainerEncoder(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=tfe_collate_fn,
            callbacks=[ClearCacheCallback(), eval_callback],
            focal_alpha_ad=loss_cfg.get("focal_alpha", 0.78),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha_reg=mmse_cfg.get("loss_weight", 1.0),
            cn_token_id=cn_id,
            ad_token_id=ad_id,
            pool_strategy=args.pool_strategy,
        )

        register_signal_refs(trainer=trainer, model=model)

        print("\n[*] Configuration T2-c :")
        print(f"    extraction      = encoder (vision_tower)")
        print(f"    pool_strategy   = {args.pool_strategy}")
        print(f"    focal_alpha_AD  = {loss_cfg.get('focal_alpha', 0.78)}")
        print(f"    focal_gamma     = {loss_cfg.get('focal_gamma', 2.0)}")
        print(f"    alpha_reg       = {mmse_cfg.get('loss_weight', 1.0)}")
        print(f"    epochs max      = {config['training']['num_train_epochs']}")
        print(f"    early_stop      = patience={es.get('patience', 2)}, "
              f"min_delta={es.get('min_delta', 0.005)}, metric={es.get('metric', 'auc')}\n")

        # ── Train ────────────────────────────────────────────────────────
        if last_checkpoint:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

        print(f"\n[✓] Entraînement T2-c terminé.")
        print(f"    Meilleur métrique : {eval_callback.best_metric:.4f}")
        print(f"    Meilleur modèle  : {eval_callback.best_path}")
        return 0

    except KeyboardInterrupt:
        print("\n[!] Interruption — sauvegarde en cours...")
        return 130
    except Exception as e:
        print(f"\n[!] Erreur : {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1
    finally:
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        release_gpu(model=model)


if __name__ == "__main__":
    sys.exit(main())
