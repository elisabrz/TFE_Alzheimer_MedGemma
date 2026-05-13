"""
train.py — Tâche 1 : Baseline classification CN/AD SANS tête MMSE.

Fine-tuning MedGemma 1.5 4B en QLoRA NF4 sur fold_0.
Loss : Focal uniquement (α_AD=0.78, γ=2.0).

Sert de baseline pour quantifier l'apport de la régression MMSE (Tâche 2).
Lancement :
    cd 01_train_no_mmse/
    python train.py
    python train.py --resume /chemin/checkpoint
"""


# 01_train_no_mmse/train.py (futur)
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from trainers import run_training

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    here = Path(__file__).parent
    p.add_argument("--config", default=str(here / "config.yaml"))
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    run_training(config_path=args.config, resume_from=args.resume)