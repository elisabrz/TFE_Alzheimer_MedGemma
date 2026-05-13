"""
train.py — Tâche 2 : Multitâche classification + régression MMSE.

Reproduction du best_model_step2 sur les nouveaux splits 5-fold.
Loss : Focal(CN/AD) + α_reg × MSE(MMSE).

Tête MMSE : LayerNorm(2560) + Linear(2560, 1) — auto-activée via config.

Lancement :
    cd 02_train_with_mmse/
    python train.py
    python train.py --resume /chemin/checkpoint
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from trainers import run_training


def main() -> None:
    p = argparse.ArgumentParser(
        description="T2 — Multitâche classification CN/AD + régression MMSE"
    )
    here = Path(__file__).parent
    p.add_argument("--config",  default=str(here / "config.yaml"))
    p.add_argument("--resume",  default=None,
                   help="Forcer reprise depuis ce checkpoint")
    args = p.parse_args()
    run_training(config_path=args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()