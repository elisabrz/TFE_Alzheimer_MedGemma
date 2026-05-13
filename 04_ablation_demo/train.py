"""
train.py — Tâche 4 : Ablation étendue (neuropsych + anthropométrie).

Retire en plus de T3 : VSWEIGHT et BMI.
→ 8 features conservées : démographie (4) + antécédents médicaux (4).

Lancement :
    cd 04_ablation_demo/
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
        description="T4 — Ablation étendue (8 features : demo + antécédents)"
    )
    here = Path(__file__).parent
    p.add_argument("--config",  default=str(here / "config.yaml"))
    p.add_argument("--resume",  default=None,
                   help="Forcer reprise depuis ce checkpoint")
    args = p.parse_args()
    run_training(config_path=args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()