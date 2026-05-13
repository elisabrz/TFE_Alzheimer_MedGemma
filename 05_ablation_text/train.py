"""
train.py — Tâche 5 : Ablation textuelle complète (IRM seules).

Prompt minimal : aucune feature tabulaire, juste les 4 IRM + question CN/AD.
Pas de tête MMSE (auto-détecté depuis config.mmse_head.enabled = false).

Mesure la contribution pure des IRM : ΔAUC(T_best - T5).

Lancement :
    cd 05_ablation_text/
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
        description="T5 — Ablation textuelle complète (prompt minimal, IRM seules)"
    )
    here = Path(__file__).parent
    p.add_argument("--config",  default=str(here / "config.yaml"))
    p.add_argument("--resume",  default=None,
                   help="Forcer reprise depuis ce checkpoint")
    args = p.parse_args()
    run_training(config_path=args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()