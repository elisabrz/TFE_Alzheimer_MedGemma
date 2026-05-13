"""
train.py — Tâche 8 : ADNI vs no-ADNI (généralisation cross-cohorte).

Compare deux entraînements sur le MÊME test set (1213 patients toutes cohortes) :
    - config_adni_only.yaml : train sur ADNI seul (~903 sujets train)
    - config_no_adni.yaml   : train sur NACC+OASIS (~5162 sujets train)

Hypothèse scientifique : si le modèle ADNI-seul généralise bien à NACC/OASIS
mais pas l'inverse, cela suggère que ADNI est dans la "distribution préférée"
de MedGemma (contamination potentielle pre-train).

Le filtrage cohorte est lu depuis la config (data.cohort_filter) et appliqué
au train + val. Pour le test, evaluate.py ignore par défaut le filtre
(sauf --apply-cohort-filter) pour garder le même test set sur les deux runs.

Lancement :
    cd 08_adni_vs_no_adni/
    python train.py --config config_adni_only.yaml    # ~10h (903 sujets)
    python train.py --config config_no_adni.yaml      # ~25h (5162 sujets)

Évaluation comparative (après training) :
    python ../evaluate.py --task 08_adni_only         # test sur 1213 complet
    python ../evaluate.py --task 08_no_adni           # test sur 1213 complet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from trainers import run_training


def main() -> None:
    p = argparse.ArgumentParser(
        description="T8 — ADNI vs no-ADNI (généralisation cross-cohorte)"
    )
    here = Path(__file__).parent
    p.add_argument("--config", required=True,
                   help="Config à utiliser : config_adni_only.yaml ou config_no_adni.yaml")
    p.add_argument("--resume", default=None,
                   help="Forcer reprise depuis ce checkpoint")
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = here / args.config
    if not config_path.exists():
        print(f"[!] Config introuvable : {args.config}")
        print(f"    Disponibles dans {here}:")
        for f in sorted(here.glob("config_*.yaml")):
            print(f"      - {f.name}")
        sys.exit(1)

    run_training(config_path=str(config_path), resume_from=args.resume)


if __name__ == "__main__":
    main()