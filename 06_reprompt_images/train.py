"""
train.py — Tâche 6 : Reprompting image-focused.

Teste l'effet de différents prompts qui orientent le modèle vers les IRM.
Mesure aussi l'impact des features tabulaires (ablation noTab).

12 variantes possibles via --config :
    - 3 variantes de prompt :
        v1 = structures + symétries (descriptif)
        v2 = chain-of-thought clinique
        v3 = rôle de radiologue expert
    - 2 modes loss : a (sans MMSE) / b (avec MMSE)
    - 2 modes tabular : avec / sans features tabulaires (noTab)

Lancement :
    cd 06_reprompt_images/
    # Commencer par v1 selon les recommandations
    python train.py --config config_v1_a.yaml          # v1 sans MMSE, avec features
    python train.py --config config_v1_b.yaml          # v1 avec MMSE
    python train.py --config config_v1_a_noTab.yaml    # ablation : v1 image-only
    python train.py --config config_v1_b_noTab.yaml    # ablation : v1 image-only + MMSE
    # Puis v2 et v3 si v1 donne des résultats prometteurs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from trainers import run_training


def main() -> None:
    p = argparse.ArgumentParser(
        description="T6 — Reprompting image-focused (3 variantes × 4 modes)"
    )
    here = Path(__file__).parent
    p.add_argument("--config", required=True,
                   help="Config à utiliser. Ex: config_v1_a.yaml. "
                        "12 disponibles : config_v{1,2,3}_{a,b,a_noTab,b_noTab}.yaml")
    p.add_argument("--resume", default=None,
                   help="Forcer reprise depuis ce checkpoint")
    args = p.parse_args()

    # Résolution du chemin config (relatif au dossier du script si nécessaire)
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        # Tenter relatif au dossier du script
        config_path = here / args.config
    if not config_path.exists():
        print(f"[!] Config introuvable : {args.config}")
        print(f"    Configs disponibles dans {here}:")
        for f in sorted(here.glob("config_*.yaml")):
            print(f"      - {f.name}")
        sys.exit(1)

    run_training(config_path=str(config_path), resume_from=args.resume)


if __name__ == "__main__":
    main()