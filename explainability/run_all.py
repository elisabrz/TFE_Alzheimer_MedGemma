"""
run_all.py — Lance les 4 méthodes d'explainability séquentiellement.

Évite de relancer 4 fois le chargement du modèle MedGemma (gain ~30s × 3
+ pas d'OOM entre runs). Tout reste isolé dans des sous-dossiers.

Lancement :
    python run_all.py --task 02_with_mmse --n-patients 100 --strategy stratified
    python run_all.py --task 01_no_mmse --skip-occlusion  # skip la plus lente
    python run_all.py --task 02_with_mmse --strategy tp_fn_mix \\
        --predictions-csv results/02_with_mmse/best_model/test_results/predictions_test.csv

Note : pour le batch sur PLUSIEURS modèles, écrire un wrapper bash :
    for task in 01_no_mmse 02_with_mmse 03_ablation_neuro; do
        python explainability/run_all.py --task $task --n-patients 100
    done
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent


METHODS = [
    ("Grad-CAM",          "gradcam_mri.py",         "skip_gradcam"),
    ("Occlusion",         "occlusion_mri.py",       "skip_occlusion"),
    ("Feature importance", "feature_importance.py", "skip_features"),
    ("Attention rollout", "attention_rollout.py",   "skip_attention"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lance les 4 méthodes explainability séquentiellement"
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-patients", type=int, default=20)
    parser.add_argument("--strategy", default="stratified",
                        choices=["random", "stratified", "tp_fn_mix"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip-gradcam",   action="store_true")
    parser.add_argument("--skip-occlusion", action="store_true")
    parser.add_argument("--skip-features",  action="store_true")
    parser.add_argument("--skip-attention", action="store_true")

    parser.add_argument("--patch-size", type=int, default=32,
                        help="Pour occlusion (défaut 32px)")
    parser.add_argument("--n-bootstrap", type=int, default=100,
                        help="Pour feature_importance (défaut 100)")

    args = parser.parse_args()

    # Args de base communs à tous les scripts
    common_args = [
        "--task", args.task,
        "--n-patients", str(args.n_patients),
        "--strategy", args.strategy,
        "--split", args.split,
        "--fold", str(args.fold),
        "--seed", str(args.seed),
    ]
    if args.config:
        common_args.extend(["--config", args.config])
    if args.checkpoint:
        common_args.extend(["--checkpoint", args.checkpoint])
    if args.predictions_csv:
        common_args.extend(["--predictions-csv", args.predictions_csv])

    print(f"\n{'#'*70}")
    print(f"#  EXPLAINABILITY — {args.task}")
    print(f"#  {len(METHODS)} méthodes : Grad-CAM → Occlusion → Features → Rollout")
    print(f"{'#'*70}\n")

    overall_start = time.time()
    errors = 0

    for method_name, script, skip_attr in METHODS:
        if getattr(args, skip_attr):
            print(f"\n>> {method_name} : SKIPPED (--{skip_attr.replace('_', '-')})")
            continue

        script_path = THIS_DIR / script
        cmd = [sys.executable, str(script_path)] + common_args

        # Args spécifiques par méthode
        if script == "occlusion_mri.py":
            cmd.extend(["--patch-size", str(args.patch_size)])
        elif script == "feature_importance.py":
            cmd.extend(["--n-bootstrap", str(args.n_bootstrap)])

        print(f"\n{'='*70}")
        print(f"  >> Lancement : {method_name}")
        print(f"  >> Commande  : {' '.join(cmd)}")
        print(f"{'='*70}\n")

        t0 = time.time()
        try:
            result = subprocess.run(cmd, check=False)
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"\n[!] {method_name} a échoué (code {result.returncode}) "
                      f"après {elapsed/60:.1f} min")
                errors += 1
            else:
                print(f"\n[✓] {method_name} terminé en {elapsed/60:.1f} min")
        except KeyboardInterrupt:
            print(f"\n[!] Interruption clavier — arrêt après {method_name}")
            break
        except Exception as e:
            print(f"\n[!] {method_name} crash : {e}")
            errors += 1

    total_min = (time.time() - overall_start) / 60
    print(f"\n{'#'*70}")
    print(f"#  EXPLAINABILITY TERMINÉ")
    print(f"#  Total       : {total_min:.1f} min")
    print(f"#  Erreurs     : {errors}/{len(METHODS)}")
    print(f"{'#'*70}\n")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())