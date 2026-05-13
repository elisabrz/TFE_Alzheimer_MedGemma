#!/bin/bash
# eval_prompts_zeroshot.sh — Sélection du prompt optimal pour T6.
#
# Compare 5 variantes de prompt sur le checkpoint T1 (modèle constant).
# Mesure l'effet causal du prompt seul, à poids de modèle identiques.
#
# IMPORTANT — calibration du seuil :
# Chaque prompt change la distribution des probas de sortie. Le seuil Youden
# sauvé dans le checkpoint T1 a été calibré pour le prompt 'full' uniquement,
# donc l'utiliser tel quel pour 'image_centric' / 'image_focused' produirait
# une fausse comparaison (Sens=1.0, Spec=0.0 vu sur la run précédente).
#
# Solution : --threshold-source val recalcule Youden sur SON propre split val
# pour CHAQUE prompt. Comparaison équitable.
#
# Durée : ~7 min/prompt × 5 prompts × 2 (val pour calibrage si split=test) = ~25-50 min
#
# Usage :
#   bash eval_prompts_zeroshot.sh           # val uniquement (recommandé)
#   bash eval_prompts_zeroshot.sh test      # val + test (après sélection validée)
#
# Sortie :
#   results/01_no_mmse/prompt_selection/
#     ├── full/val_results/
#     ├── image_centric/val_results/
#     ├── if_v1/val_results/
#     ├── if_v2/val_results/
#     ├── if_v3/val_results/
#     └── summary.csv     ← tableau comparatif trié par AUC
set -e

TASK="${TASK:-01_no_mmse}"
SPLIT_LIST="val"
[ "$1" = "test" ] && SPLIT_LIST="val test"

BASE_OUT="results/${TASK}/prompt_selection"
mkdir -p "$BASE_OUT"

echo "=========================================================="
echo "  Sélection du prompt optimal — checkpoint ${TASK}"
echo "  Référence : T1 full prompt"
echo "  Variantes : image_centric | v1 | v2 | v3"
echo "  Splits : ${SPLIT_LIST}"
echo "  Calibration seuil : Youden recalculé sur val (pour CHAQUE prompt)"
echo "  Output : ${BASE_OUT}/"
echo "=========================================================="

# Format : "id|mode|variant"
PROMPTS=(
    "full|full|"
    "image_centric|image_centric|"
    "if_v1|image_focused|v1"
    "if_v2|image_focused|v2"
    "if_v3|image_focused|v3"
)

for split in $SPLIT_LIST; do
    echo
    echo "─── Split: ${split} ───"

    for entry in "${PROMPTS[@]}"; do
        IFS="|" read -r pid pmode pvariant <<< "$entry"
        out_dir="${BASE_OUT}/${pid}/${split}_results"
        mkdir -p "$out_dir"

        label="${pid}"
        [ -n "$pvariant" ] && label="${pid} (variant=${pvariant})"

        echo
        echo "▶ ${label}"

        cmd=(python evaluate.py
             --task "$TASK"
             --split "$split"
             --output "$out_dir"
             --prompt-mode "$pmode"
             --threshold-source val)
        [ -n "$pvariant" ] && cmd+=(--prompt-variant "$pvariant")

        if "${cmd[@]}" 2>&1 | tee "${out_dir}/eval.log"; then
            echo "   ✓ done → ${out_dir}"
        else
            echo "   ✗ FAILED — on continue"
        fi
    done
done

# ── Récap final ─────────────────────────────────────────────────────────
echo
echo "=========================================================="
echo "  RÉCAPITULATIF"
echo "=========================================================="

python << 'EOF'
import json
from pathlib import Path
import pandas as pd

base = Path("results/01_no_mmse/prompt_selection")

PROMPT_LABELS = {
    "full":          "T1 — full (baseline)",
    "image_centric": "T6 — image_centric",
    "if_v1":         "T6 — image_focused v1 (structures)",
    "if_v2":         "T6 — image_focused v2 (CoT)",
    "if_v3":         "T6 — image_focused v3 (radiologue)",
}

rows = []
for pid, label in PROMPT_LABELS.items():
    for split_dir in sorted((base / pid).glob("*_results")):
        split_name = split_dir.name.replace("_results", "")
        mfile = split_dir / f"metrics_{split_name}.json"
        if not mfile.exists():
            continue
        with open(mfile) as f:
            m = json.load(f)
        # Helper : remplace None par NaN pour formatter proprement
        def _safe(v):
            return v if v is not None else float("nan")
        rows.append({
            "prompt":        label,
            "split":         split_name,
            "n":             m.get("n_samples", "?"),
            "AUC":           _safe(m.get("auc")),
            "thr":           _safe(m.get("threshold_val_optimal")),
            "F1 (Youden)":   _safe(m.get("f1_val_thr")),
            "Sens (Youden)": _safe(m.get("sensitivity_val_thr")),
            "Spec (Youden)": _safe(m.get("specificity_val_thr")),
            "F1 (0.5)":      _safe(m.get("f1")),
        })

if not rows:
    print("[!] Aucun résultat trouvé. Vérifier que evaluate.py a bien tourné.")
else:
    df = pd.DataFrame(rows).sort_values(["split", "AUC"], ascending=[True, False])
    pd.set_option("display.max_colwidth", 35)

    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}" if not pd.isna(x) else "—"
        return str(x)

    print(df.to_string(index=False, float_format=_fmt))

    out_csv = base / "summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Résumé sauvé : {out_csv}")

    print()
    print("─" * 60)
    print("  PROCHAINE ÉTAPE")
    print("─" * 60)

    val_df = df[df["split"] == "val"].sort_values("AUC", ascending=False)
    if not val_df.empty:
        best = val_df.iloc[0]
        baseline_auc = val_df[val_df["prompt"] == PROMPT_LABELS["full"]]
        baseline_auc = float(baseline_auc.iloc[0]["AUC"]) if len(baseline_auc) else None

        print(f"  Meilleur prompt (AUC val) : {best['prompt']}")
        sens_str = f"{best['Sens (Youden)']:.4f}" if not pd.isna(best['Sens (Youden)']) else "—"
        spec_str = f"{best['Spec (Youden)']:.4f}" if not pd.isna(best['Spec (Youden)']) else "—"
        thr_str  = f"{best['thr']:.4f}" if not pd.isna(best['thr']) else "—"
        print(f"    AUC={best['AUC']:.4f} | Sens={sens_str} | Spec={spec_str}")
        print(f"    Seuil Youden recalculé sur val : {thr_str}")
        print()
        if baseline_auc is not None and best['prompt'] != PROMPT_LABELS["full"]:
            delta = float(best['AUC']) - baseline_auc
            print(f"  Δ AUC vs full baseline : {delta:+.4f}")
            if abs(delta) > 0.01:
                print(f"  ⇒ Variation significative — lancer training complet :")
                print(f"     cd 06_<winner> && python train.py")
            else:
                print(f"  ⇒ Variation < 0.01 → robustesse au phrasing confirmée.")
                print(f"     Garder T1 comme référence, documenter en thèse.")
        elif best['prompt'] == PROMPT_LABELS["full"]:
            print(f"  ⇒ Le prompt 'full' reste la meilleure baseline.")
            print(f"     Robustesse de T1 au phrasing confirmée.")
    print("─" * 60)
EOF

echo
echo "=========================================================="
echo "  ✓ SÉLECTION TERMINÉE"
echo "=========================================================="