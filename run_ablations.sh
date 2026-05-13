#!/bin/bash
# run_ablations.sh — Étude d'ablation complète : texte-only et vision-only
#
# Runs : 4 modèles × 2 types d'ablation × 2 splits = 16 évaluations
# Durée : ~10 min par run × 16 = ~2h40 total
#
# Prérequis :
#   - dataset.py patché (use_visual flag)
#   - evaluate.py à jour (dual-threshold)
#   - Les 4 checkpoints présents dans results/
#
# Usage :
#   cd /home/elisa/TFE_final
#   nohup bash run_ablations.sh > logs/ablations_$(date +%Y%m%d_%H%M).log 2>&1 &
#   echo $! > ablations.pid
#
# Options :
#   bash run_ablations.sh --notab-only    # uniquement ablations text
#   bash run_ablations.sh --novis-only    # uniquement ablations vision
#   bash run_ablations.sh --val-only      # uniquement sur val
#   bash run_ablations.sh --test-only     # uniquement sur test
# ─────────────────────────────────────────────────────────────────────────

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/ablations_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── Flags ────────────────────────────────────────────────────────────────
DO_NOTAB=true
DO_NOVIS=true
DO_VAL=true
DO_TEST=true

for arg in "$@"; do
    case $arg in
        --notab-only) DO_NOVIS=false ;;
        --novis-only) DO_NOTAB=false ;;
        --val-only)   DO_TEST=false ;;
        --test-only)  DO_VAL=false ;;
        *) echo "[!] Argument inconnu : $arg" ;;
    esac
done

# ── Fonction utilitaire ───────────────────────────────────────────────────
run_eval() {
    local label="$1"   # ex: T1_noTab_test
    local checkpoint="$2"
    local config="$3"
    local split="$4"
    local output="$5"
    local logfile="$LOG_DIR/${label}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ▶ $label"
    echo "  ⏱  $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"

    local start=$(date +%s)
    if python evaluate.py \
        --checkpoint "$checkpoint" \
        --config "$config" \
        --split "$split" \
        --output "$output" \
        --threshold-source val \
        > "$logfile" 2>&1; then
        local elapsed=$(( $(date +%s) - start ))
        echo "  ✓ terminé en $((elapsed/60))min $((elapsed%60))s"
        # Extrait AUC/Sens/Spec du log pour récap
        grep -E "AUC|Sens|Spec" "$logfile" | grep -v "INFO" | tail -6
    else
        local elapsed=$(( $(date +%s) - start ))
        echo "  ✗ ÉCHEC après $((elapsed/60))min — voir $logfile"
    fi
}

# ── Définition des 4 modèles ─────────────────────────────────────────────
declare -A CHECKPOINTS
CHECKPOINTS["T1"]="results/01_no_mmse/best_model"
CHECKPOINTS["T2"]="results/02_with_mmse/best_model"
CHECKPOINTS["T6_ic"]="results/06_image_centric/best_model"
CHECKPOINTS["T6_V1"]="results/06_if_v1/best_model"
CHECKPOINTS["T6_V2"]="results/06_if_v2/best_model"
CHECKPOINTS["T6_V3"]="results/06_if_v3/best_model"

declare -A CONFIGS_NOTAB
CONFIGS_NOTAB["T1"]="01_train_no_mmse/T1_config_noTab.yaml"
CONFIGS_NOTAB["T2"]="02_train_with_mmse/T2_config_noTab.yaml"
CONFIGS_NOTAB["T6_ic"]="06_reprompt_images/T6_ic_config_noTab.yaml"
CONFIGS_NOTAB["T6_V1"]="06_reprompt_images/T6_V1_config_noTab.yaml"
CONFIGS_NOTAB["T6_V2"]="06_reprompt_images/T6_V2_config_noTab.yaml"
CONFIGS_NOTAB["T6_V3"]="06_reprompt_images/T6_V3_config_noTab.yaml"

declare -A CONFIGS_NOVIS
CONFIGS_NOVIS["T1"]="01_train_no_mmse/T1_config_noVis.yaml"
CONFIGS_NOVIS["T2"]="02_train_with_mmse/T2_config_noVis.yaml"
CONFIGS_NOVIS["T6_ic"]="06_reprompt_images/T6_ic_config_noVis.yaml"
CONFIGS_NOVIS["T6_V1"]="06_reprompt_images/T6_V1_config_noVis.yaml"
CONFIGS_NOVIS["T6_V2"]="06_reprompt_images/T6_V2_config_noVis.yaml"
CONFIGS_NOVIS["T6_V3"]="06_reprompt_images/T6_V3_config_noVis.yaml"

declare -A TASK_IDS
TASK_IDS["T1"]="01_no_mmse"
TASK_IDS["T2"]="02_with_mmse"
TASK_IDS["T6_ic"]="06_image_centric"
TASK_IDS["T6_V1"]="06_if_v1"
TASK_IDS["T6_V2"]="06_if_v2"
TASK_IDS["T6_V3"]="06_if_v3"

GLOBAL_START=$(date +%s)
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ABLATION STUDY — Texte-only et Vision-only                      ║"
echo "║  $(date '+%Y-%m-%d %H:%M:%S')                                    ║"
echo "║  Logs : $LOG_DIR                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

MODELS=("T1" "T2" "T6_ic" "T6_V1" "T6_V2" "T6_V3")

# ── ABLATION TEXTE (use_tabular=false, use_visual=true) ──────────────────
# Le modèle ne voit que les IRM → teste la robustesse sans données cliniques
if [ "$DO_NOTAB" = true ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  ABLATION TEXT-ONLY (IRM uniquement, pas de features cliniques)"
    echo "══════════════════════════════════════════════════════════════════"
    for model in "${MODELS[@]}"; do
        ckpt="${CHECKPOINTS[$model]}"
        cfg="${CONFIGS_NOTAB[$model]}"
        task="${TASK_IDS[$model]}"

        if [ ! -d "$ckpt" ]; then
            echo "  ⚠ $model : checkpoint introuvable ($ckpt) — skip"
            continue
        fi

        if [ "$DO_VAL" = true ]; then
            run_eval "${model}_noTab_val" "$ckpt" "$cfg" "val" \
                "results/ablation/${task}_noTab/val_results"
        fi
        if [ "$DO_TEST" = true ]; then
            run_eval "${model}_noTab_test" "$ckpt" "$cfg" "test" \
                "results/ablation/${task}_noTab/test_results"
        fi
    done
fi

# ── ABLATION VISION (use_tabular=true, use_visual=false) ─────────────────
# Le modèle ne voit que le texte → teste la robustesse sans images IRM
if [ "$DO_NOVIS" = true ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  ABLATION VISION-ONLY (texte clinique, IRM = zéros)"
    echo "══════════════════════════════════════════════════════════════════"
    for model in "${MODELS[@]}"; do
        ckpt="${CHECKPOINTS[$model]}"
        cfg="${CONFIGS_NOVIS[$model]}"
        task="${TASK_IDS[$model]}"

        if [ ! -d "$ckpt" ]; then
            echo "  ⚠ $model : checkpoint introuvable ($ckpt) — skip"
            continue
        fi

        if [ "$DO_VAL" = true ]; then
            run_eval "${model}_noVis_val" "$ckpt" "$cfg" "val" \
                "results/ablation/${task}_noVis/val_results"
        fi
        if [ "$DO_TEST" = true ]; then
            run_eval "${model}_noVis_test" "$ckpt" "$cfg" "test" \
                "results/ablation/${task}_noVis/test_results"
        fi
    done
fi

# ── Récapitulatif ─────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - GLOBAL_START ))
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ABLATIONS TERMINÉES                                             ║"
printf "║  Durée totale : %-49s║\n" "$((TOTAL/60))min $((TOTAL%60))s"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "AUC récap (val set) :"
echo "  Modèle       │ complet │ noTab │ noVis"
for model in "${MODELS[@]}"; do
    task="${TASK_IDS[$model]}"
    auc_full=$(grep -h "AUC" "results/${task}/test_results/cohort_metrics_test.csv" 2>/dev/null \
               | grep "GLOBAL" | cut -d',' -f6 | head -1 | xargs printf "%.3f" 2>/dev/null || echo "  n/a")
    auc_notab=$(grep -h "AUC" "results/ablation/${task}_noTab/test_results/cohort_metrics_test.csv" 2>/dev/null \
               | grep "GLOBAL" | cut -d',' -f6 | head -1 | xargs printf "%.3f" 2>/dev/null || echo "  n/a")
    auc_novis=$(grep -h "AUC" "results/ablation/${task}_noVis/test_results/cohort_metrics_test.csv" 2>/dev/null \
               | grep "GLOBAL" | cut -d',' -f6 | head -1 | xargs printf "%.3f" 2>/dev/null || echo "  n/a")
    printf "  %-12s │ %7s │ %5s │ %5s\n" "$model" "$auc_full" "$auc_notab" "$auc_novis"
done
