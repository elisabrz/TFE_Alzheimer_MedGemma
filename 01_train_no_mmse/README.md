# T1 — Baseline classification CN/AD (sans MMSE)

Premier modèle entraîné : fine-tuning QLoRA de MedGemma 1.5 4B pour la
classification binaire **CN (cognitively normal) vs AD (Alzheimer's disease)**
à partir des IRM + 16 features cliniques tabulaires en texte.

## Architecture

- Backbone : `google/medgemma-1.5-4b-it`
- QLoRA : NF4, r=16, α=32, target_modules=all-linear, ~38.5M params entraînables
- Classification : verbalizer CN/AD, focal loss (α_AD=0.783, γ=2.0)
- **Pas de tête MMSE** (single-task)
- Prompt mode : `full` (template standard avec section clinical_info)

## Lancer

```bash
cd 01_train_no_mmse/
python train.py
```

Durée : ~24h sur RTX 4080 16GB (5 epochs max, early stopping patience=2).

## Configuration

```yaml
# 01_train_no_mmse/config.yaml
task_name: 01_no_mmse
inherits_from: ../config/config_base.yaml
mmse_head:
  enabled: false
training:
  output_dir: results/01_no_mmse
```

## Évaluer

```bash
# Test set, seuil checkpoint (Youden val calibré pendant training)
python evaluate.py --task 01_no_mmse --split test --threshold-source checkpoint

# Outputs dans results/01_no_mmse/test_results/
#   metrics_test.json
#   predictions_test.csv
#   cohort_metrics_test.csv  (les deux seuils 0.5 + Youden val)
#   roc_curve_test.png
#   confusion_matrix_test_<cohort>_{thr05,thrval}.png
```

## Résultats (fold 0, seed 42)

### Validation (n=486)

| Seuil | AUC | Acc | F1 | Sens | Spec |
|---|---|---|---|---|---|
| 0.5 | 0.946 | 0.834 | 0.704 | 0.870 | 0.824 |
| Youden=0.835 | 0.946 | 0.907 | 0.796 | 0.830 | 0.929 |

### Test (n=1213)

| Seuil | AUC | Acc | F1 | Sens | Spec |
|---|---|---|---|---|---|
| 0.5 | 0.945 | 0.856 | 0.730 | 0.897 | 0.844 |

### Par cohorte (test, seuil 0.5)

| Cohorte | n | AUC | Sens | Spec |
|---|---|---|---|---|
| ADNI | 196 | 0.915 | 0.960 | 0.660 |
| NACC | 798 | 0.951 | 0.850 | 0.921 |
| OASIS | 219 | 0.867 | 0.875 | 0.613 |

Le modèle généralise bien sur NACC (la cohorte la plus grande) avec
spécificité élevée, mais reste plus permissif sur ADNI (Spec=0.66) — biais
vers la prédiction AD probablement lié aux scores cognitifs réels visibles
dans le prompt qui orientent fortement la décision.

## Ablation image-only (zero-shot)

Évaluation du checkpoint T1 avec `use_tabular: false` (toutes les 16 features
omises) :

```bash
python evaluate.py --task 01_no_mmse \
    --config 01_no_mmse/config_noTab.yaml \
    --split test \
    --output results/ablation/01_noTab \
    --threshold-source val
```

**Résultats test image-only** : AUC=0.856, Sens=0.738, Spec=0.794 (Δ AUC = −0.089)

→ Quantifie la contribution causale des 16 features tabulaires sur le
modèle entraîné. Les features apportent ~0.09 d'AUC.

## Fichiers

- `train.py` : entraînement
- `config.yaml` : config standard T1
- `config_noTab.yaml` : config ablation image-only
- `README.md` : ce fichier
