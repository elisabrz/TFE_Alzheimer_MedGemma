# T2 — Classification + régression MMSE (multitâche)

Extension de T1 avec une **tête de régression MMSE** auxiliaire. Le modèle
apprend simultanément la classification CN/AD et l'estimation du score
MMSE (Mini-Mental State Examination, 0-30).

## Architecture

- Backbone : identique à T1 (MedGemma 1.5 4B + LoRA)
- **Tête MMSE** ajoutée : `LayerNorm(2560) + Linear(2560→1)` en fp32, sigmoid × 30
- Hook sur la dernière couche du décodeur LM à la position `logit_pos = ans_pos - 1`
- **MMSE exclu du prompt** (cible de régression → fuite si présent)
- Loss combinée : `L_total = L_cls(focal) + α_reg × L_reg(MSE)`

## Pourquoi une tête de régression et pas de la génération de texte

À l'inverse de FLIQA-AD (Chen 2025) qui génère le MMSE comme tokens textuels,
ce TFE utilise une **tête de régression numérique**. Justification :
- Les LLM traitent les nombres comme tokens discrets sans relation de proximité
- Cross-Entropy pénaliserait identiquement "prédire 25 au lieu de 24" et
  "prédire 5 au lieu de 24" — pas de distance mathématique
- MSE traite le score sur une échelle continue, ce qui est cliniquement correct

Approche similaire à DEFT-VLM-AD (Cheng 2025), avec une adaptation au contexte
multi-cohorte (cf. section *Loss et normalisation*).

## Loss et normalisation

```python
loss_cls = FocalLoss(α_AD=0.783, γ=2.0)(logits_CN_AD, y_cls)
loss_reg = Σ(w_i · (ŷ_i - y_i)²) / max(1, Σw_i)        # sum/sum
loss_total = loss_cls + α_reg × loss_reg                 # α_reg = 1.0
```

**Pourquoi sum/sum plutôt que `.mean()`** :

DEFT utilise `L = (1/N) Σ MSE_i` avec **N = batch size**. Dans leur cas
(ADNI seul, 100% MMSE réels), `.mean()` ≡ `sum/sum` car tous les `w_i = 1`.

Dans le contexte multi-cohorte de ce TFE, **85% des sujets n'ont pas de
MMSE réel** (NACC, OASIS). Pour ces sujets, `w_reg = 0` (cohérent avec la
décision méthodologique d'usage exclusif des données réelles).

Diviser par N diluerait le gradient MMSE par ~5-7× selon le mix du batch.
`sum/sum` garantit que chaque mesure réelle exerce une contribution
complète et constante :

$$L_{reg} = \frac{\sum_{i=1}^{B} w_i \cdot (\hat{y}_i - y_i)^2}{\max(1, \sum_{i=1}^{B} w_i)}$$

C'est une **généralisation de DEFT** au contexte multi-cohorte avec MMSE rares,
pas un départ de leur approche.

## Lancer

```bash
cd 02_train_with_mmse/
python train.py
```

Durée : ~24h sur RTX 4080 16GB.

## Configuration

```yaml
# 02_train_with_mmse/config.yaml
task_name: 02_with_mmse
inherits_from: ../config/config_base.yaml
mmse_head:
  enabled: true
  loss_weight: 1.0          # α_reg
training:
  output_dir: results/02_with_mmse
```

## Résultats (fold 0, seed 42)

### Validation (n=486)

| Seuil | AUC | Acc | F1 | Sens | Spec | MAE MMSE | CC | R² |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.951 | 0.838 | 0.711 | 0.916 | 0.817 | — | — | — |
| Youden=0.798 | 0.951 | 0.905 | 0.798 | 0.858 | 0.918 | 3.18† | 0.715† | 0.146† |

### Test (n=1213, seuil 0.5)

| Métrique | Valeur |
|---|---|
| AUC | 0.946 |
| Sens | 0.916 |
| Spec | 0.817 |
| F1 | 0.711 |
| MAE MMSE (n_real=196 ADNI) | 2.49† |
| CC | 0.670 |
| R² | 0.330† |

† Valeurs initialement affectées par un bug d'échelle évaluation (mmse_pred
en [0,30] vs mmse_true resté en [0,1]). CC reste valide car invariant à
l'échelle linéaire. MAE/RMSE/R² ont été re-calculés après correction.

### Par cohorte (test, seuil 0.5)

| Cohorte | n | AUC | Sens | Spec |
|---|---|---|---|---|
| ADNI | 196 | 0.925 | **0.990** | 0.608 |
| NACC | 798 | 0.952 | 0.870 | 0.908 |
| OASIS | 219 | 0.859 | 0.875 | 0.535 |

Note : Sens=0.990 sur ADNI = le modèle détecte presque tous les AD, mais
la spécificité est faible (0.608) → biais "AD-permissif" sur la cohorte où
le MMSE et les scores neuropsychologiques sont réels et corrélés.

## Étude d'ablation sur la régression MMSE

Trois variantes pour mesurer la sensibilité du multitâche :

| Run | α_reg | Niveau extraction | Description |
|---|---|---|---|
| **T2** (baseline) | 1.0 | décodeur | Configuration principale |
| **T2-b** | 0.5 | décodeur | Sensibilité de la pondération |
| **T2-c** | 1.0 | encodeur (SigLIP) | Encodeur vs décodeur |

### T2-b : test α_reg = 0.5

```bash
python train.py --config 02_with_mmse/T2b_config_alpha05.yaml
```

Hypothèse : avec α_reg=1.0, le modèle peut sacrifier un peu la classification
pour mieux régresser le MMSE (compétition de gradient). α=0.5 ramène le
MMSE à un rôle de "guide" plutôt que d'objectif principal.

### T2-c : extraction au niveau encodeur

```bash
python T2c_train_encoder.py --config 02_with_mmse/T2c_config_encoder.yaml
```

La tête MMSE lit le **CLS de SigLIP** (vision_tower) au lieu du hidden state
LM. Architecture séparée dans `trainers_encoder.py` (ne touche pas
`trainers.py`). Test direct de l'hypothèse :

> *"Le MMSE est-il dérivable de l'IRM seule (encodeur), ou nécessite-t-il
> la fusion image+texte (décodeur, T2 baseline) ?"*

Si MAE_encodeur >> MAE_decodeur, ça valide que les features tabulaires
(âge, éducation, scores cognitifs) sont indispensables à la régression MMSE.

## Évaluer

```bash
# T2 standard
python evaluate.py --task 02_with_mmse --split test --threshold-source checkpoint

# Ablation image-only
python evaluate.py --task 02_with_mmse \
    --config 02_with_mmse/config_noTab.yaml \
    --split test --output results/ablation/02_noTab \
    --threshold-source val
```

## Ablation image-only (test)

| Métrique | T2 standard | T2 image-only | Δ |
|---|---|---|---|
| AUC | 0.946 | 0.851 | −0.095 |
| Sens | 0.916 | 0.730 | −0.19 |
| Spec | 0.817 | 0.816 | 0.00 |
| MAE MMSE | 2.49† | 3.47 | +0.98 |
| CC MMSE | 0.670 | 0.445 | −0.22 |

Sans features tabulaires, la régression MMSE devient nettement moins
performante. Confirme que les scores cognitifs textuels portent l'essentiel
du signal de régression — le modèle ne peut pas inférer le MMSE de l'IRM
seule de manière fiable.

## Fichiers

- `train.py` : entraînement T2 standard
- `config.yaml` : config T2 (α_reg=1.0)
- `config_noTab.yaml` : ablation image-only
- `T2b_config_alpha05.yaml` : variante α_reg=0.5
- `T2c_config_encoder.yaml` : variante extraction encodeur
- `README.md` : ce fichier
