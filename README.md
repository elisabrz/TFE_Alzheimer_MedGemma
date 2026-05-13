# TFE — Détection Automatique de la Maladie d'Alzheimer

Fine-tuning de **MedGemma 1.5 4B** en **QLoRA NF4 4-bit** pour la classification
binaire CN/AD et la régression MMSE conjointe, à partir d'IRM cérébrales 3D
(4 coupes 2D 448×448) et de 16 features cliniques tabulaires.

**Auteur** : Elisa Bourez — TFE Master Ingénieur Civil, FPMs / UMons
**Données** : 3 cohortes poolées (ADNI + NACC + OASIS, 6065 sujets)
**Hardware cible** : RTX 4080 16 GB

---

## Architecture

- **Backbone** : `google/medgemma-1.5-4b-it` (MedSigLIP + Gemma 3, ~4.34B paramètres)
- **QLoRA** : NF4 double-quant, compute dtype bfloat16, r=16, α=32, dropout=0.05
- **Classification** : verbalizer sur tokens `CN` (id=9353) / `AD` (id=2466)
- **Régression MMSE** : tête auxiliaire `LayerNorm(2560) + Linear(2560→1)` × sigmoid × 30
- **Loss** : Focal(α_AD = 0.783, γ = 2.0) + α_reg × MSE pondérée sum/sum
- **Paramètres entraînables** : 38.5M (0.89% du modèle, adaptateurs LoRA uniquement)

## Méthodologie clé

**Imputation rigoureuse** : médiane globale **train-only** appliquée aux 3 splits.
Pas de fuite val/test. Colonnes `<feature>_imputed` propagées pour traçabilité.

**Omission des features imputées du prompt** (`signal_in_prompt: false`).
Décision méthodologique : *"on n'approxime pas ce qu'on ne sait pas."* Le modèle
voit uniquement les features réellement mesurées. Apport vs littérature (DEFT, FLIQA-AD).

**MMSE multitâche** : régression numérique via tête dédiée (pas génération de
tokens textuels). Justification : les LLM traitent les nombres comme tokens
discrets sans relation de proximité — Cross-Entropy pénaliserait identiquement
"prédire 25 au lieu de 24" et "prédire 5 au lieu de 24".

**Poids w_reg = 0 pour les sujets sans MMSE réel** (NACC, OASIS).
Cohérent avec l'omission des features imputées : ces sujets ne contribuent
ni au prompt ni à la loss MMSE. Normalisation sum/sum :
`L_reg = Σ(w_i · MSE_i) / max(1, Σw_i)` — généralisation de DEFT au contexte
multi-cohorte avec MMSE rares.

## Protocole splits

5-fold stratified CV au niveau **sujet** (pas de leakage), première visite uniquement.
Par contrainte GPU (TFE), on entraîne uniquement sur **fold 0, seed 42**
(les 5 folds sont générés pour reproductibilité future et T9).

Découpage fold 0 : **72% train / 8% val / 20% test** (4366 / 486 / 1213 sujets).

## Structure du projet

```
TFE_final/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances pinnées
├── .env.example                 # Template variables d'environnement
│
├── utils.py                     # Fonctions partagées (load_config, EvalCallback, FocalLoss, MMSEHead, eval, plots)
├── dataset.py                   # TfeDataset + collate (vues IRM, features tabulaires en texte)
├── trainers.py                  # TfeMedGemmaCls, TfeMedGemmaWithMMSE, TfeMultitaskTrainer
├── trainers_encoder.py          # T2-c : tête MMSE branchée sur vision_tower
├── evaluate.py                  # Évaluation standalone (dual-threshold 0.5 + Youden val)
├── inspect_prompts.py           # Inspection des prompts générés (debug)
│
├── config/
│   └── config_base.yaml         # Hyperparamètres communs à toutes les tâches
│
├── data/
│   ├── README.md                # Format attendu des splits + description des 16 features
│   └── splits/                  # Générés par 00_prepare_splits/
│       ├── fold_0/{train,val,test}.csv
│       └── fold_1..4/...        # Générés mais utilisés uniquement par T9
│
├── 00_prepare_splits/           # T0 : génération des splits 5-fold + imputation
├── 01_train_no_mmse/            # T1 : baseline classification CN/AD
├── 02_train_with_mmse/          # T2 : + tête MMSE (multitâche)
│   ├── T2b_config_alpha05.yaml  #   T2-b : sensibilité α_reg = 0.5
│   └── T2c_config_encoder.yaml  #   T2-c : extraction encodeur
│
├── 06_reprompt_images/          # T6 : ablation du prompt (image_centric, image_focused)
├── 07_test_hospital/            # T7 : validation externe hôpital (données en attente)
├── 08_adni_vs_no_adni/          # T8 : impact cohorte ADNI
├── 09_statistical_analysis/     # T9 : analyse stats multi-fold (CPU)
├── 10_zero_shot_medgemma/       # T10 : MedGemma vanilla (baseline VLM zero-shot)
│
├── explainability/              # 4 méthodes : feature_importance, gradcam, attention_rollout, occlusion
│
└── results/                     # Sorties de chaque tâche (checkpoints, métriques, figures)
```

## Ablations text-only / image-only / suppression sélective

Les tâches T3, T4, T5 du plan initial ont été **abandonnées au profit d'ablations
d'inférence** sur les checkpoints existants (T1, T2, T6 image_centric, T6 V1).
Ces ablations ne nécessitent pas de ré-entraînement : on évalue le checkpoint
avec une config modifiée (`use_tabular: false` ou `tabular_features: []`).

Cette approche est :
- **Plus rapide** : 40 min par ablation au lieu de 24h d'entraînement
- **Plus rigoureuse scientifiquement** : on teste la dépendance d'un modèle
  déjà entraîné aux différentes modalités, pas l'impact de l'absence d'une
  modalité pendant le training (deux questions différentes)

Types d'ablations effectuées :
- **Image-only** : prompt sans aucune feature tabulaire (toutes les 16 omises)
- **Text-only** (à venir) : suppression des images, conservation du prompt textuel
- **Suppression sélective** (à venir) : retrait d'un sous-ensemble de features
  (ex: tests neuropsychologiques uniquement)

## Installation

```bash
cd TFE_final/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Renseigner HF_TOKEN (MedGemma est gated) et WANDB_API_KEY
cp .env.example .env
```

## Workflow type

```bash
# 1. Générer les splits (une seule fois)
cd 00_prepare_splits && python prepare_splits.py && cd ..

# 2. Lancer un entraînement
cd 01_train_no_mmse && python train.py && cd ..

# 3. Évaluer un checkpoint
python evaluate.py --task 01_no_mmse --split test --threshold-source checkpoint

# 4. Ablation image-only (zero-shot sur checkpoint existant)
python evaluate.py --task 01_no_mmse --config 01_no_mmse/config_noTab.yaml \
    --split test --output results/ablation/01_noTab --threshold-source val
```

## Ordre d'exécution

1. **T0** (splits) → indispensable avant tout
2. **T1, T2** → baselines classification (single-task et multitâche)
3. **T6** → identification du meilleur prompt
4. **Ablations** → zero-shot sur T1, T2, T6 (image-only, text-only)
5. **T2-b, T2-c** → étude d'ablation sur la régression MMSE
6. **T10** → baseline MedGemma vanilla zero-shot
7. **T8, T9** → en parallèle CPU
8. **T7** → validation externe hôpital (en attente données)

## Résultats actuels (fold 0, seed 42, split test n=1213)

| Modèle | AUC | Sens (0.5) | Spec (0.5) | F1 (0.5) | MAE MMSE |
|---|---|---|---|---|---|
| T1 (full prompt) | 0.945 | 0.897 | 0.844 | 0.730 | — |
| T2 (multitâche) | 0.946 | 0.916 | 0.817 | 0.711 | 2.49† |
| T6 image_centric | 0.948 | 0.890 | 0.856 | 0.738 | — |
| T6 V1 image_focused | 0.945 | 0.856 | 0.869 | 0.735 | — |

† Valeur initiale calculée avec bug d'échelle, à re-évaluer après fix.

**Ablations image-only** (zero-shot sur checkpoints existants) :

| Modèle | AUC img-only | Δ AUC vs full |
|---|---|---|
| T1 image-only | 0.856 | −0.089 |
| T2 image-only | 0.851 | −0.095 |
| T6 image_centric image-only | 0.860 | −0.088 |
| **T6 V1 image-only** | **0.866** | **−0.079** |

T6 V1 est le meilleur prompt pour l'extraction visuelle pure.

## Contraintes techniques à ne jamais oublier

- `MMSEHead` : LayerNorm **obligatoire** (pas Linear seul → crash au chargement)
- Hook MMSE **sans `.detach()`** (le gradient MSE doit remonter vers PEFT)
- `safe_serialization=False` (Gemma 3 partage `embed` et `lm_head`)
- `eval_strategy="no"` + `EvalCallback` custom (pipeline HF cassé avec `labels=None` pour ImageTextToText)
- Convention autoregressive : `logit_pos = ans_pos - 1`
- `batch_size=1` est la limite physique sur RTX 4080 16 GB
- `gradient_checkpointing_kwargs={"use_reentrant": False}` (compat PEFT 4-bit)
- `save_strategy: "no"` recommandé (crash bitsandbytes NF4 après save_steps)
- Échelle MMSE : target normalisée `/30` au chargement, dénormalisée `×30` à l'évaluation
