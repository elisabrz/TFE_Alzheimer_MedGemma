# T3 — ABANDONNÉ (remplacé par ablations zero-shot)

⚠️ **Cette tâche n'est plus active dans le pipeline final.**

## Plan initial

T3 devait être une ablation des **6 features neuropsychologiques** :
CATANIMSC, TRAASCOR, TRABSCOR, DSPANFOR, DSPANBAC, BNTTOTAL — soit un
ré-entraînement complet avec un sous-ensemble réduit de features.

## Pourquoi abandonné

Le coût d'un entraînement complet (~24h sur RTX 4080 16GB) ne se justifiait
plus une fois qu'on a réalisé que **la même question scientifique** peut
être répondue de manière plus rigoureuse en **zero-shot** sur le checkpoint
existant T1 ou T2 :

```bash
# Évaluer T1 ou T2 sans les features neuropsychologiques
python evaluate.py \
    --task 01_no_mmse \
    --config 01_no_mmse/config_noNeuropsych.yaml \
    --split test \
    --output results/ablation/01_noNeuropsych \
    --threshold-source val
```

Avec la config qui retire ces features de la liste `tabular_features` :

```yaml
data:
  tabular_features:
    - AGE
    - PTGENDER
    - PTEDUCAT
    - PTMARRY
    # CATANIMSC, TRAASCOR, TRABSCOR, DSPANFOR, DSPANBAC, BNTTOTAL retirées
    - MH14ALCH
    - MH16SMOK
    - MH4CARD
    - MH2NEURL
    - VSWEIGHT
    - BMI
```

## Question scientifique différente

**T3 original (ré-entraînement)** : *"un modèle entraîné sans neuropsycho
performe-t-il bien ?"*

**Ablation zero-shot** : *"le modèle entraîné avec neuropsycho dépend-il
fortement de ces features pour ses prédictions ?"*

Ces deux questions sont distinctes. Pour le TFE, la seconde (zero-shot)
est :
- Plus rapide (10 min vs 24h)
- Plus cohérente avec une étude d'explicabilité (on teste la dépendance
  d'un modèle déployé aux modalités)
- Comparable directement au modèle de référence sans biais d'entraînement

## Voir aussi

- `explainability/feature_importance.py` : analyse plus fine, par feature
  individuelle, sur les checkpoints existants
- `01_no_mmse/config_noNeuropsych.yaml` : config ablation neuropsycho zero-shot
