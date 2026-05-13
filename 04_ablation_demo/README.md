# T4 — ABANDONNÉ (remplacé par ablations zero-shot)

⚠️ **Cette tâche n'est plus active dans le pipeline final.**

## Plan initial

T4 devait être une ablation cumulant **neuropsycho + démographie** (sans
poids/BMI ni comorbidités), pour isoler la contribution des features
anthropométriques.

## Pourquoi abandonné

Comme T3, la même question peut être répondue **en zero-shot** sur le
checkpoint T1 ou T2 en modifiant simplement la liste `tabular_features`
dans une config dédiée.

```bash
python evaluate.py \
    --task 01_no_mmse \
    --config 01_no_mmse/config_noDemo.yaml \
    --split test \
    --output results/ablation/01_noDemo \
    --threshold-source val
```

Avec :

```yaml
data:
  tabular_features:
    # AGE, PTGENDER, PTEDUCAT, PTMARRY retirées
    - CATANIMSC
    - TRAASCOR
    - TRABSCOR
    - DSPANFOR
    - DSPANBAC
    - BNTTOTAL
    - MH14ALCH
    - MH16SMOK
    - MH4CARD
    - MH2NEURL
    - VSWEIGHT
    - BMI
```

## Voir aussi

- `T3_README.md` : justification générale de l'abandon des ablations
  par ré-entraînement au profit du zero-shot
- `explainability/feature_importance.py` : ranking d'importance par
  feature individuelle (méthode plus fine que les ablations groupées)
