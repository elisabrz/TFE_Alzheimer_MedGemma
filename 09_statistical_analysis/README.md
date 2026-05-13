# T9 — Analyse statistique multi-fold

Étude statistique de la **stabilité des effets** des 16 features cliniques
sur les 5 folds générés par T0. Tourne en **CPU** uniquement (pas de GPU
requis), peut être lancé en parallèle d'entraînements GPU.

## Objectif

Pour chaque feature et chaque fold, calculer :
- Distribution CN vs AD (médiane, IQR, test de Mann-Whitney U)
- Taille d'effet (Cohen's d ou rank-biserial pour catégoriels)
- p-value avec correction multi-comparaisons (Bonferroni ou Benjamini-Hochberg)

Puis **agréger sur les 5 folds** pour identifier :
- Features **stables** : significatives sur tous les folds (effet réel)
- Features **instables** : significatives sur certains folds seulement
  (effet possiblement dû au hasard du split)

## Pourquoi multi-fold

Une analyse statistique sur un seul fold (le fold 0 utilisé pour l'entraînement
TFE) peut donner des résultats biaisés par le split aléatoire. La répétition
sur 5 folds permet d'estimer la variance de l'effet et de distinguer signal
réel du bruit d'échantillonnage.

## Lancer

```bash
cd 09_statistical_analysis/
python analyze.py --folds 0 1 2 3 4 --split train --output results/09_stats/train
python analyze.py --folds 0 1 2 3 4 --split val --output results/09_stats/val
```

Tourne en CPU (~5-10 min pour les 5 folds × 16 features).

## Outputs

```
results/09_stats/
├── per_fold/
│   ├── fold_0_features.csv      # Stats par feature pour fold 0
│   ├── fold_1_features.csv
│   └── ...
├── aggregated/
│   ├── feature_summary.csv      # Moyenne ± std des effects sur 5 folds
│   └── stability_ranking.csv    # Tri par stabilité
└── figures/
    ├── feature_effects_<feature>.png   # Boxplot CN vs AD multi-fold
    └── stability_heatmap.png            # Vue d'ensemble
```

## Format `feature_summary.csv`

```
feature, effect_mean, effect_std, n_significant_folds, stability_class
CATANIMSC, -1.8, 0.15, 5, stable
TRABSCOR, 1.4, 0.22, 5, stable
AGE, 0.6, 0.18, 4, mostly_stable
PTGENDER, 0.05, 0.10, 1, unstable
...
```

## Lecture des résultats

**Features attendues comme stables** (signal AD bien établi cliniquement) :
- `CATANIMSC` : déficit de fluence verbale très précoce dans la MA
- `TRABSCOR` (TMT-B) : déficit exécutif progressif
- `DSPANBAC` : déficit de mémoire de travail
- `BNTTOTAL` : aphasie sémantique tardive
- `AGE` : facteur de risque démographique majeur

**Features attendues comme instables** :
- `PTGENDER`, `PTMARRY`, `MH*` (comorbidités) : effets variables selon
  l'échantillonnage des cohortes

## Utilité pour la thèse

Ces résultats statistiques fournissent une **validation indépendante** de
l'importance des features identifiées par `feature_importance.py` (méthode
de perturbation sur le modèle entraîné). Si les deux approches convergent
(ex: CATANIMSC est important dans les deux analyses), c'est un argument
fort pour l'interprétation clinique.

Si elles divergent (une feature statistiquement non-significative mais
importante pour le modèle, ou inversement), c'est un sujet de discussion
méthodologique riche pour la thèse.

## Status actuel

✅ **Peut tourner à tout moment en arrière-plan CPU.** Pas de dépendance
sur les autres tâches.

## Fichiers

- `analyze.py` : script principal
- `config.yaml` : paramètres (test statistique, correction multi-comparaisons)
- `README.md` : ce fichier
