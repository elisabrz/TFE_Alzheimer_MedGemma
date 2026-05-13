# T0 — Préparation des splits

Génère les splits stratifiés 5-fold à partir du dataset poolé multi-cohorte.

## Objectif

Produire 5 paires `(train, val, test)` au format CSV pour validation croisée
au niveau **sujet** (jamais visite), avec :
- Stratification par classe **et** par cohorte
- Imputation médiane globale **train-only** appliquée aux 3 splits
- Marquage exhaustif des valeurs imputées via colonnes `<feature>_imputed`
- Reproductibilité totale (seed=42)

## Pipeline

```
all.csv (6065 sujets)
   ↓ filter_first_visit
6065 sujets unique
   ↓ StratifiedKFold(n=5, seed=42) sur label × source
fold_0/train + val + test  (4366 / 486 / 1213)
fold_1..4/...
   ↓ pour CHAQUE fold :
   |  ↓ calcul médiane sur train (par feature)
   |  ↓ imputation des 3 splits avec cette médiane
   |  ↓ ajout colonnes <feature>_imputed (0=réel, 1=imputé)
   ↓
data/splits/fold_0..4/{train,val,test}.csv
```

## Justification méthodologique

**Imputation train-only** : la médiane utilisée pour imputer val/test est
calculée **uniquement** sur train. Pas de fuite val→test ni test→val.
C'est plus rigoureux que la littérature standard (DEFT, Tanguy CBMS) qui
impute souvent chaque split avec sa propre médiane.

**Stratification par cohorte** : sans cela, les folds pourraient être
déséquilibrés (par exemple, un fold avec 80% NACC et un autre avec 20%).
Le déséquilibre de classe et de cohorte est désormais identique sur tous
les folds.

**Première visite uniquement** : un sujet peut avoir plusieurs visites
longitudinales (surtout ADNI). On garde uniquement la première pour éviter
les corrélations intra-sujet qui fausseraient l'évaluation.

## Lancer

```bash
cd 00_prepare_splits/
python prepare_splits.py
```

Output : `data/splits/fold_0..4/{train,val,test}.csv`

Durée : ~30 secondes (CPU pur, lecture + médianes + écritures CSV).

## Validation post-génération

```bash
python -c "
import pandas as pd
for fold in range(5):
    print(f'=== Fold {fold} ===')
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(f'data/splits/fold_{fold}/{split}.csv')
        ratio = (df.label==1).mean() * 100
        print(f'  {split}: n={len(df)}, AD={ratio:.1f}%')
"
```

Tous les folds doivent montrer ~21.7% AD sur train/val/test.

## Note sur fold_0

Pour le TFE, seul fold_0 est entraîné (contrainte GPU : ~24h par run).
Les folds 1-4 sont utilisés uniquement par T9 (analyse statistique multi-fold
sur les features cliniques, qui tourne en CPU).

## Fichiers

- `prepare_splits.py` : script principal
- `config.yaml` : paramètres (chemin source `all.csv`, n_folds, seed)
- `README.md` : ce fichier
