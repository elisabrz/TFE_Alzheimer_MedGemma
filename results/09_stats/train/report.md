# Analyse statistique multi-fold — 5 folds

**Folds analysés** : [0, 1, 2, 3, 4]  
**Split** : train  
**N sujets (par fold)** : 4366 
(CN=3418, AD=948, %AD=21.7)  
**N bootstrap** : 500

> ⚠️ Toutes les statistiques sont calculées sur les valeurs 
> **réellement mesurées** uniquement. Les valeurs imputées 
> (médiane train, T0) sont exclues.

> ⚠️ Un fold = un partitionnement train/val/test différent du même 
> ensemble de 6065 sujets. La variance observée reflète la 
> sensibilité de l'analyse au choix du split.

## 1. Table descriptive (fold de référence)

Continues : médiane [Q1, Q3] (n_real). Binaires : n positifs / n_real.

| cohort   |    n |   n_CN |   n_AD | pct_AD   | AGE                        | PTGENDER          | PTEDUCAT                   | PTMARRY           | CATANIMSC                  | TRAASCOR                   | TRABSCOR                    | DSPANFOR                 | DSPANBAC                | BNTTOTAL                   | BMI                        | VSWEIGHT                      | MH14ALCH        | MH16SMOK         | MH4CARD         | MH2NEURL        | mmse_score                |
|:---------|-----:|-------:|-------:|:---------|:---------------------------|:------------------|:---------------------------|:------------------|:---------------------------|:---------------------------|:----------------------------|:-------------------------|:------------------------|:---------------------------|:---------------------------|:------------------------------|:----------------|:-----------------|:----------------|:----------------|:--------------------------|
| GLOBAL   | 4366 |   3418 |    948 | 21.7%    | 69.0 [64.0, 74.0] (n=4364) | 1658/4366 (38.0%) | 16.0 [14.0, 18.0] (n=4345) | mode=1.0 (n=4340) | 20.0 [16.0, 25.0] (n=3875) | 31.0 [24.0, 42.0] (n=3798) | 77.0 [58.0, 114.0] (n=3751) | 8.0 [7.0, 10.0] (n=1026) | 6.0 [5.0, 8.0] (n=1024) | 29.0 [27.0, 30.0] (n=1036) | 27.2 [24.1, 31.0] (n=3603) | 158.0 [136.0, 177.0] (n=3319) | 187/4087 (4.6%) | 585/4081 (14.3%) | 375/4094 (9.2%) | 116/1635 (7.1%) | 27.5 [23.5, 29.0] (n=656) |
| ADNI     |  656 |    304 |    352 | 53.7%    | 72.9 [68.3, 78.4] (n=656)  | 333/656 (50.8%)   | 16.0 [14.0, 18.0] (n=656)  | mode=1.0 (n=656)  | 17.0 [13.0, 22.0] (n=655)  | 36.0 [28.0, 50.0] (n=653)  | 92.0 [65.0, 153.0] (n=641)  | 8.0 [7.0, 10.0] (n=385)  | 6.0 [5.0, 7.0] (n=384)  | 27.0 [23.0, 29.0] (n=395)  | 28.0 [23.8, 45.5] (n=159)  | 145.0 [105.3, 170.0] (n=572)  | 17/397 (4.3%)   | 152/397 (38.3%)  | 275/397 (69.3%) | 104/397 (26.2%) | 27.5 [23.5, 29.0] (n=656) |
| NACC     | 2968 |   2574 |    394 | 13.3%    | 68.0 [63.0, 73.0] (n=2966) | 1077/2968 (36.3%) | 16.0 [14.0, 18.0] (n=2963) | mode=1.0 (n=2943) | 21.0 [17.0, 25.0] (n=2968) | 30.0 [23.0, 40.0] (n=2893) | 74.0 [56.0, 107.0] (n=2866) | 9.0 [8.0, 11.0] (n=490)  | 7.0 [6.0, 9.0] (n=490)  | 29.0 [28.0, 30.0] (n=490)  | 27.2 [24.0, 30.8] (n=2704) | 159.0 [139.0, 178.0] (n=2178) | 131/2956 (4.4%) | 103/2959 (3.5%)  | 60/2958 (2.0%)  | 1/499 (0.2%)    | NA                        |
| OASIS    |  742 |    540 |    202 | 27.2%    | 70.1 [65.7, 75.1] (n=742)  | 248/742 (33.4%)   | 16.0 [14.0, 18.0] (n=726)  | mode=1.0 (n=741)  | 20.5 [16.8, 25.0] (n=252)  | 31.0 [24.0, 42.0] (n=252)  | 79.5 [60.0, 114.0] (n=244)  | 7.0 [6.0, 8.0] (n=151)   | 4.0 [4.0, 5.0] (n=150)  | 56.0 [50.0, 58.0] (n=151)  | 27.3 [24.3, 30.9] (n=740)  | 161.0 [141.0, 180.0] (n=569)  | 39/734 (5.3%)   | 330/725 (45.5%)  | 40/739 (5.4%)   | 11/739 (1.5%)   | NA                        |


## 2. Stabilité des tailles d'effet sur les folds

Pour chaque feature : mean ± std de l'effet de Mann-Whitney 
(rank-biserial) à travers les 5 folds.

| Feature | n_CN_real (μ) | n_AD_real (μ) | Effect (μ ± σ) | p_max | Sig. folds | Decay (μ ± σ) | Decay class |
|---|---|---|---|---|---|---|---|
| `mmse_score` | 314 | 344 | -0.952 ± 0.005 | 3.01e-96 | 5/5 | +0.000 ± 0.000 | ✓ intrinsèque |
| `TRABSCOR` | 2986 | 776 | +0.732 ± 0.010 | 5.12e-211 | 5/5 | -0.048 ± 0.007 | ~ ambigu |
| `CATANIMSC` | 3066 | 811 | -0.705 ± 0.009 | 4.77e-202 | 5/5 | -0.004 ± 0.006 | ✓ intrinsèque |
| `TRAASCOR` | 3015 | 791 | +0.638 ± 0.009 | 3.13e-163 | 5/5 | -0.028 ± 0.005 | ✓ intrinsèque |
| `DSPANBAC` | 681 | 341 | -0.447 ± 0.009 | 2.82e-31 | 5/5 | -0.008 ± 0.021 | ✓ intrinsèque |
| `BNTTOTAL` | 681 | 354 | -0.382 ± 0.018 | 6.97e-22 | 5/5 | -0.070 ± 0.008 | ✓ ambigu |
| `AGE` | 3416 | 947 | +0.329 ± 0.013 | 1.89e-50 | 5/5 | +0.100 ± 0.008 | ✓ ambigu |
| `DSPANFOR` | 681 | 345 | -0.309 ± 0.013 | 3.46e-14 | 5/5 | +0.048 ± 0.017 | ~ ambigu |
| `MH4CARD` | 3209 | 888 | +0.297 ± 0.006 | 7.78e-79 | 5/5 | +0.278 ± 0.008 | ✓ FORTUIT (Simpson) |
| `MH2NEURL` | 1151 | 494 | +0.247 ± 0.013 | 5.81e-21 | 5/5 | +0.222 ± 0.012 | ✓ FORTUIT (Simpson) |
| `PTEDUCAT` | 3400 | 945 | -0.170 ± 0.010 | 3.32e-14 | 5/5 | +0.025 ± 0.010 | ✓ intrinsèque |
| `MH16SMOK` | 3196 | 886 | +0.156 ± 0.012 | 2.07e-18 | 5/5 | +0.134 ± 0.009 | ~ FORTUIT (Simpson) |
| `PTMARRY` | 3397 | 944 | -0.150 ± 0.006 | 5.73e-16 | 5/5 | +0.006 ± 0.005 | ✓ intrinsèque |
| `PTGENDER` | 3418 | 948 | +0.117 ± 0.007 | 7.67e-12 | 5/5 | -0.001 ± 0.008 | ✓ intrinsèque |
| `VSWEIGHT` | 2587 | 754 | -0.109 ± 0.010 | 7.07e-05 | 5/5 | +0.039 ± 0.015 | ~ intrinsèque |
| `BMI` | 3009 | 603 | -0.103 ± 0.018 | 0.000637 | 5/5 | -0.018 ± 0.008 | ✓ intrinsèque |
| `MH14ALCH` | 3203 | 887 | +0.008 ± 0.007 | 0.896 | 0/5 | -0.007 ± 0.006 | ✓ intrinsèque |

## 3. Détection biais Simpson — stabilité multi-fold

- **FORTUIT stable** (toutes folds = FORTUIT) : 2 features
  - `MH4CARD` : decay = +0.278 ± 0.008
  - `MH2NEURL` : decay = +0.222 ± 0.012
- **FORTUIT instable** (FORTUIT majoritaire mais pas tous) : 1 features
  - `MH16SMOK` : FORTUIT=4/5 folds, decay = +0.134 ± 0.009

## 4. Figures

- `figures/multifold_decay_overview.pdf` — barplot decay moyen ± std sur folds
- `figures/correlation_matrix.png` — corrélations Spearman (valeurs réelles, pairwise)
- `figures/pca_features.png` — PCA 2D (imputation médiane pour viz uniquement)
- `figures/tsne_features.png` — t-SNE 2D (imputation médiane pour viz uniquement)
- `per_fold/fold_X/figures/` — distributions par feature pour chaque fold
