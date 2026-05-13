# T8 — Impact de la cohorte ADNI (étude d'ablation cohorte)

à confirmer pas réalisé pour l'instant : Étude de l'impact de la cohorte **ADNI** sur la performance globale du modèle.
ADNI étant la cohorte la mieux caractérisée (scores cognitifs réels complets,
MMSE disponible), elle pourrait dominer le signal d'apprentissage de manière
disproportionnée par rapport à son poids numérique (903/6065 = 15%).

## Hypothèse

Deux scénarios à comparer :
1. **Avec ADNI** (T1/T2 standards) : training sur tout le pool 6065 sujets
2. **Sans ADNI** : training sur NACC + OASIS uniquement (5162 sujets)

Si la performance test (sur les trois cohortes) chute drastiquement sans
ADNI, le modèle apprenait surtout des patterns ADNI. Si la performance
reste stable, le modèle généralise bien à partir des features partielles
NACC/OASIS.

## Variantes

```yaml
# T8 base : T1 standard (avec ADNI) — déjà entraîné
inherits_from: 01_no_mmse/config.yaml

# T8 ablation : same pipeline mais train sans ADNI
inherits_from: ../config/config_base.yaml
data:
  cohort_filter: ["NACC", "OASIS"]   # exclut ADNI au chargement
training:
  output_dir: results/08_no_adni
```

## Lancer

```bash
# La version "avec ADNI" est déjà entraînée (= T1 baseline)
# Donc seul le run "sans ADNI" est nécessaire

cd 08_adni_vs_no_adni/
python train_no_adni.py
```

Durée : ~24h pour le run sans ADNI.

## Évaluation

Le critère clé est la performance **sur le test set complet** (incluant ADNI)
avec un modèle entraîné sans ADNI. Cela mesure la capacité de généralisation
du modèle non-ADNI aux patients ADNI.

```bash
# Test sur ensemble du fold 0 (ADNI + NACC + OASIS)
python evaluate.py \
    --checkpoint results/08_no_adni/best_model \
    --config 08_adni_vs_no_adni/config_no_adni.yaml \
    --split test \
    --output results/08_no_adni/test_results \
    --threshold-source checkpoint
```

## Résultats attendus

Si l'hypothèse "ADNI domine" est vraie :
- Performance ADNI : chute significative (-0.05 à -0.10 AUC)
- Performance NACC : peu impactée
- Performance OASIS : potentiellement meilleure (moins de bruit imputation
  vu que les features réelles dominantes ADNI n'orientent plus l'apprentissage)

Si l'hypothèse est fausse :
- Performance globalement stable, le modèle généralise bien à partir des
  features partielles NACC/OASIS

## Status actuel

🟡 **À lancer si temps GPU disponible.** Priorité plus basse que T2-b et T2-c
(étude d'ablation MMSE) car la question d'impact cohorte est plus annexe
dans la problématique TFE.

## Fichiers

- `train_no_adni.py` : script d'entraînement avec exclusion ADNI
- `config_no_adni.yaml` : config NACC + OASIS uniquement
- `README.md` : ce fichier
