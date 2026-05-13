# T7 — Validation externe sur données hospitalières

⏳ **En attente des données** (rétention par l'hôpital partenaire).

## Objectif

Évaluer le modèle final (T1 ou T2, à décider) sur un **jeu de données
externe complètement non vu pendant le training**, issu d'un hôpital
partenaire. Mesure de généralisation hors des cohortes de recherche
standardisées (ADNI, NACC, OASIS).

## Pipeline prévu

Inference-only sur le checkpoint best_model existant. Aucun ré-entraînement.

```bash
python 07_test_hospital/test_hospital.py \
    --checkpoint results/02_with_mmse/best_model \
    --hospital-csv data/hospital/hospital_test.csv \
    --output results/07_hospital/
```

## Pourquoi c'est important

ADNI, NACC, OASIS sont des cohortes **de recherche** :
- Protocoles d'acquisition IRM standardisés
- Examens neuropsychologiques rigoureux
- Patients sélectionnés (critères d'inclusion stricts)

Un hôpital représente la **réalité clinique** :
- Acquisitions IRM hétérogènes (machines différentes, protocoles variés)
- Données cliniques souvent incomplètes (moins de scores cognitifs détaillés)
- Population plus large (comorbidités, démence d'autres étiologies)

Une dégradation de performance importante (>0.1 d'AUC) entre ADNI/NACC/OASIS
et l'hôpital serait un signal fort que le modèle a surappris les
caractéristiques des cohortes de recherche.

## Format attendu des données

CSV minimal :

```
subject_id, scan_path, label, mmse_score (optionnel),
AGE, PTGENDER, PTEDUCAT, ..., BMI  (16 colonnes features)
```

Les valeurs manquantes doivent être marquées par `NaN` ou colonne vide —
le pipeline d'imputation **train-only** s'appliquera : médiane calculée
sur le train ADNI/NACC/OASIS, appliquée à l'hôpital. Cohérent avec
la décision méthodologique d'omission des features imputées au prompt
(`signal_in_prompt: false`).

## Précautions méthodologiques

- **Pas de re-fit de l'imputation sur l'hôpital** : utiliser exclusivement
  les médianes train calculées par T0. Toute autre approche serait du leakage.
- **Calibration du seuil** : utiliser le seuil Youden val calibré pendant
  T1/T2 training, **pas un seuil recalculé sur l'hôpital**. Recalculer
  sur l'hôpital donnerait des métriques optimistes biaisées.
- **Analyse par cohorte hospitalière** : si l'hôpital fournit plusieurs
  sources (services différents), stratifier les métriques.

## Status actuel

🟡 **Bloqué côté livraison données.** Si les données arrivent à temps :
- 20 min d'inférence sur ~200 patients
- Génère mêmes outputs que `evaluate.py` standard
- À intégrer dans la section "Validation externe" du manuscrit

Si les données n'arrivent pas avant la deadline TFE, mentionner cette
validation comme **perspective immédiate** dans la conclusion du manuscrit.

## Fichiers

- `test_hospital.py` : script d'évaluation (à finaliser quand format CSV connu)
- `config.yaml` : config d'inférence
- `README.md` : ce fichier
