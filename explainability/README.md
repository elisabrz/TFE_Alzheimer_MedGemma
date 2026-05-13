# Explicabilité — 4 méthodes complémentaires

Suite d'analyses post-hoc sur les 4 modèles entraînés (T1, T2, T6 image_centric,
T6 V1) pour comprendre **ce que le modèle a appris** et **où il regarde**.

## Méthodes

| Script | Méthode | Cible | Coût |
|---|---|---|---|
| `feature_importance.py` | Perturbation + LOCO | Importance des 16 features tabulaires | 30 min/modèle |
| `gradcam_mri.py` | Grad-CAM sur SigLIP | Régions IRM activées | 10 min/modèle |
| `attention_rollout.py` | Attention rollout sur LM | Tokens prompt regardés | 5 min/modèle |
| `occlusion_mri.py` | Occlusion 3D patches | Régions IRM essentielles | 1-3h/modèle |

`run_all.py` orchestre les 4 méthodes en séquence.

## Lancer

```bash
cd /home/elisa/TFE_final
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Tout en un (4 méthodes pour un modèle)
python explainability/run_all.py \
    --task 02_with_mmse \
    --split test \
    --n-patients 20 \
    --strategy stratified \
    --predictions-csv results/02_with_mmse/test_results/predictions_test.csv \
    --filter-real-only \
    --skip-occlusion          # occlusion à part car très lente
```

## Argument crucial : `--filter-real-only`

**Restreint la sélection aux patients avec ≥50% de features réelles.**

Cascade de détection :
1. Colonnes `<feature>_imputed` dans le CSV (méthode préférentielle)
2. `has_real_measures` (proxy MMSE)
3. Fallback : tous gardés

**Pourquoi c'est critique** :
- **Feature importance** : si un patient NACC a CATANIMSC=médiane imputée,
  remplacer par la médiane = ΔAUC nul artificiel → biais "feature pas
  importante" alors qu'elle est juste imputée.
- **Grad-CAM** : permet de croiser l'activation visuelle avec les vrais
  scores cliniques du patient (MMSE réel, scores cognitifs réels) pour
  une validation cohérente — impossible si tout est imputé.

Sur test n=1213 : avec `--filter-real-only`, on garde ~196 patients ADNI
(les seuls avec features cognitives réelles). C'est normal et voulu.

## 1. Feature importance

**Deux sous-méthodes** dans le même script :

### Perturbation globale

Pour chaque feature F, on remplace sa valeur par la médiane train **pour
tous les patients sélectionnés**, on refait l'inférence, on calcule
`ΔAUC = AUC_orig - AUC_perturb`.

ΔAUC élevé → feature très importante pour les prédictions globales.
ΔAUC ≈ 0 → feature ignorée par le modèle.

### LOCO (Leave-One-Out per patient)

Pour chaque patient × feature, on retire la feature du prompt et on mesure
`|Δprob_AD|`. Donne une carte d'importance individualisée :

> *"Pour ce patient précis, retirer son score TMT-B a fait varier P(AD)
> de 0.18 → décision sensible à cette feature."*

Plus interprétable cliniquement que la perturbation globale.

### Output

```
results/explain/<task>/feature_importance/
├── perturbation_ranking.png        # Bar chart des 16 features triées
├── perturbation_ranking.csv
├── loco_per_patient.csv             # Importance par patient × feature
└── summary.json
```

## 2. Grad-CAM sur les IRM

**Backward uniquement dans `vision_tower` SigLIP**, pas dans le LM entier.

Pourquoi : faire le backward à travers les 27 couches Gemma 3 + le projecteur
dépassait les 16 GB VRAM (OOM crashes répétés). La solution architecturale :

1. Forward complet **no_grad** → P(AD) pour annotation
2. Forward **vision_tower seul avec grad** → score = norme L2 du token CLS
3. Backward dans SigLIP uniquement → ~3-4 GB VRAM

C'est scientifiquement valide : on explique **ce que le module vision
trouve important**, ce qui est exactement la question.

### Score proxy : norme du CLS

Pour le backward, on ne peut pas utiliser le logit CN/AD (qui nécessite
le LM). On utilise comme score proxy la norme L2 du CLS token de la
dernière couche SigLIP — c'est l'agrégation visuelle globale apprise par
le modèle. Plus le CLS a une "énergie" élevée, plus le modèle a extrait
de patterns visuels saillants.

### Les 4 vues

Le modèle reçoit 4 coupes par patient :
- 2 **coronales** (région hippocampe)
- 2 **axiales** (région ventricules)

Grad-CAM produit une heatmap par vue, superposée à la coupe brute.

### Output

```
results/explain/<task>/gradcam/
├── patient_<id>/
│   ├── coronal_1.png      # Vue brute
│   ├── coronal_1_cam.png  # Vue + heatmap
│   ├── coronal_2.png
│   ├── axial_1.png
│   ├── axial_2.png
│   └── mosaic.png         # Composite 4 vues
└── summary.csv
```

### Interprétation attendue

Sur les vrais AD bien classés (P(AD) > 0.7) :
- Activation hippocampe (coronal) → atrophie hippocampique = biomarqueur AD #1
- Activation ventricules latéraux (axial) → élargissement = signe AD secondaire

Sur les vrais CN bien classés :
- Activation diffuse ou faible → pas de pattern AD spécifique

Sur les erreurs (FN ou FP) :
- Activations atypiques → cas instructifs pour le manuscrit

## 3. Attention rollout (LM)

Trace l'attention agrégée sur les couches du décodeur LM, pour identifier
**quels tokens du prompt le modèle regarde** au moment de la décision.

```bash
python explainability/attention_rollout.py \
    --task 02_with_mmse --split test \
    --n-patients 50 --strategy stratified \
    --filter-real-only
```

### Output

Heatmaps token × layer : pour chaque patient, on voit quels tokens du
prompt (Age, CATANIMSC, TRABSCOR, etc.) reçoivent le plus d'attention
des têtes du LM.

Permet de répondre : *"Le modèle utilise-t-il vraiment les scores
neuropsychologiques mentionnés dans le prompt, ou ignore-t-il certains
tokens ?"*

## 4. Occlusion 3D (méthode la plus lente)

Pour chaque patient sélectionné, on occlut (met à zéro) un cube 3D de
l'IRM par sliding window, et on mesure la chute de P(AD). On obtient
une carte 3D des régions essentielles à la prédiction.

```bash
python explainability/occlusion_mri.py \
    --task 02_with_mmse --split test \
    --n-patients 5 --strategy stratified \
    --filter-real-only
```

⚠️ **Très lent** : 196 forward passes par patient. Limité à 5 patients
pour des raisons pratiques (sinon 3+ heures). Réservé aux figures
illustratives du manuscrit (1-2 cas représentatifs).

## Stratégies de sélection des patients

Toutes les méthodes utilisent `select_patients` avec :

- `--strategy random` : tirage aléatoire stratifié CN/AD
- `--strategy stratified` : équilibre CN/AD × cohorte
- `--strategy tp_fn_mix` : nécessite `predictions_csv`, sélectionne :
  * True Positive (AD prédits AD, prob > 0.7)
  * False Negative (AD prédits CN, prob < 0.3)
  * True Negative (CN prédits CN, prob < 0.3)
  * Hard cases (prob ∈ [0.4, 0.6])

Pour le manuscrit, `stratified` + `--filter-real-only` est généralement le
meilleur choix : représentativité + interprétabilité.

## Fichiers

- `_common.py` : fonctions partagées (chargement modèle, sélection patients,
  `get_real_mask`, `select_patients`)
- `feature_importance.py` : perturbation + LOCO
- `gradcam_mri.py` : Grad-CAM vision encoder (OOM-safe)
- `attention_rollout.py` : attention rollout LM
- `occlusion_mri.py` : occlusion 3D (lent)
- `heatmaps_occlusion.py` : variante d'occlusion (plots)
- `run_all.py` : orchestre les 4 méthodes
- `README.md` : ce fichier

## Référence pour le manuscrit

Suggestion de présentation dans la section "Résultats — Explicabilité" :

1. **Feature importance** : barplot des 16 features triées par perturbation ΔAUC.
   Mettre en avant les 5 features dominantes et discuter leur cohérence
   avec la littérature clinique AD.

2. **Grad-CAM** : 3 figures choisies à la main :
   - 1 vrai positif AD avec activation hippocampe nette
   - 1 vrai négatif CN avec activation diffuse/faible
   - 1 erreur instructive (FN ou FP)

3. **Attention rollout** : un seul exemple de patient AD, montrant l'attention
   sur les tokens CATANIMSC / TRABSCOR / DSPANBAC dans le prompt.

4. **Occlusion** : optionnelle, comme validation visuelle d'un cas Grad-CAM.
