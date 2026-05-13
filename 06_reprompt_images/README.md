# T6 — Ablation du prompt (sélection du meilleur phrasing)

Étude de la sensibilité du modèle au **phrasing du prompt**. Question :
quel prompt extrait le mieux l'information visuelle de l'IRM ?

## Approche

Plusieurs variantes de prompt testées, toutes héritant de la même
configuration base que T1 (single-task classification, pas de tête MMSE) :

- **`full`** (baseline T1) : structure standard, section clinical_info en tête,
  puis IRM en bullets descriptifs.
- **`image_centric`** : restructuration mettant l'IRM au premier plan,
  clinical_info traitée comme contexte secondaire.
- **`image_focused`** (avec variants v1/v2/v3) : prompts explicites demandant
  au modèle d'analyser des structures cérébrales précises (hippocampe,
  ventricules, asymétries). Conçu pour guider l'attention visuelle.

## Variantes retenues pour le TFE

Par contrainte de temps GPU (chaque variante = ~24h d'entraînement) :

- **T6 image_centric** : entraîné, ~24h
- **T6 V1** (image_focused variant=v1) : entraîné, ~24h
- T6 V2, V3 : entrainé, ~24h

## Configuration

Chaque variante a son propre YAML :

```yaml
# 06_reprompt_images/T6_config_image_centric.yaml
task_name: 06_image_centric
inherits_from: ../config/config_base.yaml
prompt:
  mode: image_centric
mmse_head:
  enabled: false
training:
  output_dir: results/06_image_centric
```

```yaml
# 06_reprompt_v1_a/T6_config_v1_a.yaml
task_name: 06_if_v1
inherits_from: ../config/config_base.yaml
prompt:
  mode: image_focused
  variant: v1
mmse_head:
  enabled: false
```

## Lancer

```bash
# T6 image_centric
cd 06_reprompt_images/
python ../trainers.py --config T6_config_image_centric.yaml

# T6 V1
cd ../06_reprompt_v1_a/
python ../trainers.py --config T6_config_v1_a.yaml
```

## Résultats (fold 0, seed 42, test n=1213, seuil 0.5)

| Variante | AUC | Sens | Spec | F1 |
|---|---|---|---|---|
| T6 image_centric | **0.948** | 0.890 | 0.856 | 0.738 |
| T6 V1 image_focused | 0.945 | 0.856 | **0.869** | 0.735 |
| (rappel T1 full) | 0.945 | 0.897 | 0.844 | 0.730 |

**Observations** :
- Tous les prompts convergent vers AUC ≈ 0.945-0.948 quand les features
  sont disponibles → les features tabulaires dominent le signal.
- L'image_centric maximise très légèrement l'AUC (0.948 vs 0.945).
- L'image_focused V1 favorise la spécificité (0.869) au détriment de la
  sensibilité (0.856).

## Ablation image-only — la vraie question scientifique

C'est en mode **image-only** (sans features tabulaires) que les différences
deviennent claires : quel prompt permet d'extraire le plus d'information
visuelle de l'IRM seule ?

```bash
# Pour chaque variante
python evaluate.py \
    --checkpoint results/06_image_centric/best_model \
    --config 06_reprompt_images/T6_config_image_centric_noTab.yaml \
    --split test \
    --output results/ablation/06_image_centric_noTab \
    --threshold-source val

python evaluate.py \
    --checkpoint results/06_if_v1/best_model \
    --config 06_reprompt_v1_a/T6_config_v1_a_noTab.yaml \
    --split test \
    --output results/ablation/06_if_v1_noTab \
    --threshold-source val
```

### Résultats image-only (test n=1213)

| Variante | AUC img-only | Sens | Spec | Δ AUC vs complet |
|---|---|---|---|---|
| T1 full | 0.856 | 0.738 | 0.794 | −0.089 |
| T2 full | 0.851 | 0.730 | 0.816 | −0.095 |
| T6 image_centric | 0.860 | 0.795 | 0.725 | −0.088 |
| **T6 V1 image_focused** | **0.866** | **0.844** | 0.728 | **−0.079** |

**Conclusion** : T6 V1 est le **meilleur prompt pour l'extraction visuelle pure**.
En guidant explicitement l'analyse vers les structures cérébrales pertinentes
(hippocampe, ventricules), le prompt aide le modèle à mieux utiliser l'IRM
quand les features cliniques sont absentes.

Le Δ AUC plus petit pour T6 V1 (−0.079 vs ~−0.09 pour les autres) confirme
qu'il **dépend moins des features** que les autres prompts.

## Avec features, tous convergent

Avec les 16 features présentes, tous les prompts donnent AUC ≈ 0.945-0.948.
Ce résultat suggère que :
- Les features tabulaires **saturent le signal disponible**
- Le choix du prompt n'a d'impact significatif que dans le régime image-only
- En déploiement avec données cliniques complètes, le choix du prompt
  est secondaire

## Fichiers

- `T6_config_image_centric.yaml` : variante image_centric
- `T6_config_v1_a.yaml` : variante image_focused v1
- `T6_config_*_noTab.yaml` : configs ablation image-only correspondantes
- `T6_config_base.yaml` : tronc commun pour les variantes image_focused
- README.md : ce fichier
