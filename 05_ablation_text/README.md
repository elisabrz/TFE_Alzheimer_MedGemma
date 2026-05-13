# T5 — ABANDONNÉ (remplacé par ablations zero-shot)

⚠️ **Cette tâche n'est plus active dans le pipeline final.**

## Plan initial

T5 devait être une ablation **texte minimal** : ré-entraînement avec un
prompt drastiquement réduit (uniquement l'instruction de classification,
aucune feature). Objectif : tester la performance du modèle sur les IRM
quasiment seules.

## Pourquoi abandonné

L'ablation **image-only zero-shot** (déjà réalisée sur T1, T2, T6 image_centric
et T6 V1) répond exactement à cette question, à moindre coût :

```bash
python evaluate.py \
    --task 01_no_mmse \
    --config 01_no_mmse/config_noTab.yaml \
    --split test \
    --output results/ablation/01_noTab \
    --threshold-source val
```

Avec `use_tabular: false`, le `_format_clinical_info` retourne simplement
`"No clinical data provided."` — le prompt ne contient plus que l'instruction
et les coupes IRM.

## Résultats déjà obtenus (image-only zero-shot, test n=1213)

| Modèle | AUC img-only | AUC complet | Δ |
|---|---|---|---|
| T1 | 0.856 | 0.945 | −0.089 |
| T2 | 0.851 | 0.946 | −0.095 |
| T6 image_centric | 0.860 | 0.948 | −0.088 |
| T6 V1 image_focused | **0.866** | 0.945 | **−0.079** |

**Observation clé** : T6 V1 (prompt image_focused) est le meilleur extracteur
d'information visuelle pure (AUC=0.866 sans aucune feature). Cela suggère
que le phrasing du prompt influence l'attention visuelle du modèle même
sans contenu textuel additionnel.

## Voir aussi

- `T6_README.md` : étude détaillée des prompts (image_centric vs image_focused)
- `06_reprompt_images/T6_config_v1_a_noTab.yaml` : config ablation image-only
  pour T6 V1
- Tableau récapitulatif dans `README.md` racine
