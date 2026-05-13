# T10 — Baseline zero-shot MedGemma vanilla

Évaluation de **MedGemma 1.5 4B sans fine-tuning** sur le test set du TFE.
Mesure la performance "out-of-the-box" du modèle médical pré-entraîné,
pour quantifier le gain réel apporté par le QLoRA fine-tuning.

## Objectif

Répondre à la question : *"De combien le fine-tuning améliore-t-il la
performance par rapport à MedGemma utilisé tel quel ?"*

MedGemma a été pré-entraîné sur un large corpus médical incluant probablement
de la littérature sur la maladie d'Alzheimer. Il dispose donc d'un certain
"prior" clinique. Le fine-tuning QLoRA spécialise ce prior aux patterns
spécifiques de notre pool ADNI+NACC+OASIS.

Si T10 atteint déjà AUC > 0.8 zero-shot, le gain du fine-tuning est marginal
(~0.10 d'AUC). Si T10 plafonne à ~0.6, le fine-tuning apporte beaucoup.

## Pipeline

Pas d'entraînement. Inférence directe avec le même prompt que T1 et collecte
des logits CN/AD via le verbalizer. Pas de tête MMSE.

```bash
# Évaluation zero-shot sur val (pour calibrer un seuil)
python T10_zero_shot.py \
    --config T10_config.yaml \
    --split val \
    --output results/10_zero_shot/val_results

# Évaluation zero-shot sur test (pour les chiffres finaux)
python T10_zero_shot.py \
    --config T10_config.yaml \
    --split test \
    --output results/10_zero_shot/test_results
```

Durée : ~30 min par split (1213 patients × 1.5 sec/inférence).

## Configuration

```yaml
# T10_config.yaml
task_name: 10_zero_shot
inherits_from: config/config_base.yaml

# Pas de LoRA, pas de tête MMSE — MedGemma brut
zero_shot: true

# Même prompt que T1 pour comparabilité
prompt:
  mode: full

mmse_head:
  enabled: false

training:
  output_dir: results/10_zero_shot
```

## Résultats attendus

Si MedGemma a un bon prior clinique : AUC val/test entre 0.70 et 0.85.
Si le prior est faible : AUC autour de 0.60-0.65 (légèrement mieux que random).

**Métrique critique** : Sens vs Spec. MedGemma vanilla a tendance à être
"timide" (prédire CN par défaut), donc on peut s'attendre à Sens basse et
Spec élevée. Le seuil Youden val permet de re-calibrer.

## Lecture pour la thèse

Format de présentation dans le tableau de résultats :

| Modèle | AUC test |
|---|---|
| MedGemma zero-shot (T10) | 0.XX |
| T1 (fine-tuned) | 0.945 |
| **Gain absolu** | **+0.YY** |

C'est cette dernière ligne qui justifie le coût de calcul du fine-tuning
QLoRA (~24h par modèle). Un gain de +0.10 AUC est considéré comme très
significatif en pratique clinique.

## Status actuel

🟢 **Peut être lancé à tout moment**, rapide (~30 min/split), pas de
dépendance sur les autres tâches.

## Fichiers

- `T10_zero_shot.py` : script d'inférence zero-shot
- `T10_config.yaml` : config (utilise le prompt T1 standard)
- `README.md` : ce fichier
