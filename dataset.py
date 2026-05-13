"""
dataset.py — Datasets multimodaux pour MedGemma (IRM 3D + 16 features cliniques).

Refonte unifiée vs l'ancien pipeline :
    - Une SEULE classe `TfeDataset` (au lieu de 4 classes d'ablation)
    - Paramétrée par config (features actives, prompt mode, MMSE on/off)
    - `SliceExtractor` + `MultiViewSliceExtractor` réécrits localement
      (FIDÈLES à l'implémentation Tanguy : code_tanguy/medgemma/utils/slice_extractor.py)
    - `tfe_collate_fn` factorisé localement (plus d'import code_tanguy/)

Modes de prompt (config.prompt.mode) :
    - "full"          : 16 features détaillées + question CN/AD       (T1, T2)
    - "ablation"      : sous-ensemble de features selon config.data.tabular_features
                        (T3 : 10 features, T4 : 8 features)
    - "minimal"       : aucune feature, juste 4 IRM + question         (T5)
    - "image_centric" : prompt recentré sur les IRM                    (T6)

Usage :
    from dataset import TfeDataset, tfe_collate_fn
    ds = TfeDataset(csv_path, processor, config)
    loader = DataLoader(ds, batch_size=1, collate_fn=tfe_collate_fn)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. EXTRACTION SLICE — réécrite à l'identique de
#    code_tanguy/medgemma/utils/slice_extractor.py
# ═══════════════════════════════════════════════════════════════════════════

class SliceExtractor:
    """
    Extrait des coupes 2D depuis un volume IRM 3D.

    Implémentation fidèle à Tanguy `slice_extractor.py` :
        - Percentile clipping (1, 99) sur les voxels cerveau (>0)
        - Normalisation dans `normalize_range` (par défaut [-1, 1])
        - Resize via `scipy.ndimage.zoom(order=3)` (bicubique)
        - Sortie PIL RGB (3 canaux dupliqués)
        - Pas de transpose ni flip — orientation native du volume conservée
          (les volumes Tanguy sont déjà alignés MNI-152, orientation correcte)
    """

    def __init__(
        self,
        view: str = "coronal",
        n_slices: int = 5,
        region_start: float = 0.40,
        region_end: float = 0.60,
        output_size: int = 896,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.view = view.lower()
        self.n_slices = n_slices
        self.region_start = region_start
        self.region_end = region_end
        self.output_size = output_size
        self.normalize_range = normalize_range

        # Mapping axes (convention NIfTI standard)
        self.axis_map = {
            "sagittal": 0,   # x : gauche-droite
            "coronal":  1,   # y : avant-arrière (région hippocampe)
            "axial":    2,   # z : pied-tête (région ventricules)
        }
        if self.view not in self.axis_map:
            raise ValueError(
                f"View invalide : {view}. Choix : 'axial', 'coronal', 'sagittal'"
            )

    def extract_from_nifti(
        self, nifti_path: Union[str, Path]
    ) -> List[Image.Image]:
        """Charge un NIfTI puis appelle extract_from_volume."""
        nifti = nib.load(str(nifti_path))
        volume = nifti.get_fdata().astype(np.float32)
        return self.extract_from_volume(volume)

    def extract_from_volume(self, volume: np.ndarray) -> List[Image.Image]:
        """Extrait `n_slices` coupes 2D régulièrement espacées dans la région."""
        axis = self.axis_map[self.view]
        depth = volume.shape[axis]

        start_idx = int(self.region_start * depth)
        end_idx = int(self.region_end * depth)
        indices = np.linspace(start_idx, end_idx, self.n_slices, dtype=int)

        slices: List[Image.Image] = []
        for idx in indices:
            if axis == 0:
                slice_2d = volume[idx, :, :]
            elif axis == 1:
                slice_2d = volume[:, idx, :]
            else:
                slice_2d = volume[:, :, idx]
            slices.append(self._process_slice(slice_2d))
        return slices

    def _process_slice(self, slice_2d: np.ndarray) -> Image.Image:
        """Normalisation → resize bicubique → PIL RGB."""
        # 1. Normalisation dans normalize_range (typiquement [-1, 1])
        slice_2d = self._normalize_intensity(slice_2d)

        # 2. Resize bicubique (scipy.ndimage.zoom order=3)
        if (slice_2d.shape[0] != self.output_size
                or slice_2d.shape[1] != self.output_size):
            zoom_factors = (
                self.output_size / slice_2d.shape[0],
                self.output_size / slice_2d.shape[1],
            )
            slice_2d = zoom(slice_2d, zoom_factors, order=3)

        # 3. Mappe vers [0, 255] uint8 pour PIL
        min_val, max_val = self.normalize_range
        slice_2d = (slice_2d - min_val) / (max_val - min_val)  # [0, 1]
        slice_2d = np.clip(slice_2d * 255, 0, 255).astype(np.uint8)

        # 4. RGB par duplication (MedGemma attend 3 canaux)
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        return Image.fromarray(slice_rgb, mode="RGB")

    def _normalize_intensity(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Percentile clipping (1, 99) sur les voxels cerveau (>0), puis rescale
        dans `normalize_range`. Robuste aux outliers et aux coupes vides.
        """
        brain_mask = slice_2d > 0
        if brain_mask.sum() > 0:
            brain_voxels = slice_2d[brain_mask]
            p1, p99 = np.percentile(brain_voxels, (1, 99))
            slice_2d = np.clip(slice_2d, p1, p99)
            slice_2d = (slice_2d - p1) / (p99 - p1 + 1e-8)
        else:
            slice_2d = np.zeros_like(slice_2d)

        # Rescale vers normalize_range
        min_val, max_val = self.normalize_range
        slice_2d = slice_2d * (max_val - min_val) + min_val
        return slice_2d


class MultiViewSliceExtractor:
    """
    Extrait des coupes multi-vues (coronal + axial) d'un volume IRM 3D.

    Implémentation fidèle à Tanguy `slice_extractor.py` (classe identique).
    Compose deux `SliceExtractor` (un par vue) avec des régions différentes :
        - Coronal : 2 coupes dans la région hippocampe
        - Axial   : 2 coupes dans la région ventricules

    Sortie de `extract_all()` : liste de 4 PIL Images dans l'ordre
        [coronal_1, coronal_2, axial_1, axial_2]
    """

    def __init__(
        self,
        n_coronal: int = 2,
        n_axial: int = 2,
        coronal_region: Tuple[float, float] = (0.45, 0.55),
        axial_region: Tuple[float, float] = (0.35, 0.45),
        output_size: int = 448,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.n_coronal = n_coronal
        self.n_axial = n_axial
        self.output_size = output_size

        self.coronal_extractor = SliceExtractor(
            view="coronal",
            n_slices=n_coronal,
            region_start=coronal_region[0],
            region_end=coronal_region[1],
            output_size=output_size,
            normalize_range=normalize_range,
        )
        self.axial_extractor = SliceExtractor(
            view="axial",
            n_slices=n_axial,
            region_start=axial_region[0],
            region_end=axial_region[1],
            output_size=output_size,
            normalize_range=normalize_range,
        )

    def extract_from_nifti(
        self, nifti_path: Union[str, Path]
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """Retourne (coronal_slices, axial_slices) depuis un fichier NIfTI."""
        nifti = nib.load(str(nifti_path))
        volume = nifti.get_fdata().astype(np.float32)
        return self.extract_from_volume(volume)

    def extract_from_volume(
        self, volume: np.ndarray
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """Retourne (coronal_slices, axial_slices) depuis un ndarray 3D."""
        coronal = self.coronal_extractor.extract_from_volume(volume)
        axial = self.axial_extractor.extract_from_volume(volume)
        return coronal, axial

    def extract_all(
        self, nifti_path: Union[str, Path]
    ) -> List[Image.Image]:
        """Coupes coronales puis axiales, concaténées en une liste plate."""
        coronal, axial = self.extract_from_nifti(nifti_path)
        return coronal + axial

    @property
    def total_slices(self) -> int:
        return self.n_coronal + self.n_axial


# ═══════════════════════════════════════════════════════════════════════════
# 2. FORMATAGE DES FEATURES CLINIQUES (en texte naturel pour le LLM)
# ═══════════════════════════════════════════════════════════════════════════

# Mapping feature → libellé humain
FEATURE_LABELS: Dict[str, str] = {
    "AGE":       "Age",
    "PTGENDER":  "Gender",          # rendu "Male"/"Female"
    "PTEDUCAT":  "Years of education",
    "PTMARRY":   "Marital status",  # rendu via PTMARRY_LABELS
    "CATANIMSC": "Animal Fluency",
    "TRAASCOR":  "Trail Making A",
    "TRABSCOR":  "Trail Making B",
    "DSPANFOR":  "Digit Span Forward",
    "DSPANBAC":  "Digit Span Backward",
    "BNTTOTAL":  "Boston Naming Test",
    "BMI":       "BMI",
    "VSWEIGHT":  "Weight",
    "MH14ALCH":  "Alcohol use",
    "MH16SMOK":  "Smoking history",
    "MH4CARD":   "Cardiovascular history",
    "MH2NEURL":  "Neurological history",
}

PTMARRY_LABELS = {
    1: "Married", 2: "Widowed", 3: "Divorced", 4: "Never married",
}


def _format_value(feature: str, value: Any) -> str:
    """Format une valeur de feature pour affichage dans le prompt."""
    if pd.isna(value):
        return "Unknown"
    try:
        if feature == "PTGENDER":
            return "Male" if int(value) == 1 else "Female"
        if feature == "PTMARRY":
            return PTMARRY_LABELS.get(int(value), "Unknown")
        if feature in ("AGE", "PTEDUCAT"):
            return f"{float(value):.0f} years"
        if feature in ("TRAASCOR", "TRABSCOR"):
            return f"{float(value):.0f}s"
        if feature == "VSWEIGHT":
            return f"{float(value):.0f} kg"
        if feature == "BMI":
            return f"{float(value):.1f}"
        if feature in ("MH14ALCH", "MH16SMOK", "MH4CARD", "MH2NEURL"):
            return "Yes" if int(value) == 1 else "No"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.1f}"
        return str(value)
    except (ValueError, TypeError):
        return str(value)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_FULL = """Patient clinical information:
{clinical_info}

Brain MRI: 2 coronal slices (hippocampus) + 2 axial slices (ventricles).

Classify this patient as:
- CN: Cognitively Normal
- AD: Alzheimer's Disease

Respond with only: CN or AD"""


PROMPT_MINIMAL = """Brain MRI: 2 coronal slices (hippocampus) + 2 axial slices (ventricles).

Classify as CN (Cognitively Normal) or AD (Alzheimer's Disease).

Respond with only: CN or AD"""


PROMPT_IMAGE_CENTRIC = """Look carefully at the 4 brain MRI slices below (2 coronal showing the hippocampus, 2 axial showing the ventricles). Focus on:
- Hippocampal atrophy (typical AD marker)
- Ventricular enlargement
- Cortical thinning

Additional clinical context:
{clinical_info}

Based primarily on the MRI findings, classify this patient as:
- CN: Cognitively Normal
- AD: Alzheimer's Disease

Respond with only: CN or AD"""


# ═══════════════════════════════════════════════════════════════════════════
# T6 PROMPTS — 3 variantes pour reprompting image-focused
# ═══════════════════════════════════════════════════════════════════════════
# Toutes utilisent {clinical_section} qui est rempli différemment selon
# la config :
#   - Avec features tabulaires : "Clinical context...\n{clinical_info}\n"
#   - Sans features (ablation) : "(No clinical context provided.)\n"
# Ainsi le MÊME prompt sert pour T6a/T6b avec ou sans features tabulaires,
# garantissant que la seule différence mesurable est la présence des features.

# ── Variante 1 : focus structures + symétries ──
# Approche descriptive directe : on liste les structures cibles
# Hypothèse : oblige le modèle à "regarder" chaque région
PROMPT_IMAGE_FOCUSED_V1 = """You are analyzing 4 brain MRI slices from a single patient for signs of Alzheimer's disease.

The 4 images show:
- Slices 1-2 (coronal view): hippocampus and medial temporal lobe region
- Slices 3-4 (axial view): ventricles and cortical surface

Examine the MRI scans for the following anatomical markers:

1. HIPPOCAMPAL ATROPHY (visible in coronal slices):
   - Look at the hippocampus on both sides
   - Is it shrunken/atrophied or normal in size?
   - Note any asymmetry between left and right hippocampi

2. VENTRICULAR ENLARGEMENT (visible in axial slices):
   - Are the lateral ventricles enlarged or normal?
   - Look for ex-vacuo dilation (enlargement due to brain tissue loss)

3. CORTICAL ATROPHY (visible in all slices):
   - Are the cortical sulci widened?
   - Is the cortical mantle thinned?
   - Note any asymmetric atrophy patterns
{clinical_section}
Based on the MRI findings, classify this patient as:
- CN: Cognitively Normal (intact structures, no atrophy)
- AD: Alzheimer's Disease (hippocampal atrophy, ventricular enlargement, cortical thinning)

Respond with only: CN or AD"""


# ── Variante 2 : chain-of-thought clinique ──
# Approche raisonnement guidé : invite le modèle à construire son diagnostic
# par étapes successives. Hypothèse : meilleur ancrage de la décision.
PROMPT_IMAGE_FOCUSED_V2 = """You are analyzing 4 brain MRI slices from a single patient for signs of Alzheimer's disease.

The 4 images show:
- Slices 1-2 (coronal view): hippocampus region
- Slices 3-4 (axial view): ventricles region

Reason step by step through your analysis:

Step 1 — Observe the hippocampus on the coronal slices.
Is the hippocampal structure preserved (normal volume, sharp boundaries) or atrophied (shrunken, thinned, with surrounding CSF)?

Step 2 — Observe the lateral ventricles on the axial slices.
Are they within normal range or enlarged? Pay attention to the temporal horn dilation, a specific marker of medial temporal lobe atrophy.

Step 3 — Look for cortical atrophy across all slices.
Are the sulci widened? Is the cortical surface thinning?

Step 4 — Integrate observations.
Multiple positive markers (hippocampal atrophy + ventricular enlargement + cortical thinning) strongly suggest Alzheimer's pathology.
{clinical_section}
Step 5 — Final classification.
Based on your step-by-step analysis, classify this patient as:
- CN: Cognitively Normal
- AD: Alzheimer's Disease

Respond with only: CN or AD"""


# ── Variante 3 : rôle de radiologue expert ──
# Approche role-prompting : active le corpus radiologique pré-entraîné
# Hypothèse : MedGemma a été pré-entraîné sur de la littérature radio,
# le rôle l'oriente vers ce sous-domaine.
PROMPT_IMAGE_FOCUSED_V3 = """You are an expert neuroradiologist with 20 years of experience in dementia imaging. You are reviewing T1-weighted brain MRI scans for signs of Alzheimer's disease.

You have 4 MRI slices to interpret:
- Slices 1-2 (coronal): centered on the hippocampal region
- Slices 3-4 (axial): centered on the ventricular system

As an expert, focus your radiological assessment on:
- Medial temporal atrophy, particularly the hippocampus and entorhinal cortex
- Ventricular morphology (notably the temporal horn)
- Global and focal cortical atrophy patterns
- Asymmetries that may indicate localized neurodegeneration
{clinical_section}
Render your radiological diagnosis:
- CN: Cognitively Normal — no significant atrophy
- AD: Alzheimer's Disease — typical neurodegenerative pattern

Respond with only: CN or AD"""


# Mapping pour résolution depuis la config
IMAGE_FOCUSED_TEMPLATES = {
    "v1": PROMPT_IMAGE_FOCUSED_V1,
    "v2": PROMPT_IMAGE_FOCUSED_V2,
    "v3": PROMPT_IMAGE_FOCUSED_V3,
}


def get_prompt_template(mode: str, variant: Optional[str] = None) -> str:
    """
    Retourne le template de prompt selon le mode demandé.

    Args:
        mode    : 'full' / 'ablation' / 'minimal' / 'image_centric' /
                  'image_focused' (T6)
        variant : pour 'image_focused' uniquement, choix de la variante :
                  'v1' (structures), 'v2' (chain-of-thought), 'v3' (radiologue)
    """
    if mode == "image_focused":
        v = variant or "v1"
        if v not in IMAGE_FOCUSED_TEMPLATES:
            raise ValueError(
                f"Variante image_focused inconnue : '{v}'. "
                f"Choix : {list(IMAGE_FOCUSED_TEMPLATES.keys())}"
            )
        return IMAGE_FOCUSED_TEMPLATES[v]

    templates = {
        "full":          PROMPT_FULL,
        "ablation":      PROMPT_FULL,           # même template, features filtrées
        "minimal":       PROMPT_MINIMAL,
        "image_centric": PROMPT_IMAGE_CENTRIC,
    }
    if mode not in templates:
        raise ValueError(
            f"Mode prompt '{mode}' inconnu. "
            f"Choix : {list(templates.keys()) + ['image_focused']}"
        )
    return templates[mode]


# ═══════════════════════════════════════════════════════════════════════════
# 4. DATASET UNIFIÉ
# ═══════════════════════════════════════════════════════════════════════════

class TfeDataset(Dataset):
    """
    Dataset multimodal MedGemma — pipeline TFE Alzheimer.

    Une seule classe pour toutes les tâches T1-T6, paramétrée par config.

    Args:
        csv_path     : chemin vers train.csv / val.csv / test.csv
                       (généré par 00_prepare_splits/prepare_splits.py)
        processor    : MedGemma AutoProcessor (texte + images)
        config       : dict de configuration (depuis utils.load_config)
        is_training  : si False, désactive l'ajout du label assistant
                       (utile pour évaluation zero-shot ou inférence)
        max_samples  : limite le nombre d'échantillons (debug)

    Lecture de la config :
        config["data"]["tabular_features"]      : liste des features actives
                                                  (alias accepté : config["data"]["features"])
        config["data"]["use_tabular"]           : si False, force prompt minimal
        config["data"]["slice_extraction"]      : params MultiViewSliceExtractor
        config["mmse_head"]["enabled"]          : si True, ajoute mmse_target
        config.get("prompt", {}).get("mode")    : "full" / "ablation" / "minimal" / "image_centric"
                                                  (auto-détecté sinon)

    Format de sortie de __getitem__ :
        {
            "input_ids":       torch.LongTensor (seq_len,)
            "attention_mask":  torch.LongTensor (seq_len,)
            "pixel_values":    torch.FloatTensor (n_views, C, H, W)
            "labels":          torch.LongTensor (seq_len,)         # si is_training
            "mmse_target":     torch.FloatTensor scalar dans [0,1] # si MMSE actif
            "reg_weight":      torch.FloatTensor scalar             # 1.0 ou 0.1
        }
    """

    def __init__(
        self,
        csv_path: str,
        processor: Any,
        config: Dict[str, Any],
        is_training: bool = True,
        max_samples: Optional[int] = None,
        cohort_filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            cohort_filter : optionnel, pour T8 (ADNI vs no-ADNI) ou tests OOD.
                Format : {"include": ["ADNI"]}  → ne garde QUE ADNI
                         {"exclude": ["ADNI"]}  → exclut ADNI (garde NACC + OASIS)
                         {"include": ["NACC", "OASIS"]}  → équivalent à exclude ADNI
                Le filtrage est appliqué APRÈS le chargement CSV, AVANT max_samples.
        """
        self.csv_path = csv_path
        self.processor = processor
        self.config = config
        self.is_training = is_training
        self.cohort_filter = cohort_filter

        # ── Lecture de la config ─────────────────────────────────────────
        data_cfg  = config.get("data", {})
        slice_cfg = data_cfg.get("slice_extraction", {})

        # Accepte 'tabular_features' OU 'features' (rétro-compat configs)
        feats = data_cfg.get("tabular_features", None)
        if feats is None:
            feats = data_cfg.get("features", [])
        self.tabular_features: List[str] = list(feats or [])

        self.use_tabular: bool = bool(data_cfg.get("use_tabular", True))
        self.use_visual: bool  = bool(data_cfg.get("use_visual",  True))
        self.mmse_enabled: bool = bool(
            config.get("mmse_head", {}).get("enabled", False)
        )

        # Mode prompt : explicite, ou auto-détecté
        prompt_cfg = config.get("prompt", {}) or {}
        self.prompt_mode: str = prompt_cfg.get("mode") or self._auto_detect_mode()
        self.prompt_variant: Optional[str] = prompt_cfg.get("variant", None)
        self.prompt_template: str = get_prompt_template(
            self.prompt_mode, variant=self.prompt_variant,
        )

        # ── Slice extractor ──────────────────────────────────────────────
        self.output_size = int(slice_cfg.get("output_size", 448))
        self.slice_extractor = MultiViewSliceExtractor(
            n_coronal=int(slice_cfg.get("n_coronal", 2)),
            n_axial=int(slice_cfg.get("n_axial", 2)),
            coronal_region=tuple(slice_cfg.get("coronal_region", [0.45, 0.55])),
            axial_region=tuple(slice_cfg.get("axial_region", [0.35, 0.45])),
            output_size=self.output_size,
            normalize_range=tuple(slice_cfg.get("normalize_range", [-1.0, 1.0])),
        )

        # ── Chargement CSV ───────────────────────────────────────────────
        self.df = pd.read_csv(csv_path)
        n_before_filter = len(self.df)

        # ── Filtrage cohorte (T8 ADNI vs no-ADNI, etc.) ──────────────────
        if cohort_filter is not None and "source" in self.df.columns:
            include = cohort_filter.get("include")
            exclude = cohort_filter.get("exclude")
            if include is not None:
                self.df = self.df[self.df["source"].isin(include)]
                logger.info(f"  Cohort filter (include) : {include}")
            if exclude is not None:
                self.df = self.df[~self.df["source"].isin(exclude)]
                logger.info(f"  Cohort filter (exclude) : {exclude}")
            n_after = len(self.df)
            logger.info(
                f"  Cohort filter : {n_before_filter} → {n_after} sujets "
                f"({n_after/n_before_filter*100:.1f}% conservés)"
            )

        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        # Mappings classes
        self.label_to_text = {0: "CN", 1: "AD"}
        self.text_to_label = {"CN": 0, "AD": 1}

        self._log_stats()

    # ── Auto-détection du mode prompt ─────────────────────────────────────

    def _auto_detect_mode(self) -> str:
        """Devine le mode prompt depuis la config si non explicite."""
        if not self.use_tabular or len(self.tabular_features) == 0:
            return "minimal"
        return "full" if len(self.tabular_features) == 16 else "ablation"

    # ── Logging ─────────────────────────────────────────────────────────

    def _log_stats(self) -> None:
        n = len(self.df)
        logger.info(
            f"TfeDataset chargé : {n} échantillons depuis {Path(self.csv_path).name}"
        )
        logger.info(f"  Mode prompt        : {self.prompt_mode}"
                    + (f" (variant={self.prompt_variant})"
                       if self.prompt_variant else ""))
        logger.info(f"  use_tabular        : {self.use_tabular}")
        logger.info(f"  use_visual         : {self.use_visual}"
                    + ("  ← ablation image-only" if not self.use_visual else ""))
        logger.info(f"  Features actives   : {len(self.tabular_features)}")
        if self.tabular_features:
            available = [f for f in self.tabular_features if f in self.df.columns]
            missing = set(self.tabular_features) - set(available)
            logger.info(f"  Features dans CSV  : {len(available)}")
            if missing:
                logger.warning(f"  ⚠ Features manquantes : {missing}")
        logger.info(f"  MMSE head enabled  : {self.mmse_enabled}")
        if "label" in self.df.columns:
            counts = self.df["label"].value_counts().sort_index()
            for lbl, count in counts.items():
                name = self.label_to_text.get(int(lbl), str(lbl))
                logger.info(f"  {name}: {count} ({100*count/n:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    # ── Construction du clinical_info textuel ────────────────────────────

    def _format_clinical_info(self, row: pd.Series) -> str:
        """
        Formate les features actives en bullet list textuelle.

        Si `tabular_features` est vide ou si une feature n'est pas dans le CSV,
        elle est silencieusement skippée. Les valeurs NaN sont rendues "Unknown"
        (en pratique, jamais NaN après imputation train-only).
        """
        if not self.tabular_features:
            return "No clinical data provided."

        info_parts: List[str] = []
        for feature in self.tabular_features:
            if feature not in row.index:
                continue
            value = row[feature]
            label = FEATURE_LABELS.get(feature, feature)
            formatted = _format_value(feature, value)
            info_parts.append(f"- {label}: {formatted}")

        if not info_parts:
            return "No clinical data provided."

        return "\n".join(info_parts) + (
            "\n\nTask: Analyze the 4 MRI slices and this clinical report. "
            "Classify as CN (Cognitively Normal) or AD (Alzheimer's Disease)."
        )

    # ── Construction des messages de conversation ────────────────────────

    def _build_messages(
        self,
        slices: List[Image.Image],
        prompt_text: str,
        label_text: str,
    ) -> List[Dict[str, Any]]:
        """Format MedGemma : messages [{role, content: [image*N, text]}]."""
        user_content: List[Dict[str, Any]] = []
        for img in slices:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        if self.is_training and label_text:
            messages.append({"role": "assistant", "content": label_text})
        return messages

    # ── Tokenisation via le processor MedGemma ───────────────────────────

    def _process_messages(
        self,
        messages: List[Dict[str, Any]],
        slices: List[Image.Image],
    ) -> Dict[str, torch.Tensor]:
        """chat_template + processor → input_ids / labels / pixel_values."""
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not self.is_training,
        )
        processed = self.processor(
            text=text,
            images=slices,
            return_tensors="pt",
            padding=False,
        )

        result: Dict[str, torch.Tensor] = {
            "input_ids":      processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
        }
        if "pixel_values" in processed:
            result["pixel_values"] = processed["pixel_values"].squeeze(0)

        # En training, labels = input_ids (cross-entropy autoregressive)
        if self.is_training:
            result["labels"] = result["input_ids"].clone()

        return result

    # ── __getitem__ ──────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        scan_path = row["scan_path"]

        # 1. Extraction des coupes IRM (4 PIL Images 448×448 RGB)
        try:
            slices = self.slice_extractor.extract_all(scan_path)
        except Exception as e:
            logger.warning(f"Erreur chargement {scan_path} : {e}. Images noires.")
            slices = [
                Image.new("RGB", (self.output_size, self.output_size), color=0)
                for _ in range(self.slice_extractor.total_slices)
            ]

        # 2. Construction du prompt texte
        if self.prompt_mode == "minimal":
            prompt_text = self.prompt_template
        elif self.prompt_mode == "image_focused":
            # Mode T6 : utilise {clinical_section} qui peut être vide
            # ou contenir les features tabulaires.
            if self.tabular_features and self.use_tabular:
                clinical_info = self._format_clinical_info(row)
                clinical_section = (
                    f"\nAdditional clinical context (supplementary):\n"
                    f"{clinical_info}\n"
                )
            else:
                clinical_section = (
                    "\n(No clinical context provided. Base your decision "
                    "on MRI findings only.)\n"
                )
            prompt_text = self.prompt_template.format(
                clinical_section=clinical_section,
            )
        else:
            clinical_info = self._format_clinical_info(row)
            prompt_text = self.prompt_template.format(clinical_info=clinical_info)

        # 3. Label string ("CN"/"AD")
        label = int(row["label"]) if "label" in row.index else None
        label_text = self.label_to_text.get(label, "") if label is not None else ""

        # 4. Tokenisation via processor MedGemma
        messages = self._build_messages(slices, prompt_text, label_text)
        item = self._process_messages(messages, slices)

        # Ablation visuelle : remplace les coupes IRM par des images noires (zéros).
        # Le modèle traite la séquence textuelle seule ; MedSigLIP produit un
        # embedding "neutre" à partir d'un input nul.
        # La structure du prompt (tokens <image>) est conservée à l'identique
        # pour ne pas perturber la longueur de séquence.
        if not self.use_visual:
            item["pixel_values"] = torch.zeros_like(item["pixel_values"])

        # 5. Cibles multitâche (si MMSE activé)
        if self.mmse_enabled:
            mmse_val = float(row["mmse_score"]) if "mmse_score" in row.index else 0.0
            # Normalisation [0,30] → [0,1] (cohérent avec sigmoid côté training)
            item["mmse_target"] = torch.tensor(mmse_val / 30.0, dtype=torch.float32)
            # Poids : 1.0 si vraie mesure ADNI, 0.1 si imputée
            has_real = (
                int(row["has_real_measures"])
                if "has_real_measures" in row.index else 0
            )
            item["reg_weight"] = torch.tensor(
                1.0 if has_real == 1 else 0.1, dtype=torch.float32
            )

        return item


# ═══════════════════════════════════════════════════════════════════════════
# 5. COLLATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def tfe_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate avec right-padding pour batchs MedGemma de longueur variable.

    Compatible avec les deux modes :
        - Classification seule : input_ids / attention_mask / pixel_values / labels
        - Multitâche MMSE      : + mmse_score / regression_weight

    pixel_values en sortie : (batch_size * n_views, C, H, W) — concaténé sur
    l'axe 0, format attendu par MedGemma pour ses VLM.
    """
    if not batch:
        raise ValueError("tfe_collate_fn : batch vide")

    # ── Cibles multitâche ───────────────────────────────────────────────
    has_mmse = "mmse_target" in batch[0]
    if has_mmse:
        mmse_targets = torch.stack(
            [item.pop("mmse_target") for item in batch]
        )
        reg_weights = torch.stack(
            [item.pop("reg_weight", torch.tensor(1.0)) for item in batch]
        )

    # ── Padding séquences ────────────────────────────────────────────────
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)
    has_labels = "labels" in batch[0]

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = (
        torch.full((batch_size, max_len), -100, dtype=torch.long)
        if has_labels else None
    )

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        if has_labels:
            labels[i, :seq_len] = item["labels"]

    # ── pixel_values : (batch_size * n_views, C, H, W) ───────────────────
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)

    out: Dict[str, torch.Tensor] = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pixel_values":   pixel_values,
    }
    if has_labels:
        out["labels"] = labels
    if has_mmse:
        out["mmse_score"] = mmse_targets
        out["regression_weight"] = reg_weights

    return out