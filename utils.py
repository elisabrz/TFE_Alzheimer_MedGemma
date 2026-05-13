"""
utils.py — Fonctions et classes partagées entre toutes les tâches du TFE.

Centralise tout ce qui était dupliqué entre :
    - train_step_2_v3.py (multitâche classification + MMSE)
    - train_baseline_cls_only.py (classification seule)
    - train_step_3.py (variante step 3)
    - tfe_eval.py (find_answer_position, evaluate_dataset, métriques)

Sections :
    1. Configuration (load_config avec inherits_from)
    2. Environnement (set_seed, setup_env, gestion .env)
    3. GPU (check_vram, release_gpu, signal_handler, ClearCacheCallback)
    4. Checkpoints (is_valid_checkpoint, get_last_checkpoint)
    5. Loss & têtes (FocalLoss, MMSEHead)
    6. Verbalizer (find_answer_position)
    7. Évaluation (evaluate_dataset, compute_mmse_metrics, set_token_ids)
    8. Plots (plot_roc_curve, plot_confusion_matrix, plot_training_curves,
              plot_mmse_scatter)
    9. EvalCallback (custom — remplace pipeline HF cassé)
   10. Sauvegarde (save_metrics_json)

Convention d'usage :
    from utils import (
        load_config, set_seed, check_vram, release_gpu, signal_handler,
        FocalLoss, MMSEHead, find_answer_position, evaluate_dataset,
        ClearCacheCallback, EvalCallback, set_token_ids,
        plot_roc_curve, plot_confusion_matrix,
    )
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import signal
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # backend non-interactif (serveur)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, recall_score, roc_auc_score, roc_curve, auc as sklearn_auc,
)
from tqdm import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None  # géré dans compute_mmse_metrics

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ═══════════════════════════════════════════════════════════════════════════
# 0. CONSTANTES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════

# Token IDs CN/AD — initialisés via set_token_ids() depuis le main de chaque tâche
CN_TOKEN_ID: Optional[int] = None
AD_TOKEN_ID: Optional[int] = None


def set_token_ids(cn_id: int, ad_id: int) -> None:
    """À appeler AVANT toute évaluation, depuis le main du script de tâche."""
    global CN_TOKEN_ID, AD_TOKEN_ID
    CN_TOKEN_ID = cn_id
    AD_TOKEN_ID = ad_id


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION (avec héritage inherits_from)
# ═══════════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    """Merge récursif : `override` écrase `base`, listes complètement remplacées."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str, _seen: Optional[set] = None) -> dict:
    """
    Charge un YAML avec support du champ `inherits_from`.

    Si la config contient `inherits_from: ../config/config_base.yaml`, charge
    récursivement le parent puis applique l'override.

    Détecte les cycles d'héritage. Les chemins relatifs sont résolus depuis
    le dossier du fichier qui les déclare.
    """
    if _seen is None:
        _seen = set()
    abs_path = os.path.abspath(path)
    if abs_path in _seen:
        raise ValueError(f"Cycle d'héritage détecté : {abs_path} déjà visité")
    _seen.add(abs_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config introuvable : {abs_path}")

    with open(abs_path) as f:
        cfg = yaml.safe_load(f) or {}

    parent_ref = cfg.pop("inherits_from", None)
    if parent_ref:
        parent_path = parent_ref
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(os.path.dirname(abs_path), parent_path)
        parent_cfg = load_config(parent_path, _seen=_seen)
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# 2. ENVIRONNEMENT
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Reproductibilité : Python, NumPy, PyTorch, CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_env(
    seed: int = 42,
    load_dotenv_file: bool = True,
    cuda_alloc_conf: Optional[str] = "max_split_size_mb:256",
) -> None:
    """
    À appeler en début de script :
        - charge .env si présent (HF_TOKEN, WANDB_API_KEY)
        - fixe les seeds
        - configure CUDA pour limiter la fragmentation VRAM
    """
    if load_dotenv_file:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv non installé, pas grave

    set_seed(seed)

    if cuda_alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf

    # Évite le warning tokenizer dans les sub-processes des dataloaders
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ═══════════════════════════════════════════════════════════════════════════
# 3. GESTION GPU
# ═══════════════════════════════════════════════════════════════════════════

def check_vram(min_gb: float = 10.0) -> None:
    """Vérifie qu'on a au moins `min_gb` de VRAM libre, sinon raise."""
    if not torch.cuda.is_available():
        raise RuntimeError("Aucun GPU CUDA disponible.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1e9
    total_gb = total_bytes / 1e9
    print(f"[*] VRAM : {free_gb:.1f} GB libres / {total_gb:.1f} GB total")
    if free_gb < min_gb:
        raise RuntimeError(
            f"VRAM insuffisante : {free_gb:.1f} GB libres, {min_gb} GB requis."
        )


def release_gpu(model=None, trainer=None) -> None:
    """Libère un maximum de VRAM. Idempotent (peut être appelé plusieurs fois)."""
    print("\n[*] Libération du GPU...")
    if trainer is not None:
        try:
            del trainer.model
            del trainer.optimizer
        except Exception:
            pass
    if model is not None:
        try:
            del model
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
            print(f"[✓] VRAM libre : {free_gb:.1f} GB")
        except RuntimeError as e:
            # Workers dataloader sans CUDA init, ou GPU déjà libéré
            print(f"[*] (CUDA non disponible ici : {type(e).__name__})")


# Références globales pour le signal handler (set par les scripts main)
_TRAINER_REF: Optional[Any] = None
_MODEL_REF:   Optional[Any] = None
# PID du process main, capturé au moment de register_signal_refs.
# Permet de détecter quand signal_handler est appelé depuis un dataloader worker
# (subprocess) — auquel cas on quitte sans rien toucher (CUDA non init dans worker).
_MAIN_PID:    Optional[int] = None


def register_signal_refs(trainer=None, model=None) -> None:
    """
    Permet aux scripts de tâche d'enregistrer leur trainer/model pour que
    signal_handler puisse les sauvegarder en cas de Ctrl+C.

    Capture aussi le PID du process appelant comme PID "main", pour que
    les dataloader workers (qui héritent du handler à fork) puissent être
    distingués et sortir proprement sans toucher CUDA.
    """
    global _TRAINER_REF, _MODEL_REF, _MAIN_PID
    _TRAINER_REF = trainer
    _MODEL_REF = model
    if _MAIN_PID is None:
        _MAIN_PID = os.getpid()


def signal_handler(sig, frame) -> None:
    """
    Handler SIGINT/SIGTERM : sauvegarde d'urgence + libération propre.
    À enregistrer avec :
        signal.signal(signal.SIGINT,  signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    """
    # Si on est dans un dataloader worker (subprocess), on sort sans rien
    # faire — CUDA n'y est pas initialisé, et le main process s'occupe du
    # nettoyage. Sans ce check, chaque worker spam des "[!] Sauvegarde
    # échouée : CUDA error" en parallèle.
    if _MAIN_PID is not None and os.getpid() != _MAIN_PID:
        sys.exit(0)

    sig_name = "SIGTERM" if sig == signal.SIGTERM else "SIGINT (Ctrl+C)"
    print(f"\n[!] Signal reçu : {sig_name} — arrêt propre...")

    # Sauvegarde d'urgence — uniquement si CUDA est utilisable
    cuda_ok = (
        torch.cuda.is_available()
        and torch.cuda.is_initialized()
    )
    if _TRAINER_REF is not None and cuda_ok:
        try:
            emergency_path = os.path.join(
                _TRAINER_REF.args.output_dir, "checkpoint_emergency"
            )
            print(f"[*] Sauvegarde d'urgence dans : {emergency_path}")
            # save_model() délègue à model.save_pretrained() qui sauve déjà
            # mmse_head.pt (cf. TfeMedGemmaWithMMSE.save_pretrained)
            _TRAINER_REF.save_model(emergency_path)
        except Exception as e:
            print(f"[!] Sauvegarde d'urgence échouée : {e}")

    if HAS_WANDB:
        try:
            wandb.finish()
        except Exception:
            pass

    # Nettoyage GPU défensif (skip si CUDA pas initialisé)
    if cuda_ok:
        release_gpu(model=_MODEL_REF, trainer=_TRAINER_REF)
    print("[✓] Arrêt propre terminé.")
    sys.exit(0)


class ClearCacheCallback(TrainerCallback):
    """Libère le cache GPU à chaque fin d'époque."""

    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
            print(
                f"\n[GPU] Fin époque {state.epoch:.0f} — VRAM libre : {free_gb:.1f} GB",
                flush=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════

def is_valid_checkpoint(path: str, require_mmse: bool = False) -> bool:
    """
    Un checkpoint HF-PEFT valide contient au minimum :
        - adapter_config.json    (config LoRA)
        - adapter_model.safetensors  (poids LoRA — ou .bin pour anciens runs)

    Args:
        require_mmse : si True, exige aussi mmse_head.pt (multitâche).
    """
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        return False
    has_weights = (
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_model.bin"))
    )
    if not has_weights:
        return False
    if require_mmse and not os.path.exists(os.path.join(path, "mmse_head.pt")):
        return False
    return True


def get_last_checkpoint(output_dir: str) -> Optional[str]:
    """
    Détecte le dernier checkpoint VALIDE (supprime les corrompus en chemin).
    Retourne None si aucun checkpoint valide.
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))

    for ckpt in reversed(checkpoints):
        ckpt_path = os.path.join(output_dir, ckpt)
        if is_valid_checkpoint(ckpt_path):
            print(f"[*] Checkpoint valide détecté : {ckpt_path}")
            return ckpt_path
        print(f"[!] Checkpoint corrompu — suppression : {ckpt_path}")
        try:
            shutil.rmtree(ckpt_path)
        except Exception as e:
            print(f"    (suppression échouée : {e})")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 5. LOSS & TÊTES
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss binaire pondérée (Lin et al. 2017) pour le déséquilibre AD/CN.

    α_AD : poids de la classe AD (positive). Avec ~21.7% AD dans le dataset,
           α=0.78 compense bien (poids inverse ~0.78 pour AD, 0.22 pour CN).
    γ    : focusing parameter. γ=2 réduit la perte pour les exemples bien
           classés et concentre l'apprentissage sur les difficiles.
    """

    def __init__(self, alpha_ad: float = 0.78, gamma: float = 2.0):
        super().__init__()
        self.alpha_ad = alpha_ad
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(ce_loss, self.alpha_ad),
            torch.full_like(ce_loss, 1.0 - self.alpha_ad),
        )
        return (alpha_t * (1 - pt) ** self.gamma * ce_loss).mean()


class MMSEHead(nn.Module):
    """
    Tête de régression MMSE : LayerNorm(2560) + Linear(2560 → 1).

    Architecture FIXÉE par le pipeline : LayerNorm avant Linear stabilise
    les gradients en bf16 et empêche le collapse (tous les patients → MMSE=0)
    observé en step 1/2 avec un Linear seul.

    Sortie brute : passer par sigmoid × 30 côté training pour borner dans [0, 30].

    Clés du state_dict (mmse_head.pt) :
        - norm.weight, norm.bias
        - fc.weight,   fc.bias
    """

    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, 1, dtype=torch.float32)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class MMSEHeadLegacy(nn.Module):
    """Ancien format (Linear seul) — pour rétro-compatibilité au chargement."""

    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_mmse_head(path: str, hidden_size: int = 2560) -> nn.Module:
    """
    Recharge une tête MMSE depuis un checkpoint avec auto-détection du format
    (LayerNorm+Linear actuel vs Linear seul legacy). Retourne une instance
    initialisée aléatoirement si le fichier est absent.
    """
    head_path = os.path.join(path, "mmse_head.pt")
    if not os.path.exists(head_path):
        print(f"[!] mmse_head.pt absent dans {path} — initialisation par défaut")
        return MMSEHead(hidden_size)

    try:
        state = torch.load(head_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[!] Lecture mmse_head.pt échouée : {e} — init par défaut")
        return MMSEHead(hidden_size)

    keys = set(state.keys())
    if any(k.startswith("norm.") for k in keys) and any(k.startswith("fc.") for k in keys):
        head = MMSEHead(hidden_size)
        head.load_state_dict(state)
        print("[*] Tête MMSE (LayerNorm + Linear) chargée")
    elif any(k.startswith("fc.") for k in keys):
        head = MMSEHeadLegacy(hidden_size)
        head.load_state_dict({k: v for k, v in state.items() if k.startswith("fc.")})
        print("[*] Tête MMSE legacy (fc.) chargée")
    elif "weight" in keys and "bias" in keys:
        head = MMSEHeadLegacy(hidden_size)
        head.fc.weight.data = state["weight"]
        head.fc.bias.data = state["bias"]
        print("[*] Tête MMSE ancienne (Linear direct) rechargée via remapping")
    else:
        print(f"[!] Format mmse_head.pt inconnu : {list(keys)[:5]}")
        head = MMSEHead(hidden_size)
    return head


# ═══════════════════════════════════════════════════════════════════════════
# 6. VERBALIZER (position du token de réponse)
# ═══════════════════════════════════════════════════════════════════════════

def find_answer_position(
    labels_row: torch.Tensor | np.ndarray,
    cn_id: int,
    ad_id: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Trouve la position du token CN/AD et le logit_pos correspondant.

    CONVENTION AUTOREGRESSIVE :
        logits[k-1] prédit labels[k] → on lit logits à logit_pos = ans_pos - 1.

    Retourne (logit_pos, true_label) ou (None, None) si séquence vide.
    """
    for pos in range(len(labels_row)):
        tok = int(labels_row[pos])
        if tok == cn_id:
            return max(0, pos - 1), 0
        if tok == ad_id:
            return max(0, pos - 1), 1

    # Fallback : dernier token non-(-100). On NE FABRIQUE PLUS un label CN
    # par défaut (ancien bug). Si le token n'est ni CN ni AD, on retourne
    # (None, None) pour signaler le mismatch et laisser l'appelant skipper
    # la sample (alignement préservé via index explicite).
    if hasattr(labels_row, "nonzero") and not isinstance(labels_row, np.ndarray):
        valid = (labels_row != -100).nonzero(as_tuple=True)[0]
    else:
        valid = np.where(np.asarray(labels_row) != -100)[0]

    if len(valid) == 0:
        return None, None

    ans_pos = int(valid[-1])
    tok = int(labels_row[ans_pos])
    if tok == cn_id:
        return max(0, ans_pos - 1), 0
    if tok == ad_id:
        return max(0, ans_pos - 1), 1
    # Token inattendu (verbalizer mismatch ou tokenizer drift) → signal explicite
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# 7. ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_mmse_metrics(
    mmse_pred: np.ndarray, mmse_true: np.ndarray, real_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calcule MAE, RMSE, R², CC pour la régression MMSE.

    real_mask : True = vrai score MMSE (poids 1), False = placeholder imputé.
    Les métriques `*_real` ne sont calculées que sur les vrais scores ADNI.
    """
    out: Dict[str, float] = {
        "mae": float("nan"), "rmse": float("nan"),
        "r2": float("nan"),  "cc":   float("nan"),
        "mae_real": float("nan"), "rmse_real": float("nan"),
        "r2_real":  float("nan"), "cc_real":   float("nan"),
        "n_real": 0,
    }

    if len(mmse_pred) == 0:
        return out

    p, t = np.asarray(mmse_pred, dtype=float), np.asarray(mmse_true, dtype=float)
    out["mae"] = float(mean_absolute_error(t, p))
    out["rmse"] = float(np.sqrt(mean_squared_error(t, p)))

    mask = np.asarray(real_mask, dtype=bool)
    if mask.sum() >= 2:
        p_real, t_real = p[mask], t[mask]
        out["n_real"] = int(mask.sum())
        out["mae_real"] = float(mean_absolute_error(t_real, p_real))
        out["rmse_real"] = float(np.sqrt(mean_squared_error(t_real, p_real)))

        if np.var(t_real) > 1e-8:
            ss_res = float(np.sum((t_real - p_real) ** 2))
            ss_tot = float(np.sum((t_real - np.mean(t_real)) ** 2))
            out["r2_real"] = 1.0 - ss_res / ss_tot

        if pearsonr is not None and np.var(t_real) > 1e-8 and np.var(p_real) > 1e-8:
            try:
                cc, _ = pearsonr(t_real, p_real)
                out["cc_real"] = float(cc)
            except Exception:
                pass

    return out


@torch.no_grad()
def _find_text_decoder_last_layer_for_eval(model) -> Optional[torch.nn.Module]:
    """
    Localise EXPLICITEMENT la dernière couche du décodeur texte (Gemma).
    Cohérent avec trainers._find_text_decoder_last_layer (même logique).
    Évite l'ancien bug "dernière ModuleList rencontrée" qui pouvait pointer
    sur le vision encoder.
    """
    candidates: list = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if "vision" in name.lower():
            continue
        if name.endswith("layers") or "language_model" in name or "text_model" in name:
            candidates.append((name, module))
    if not candidates:
        return None
    for name, module in candidates:
        if "language_model" in name:
            return module[-1]
    for name, module in candidates:
        if "text_model" in name:
            return module[-1]
    name, module = max(candidates, key=lambda x: len(x[1]))
    return module[-1]


def evaluate_dataset(
    model,
    dataset,
    collate_fn: Callable,
    cn_id: int,
    ad_id: int,
    batch_size: int = 1,
    device: str = "cuda",
    return_indices: bool = False,
) -> Tuple:
    """
    Boucle d'évaluation complète, équivalente à tfe_eval.evaluate_dataset().

    Compatible avec :
        - modèle multitâche (avec MMSEHead, hook hidden state)
        - modèle classification seule (pas de tête MMSE)

    Args:
        return_indices : si True, retourne en plus la liste des indices df
            des samples conservés (alignement strict pour évaluations stratifiées).

    Retourne (results, preds, probs, true_cls, mmse_pred, mmse_true)
        ou (..., kept_indices) si return_indices=True.

    `results_dict` contient :
        accuracy, auc, f1, sensitivity, specificity, optimal_threshold,
        f1_calibrated, sensitivity_calibrated, specificity_calibrated,
        + mae, rmse, r2, cc, mae_real, rmse_real, r2_real, cc_real, n_real

    ⚠️ ALIGNEMENT : les listes retournées (preds/probs/true_cls/...) ne
    contiennent QUE les samples où find_answer_position a réussi. Si un sample
    est skippé (verbalizer mismatch), il n'apparaît dans aucune liste.
    Pour faire une analyse par cohorte/subject_id, utiliser return_indices=True
    et indexer dataset.df.iloc[kept_indices] (jamais dataset.df directement).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Anchoring explicite de la couche cible (cohérent avec trainers.py)
    target_layer = _find_text_decoder_last_layer_for_eval(model)
    if target_layer is None:
        print("[!] evaluate_dataset : impossible de localiser la couche décodeur texte. "
              "Le hook MMSE ne fonctionnera pas.")

    has_mmse_head = hasattr(model, "mmse_head")

    all_preds: List[int] = []
    all_probs: List[float] = []
    all_true_cls: List[int] = []
    all_mmse_pred: List[float] = []
    all_mmse_true: List[float] = []
    all_has_real: List[bool] = []
    kept_indices: List[int] = []
    sample_idx = 0

    # Libération agressive AVANT éval — le training laisse ~5 GB de gradients
    # et cache d'optimizer en VRAM. Sans ça, OOM sur RTX 4080 16GB.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"[EVAL] VRAM libre avant éval : {free_gb:.1f} GB")

    for batch_i, batch in enumerate(tqdm(loader, desc="Évaluation")):
        # MMSE targets (présents en multitâche, absents sinon)
        mmse_true_batch = batch.pop("mmse_score", None)
        _ = batch.pop("regression_weight", None)
        labels = batch.get("labels")

        # Reshape pixel_values 5D → 4D (n_views collapsed)
        if "pixel_values" in batch and batch["pixel_values"].ndim == 5:
            b, n, c, h, w = batch["pixel_values"].shape
            batch["pixel_values"] = batch["pixel_values"].view(b * n, c, h, w)
        if "pixel_attention_mask" in batch and batch["pixel_attention_mask"].ndim == 4:
            b, n, h, w = batch["pixel_attention_mask"].shape
            batch["pixel_attention_mask"] = batch["pixel_attention_mask"].view(b * n, h, w)

        batch_gpu = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        # Hook hidden state — UNIQUEMENT si on a une tête MMSE.
        # Sinon le hook stocke ~1.5 GB d'activations fp32 en VRAM par sample
        # (1274 tokens × 2560 hidden × float32) → cumul → OOM.
        last_hidden_state: List[Optional[torch.Tensor]] = [None]
        hook_handle = None

        if has_mmse_head and target_layer is not None:
            def _hook(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                # Garde en bf16 et seulement le token d'intérêt en fp32 plus tard
                last_hidden_state[0] = h.detach()

            hook_handle = target_layer.register_forward_hook(_hook)

        try:
            with torch.inference_mode():
                outputs = model(**batch_gpu, output_hidden_states=False)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        # ── Classification ──
        batch_logit_pos: List[Optional[int]] = (
            [None] * (labels.shape[0] if labels is not None else 0)
        )

        if labels is not None:
            for i in range(labels.shape[0]):
                # FIX : utilise true_lab retourné par find_answer_position
                # (anciennement dataset.df.iloc[sample_idx + i]["label"], qui se
                # désalignait dès qu'un sample était skippé)
                logit_pos, true_lab = find_answer_position(labels[i], cn_id, ad_id)
                batch_logit_pos[i] = logit_pos
                if logit_pos is None or true_lab is None:
                    # sample skippé → on N'AVANCE PAS dans les listes de retour,
                    # mais sample_idx avance plus bas pour la prochaine batch
                    continue

                cn_logit = float(outputs.logits[i, logit_pos, cn_id].cpu())
                ad_logit = float(outputs.logits[i, logit_pos, ad_id].cpu())
                exp_cn, exp_ad = np.exp(cn_logit), np.exp(ad_logit)
                ad_prob = exp_ad / (exp_cn + exp_ad)
                pred = 1 if ad_prob > 0.5 else 0

                all_preds.append(pred)
                all_probs.append(ad_prob)
                all_true_cls.append(int(true_lab))
                kept_indices.append(sample_idx + i)

        # ── Régression MMSE (si tête présente) ──
        if has_mmse_head and last_hidden_state[0] is not None:
            h = last_hidden_state[0]
            for i in range(h.shape[0] if h.dim() == 3 else 1):
                lp = batch_logit_pos[i] if i < len(batch_logit_pos) else None
                if lp is None:
                    # sample skippé en classif → skip aussi en MMSE pour
                    # préserver l'alignement avec all_preds/all_probs
                    continue
                if h.dim() == 3:
                    h_token = h[i, lp, :].unsqueeze(0).float()
                elif h.dim() == 2:
                    h_token = h[lp, :].unsqueeze(0).float()
                else:
                    continue

                model.mmse_head.to(h_token.device)
                # Prédiction : sigmoid donne [0,1], × 30 → échelle clinique [0,30]
                mmse_norm = torch.sigmoid(model.mmse_head(h_token).squeeze(-1))
                mmse_pts = float((mmse_norm * 30.0).cpu().item())
                all_mmse_pred.append(mmse_pts)

                # Cible : le dataset stocke mmse_target = mmse_val/30 ∈ [0,1]
                # (cohérent avec sigmoid côté training, où la MSE travaille
                # dans l'espace normalisé). On dénormalise ICI pour matcher
                # l'échelle clinique de mmse_pts.
                # Sans cette dénormalisation, MAE/RMSE/R² compareraient
                # des échelles incompatibles ([0,30] vs [0,1]) — bug critique.
                # CC reste invariant à l'échelle, donc valide dans les deux cas.
                if mmse_true_batch is not None:
                    mmse_true_norm = float(mmse_true_batch[i].cpu().item())
                    all_mmse_true.append(mmse_true_norm * 30.0)
                else:
                    all_mmse_true.append(float("nan"))

                if "has_real_measures" in dataset.df.columns:
                    all_has_real.append(
                        bool(dataset.df.iloc[sample_idx + i]["has_real_measures"])
                    )
                else:
                    all_has_real.append(False)

        sample_idx += labels.shape[0] if labels is not None else 1

        # Libération VRAM aggressive après chaque batch
        # (outputs.logits ~= 50-100 MB par sample en fp32, hidden_state ~= 1.5 GB)
        del outputs, batch_gpu, last_hidden_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Métriques classification ──
    results: Dict[str, float] = {}
    true_arr = np.asarray(all_true_cls)
    preds_arr = np.asarray(all_preds)
    probs_arr = np.asarray(all_probs)

    if len(true_arr) == 0:
        print("[!] evaluate_dataset : aucune prédiction collectée")
        return results, all_preds, all_probs, all_true_cls, all_mmse_pred, all_mmse_true

    results["accuracy"] = float(accuracy_score(true_arr, preds_arr))
    results["f1"] = float(f1_score(true_arr, preds_arr, zero_division=0))
    results["sensitivity"] = float(
        recall_score(true_arr, preds_arr, pos_label=1, zero_division=0)
    )
    results["specificity"] = float(
        recall_score(true_arr, preds_arr, pos_label=0, zero_division=0)
    )

    if len(np.unique(true_arr)) < 2:
        results["auc"] = 0.5
    else:
        results["auc"] = float(roc_auc_score(true_arr, probs_arr))

        # Seuil optimal (Youden J)
        fpr, tpr, thresholds = roc_curve(true_arr, probs_arr)
        opt_idx = int(np.argmax(tpr - fpr))
        opt_thr = float(thresholds[opt_idx])
        preds_opt = (probs_arr >= opt_thr).astype(int)
        results["optimal_threshold"] = opt_thr
        results["f1_calibrated"] = float(f1_score(true_arr, preds_opt, zero_division=0))
        results["sensitivity_calibrated"] = float(
            recall_score(true_arr, preds_opt, pos_label=1, zero_division=0)
        )
        results["specificity_calibrated"] = float(
            recall_score(true_arr, preds_opt, pos_label=0, zero_division=0)
        )

    # ── Métriques MMSE ──
    if all_mmse_pred:
        mmse_metrics = compute_mmse_metrics(
            np.asarray(all_mmse_pred),
            np.asarray(all_mmse_true),
            np.asarray(all_has_real),
        )
        results.update(mmse_metrics)

    if return_indices:
        return (results, all_preds, all_probs, all_true_cls,
                all_mmse_pred, all_mmse_true, kept_indices)
    return results, all_preds, all_probs, all_true_cls, all_mmse_pred, all_mmse_true


# ═══════════════════════════════════════════════════════════════════════════
# 8. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_roc_curve(
    y_true: np.ndarray | List[int],
    y_prob: np.ndarray | List[float],
    out_path: str | Path,
    title: Optional[str] = None,
    extra_caption: Optional[str] = None,
) -> Path:
    """Courbe ROC standard avec AUC affiché. Retourne le chemin sauvegardé."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = sklearn_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2C7BB6", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2C7BB6")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("FPR (1 − Spécificité)", fontsize=10)
    ax.set_ylabel("TPR (Sensibilité)", fontsize=10)
    ttl = title or "Courbe ROC"
    if extra_caption:
        ttl += f"\n{extra_caption}"
    ax.set_title(ttl, fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    y_true: np.ndarray | List[int],
    y_pred: np.ndarray | List[int],
    out_path: str | Path,
    labels: Tuple[str, str] = ("CN", "AD"),
    title: Optional[str] = None,
) -> Path:
    """Matrice de confusion 2×2 avec annotations. Retourne le chemin sauvegardé."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")

    # Annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    ax.set_title(title or "Matrice de confusion")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_curves(
    history: List[Dict[str, Any]],
    out_path: str | Path,
    title: Optional[str] = None,
) -> Path:
    """
    history : list des dicts par époque (ce qu'EvalCallback.history accumule).
    Trace train loss (si dispo) + val AUC + val loss sur 2 panels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    val_auc = [h.get("auc", float("nan")) for h in history]
    val_loss = [h.get("val_loss", float("nan")) for h in history]
    val_f1 = [h.get("f1", float("nan")) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(epochs, val_auc, "o-", label="Val AUC", color="#2C7BB6", lw=2)
    ax1.plot(epochs, val_f1, "s--", label="Val F1", color="#D7191C", lw=1.5, alpha=0.7)
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Score")
    ax1.set_ylim([0, 1.02])
    ax1.set_title("Métriques validation")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    if any(not np.isnan(v) for v in val_loss):
        ax2.plot(epochs, val_loss, "o-", color="#FDAE61", lw=2, label="Val loss (BCE)")
        ax2.set_xlabel("Époque")
        ax2.set_ylabel("Loss")
        ax2.set_title("Val loss")
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "val_loss non disponible",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mmse_scatter(
    y_true: np.ndarray | List[float],
    y_pred: np.ndarray | List[float],
    out_path: str | Path,
    real_mask: Optional[np.ndarray | List[bool]] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Nuage de points MMSE prédit vs MMSE vrai. Si `real_mask` fourni, distingue
    les vrais scores ADNI (cercles pleins) des imputés (croix grises).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    if real_mask is not None:
        real_mask = np.asarray(real_mask, dtype=bool)
        if (~real_mask).any():
            ax.scatter(
                y_true[~real_mask], y_pred[~real_mask],
                marker="x", color="#888888", alpha=0.4, s=20, label="Imputé",
            )
        if real_mask.any():
            ax.scatter(
                y_true[real_mask], y_pred[real_mask],
                marker="o", color="#2C7BB6", alpha=0.6, s=28, label="ADNI réel",
            )
            mae = float(mean_absolute_error(y_true[real_mask], y_pred[real_mask]))
            ax.text(
                0.05, 0.95, f"MAE (ADNI) = {mae:.2f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    else:
        ax.scatter(y_true, y_pred, alpha=0.5, s=24, color="#2C7BB6")

    # Diagonale y=x
    lo, hi = 0, 30
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)

    ax.set_xlim([lo, hi])
    ax.set_ylim([lo, hi])
    ax.set_xlabel("MMSE vrai")
    ax.set_ylabel("MMSE prédit")
    ax.set_title(title or "MMSE prédit vs vrai")
    if real_mask is not None:
        ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 9. EVAL CALLBACK (custom — remplace pipeline HF cassé)
# ═══════════════════════════════════════════════════════════════════════════

class EvalCallback(TrainerCallback):
    """
    Évaluation custom + early stopping + monitoring WandB en fin d'époque.

    Pourquoi ce callback custom :
        Le Trainer HF ne passe pas correctement `labels` à
        preprocess_logits_for_metrics pour les modèles ImageTextToText
        → compute_metrics reçoit des métriques vides → AUC=0.0 bidon.
        On contourne en évaluant nous-mêmes via evaluate_dataset() qui
        marche parfaitement en standalone.

    Fonctionnement :
        1. Évalue via evaluate_dataset()
        2. Calcule la val loss (BCE sur P(AD)) — détection précoce overfitting
        3. Génère et sauve la courbe ROC (+ upload WandB)
        4. Logge toutes les métriques sur WandB
        5. Sauvegarde best_model si AUC ≥ best_auc + min_delta
        6. Early stopping si patience épuisée

    Args:
        val_dataset    : dataset de validation
        collate_fn     : fonction de collate (TFE custom)
        cn_id, ad_id   : token IDs
        output_dir     : où sauver le best_model
        best_name      : nom du sous-dossier best_model (par tâche)
        patience       : nombre d'époques sans amélioration avant arrêt
        min_delta      : amélioration minimale pour reset patience
        use_wandb      : si False, skippe les wandb.log (sécurité)
    """

    def __init__(
        self,
        val_dataset,
        collate_fn: Callable,
        cn_id: int,
        ad_id: int,
        output_dir: str,
        best_name: str = "best_model",
        patience: int = 2,
        min_delta: float = 0.001,
        use_wandb: bool = True,
        processor=None,
        metric_name: str = "auc",      # "auc" | "f1" | "sens" | "spec"
    ):
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.cn_id = cn_id
        self.ad_id = ad_id
        self.output_dir = output_dir
        self.best_name = best_name
        self.best_path = os.path.join(output_dir, best_name)
        self.metric_name = metric_name
        self.best_metric = -1.0          # générique : remplace best_auc
        self.best_auc = -1.0             # gardé pour compat logs/wandb
        self.history: List[Dict[str, Any]] = []
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch: Optional[float] = None
        self.use_wandb = use_wandb and HAS_WANDB
        self.roc_dir = Path(output_dir) / "roc_curves"
        self.roc_dir.mkdir(parents=True, exist_ok=True)
        self.processor = processor  # pour sauvegarde tokenizer dans best_model/

    def on_epoch_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> TrainerControl:
        if model is None:
            return control

        epoch = state.epoch
        print(f"\n{'='*70}", flush=True)
        print(f"[EVAL CUSTOM] Fin époque {epoch:.2f}", flush=True)
        print(f"{'='*70}", flush=True)

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        # Libération VRAM agressive AVANT éval :
        # 1. Zero_grad sur tous les params PEFT (libère gradients accumulés)
        # 2. gc.collect + empty_cache pour défragmenter
        # Sans ça : OOM sur RTX 4080 16GB car training laisse ~5 GB de gradients
        # et l'éval forward demande ~270 MB d'allocation contiguë.
        try:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
            print(f"[EVAL] VRAM libre : {free_gb:.1f} GB (après cleanup)", flush=True)

        try:
            results, preds, probs, true_cls, mmse_pred, mmse_true = evaluate_dataset(
                model, self.val_dataset, self.collate_fn,
                self.cn_id, self.ad_id, batch_size=1, device=device,
            )

            auc = results.get("auc", 0.0)
            f1 = results.get("f1", 0.0)
            acc = results.get("accuracy", 0.0)
            sens = results.get("sensitivity", 0.0)
            spec = results.get("specificity", 0.0)

            print(f"\n📊 CLASSIFICATION :", flush=True)
            print(f"  ACC={acc:.4f} | AUC={auc:.4f} | F1={f1:.4f}", flush=True)
            print(f"  Sens={sens:.4f} | Spec={spec:.4f}", flush=True)

            # MMSE (si tête présente)
            mae_mmse = results.get("mae_real", float("nan"))
            rmse_mmse = results.get("rmse_real", float("nan"))
            cc_mmse = results.get("cc_real", float("nan"))
            n_real = results.get("n_real", 0)

            if not np.isnan(mae_mmse) and n_real > 0:
                print(f"\n🧠 MMSE (réel, n={n_real}) :", flush=True)
                print(
                    f"  MAE={mae_mmse:.2f} | RMSE={rmse_mmse:.2f} | CC={cc_mmse:.4f}",
                    flush=True,
                )

            # Val loss BCE
            probs_t = torch.tensor(probs, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
            labels_t = torch.tensor(true_cls, dtype=torch.float32)
            val_loss = F.binary_cross_entropy(probs_t, labels_t).item()
            print(f"\n📉 Val loss (BCE) : {val_loss:.4f}", flush=True)

            # Courbe ROC
            roc_img_path: Optional[Path] = None
            try:
                roc_img_path = self.roc_dir / f"roc_epoch_{epoch:.0f}.png"
                plot_roc_curve(
                    true_cls, probs, roc_img_path,
                    title=f"ROC — Époque {epoch:.0f}",
                    extra_caption=f"Sens={sens:.3f} | Spec={spec:.3f} | val_loss={val_loss:.4f}",
                )
            except Exception as e_roc:
                print(f"[!] Courbe ROC échouée : {e_roc}", flush=True)
                roc_img_path = None

            # Historique
            epoch_metrics: Dict[str, Any] = {
                "epoch":    float(epoch),
                "step":     int(state.global_step),
                "auc":      auc,
                "f1":       f1,
                "accuracy": acc,
                "sens":     sens,
                "spec":     spec,
                "val_loss": val_loss,
                "es_wait":  self.wait,
            }
            if not np.isnan(mae_mmse):
                epoch_metrics.update({
                    "mmse_mae":  mae_mmse,
                    "mmse_rmse": rmse_mmse,
                    "mmse_cc":   cc_mmse,
                    "mmse_n_real": n_real,
                })
            self.history.append(epoch_metrics)

            # Log WandB
            if self.use_wandb:
                wandb_metrics = {
                    "eval/accuracy":          acc,
                    "eval/auc":               auc,
                    "eval/f1":                f1,
                    "eval/sensitivity":       sens,
                    "eval/specificity":       spec,
                    "eval/val_loss":          val_loss,
                    "eval/epoch":             epoch,
                    "early_stopping/wait":      self.wait,
                    "early_stopping/patience":  self.patience,
                    "early_stopping/best_auc":  max(self.best_auc, 0.0),
                }
                if not np.isnan(mae_mmse):
                    wandb_metrics["eval/mmse_mae"] = mae_mmse
                    wandb_metrics["eval/mmse_rmse"] = rmse_mmse
                    wandb_metrics["eval/mmse_cc"] = cc_mmse
                if roc_img_path is not None and roc_img_path.exists():
                    wandb_metrics["eval/roc_curve"] = wandb.Image(str(roc_img_path))
                try:
                    wandb.log(wandb_metrics, step=state.global_step)
                except Exception as e:
                    print(f"[!] WandB log échoué : {e}", flush=True)

            # Récupère la métrique de monitoring (auc, f1, sens, spec)
            current_metric = epoch_metrics.get(self.metric_name, auc)

            # Sauvegarde best_model + early stopping
            if current_metric >= self.best_metric + self.min_delta:
                improvement = current_metric - max(self.best_metric, 0.0)
                print(f"\n[✓] NOUVEAU MEILLEUR {self.metric_name.upper()} : "
                      f"{current_metric:.4f} (+{improvement:.4f})", flush=True)
                self.best_metric = current_metric
                self.best_auc = auc            # garder synchronisé pour logs
                self.wait = 0
                try:
                    model.save_pretrained(self.best_path)
                    # Sauve aussi le processor pour pouvoir reload best_model
                    # sans dépendre du base model HF (drift de version possible)
                    if self.processor is not None:
                        try:
                            self.processor.save_pretrained(self.best_path)
                        except Exception as e:
                            print(f"[!] Processor save échoué : {e}", flush=True)
                    # Inclut les token IDs CN/AD utilisés à l'entraînement
                    # → permet à evaluate.py d'asserter la cohérence au reload
                    epoch_metrics_with_ids = dict(epoch_metrics)
                    epoch_metrics_with_ids["cn_token_id"] = int(self.cn_id)
                    epoch_metrics_with_ids["ad_token_id"] = int(self.ad_id)
                    save_metrics_json(
                        epoch_metrics_with_ids,
                        os.path.join(self.best_path, "eval_metrics.json"),
                    )
                    save_metrics_json(
                        self.history,
                        os.path.join(self.best_path, "eval_history.json"),
                    )
                    print(f"[✓] Sauvegardé : {self.best_path}", flush=True)
                except Exception as e:
                    print(f"[!] Sauvegarde échouée : {e}", flush=True)
            else:
                self.wait += 1
                print(
                    f"\n  [~] Pas d'amélioration — patience : {self.wait}/{self.patience}",
                    flush=True,
                )

            if self.wait >= self.patience:
                self.stopped_epoch = float(epoch)
                print(f"\n{'!'*70}", flush=True)
                print(f"  EARLY STOPPING — époque {epoch:.0f}", flush=True)
                print(f"  Meilleur AUC : {self.best_auc:.4f}", flush=True)
                print(f"{'!'*70}\n", flush=True)
                if self.use_wandb:
                    try:
                        wandb.log({
                            "early_stopping/triggered":      True,
                            "early_stopping/stopped_epoch":  epoch,
                            "early_stopping/final_best_auc": self.best_auc,
                        }, step=state.global_step)
                    except Exception:
                        pass
                control.should_training_stop = True

        except Exception as e:
            print(f"\n[!] EvalCallback CRASH : {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()

        finally:
            if was_training:
                model.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{'='*70}\n", flush=True)

        return control


# ═══════════════════════════════════════════════════════════════════════════
# 10. SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════

def _json_safe(obj: Any) -> Any:
    """Convertit récursivement en types JSON-sérialisables (numpy → Python)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_metrics_json(metrics: Any, path: str | Path) -> None:
    """Écrit un dict (ou liste de dicts) en JSON, en gérant les types numpy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(metrics), f, indent=2, ensure_ascii=False)