"""
trainers.py — Module partagé : Trainers HuggingFace custom + run_training().

Centralise pour T1-T5 (et tâches futures) :
    - TfeMedGemmaCls         : wrapper modèle SANS tête MMSE      (T1, T5)
    - TfeMedGemmaWithMMSE    : wrapper modèle AVEC tête MMSE      (T2, T3, T4)
    - TfeClsTrainer          : Trainer HF, Focal Loss seule       (T1, T5)
    - TfeMultitaskTrainer    : Trainer HF, Focal + MSE pondérée   (T2, T3, T4)
    - run_training()         : point d'entrée unifié

Auto-sélection (dans run_training) selon config["mmse_head"]["enabled"] :
    - True  → TfeMedGemmaWithMMSE + TfeMultitaskTrainer
    - False → TfeMedGemmaCls + TfeClsTrainer

Usage depuis chaque tâche (0X_*/train.py) :
    from trainers import run_training
    run_training(config_path=Path(__file__).parent / "config.yaml")

Points critiques préservés à l'identique :
    - Hook MMSE SANS .detach() (gradient MSE remonte vers PEFT)
    - logit_pos = ans_pos - 1 (convention autoregressive)
    - safe_serialization=False (Gemma 3 partage embed_tokens / lm_head)
    - eval_strategy="no" + EvalCallback custom
    - _load_from_checkpoint surchargé (PEFT, pas de pytorch_model.bin)
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import wandb
from peft import (
    LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from utils import (
    ClearCacheCallback, EvalCallback, FocalLoss, MMSEHead,
    check_vram, get_last_checkpoint, is_valid_checkpoint,
    load_config, load_mmse_head, register_signal_refs, release_gpu,
    save_metrics_json, set_token_ids, setup_env, signal_handler,
)
from dataset import TfeDataset, tfe_collate_fn


# ═══════════════════════════════════════════════════════════════════════════
# 1. WRAPPERS DE MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

class TfeMedGemmaCls(nn.Module):
    """
    Wrapper minimal autour du peft_model — SANS tête MMSE (T1, T5).

    Surcharge save_pretrained pour forcer safe_serialization=False
    (Gemma 3 partage embed_tokens / lm_head → safetensors crash sinon).
    """

    def __init__(self, peft_model):
        super().__init__()
        self.model = peft_model

    def forward(self, **kwargs):
        kwargs["use_cache"] = False  # DynamicCache crash le padding Accelerate
        return self.model(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.model.enable_input_require_grads()

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.model, "is_gradient_checkpointing", False)

    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=False, **kwargs)

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return next(self.model.parameters()).device


class TfeMedGemmaWithMMSE(nn.Module):
    """
    Wrapper MedGemma + tête MMSE — AVEC tête (T2, T3, T4).

    Architecture validée (best_model_step2, MAE=1.78 pts) :
        peft_model + MMSEHead(LayerNorm(2560) + Linear(2560, 1), fp32)

    Sauvegarde :
        - adapter_model* : poids LoRA  (peft_model.save_pretrained)
        - mmse_head.pt   : poids tête  (torch.save séparé)
        - safe_serialization=False OBLIGATOIRE (Gemma 3 shared tensors)
    """

    def __init__(self, peft_model, hidden_size: int):
        super().__init__()
        self.model = peft_model
        self.mmse_head = MMSEHead(hidden_size)

    def forward(self, **kwargs):
        kwargs["use_cache"] = False
        return self.model(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.model.enable_input_require_grads()

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.model, "is_gradient_checkpointing", False)

    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=False, **kwargs)
        torch.save(self.mmse_head.state_dict(), os.path.join(path, "mmse_head.pt"))

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return next(self.model.parameters()).device


# ═══════════════════════════════════════════════════════════════════════════
# 2. TRAINERS
# ═══════════════════════════════════════════════════════════════════════════

def _find_text_decoder_last_layer(model: nn.Module) -> nn.Module:
    """
    Trouve EXPLICITEMENT la dernière couche du décodeur texte (Gemma).

    L'ancienne approche "dernière ModuleList rencontrée" était fragile : avec
    MedGemma il y a au moins 3 ModuleList (vision encoder, multimodal projector,
    text decoder), et l'ordre d'itération `named_modules()` n'est pas garanti.

    On cherche par préfixe de nom — ordre de priorité :
        1. ...language_model.model.layers          (MedGemma 1.5 / Gemma 3)
        2. ...language_model.layers
        3. ...text_model.layers
        4. fallback : dernière ModuleList nommée 'layers'
    """
    candidates: list = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        # On veut le décodeur texte, pas le vision encoder
        if "vision" in name.lower():
            continue
        if name.endswith("layers") or "language_model" in name or "text_model" in name:
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError(
            "Aucune ModuleList de décodeur texte trouvée. "
            "Vérifier la structure du modèle (MedGemma a-t-il changé ?)"
        )

    # Préférer 'language_model' > 'text_model' > 'layers' générique
    for name, module in candidates:
        if "language_model" in name:
            print(f"[*] Hidden state hook → {name}[-1] "
                  f"(class: {module[-1].__class__.__name__})", flush=True)
            return module[-1]
    for name, module in candidates:
        if "text_model" in name:
            print(f"[*] Hidden state hook → {name}[-1]", flush=True)
            return module[-1]

    # Fallback : prendre la plus longue (le décodeur texte a typiquement le plus
    # de couches que le vision encoder)
    name, module = max(candidates, key=lambda x: len(x[1]))
    print(f"[*] Hidden state hook → {name}[-1] "
          f"(fallback : {len(module)} couches, "
          f"class: {module[-1].__class__.__name__})", flush=True)
    return module[-1]


def _verify_verbalizer_tokens(processor, cn_id: int, ad_id: int) -> None:
    """
    Vérifie que les IDs CN/AD sont cohérents avec le tokenizer dans le contexte
    du chat template.

    Encode les variantes "CN", " CN", "\\nCN" et confirme que cn_id apparaît
    dans la liste. Idem pour AD. Émet un AVERTISSEMENT (pas un crash) si
    aucune des variantes n'a un token unique correspondant — le fallback
    "dernier token non -100" du compute_loss prendra alors le relais, mais
    il faut le savoir.
    """
    tokenizer = processor.tokenizer
    issues = []
    for label, tok_id in [("CN", cn_id), ("AD", ad_id)]:
        variants_ids = {
            v: tokenizer.encode(v, add_special_tokens=False)
            for v in [label, f" {label}", f"\n{label}"]
        }
        # Cherche si tok_id apparaît comme dernier token de l'une des variantes
        found = any(ids and ids[-1] == tok_id for ids in variants_ids.values())
        if not found:
            issues.append((label, tok_id, variants_ids))

    if issues:
        print("[!] AVERTISSEMENT : verbalizer ID mismatch potentiel", flush=True)
        for label, tok_id, variants in issues:
            print(f"    {label} (id={tok_id}) non retrouvé dans variants : "
                  f"{variants}", flush=True)
        print("    Le fallback 'dernier token non -100' prendra le relais "
              "dans compute_loss.", flush=True)
    else:
        print(f"[✓] Verbalizer cohérent : CN={cn_id}, AD={ad_id}", flush=True)


class _BaseTfeTrainer(Trainer):
    """
    Méthodes communes aux deux Trainers (load checkpoint PEFT, save model).
    Évite la duplication entre TfeClsTrainer et TfeMultitaskTrainer.
    """

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Reprise PEFT propre : optimizer + scheduler + state."""
        print(f"[*] Reprise PEFT depuis : {resume_from_checkpoint}", flush=True)
        for fname, attr, label in [
            ("optimizer.pt", "optimizer",    "Optimizer"),
            ("scheduler.pt", "lr_scheduler", "Scheduler"),
        ]:
            fpath = os.path.join(resume_from_checkpoint, fname)
            obj = getattr(self, attr.split(".")[0], None)
            if os.path.exists(fpath) and obj is not None:
                try:
                    obj.load_state_dict(
                        torch.load(fpath, map_location="cpu", weights_only=False)
                    )
                    print(f"[*] {label} rechargé")
                except Exception as e:
                    print(f"[!] {label} non rechargé : {e}")
        state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(state_path):
            try:
                self.state = self.state.load_from_json(state_path)
                print(f"[*] Trainer state rechargé — step : {self.state.global_step}")
            except Exception as e:
                print(f"[!] Trainer state non rechargé : {e}")

    def save_model(self, output_dir=None, _internal_call=False):
        """Force safe_serialization=False (Gemma 3 shared tensors)."""
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)  # délégué au wrapper
        proc = (
            getattr(self, "tokenizer", None)
            or getattr(self, "processing_class", None)
        )
        if proc is not None:
            proc.save_pretrained(output_dir)


class TfeClsTrainer(_BaseTfeTrainer):
    """
    Trainer pour classification CN/AD seule (T1, T5).
    Loss : Focal Loss sur tokens CN/AD (verbalizer).
    """

    def __init__(self, *args,
                 focal_alpha_ad: float = 0.78,
                 focal_gamma: float = 2.0,
                 cn_token_id: int = None,
                 ad_token_id: int = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(alpha_ad=focal_alpha_ad, gamma=focal_gamma)
        self.cn_token_id = cn_token_id
        self.ad_token_id = ad_token_id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # MMSE keys ignorées (le collate les ajoute toujours)
        inputs.pop("mmse_score", None)
        inputs.pop("regression_weight", None)

        if "token_type_ids" not in inputs and "input_ids" in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        # pixel_values : 5D → 4D (n_views collapsé sur batch)
        if "pixel_values" in inputs and inputs["pixel_values"].ndim == 5:
            b, n, c, h, w = inputs["pixel_values"].shape
            inputs["pixel_values"] = inputs["pixel_values"].view(b * n, c, h, w)
        if "pixel_attention_mask" in inputs and inputs["pixel_attention_mask"].ndim == 4:
            b, n, h, w = inputs["pixel_attention_mask"].shape
            inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"].view(b * n, h, w)

        labels = inputs.get("labels", None)
        outputs = model(**inputs, output_hidden_states=False)
        loss_cls = outputs.loss

        if labels is not None and self.cn_token_id is not None:
            cls_logits_list, cls_true_list = [], []
            for i in range(labels.shape[0]):
                logit_pos = None
                true_val = None
                for pos in range(labels.shape[1]):
                    tok = labels[i, pos].item()
                    if tok == self.cn_token_id or tok == self.ad_token_id:
                        logit_pos = max(0, pos - 1)
                        true_val = 1 if tok == self.ad_token_id else 0
                        break
                if logit_pos is None:
                    valid = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if len(valid) == 0:
                        continue
                    ans_pos = valid[-1].item()
                    logit_pos = max(0, ans_pos - 1)
                    ans_tok = labels[i, ans_pos].item()
                    true_val = 1 if ans_tok == self.ad_token_id else 0

                cn_logit = outputs.logits[i, logit_pos, self.cn_token_id]
                ad_logit = outputs.logits[i, logit_pos, self.ad_token_id]
                cls_logits_list.append(torch.stack([cn_logit, ad_logit]))
                cls_true_list.append(true_val)

            if cls_logits_list:
                cls_logits = torch.stack(cls_logits_list)
                cls_true = torch.tensor(cls_true_list, device=model.device)
                loss_cls = self.focal_loss_fn(cls_logits, cls_true)

        return (loss_cls, outputs) if return_outputs else loss_cls


class TfeMultitaskTrainer(_BaseTfeTrainer):
    """
    Trainer multitâche CN/AD + régression MMSE (T2, T3, T4).

    compute_loss :
        1. Hook hidden state SANS .detach() (gradient MSE → PEFT)
        2. Focal Loss sur tokens CN/AD au logit_pos
        3. MSE sur MMSE normalisé [0,1], pondérée par reg_weight
        4. Loss = Focal + α_reg × MSE
    """

    def __init__(self, *args,
                 focal_alpha_ad: float = 0.78,
                 focal_gamma: float = 2.0,
                 alpha_reg: float = 1.0,
                 cn_token_id: int = None,
                 ad_token_id: int = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(alpha_ad=focal_alpha_ad, gamma=focal_gamma)
        self.mse_loss_fn = nn.MSELoss(reduction="none")
        self.alpha_reg = alpha_reg
        self.cn_token_id = cn_token_id
        self.ad_token_id = ad_token_id
        # Cache de la couche cible pour éviter le re-scan à chaque step
        self._target_layer = None

    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """Lazy-init du target_layer (a besoin du modèle, donc pas dans __init__)."""
        if self._target_layer is None:
            self._target_layer = _find_text_decoder_last_layer(model)
        return self._target_layer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mmse_true = inputs.pop("mmse_score", None)
        reg_weight = inputs.pop("regression_weight", None)

        if "token_type_ids" not in inputs and "input_ids" in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        if "pixel_values" in inputs and inputs["pixel_values"].ndim == 5:
            b, n, c, h, w = inputs["pixel_values"].shape
            inputs["pixel_values"] = inputs["pixel_values"].view(b * n, c, h, w)
        if "pixel_attention_mask" in inputs and inputs["pixel_attention_mask"].ndim == 4:
            b, n, h, w = inputs["pixel_attention_mask"].shape
            inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"].view(b * n, h, w)

        labels = inputs.get("labels", None)

        # ── Hook hidden state — SANS .detach() ───────────────────────────
        # ⚠️ PAS de .detach() : gradient MSE doit remonter vers PEFT
        last_hidden_state = [None]

        def _hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            last_hidden_state[0] = h.float()

        target_layer = self._get_target_layer(model)
        hook_handle = target_layer.register_forward_hook(_hook)

        try:
            outputs = model(**inputs, output_hidden_states=False)
        finally:
            hook_handle.remove()

        # ── Focal Loss CN/AD ──────────────────────────────────────────────
        loss_cls = outputs.loss
        batch_logit_pos = [None] * (labels.shape[0] if labels is not None else 0)

        if labels is not None and self.cn_token_id is not None:
            cls_logits_list, cls_true_list = [], []
            for i in range(labels.shape[0]):
                logit_pos = None
                true_val = None
                for pos in range(labels.shape[1]):
                    tok = labels[i, pos].item()
                    if tok == self.cn_token_id or tok == self.ad_token_id:
                        logit_pos = max(0, pos - 1)
                        true_val = 1 if tok == self.ad_token_id else 0
                        break
                if logit_pos is None:
                    valid = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if len(valid) == 0:
                        continue
                    ans_pos = valid[-1].item()
                    logit_pos = max(0, ans_pos - 1)
                    ans_tok = labels[i, ans_pos].item()
                    true_val = 1 if ans_tok == self.ad_token_id else 0

                batch_logit_pos[i] = logit_pos
                cn_l = outputs.logits[i, logit_pos, self.cn_token_id]
                ad_l = outputs.logits[i, logit_pos, self.ad_token_id]
                cls_logits_list.append(torch.stack([cn_l, ad_l]))
                cls_true_list.append(true_val)

            if cls_logits_list:
                cls_logits = torch.stack(cls_logits_list)
                cls_true = torch.tensor(cls_true_list, device=model.device)
                loss_cls = self.focal_loss_fn(cls_logits, cls_true)

        # ── MSE MMSE au logit_pos ────────────────────────────────────────
        loss_reg = torch.tensor(0.0, device=model.device)

        if mmse_true is not None and last_hidden_state[0] is not None:
            h = last_hidden_state[0]
            model.mmse_head = model.mmse_head.to(h.device)

            if h.dim() == 2:
                last_h = h[-1, :].unsqueeze(0)
                mmse_pred = torch.sigmoid(model.mmse_head(last_h).squeeze(-1))
            else:
                mmse_preds = []
                for i in range(h.shape[0]):
                    lp = batch_logit_pos[i] if i < len(batch_logit_pos) else None
                    if lp is None:
                        lp = h.shape[1] - 1
                    lh = h[i, lp, :].unsqueeze(0)
                    mmse_preds.append(
                        torch.sigmoid(model.mmse_head(lh).squeeze(-1))
                    )
                mmse_pred = torch.cat(mmse_preds, dim=0)

            mse_raw = self.mse_loss_fn(
                mmse_pred, mmse_true.to(model.device).float()
            )
            eff_weight = (
                reg_weight.to(model.device)
                if reg_weight is not None
                else torch.ones_like(mse_raw)
            )
            # Normalisation sum/sum : divise par la somme des poids effectifs
            # (= nombre de vrais MMSE dans le batch), pas par N (taille batch).
            # Formule thèse : L_reg = Σ(MSE_i · w_i) / max(1, Σw_i)
            #
            # Justification : avec ~85% des MMSE manquants (NACC + OASIS) et
            # w_reg=0 pour ces samples, .mean() diluerait le gradient par ~5-7×
            # selon le mix du batch. sum/sum garantit que chaque vraie mesure
            # exerce une contribution complète et constante à la loss,
            # indépendamment de la proportion de données manquantes du batch.
            #
            # Cas DEFT : tous les sujets ont MMSE (ADNI seul) → Σw_i = N
            # → .mean() ≡ sum/sum. Notre formulation est une généralisation
            # de DEFT au contexte multi-cohorte avec MMSE rares.
            weight_sum = eff_weight.sum().clamp(min=1.0)
            loss_reg = (mse_raw * eff_weight).sum() / weight_sum

        total_loss = loss_cls + self.alpha_reg * loss_reg

        # Logging séparé des composantes (utile pour diagnostiquer T2 vs T6)
        # Frequence : alignée sur self.args.logging_steps via state.global_step
        if self.state.global_step % self.args.logging_steps == 0:
            try:
                self.log({
                    "loss_cls": loss_cls.detach().item(),
                    "loss_reg": (loss_reg.detach().item()
                                 if isinstance(loss_reg, torch.Tensor)
                                 else float(loss_reg)),
                })
            except Exception:
                pass  # log non bloquant

        return (total_loss, outputs) if return_outputs else total_loss


# ═══════════════════════════════════════════════════════════════════════════
# 3. POINT D'ENTRÉE UNIFIÉ
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_path(path: Union[str, Path], project_root: Path) -> Path:
    """Résout un chemin relatif depuis project_root, conserve les chemins absolus."""
    p = Path(path)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def _build_quantization(config: dict) -> BitsAndBytesConfig:
    q = config["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if q["bnb_4bit_compute_dtype"] == "bfloat16"
            else torch.float16
        ),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def _load_base_and_lora(config: dict, last_checkpoint: Optional[str]):
    """Charge base_model + applique LoRA (from scratch ou depuis checkpoint)."""
    bnb_config = _build_quantization(config)

    base_model = AutoModelForImageTextToText.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    if last_checkpoint and is_valid_checkpoint(last_checkpoint):
        print(f"[*] Reprise LoRA depuis checkpoint...")
        peft_model = PeftModel.from_pretrained(
            base_model, last_checkpoint, is_trainable=True
        )
    else:
        print(f"[*] LoRA from scratch...")
        lora = config["lora"]
        peft_config = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["lora_alpha"],
            target_modules=lora["target_modules"],
            lora_dropout=lora["lora_dropout"],
            task_type=lora["task_type"],
        )
        peft_model = get_peft_model(base_model, peft_config)

    peft_model.print_trainable_parameters()
    return base_model, peft_model


def _build_training_args(config: dict, out_dir: str) -> TrainingArguments:
    t = config["training"]
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_steps=t.get("warmup_steps", 100),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        bf16=t.get("bf16", True),
        tf32=t.get("tf32", True),
        logging_steps=t.get("logging_steps", 20),
        eval_strategy="no",  # OBLIGATOIRE — pipeline HF cassé pour VLMs
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 2),
        load_best_model_at_end=False,
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.get("optim", "paged_adamw_8bit"),
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        report_to="wandb",
        remove_unused_columns=False,
        seed=t.get("seed", 42),
    )


def run_training(
    config_path: Union[str, Path],
    resume_from: Optional[str] = None,
) -> None:
    """
    Point d'entrée unifié pour T1-T5 (et tâches futures avec même structure).

    Auto-sélection :
        config["mmse_head"]["enabled"] = True  → Multitask (Focal + MSE)
        config["mmse_head"]["enabled"] = False → Cls seul   (Focal)

    Args:
        config_path : chemin vers le YAML de la tâche
        resume_from : checkpoint à reprendre (override auto-detect)
    """
    config_path = Path(config_path).resolve()
    config = load_config(str(config_path))

    # Racine projet = parent du dossier de tâche (où vivent utils.py, dataset.py)
    project_root = config_path.parent.parent

    # ── Setup ────────────────────────────────────────────────────────────
    setup_env(seed=config["training"]["seed"])
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Résolution chemins
    out_dir = str(_resolve_path(config["training"]["output_dir"], project_root))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Output dir : {out_dir}")

    splits_dir = _resolve_path(config["data"]["splits_dir"], project_root)
    fold = config["data"]["fold"]
    fold_dir = splits_dir / f"fold_{fold}"
    print(f"[*] Splits     : {fold_dir}")
    if not fold_dir.exists():
        raise FileNotFoundError(
            f"Splits introuvables : {fold_dir}\n"
            f"Lance d'abord : cd 00_prepare_splits && python prepare_splits.py"
        )

    # ── Auto-detection mode (MMSE on/off) ────────────────────────────────
    use_mmse = bool(config.get("mmse_head", {}).get("enabled", False))
    print(f"[*] Mode       : {'multitâche (MMSE)' if use_mmse else 'classification seule'}")

    # ── Checkpoint auto-detect ───────────────────────────────────────────
    last_checkpoint = resume_from if resume_from else get_last_checkpoint(out_dir)
    if last_checkpoint:
        print(f"[*] Reprise depuis : {last_checkpoint}")
    else:
        print("[*] Démarrage from scratch")

    trainer = None
    model = None

    try:
        check_vram(min_gb=10.0)

        # ── WandB ────────────────────────────────────────────────────────
        log_cfg = config.get("logging", {})
        wandb.init(
            project=log_cfg.get("wandb_project", "tfe-alzheimer-medgemma"),
            name=log_cfg.get("run_name", config.get("task_name", "tfe_run")),
            config=config,
            resume="allow",
        )

        # ── Processor + token IDs ────────────────────────────────────────
        processor = AutoProcessor.from_pretrained(config["model"]["name"])
        cn_id = processor.tokenizer.encode("CN", add_special_tokens=False)[0]
        ad_id = processor.tokenizer.encode("AD", add_special_tokens=False)[0]
        print(f"[*] Token IDs → CN: {cn_id}, AD: {ad_id}")
        set_token_ids(cn_id, ad_id)
        _verify_verbalizer_tokens(processor, cn_id, ad_id)

        # ── Modèle (base + LoRA + wrapper) ───────────────────────────────
        base_model, peft_model = _load_base_and_lora(config, last_checkpoint)

        if use_mmse:
            hidden_size = base_model.config.text_config.hidden_size
            model = TfeMedGemmaWithMMSE(peft_model, hidden_size=hidden_size)
            print(f"[*] Tête MMSE : LayerNorm({hidden_size}) + Linear({hidden_size}→1) fp32")
            if last_checkpoint:
                model.mmse_head = load_mmse_head(last_checkpoint, hidden_size=hidden_size)
                model.mmse_head = model.mmse_head.to("cuda")
        else:
            model = TfeMedGemmaCls(peft_model)

        register_signal_refs(model=model)

        # ── Datasets ─────────────────────────────────────────────────────
        # Lecture du filtre cohorte (T8) si présent
        cohort_filter = config.get("data", {}).get("cohort_filter")
        if cohort_filter:
            print(f"[*] Cohort filter : {cohort_filter}")

        train_ds = TfeDataset(
            str(fold_dir / "train.csv"), processor, config,
            is_training=True, cohort_filter=cohort_filter,
        )
        val_ds = TfeDataset(
            str(fold_dir / "val.csv"), processor, config,
            is_training=True, cohort_filter=cohort_filter,
        )
        print(f"[*] Train : {len(train_ds)} | Val : {len(val_ds)}")

        # ── EvalCallback ─────────────────────────────────────────────────
        es = config.get("early_stopping", {})
        eval_callback = EvalCallback(
            val_dataset=val_ds,
            collate_fn=tfe_collate_fn,
            cn_id=cn_id,
            ad_id=ad_id,
            output_dir=out_dir,
            best_name=config.get("training", {}).get("best_model_name", "best_model"),
            patience=es.get("patience", 2),
            min_delta=es.get("min_delta", 0.001),
            use_wandb=True,
            processor=processor,
            metric_name=es.get("metric", "auc"),
        )

        # ── TrainingArguments ─────────────────────────────────────────────
        training_args = _build_training_args(config, out_dir)

        # ── Trainer (selon mode) ──────────────────────────────────────────
        loss_cfg = config.get("loss", {})
        if use_mmse:
            mmse_cfg = config.get("mmse_head", {})
            trainer = TfeMultitaskTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=tfe_collate_fn,
                callbacks=[ClearCacheCallback(), eval_callback],
                focal_alpha_ad=loss_cfg.get("focal_alpha", 0.78),
                focal_gamma=loss_cfg.get("focal_gamma", 2.0),
                alpha_reg=mmse_cfg.get("loss_weight", 1.0),
                cn_token_id=cn_id,
                ad_token_id=ad_id,
            )
        else:
            trainer = TfeClsTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=tfe_collate_fn,
                callbacks=[ClearCacheCallback(), eval_callback],
                focal_alpha_ad=loss_cfg.get("focal_alpha", 0.78),
                focal_gamma=loss_cfg.get("focal_gamma", 2.0),
                cn_token_id=cn_id,
                ad_token_id=ad_id,
            )
        register_signal_refs(trainer=trainer, model=model)

        # ── Récap ────────────────────────────────────────────────────────
        t = config["training"]
        print(f"\n[*] Configuration :")
        print(f"    task_name      = {config.get('task_name', 'unnamed')}")
        print(f"    focal_alpha_AD = {loss_cfg.get('focal_alpha', 0.78)}")
        print(f"    focal_gamma    = {loss_cfg.get('focal_gamma', 2.0)}")
        if use_mmse:
            print(f"    alpha_reg      = {config.get('mmse_head', {}).get('loss_weight', 1.0)}")
        print(f"    epochs max     = {t['num_train_epochs']}")
        print(f"    early_stop     = patience={es.get('patience', 2)}, "
              f"min_delta={es.get('min_delta', 0.001)}")

        # ── Entraînement ─────────────────────────────────────────────────
        trainer.train(resume_from_checkpoint=last_checkpoint)

        print(f"\n[✓] Entraînement terminé.")
        print(f"    Meilleur AUC : {eval_callback.best_auc:.4f}")
        print(f"    Meilleur modèle : {eval_callback.best_path}")
        if eval_callback.stopped_epoch:
            print(f"    Early stopping : époque {eval_callback.stopped_epoch:.0f}")

        save_metrics_json(
            eval_callback.history,
            os.path.join(out_dir, "training_history.json"),
        )

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"\n[!] OOM — VRAM épuisée : {e}")
            print("[!] RTX 4080 16GB + MedGemma 4B QLoRA : batch_size=1 est le minimum")
        else:
            print(f"\n[!] RuntimeError : {e}")
        raise

    except KeyboardInterrupt:
        print("\n[!] Interruption clavier.")

    except Exception as e:
        print(f"\n[!] Erreur inattendue : {type(e).__name__}: {e}")
        raise

    finally:
        print("\n[*] Clôture...")
        try:
            wandb.finish()
        except Exception:
            pass
        release_gpu(model=model, trainer=trainer)