"""
training_logger.py — Logging append-only pendant l'entraînement.

Persiste TOUS les logs HF Trainer dans un CSV simple, indépendant de l'EvalCallback.
Permet de reconstruire après-coup train/val curves, ROC, etc. même si l'eval crashe.

Usage :
    from training_logger import TrainingLoggerCallback

    logger = TrainingLoggerCallback(
        output_dir=cfg.training.output_dir,
        run_name=cfg.task_name,
    )
    trainer = Trainer(..., callbacks=[..., logger])

Outputs (sous output_dir/training_logs/) :
    - train_log.csv      : 1 ligne par log step (loss, lr, grad_norm, epoch, step)
    - eval_log.csv       : 1 ligne par fin d'epoch (auc, val_loss, sens, spec, mmse_*)
    - epoch_summary.csv  : 1 ligne par epoch (train_loss moyenné + métriques eval)
    - run_info.json      : métadonnées (date, config, token IDs, dataset sizes)

Robustesse :
    - Append-only (chaque ligne flushée immédiatement → survit au crash)
    - Pas d'imports lourds (matplotlib/sklearn non requis ici → générés post-hoc)
    - Format CSV stable (pas de pickle, pas de versioning)
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState


class TrainingLoggerCallback(TrainerCallback):
    """
    Logger append-only qui survit au crash. À ajouter en premier dans la liste
    de callbacks du Trainer pour qu'il logue avant tout autre callback.

    Args:
        output_dir   : dossier de la tâche (ex: results/01_no_mmse/)
        run_name     : nom de la tâche (ex: "01_no_mmse")
        run_metadata : dict optionnel sauvé dans run_info.json (config, token IDs...)
    """

    TRAIN_FIELDS = ["timestamp", "epoch", "step", "loss", "learning_rate",
                    "grad_norm", "loss_cls", "loss_reg"]
    EVAL_FIELDS = ["timestamp", "epoch", "step", "auc", "f1", "accuracy",
                   "sensitivity", "specificity", "val_loss",
                   "mmse_mae", "mmse_rmse", "mmse_cc", "mmse_n_real",
                   "best_auc_so_far", "es_wait"]

    def __init__(
        self,
        output_dir: str,
        run_name: str,
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = Path(output_dir) / "training_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.train_csv = self.log_dir / "train_log.csv"
        self.eval_csv = self.log_dir / "eval_log.csv"
        self.run_info = self.log_dir / "run_info.json"

        self._init_csv(self.train_csv, self.TRAIN_FIELDS)
        self._init_csv(self.eval_csv, self.EVAL_FIELDS)

        info = {
            "run_name": run_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            **(run_metadata or {}),
        }
        with open(self.run_info, "w") as f:
            json.dump(info, f, indent=2, default=str)

        self._best_auc = -1.0
        self._wait = 0

    def _init_csv(self, path: Path, fields: List[str]) -> None:
        """Crée le CSV avec header si absent. N'écrase PAS un fichier existant
        (permet la reprise depuis checkpoint sans perdre l'historique)."""
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def _append(self, path: Path, fields: List[str], row: Dict[str, Any]) -> None:
        """Append + flush immédiat (résiste au kill -9)."""
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(k, "") for k in fields])
            f.flush()
            os.fsync(f.fileno())

    # ───── HF callback hooks ─────

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs=None, **kwargs):
        """Appelé à chaque logging_step (HF appelle ici aussi à la fin)."""
        if logs is None:
            return control

        # On ne logue que les training logs ici (pas les eval logs HF)
        # → eval HF n'est jamais utilisé dans ce pipeline (eval_strategy="no")
        is_train_log = "loss" in logs and "eval_loss" not in logs
        if not is_train_log:
            return control

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": state.epoch,
            "step": state.global_step,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "loss_cls": logs.get("loss_cls"),  # multitâche T2-T4 only
            "loss_reg": logs.get("loss_reg"),
        }
        self._append(self.train_csv, self.TRAIN_FIELDS, row)
        return control

    def log_eval(self, epoch: float, step: int, results: Dict[str, Any],
                 val_loss: float, best_auc: float, wait: int) -> None:
        """
        À appeler depuis l'EvalCallback custom pour persister les métriques eval.
        Si l'EvalCallback crashe AVANT cet appel, on n'a pas l'eval — mais on a
        toujours train_log.csv complet pour reconstruire la courbe train.
        """
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "step": step,
            "auc": results.get("auc"),
            "f1": results.get("f1"),
            "accuracy": results.get("accuracy"),
            "sensitivity": results.get("sensitivity"),
            "specificity": results.get("specificity"),
            "val_loss": val_loss,
            "mmse_mae": results.get("mae_real"),
            "mmse_rmse": results.get("rmse_real"),
            "mmse_cc": results.get("cc_real"),
            "mmse_n_real": results.get("n_real"),
            "best_auc_so_far": best_auc,
            "es_wait": wait,
        }
        self._append(self.eval_csv, self.EVAL_FIELDS, row)

    def log_eval_predictions(
        self,
        epoch: float,
        probs: List[float],
        true_cls: List[int],
        mmse_pred: Optional[List[float]] = None,
        mmse_true: Optional[List[float]] = None,
    ) -> None:
        """
        Sauve les probas brutes + labels pour pouvoir regénérer ROC/MMSE-scatter
        post-hoc. Un fichier par epoch sous training_logs/predictions/.
        """
        pred_dir = self.log_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        path = pred_dir / f"epoch_{epoch:.0f}.csv"

        fields = ["index", "true_label", "ad_prob"]
        if mmse_pred is not None:
            fields += ["mmse_pred", "mmse_true"]

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for i in range(len(probs)):
                row = [i, true_cls[i], probs[i]]
                if mmse_pred is not None:
                    mp = mmse_pred[i] if i < len(mmse_pred) else ""
                    mt = mmse_true[i] if i < len(mmse_true) else ""
                    row += [mp, mt]
                w.writerow(row)
