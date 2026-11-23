"""Pipeline fine-tuning IndoBERT yang bisa dipakai baseline, HPO, dan final.

Prinsip yang sama dengan BiLSTM:

- Holdout tidak pernah dipakai untuk training/early stopping/pemilihan model.
- Pemilihan model memakai validation set saja.
- Semua metrik utama dihitung pada threshold tetap 0,5.
- Setiap trial mencatat konfigurasi, log training, probabilitas, dan metrik.

Modul ini mengimpor Torch/Transformers secara **lazy** agar notebook data
dan test ringan tidak perlu menginstal stack BERT.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.experiment.artifacts import environment_snapshot, write_json
from app.experiment.configs import BertConfig
from app.experiment.data_pipeline import DataBundle
from app.experiment.reporting import bootstrap_ci, classification_metrics

logger = logging.getLogger(__name__)

LABEL2ID = {"safe": 0, "unsafe": 1}
ID2LABEL = {0: "safe", 1: "unsafe"}


@dataclass
class BertTrialResult:
    """Hasil satu trial IndoBERT."""

    metrics: dict[str, float]
    history: list[dict[str, Any]]
    val_prob: np.ndarray
    holdout_prob: np.ndarray | None
    artifacts: dict[str, str] = field(default_factory=dict)
    model: Any = None
    tokenizer: Any = None


class TextDataset:
    """Dataset ringan tanpa subclass torch agar import tetap lazy."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def _require_bert_stack():
    import importlib.util

    missing = [
        name for name in ("torch", "transformers", "accelerate")
        if importlib.util.find_spec(name) is None
    ]
    if missing:  # pragma: no cover
        raise ImportError(
            "Training IndoBERT membutuhkan paket: "
            + ", ".join(missing)
            + ". Aktifkan environment Python 3.10–3.12 lalu "
            "`pip install -r requirements-bert.txt`."
        )


def _encode(tokenizer: Any, texts: list[str], max_len: int) -> dict[str, list[list[int]]]:
    return tokenizer(
        [str(text) if text is not None else "" for text in texts],
        truncation=True,
        padding=False,
        max_length=max_len,
    )


def _labels_from_ids(y: np.ndarray) -> list[int]:
    return [int(value) for value in np.asarray(y).astype(int).ravel()]


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob_unsafe = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
    metrics = classification_metrics(np.asarray(labels).astype(int), prob_unsafe)
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")
    return {key: float(metrics[key]) for key in keys}


def _make_trainer_class():
    """Bangun subclass Trainer yang mendukung class weight + label smoothing."""
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *, class_weight=None, label_smoothing: float = 0.0, **kwargs):
            super().__init__(**kwargs)
            self._class_weight = class_weight
            self._label_smoothing = float(label_smoothing)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if self._class_weight is None and self._label_smoothing <= 0:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, **kwargs
                )
            import torch
            from torch import nn

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = None
            if self._class_weight is not None:
                weight = torch.as_tensor(
                    self._class_weight, dtype=torch.float32, device=logits.device
                )
            loss_fn = nn.CrossEntropyLoss(
                weight=weight, label_smoothing=self._label_smoothing
            )
            loss = loss_fn(
                logits.view(-1, model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _class_weight_tensor(y_train: np.ndarray) -> list[float] | None:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    mapping = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    return [mapping.get(0, 1.0), mapping.get(1, 1.0)]


def _freeze_layers(model: Any, freeze_layers: int) -> dict[str, int]:
    """Bekukan embeddings dan/atau N encoder layer pertama.

    Semantics:
       0  -> semua parameter dilatih.
      -1  -> hanya embeddings yang dibekukan.
       N  -> embeddings + N encoder layer pertama dibekukan.
    """
    if freeze_layers == 0:
        return {
            "frozen_layers": 0,
            "embeddings_frozen": False,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    base = None
    for attribute in ("bert", "roberta", "distilbert", "model"):
        candidate = getattr(model, attribute, None)
        if candidate is not None and hasattr(candidate, "embeddings"):
            base = candidate
            break
    if base is None:
        logger.warning("Arsitektur tidak dikenali; freeze_layers diabaikan.")
        return {
            "frozen_layers": 0,
            "embeddings_frozen": False,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    embeddings = getattr(base, "embeddings", None)
    if embeddings is not None:
        for parameter in embeddings.parameters():
            parameter.requires_grad = False

    frozen = 0
    if freeze_layers > 0:
        encoder = getattr(base, "encoder", None)
        layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None) or []
        for layer in list(layers)[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
            frozen += 1
    return {
        "frozen_layers": frozen,
        "embeddings_frozen": embeddings is not None,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def _build_training_arguments(config: BertConfig, output_dir: Path, train_size: int):
    """Buat TrainingArguments yang kompatibel dengan beberapa versi Transformers."""
    from transformers import TrainingArguments

    import inspect

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)

    precision = config.precision
    if precision != "fp32":
        import torch

        if not torch.cuda.is_available():
            logger.warning("Presisi %s diminta tetapi CUDA tidak tersedia; memakai fp32.", precision)
            precision = "fp32"
        elif precision == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 tidak didukung GPU ini; memakai fp16.")
            precision = "fp16"

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optimizer,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "seed": config.seed,
        "data_seed": config.seed,
        "logging_steps": config.logging_steps,
        "report_to": "none",
        "group_by_length": config.group_by_length,
        "dataloader_num_workers": config.dataloader_num_workers,
        "label_smoothing_factor": 0.0,  # ditangani WeightedTrainer
    }
    if config.save_total_limit > 0:
        kwargs["save_total_limit"] = config.save_total_limit

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = config.warmup_ratio
    elif "warmup_steps" in supported:
        # Fallback untuk versi Transformers lama yang belum punya warmup_ratio.
        effective_batch = max(1, config.batch_size * config.gradient_accumulation_steps)
        steps_per_epoch = max(1, train_size // effective_batch)
        kwargs["warmup_steps"] = int(config.warmup_ratio * steps_per_epoch * config.epochs)

    if precision == "fp16" and "fp16" in supported:
        kwargs["fp16"] = True
    if precision == "bf16" and "bf16" in supported:
        kwargs["bf16"] = True

    # Buang key yang tidak dikenal versi ini.
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(**kwargs)


def run_bert_trial(
    bundle: DataBundle,
    config: BertConfig,
    output_dir: str | os.PathLike,
    *,
    save_model: bool = False,
    save_predictions: bool = True,
    evaluate_holdout: bool = True,
    verbose: bool = True,
) -> BertTrialResult:
    """Fine-tune satu konfigurasi IndoBERT dan evaluasi val + holdout."""
    _require_bert_stack()
    from app.repro import apply_reproducibility_lock

    apply_reproducibility_lock(config.seed)
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        set_seed,
    )

    set_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model_config = AutoConfig.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if config.hidden_dropout_prob is not None:
        model_config.hidden_dropout_prob = config.hidden_dropout_prob
    if config.attention_probs_dropout_prob is not None:
        model_config.attention_probs_dropout_prob = config.attention_probs_dropout_prob
    if config.classifier_dropout is not None:
        model_config.classifier_dropout = config.classifier_dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, config=model_config
    )
    freeze_info = _freeze_layers(model, config.freeze_layers)

    tr_enc = _encode(tokenizer, bundle.X_train_text, config.max_len)
    va_enc = _encode(tokenizer, bundle.X_val_text, config.max_len)
    tr_y = _labels_from_ids(bundle.y_train)
    va_y = _labels_from_ids(bundle.y_val)
    train_dataset = TextDataset(tr_enc, tr_y)
    val_dataset = TextDataset(va_enc, va_y)

    training_args = _build_training_arguments(config, out, train_size=len(tr_y))
    uses_weighted_trainer = bool(
        config.use_class_weight or config.label_smoothing_factor > 0
    )
    if uses_weighted_trainer:
        trainer_cls = _make_trainer_class()
    else:
        from transformers import Trainer

        trainer_cls = Trainer
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer, padding="longest"),
        "compute_metrics": _compute_metrics,
        "callbacks": (
            [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
            if config.early_stopping_patience > 0 else []
        ),
    }
    if uses_weighted_trainer:
        trainer_kwargs["class_weight"] = (
            _class_weight_tensor(bundle.y_train) if config.use_class_weight else None
        )
        trainer_kwargs["label_smoothing"] = config.label_smoothing_factor
    trainer = trainer_cls(**trainer_kwargs)

    import time

    started = time.time()
    trainer.train()
    train_seconds = time.time() - started
    history = list(trainer.state.log_history)
    write_json(out / "training_log.json", history)

    def _predict(
        active_trainer: Any,
        texts: list[str],
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        enc = _encode(tokenizer, texts, config.max_len)
        dataset = TextDataset(enc, _labels_from_ids(y))
        prediction = active_trainer.predict(dataset)
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        prob = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
        return np.asarray(y).astype(int), prob, float(prediction.metrics.get("test_loss", float("nan")))

    y_val, val_prob, val_loss = _predict(trainer, bundle.X_val_text, bundle.y_val)
    y_hold = None
    hold_prob = None
    hold_loss = float("nan")
    if evaluate_holdout:
        y_hold, hold_prob, hold_loss = _predict(
            trainer, bundle.X_holdout_text, bundle.y_holdout
        )

    val_metrics = classification_metrics(y_val, val_prob)
    hold_metrics = (
        classification_metrics(y_hold, hold_prob)
        if y_hold is not None and hold_prob is not None else {}
    )
    metrics = {
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"holdout_{key}": value for key, value in hold_metrics.items()},
        "val_loss": val_loss,
        "train_seconds": float(train_seconds),
        "epochs_ran": float(trainer.state.epoch or 0.0),
        "best_checkpoint": str(trainer.state.best_model_checkpoint or ""),
        "precision_used": (
            "fp16" if getattr(training_args, "fp16", False)
            else "bf16" if getattr(training_args, "bf16", False)
            else "fp32"
        ),
        **freeze_info,
    }
    if evaluate_holdout:
        metrics["holdout_loss"] = hold_loss
    write_json(out / "metrics.json", metrics)
    write_json(
        out / "training_summary.json",
        {
            "train_seconds": train_seconds,
            "epochs_ran": trainer.state.epoch,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "freeze_info": freeze_info,
            "config": config.to_dict(),
        },
    )

    artifacts: dict[str, str] = {
        "metrics": str(out / "metrics.json"),
        "training_log": str(out / "training_log.json"),
    }
    if save_predictions:
        val_df = pd.DataFrame(
            {
                "text": bundle.X_val_text,
                "label": np.where(bundle.y_val == 1, "unsafe", "safe"),
                "label_id": bundle.y_val.astype(int),
                "prob_unsafe": val_prob.astype(float),
            }
        )
        val_df.to_csv(out / "probs_val.csv", index=False)
        artifacts["probs_val"] = str(out / "probs_val.csv")
        if hold_prob is not None and y_hold is not None:
            hold_df = pd.DataFrame(
                {
                    "text": bundle.X_holdout_text,
                    "label": np.where(bundle.y_holdout == 1, "unsafe", "safe"),
                    "label_id": bundle.y_holdout.astype(int),
                    "prob_unsafe": hold_prob.astype(float),
                }
            )
            hold_df.to_csv(out / "probs_holdout.csv", index=False)
            artifacts["probs_holdout"] = str(out / "probs_holdout.csv")
        from sklearn.metrics import roc_curve

        roc_payload = {}
        for name, y_true, prob in (("val", y_val, val_prob), ("holdout", y_hold, hold_prob)):
            if y_true is None or prob is None:
                continue
            fpr, tpr, _ = roc_curve(y_true, prob)
            roc_payload[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        write_json(out / "roc_data.json", roc_payload)
        artifacts["roc_data"] = str(out / "roc_data.json")

    if save_model:
        model_dir = out / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        # Metadata minimum agar ml-service bisa membaca max_len tanpa drift.
        write_json(model_dir / "threshold.json", {"threshold": 0.5, "fixed": True})
        write_json(
            model_dir / "metrics.json",
            {
                "model_name": config.model_name,
                "max_len": config.max_len,
                "seed": config.seed,
            },
        )
        artifacts["model"] = str(model_dir)
    else:
        # HPO tidak butuh checkpoint per trial; hapus agar disk tidak penuh.
        import shutil

        shutil.rmtree(out / "checkpoints", ignore_errors=True)

    result = BertTrialResult(
        metrics=metrics,
        history=history,
        val_prob=val_prob,
        holdout_prob=hold_prob,
        artifacts=artifacts,
        model=None,  # model sudah disimpan (jika diminta); hindari GPU memory leak
        tokenizer=tokenizer,
    )
    # Bebaskan GPU secepat mungkin saat HPO.
    try:
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def train_final_bert(
    bundle: DataBundle,
    config: BertConfig,
    output_dir: str | os.PathLike,
    *,
    hpo_info: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Latih IndoBERT final dan simpan artefak kompatibel dengan paper lama."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = run_bert_trial(
        bundle, config, out,
        save_model=True, save_predictions=True, verbose=verbose,
    )

    assert result.holdout_prob is not None, "Training final wajib mengevaluasi holdout."
    validation = {
        key.replace("val_", ""): value
        for key, value in result.metrics.items()
        if key.startswith("val_")
    }
    holdout = {
        key.replace("holdout_", ""): value
        for key, value in result.metrics.items()
        if key.startswith("holdout_")
    }
    metrics = {
        "model_name": config.model_name,
        "seed": config.seed,
        "epochs": config.epochs,
        "lr": config.learning_rate,
        "batch_size": config.batch_size,
        "max_len": config.max_len,
        "train_file": "train_combined.csv",
        "validation": validation,
        "holdout": holdout,
        "config": config.to_dict(),
        "hpo": hpo_info or {},
        "environment": environment_snapshot(seed=config.seed),
    }
    write_json(out / "metrics.json", metrics)
    write_json(out / "threshold.json", {"threshold": 0.5, "fixed": True})
    write_json(
        out / "model_card.json",
        {
            "model_name": config.model_name,
            "config": config.to_dict(),
            "metrics": metrics,
            "data": {
                "csv_sha256": bundle.metadata.get("csv_sha256"),
                "frozen_holdout_sha256": bundle.frozen_sha256,
                "pool_sizes": bundle.pool_sizes,
            },
            "environment": environment_snapshot(seed=config.seed),
        },
    )

    # Tabel evaluasi + bootstrap CI.
    rows = []
    for split_name, y_true, y_prob, split_metrics in (
        ("validation", bundle.y_val, result.val_prob, validation),
        ("frozen_holdout", bundle.y_holdout, result.holdout_prob, holdout),
    ):
        row = {"split": split_name, **split_metrics}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "evaluation_table_bert.csv", index=False)

    ci_rows = []
    for split_name, y_true, y_prob in (
        ("validation", bundle.y_val, result.val_prob),
        ("frozen_holdout", bundle.y_holdout, result.holdout_prob),
    ):
        table = bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstrap=2000, seed=config.seed)
        table.insert(0, "split", split_name)
        ci_rows.append(table)
    pd.concat(ci_rows, ignore_index=True).to_csv(out / "bootstrap_ci_bert.csv", index=False)

    return {
        "metrics": metrics,
        "config": config.to_dict(),
        "artifacts": result.artifacts,
        "output_dir": str(out),
        "result": result,
    }
