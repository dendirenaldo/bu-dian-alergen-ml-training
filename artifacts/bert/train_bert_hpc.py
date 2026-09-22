"""Fine-tuning BERT di HPC — kontrak apel-vs-apel dengan BiLSTM.

Input : artifacts/bert/input/{train_combined,val,holdout}.csv dari scripts/rebuild_split.py
Aturan:
  - Early-stop & model-selection HANYA dari val.csv.
  - holdout.csv HANYA diprediksi sekali di akhir (tidak untuk tuning).
  - Metrik utama memakai threshold FIXED 0.5 (sama seperti BiLSTM).

Contoh:
  python train_bert_hpc.py --data-dir ./artifacts/bert/input --output-dir ./artifacts/bert/results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL2ID = {"safe": 0, "unsafe": 1}


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def metrics_at_threshold(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


def compute_eval_metrics(eval_pred):
    """Metrik evaluasi tiap epoch ala verbose TensorFlow.

    Dipakai via Trainer(compute_metrics=...). Threshold fixed 0.5,
    sama seperti metrik utama. Tidak memengaruhi seleksi model
    (tetap eval_loss) maupun bobot.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    prob = (e / e.sum(axis=1, keepdims=True))[:, 1]
    full = metrics_at_threshold(np.asarray(labels), prob, threshold=0.5)
    return {k: v for k, v in full.items()
            if k in ("accuracy", "precision", "recall", "f1",
                     "roc_auc", "average_precision")}


def main():
    # Senyapkan warning kosmetik internal HF Trainer (agregasi loss skalar).
    # Tidak memengaruhi bobot/hasil.
    warnings.filterwarnings(
        "ignore", message="Was asked to gather along dimension 0.*")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./input")
    ap.add_argument("--output-dir", default="./results")
    ap.add_argument("--model-name", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--train-file", default="train_combined.csv")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    hold_df = pd.read_csv(os.path.join(args.data_dir, "holdout.csv"))
    for df, name in ((train_df, "train"), (val_df, "val"), (hold_df, "holdout")):
        assert set(df["label"].str.lower().unique()) <= {"safe", "unsafe"}, name
    print(f"train={len(train_df)} val={len(val_df)} holdout={len(hold_df)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def encode(texts):
        return tok(list(texts), truncation=True, padding="max_length",
                   max_length=args.max_len)

    tr_enc = encode(train_df["text"].fillna("").astype(str))
    va_enc = encode(val_df["text"].fillna("").astype(str))
    tr_y = train_df["label"].str.lower().map(LABEL2ID).astype(int).tolist()
    va_y = val_df["label"].str.lower().map(LABEL2ID).astype(int).tolist()

    # Peta label eksplisit: menimpa id2label basi (5 label) bawaan config
    # IndoBERT sekaligus membuat artefak model/ bersih untuk serving.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
        id2label={0: "safe", 1: "unsafe"},
        label2id={"safe": 0, "unsafe": 1})

    import inspect as _inspect
    _TA_PARAMS = set(_inspect.signature(TrainingArguments.__init__).parameters)

    ta_kwargs = dict(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        logging_steps=50,
    )
    # --- kompatibilitas lintas versi transformers ---
    # transformers lama memakai 'evaluation_strategy', bukan 'eval_strategy'.
    if "eval_strategy" in _TA_PARAMS:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"
    # 'warmup_ratio' tidak ada di transformers lama -> hitung warmup_steps
    # manual (±10% total step). Bila keduanya tak ada, abaikan warmup.
    if "warmup_ratio" in _TA_PARAMS:
        ta_kwargs["warmup_ratio"] = 0.1
    elif "warmup_steps" in _TA_PARAMS:
        steps_per_epoch = max(1, len(tr_y) // args.batch_size)
        ta_kwargs["warmup_steps"] = int(0.1 * steps_per_epoch * args.epochs)
    for _k, _v in (("data_seed", args.seed), ("save_total_limit", 1),
                   ("report_to", "none"), ("max_grad_norm", 1.0)):
        if _k in _TA_PARAMS:
            ta_kwargs[_k] = _v
    targs = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=TextDataset(tr_enc, tr_y),
        eval_dataset=TextDataset(va_enc, va_y),
        compute_metrics=compute_eval_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    # Riwayat log per-epoch (loss/train + eval_*) untuk kurva skripsi.
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=float)

    def predict_probs(df):
        enc = encode(df["text"].fillna("").astype(str))
        y = df["label"].str.lower().map(LABEL2ID).astype(int).tolist()
        out = trainer.predict(TextDataset(enc, y))
        logits = out.predictions
        # softmax biner -> prob kelas unsafe (id=1)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return np.asarray(y), (e / e.sum(axis=1, keepdims=True))[:, 1]

    y_val, p_val = predict_probs(val_df)
    y_hold, p_hold = predict_probs(hold_df)

    metrics = {
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "train_file": args.train_file,
        "validation": metrics_at_threshold(y_val, p_val),
        "holdout": metrics_at_threshold(y_hold, p_hold),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": 0.5, "fixed": True}, f)

    for name, df, y, p in (("val", val_df, y_val, p_val),
                           ("holdout", hold_df, y_hold, p_hold)):
        pd.DataFrame({"text": df["text"].values, "label": df["label"].values,
                      "prob_unsafe": p}).to_csv(
            os.path.join(args.output_dir, f"probs_{name}.csv"), index=False)

    roc_data = {}
    for name, y, p in (("val", y_val, p_val), ("holdout", y_hold, p_hold)):
        fpr, tpr, _ = roc_curve(y, p)
        roc_data[name] = {"fpr": [float(x) for x in fpr],
                          "tpr": [float(x) for x in tpr]}
    with open(os.path.join(args.output_dir, "roc_data.json"), "w") as f:
        json.dump(roc_data, f)

    with open(os.path.join(args.output_dir, "model_card.json"), "w") as f:
        json.dump({"model_name": args.model_name, "seed": args.seed,
                   "epochs": args.epochs, "lr": args.lr,
                   "batch_size": args.batch_size, "max_len": args.max_len,
                   "train_file": args.train_file,
                   "torch": torch.__version__}, f, indent=2)

    trainer.save_model(os.path.join(args.output_dir, "model"))
    tok.save_pretrained(os.path.join(args.output_dir, "model"))
    print("VAL :", json.dumps(metrics["validation"], indent=2), flush=True)
    print("HOLD:", json.dumps(metrics["holdout"], indent=2), flush=True)
    print(f"Selesai -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
