"""Figure BERT untuk bagian Arsitektur & Evaluasi (disisipkan ke Bagian Hasil).

Semua 300 DPI, serif, TANPA judul dalam gambar (caption di naskah).
Data dibaca dari artefak aktual (artifacts/bert/results/ + probs/roc_data).
Jalankan dari root repo: venv/bin/python paper/make_bert_figs.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300,
})
OUT = ROOT / "paper_figs"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, RED, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"

# ============================================================
# 1. Diagram arsitektur end-to-end (Gambar 4.6)
# ============================================================
stages = [
    ("Teks komposisi\n(mentah)", "", "#EFEFEF"),
    ("Tokenisasi\nWordPiece", "vocab 50 rb\npotong/pad 256", "#DCE9F7"),
    ("Embedding\ntoken+posisi", "768 dim", "#DCE9F7"),
    ("12 × Transformer\nEncoder", "12 kepala · FFN 3072\ndropout 0,1", "#F7E3C3"),
    ("[CLS] + Dense\n768 → 2", "kepala 768 → 2\ndropout 0,1", "#C9E2C0"),
    ("Softmax", "p(safe), p(unsafe)\nkeputusan @0,5", "#E9D9F2"),
]
fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.set_xlim(0, 21.4); ax.set_ylim(0, 3.4); ax.axis("off")
bw, bh, gap = 2.5, 1.5, 0.55
for i, (title, sub, fc) in enumerate(stages):
    x0 = 0.2 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x0, 1.05), bw, bh, boxstyle="round,pad=0.04",
                                facecolor=fc, edgecolor="#1f3a5f", linewidth=1.1))
    nlines = title.count("\n") + 1
    ax.text(x0 + bw / 2, 1.05 + bh - 0.32 if nlines == 1 else 1.05 + bh - 0.42,
            title, ha="center", va="center", fontsize=7.6, weight="bold")
    if sub:
        ax.text(x0 + bw / 2, 1.34, sub, ha="center", va="center",
                fontsize=5.9, style="italic")
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x0 + bw, 1.05 + bh / 2),
                                     (x0 + bw + gap, 1.05 + bh / 2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.3, color="black"))
# satu baris anotasi di tengah bawah (tanpa tumpang tindih)
ax.annotate(
    "max_length = 256 token  ·  12 layer · hidden 768 · 12 kepala · posisi ≤ 512"
    "  ·  label: 0 = safe, 1 = unsafe",
    xy=((len(stages) * (bw + gap)) / 2.0, 0.70), ha="center", fontsize=6.4,
    color="#1f3a5f")
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "bert_diagram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 2. Kurva pelatihan (Gambar 4.7)
# ============================================================
blog = json.load(open("artifacts/bert/results/training_log.json"))
ev = [(e["epoch"], e["eval_loss"], e["eval_f1"], e["eval_roc_auc"],
       e["eval_precision"], e["eval_recall"]) for e in blog if "eval_loss" in e]
tr = [(e["epoch"], e["loss"]) for e in blog
      if "loss" in e and "eval_loss" not in e and "train_runtime" not in e]
best = min(ev, key=lambda t: t[1])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axes[0]
ax.plot([t[0] for t in ev], [t[1] for t in ev], color=ORANGE, lw=1.8,
        marker="s", ms=5, label="Validation loss")
ax.scatter([t[0] for t in tr], [t[1] for t in tr], color=BLUE, s=32,
           zorder=3, label="Train loss (sampling)")
ax.axvline(best[0], color="gray", lw=0.9, ls=":",
           label=f"Epoch terbaik ({best[0]:g})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(frameon=True, loc="upper right")
ax = axes[1]
ax.plot([t[0] for t in ev], [t[2] for t in ev], color=GREEN, lw=1.6,
        marker="o", ms=4, label="F1")
ax.plot([t[0] for t in ev], [t[3] for t in ev], color=PURPLE, lw=1.6,
        marker="s", ms=4, label="ROC-AUC")
ax.plot([t[0] for t in ev], [t[4] for t in ev], color=BLUE, lw=1.2,
        ls="--", label="Precision")
ax.plot([t[0] for t in ev], [t[5] for t in ev], color=ORANGE, lw=1.2,
        ls="--", label="Recall")
ax.axvline(best[0], color="gray", lw=0.9, ls=":")
ax.set_xlabel("Epoch"); ax.set_ylabel("Skor (validasi)")
ax.set_ylim(0.90, 1.01)
ax.legend(frameon=True, loc="lower left", ncol=2)
fig.tight_layout(pad=1.0)
fig.savefig(OUT / "bert_training.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 3. Matriks konfusi 1×2 (Gambar 4.8)
# ============================================================
def cm_from(path, label_col):
    d = pd.read_csv(path)
    if d[label_col].dtype == object:
        yt = (d[label_col].astype(str).str.lower() == "unsafe").astype(int).values
    else:
        yt = d[label_col].astype(int).values
    yp = (d["prob_unsafe"].astype(float).values >= 0.5).astype(int)
    return confusion_matrix(yt, yp, labels=[0, 1])

cm_v = cm_from("artifacts/bert/results/probs_val.csv", "label")
cm_h = cm_from("artifacts/bert/results/probs_holdout.csv", "label")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, cm, tag in zip(axes, [cm_v, cm_h],
                       ["(a) Validasi (n = 50)", "(b) Holdout beku (n = 50)"]):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=25)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. safe", "Pred. unsafe"])
    ax.set_yticklabels(["Aktual safe", "Aktual unsafe"])
    ax.set_xlabel(tag, fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=13, weight="bold",
                    color="white" if cm[i, j] > 12 else "black")
fig.tight_layout(pad=1.2)
fig.savefig(OUT / "bert_confusion.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 4. ROC + PR BERT (Gambar 4.9)
# ============================================================
d_v = pd.read_csv("artifacts/bert/results/probs_val.csv")
d_h = pd.read_csv("artifacts/bert/results/probs_holdout.csv")
y_v = (d_v["label"].str.lower() == "unsafe").astype(int).values
y_h = (d_h["label"].str.lower() == "unsafe").astype(int).values
p_v, p_h = d_v["prob_unsafe"].astype(float).values, d_h["prob_unsafe"].astype(float).values

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax = axes[0]
fpr_v, tpr_v, _ = roc_curve(y_v, p_v)
fpr_h, tpr_h, _ = roc_curve(y_h, p_h)
ax.plot(fpr_v, tpr_v, lw=1.8, color=BLUE, label=f"Validasi (AUC = {auc(fpr_v, tpr_v):.3f})")
ax.plot(fpr_h, tpr_h, lw=1.8, color=ORANGE, label=f"Holdout beku (AUC = {auc(fpr_h, tpr_h):.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=0.9, ls=":", label="Acak")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(frameon=True, loc="lower right")
ax = axes[1]
prec_v, rec_v, _ = precision_recall_curve(y_v, p_v)
prec_h, rec_h, _ = precision_recall_curve(y_h, p_h)
ax.plot(rec_v, prec_v, lw=1.8, color=BLUE, label="Validasi")
ax.plot(rec_h, prec_h, lw=1.8, color=ORANGE, label="Holdout beku")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_ylim(0.45, 1.02)
ax.legend(frameon=True, loc="lower left")
fig.tight_layout(pad=1.0)
fig.savefig(OUT / "bert_roc_pr.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 5. Persamaan (5) softmax, (6) AUC
# ============================================================
eqs = {
    "bert_eq_softmax": r"$p(y=j \mid \mathbf{z}) = \dfrac{\exp(z_j)}{\sum_{k=0}^{1} \exp(z_k)}, \quad j \in \{0,1\}$",
    "bert_eq_ce": r"$\mathcal{L}_{\mathrm{CE}} = -\dfrac{1}{N}\sum_{i=1}^{N} \log\, p\!\left(y_i \mid \mathbf{z}_i\right)$",
    "bert_eq_auc": r"$\mathrm{AUC} = \int_{0}^{1} \mathrm{TPR}(t)\; \mathrm{d}\,\mathrm{FPR}(t)$",
    "bert_eq_f1": r"$P=\dfrac{TP}{TP+FP},\;\; R=\dfrac{TP}{TP+FN},\;\; F_1=\dfrac{2PR}{P+R}$",
}
for name, tex in eqs.items():
    fig = plt.figure(figsize=(6.4, 0.9))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=11.5)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)

print("bert figures:", sorted(p.name for p in OUT.glob("bert_*")))
