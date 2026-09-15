"""Regenerasi figure publikasi 300 DPI (tanpa judul dalam gambar; caption di naskah)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300,
})
OUT = "paper_figs"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"

# ---------- Gambar: distribusi label ----------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
cats = ["safe", "unsafe"]
real = [248, 251]
ax = axes[0]
ax.bar(cats, real, color=[BLUE, ORANGE], edgecolor="black", linewidth=0.6)
for i, v in enumerate(real):
    ax.text(i, v + 4, str(v), ha="center", fontsize=9)
ax.set_ylabel("Jumlah sampel")
ax.set_xlabel("Label acuan (gold label)")
ax.set_ylim(0, 290)
splits = ["train\n(n=399)", "validation\n(n=50)", "holdout\n(n=50)"]
safe = [198, 25, 25]; unsafe = [201, 25, 25]
x = np.arange(3); w = 0.36
ax = axes[1]
ax.bar(x - w/2, safe, w, label="safe", color=BLUE, edgecolor="black", linewidth=0.6)
ax.bar(x + w/2, unsafe, w, label="unsafe", color=ORANGE, edgecolor="black", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel("Jumlah sampel"); ax.legend(frameon=True)
for i in range(3):
    ax.text(i - w/2, safe[i] + 4, str(safe[i]), ha="center", fontsize=8)
    ax.text(i + w/2, unsafe[i] + 4, str(unsafe[i]), ha="center", fontsize=8)
ax.set_ylim(0, 235)
fig.tight_layout(pad=1.2)
fig.savefig(f"{OUT}/fig_distribusi_label.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Gambar: diagram alir metodologi ----------
stages = [
    ("Studi\nLiteratur", None),
    ("Pengumpulan\nData", "foto kemasan +\nteks komposisi"),
    ("Pra-pemrosesan\n(OCR + teks)", "normalisasi"),
    ("Knowledge\nBase Alergen", "8 kategori\n(diagnostik)"),
    ("Pelabelan\nAcuan", "gold label"),
    ("Partisi\nAnti-Leakage", "union-find +\nfrozen holdout"),
    ("Data\nSintetik", "real-train\nonly"),
    ("Word2Vec +\nBiLSTM / BERT", "2 model"),
    ("Evaluasi\n+ Audit", "ROC, CI,\ntriase"),
]
fig, ax = plt.subplots(figsize=(6.5, 2.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
n = len(stages); bw, bh, gap = 1.15, 1.1, 0.28
for i, (title, sub) in enumerate(stages):
    x0 = 0.15 + i * (bw + gap)
    box = FancyBboxPatch((x0, 0.95), bw, bh, boxstyle="round,pad=0.03",
                         facecolor="#DCE9F7", edgecolor="#1f3a5f", linewidth=1.0)
    ax.add_patch(box)
    ax.text(x0 + bw/2, 1.72, title, ha="center", va="center", fontsize=7.2, weight="bold")
    if sub:
        ax.text(x0 + bw/2, 1.22, sub, ha="center", va="center", fontsize=6.2, style="italic")
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x0 + bw, 1.5), (x0 + bw + gap, 1.5),
                                     arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color="black"))
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/fig_diagram_metodologi.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Gambar: arsitektur BiLSTM ----------
layers = [("Input\n(indeks kata)\nmax_len=120", 1.15), ("Embedding\nWord2Vec\n100-dim", 1.15),
          ("BiLSTM\n128 unit", 1.0), ("Dropout\n0,3", 0.7), ("BiLSTM\n64 unit", 1.0),
          ("Dropout\n0,3", 0.7), ("Dense\n64, ReLU", 0.95), ("Dropout\n0,2", 0.7),
          ("Sigmoid\n(prob unsafe)", 1.1)]
fig, ax = plt.subplots(figsize=(6.5, 1.9))
ax.set_xlim(0, 13); ax.set_ylim(0, 2.6); ax.axis("off")
bw, gap = 1.12, 0.28
for i, (title, _) in enumerate(layers):
    x0 = 0.12 + i * (bw + gap)
    fc = "#F7E3C3" if "Dropout" in title else ("#C9E2C0" if "Sigmoid" in title else "#DCE9F7")
    ax.add_patch(FancyBboxPatch((x0, 0.7), bw, 1.15, boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor="#1f3a5f", linewidth=1.0))
    ax.text(x0 + bw/2, 1.27, title, ha="center", va="center", fontsize=6.4, weight="bold")
    if i < len(layers) - 1:
        ax.add_patch(FancyArrowPatch((x0 + bw, 1.27), (x0 + bw + gap, 1.27),
                                     arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color="black"))
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/fig_arsitektur_bilstm.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Gambar: kurva training BiLSTM ----------
h = pd.read_csv("artifacts/bilstm/output/training_history_bilstm_word2vec_leakage_safe.csv")
summ = pd.read_csv("artifacts/bilstm/output/training_summary_bilstm_word2vec_leakage_safe.csv").iloc[0]
BEST_EP = int(summ["best_epoch"])
ep = np.arange(1, len(h) + 1)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
ax = axes[0]
ax.plot(ep, h["accuracy"], color=BLUE, lw=1.6, label="Train accuracy")
ax.plot(ep, h["val_accuracy"], color=ORANGE, lw=1.6, marker="o", ms=3, label="Validation accuracy")
ax.plot(ep, h["f1"], color=BLUE, lw=1.0, ls="--", label="Train F1")
ax.plot(ep, h["val_f1"], color=ORANGE, lw=1.0, ls="--", label="Validation F1")
ax.axvline(BEST_EP, color="gray", lw=0.9, ls=":", label=f"Best epoch ({BEST_EP})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_ylim(0.4, 1.02)
ax.legend(frameon=True, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
ax = axes[1]
ax.plot(ep, h["loss"], color=BLUE, lw=1.6, label="Train loss")
ax.plot(ep, h["val_loss"], color=ORANGE, lw=1.6, marker="o", ms=3, label="Validation loss")
ax.axvline(BEST_EP, color="gray", lw=0.9, ls=":")
ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy loss"); ax.legend(frameon=True)
fig.tight_layout(pad=1.0)
fig.savefig(f"{OUT}/fig_training_bilstm.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Gambar: kurva training BERT ----------
blog = json.load(open("artifacts/bert/bert_hpc_results/training_log.json"))
ev = [(e["epoch"], e["eval_loss"], e.get("eval_f1"), e.get("eval_roc_auc"))
      for e in blog if "eval_loss" in e]
tr = [(e["epoch"], e["loss"]) for e in blog if "loss" in e and "eval_loss" not in e]
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
ax = axes[0]
ax.plot([t[0] for t in tr], [t[1] for t in tr], color=BLUE, lw=1.6, marker="o", ms=4, label="Train loss")
ax.plot([t[0] for t in ev], [t[1] for t in ev], color=ORANGE, lw=1.6, marker="s", ms=4, label="Validation loss")
BEST_BERT = min(ev, key=lambda t: t[1])[0]
ax.axvline(BEST_BERT, color="gray", lw=0.9, ls=":", label=f"Best epoch ({BEST_BERT:g})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(frameon=True)
ax = axes[1]
ax.plot([t[0] for t in ev], [t[2] for t in ev], color=GREEN, lw=1.6, marker="o", ms=4, label="Validation F1")
ax.plot([t[0] for t in ev], [t[3] for t in ev], color=PURPLE, lw=1.6, marker="s", ms=4, label="Validation ROC-AUC")
ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_ylim(0.9, 1.005); ax.legend(frameon=True)
fig.tight_layout(pad=1.0)
fig.savefig(f"{OUT}/fig_training_bert.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- ROC + PR ----------
def _cm_from_probs(path, label_col=None, prob_col="prob_unsafe", thr=0.5):
    d = pd.read_csv(path)
    lc = label_col or ("label_id" if "label_id" in d.columns else "label")
    if d[lc].dtype == object:
        yt = (d[lc].astype(str).str.lower() == "unsafe").astype(int).values
    else:
        yt = d[lc].astype(int).values
    pr = (d[prob_col].astype(float).values >= thr).astype(int)
    return confusion_matrix(yt, pr, labels=[0, 1])


def _cm_from_triage(path, label_col="gold_label", prob_col="prob_unsafe", thr=0.5):
    d = pd.read_csv(path)
    assert (d["error_triage"] != "UNCLASSIFIED").all()
    return _cm_from_probs(path, label_col=label_col, prob_col=prob_col, thr=thr)


def _probs(path, label_col=None, prob_col="prob_unsafe"):
    d = pd.read_csv(path)
    lc = label_col or ("label_id" if "label_id" in d.columns else "label")
    if d[lc].dtype == object:
        y = (d[lc].astype(str).str.lower() == "unsafe").astype(int).tolist()
    else:
        y = d[lc].astype(int).tolist()
    return {"y": y, "p": d[prob_col].astype(float).tolist()}


bp = {"hold": _probs("artifacts/bilstm/output/gold_kb_bilstm_triage.csv", label_col="gold_label"),
      "val": _probs("artifacts/bilstm/output/val_probs.csv", label_col="label_id")}
roc_bert = json.load(open("artifacts/bert/bert_hpc_results/roc_data.json"))
bval = bp["val"]; bhold = bp["hold"]
bv_fpr, bv_tpr, _ = roc_curve(bval["y"], bval["p"])
bh_fpr, bh_tpr, _ = roc_curve(bhold["y"], bhold["p"])
from sklearn.metrics import roc_auc_score as _auc
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
ax = axes[0]
ax.plot(bv_fpr, bv_tpr, lw=1.8, color=BLUE, label=f"BiLSTM val (AUC={_auc(bval['y'], bval['p']):.3f})")
ax.plot(bh_fpr, bh_tpr, lw=1.8, color=ORANGE, label=f"BiLSTM holdout (AUC={_auc(bhold['y'], bhold['p']):.3f})")
ax.plot(roc_bert["val"]["fpr"], roc_bert["val"]["tpr"], lw=1.4, ls="--", color=GREEN, label="BERT val (AUC=0.995)")
ax.plot(roc_bert["holdout"]["fpr"], roc_bert["holdout"]["tpr"], lw=1.4, ls="--", color=RED, label="BERT holdout (AUC=0.984)")
ax.plot([0, 1], [0, 1], color="gray", lw=0.9, ls=":", label="Acak")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(frameon=True, loc="lower right")
# PR
ax = axes[1]
for name, yy, pp, col, ls in [("BiLSTM val", bval["y"], bval["p"], BLUE, "-"),
                              ("BiLSTM holdout", bhold["y"], bhold["p"], ORANGE, "-")]:
    prec, rec, _ = precision_recall_curve(yy, pp)
    ax.plot(rec, prec, lw=1.6, color=col, ls=ls, label=name)
bpv = pd.read_csv("artifacts/bert/bert_hpc_results/probs_val.csv")
bph = pd.read_csv("artifacts/bert/bert_hpc_results/probs_holdout.csv")
for name, df, col in [("BERT val", bpv, GREEN), ("BERT holdout", bph, RED)]:
    yy = (df["label"].str.lower() == "unsafe").astype(int).values
    prec, rec, _ = precision_recall_curve(yy, df["prob_unsafe"].values)
    ax.plot(rec, prec, lw=1.4, ls="--", color=col, label=name)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(frameon=True, loc="lower left")
fig.tight_layout(pad=1.0)
fig.savefig(f"{OUT}/fig_roc_pr.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Confusion matrices 2x2 ----------
def _cm_from_triage(path, label_col="gold_label", prob_col="prob_unsafe", thr=0.5):
    d = pd.read_csv(path)
    yt = (d[label_col].astype(str).str.lower() == "unsafe").astype(int).values
    pr = (d[prob_col].astype(float).values >= thr).astype(int)
    return confusion_matrix(yt, pr, labels=[0, 1])

cm_bh = _cm_from_triage("artifacts/bilstm/output/gold_kb_bilstm_triage.csv")
cm_bv = _cm_from_probs("artifacts/bilstm/output/val_probs.csv")
cm_tv = _cm_from_probs("artifacts/bert/bert_hpc_results/probs_val.csv")
cm_th = _cm_from_probs("artifacts/bert/bert_hpc_results/probs_holdout.csv")


def _cm_from_probs(path, label_col=None, prob_col="prob_unsafe", thr=0.5):
    d = pd.read_csv(path)
    lc = label_col or ("label_id" if "label_id" in d.columns else "label")
    if d[lc].dtype == object:
        yt = (d[lc].astype(str).str.lower() == "unsafe").astype(int).values
    else:
        yt = d[lc].astype(int).values
    pr = (d[prob_col].astype(float).values >= thr).astype(int)
    return confusion_matrix(yt, pr, labels=[0, 1])
fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.2))
for ax, cm, tag in zip(axes.ravel(),
                       [cm_bv, cm_bh, cm_tv, cm_th],
                       ["(a) BiLSTM — validasi", "(b) BiLSTM — holdout",
                        "(c) BERT — validasi", "(d) BERT — holdout"]):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=25)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. safe", "Pred. unsafe"])
    ax.set_yticklabels(["Aktual safe", "Aktual unsafe"])
    ax.set_xlabel(tag, fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=11, weight="bold",
                    color="white" if cm[i, j] > 12 else "black")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="Jumlah sampel")
fig.tight_layout(pad=1.4)
fig.savefig(f"{OUT}/fig_confusion.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Persamaan (render mathtext) ----------
eqs = {
    "eq_lstm": r"$f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\; i_t=\sigma(W_i[h_{t-1},x_t]+b_i),\; o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$"
               "\n" r"$\tilde{C}_t=\tanh(W_C[h_{t-1},x_t]+b_C),\; C_t=f_t \odot C_{t-1}+i_t \odot \tilde{C}_t,\; h_t=o_t \odot \tanh(C_t)$",
    "eq_bilstm": r"$h_t = [\overrightarrow{h}_t \,;\, \overleftarrow{h}_t], \quad \hat{y}=\sigma(w^{\top}h_T+b)$",
    "eq_bce": r"$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)\right]$",
    "eq_f1": r"$P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}$",
}
for name, tex in eqs.items():
    fig = plt.figure(figsize=(6.2, 0.9 if name != "eq_lstm" else 1.2))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=11)
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)

print("figures done:", sorted(os.listdir(OUT)))
