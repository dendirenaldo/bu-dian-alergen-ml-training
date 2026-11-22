"""Pelaporan dan analisis statistik untuk eksperimen.

Isi modul:

- Metrik klasifikasi biner pada threshold tetap 0,5.
- Bootstrap CI (stratified).
- Uji McNemar untuk perbandingan dua model pada data yang sama.
- Paired bootstrap untuk selisih ROC-AUC.
- Plot HPO (progres trial, importance parameter).
- Pembacaan file probabilitas dengan skema lama maupun baru.

Modul ini tidak mengimpor TensorFlow/Torch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.core.model.metrics import bootstrap_ci_metrics


def classification_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Hitung metrik klasifikasi biner pada threshold tertentu.

    Metrik yang dikembalikan: n, accuracy, precision, recall, f1, roc_auc,
    average_precision, brier, dan threshold. Bila hanya ada satu kelas,
    ROC-AUC/AP bernilai ``nan`` (tidak terdefinisi).
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if y_true.size != y_prob.size:
        raise ValueError("Panjang y_true dan y_prob harus sama.")
    pred = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "n": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "average_precision": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }


def bootstrap_ci(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    *,
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap CI stratified (mempertahankan proporsi kelas)."""
    return bootstrap_ci_metrics(
        np.asarray(y_true), np.asarray(y_prob),
        threshold=threshold, n_bootstrap=n_bootstrap, seed=seed, ci=ci,
    )


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def paired_bootstrap_auc_diff(
    y_true: Iterable[int],
    prob_a: Iterable[float],
    prob_b: Iterable[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """Uji selisih ROC-AUC dua model dengan resampling berpasangan.

    Karena kedua model dievaluasi pada sampel yang sama, resampling harus
    memakai indeks yang sama untuk keduanya (paired). Ini lebih tepat
    daripada membandingkan dua CI marginal yang saling independen.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    prob_a = np.asarray(prob_a).astype(float).ravel()
    prob_b = np.asarray(prob_b).astype(float).ravel()
    if not (len(y_true) == len(prob_a) == len(prob_b)):
        raise ValueError("Panjang y_true, prob_a, dan prob_b harus sama.")
    observed = _auc(y_true, prob_a) - _auc(y_true, prob_b)
    rng = np.random.RandomState(seed)
    indices_by_class = [np.where(y_true == cls)[0] for cls in (0, 1)]
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        sampled = [
            rng.choice(idx, size=len(idx), replace=True) if len(idx) else np.array([], dtype=int)
            for idx in indices_by_class
        ]
        idx = np.concatenate(sampled)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(_auc(y_true[idx], prob_a[idx]) - _auc(y_true[idx], prob_b[idx]))
    if not diffs:
        return {
            "auc_a_minus_b": observed,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "n_bootstrap": 0,
        }
    arr = np.asarray(diffs)
    alpha = 1.0 - ci
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    # p-value dua sisi: proporsi resampling yang berlawanan arah / melewati nol.
    p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
    return {
        "auc_a_minus_b": float(observed),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": min(1.0, max(0.0, p)),
        "n_bootstrap": int(len(arr)),
        "ci": float(ci),
    }


def mcnemar_test(
    y_true: Iterable[int],
    pred_a: Iterable[int],
    pred_b: Iterable[int],
) -> dict[str, Any]:
    """Uji McNemar untuk dua model pada sampel yang sama.

    Tabel 2x2:

    - ``b`` = model A benar, model B salah.
    - ``c`` = model A salah, model B benar.

    Uji binomial exact dipakai bila SciPy tersedia; jika tidak, pakai
    aproksimasi chi-square dengan koreksi kontinuitas.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    pred_a = np.asarray(pred_a).astype(int).ravel()
    pred_b = np.asarray(pred_b).astype(int).ravel()
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    try:
        from scipy.stats import binomtest

        result = binomtest(b, n, 0.5, alternative="two-sided")
        return {
            "b": b, "c": c,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "exact_binomial",
        }
    except Exception:
        statistic = (abs(b - c) - 1) ** 2 / max(1, n)
        # Aproksimasi chi-square 1 derajat bebas.
        p_value = math.erfc(math.sqrt(statistic / 2.0))
        return {
            "b": b, "c": c,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "method": "chi2_continuity",
        }


def compare_two_models(
    y_true: Iterable[int],
    prob_a: Iterable[float],
    prob_b: Iterable[float],
    *,
    name_a: str = "model_a",
    name_b: str = "model_b",
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Ringkasan perbandingan dua model pada sampel yang sama."""
    y_true = np.asarray(y_true).astype(int).ravel()
    prob_a = np.asarray(prob_a).astype(float).ravel()
    prob_b = np.asarray(prob_b).astype(float).ravel()
    metrics_a = classification_metrics(y_true, prob_a, threshold)
    metrics_b = classification_metrics(y_true, prob_b, threshold)
    pred_a = (prob_a >= threshold).astype(int)
    pred_b = (prob_b >= threshold).astype(int)
    return {
        "name_a": name_a,
        "name_b": name_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "delta": {
            key: metrics_a[key] - metrics_b[key]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")
        },
        "mcnemar": mcnemar_test(y_true, pred_a, pred_b),
        "paired_bootstrap_auc": paired_bootstrap_auc_diff(
            y_true, prob_a, prob_b,
            n_bootstrap=n_bootstrap, seed=seed,
        ),
    }


# ----------------------------------------------------------------------
# Pembacaan probabilitas
# ----------------------------------------------------------------------
def load_probabilities(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Baca CSV probabilitas dengan skema lama maupun baru.

    Skema yang didukung:

    - ``label_id,prob_unsafe`` (BiLSTM lama)
    - ``text,label,prob_unsafe`` (BERT lama)
    - ``text,label,label_id,prob_unsafe`` (skema baru)

    Returns:
        ``(y_true, y_prob, dataframe)``.
    """
    df = pd.read_csv(path)
    if "prob_unsafe" not in df.columns:
        raise ValueError(f"{path} tidak punya kolom 'prob_unsafe'.")
    if "label_id" in df.columns:
        y_true = df["label_id"].astype(int).values
    elif "label" in df.columns:
        y_true = (df["label"].astype(str).str.lower() == "unsafe").astype(int).values
    else:
        raise ValueError(f"{path} tidak punya kolom 'label_id' atau 'label'.")
    return y_true, df["prob_unsafe"].astype(float).values, df


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_roc_pr(
    curves: dict[str, tuple[Iterable[int], Iterable[float]]],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> str:
    """Plot ROC dan PR untuk beberapa model dalam satu gambar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y_true, y_prob) in curves.items():
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, linewidth=1.8, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.4f})")
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[1].plot(
            recall, precision, linewidth=1.8,
            label=f"{name} (AP={average_precision_score(y_true, y_prob):.4f})",
        )
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def plot_confusion(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    output_path: str | Path,
    *,
    threshold: float = 0.5,
    title: str | None = None,
) -> str:
    """Plot confusion matrix sederhana tanpa seaborn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob).astype(float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred safe", "Pred unsafe"])
    ax.set_yticks([0, 1], ["Actual safe", "Actual unsafe"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=13, weight="bold",
            )
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def plot_hpo_progress(
    trials: pd.DataFrame,
    output_path: str | Path,
    *,
    objective_col: str = "objective",
    mode: str = "max",
) -> str:
    """Plot objective per trial + garis best-so-far."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = trials.copy()
    df = df[df["status"] == "ok"].reset_index(drop=True)
    if df.empty:
        raise ValueError("Tidak ada trial sukses untuk diplot.")
    values = pd.to_numeric(df[objective_col], errors="coerce")
    best = values.cummax() if mode == "max" else values.cummin()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(range(len(df)), values, s=28, label="Trial", alpha=0.75)
    ax.plot(range(len(df)), best, linewidth=2, label="Best so far")
    ax.set_xlabel("Trial (urut kronologis)")
    ax.set_ylabel(objective_col)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def _parse_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def plot_param_importance(
    trials: pd.DataFrame,
    output_path: str | Path,
    *,
    objective_col: str = "objective",
) -> str:
    """Plot korelasi/effect size tiap parameter terhadap objective.

    - Parameter numerik: korelasi Pearson absolut.
    - Parameter kategorikal: eta-squared (proporsi varians antar kelompok).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = trials[trials["status"] == "ok"].copy()
    if df.empty:
        raise ValueError("Tidak ada trial sukses untuk diplot.")
    df["objective_value"] = pd.to_numeric(df[objective_col], errors="coerce")
    df = df.dropna(subset=["objective_value"])
    if "params_json" in df.columns:
        params_df = pd.json_normalize(df["params_json"].map(_parse_params))
    else:
        params_df = pd.DataFrame([_parse_params(v) for v in df.get("params", [])])
    rows = []
    total_var = float(np.var(df["objective_value"])) or 1.0
    for column in params_df.columns:
        values = params_df[column]
        if values.isna().all():
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() >= max(3, 0.8 * len(values)):
            score = abs(float(np.corrcoef(numeric.fillna(numeric.mean()), df["objective_value"])[0, 1]))
            kind = "correlation"
        else:
            groups = [
                group["objective_value"].values
                for _, group in df.assign(param=values.astype(str)).groupby("param")
                if len(group)
            ]
            if len(groups) < 2:
                continue
            grand_mean = df["objective_value"].mean()
            between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            score = float(between / (total_var * len(df)))
            kind = "eta_squared"
        if np.isnan(score):
            continue
        rows.append({"parameter": column, "score": score, "kind": kind})
    importance = pd.DataFrame(rows).sort_values("score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(importance) + 1)))
    ax.barh(importance["parameter"], importance["score"], color="#4C78A8")
    ax.set_xlabel("Pengaruh terhadap objective (semakin besar semakin penting)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def write_markdown_summary(
    path: str | Path,
    *,
    title: str,
    sections: list[tuple[str, str]],
) -> str:
    """Tulis ringkasan Markdown sederhana untuk lampiran paper."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return str(target)
