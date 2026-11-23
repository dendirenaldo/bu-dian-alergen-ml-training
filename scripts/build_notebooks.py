"""Bangun notebook pembelajaran dari satu sumber agar konsisten.

Jalankan dari root repo:

    python scripts/build_notebooks.py

Setiap notebook berisi markdown bahasa Indonesia + kode tipis yang memanggil
modul ``app.experiment``. Logika berat tidak diduplikasi di notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list[dict]) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / name
    path.write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("written:", path)


SETUP = r"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Notebook biasanya dijalankan dari folder notebooks/. Kita pindah ke root
# project agar semua path artefak konsisten.
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "artifacts" / "data"
BERT_DIR = PROJECT_ROOT / "artifacts" / "bert" / "input"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "notebook_runs"
for path in (DATA_DIR, BERT_DIR, RUNS_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("Project root :", PROJECT_ROOT)
print("Data dir     :", DATA_DIR)
print("BERT input   :", BERT_DIR)
print("Runs dir     :", RUNS_DIR)
"""


LOAD_DATA = r"""
from app.experiment.configs import DataConfig
from app.experiment.data_pipeline import prepare_data

config_path = DATA_DIR / "data_config.json"
if config_path.exists():
    data_config = DataConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
else:
    candidates = [
        PROJECT_ROOT / "ocr_output" / "data-mengandung.csv",
        PROJECT_ROOT.parent / "old" / "ocr_output" / "data-mengandung.csv",
        PROJECT_ROOT / "data.csv",
    ]
    csv_path = next((path for path in candidates if path.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "CSV tidak ditemukan. Set DataConfig.csv_input secara manual "
            "ke file dengan kolom: nama_produk;text;label"
        )
    data_config = DataConfig(
        csv_input=str(csv_path),
        output_dir=str(DATA_DIR),
        bert_input_dir=str(BERT_DIR),
        sort_source=True,
    )

# prepare_data bersifat deterministik dan cepat; aman dijalankan ulang.
bundle = prepare_data(data_config)
print(bundle.summary_frame().to_string(index=False))
"""


# ======================================================================
# 00 — Panduan dan setup
# ======================================================================
cells_00 = [
    md(
        """
# 00 — Panduan, Setup, dan Peta Project

Notebook ini adalah **titik masuk** untuk memahami project pemodelan deteksi
alergen. Tidak ada training di sini. Tujuannya agar Anda tahu:

- apa yang dimaksud dengan istilah-istilah teknis,
- notebook mana yang harus dijalankan lebih dulu,
- berapa lama waktu yang dibutuhkan,
- dan aturan eksperimen yang tidak boleh dilanggar.
"""
    ),
    md(
        """
## Untuk siapa notebook ini?

Notebook ini ditulis untuk orang yang **belum tentu berlatar belakang machine
learning**. Setiap langkah penting dijelaskan dengan bahasa sehari-hari.
Kode tetap ada, tetapi logika berat disembunyikan di folder `app/experiment/`
agar notebook mudah dibaca.
"""
    ),
    md(
        """
## Peta notebook

| Notebook | Isi | Untuk siapa | Estimasi waktu |
|---|---|---|---|
| `00_panduan_dan_setup.ipynb` | Istilah, setup, peta project | semua | 10 menit |
| `01_data_dan_split.ipynb` | Baca data, audit, split anti-bocor, data sintetik | semua | 10–20 menit |
| `02_bilstm_dasar.ipynb` | Baseline Word2Vec + BiLSTM | pemula | 10–30 menit |
| `03_bilstm_tuning.ipynb` | Hyperparameter tuning BiLSTM lengkap | peneliti | 1–6 jam |
| `04_indobert_dasar.ipynb` | Baseline fine-tuning IndoBERT | pemula | 10–40 menit (GPU) |
| `05_indobert_tuning.ipynb` | Hyperparameter tuning IndoBERT lengkap | peneliti | 2–12 jam (GPU) |
| `06_perbandingan_dan_laporan.ipynb` | Bandingkan model, uji statistik, ekspor laporan | peneliti | 10 menit |
"""
    ),
    md(
        """
## Istilah penting

| Istilah | Arti sederhana |
|---|---|
| **Komposisi** | Daftar bahan pada kemasan makanan. |
| **OCR** | Proses mengubah gambar tulisan menjadi teks. |
| **Gold label** | Label acuan yang dianggap benar (dibuat manusia). |
| **Knowledge Base (KB)** | Aturan kata kunci alergen, dipakai sebagai pembanding/diagnostik. KB bukan pengganti gold label. |
| **Leakage** | Kebocoran informasi: data ujian ikut masuk ke data latihan. Ini membuat skor terlihat bagus padahal tidak jujur. |
| **Split** | Pembagian data menjadi train (latihan), validation (pemilihan model), dan holdout (ujian akhir). |
| **Frozen holdout** | Holdout yang sudah dikunci dan tidak boleh diubah-ubah. |
| **Word2Vec** | Mengubah kata menjadi vektor angka yang menyimpan makna. |
| **BiLSTM** | Model urutan dua arah yang membaca konteks kata dari kiri dan kanan. |
| **IndoBERT** | Model bahasa Transformer pra-latih untuk Bahasa Indonesia. |
| **Hyperparameter** | Pengaturan model yang dipilih sebelum training, mis. learning rate, jumlah unit, dropout. |
| **HPO** | Hyperparameter Optimization: mencari kombinasi hyperparameter terbaik secara sistematis. |
| **ROC-AUC** | Kemampuan model membedakan dua kelas; 0,5 = acak, 1,0 = sempurna. |
| **F1** | Keseimbangan precision dan recall. |
| **Bootstrap CI** | Rentang ketidakpastian metrik, dihitung dengan resampling. |
"""
    ),
    md(
        """
## Cara menjalankan

> Jalankan notebook dari sel pertama (`Restart Kernel and Run All Cells`).
> Notebook memakai state antar sel; melompat ke sel tengah bisa memunculkan
> `NameError`.

Disarankan memakai **Python 3.10–3.12**. TensorFlow 2.16 belum mendukung
Python 3.14, jadi environment yang dipakai untuk training harus Python 3.12
(atau 3.10/3.11).

```bash
# dari root project
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements-train.txt
pip install -r requirements-notebook.txt

# agar hash Python deterministik, set SEBELUM python berjalan
export PYTHONHASHSEED=42           # Windows PowerShell: $env:PYTHONHASHSEED=42

jupyter lab
```

Untuk IndoBERT, install stack terpisah:

```bash
pip install -r requirements-bert.txt
```

Untuk strategi HPO Bayesian (opsional):

```bash
pip install -r requirements-hpo.txt
```
"""
    ),
    code(SETUP),
    md(
        """
## Cek environment

Sel berikut menampilkan versi paket yang terpasang. Jika TensorFlow/Gensim
belum ada, notebook data tetap bisa jalan, tetapi notebook training BiLSTM
akan memberi pesan yang jelas.
"""
    ),
    code(
        r"""
from app.experiment.artifacts import environment_snapshot

snapshot = environment_snapshot(seed=42)
print("Python:", sys.version.split()[0])
print("Git SHA:", snapshot["git_sha"])
print("GPU:", snapshot["gpu"])
print("\nPaket penting:")
for name, version in snapshot["packages"].items():
    print(f"  {name:15s} {version}")

if snapshot["reproducibility_warnings"]:
    print("\nPeringatan reproducibility:")
    for warning in snapshot["reproducibility_warnings"]:
        print("  -", warning)
"""
    ),
    md(
        """
## Cari file dataset

Dataset yang dipakai di project ini memiliki kolom:

- `nama produk` — identitas produk,
- `text` — teks komposisi hasil OCR,
- `label` — `safe` atau `unsafe`.

Sel berikut mencari file CSV di lokasi yang umum.
"""
    ),
    code(
        r"""
candidates = [
    PROJECT_ROOT / "ocr_output" / "data-mengandung.csv",
    PROJECT_ROOT.parent / "old" / "ocr_output" / "data-mengandung.csv",
    PROJECT_ROOT / "data.csv",
]
for path in candidates:
    print(("[ADA]    " if path.exists() else "[TIDAK]  ") + str(path))
"""
    ),
    md(
        """
## Aturan main eksperimen (penting untuk paper)

1. **Holdout tidak boleh disentuh** untuk training, early stopping, atau
   memilih hyperparameter. Holdout hanya diprediksi sekali di akhir.
2. **Validation boleh dipakai untuk memilih model.** Itulah tugasnya.
3. **Threshold utama tetap 0,5.** Jangan mencari threshold yang paling bagus
   di holdout, karena itu termasuk tuning di data ujian.
4. **Setiap angka harus menyebut n dan interval kepercayaan.**
5. **Klaim "model A lebih baik" harus memakai uji berpasangan** (McNemar /
   paired bootstrap), bukan hanya selisih angka.
6. **Jalankan beberapa seed** untuk HPO agar hasil tidak bergantung pada satu
   undian acak.
"""
    ),
    md(
        """
## Estimasi waktu dan kebutuhan

| Kegiatan | CPU | GPU |
|---|---|---|
| Notebook 01 (data + split) | < 1 menit | < 1 menit |
| Baseline BiLSTM 20 epoch | 5–20 menit | 1–3 menit |
| HPO BiLSTM `smoke` (4 trial) | 15–60 menit | 5–15 menit |
| HPO BiLSTM `standard` (30 trial) | 2–8 jam | 30–90 menit |
| Baseline IndoBERT 4 epoch | 1–3 jam | 5–15 menit |
| HPO IndoBERT `smoke` (4 trial) | 4–12 jam | 20–60 menit |

Waktu sangat bergantung pada CPU/GPU, panjang teks, dan batch size.
"""
    ),
    md(
        """
## Troubleshooting singkat

| Gejala | Penyebab umum | Solusi |
|---|---|---|
| `ModuleNotFoundError: tensorflow` | Environment salah | Pakai Python 3.12 + `pip install -r requirements-train.txt` |
| `ModuleNotFoundError: gensim` | Belum install | `pip install gensim==4.3.3` |
| Training sangat lambat | CPU / max_len besar | Turunkan `epochs`, `max_len`, atau pakai GPU |
| `CUDA out of memory` | Batch terlalu besar | Turunkan `batch_size` atau aktifkan `gradient_accumulation_steps` |
| Hasil berbeda setiap run | Hash randomization / seed | Set `PYTHONHASHSEED=42` sebelum Python, pakai seed yang sama |
| `Deterministic GPU implementation ... not available` | Metric Keras/ops tertentu tidak deterministik di GPU | Versi ini sudah memakai metric custom; jika masih muncul, set `config.deterministic = False` |
| `FileNotFoundError` CSV | Path salah | Set `DataConfig.csv_input` ke path absolut |
| Model BERT tidak download | Tidak ada internet | Unduh di login node, set `HF_HUB_OFFLINE=1` |
"""
    ),
]


# ======================================================================
# 01 — Data dan split
# ======================================================================
cells_01 = [
    md(
        """
# 01 — Data, Audit, Split Anti-Bocor, dan Data Sintetik

Notebook ini menjawab pertanyaan:

1. Data apa yang kita punya?
2. Apakah labelnya seimbang?
3. Apakah teks komposisinya panjang?
4. Apakah Knowledge Base setuju dengan gold label?
5. Bagaimana membagi data tanpa kebocoran?
6. Bagaimana data sintetik dibuat dan diaudit?

Output notebook ini dipakai oleh semua notebook training.
"""
    ),
    code(SETUP),
    code(
        r"""
from app.experiment.configs import DataConfig
from app.experiment.data_pipeline import prepare_data, write_bert_inputs
from app.experiment.artifacts import write_json

pd.set_option("display.max_colwidth", 120)
print("Siap.")
"""
    ),
    md(
        """
## Langkah 1 — Baca CSV mentah

Kita baca apa adanya dulu. Delimiter dataset ini adalah titik koma (`;`),
bukan koma. Kalau salah delimiter, semua kolom akan terlihat menjadi satu.
"""
    ),
    code(
        r"""
candidates = [
    PROJECT_ROOT / "ocr_output" / "data-mengandung.csv",
    PROJECT_ROOT.parent / "old" / "ocr_output" / "data-mengandung.csv",
    PROJECT_ROOT / "data.csv",
]
csv_path = next((path for path in candidates if path.exists()), None)
if csv_path is None:
    raise FileNotFoundError("CSV tidak ditemukan. Set csv_path secara manual.")

df_raw = pd.read_csv(csv_path, delimiter=";", encoding="utf-8")
print("File :", csv_path)
print("Bentuk:", df_raw.shape)
print("Kolom :", list(df_raw.columns))
df_raw.head(5)
"""
    ),
    md(
        """
## Langkah 2 — Distribusi label

Dataset yang sehat untuk klasifikasi biner sebaiknya tidak terlalu timpang.
Kalau 95% data hanya satu kelas, model bisa "pintar" hanya dengan menebak kelas
mayoritas.
"""
    ),
    code(
        r"""
label_counts = df_raw["label"].astype(str).str.lower().value_counts()
print(label_counts)

fig, ax = plt.subplots(figsize=(5, 3.5))
label_counts.plot(kind="bar", color=["#4C78A8", "#F58518"], ax=ax)
ax.set_xlabel("Label")
ax.set_ylabel("Jumlah sampel")
ax.set_title("Distribusi label dataset mentah")
for i, value in enumerate(label_counts.values):
    ax.text(i, value + 1, str(value), ha="center")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Langkah 3 — Panjang teks komposisi

Panjang teks menentukan `max_len`. Kalau `max_len` terlalu kecil, banyak teks
terpotong. Kalau terlalu besar, training lebih lambat dan padding lebih banyak.
"""
    ),
    code(
        r"""
from app.core.preprocessing.text import simple_tokenize

token_lengths = df_raw["text"].fillna("").astype(str).map(lambda t: len(simple_tokenize(t)))
stats = token_lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
print(stats)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(token_lengths, bins=40, color="#4C78A8", edgecolor="white")
ax.axvline(token_lengths.median(), color="black", linestyle="--", label=f"median={token_lengths.median():.0f}")
ax.axvline(token_lengths.quantile(0.95), color="red", linestyle=":", label=f"p95={token_lengths.quantile(0.95):.0f}")
ax.set_xlabel("Jumlah token per teks")
ax.set_ylabel("Frekuensi")
ax.legend()
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Langkah 4 — Gold label vs Knowledge Base

**Gold label** adalah acuan manusia. **KB** adalah aturan kata kunci yang
dibuat otomatis. KB berguna untuk:

- mengecek apakah aturan kita masuk akal,
- membuat data sintetik,
- menjelaskan error model.

Tetapi KB **tidak boleh menggantikan gold label**.
"""
    ),
    code(
        r"""
config = DataConfig(
    csv_input=str(csv_path),
    output_dir=str(DATA_DIR),
    bert_input_dir=str(BERT_DIR),
    sort_source=True,   # split tidak berubah walau urutan baris CSV diacak
)
bundle = prepare_data(config)

# Simpan konfigurasi agar notebook lain memakai split yang sama.
write_json(DATA_DIR / "data_config.json", config.to_dict())

print("Ringkasan split:")
print(bundle.summary_frame().to_string(index=False))
"""
    ),
    md(
        """
## Langkah 5 — Audit split

Audit memastikan tidak ada produk atau teks yang sama muncul di dua split.
Kalau ada, model bisa "menghafal" jawaban dan skor menjadi tidak jujur.
"""
    ),
    code(
        r"""
print("Audit leakage:")
print(bundle.split_audit.to_string(index=False))
print("\nKB vs Gold:")
print(bundle.kb_audit["summary"].to_string(index=False))
print("\nConfusion matrix KB vs Gold:")
print(bundle.kb_audit["confusion_matrix"].to_string())
"""
    ),
    md(
        """
## Langkah 6 — Data sintetik

Data sintetik dibuat **hanya dari real-train**, bukan dari validation/holdout.
Tujuannya menambah variasi komposisi. Setiap baris sintetik dicek ulang oleh
KB; kalau labelnya tidak konsisten, proses berhenti.

Perhatikan: data sintetik adalah **prior KB**, bukan data dunia nyata. Karena
itu kita selalu menyediakan opsi ablasi `synthetic_total=0`.
"""
    ),
    code(
        r"""
print("Jumlah synthetic:", len(bundle.df_synthetic))
bundle.df_synthetic.head(8)[["text", "label"]]
"""
    ),
    md(
        """
## Langkah 7 — File split untuk IndoBERT

IndoBERT memakai file CSV yang sama persis dengan BiLSTM. Ini penting agar
perbandingan adil (apel vs apel). File yang dihasilkan:

- `train_real.csv` — 399 data real,
- `synthetic.csv` — data sintetik,
- `train_combined.csv` — real + sintetik (di-shuffle dengan seed),
- `val.csv` — 50 data validasi,
- `holdout.csv` — 50 data ujian akhir,
- `split_contract.json` — hash untuk memastikan file tidak berubah.
"""
    ),
    code(
        r"""
contract = json.loads((BERT_DIR / "split_contract.json").read_text(encoding="utf-8"))
print(json.dumps(contract, indent=2, ensure_ascii=False))
"""
    ),
    md(
        """
## Checklist setelah notebook ini

- [ ] CSV berhasil dibaca dan punya kolom produk/teks/label.
- [ ] Distribusi label tidak ekstrem.
- [ ] Panjang teks masuk akal; `max_len` bisa dipilih.
- [ ] Audit leakage semua **PASS**.
- [ ] Tidak ada produk yang sama di train dan holdout.
- [ ] `split_contract.json` tersimpan.
- [ ] Notebook 02/03/04/05 memakai `data_config.json` yang sama.
"""
    ),
]


# ======================================================================
# 02 — Baseline BiLSTM
# ======================================================================
cells_02 = [
    md(
        """
# 02 — Baseline Word2Vec + BiLSTM

Notebook ini melatih model baseline dengan konfigurasi final yang sudah
dipakai di paper. Tujuannya:

1. Memahami alur Word2Vec → BiLSTM.
2. Melihat cara membaca metrik dan kurva training.
3. Menyimpan artefak model yang bisa diekspor ke layanan inferensi.

Model ini menjadi pembanding untuk hasil HPO di notebook 03.
"""
    ),
    md(
        """
## Alur model

```
Teks komposisi
   │
   ▼
Tokenisasi (Keras Tokenizer)
   │
   ▼
Word2Vec 100 dimensi (dilatih hanya di final-train)
   │
   ▼
Embedding matrix
   │
   ▼
BiLSTM 128 → Dropout → BiLSTM 64 → Dropout
   │
   ▼
Dense 64 → Dropout → Sigmoid
   │
   ▼
Probabilitas unsafe
```

Keputusan `unsafe` jika probabilitas ≥ 0,5.
"""
    ),
    code(SETUP),
    code(LOAD_DATA),
    md(
        """
## Konfigurasi baseline

`QUICK_RUN = True` memakai 5 epoch agar notebook selesai cepat. Untuk hasil
paper, ubah ke `False` (20 epoch + early stopping).
"""
    ),
    code(
        r"""
from app.experiment.configs import BiLSTMConfig

QUICK_RUN = True

config = BiLSTMConfig()
if QUICK_RUN:
    config.epochs = 5
    config.early_stopping_patience = 2

pd.DataFrame([config.to_dict()]).T.rename(columns={0: "nilai"})
"""
    ),
    md(
        """
> **Catatan perbandingan:** baseline baru memakai `shuffle=True` agar setara
> dengan IndoBERT yang selalu mengacak data latih. Konfigurasi legacy paper
> memakai `shuffle=False`; jangan mencampur angka dari kedua protokol tanpa
> menjelaskannya di laporan.
"""
    ),
    md(
        """
## Vektorisasi: Tokenizer + Word2Vec

Tokenizer dan Word2Vec **hanya dilatih pada final-train**. Validation dan
holdout hanya diubah menjadi angka memakai kosakata yang sudah ada. Ini
mencegah kebocoran kosakata.
"""
    ),
    code(
        r"""
from app.experiment.bilstm_pipeline import build_vectorizers

vectorized = build_vectorizers(bundle, config)
print("Ukuran vocabulary tokenizer :", len(vectorized.tokenizer.word_index))
print("Ukuran vocabulary Word2Vec  :", len(vectorized.w2v_model.wv))
print("Bentuk embedding matrix     :", vectorized.embedding_matrix.shape)
print(f"Cakupan Word2Vec            : {vectorized.w2v_coverage:.1%}")
print("\nOOV per split:")
print(vectorized.oov_table.to_string(index=False))
"""
    ),
    md(
        """
## Latih model final baseline

Sel ini menjalankan training, evaluasi validation + holdout, dan menyimpan
artefak ke `artifacts/notebook_runs/bilstm_baseline/`.

Holdout hanya diprediksi **sekali** di akhir. Tidak ada early stopping dari
holdout.
"""
    ),
    code(
        r"""
from app.experiment.bilstm_pipeline import train_final_bilstm

OUT_DIR = RUNS_DIR / "bilstm_baseline"
MODEL_DIR = OUT_DIR / "models"

output = train_final_bilstm(
    bundle,
    config,
    output_dir=OUT_DIR,
    model_dir=MODEL_DIR,
    hpo_info={"mode": "baseline_notebook"},
    verbose=1,
)
print("Selesai. Artefak:", OUT_DIR)
"""
    ),
    md(
        """
## Baca metrik

Metrik utama dihitung pada threshold 0,5:

- **Accuracy**: proporsi prediksi benar.
- **Precision**: dari yang diprediksi unsafe, berapa yang benar unsafe.
- **Recall**: dari yang benar unsafe, berapa yang berhasil ditemukan.
- **F1**: keseimbangan precision dan recall.
- **ROC-AUC**: kemampuan membedakan kelas di semua threshold.
"""
    ),
    code(
        r"""
print(output["eval_table"].to_string(index=False))
"""
    ),
    code(
        r"""
result = output["result"]
history = result.history

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["loss"], label="Train loss")
if "val_loss" in history:
    axes[0].plot(history["val_loss"], label="Validation loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["accuracy"], label="Train accuracy")
if "val_accuracy" in history:
    axes[1].plot(history["val_accuracy"], label="Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

for column in ("precision", "recall", "f1", "val_precision", "val_recall", "val_f1"):
    if column in history:
        axes[2].plot(history[column], label=column)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Score")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
from app.experiment.reporting import plot_confusion, plot_roc_pr

plot_confusion(
    bundle.y_holdout, result.holdout_prob,
    OUT_DIR / "confusion_holdout.png",
    title="BiLSTM baseline — frozen holdout",
)
plot_roc_pr(
    {
        "Validation": (bundle.y_val, result.val_prob),
        "Frozen holdout": (bundle.y_holdout, result.holdout_prob),
    },
    OUT_DIR / "roc_pr.png",
    title="BiLSTM baseline",
)
print("Plot tersimpan di", OUT_DIR)
"""
    ),
    md(
        """
## Cara membaca hasil

- Jika **train accuracy naik** tetapi **validation accuracy turun**, model
  mulai menghafal (overfitting). Early stopping akan mengembalikan bobot
  terbaik.
- Jika **recall tinggi tetapi precision rendah**, model terlalu sering
  menebak `unsafe`. Untuk aplikasi alergi, recall tinggi sering lebih penting,
  tetapi false positive tetap mengganggu pengguna.
- Jika **ROC-AUC jauh di atas 0,5**, model punya kemampuan diskriminasi yang
  baik, meskipun accuracy bisa terlihat biasa saja.

Bandingkan angka ini dengan notebook 03 (setelah HPO) dan notebook 04/05
(IndoBERT). Gunakan holdout yang sama dan uji statistik di notebook 06.
"""
    ),
    md(
        """
## Masalah umum

| Gejala | Solusi |
|---|---|
| Loss tidak turun | Naikkan learning rate atau cek tokenisasi |
| Validation loss naik cepat | Turunkan epoch, naikkan dropout, pakai early stopping |
| Cakupan Word2Vec rendah | Samakan tokenisasi atau turunkan `w2v_min_count` |
| Hasil berubah-ubah | Set `PYTHONHASHSEED` sebelum Python, pakai seed sama |
"""
    ),
]


# ======================================================================
# 03 — HPO BiLSTM
# ======================================================================
cells_03 = [
    md(
        """
# 03 — Hyperparameter Tuning BiLSTM (Lengkap)

Notebook ini melakukan pencarian hyperparameter secara sistematis. Inilah
bagian yang menjawab kritik reviewer: **bukan satu resep, tetapi ruang
pencarian yang jelas + protokol seleksi yang jujur.**

## Protokol

1. HPO hanya memakai **train + validation**.
2. **Frozen holdout tidak disentuh** sampai model terbaik dipilih.
3. Setiap trial dicatat: parameter, metrik, durasi, status.
4. Model dipilih berdasarkan metrik validation (default F1).
5. Setelah terbaik dipilih, model di-retrain dan diuji **sekali** di holdout.
6. Untuk paper, jalankan beberapa seed dan laporkan varians.
"""
    ),
    md(
        """
## Ruang pencarian

Preset:

- `smoke` — 4 trial, untuk memastikan pipeline jalan.
- `standard` — 30 trial, disarankan untuk paper.
- `full` — 100+ trial, eksplorasi maksimal.

Parameter yang dicari mencakup:

- **Vektorisasi**: vocab, max_len, dimensi embedding, window, min_count,
  epoch Word2Vec, CBOW/skip-gram, trainable embedding, mask_zero, digit_fold.
- **Arsitektur**: LSTM/GRU, bidirectional/tidak, 1–2 lapisan, unit, dropout,
  recurrent dropout, dense, MaxNorm.
- **Optimasi**: Adam/AdamW/RMSprop/SGD, learning rate, weight decay, batch,
  gradient clip, class weight, label smoothing.
- **Callback**: patience early stopping, patience/factor reduce LR.
"""
    ),
    code(SETUP),
    code(LOAD_DATA),
    code(
        r"""
from app.experiment.configs import BiLSTMConfig, HPOConfig
from app.experiment.search_spaces import bilstm_search_space, preset_trials, space_summary
from app.experiment.hpo import HPOEngine
from app.experiment.bilstm_pipeline import build_vectorizers, run_bilstm_trial, train_final_bilstm

PRESET = "smoke"          # ganti ke "standard" atau "full" untuk paper
N_TRIALS = preset_trials(PRESET)
space = bilstm_search_space(PRESET)
print(f"Preset={PRESET}, target trial={N_TRIALS}, jumlah parameter={len(space.params)}")
pd.DataFrame(space_summary(space))
"""
    ),
    md(
        """
## Fungsi objective

Satu trial = satu konfigurasi. Fungsi ini:

1. mengubah dict parameter menjadi `BiLSTMConfig` (validasi otomatis),
2. memakai cache vektorisasi bila parameter Word2Vec sama,
3. melatih model,
4. mengembalikan metrik validation.

Holdout tidak dipanggil di sini.
"""
    ),
    code(
        r"""
from app.experiment.bilstm_pipeline import vectorizer_key

_vector_cache = {}

def objective(params, trial_index, trial_dir):
    config = BiLSTMConfig.from_dict(params)
    key = vectorizer_key(config)
    vectorized = _vector_cache.get(key)
    if vectorized is None:
        # Cache dibatasi agar memori tidak membengkak saat banyak trial.
        if len(_vector_cache) >= 2:
            _vector_cache.clear()
        vectorized = build_vectorizers(bundle, config)
        _vector_cache[key] = vectorized
    result = run_bilstm_trial(
        bundle,
        config,
        trial_dir,
        vectorized=vectorized,
        save_model=False,
        save_predictions=False,
        evaluate_holdout=False,
        verbose=0,
    )
    return {"metrics": result.metrics, "artifacts": result.artifacts}
"""
    ),
    md(
        """
## Jalankan HPO

`timeout_minutes` membatasi waktu. `resume=True` membuat notebook bisa
dilanjutkan tanpa mengulang trial yang sudah sukses.
"""
    ),
    code(
        r"""
HPO_DIR = RUNS_DIR / "bilstm_hpo"
hpo_config = HPOConfig(
    strategy="random",          # coba juga "grid" (jika grid kecil) atau "bayesian"
    n_trials=N_TRIALS,
    seed=42,
    timeout_minutes=120,
    objective_metric="val_f1",
    objective_mode="max",
    output_dir=str(HPO_DIR),
    study_name=f"bilstm_{PRESET}",
    resume=True,
    save_trial_models=False,
)

engine = HPOEngine(hpo_config, space, run_dir=HPO_DIR / f"{PRESET}_seed42", verbose=True)
hpo_result = engine.run(objective)
print("Run dir:", hpo_result.run_dir)
if hpo_result.best is None:
    raise RuntimeError("Tidak ada trial sukses. Cek error di trials.jsonl.")
"""
    ),
    md(
        """
## Leaderboard

Semua trial diurutkan berdasarkan objective. Perhatikan bahwa trial yang
gagal tetap dicatat — ini penting untuk audit.
"""
    ),
    code(
        r"""
leaderboard = pd.read_csv(hpo_result.run_dir / "leaderboard.csv")
columns = ["trial_id", "status", "objective", "metric_val_f1", "metric_val_roc_auc", "metric_val_loss", "duration_sec"]
columns = [column for column in columns if column in leaderboard.columns]
leaderboard[columns].head(15)
"""
    ),
    code(
        r"""
best = hpo_result.best
print("Best trial :", best.trial_id)
print("Best params:")
print(json.dumps(best.params, indent=2, ensure_ascii=False))
print("\nBest metrics:")
print(json.dumps(best.metrics, indent=2, ensure_ascii=False))
"""
    ),
    md(
        """
## Plot progres dan importance

- **Progres**: apakah pencarian menemukan perbaikan dari trial ke trial?
- **Importance**: parameter mana yang paling berpengaruh? Ini bahan bagus
  untuk diskusi paper.
"""
    ),
    code(
        r"""
from app.experiment.reporting import plot_hpo_progress, plot_param_importance

plot_hpo_progress(
    leaderboard,
    hpo_result.run_dir / "hpo_progress.png",
    mode=hpo_config.objective_mode,
)
plot_param_importance(leaderboard, hpo_result.run_dir / "hpo_importance.png")
print("Plot tersimpan di", hpo_result.run_dir)
"""
    ),
    md(
        """
## Retrain model terbaik dan uji holdout sekali

Hyperparameter terbaik dari validation sekarang dilatih ulang pada
final-train, lalu diuji di frozen holdout. Ini satu-satunya saat holdout
dipakai.
"""
    ),
    code(
        r"""
best_config = BiLSTMConfig.from_dict(best.params)
BEST_OUT = RUNS_DIR / "bilstm_tuned"
BEST_MODEL = BEST_OUT / "models"

best_output = train_final_bilstm(
    bundle,
    best_config,
    output_dir=BEST_OUT,
    model_dir=BEST_MODEL,
    hpo_info={
        "source": str(hpo_result.run_dir),
        "best_trial": best.trial_id,
        "best_params": best.params,
        "best_metrics": best.metrics,
        "search_space": space.to_dict(),
        "n_trials": len(hpo_result.trials),
    },
    verbose=1,
)
print(best_output["eval_table"].to_string(index=False))
"""
    ),
    md(
        """
## Bandingkan baseline vs tuned

Kalau notebook 02 sudah dijalankan, kita bisa membandingkan pada holdout yang
sama memakai uji McNemar + paired bootstrap AUC.
"""
    ),
    code(
        r"""
from app.experiment.reporting import compare_two_models, load_probabilities

baseline_path = RUNS_DIR / "bilstm_baseline" / "probs_holdout.csv"
if baseline_path.exists():
    y_base, p_base, _ = load_probabilities(baseline_path)
    comparison = compare_two_models(
        y_base, p_base, best_output["result"].holdout_prob,
        name_a="BiLSTM baseline", name_b="BiLSTM tuned",
        n_bootstrap=2000, seed=42,
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
else:
    print("Jalankan notebook 02 dulu untuk perbandingan baseline.")
"""
    ),
    md(
        """
## Cara melaporkan HPO di paper

Sertakan minimal:

1. **Search space** (tabel parameter + rentang/nilai).
2. **Jumlah trial** dan strategi (grid/random/Bayesian).
3. **Metrik seleksi** (mis. F1 validation) dan arahnya.
4. **Protokol holdout**: holdout tidak dipakai untuk tuning.
5. **Seed** dan varians antar run.
6. **Uji statistik** baseline vs tuned.
7. **Ablasi** synthetic vs real-only (set `synthetic_total=0`).

Contoh kalimat:

> Pencarian hiperparameter dilakukan dengan random search sebanyak 30 trial
> pada ruang 33 parameter (Tabel X). Model dipilih berdasarkan F1 pada
> validation set; frozen holdout (n=50) hanya diprediksi sekali setelah
> konfigurasi terbaik dibekukan. Perbedaan performa terhadap baseline diuji
> dengan McNemar dan paired bootstrap (2.000 resampling).
"""
    ),
    md(
        """
## Peringatan overfitting validation

Validation hanya 50 sampel. Jika Anda menjalankan 100+ trial dan memilih yang
terbaik, sebagian keunggulan bisa berasal dari keberuntungan. Untuk paper:

- jalankan minimal 3 seed,
- laporkan mean ± std,
- pertimbangkan repeated group-aware cross-validation,
- jangan mengubah holdout untuk "memperbaiki" hasil.
"""
    ),
]


# ======================================================================
# 04 — Baseline IndoBERT
# ======================================================================
cells_04 = [
    md(
        """
# 04 — Baseline Fine-Tuning IndoBERT

Notebook ini melatih IndoBERT pada split yang sama dengan BiLSTM.

## Apa yang terjadi saat fine-tuning?

1. Teks dipecah menjadi subword oleh tokenizer IndoBERT.
2. Setiap token menjadi vektor 768 dimensi.
3. 12 lapisan Transformer membaca konteks dua arah.
4. Representasi `[CLS]` masuk ke kepala klasifikasi 2 neuron.
5. Softmax menghasilkan probabilitas `safe` dan `unsafe`.
6. Bobot pre-trained disesuaikan (fine-tuned) dengan data kita.

Karena model jauh lebih besar dari BiLSTM, GPU sangat disarankan.
"""
    ),
    code(SETUP),
    code(LOAD_DATA),
    code(
        r"""
from app.experiment.configs import BertConfig

QUICK_RUN = True

config = BertConfig()
if QUICK_RUN:
    config.epochs = 1
    config.max_len = 128
    config.early_stopping_patience = 1

pd.DataFrame([config.to_dict()]).T.rename(columns={0: "nilai"})
"""
    ),
    md(
        """
## Lihat tokenisasi

Contoh di bawah menunjukkan bagaimana kalimat komposisi dipecah menjadi
subword. Token `[CLS]` dipakai untuk klasifikasi, `[SEP]` menandai akhir
segmen.
"""
    ),
    code(
        r"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
example = bundle.X_val_text[0]
encoded = tokenizer(example, truncation=True, max_length=config.max_len)
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
print("Teks:", example[:300])
print("Jumlah token:", len(tokens))
print(tokens[:60])
"""
    ),
    md(
        """
## Fine-tuning dan evaluasi

Holdout hanya diprediksi sekali di akhir. Early stopping dan pemilihan model
memakai validation.
"""
    ),
    code(
        r"""
from app.experiment.bert_pipeline import train_final_bert

OUT_DIR = RUNS_DIR / "bert_baseline"
output = train_final_bert(
    bundle,
    config,
    output_dir=OUT_DIR,
    hpo_info={"mode": "baseline_notebook"},
    verbose=True,
)
print(json.dumps(output["metrics"]["validation"], indent=2))
print(json.dumps(output["metrics"]["holdout"], indent=2))
"""
    ),
    md(
        """
## Visualisasi hasil

Kurva training BERT dibaca dari `training_log.json`; metrik validation
dihitung setiap epoch. Perhatikan epoch terbaik dan apakah validation loss
mulai naik.
"""
    ),
    code(
        r"""
log = json.loads((OUT_DIR / "training_log.json").read_text(encoding="utf-8"))
eval_rows = [row for row in log if "eval_loss" in row]
train_rows = [row for row in log if "loss" in row and "eval_loss" not in row]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if train_rows:
    axes[0].scatter([row["epoch"] for row in train_rows], [row["loss"] for row in train_rows], label="Train loss")
if eval_rows:
    axes[0].plot([row["epoch"] for row in eval_rows], [row["eval_loss"] for row in eval_rows], marker="o", label="Eval loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

for key, label in (("eval_f1", "F1"), ("eval_roc_auc", "ROC-AUC"), ("eval_average_precision", "AP")):
    points = [(row["epoch"], row[key]) for row in eval_rows if key in row]
    if points:
        axes[1].plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
from app.experiment.reporting import load_probabilities, plot_confusion, plot_roc_pr

y_val, p_val, _ = load_probabilities(OUT_DIR / "probs_val.csv")
y_hold, p_hold, _ = load_probabilities(OUT_DIR / "probs_holdout.csv")

plot_confusion(y_hold, p_hold, OUT_DIR / "confusion_holdout.png", title="IndoBERT baseline — frozen holdout")
plot_roc_pr(
    {"Validation": (y_val, p_val), "Frozen holdout": (y_hold, p_hold)},
    OUT_DIR / "roc_pr.png",
    title="IndoBERT baseline",
)
print("Plot tersimpan di", OUT_DIR)
"""
    ),
    md(
        """
## Catatan GPU dan memori

| Gejala | Solusi |
|---|---|
| CUDA out of memory | Turunkan `batch_size`, `max_len`, atau naikkan `gradient_accumulation_steps` |
| Training sangat lambat | Pastikan `torch.cuda.is_available()` bernilai True |
| Download model gagal | Unduh di login node, set `HF_HUB_OFFLINE=1` |
| Hasil berbeda antar run | Set seed yang sama + `PYTHONHASHSEED` sebelum Python |

Untuk HPO IndoBERT, gunakan notebook 05. Baseline ini menjadi pembanding.
"""
    ),
]


# ======================================================================
# 05 — HPO IndoBERT
# ======================================================================
cells_05 = [
    md(
        """
# 05 — Hyperparameter Tuning IndoBERT (Lengkap)

Notebook ini melakukan HPO untuk fine-tuning IndoBERT dengan protokol yang
sama seperti BiLSTM:

- tuning hanya di train + validation,
- holdout hanya diuji sekali,
- semua trial dicatat,
- model terbaik di-retrain lalu diuji di holdout.

Ruang pencarian mencakup model checkpoint, panjang token, learning rate,
batch, weight decay, warmup, scheduler, optimizer, gradient clipping,
gradient accumulation, freeze layer, dropout, class weight, label smoothing,
presisi (fp32/fp16/bf16), dan metrik seleksi.
"""
    ),
    code(SETUP),
    code(LOAD_DATA),
    code(
        r"""
from app.experiment.configs import BertConfig, HPOConfig
from app.experiment.search_spaces import bert_search_space, preset_trials, space_summary
from app.experiment.hpo import HPOEngine
from app.experiment.bert_pipeline import run_bert_trial, train_final_bert

PRESET = "smoke"       # ganti ke "standard" atau "full" untuk paper
N_TRIALS = preset_trials(PRESET)
space = bert_search_space(PRESET)
print(f"Preset={PRESET}, target trial={N_TRIALS}, jumlah parameter={len(space.params)}")
pd.DataFrame(space_summary(space))
"""
    ),
    md(
        """
## Objective function

Setiap trial memuat tokenizer + model dari checkpoint, melatih, lalu
mengembalikan metrik validation. Model tidak disimpan per trial agar disk
tidak penuh; hanya model terbaik yang disimpan ulang di tahap akhir.
"""
    ),
    code(
        r"""
def objective(params, trial_index, trial_dir):
    config = BertConfig.from_dict(params)
    result = run_bert_trial(
        bundle,
        config,
        trial_dir,
        save_model=False,
        save_predictions=False,
        evaluate_holdout=False,
        verbose=False,
    )
    return {"metrics": result.metrics, "artifacts": result.artifacts}
"""
    ),
    md(
        """
## Jalankan HPO

Untuk GPU 16 GB, mulai dari `smoke`. Setelah pipeline terbukti jalan, naikkan
ke `standard`. Untuk `full`, pertimbangkan menjalankan di HPC dan menyimpan
`run_dir` agar bisa di-resume.
"""
    ),
    code(
        r"""
HPO_DIR = RUNS_DIR / "bert_hpo"
hpo_config = HPOConfig(
    strategy="random",
    n_trials=N_TRIALS,
    seed=42,
    timeout_minutes=240,
    objective_metric="val_f1",
    objective_mode="max",
    output_dir=str(HPO_DIR),
    study_name=f"bert_{PRESET}",
    resume=True,
    save_trial_models=False,
)

engine = HPOEngine(hpo_config, space, run_dir=HPO_DIR / f"{PRESET}_seed42", verbose=True)
hpo_result = engine.run(objective)
print("Run dir:", hpo_result.run_dir)
if hpo_result.best is None:
    raise RuntimeError("Tidak ada trial sukses. Cek error di trials.jsonl.")
"""
    ),
    code(
        r"""
leaderboard = pd.read_csv(hpo_result.run_dir / "leaderboard.csv")
columns = ["trial_id", "status", "objective", "metric_val_f1", "metric_val_roc_auc", "metric_val_loss", "duration_sec"]
columns = [column for column in columns if column in leaderboard.columns]
leaderboard[columns].head(15)
"""
    ),
    md(
        """
## Plot progres dan importance

Grafik ini membantu menjawab: apakah pencarian membaik, dan parameter mana
yang paling berpengaruh? Sertakan interpretasinya di pembahasan paper.
"""
    ),
    code(
        r"""
from app.experiment.reporting import plot_hpo_progress, plot_param_importance

plot_hpo_progress(
    leaderboard,
    hpo_result.run_dir / "hpo_progress.png",
    mode=hpo_config.objective_mode,
)
plot_param_importance(leaderboard, hpo_result.run_dir / "hpo_importance.png")
print("Plot tersimpan di", hpo_result.run_dir)
"""
    ),
    md(
        """
## Retrain model terbaik dan uji holdout

Setelah konfigurasi terbaik dibekukan dari validation, latih ulang dan
evaluasi holdout satu kali.
"""
    ),
    code(
        r"""
best = hpo_result.best
best_config = BertConfig.from_dict(best.params)
BEST_OUT = RUNS_DIR / "bert_tuned"

best_output = train_final_bert(
    bundle,
    best_config,
    output_dir=BEST_OUT,
    hpo_info={
        "source": str(hpo_result.run_dir),
        "best_trial": best.trial_id,
        "best_params": best.params,
        "best_metrics": best.metrics,
        "search_space": space.to_dict(),
        "n_trials": len(hpo_result.trials),
    },
    verbose=True,
)
print(json.dumps(best_output["metrics"]["validation"], indent=2))
print(json.dumps(best_output["metrics"]["holdout"], indent=2))
"""
    ),
    md(
        """
## Bandingkan baseline vs tuned

Notebook 04 harus sudah dijalankan agar perbandingan tersedia.
"""
    ),
    code(
        r"""
from app.experiment.reporting import compare_two_models, load_probabilities

baseline_path = RUNS_DIR / "bert_baseline" / "probs_holdout.csv"
tuned_path = BEST_OUT / "probs_holdout.csv"
if baseline_path.exists() and tuned_path.exists():
    y_base, p_base, _ = load_probabilities(baseline_path)
    y_tuned, p_tuned, _ = load_probabilities(tuned_path)
    comparison = compare_two_models(
        y_base, p_base, p_tuned,
        name_a="IndoBERT baseline", name_b="IndoBERT tuned",
        n_bootstrap=2000, seed=42,
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
else:
    print("Jalankan notebook 04 dulu untuk perbandingan baseline.")
"""
    ),
    md(
        """
## Yang perlu dilaporkan untuk HPO BERT

1. Checkpoint/model yang dicoba.
2. Rentang learning rate, batch, epoch, max_len.
3. Regularisasi: weight decay, dropout, label smoothing.
4. Strategi freeze layer.
5. Presisi (fp32/fp16/bf16) dan hardware.
6. Jumlah trial + strategi.
7. Metrik seleksi dan protokol holdout.
8. Uji statistik vs baseline.

Untuk paper, jalankan minimal 2–3 seed pada konfigurasi terbaik dan laporkan
mean ± std. Jangan memilih model berdasarkan holdout.
"""
    ),
]


# ======================================================================
# 06 — Perbandingan dan laporan
# ======================================================================
cells_06 = [
    md(
        """
# 06 — Perbandingan BiLSTM vs IndoBERT dan Ekspor Laporan

Notebook ini menggabungkan hasil dari notebook 02–05:

- baseline BiLSTM,
- BiLSTM setelah HPO,
- baseline IndoBERT,
- IndoBERT setelah HPO.

Yang dilakukan:

1. Tabel metrik holdout + bootstrap CI.
2. Kurva ROC/PR gabungan.
3. Uji McNemar dan paired bootstrap AUC.
4. Analisis error (false positive/negative).
5. Ekspor tabel + ringkasan Markdown untuk paper.
"""
    ),
    md(
        """
## Prinsip perbandingan

Semua model harus dievaluasi pada **holdout yang sama**. Jangan membandingkan
angka dari split berbeda. Selisih point estimate saja tidak cukup; sertakan
interval kepercayaan dan uji berpasangan.
"""
    ),
    code(SETUP),
    code(
        r"""
from app.experiment.reporting import (
    bootstrap_ci,
    classification_metrics,
    compare_two_models,
    load_probabilities,
    plot_roc_pr,
)

pd.set_option("display.max_colwidth", 200)
"""
    ),
    code(
        r"""
RUN_SPECS = {
    "BiLSTM baseline": RUNS_DIR / "bilstm_baseline" / "probs_holdout.csv",
    "BiLSTM tuned": RUNS_DIR / "bilstm_tuned" / "probs_holdout.csv",
    "IndoBERT baseline": RUNS_DIR / "bert_baseline" / "probs_holdout.csv",
    "IndoBERT tuned": RUNS_DIR / "bert_tuned" / "probs_holdout.csv",
}

available = {name: path for name, path in RUN_SPECS.items() if path.exists()}
missing = [name for name in RUN_SPECS if name not in available]
print("Tersedia:", list(available))
print("Belum ada:", missing)
if not available:
    raise RuntimeError("Belum ada hasil model. Jalankan notebook 02–05 terlebih dahulu.")
"""
    ),
    md(
        """
## Tabel metrik holdout + 95% CI

Setiap baris adalah satu model. CI dihitung dengan stratified bootstrap
(2.000 resampling).
"""
    ),
    code(
        r"""
rows = []
probability_sets = {}
for name, path in available.items():
    y_true, y_prob, _ = load_probabilities(path)
    probability_sets[name] = (y_true, y_prob)
    metrics = classification_metrics(y_true, y_prob)
    ci = bootstrap_ci(y_true, y_prob, n_bootstrap=2000, seed=42)
    ci_map = {row["metric"]: (row["lo"], row["hi"]) for _, row in ci.iterrows()}
    rows.append({
        "model": name,
        "n": int(metrics["n"]),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "f1_ci": f"{ci_map['f1'][0]:.3f}–{ci_map['f1'][1]:.3f}",
        "auc_ci": f"{ci_map['roc_auc'][0]:.3f}–{ci_map['roc_auc'][1]:.3f}",
    })

comparison_table = pd.DataFrame(rows)
comparison_table
"""
    ),
    code(
        r"""
if probability_sets:
    plot_roc_pr(probability_sets, RUNS_DIR / "comparison_roc_pr.png", title="Perbandingan model pada frozen holdout")
    print("Plot tersimpan di", RUNS_DIR / "comparison_roc_pr.png")
"""
    ),
    md(
        """
## Uji statistik berpasangan

- **McNemar**: apakah pola benar/salah dua model berbeda secara signifikan?
- **Paired bootstrap AUC**: selisih ROC-AUC + CI + p-value.

Keduanya memakai sampel holdout yang sama.
"""
    ),
    code(
        r"""
names = list(probability_sets)
statistical_rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name_a, name_b = names[i], names[j]
        y_a, p_a = probability_sets[name_a]
        y_b, p_b = probability_sets[name_b]
        comparison = compare_two_models(
            y_a, p_a, p_b,
            name_a=name_a, name_b=name_b,
            n_bootstrap=2000, seed=42,
        )
        statistical_rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "delta_f1": comparison["delta"]["f1"],
            "delta_auc": comparison["delta"]["roc_auc"],
            "mcnemar_p": comparison["mcnemar"]["p_value"],
            "auc_diff": comparison["paired_bootstrap_auc"]["auc_a_minus_b"],
            "auc_diff_ci": (
                f"{comparison['paired_bootstrap_auc']['ci_lo']:.3f}–"
                f"{comparison['paired_bootstrap_auc']['ci_hi']:.3f}"
            ),
            "auc_diff_p": comparison["paired_bootstrap_auc"]["p_value"],
        })

pd.DataFrame(statistical_rows)
"""
    ),
    md(
        """
## Analisis error

Kita lihat contoh false positive (diprediksi unsafe padahal gold safe) dan
false negative (diprediksi safe padahal gold unsafe). Ini bahan diskusi yang
kuat di pembahasan paper.
"""
    ),
    code(
        r"""
best_model_name = comparison_table.sort_values("f1", ascending=False).iloc[0]["model"]
y_true, y_prob = probability_sets[best_model_name]
pred = (y_prob >= 0.5).astype(int)
probs_df = pd.read_csv(available[best_model_name])

false_positives = probs_df[(y_true == 0) & (pred == 1)].copy()
false_negatives = probs_df[(y_true == 1) & (pred == 0)].copy()

print("Model:", best_model_name)
print("False positives:", len(false_positives))
print(false_positives.head(10).to_string(index=False))
print("\nFalse negatives:", len(false_negatives))
print(false_negatives.head(10).to_string(index=False))
"""
    ),
    md(
        """
## Ekspor untuk paper

Sel ini menulis:

- `comparison_holdout.csv` — tabel metrik,
- `statistical_tests.csv` — uji berpasangan,
- `comparison_summary.md` — ringkasan siap tempel.
"""
    ),
    code(
        r'''
comparison_table.to_csv(RUNS_DIR / "comparison_holdout.csv", index=False)
pd.DataFrame(statistical_rows).to_csv(RUNS_DIR / "statistical_tests.csv", index=False)

best_row = comparison_table.sort_values("f1", ascending=False).iloc[0]
summary = f"""# Ringkasan Perbandingan Model

Holdout: frozen, n = {int(best_row['n'])}.

| Model | F1 | CI F1 | ROC-AUC | CI AUC |
|---|---:|---|---:|---|
"""
for _, row in comparison_table.iterrows():
    summary += f"| {row['model']} | {row['f1']:.3f} | {row['f1_ci']} | {row['roc_auc']:.3f} | {row['auc_ci']} |\n"

summary += f"""
Model dengan F1 tertinggi: **{best_row['model']}**.

Catatan: perbedaan antar model harus dibaca bersama uji McNemar dan paired
bootstrap AUC di `statistical_tests.csv`. Jangan mengklaim keunggulan hanya
dari selisih point estimate.
"""
(RUNS_DIR / "comparison_summary.md").write_text(summary, encoding="utf-8")
print(summary)
'''
    ),
    md(
        """
## Checklist laporan paper

- [ ] Search space HPO dicantumkan (parameter + rentang).
- [ ] Jumlah trial dan strategi disebutkan.
- [ ] Metrik seleksi validation disebutkan.
- [ ] Holdout tidak dipakai untuk tuning.
- [ ] Setiap metrik menyebut n + 95% CI.
- [ ] Perbandingan model memakai uji berpasangan.
- [ ] Ada ablasi synthetic vs real-only.
- [ ] Ada analisis error (false positive/negative).
- [ ] Seed dan varians antar run dilaporkan.
- [ ] Keterbatasan (n kecil, data sintetik, OCR noise) ditulis jujur.
"""
    ),
]


def main() -> None:
    write("00_panduan_dan_setup.ipynb", cells_00)
    write("01_data_dan_split.ipynb", cells_01)
    write("02_bilstm_dasar.ipynb", cells_02)
    write("03_bilstm_tuning.ipynb", cells_03)
    write("04_indobert_dasar.ipynb", cells_04)
    write("05_indobert_tuning.ipynb", cells_05)
    write("06_perbandingan_dan_laporan.ipynb", cells_06)


if __name__ == "__main__":
    main()
