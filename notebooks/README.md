# Notebook Pembelajaran & Eksperimen

Folder ini berisi notebook yang menjelaskan pipeline pemodelan dari nol.
Kode berat ada di `app/experiment/`; notebook hanya mengatur alur, menampilkan
hasil, dan memberi penjelasan.

## Urutan menjalankan

| # | Notebook | Waktu | Prasyarat |
|---|---|---|---|
| 00 | `00_panduan_dan_setup.ipynb` | 10 menit | tidak ada |
| 01 | `01_data_dan_split.ipynb` | 10–20 menit | CSV dataset |
| 02 | `02_bilstm_dasar.ipynb` | 10–30 menit | environment TensorFlow |
| 03 | `03_bilstm_tuning.ipynb` | 1–6 jam | notebook 02 (untuk pembanding) |
| 04 | `04_indobert_dasar.ipynb` | 10–40 menit | GPU + `requirements-bert.txt` |
| 05 | `05_indobert_tuning.ipynb` | 2–12 jam | notebook 04 |
| 06 | `06_perbandingan_dan_laporan.ipynb` | 10 menit | notebook 02–05 |

## Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -r requirements-notebook.txt

# BERT (terpisah, disarankan GPU)
pip install -r requirements-bert.txt

# set sebelum Python berjalan agar hash deterministik
export PYTHONHASHSEED=42
jupyter lab
```

## Output

| Folder | Isi |
|---|---|
| `artifacts/data/` | ringkasan data, audit split, konfigurasi data |
| `artifacts/bert/input/` | CSV split identik untuk IndoBERT |
| `artifacts/notebook_runs/bilstm_baseline/` | model + metrik baseline BiLSTM |
| `artifacts/notebook_runs/bilstm_hpo/` | trial log + leaderboard HPO BiLSTM |
| `artifacts/notebook_runs/bilstm_tuned/` | model final hasil HPO BiLSTM |
| `artifacts/notebook_runs/bert_baseline/` | model + metrik baseline IndoBERT |
| `artifacts/notebook_runs/bert_hpo/` | trial log + leaderboard HPO IndoBERT |
| `artifacts/notebook_runs/bert_tuned/` | model final hasil HPO IndoBERT |

Semua folder di atas bersifat **generated** dan tidak di-commit.

## Aturan penting

1. Jalankan notebook 01 dulu; notebook lain membaca `artifacts/data/data_config.json`.
2. Jangan mengubah split di tengah eksperimen. Kalau data berubah, regenerate
   dan latih ulang semua model.
3. Holdout hanya dipakai sekali di akhir.
4. Untuk paper, gunakan preset `standard` atau `full`, bukan `smoke`.
5. Laporkan CI dan uji statistik, bukan hanya point estimate.

## Regenerasi notebook

Notebook di folder ini dihasilkan dari satu sumber agar konsisten:

```bash
python scripts/build_notebooks.py
```

Edit `scripts/build_notebooks.py`, lalu jalankan ulang perintah di atas.
