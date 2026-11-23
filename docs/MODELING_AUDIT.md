# Audit End-to-End Pipeline Pemodelan (BiLSTM + IndoBERT)

Tanggal audit: 2026-11-21  
Ruang lingkup: `bu-dian-alergen-ml-training` (data, split, training, evaluasi,
artefak, test, export serving, dan paper builder).  
Metode: pembacaan kode statis, inspeksi artefak, penelusuran kontrak
training→serving, dan eksekusi test ringan yang tidak membutuhkan TensorFlow.

---

## 1. Ringkasan Eksekutif

Pipeline saat ini **secara ilmiah sudah jauh lebih baik** daripada notebook awal:
split leakage-safe, frozen holdout, audit kebocoran, threshold tetap 0,5, dan
manifest eksperimen sudah ada. Namun ada dua masalah besar yang menjelaskan
kenapa paper bisa ditolak:

1. **Hyperparameter tuning praktis tidak ada.** Konfigurasi final dikunci mati
   di `V5Config`, dan `audit_final_config_v5()` justru **menggagalkan run** bila
   nilai seperti `learning_rate`, `mask_zero`, `gradient_clip_norm`, atau
   `w2v_min_count` diubah. Jadi bukan hanya "opsinya sedikit" — secara kode,
   perubahan hiperparameter malah dianggap pelanggaran kontrak.
2. **IndoBERT hanya punya 6 argumen CLI** (`epochs`, `lr`, `batch_size`,
   `max_len`, `seed`, `patience`). `weight_decay`, `warmup`, `scheduler`,
   `dropout`, `freeze layer`, `class weight`, `label smoothing`, dan pemilihan
   metrik model semuanya hardcoded. Tidak ada pencarian hiperparameter sama
   sekali untuk BERT.

Selain itu, ada beberapa isu yang bisa melemahkan klaim paper meskipun modelnya
sudah bagus: urutan training BiLSTM dan BERT tidak setara, `PYTHONHASHSEED`
diset setelah interpreter berjalan (tidak efektif), tuning hanya memakai 50
sampel validasi tanpa protokol berulang, dan perbandingan model tidak memakai
uji statistik berpasangan.

**Status: PASS WITH MAJOR GAPS.** Fondasi data sudah kuat; yang perlu
direfactor adalah konfigurasi, orkestrasi training, HPO, dan pelaporan.

Baseline verifikasi:

| Pemeriksaan | Hasil |
|---|---|
| `pytest -q` | 24 passed, 1 skipped (test TensorFlow di-skip) |
| TensorFlow/Gensim di environment audit | Tidak tersedia (Python 3.14) |
| Artefak BiLSTM final | Ada: metrics, threshold, manifest, triage, CI |
| Artefak IndoBERT | Ada 2 run (`results`, `results_2`) dengan hasil berbeda |
| Notebook | Tidak ada; semua lewat CLI |
| Serving contract | Ada `export_for_serving.py`, tetapi ada hardcode `max_len`/`vocab` |

---

## 2. Yang Sudah Bagus (Pertahankan)

- **Leakage-safe split** memakai union-find pada `group_key` dan hash teks.
- **Frozen holdout** dikunci per nama produk + hash anti-tamper.
- **Synthetic hanya dari real-train**, bukan dari val/holdout.
- **Tokenizer + Word2Vec hanya fit di final-train** (real + synthetic).
- **Threshold evaluasi tetap 0,5** dan tidak di-tuning di holdout.
- **Audit eksplisit**: group leakage, exact-text overlap, split manifest.
- **Bootstrap CI** sudah ada untuk BiLSTM dan BERT.
- **Test kontrak** untuk threshold, frozen hash, audit, synthetic, dan metrik.
- **Pemisahan artefak** `models/` dan `output/` cukup rapi.

Fondasi ini tidak perlu dibongkar. Refactor harus **mempertahankan kontrak
leakage-safe** sambil membuka ruang HPO yang terkontrol.

---

## 3. Temuan Audit

Skala severity:

- **P0** = merusak hasil / wajib diperbaiki sebelum klaim ilmiah.
- **P1** = menghambat HPO, reproduksibilitas, atau validitas klaim.
- **P2** = risiko maintainability / konsistensi / kualitas laporan.
- **P3** = perbaikan kecil.

### 3.1 Temuan Prioritas Tinggi

| ID | Severity | Area | Temuan | Bukti | Dampak | Rekomendasi |
|---|---|---|---|---|---|---|
| A1 | P1 | HPO | HPO BiLSTM diblokir oleh audit konfigurasi final | `app/core/model/audit.py:173-194` meng-hard-fail `learning_rate==1e-4`, `mask_zero=True`, `gradient_clip_norm==1.0`, `w2v_min_count==1`, dll. | Setiap trial HPO akan error sebelum training. | Pisahkan "kontrak final" dan "mode eksperimen"; audit final hanya untuk run pelaporan, bukan HPO. |
| A2 | P1 | HPO | Hanya satu konfigurasi BiLSTM yang bisa dijalankan | `app/config.py:70-114`: `vocab_size`, `embed_dim`, `w2v_window`, `w2v_epochs`, `lstm_units_1/2`, dropout, dense, early-stopping, optimizer semuanya hardcoded. | Tidak ada ruang pencarian; paper mudah ditolak. | Buat `BiLSTMConfig` lengkap + search space grid/random/Bayesian. |
| A3 | P1 | HPO | IndoBERT tidak punya HPO dan hanya 6 opsi CLI | `artifacts/bert/train_bert_hpc.py:100-107`; `weight_decay`, `warmup_ratio`, `scheduler`, `dropout`, `freeze layer`, `class weight`, `label smoothing`, `fp16/bf16` hardcoded di `:142-168`. | Klaim "IndoBERT" hanya satu resep; tidak ada bukti eksplorasi. | Buat `BertConfig` + search space lengkap + runner HPO. |
| A4 | P1 | Perbandingan | Urutan training BiLSTM vs BERT tidak setara | BiLSTM `train_shuffle=False` (`app/training/trainer_v5.py:233-237`), synthetic ditempel di akhir pool (`app/core/data/synthetic.py:119-126`); BERT `train_combined.csv` di-shuffle (`scripts/rebuild_split.py:86-91`). | Perbedaan performa bisa berasal dari urutan data, bukan arsitektur. | Default `shuffle=True` untuk kedua model, atau laporkan ablasi urutan. |
| A5 | P1 | Reproducibility | `PYTHONHASHSEED` diset saat runtime, tidak efektif | `app/repro.py:24-34`; Python menetapkan hash seed saat startup. Audit hanya mengecek nilai env, bukan perilaku proses. | Klaim reproducible bisa false-positive. | Dokumentasikan `PYTHONHASHSEED` di shell/launcher; tambahkan audit proses (mis. `sys.flags.hash_randomization`) dan catat di manifest. |
| A6 | P1 | Metodologi | Tuning hanya di 50 sampel validasi, tanpa pengulangan/nested protocol | Split val 25/25 (`app/config.py:47-56`); HPO akan memilih dari val yang sama berulang kali. | Selection bias; model "terbaik" bisa overfit ke val. | Gunakan repeated seeds / repeated group-aware CV untuk HPO, atau minimal laporkan varians dan gunakan holdout hanya sekali. |
| A7 | P1 | Klaim ilmiah | Tidak ada uji statistik berpasangan antar model | Hanya bootstrap CI marginal (`app/core/model/metrics.py:314-380`); tidak ada McNemar/DeLong/paired bootstrap. | Klaim "BiLSTM lebih baik" tidak terbukti secara statistik. | Tambahkan McNemar + paired bootstrap AUC difference + CI. |
| A8 | P1 | Serving parity | `max_len` dan `vocab_size` serving hardcoded, tidak dibaca dari artefak training | `bu-dian-alergen-ml-service/app/config.py:8-10`; `detection_pipeline.py:276-283` memakai `settings.MAX_LEN`/`VOCAB_SIZE`. | Jika HPO memilih `max_len`/`vocab` lain, inference bisa salah tanpa error. | Simpan `max_len`, `vocab_size`, `model_type` di `metadata.json`/`model_card.json`; serving wajib membacanya. |
| A9 | P1 | Provenance | Dua run BERT dengan konfigurasi sama menghasilkan metrik berbeda | `artifacts/bert/results/metrics.json` holdout acc 0,90 vs `results_2` 0,96; `model_card.json` tidak mencatat versi `transformers`, git SHA, atau command. | Hasil tidak bisa diaudit ulang; sulit menjawab reviewer. | Catat environment, git SHA, hash data, dan command di setiap run; buat run directory unik. |

### 3.2 Temuan Prioritas Menengah

| ID | Severity | Area | Temuan | Bukti | Dampak | Rekomendasi |
|---|---|---|---|---|---|---|
| B1 | P2 | Maintainability | `trainer_v5.py` monolitik, campur data/vektorisasi/model/training/evaluasi | `app/training/trainer_v5.py` 323 baris | Sulit dites, diubah, dan dipahami pemula. | Pecah jadi `prepare_data`, `build_vectorizers`, `run_trial`, `train_final`, `evaluate`. |
| B2 | P2 | Tokenisasi | Word2Vec memakai `simple_tokenize`, Keras Tokenizer memakai filter bawaan | `trainer_v5.py:172-174` vs `app/core/model/tokenizer.py:14` | Distribusi token bisa beda; coverage W2V turun. | Satukan fungsi tokenisasi atau dokumentasikan perbedaannya + audit coverage. |
| B3 | P2 | Data | Synthetic 1000 vs real 399 (71,5% synthetic); belum ada ablasi BiLSTM tanpa synthetic | `artifacts/bert/input/split_contract.json`; BERT punya `train_real.csv`, BiLSTM tidak. | Model bisa belajar aturan KB, bukan pola komposisi nyata. | Tambahkan opsi `use_synthetic=False` dan laporkan ablasi. |
| B4 | P2 | Artefak | Skema file tidak konsisten | BiLSTM `val_probs.csv` (`label_id,prob_unsafe`) vs BERT `probs_val.csv` (`text,label,prob_unsafe`); `threshold.json` vs `thresholds.json`. | Script paper harus menangani banyak format. | Standarkan skema `probs_*.csv` dan nama `threshold(s).json`; simpan alias kompatibilitas. |
| B5 | P2 | Split | Hasil split bergantung urutan baris CSV | `run_v5_split` memakai index baris; komponen dipilih dari urutan `min(root)`. | CSV yang isinya sama tapi urutannya beda bisa menghasilkan val berbeda. | Sort sumber secara deterministik (mis. `group_key`, hash teks) sebelum split. |
| B6 | P2 | Tracking | Tidak ada run directory / trial log / resume | `output_dir` ditimpa setiap run. | Tidak bisa membandingkan trial atau mengulang run tertentu. | Buat `artifacts/runs/<timestamp>_<seed>/`, `trials.jsonl`, `leaderboard.csv`, resume. |
| B7 | P2 | Test | Tidak ada test untuk arsitektur, trainer, BERT, atau notebook | `tests/` hanya kontrak data/metrik; test TF di-skip. | Refactor bisa merusak training tanpa ketahuan. | Tambahkan test search-space, HPO engine dengan dummy trainer, config validation, notebook JSON. |
| B8 | P2 | Tooling | Tidak ada `pyproject.toml`, lint/typecheck, CI, atau pemisahan requirements | `requirements.txt` mencampur TF + torch; versi BERT tidak dipin. | Onboarding lambat; environment mudah drift. | Tambahkan `pyproject.toml`, `requirements-train.txt`, `requirements-bert.txt`, `requirements-dev.txt`, CI. |
| B9 | P2 | Paper | Paper builder masih menyatakan "tanpa pencarian acak" | `paper/build_paper.py:306` | Bertentangan dengan refactor HPO. | Setelah HPO selesai, ubah metodologi + tabel search space + jumlah trial. |
| B10 | P2 | Evaluasi | Tidak ada kalibrasi, error analysis per kategori, atau robustness test | Hanya metrik agregat + triage. | Reviewer bisa meminta analisis lebih dalam. | Tambahkan Brier score, reliability curve, error per kategori alergen/panjang teks, uji typo/case. |
| B11 | P2 | Split | Pencarian subset komponen memakai rekursi eksponensial | `app/core/data/leakage_split.py:114-143` | Untuk dataset jauh lebih besar bisa lambat/recursion limit. | Tambahkan guard ukuran + fallback greedy/ILP; dokumentasikan batas. |
| B12 | P2 | Config | Duplikasi `text_col` di `Config` dan `V5Config`, fallback tidak konsisten | `app/config.py:119,136`; trainer memakai `config.text_col` lalu fallback `v5.text_col` | Potensi bug saat nama kolom dikustomisasi. | Satu sumber kebenaran `DataConfig.text_col`. |

### 3.3 Temuan Prioritas Rendah

| ID | Severity | Temuan | Rekomendasi |
|---|---|---|---|
| C1 | P3 | Docstring banyak merujuk "Cell 36/47/72" yang tidak bermakna bagi pembaca baru | Ganti dengan penjelasan bahasa manusia. |
| C2 | P3 | Bahasa log/error campur Indonesia-Inggris | Standarkan bahasa Indonesia untuk pesan user, Inggris untuk nama teknis. |
| C3 | P3 | Magic numbers (0.6 init scale, 92.0 fuzzy, 1e-8) | Jadikan konstanta bernama. |
| C4 | P3 | `results` vs `results_2` tanpa penjelasan | Tambahkan `README` run atau hapus salah satu setelah final. |
| C5 | P3 | `paper/` membaca artefak yang di-ignore git | Dokumentasikan regenerasi atau gunakan artefak kecil yang di-track. |
| C6 | P3 | `capture_training.ps1` hanya Windows | Sediakan padanan bash untuk macOS/Linux. |
| C7 | P3 | Tidak ada benchmark latency/model size | Tambahkan di laporan perbandingan. |

---

## 4. Analisis Gap HPO

### 4.1 BiLSTM — kondisi saat ini vs target

| Kelompok | Saat ini | Target HPO |
|---|---|---|
| Data/vektorisasi | `vocab_size`, `embed_dim`, `w2v_window`, `w2v_epochs` fixed | `vocab_size`, `max_len`, `embed_dim`, `w2v_window`, `w2v_min_count`, `w2v_epochs`, `w2v_sg` (CBOW/skip-gram), `embed_trainable`, `mask_zero`, `digit_fold` |
| Arsitektur | BiLSTM 128→64, dropout 0,3/0,3, dense 64 | `rnn_type` (LSTM/GRU), `use_bidirectional`, `num_rnn_layers`, units 1/2, dropout 1/2, recurrent dropout, dense units, dense dropout, `use_maxnorm` |
| Optimasi | Adam, LR 1e-4, batch 16, clip 1,0 | optimizer (Adam/AdamW/RMSprop/SGD), LR log-uniform, weight decay, batch, gradient clip, label smoothing, class weight |
| Callback | Early stop 4, ReduceLROnPlateau 3 | patience, factor, min LR, monitor |
| Ordering | `shuffle=False` | `shuffle=True` (default) + opsi preshuffle |
| Pemilihan | `val_loss` | `val_loss`, `val_f1`, `val_roc_auc`, `val_average_precision` |

Strategi yang disediakan: **grid**, **random**, dan **Bayesian (Optuna TPE,
opsional)**. Preset: `smoke` (cepat), `standard` (30–50 trial), `full`
(100–300 trial).

### 4.2 IndoBERT — kondisi saat ini vs target

| Kelompok | Saat ini | Target HPO |
|---|---|---|
| Model | `indobert-base-p1` fixed | base-p1, base-p2, large-p2, lite (opsional) |
| Tokenisasi | `max_len=256` fixed | 128–512 |
| Optimasi | LR 2e-5, batch 16, epoch 4, weight decay 0,01, warmup 0,1 | LR log-uniform, batch 8/16/32, epoch 3–10, weight decay, warmup, scheduler (linear/cosine/polynomial/constant), optimizer (AdamW/Adafactor), max grad norm, gradient accumulation |
| Regularisasi | dropout model default | `hidden_dropout_prob`, `attention_probs_dropout_prob`, `classifier_dropout`, `label_smoothing_factor` |
| Arsitektur | full fine-tuning | `freeze_layers` (0, embeddings, first N encoder layers) |
| Kelas | tanpa class weight | `use_class_weight` |
| Hardware | tanpa fp16/bf16 | `fp16`, `bf16` (dengan validasi CUDA) |
| Pemilihan | `eval_loss` | `eval_loss`, `eval_f1`, `eval_roc_auc`, `eval_average_precision` |
| Data | `train_combined.csv` | sama, plus opsi `train_real.csv` untuk ablasi |

---

## 5. Rencana Refactor

### 5.1 Prinsip

1. **Kontrak leakage-safe tidak berubah.** Holdout tetap beku; HPO hanya
   menyentuh train + val.
2. **Kode berat tetap di modul**, bukan di notebook. Notebook jadi alat
   belajar + orkestrasi yang mudah dibaca.
3. **Satu sumber kebenaran konfigurasi** (`BiLSTMConfig`, `BertConfig`,
   `DataConfig`) dengan validasi.
4. **Setiap run punya direktori sendiri** + log trial + manifest environment.
5. **Backward compatibility**: CLI lama dan nama artefak final tetap jalan.

### 5.2 Struktur Baru (ringkas)

```
app/experiment/
  configs.py          # DataConfig, BiLSTMConfig, BertConfig, HPOConfig
  search_spaces.py    # ParamSpec, grid/random/Bayesian, preset smoke/standard/full
  hpo.py              # engine trial, resume, leaderboard, best config
  data_pipeline.py    # load -> gold -> split -> synthetic -> bundle
  bilstm_pipeline.py  # vectorizer, model, trial, final training
  bert_pipeline.py    # tokenizer, HF trainer, trial, final training
  reporting.py        # metrik, CI, McNemar, paired bootstrap, plot
  artifacts.py        # run dir, provenance, JSON/CSV/JSONL writer
notebooks/
  00_panduan_dan_setup.ipynb
  01_data_dan_split.ipynb
  02_bilstm_dasar.ipynb
  03_bilstm_tuning.ipynb
  04_indobert_dasar.ipynb
  05_indobert_tuning.ipynb
  06_perbandingan_dan_laporan.ipynb
```

### 5.3 Notebook Map

| Notebook | Untuk siapa | Isi | Estimasi |
|---|---|---|---|
| 00 | Semua | Istilah, setup, peta project, aturan eksperimen | 10 menit |
| 01 | Semua | Load data, EDA, KB audit, split, synthetic, simpan input BERT | 10–20 menit |
| 02 | Pemula | Baseline BiLSTM + Word2Vec, training, evaluasi | 10–30 menit (GPU lebih cepat) |
| 03 | Peneliti | HPO BiLSTM lengkap, leaderboard, retrain, CI, uji statistik | 1–6 jam tergantung preset |
| 04 | Pemula | Baseline IndoBERT, tokenizer, fine-tuning, evaluasi | 10–40 menit (GPU) |
| 05 | Peneliti | HPO IndoBERT lengkap, leaderboard, retrain, CI | 2–12 jam (GPU) |
| 06 | Peneliti | Perbandingan BiLSTM vs IndoBERT, error analysis, export laporan | 10 menit |

---

## 6. Rencana Validasi

Yang bisa divalidasi tanpa GPU/TensorFlow:

- Unit test search space: sampling valid, grid unik, parameter di luar batas
  ditolak.
- Unit test config: validasi dropout/learning rate/freeze layer.
- Unit test HPO engine dengan **dummy trainer** (fungsi matematis): logging,
  resume, pemilihan best, timeout, trial gagal.
- Unit test metrik: bootstrap CI, McNemar, paired bootstrap.
- Validasi JSON semua notebook.
- `pytest` lama harus tetap lulus.

Yang membutuhkan environment training (harus dijalankan user):

- Smoke training BiLSTM 1–2 epoch.
- Smoke fine-tuning IndoBERT 1 epoch pada GPU.
- HPO `smoke` preset (4–6 trial) untuk kedua model.
- Run final + evaluasi holdout + bootstrap CI.

---

## 7. Risiko dan Batasan Audit

- Audit ini **tidak menjalankan training TensorFlow** karena environment audit
  memakai Python 3.14 dan TensorFlow/Gensim tidak tersedia. Temuan training
  berbasis pembacaan kode + artefak, bukan eksekusi ulang.
- Hasil BERT di `results` dan `results_2` tidak bisa direkonsiliasi tanpa log
  environment; ini justru salah satu temuan A9.
- Holdout hanya 50 sampel. Semua klaim harus disertai CI dan diuji berpasangan;
  jangan membandingkan point estimate saja.
- HPO pada dataset kecil mudah overfit ke validation. Preset `full` harus
  disertai repeated seeds dan pelaporan varians.

---

## 8. Urutan Pengerjaan yang Disarankan

1. Buat `app/experiment` (config, search space, HPO engine, pipeline).
2. Refactor trainer lama agar memakai pipeline baru (backward compatible).
3. Tambahkan notebook 00–06.
4. Tambahkan test + `pyproject.toml` + requirements terpisah.
5. Jalankan smoke test di environment Python 3.12 + TensorFlow.
6. Jalankan HPO `smoke`, lalu `standard`/`full` di GPU.
7. Perbarui paper: search space, jumlah trial, protokol seleksi, CI, uji
   statistik, dan ablasi synthetic.

---

## 9. Kesimpulan

Fondasi data dan anti-leakage sudah layak. Masalah utama bukan pada splitting,
tetapi pada **kurangnya ruang eksplorasi hiperparameter, orkestrasi eksperimen,
dan pelaporan statistik**. Refactor yang diusulkan menjaga kontrak ilmiah yang
sudah ada, membuka HPO yang lengkap dan terdokumentasi, serta membuat pipeline
bisa dipahami pemula lewat notebook.

---

## 10. Status Setelah Refactor (Post-Audit)

Refactor yang diusulkan pada Bagian 5 sudah dieksekusi. Ringkasan:

### 10.1 Modul baru

| Modul | Fungsi |
|---|---|
| `app/experiment/configs.py` | `DataConfig`, `BiLSTMConfig`, `BertConfig`, `HPOConfig` + validasi |
| `app/experiment/search_spaces.py` | 33 parameter BiLSTM, 21 parameter IndoBERT, preset smoke/standard/full |
| `app/experiment/hpo.py` | engine grid/random/Bayesian, trial log, leaderboard, resume, timeout |
| `app/experiment/data_pipeline.py` | load → gold → split → synthetic → audit → CSV BERT |
| `app/experiment/bilstm_pipeline.py` | vektorisasi, training trial, final training, artefak |
| `app/experiment/bert_pipeline.py` | fine-tuning HF, freeze layer, class weight, label smoothing, presisi |
| `app/experiment/reporting.py` | metrik, bootstrap CI, McNemar, paired bootstrap AUC, plot HPO |
| `app/experiment/artifacts.py` | run directory, provenance, writer JSON/CSV/JSONL |

### 10.2 Notebook

Tujuh notebook dibuat di `notebooks/` (00–06). Semua kode notebook sudah
divalidasi sintaksnya; notebook 00 dan 01 juga dijalankan end-to-end dengan
data asli pada direktori sementara.

### 10.3 Perbaikan determinisme

Ditemukan sumber non-determinisme tambahan saat validasi: `app/core/kb/rules.py`
mengiterasi `set(...)` untuk fuzzy matching, sehingga pilihan `best_term` bisa
berbeda antar proses meskipun seed sama. Ini membuat data sintetik dan
`split_contract.json` berubah antar `PYTHONHASHSEED`.

Perbaikan: iterasi memakai `dict.fromkeys(...)` yang menjaga urutan.

Bukti:
- `PYTHONHASHSEED=1`, `=2`, `=3` menghasilkan hash data sintetik yang **identik**.
- `scripts/rebuild_split.py` dijalankan dua kali dengan hash seed berbeda
  menghasilkan `split_contract.json` dan `synthetic.csv` yang **identik**.

Catatan: artefak lama di `artifacts/bert/input/` dan hasil BERT lama dibuat
sebelum perbaikan ini. Regenerate split dan latih ulang BERT sekali agar hasil
konsisten dengan kode saat ini.

### 10.4 Guard protokol holdout

Ditemukan saat audit lanjutan: runner trial HPO awalnya tetap memprediksi
holdout meskipun pemilihan memakai validation. Ini melanggar aturan "holdout
hanya diprediksi sekali". Perbaikan:

- `run_bilstm_trial(..., evaluate_holdout=False)` dan
  `run_bert_trial(..., evaluate_holdout=False)` dipakai semua objective HPO.
- `HPOEngine` menolak trial yang mengembalikan metrik berawalan
  `holdout_`/`test_` (`forbid_holdout_metrics=True`).
- Training final tetap `evaluate_holdout=True` dan diakhiri assert bahwa
  holdout benar-benar dievaluasi.

### 10.5 Validasi

| Pemeriksaan | Hasil |
|---|---|
| `pytest -q` | 47 passed, 1 skipped (48 collected) |
| Import modul baru tanpa TF/Torch | PASS |
| Notebook JSON + syntax | PASS (7 notebook) |
| Data pipeline dengan CSV 499 baris | PASS, split 399/50/50 |
| Determinisme synthetic lintas hash seed | PASS (hash identik) |
| Determinisme split saat urutan CSV diacak (`sort_source=True`) | PASS (train/val/holdout identik) |
| Training BiLSTM/IndoBERT | Belum dieksekusi di environment audit (TF/GPU tidak tersedia) |

### 10.6 Yang masih perlu dijalankan user

1. Smoke test BiLSTM 1–2 epoch di environment Python 3.12 + TensorFlow.
2. Smoke fine-tuning IndoBERT 1 epoch di GPU.
3. HPO `smoke` → `standard` → `full` sesuai waktu.
4. Regenerate split BERT setelah perbaikan determinisme.
5. Perbarui paper: search space, jumlah trial, protokol seleksi, CI, uji
   statistik, dan ablasi synthetic.

---

## 11. Audit Lanjutan: Import, Runtime Notebook, dan Cross-Project

Audit ini dilakukan setelah push pertama, karena ditemukan beberapa error
import saat notebook dijalankan.

### 11.1 Temuan import notebook

| ID | Severity | Temuan | Perbaikan |
|---|---|---|---|
| N1 | P1 | Notebook 02, 03, 04, 05 memakai `pd`/`plt` tanpa import di sel tersebut | Semua notebook sekarang mengimpor `numpy`, `pandas`, dan `matplotlib.pyplot` di sel setup pertama. |
| N2 | P1 | Notebook 03/05 memakai `pd` sebelum sel `import pandas as pd` | Import dipindahkan ke setup; duplikasi dihapus. |
| N3 | P2 | Notebook 02/04 memakai `plt` tanpa import | Import `matplotlib.pyplot` ditambahkan ke setup. |
| N4 | P2 | Notebook 06 mengimpor ulang `json`/`numpy`/`pandas`/`plt` | Duplikasi dihapus; setup menjadi satu sumber import umum. |
| N5 | P2 | Pengguna bisa menjalankan sel tengah dan mendapat `NameError` | `notebooks/README.md` dan notebook 00 sekarang menekankan **Restart & Run All**. |

### 11.2 Temuan runtime Python

| ID | Severity | Temuan | Perbaikan |
|---|---|---|---|
| R1 | P1 | `app/experiment/bert_pipeline.py` memakai closure `trainer` yang kemudian di-`del`; pyflakes menandai undefined name dan berisiko `NameError` jika urutan berubah | `_predict()` sekarang menerima `trainer` sebagai argumen eksplisit. |
| R2 | P1 | Setiap trial HPO tetap memprediksi holdout | Ditambahkan `evaluate_holdout=False` untuk semua objective HPO; `HPOEngine` menolak metrik `holdout_*`/`test_*`. Training final tetap mengevaluasi holdout sekali dan diakhiri assert. |
| R3 | P2 | `_require_bert_stack()` mengimpor `torch`/`transformers` hanya untuk cek | Diganti `importlib.util.find_spec()` agar tidak memuat library berat. |
| R4 | P2 | Unused import di `word2vec.py`, `scripts/train.py`, `paper/`, `train_bert_hpc.py` | Dibersihkan; pyflakes bersih. |
| R5 | P3 | `plot_param_importance` memunculkan `RuntimeWarning` untuk parameter konstan | Ditambahkan guard `std == 0`; diuji dengan `-W error::RuntimeWarning`. |

### 11.3 Validasi eksekusi notebook (dengan mock untuk bagian berat)

| Notebook | Metode | Hasil |
|---|---|---|
| 00 | eksekusi penuh | PASS |
| 01 | eksekusi penuh + data asli 499 baris | PASS |
| 02 | mock vektorisasi + training | PASS |
| 03 | mock trial + HPO engine 4 trial | PASS |
| 04 | mock fine-tuning (tokenizer download dilewati) | PASS |
| 05 | mock trial + HPO engine 4 trial | PASS |
| 06 | data probabilitas dummy 4 model | PASS |

Mock dipakai hanya karena TensorFlow/GPU tidak tersedia di environment audit;
tujuannya memvalidasi alur kode, import, dan penamaan variabel, bukan kualitas
model.

### 11.4 Audit cross-project

| Proyek | Pemeriksaan | Hasil |
|---|---|---|
| `bu-dian-alergen-ml-training` | `pytest`, `pyflakes`, compile, notebook syntax | PASS (47 test, 1 skip) |
| `bu-dian-alergen-ml-service` | `pyflakes`, `compileall` | PASS |
| `bu-dian-alergen-backend` | `npm run typecheck`, `npm run test:unit` | PASS (40 test) |
| `bu-dian-alergen-web` | `npm run typecheck`, `npm test` | PASS (56 test) |
| `bu-dian-alergen-mobile` | `flutter analyze` | PASS (no issues) |
| `old/` (legacy) | `pyflakes` | Hanya unused import; tidak ada undefined name |

### 11.5 Catatan

- Pyflakes pada notebook hanya menyisakan **unused import** dari setup umum
  (`numpy`/`matplotlib` di notebook yang tidak memakainya). Ini disengaja agar
  setiap notebook siap dipakai dan tidak ada `NameError`.
- `old/` adalah kode legacy dan tidak dipakai pipeline baru.
- Training TensorFlow/IndoBERT tetap perlu dijalankan di environment Python
  3.12 + GPU/HPC untuk validasi kualitas model sebenarnya.
