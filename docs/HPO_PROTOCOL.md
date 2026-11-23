# Protokol Hyperparameter Tuning (HPO) — BiLSTM & IndoBERT

Dokumen ini adalah versi siap-pakai untuk metodologi paper. Isinya menjelaskan
ruang pencarian, protokol seleksi, cara menjalankan, dan cara melaporkan hasil.

---

## 1. Prinsip Dasar

1. **Holdout beku.** Frozen holdout (n=50) tidak dipakai untuk training,
   early stopping, pemilihan hyperparameter, atau pemilihan threshold.
   Runner trial HPO dipanggil dengan `evaluate_holdout=False`, dan
   `HPOEngine` menolak metrik yang berawalan `holdout_`/`test_`.
2. **Tuning hanya di train + validation.** Validation (n=50) dipakai untuk
   memilih konfigurasi. Holdout diprediksi sekali setelah konfigurasi dibekukan.
3. **Threshold tetap 0,5.** Tidak ada pencarian threshold di holdout.
4. **Semua trial dicatat.** Parameter, metrik, durasi, status, dan error
   tersimpan di `trials.jsonl` + `leaderboard.csv`.
5. **Uji berpasangan.** Perbandingan model memakai McNemar dan paired
   bootstrap ROC-AUC, bukan selisih point estimate.
6. **Beberapa seed.** Untuk paper, jalankan minimal 3 seed pada konfigurasi
   terbaik dan laporkan mean ± std.

---

## 2. Search Space BiLSTM

Preset `full` mencakup seluruh opsi berikut. Preset `standard` mempersempit
nilai agar waktu training wajar. Preset `smoke` hanya untuk uji pipeline.

### 2.1 Vektorisasi teks

| Parameter | Nilai / rentang | Arti |
|---|---|---|
| `vocab_size` | 10.000, 20.000, 30.000, 50.000 | ukuran kosakata tokenizer |
| `max_len` | 60–240 (step 20) | panjang maksimum token |
| `embed_dim` | 50, 100, 200, 300 | dimensi Word2Vec |
| `w2v_window` | 3–10 | jendela konteks Word2Vec |
| `w2v_min_count` | 1–5 | kata dengan frekuensi di bawah ini diabaikan |
| `w2v_epochs` | 10–50 | epoch training Word2Vec |
| `w2v_sg` | 0, 1 | 0 = CBOW, 1 = skip-gram |
| `embed_trainable` | True/False | embedding ikut dilatih atau dibekukan |
| `mask_zero` | True/False | padding id 0 diabaikan |
| `digit_fold` | True/False | angka dilipat menjadi `num` |

### 2.2 Arsitektur

| Parameter | Nilai / rentang | Arti |
|---|---|---|
| `rnn_type` | lstm, gru | jenis sel rekuren |
| `use_bidirectional` | True/False | membaca konteks dua arah |
| `num_rnn_layers` | 1, 2 | jumlah lapisan rekuren |
| `lstm_units_1` | 32, 64, 128, 256 | unit lapisan pertama |
| `lstm_units_2` | 16, 32, 64, 128 | unit lapisan kedua |
| `dropout_1` | 0,0–0,6 (step 0,1) | dropout setelah lapisan pertama |
| `dropout_2` | 0,0–0,6 (step 0,1) | dropout setelah lapisan kedua |
| `recurrent_dropout` | 0,0–0,4 (step 0,1) | dropout koneksi rekuren |
| `dense_units` | 32, 64, 128, 256 | unit Dense sebelum output |
| `dropout_dense` | 0,0–0,6 (step 0,1) | dropout setelah Dense |
| `use_maxnorm` | True/False | batasi bobot Dense dengan MaxNorm |

### 2.3 Optimasi dan callback

| Parameter | Nilai / rentang | Arti |
|---|---|---|
| `optimizer` | adam, adamw, rmsprop, sgd | algoritma optimasi |
| `learning_rate` | 1e-5–1e-2 (log) | laju belajar |
| `weight_decay` | 0, 1e-5, 1e-4, 1e-3, 1e-2 | regularisasi L2 |
| `batch_size` | 8, 16, 32, 64 | ukuran batch |
| `epochs` | 20, 30, 50, 80 | epoch maksimum (early stopping aktif) |
| `gradient_clip_norm` | None, 0,5, 1,0, 2,0, 5,0 | clipping gradien |
| `early_stopping_patience` | 2, 3, 4, 5, 8, 10 | patience early stopping |
| `lr_reduce_patience` | 2, 3, 4, 5 | patience ReduceLROnPlateau |
| `lr_reduce_factor` | 0,1, 0,3, 0,5, 0,7 | faktor penurunan LR |
| `use_class_weight` | True/False | bobot kelas seimbang |
| `label_smoothing` | 0,0, 0,05, 0,1 | smoothing label BCE |
| `shuffle` | True/False | acak urutan training |

### 2.4 Pemilihan model

Default: `val_f1` (maximize). Pilihan lain: `val_loss` (minimize),
`val_roc_auc` (maximize), `val_average_precision` (maximize).

---

## 3. Search Space IndoBERT

Preset `full` mencakup seluruh opsi berikut.

### 3.1 Model dan tokenisasi

| Parameter | Nilai | Arti |
|---|---|---|
| `model_name` | indobert-base-p1, base-p2, large-p2 | checkpoint pra-latih |
| `max_len` | 128, 192, 256, 320, 384, 512 | panjang token maksimum |

### 3.2 Optimasi

| Parameter | Nilai / rentang | Arti |
|---|---|---|
| `learning_rate` | 5e-6–1e-4 (log) | laju belajar fine-tuning |
| `batch_size` | 8, 16, 32 | batch per device |
| `epochs` | 3, 4, 5, 6, 8, 10 | epoch maksimum |
| `weight_decay` | 0, 0,01, 0,05, 0,1 | regularisasi |
| `warmup_ratio` | 0, 0,06, 0,1, 0,2 | proporsi langkah warmup |
| `lr_scheduler_type` | linear, cosine, cosine_with_restarts, polynomial, constant_with_warmup | scheduler |
| `optimizer` | adamw_torch, adamw_torch_fused, adafactor | optimizer HF |
| `max_grad_norm` | 0,5, 1,0, 2,0, 5,0 | clipping gradien |
| `gradient_accumulation_steps` | 1, 2, 4 | akumulasi gradien |

### 3.3 Regularisasi dan arsitektur

| Parameter | Nilai | Arti |
|---|---|---|
| `early_stopping_patience` | 1, 2, 3, 4 | patience early stopping |
| `metric_for_best_model` | eval_loss, eval_f1, eval_roc_auc, eval_average_precision, eval_accuracy | metrik seleksi |
| `freeze_layers` | 0, -1, 2, 4, 6, 8 | 0 = semua dilatih; -1 = embeddings beku; N = N layer awal beku |
| `use_class_weight` | True/False | bobot kelas |
| `label_smoothing_factor` | 0,0, 0,05, 0,1 | smoothing label |
| `hidden_dropout_prob` | 0,05, 0,1, 0,2, 0,3 | dropout hidden |
| `attention_probs_dropout_prob` | 0,05, 0,1, 0,2, 0,3 | dropout attention |
| `classifier_dropout` | None, 0,1, 0,2, 0,3 | dropout kepala klasifikasi |
| `precision` | fp32, fp16, bf16 | presisi training (butuh GPU yang sesuai) |
| `group_by_length` | True/False | kelompokkan panjang serupa untuk efisiensi |

---

## 4. Strategi Pencarian

| Strategi | Kapan dipakai | Catatan |
|---|---|---|
| `grid` | search space kecil, ingin audit lengkap | jumlah kombinasi dibatasi `max_grid_combinations` |
| `random` | default, ruang besar | setiap trial independen, mudah di-resume |
| `bayesian` | ingin efisien di ruang besar | butuh `optuna` (`pip install -r requirements-hpo.txt`); TPE menyarankan trial berikutnya |

Rekomendasi:

- `smoke` (4 trial) — memastikan pipeline dan environment benar.
- `standard` (30 trial) — cukup untuk paper skripsi.
- `full` (100+ trial) — eksplorasi maksimal; jalankan di HPC.

---

## 5. Cara Menjalankan

### Notebook

```text
03_bilstm_tuning.ipynb
05_indobert_tuning.ipynb
```

Ubah `PRESET` dari `"smoke"` ke `"standard"` atau `"full"`.

### CLI (untuk run panjang)

```bash
# siapkan environment
export PYTHONHASHSEED=42

# BiLSTM HPO
python scripts/tune_bilstm.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --preset standard --trials 30 --strategy random \
  --output-dir artifacts/hpo/bilstm_standard

# IndoBERT HPO (GPU)
python scripts/tune_bert.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --preset standard --trials 30 --strategy random \
  --output-dir artifacts/hpo/bert_standard
```

Resume cukup dengan menjalankan perintah yang sama; trial sukses tidak diulang.

### Ekspor tabel search space untuk lampiran

```bash
python scripts/export_search_space.py --model bilstm --preset full \
    --out artifacts/hpo/search_space_bilstm.csv
python scripts/export_search_space.py --model bert --preset full \
    --out artifacts/hpo/search_space_bert.md
```

---

## 6. Metrik dan Uji Statistik

Metrik utama (threshold 0,5):

- Accuracy
- Precision (kelas unsafe)
- Recall (kelas unsafe)
- F1
- ROC-AUC
- Average Precision
- Brier score (kalibrasi)

Interval kepercayaan: **stratified bootstrap 2.000 resampling**.

Perbandingan model:

- **McNemar** untuk pola benar/salah.
- **Paired bootstrap ROC-AUC** untuk selisih AUC + CI + p-value.

Jangan membandingkan dua CI marginal seolah-olah itu uji signifikansi.

---

## 7. Ablasi yang Disarankan

| Ablasi | Cara | Pertanyaan yang dijawab |
|---|---|---|
| Tanpa synthetic | `DataConfig(synthetic_total=0)` | apakah synthetic membantu? |
| Real-only BERT | `train_real.csv` | apakah BERT hanya butuh data real? |
| Tanpa Word2Vec | embedding acak / `embed_trainable=True` | nilai tambah pre-training Word2Vec |
| Tanpa KB | hapus fitur/seed KB | seberapa besar model belajar dari KB? |
| Tanpa freeze | `freeze_layers=0` vs `-1` | kontribusi fine-tuning penuh |
| Tanpa class weight | `use_class_weight=False` | efek imbalance |

---

## 8. Contoh Paragraf Metodologi (siap edit)

> Pencarian hyperparameter dilakukan secara terpisah untuk BiLSTM dan
> IndoBERT pada himpunan latih dan validasi; frozen holdout (n=50) tidak
> digunakan selama pencarian. Ruang pencarian BiLSTM mencakup 33 parameter
> (Tabel X), meliputi konfigurasi Word2Vec, arsitektur LSTM/GRU, dropout,
> optimizer, learning rate, ukuran batch, gradient clipping, class weight,
> dan label smoothing. Pencarian dilakukan dengan random search sebanyak 30
> trial menggunakan seed 42; setiap trial dicatat dalam `trials.jsonl`.
> Konfigurasi terbaik dipilih berdasarkan F1 pada validation set, kemudian
> dilatih ulang pada final-train dan dievaluasi satu kali pada frozen holdout.
> Interval kepercayaan 95% dihitung dengan stratified bootstrap (2.000
> resampling). Perbedaan performa terhadap baseline diuji dengan uji McNemar
> dan paired bootstrap untuk selisih ROC-AUC. Untuk mengurangi varians,
> konfigurasi terbaik dijalankan ulang pada tiga seed dan dilaporkan sebagai
> mean ± standar deviasi.

Versi Inggris singkat:

> Hyperparameter search was performed on the training and validation sets
> only; the frozen holdout (n=50) was never used for tuning. The BiLSTM search
> space covered 33 hyperparameters, including Word2Vec settings, LSTM/GRU
> architecture, dropout, optimizer, learning rate, batch size, gradient
> clipping, class weighting, and label smoothing. We ran 30 random-search
> trials with seed 42 and selected the best configuration by validation F1.
> The selected model was retrained on the final training pool and evaluated
> once on the frozen holdout. Confidence intervals were obtained via
> stratified bootstrap (2,000 resamples). Baseline comparisons used McNemar's
> test and paired bootstrap for ROC-AUC differences. The best configuration
> was repeated across three seeds and reported as mean ± standard deviation.

---

## 9. Checklist Pelaporan HPO

- [ ] Search space (parameter + nilai/rentang) dicantumkan.
- [ ] Jumlah trial + strategi + seed disebutkan.
- [ ] Metrik seleksi dan arahnya disebutkan.
- [ ] Holdout tidak dipakai untuk tuning (dinyatakan eksplisit).
- [ ] Semua metrik menyebut n + 95% CI.
- [ ] Uji McNemar + paired bootstrap AUC dilaporkan.
- [ ] Varians antar seed dilaporkan.
- [ ] Ablasi synthetic vs real-only ada.
- [ ] Keterbatasan (n kecil, synthetic, OCR noise) ditulis jujur.
- [ ] Kode + trial log tersedia sebagai lampiran/repositori.

---

## 10. Catatan Reproducibility

- Jalankan `PYTHONHASHSEED=<seed>` **sebelum** Python mulai. Menyetel
  `os.environ` di dalam program tidak menonaktifkan hash randomization.
- `app/core/kb/rules.py` sudah diperbaiki agar urutan term deterministik.
  Data sintetik sekarang identik lintas proses untuk seed yang sama.
- Artefak lama di `artifacts/bert/input/` dibuat sebelum perbaikan
  determinisme. Regenerate sekali dengan `scripts/rebuild_split.py`, lalu
  latih ulang model BERT agar hash data dan hasil konsisten.
- Word2Vec memakai `workers=1` agar deterministik.
