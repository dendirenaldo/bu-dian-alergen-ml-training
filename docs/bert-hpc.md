# Training BERT di HPC — Kontrak Perbandingan Apel-vs-apel

Model BiLSTM final sudah dikunci lokal (`models_final/`, metrik `output_final/`).
BERT dilatih di HPC dengan **data split yang byte-identik**, memakai skrip
siap-pakai `artifacts/bert/train_bert_hpc.py` (standalone: hanya butuh
`torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `accelerate`).

## 0. Ringkasan langkah di HPC (copy-paste)

```bash
# 1. Upload ke HPC: direktori `artifacts/bert/` (berisi `train_bert_hpc.py` + `input/`),
#    lalu masuk ke input:
cd artifacts/bert

# 2. Siapkan environment (GPU node, Python >=3.10)
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch transformers scikit-learn pandas numpy accelerate

# 3. Verifikasi identitas split (WAJIB, cocokkan dengan split_contract.json)
sha256sum input/train_combined.csv input/train_real.csv input/val.csv input/holdout.csv input/synthetic.csv

# 4. (Opsional, bila HF Hub diblokir) unduh model di login node dulu, lalu:
# export HF_HUB_OFFLINE=1 HF_HOME=/path/cache

# 5. Fine-tuning utama (IndoBERT, pool sama dengan BiLSTM).
#    Cukup 1x GPU 16GB (batch 16, max_len 256). Estimasi ±10-20 menit.
python train_bert_hpc.py --data-dir ./input --output-dir ./bert_hpc_results

# 6. Hasil VAL + HOLDOUT @threshold 0.5 tercetak otomatis di akhir.
#    Simpan/copy output terminal ini ke laporan.

# 7. Ablasi opsional (tanpa synthetic — menjawab "synthetic membantu?")
python train_bert_hpc.py --data-dir ./input --output-dir ./bert_hpc_results_realonly \
    --train-file train_real.csv

# 8. Bawa pulang SELURUH direktori bert_hpc_results*/
```

## 1. File yang dibawa ke HPC

Dari `bu-dian-alergen-ml-training/artifacts/bert/input/` (dibuat oleh `scripts/rebuild_split.py`):

| File | Isi | Peran |
|---|---|---|
| `train_combined.csv` | 1399 teks (399 real + 1000 synthetic), kolom `text,label` | **Training utama** (sama dengan pool BiLSTM) |
| `train_real.csv` | 399 teks real | Ablasi opsional (tanpa synthetic) |
| `val.csv` | 50 real gold (25/25) | Early-stop + evaluasi |
| `holdout.csv` | 50 real gold (25/25) | **Evaluasi final saja. DILARANG untuk tuning/early-stop/model-selection.** |
| `split_contract.json` | komposisi + sha256 tiap file | Verifikasi identitas split |
| `split_manifest.csv` |asal split per baris | Audit |

Verifikasi pertama di HPC: regenerate dengan skrip yang sama lalu
`diff split_contract.json` — sha256 harus sama persis.

## 2. Yang dilakukan skrip (cukup pakai argumen, tidak perlu edit kode)

- Checkpoint default `indobenchmark/indobert-base-p1` (ganti via `--model-name`;
  ID persis tercatat otomatis di `model_card.json`).
- Teks dipakai apa adanya; tokenisasi subword bawaan HF (`--max-len 256`).
- Default: `--epochs 4 --lr 2e-5 --batch-size 16`, `weight_decay=0.01`,
  `warmup_ratio=0.1`, seed 42.
- Early stopping (`--patience 2`, monitor `eval_loss`) + best-model-restore —
  semuanya dari **val saja**. Holdout hanya diprediksi sekali di akhir.
- Metrik utama threshold **fixed 0.5** (sama seperti BiLSTM).

Jika job antre/lama: turunkan `--batch-size 8` dan/atau `--max-len 128`,
lalu catat perubahan di laporan (hasil tidak lagi identik dengan default).

## 3. Artefak yang wajib dibawa pulang

```
bert_hpc_results/
  metrics.json        # akurasi/presisi/recall/f1/roc_auc/ap @0.5, untuk val DAN holdout
  threshold.json      # {"threshold": 0.5, "fixed": true}
  training_log.json   # riwayat log per-epoch (loss + eval_accuracy/precision/recall/f1/roc_auc) u/ kurva
  probs_val.csv       # kolom: text, label, prob_unsafe
  probs_holdout.csv   # kolom: text, label, prob_unsafe
  roc_data.json       # fpr/tpr val + holdout (untuk plot gabungan + CI)
  model_card.json     # checkpoint id, seed, epochs, batch, lr, pool yang dipakai
  model/              # direktori HF (config.json + bobot) untuk serving ensemble
```

Dengan `probs_*.csv`, CI bootstrap + kurva ROC/PR gabungan BiLSTM-vs-BERT
dihitung lokal tanpa akses HPC (skrip `scripts/bootstrap_final.py` sebagai pola.

> Hasil training diletakkan di `artifacts/bert/results/` (setara `artifacts/bilstm/`).).

## 4. Troubleshooting

- **Baris `MISSING: those params were newly initialized...` saat load model.**
  Ini NORMAL, bukan error: head klasifikasi (2 label) memang diinisialisasi
  acak karena checkpoint pretrain tidak punya head tersebut. Training
  (`trainer.train()`) yang akan melatihnya. Lanjutkan saja.
- **`TypeError: TrainingArguments.__init__() got an unexpected keyword...`**
  Versi `transformers` di mesin terlalu tua/berbeda nama argumen. Skrip sudah
  dibuat toleran (otomatis pakai `evaluation_strategy`/`warmup_steps` bila
  `eval_strategy`/`warmup_ratio` tak ada). Bila masih error, upgrade:
  `pip install -U "transformers>=4.40"`. Cek versi: `pip show transformers`.
- **Windows path berspasi** (mis. `D:\Dendi S3\...`): selalu jalankan dari
  dalam direktori `input` dan pakai path relatif (`.`, `./bert_hpc_results`)
  agar terhindar dari masalah quoting.

## 5. Aturan main (berlaku untuk BERT juga)

1. Holdout tidak pernah menyentuh training/early-stop/tuning.
2. Metrik utama = threshold 0.5, bukan threshold yang di-tuning di holdout.
3. Setiap angka di laporan wajib menyebut `n` dan interval kepercayaan.
4. Klaim "lebih baik" harus menyebut uji di split yang sama + selisih CI
   (bukan selisih point-estimate).
