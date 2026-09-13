# Training BERT di HPC — Kontrak Perbandingan Apel-vs-apel

Model BiLSTM final sudah dikunci lokal (`models_final/`, metrik `output_final/`).
BERT dilatih di HPC dengan **data split yang byte-identik**.

## 1. File yang dibawa ke HPC

Dari `bu-dian-alergen-ml-training/bert_hpc/` (dibuat oleh `scripts/rebuild_split.py`):

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

## 2. Rekomendasi training (IndoBERT)

- Checkpoint: `indobenchmark/indobert-base-p1` (pluggable, catat ID persis).
- Input: kolom `text` apa adanya (jangan lower-case agresif / buang stopwords;
  biarkan subword tokenizer bawaan).
- Pool utama: `train_combined.csv` (rekomendasi; banding adil vs BiLSTM).
  Opsional: ulangi dengan `train_real.csv` sebagai ablasi synthetic.
- Validasi: `val.csv` untuk early-stop. Seed 42.
- Threshold: **fixed 0.5** untuk tabel utama (sama seperti BiLSTM).
  Threshold hasil tuning-val boleh dilaporkan sebagai kolom sekunder, tidak
  dipakai untuk klaim utama.

## 3. Artefak yang wajib dibawa pulang

```
bert_hpc_results/
  metrics.json        # akurasi/presisi/recall/f1/roc_auc/ap @0.5, untuk val DAN holdout
  threshold.json      # {"threshold": 0.5, "fixed": true}
  probs_val.csv       # kolom: text, label, prob_unsafe
  probs_holdout.csv   # kolom: text, label, prob_unsafe
  roc_data.json       # fpr/tpr val + holdout (untuk plot gabungan + CI)
  model_card.json     # checkpoint id, seed, epochs, batch, lr, pool yang dipakai
  model/              # direktori HF (config.json + bobot) untuk serving ensemble
```

Dengan `probs_*.csv`, CI bootstrap + kurva ROC/PR gabungan BiLSTM-vs-BERT
dihitung lokal tanpa akses HPC (skrip `scripts/bootstrap_final.py` sebagai pola).

## 4. Aturan main (berlaku untuk BERT juga)

1. Holdout tidak pernah menyentuh training/early-stop/tuning.
2. Metrik utama = threshold 0.5, bukan threshold yang di-tuning di holdout.
3. Setiap angka di laporan wajib menyebut `n` dan interval kepercayaan.
4. Klaim "lebih baik" harus menyebut uji di split yang sama + selisih CI
   (bukan selisih point-estimate).
