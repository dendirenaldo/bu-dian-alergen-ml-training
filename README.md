# Bu Dian ML Training — BiLSTM Leakage-Safe

Pipeline training tunggal: Word2Vec + BiLSTM dengan kontrak leakage-safe
(frozen holdout, synthetic hanya dari real-train, threshold fixed 0.5).
BERT dilatih di HPC (`artifacts/bert/` + `docs/bert-hpc.md`) memakai split
yang identik.

## Tech Stack (aktual)

- Python 3.12, TensorFlow 2.16 (CPU), Gensim 4.3, scikit-learn, Click, Pytest
- Reproducibility: `app/repro.py` (env lock sebelum import TF + re-seed per tahap)

## CLI Usage

```bash
# Training (baca .env bila ada)
python scripts/train.py train -c .env --csv-input ./data.csv

# Lihat konfigurasi
python scripts/train.py info

# Bootstrap CI tanpa retraining (butuh artefak run)
python scripts/bootstrap_final.py --csv-input ./data.csv \
    --frozen-holdout app/core/data/frozen_holdout.json

# Rebuild split identik untuk HPC
python scripts/rebuild_split.py --csv-input ./data.csv \
    --frozen-holdout app/core/data/frozen_holdout.json

# Export artefak ke format serving
python scripts/export_for_serving.py
```

| Opsi `train` | Deskripsi | Default |
|---|---|---|
| `--config-file` | Path file .env | - |
| `--csv-input` | CSV `produk;text;label` (delimiter `;`) | dari .env |
| `--output-dir` | Direktori output | `./artifacts/bilstm/output` |
| `--model-dir` | Direktori model | `./artifacts/bilstm/models` |
| `--seed` | Random seed | 42 |
| `--frozen-holdout` | JSON frozen holdout | kemasan (`app/core/data/frozen_holdout.json`) |

Penyetelan lanjutan via env `V5_*` (lihat `.env.example`):
`V5_LEARNING_RATE`, `V5_EPOCHS`, `V5_BATCH_SIZE`, `V5_MASK_ZERO`,
`V5_W2V_MIN_COUNT`, `V5_MAX_LEN`, `V5_DIGIT_FOLD`, `V5_SHUFFLE`,
`V5_GRADIENT_CLIP_NORM`, `V5_HOLDOUT_*`, `V5_VAL_*`, `V5_SYNTHETIC_TOTAL`.

## Pipeline

1. **Gold merge** — label CSV dikunci sebagai `gold_label` + `group_key`.
2. **Split leakage-safe** — union-find `group_key` + hash teks; holdout &
   val real-gold; `split_manifest.csv` + audit PASS.
3. **Synthetic** — 1000 baris, hanya dari real-train, re-check KB.
4. **NLP** — Tokenizer + Word2Vec fit hanya final-train (real+synthetic).
5. **Train** — BiLSTM fixed single-run, `shuffle=False`, threshold fixed 0.5.
6. **Evaluasi** — val vs frozen holdout (akurasi/presisi/recall/F1/ROC-AUC/AP,
   ROC+PR, confusion matrix, triase Gold/KB/BiLSTM) + bootstrap CI.

## Konfigurasi Final (default)

| Parameter | Default | Keterangan |
|---|---|---|
| Split | train 399 / val 50 (25/25) / holdout 50 (25/25) | deterministik, frozen |
| Synthetic | 1000 (500/500), `real_train_only` | - |
| `learning_rate` | `1e-4` | deviasi D2 dari notebook 1e-3 (collapse) |
| `epochs` / `batch` | 20 / 16 | early-stop patience 4 |
| `mask_zero` | True | deviasi D3 (padding 71-85%) |
| `gradient_clip_norm` | 1.0 | deviasi D1 |
| `fixed_threshold` | 0.5 | assert, never tuned |
| LSTM | 128/64, dropout 0.3/0.3, dense 64/0.2, embed 100 | - |

## Output Files (`artifacts/bilstm/`)

### models/
- `bilstm_word2vec_v5.keras`, `word2vec_v5.model`, `tokenizer_v5.pkl`,
  `tokenizer_bilstm_v5.json`, `thresholds.json` (`{"bilstm": thr, "fixed": true}`)
  (alias `bilstm_model.keras` hanya ditulis `export_for_serving.py`)

### output/
- `evaluation_table_v5.csv`, `bootstrap_ci_v5.csv`, `training_history_*.csv`,
  `training_summary_*.csv`, `split_manifest.csv`, `oov_audit_v5.csv`,
  `gold_kb_bilstm_triage.csv`, `frozen_holdout_false_{negatives,positives}.csv`,
  `experiment_manifest.json`, kurva ROC/PR/loss/akurasi (tanpa judul figure)

## Struktur Project

```
ml-training/
├── app/
│   ├── config.py              # Config infra + V5Config (satu-satunya model config)
│   ├── repro.py               # Reproducibility lock
│   ├── core/
│   │   ├── data/              # gold_merge, leakage_split, synthetic (+ frozen_holdout.json)
│   │   ├── kb/                # Knowledge Base 8 kategori (diagnostik)
│   │   ├── preprocessing/     # text (tanpa stopwords) + image
│   │   ├── ocr/               # engine, composition, bold
│   │   ├── embedding/         # word2vec
│   │   └── model/             # architecture, tokenizer, metrics, audit
│   └── training/
│       └── trainer_v5.py      # Orkestrator pipeline tunggal
├── scripts/
│   ├── train.py               # CLI training
│   ├── bootstrap_final.py     # CI tanpa retraining
│   ├── rebuild_split.py       # Split identik untuk HPC
│   └── export_for_serving.py  # Artefak -> format ml-service
├── artifacts/
│   ├── bilstm/{models,output}/# Artefak final (biner besar di-ignore git)
│   └── bert/{input,results}/  # Input HPC + hasil (bawa pulang dari HPC)
├── docs/
│   └── bert-hpc.md            # Kontrak + langkah training BERT di HPC
├── tests/
│   └── test_v5_parity.py      # Kontrak KB/split/synthetic/eval/repro
├── requirements.txt
└── .env.example
```

## Testing

```bash
pytest tests/ -v
```
