# Bu Dian ML Training

Pipeline training model untuk deteksi alergen makanan. Dual-model: Word2Vec + BiLSTM/LSTM dan BERT (HuggingFace PyTorch, pluggable checkpoint) dengan hyperparameter tuning dan evaluasi komprehensif.

## Tech Stack

- Python 3.11
- TensorFlow 2.17 (BiLSTM/LSTM)
- PyTorch + Transformers + Datasets + Accelerate (BERT)
- Gensim 4.3 (Word2Vec embeddings)
- Sastrawi 1.0 (Indonesian NLP - stopword removal, hanya BiLSTM)
- OpenCV 4.11 (image preprocessing)
- Tesseract OCR via Pytesseract
- Scikit-learn (metrics, preprocessing)
- Click (CLI framework)
- Pytest (testing)

## Setup

### 1. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# atau
venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer dari https://github.com/UB-Mannheim/tesseract/wiki
```

### 4. Konfigurasi environment

```bash
cp .env.example .env
```

```env
DATASET_DIR=./dataset
OUTPUT_DIR=./output
MODEL_DIR=./models
CSV_INPUT=./ocr_output/data-mengandung.csv
SEED=42
```

## CLI Usage

### Training

```bash
# Training dengan konfigurasi default
python scripts/train.py train

# Training dengan opsi custom
python scripts/train.py train --data-source CSV --csv-input ./data/composition.csv --epochs 100

# Training dengan config file
python scripts/train.py train -c .env

# Lihat konfigurasi saat ini
python scripts/train.py info

# Jalankan dengan verbose logging
python scripts/train.py -v train
```

### Opsi Training

| Opsi | Deskripsi | Default |
|------|-----------|---------|
| `--config-file` | Path file .env config | - |
| `--data-source` | Sumber data: `CSV` atau `OCR` | CSV |
| `--csv-input` | Path file CSV input | dari .env |
| `--output-dir` | Direktori output | `./output` |
| `--model-dir` | Direktori model | `./models` |
| `--epochs` | Jumlah epochs | 150 |
| `--batch-size` | Batch size | 64 |
| `--seed` | Random seed | 42 |
| `--model` | Model: `bilstm`, `lstm`, `bert`, `all` | `all` |
| `--bert-model` | Checkpoint HF (pluggable, mis. `indobenchmark/indobert-base-p1`) | dari .env |
| `--bert-epochs` | Epoch training BERT | 4 |
| `--bert-lr` | Learning rate BERT | 2e-5 |

### Training BERT

```bash
# BERT saja (default IndoBERT, bisa diganti varian lain tanpa ubah kode)
python scripts/train.py train --model bert

# Coba varian BERT lain
python scripts/train.py train --model bert --bert-model bert-base-multilingual-cased
python scripts/train.py train --model bert --bert-model xlm-roberta-base --bert-epochs 3

# Semua model (BiLSTM + LSTM + BERT) di split test yang sama
python scripts/train.py train --model all
```

> Catatan: BERT memakai teks mentah + normalisasi ringan (tanpa stopword
> removal) karena WordPiece butuh konteks. BiLSTM tetap memakai
> `cleanse → tokenize → filter`.

### Evaluasi

```bash
# Bandingkan model BiLSTM dan LSTM (pakai artefak training, anti-leakage)
python scripts/evaluate.py compare \
    --model-bilstm ./models/bilstm_word2vec.keras \
    --model-lstm ./models/lstm_word2vec.keras \
    --csv-input ./ocr_output/data-mengandung.csv \
    --model-dir ./models \
    --output-dir ./output

# Sertakan BERT dalam perbandingan
python scripts/evaluate.py compare \
    --model-bilstm ./models/bilstm_word2vec.keras \
    --model-lstm ./models/lstm_word2vec.keras \
    --model-bert ./models/bert \
    --csv-input ./ocr_output/data-mengandung.csv \
    --model-dir ./models \
    --output-dir ./output
```

> `--model-dir` wajib agar evaluasi me-load `tokenizer.pkl`,
> `label_encoder.pkl`, `splits.json`, dan `thresholds.json` dari training
> (bukan fit ulang yang menyebabkan leakage + mismatch vocab).

### Testing

```bash
pytest tests/ -v
```

### Sebagai Python Module

```python
from app.config import Config
from app.training.trainer import run_training

config = Config()
config.epochs = 100
config.batch_size = 32

results = run_training(config)
print(results["comparison"])
```

## Konfigurasi

Semua pengaturan ada di `app/config.py`:

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `data_source_mode` | `CSV` | Sumber data: `CSV` atau `OCR` |
| `test_size` | `0.2` | Rasio train/test split |
| `max_len` | `120` | Panjang sekuens maksimal |
| `vocab_size` | `20000` | Ukuran vocabulary maksimal |
| `embed_dim` | `100` | Dimensi embedding Word2Vec |
| `min_word_count` | `3` | Jumlah minimum kata untuk Word2Vec |
| `w2v_epochs` | `30` | Epochs training Word2Vec |
| `lstm_units_1` | `128` | Unit LSTM layer pertama |
| `lstm_units_2` | `64` | Unit LSTM layer kedua |
| `dropout_rate_1` | `0.3` | Dropout rate pertama |
| `dropout_rate_2` | `0.3` | Dropout rate kedua |
| `learning_rate` | `1e-4` | Learning rate Adam optimizer |
| `batch_size` | `64` | Batch size training |
| `epochs` | `150` | Maximum training epochs |
| `num_trials` | `10` | Jumlah trial hyperparameter tuning |

## Output Files

Setelah training, file berikut dihasilkan:

### Model

- `models/bilstm_word2vec.keras` — Model BiLSTM
- `models/lstm_word2vec.keras` — Model LSTM
- `models/bilstm_model.keras` — Alias BiLSTM untuk ML service
- `models/bert/` — Model + tokenizer BERT (format HuggingFace)
- `models/tokenizer.pkl` + `models/tokenizer_bilstm.json` — Tokenizer
- `models/label_encoder.pkl` + `models/label_map.json` — Label map
- `models/thresholds.json` — Threshold per model (hasil tuning-val)
- `models/metadata.json` — Identitas artefak (git sha, metrik, label map)

### Metrik & Laporan

- `output/splits.json` — Indeks train/test (dipakai ulang semua model)
- `output/thresholds.json` — Threshold per model
- `output/bert_metrics.json` — Metrik BERT
- `output/comparison_models.csv` — Perbandingan semua model (nama lama `comparison_bilstm_vs_lstm.csv` tetap ditulis)

### Metrik & Laporan

- `output/training_history_bilstm_word2vec.csv` — History training BiLSTM
- `output/training_history_lstm_word2vec.csv` — History training LSTM
- `output/evaluation_table_bilstm_word2vec.csv` — Tabel evaluasi BiLSTM
- `output/evaluation_table_lstm_word2vec.csv` — Tabel evaluasi LSTM
- `output/classification_report_bilstm_word2vec.csv` — Classification report BiLSTM
- `output/classification_report_lstm_word2vec.csv` — Classification report LSTM
- `output/predictions_test_bilstm_word2vec.csv` — Prediksi BiLSTM
- `output/predictions_test_lstm_word2vec.csv` — Prediksi LSTM
- `output/comparison_bilstm_vs_lstm.csv` — Perbandingan kedua model

## Pipeline Training

1. **Data Loading** — Load dari CSV atau jalankan pipeline OCR
2. **Text Preprocessing** — Cleansing teks, stopword removal (Sastrawi)
3. **Tokenization** — Keras Tokenizer dengan padding
4. **Word2Vec Training** — Training embedding pada korpus
5. **Hyperparameter Tuning** — Random search atas ruang parameter
6. **Model Training** — Training BiLSTM dan LSTM dengan parameter terbaik
7. **Evaluasi** — Generate metrik, confusion matrix, ROC curves
8. **Export** — Simpan model, metrik, dan tabel perbandingan

## Struktur Project

```
ml-training/
├── app/
│   ├── config.py                    # Konfigurasi sentral (+ BertConfig, model_type)
│   ├── core/
│   │   ├── preprocessing/           # Preprocessing gambar & teks
│   │   │   ├── image.py
│   │   │   └── text.py
│   │   ├── ocr/                     # Tesseract OCR engine
│   │   │   ├── engine.py
│   │   │   ├── composition.py
│   │   │   └── bold.py
│   │   ├── embedding/
│   │   │   └── word2vec.py          # Word2Vec training & embedding
│   │   └── model/
│   │       ├── architecture.py      # LSTM/BiLSTM model builder
│   │       ├── tokenizer.py         # Keras tokenizer wrapper (+ JSON)
│   │       ├── tuner.py             # Hyperparameter tuning (stratified)
│   │       ├── evaluator.py         # Metrik & plot evaluasi (+ threshold)
│   │       ├── protocols.py         # Abstraksi Tokenizer/ClassifierBase
│   │       ├── model_factory.py     # Factory BiLSTM/LSTM/BERT
│   │       └── bert/                # Modul BERT (pluggable checkpoint)
│   │           ├── dataset.py       # Normalisasi ringan (tanpa stopword)
│   │           ├── trainer_bert.py  # HF Trainer + threshold-tuning
│   │           └── evaluate_bert.py # Evaluasi direktori model BERT
│   └── training/
│       └── trainer.py               # Training orchestrator (dual-model)
├── scripts/
│   ├── train.py                     # CLI training (--model, --bert-model)
│   └── evaluate.py                  # CLI evaluasi (--model-dir, --model-bert)
├── tests/
│   └── test_dual_model.py           # Regression tests
├── requirements.txt
└── .env.example
```
