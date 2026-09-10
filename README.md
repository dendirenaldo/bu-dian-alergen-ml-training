# Bu Dian Allergen Detection - ML Training

Standalone ML training pipeline for the Bu Dian Allergen Detection system. This project handles model training, hyperparameter tuning, and evaluation using Word2Vec + BiLSTM/LSTM.

## Project Structure

```
ml-training/
├── app/
│   ├── __init__.py
│   ├── config.py                    # Central configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── image.py             # Image preprocessing for OCR
│   │   │   └── text.py              # Text preprocessing & Sastrawi stopwords
│   │   ├── ocr/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py            # Tesseract OCR engine
│   │   │   ├── composition.py       # Composition text extraction
│   │   │   └── bold.py              # Bold text detection for allergens
│   │   ├── embedding/
│   │   │   ├── __init__.py
│   │   │   └── word2vec.py          # Word2Vec training & embedding matrix
│   │   └── model/
│   │       ├── __init__.py
│   │       ├── architecture.py      # LSTM/BiLSTM model builder
│   │       ├── tokenizer.py         # Keras tokenizer wrapper
│   │       ├── tuner.py             # Hyperparameter tuning
│   │       └── evaluator.py         # Evaluation metrics & plots
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py               # Main training orchestrator
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py             # Model evaluation & export
├── scripts/
│   ├── __init__.py
│   ├── train.py                     # CLI training script
│   └── evaluate.py                  # CLI evaluation script
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Install Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Training

```bash
# Train with default configuration
python scripts/train.py train

# Train with custom options
python scripts/train.py train --data-source CSV --csv-input ./data/composition.csv --epochs 100

# Show current configuration
python scripts/train.py info
```

### Evaluation

```bash
# Compare BiLSTM and LSTM models
python scripts/evaluate.py compare \
    --model-bilstm ./models/bilstm_word2vec.keras \
    --model-lstm ./models/lstm_word2vec.keras \
    --csv-input ./ocr_output/data-mengandung.csv
```

### As Python Module

```python
from app.config import Config
from app.training.trainer import run_training

config = Config()
config.epochs = 100
config.batch_size = 32

results = run_training(config)
print(results["comparison"])
```

## Configuration

All settings are in `app/config.py` (extracted from the notebook's PANEL KONFIGURASI UTAMA):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_source_mode` | `CSV` | Data source: `CSV` or `OCR` |
| `test_size` | `0.2` | Train/test split ratio |
| `max_len` | `120` | Max sequence length |
| `vocab_size` | `20000` | Maximum vocabulary size |
| `embed_dim` | `100` | Word2Vec embedding dimension |
| `min_word_count` | `3` | Min word count for Word2Vec |
| `w2v_epochs` | `30` | Word2Vec training epochs |
| `lstm_units_1` | `128` | First LSTM layer units |
| `lstm_units_2` | `64` | Second LSTM layer units |
| `dropout_rate_1` | `0.3` | First dropout rate |
| `dropout_rate_2` | `0.3` | Second dropout rate |
| `learning_rate` | `1e-4` | Adam optimizer learning rate |
| `batch_size` | `64` | Training batch size |
| `epochs` | `150` | Maximum training epochs |
| `num_trials` | `10` | Hyperparameter tuning trials |

## Output Files

After training, the following files are generated:

- `output/training_history_bilstm_word2vec.csv` - BiLSTM training history
- `output/training_history_lstm_word2vec.csv` - LSTM training history
- `output/evaluation_table_bilstm_word2vec.csv` - BiLSTM evaluation metrics
- `output/evaluation_table_lstm_word2vec.csv` - LSTM evaluation metrics
- `output/classification_report_bilstm_word2vec.csv` - BiLSTM classification report
- `output/classification_report_lstm_word2vec.csv` - LSTM classification report
- `output/predictions_test_bilstm_word2vec.csv` - BiLSTM predictions
- `output/predictions_test_lstm_word2vec.csv` - LSTM predictions
- `output/comparison_bilstm_vs_lstm.csv` - Model comparison
- `models/bilstm_word2vec.keras` - Trained BiLSTM model
- `models/lstm_word2vec.keras` - Trained LSTM model

## Pipeline Overview

1. **Data Loading**: Load from CSV or run OCR pipeline
2. **Text Preprocessing**: Cleanse text, remove stopwords (Sastrawi)
3. **Tokenization**: Keras Tokenizer with padding
4. **Word2Vec Training**: Train embeddings on corpus
5. **Hyperparameter Tuning**: Random search over parameter space
6. **Model Training**: Train BiLSTM and LSTM with best params
7. **Evaluation**: Generate metrics, confusion matrix, ROC curves
8. **Export**: Save models, metrics, and comparison tables

## Development

```bash
# Run with verbose logging
python scripts/train.py -v train

# Quick test with fewer epochs
python scripts/train.py train --epochs 10 --num-trials 2
```
