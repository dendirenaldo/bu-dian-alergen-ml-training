# Bu Dian ML Training

Pipeline training model deteksi alergen makanan menggunakan Word2Vec + BiLSTM.

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- Gensim (Word2Vec)
- Scikit-learn
- OpenCV + Tesseract OCR
- Sastrawi (Indonesian NLP)
- Jupyter Notebook

## Pipeline

1. **Image Preprocessing** — Grayscale, bilateral filter, Otsu threshold
2. **OCR Extraction** — Tesseract with layout-aware composition parsing
3. **Bold Text Detection** — Adaptive thresholding for allergen highlighting
4. **Text Preprocessing** — Lowercasing, regex, Indonesian stopword removal
5. **Word2Vec Embedding** — Skip-gram, dim=100, vocab_size=20000
6. **BiLSTM Classification** — 2-layer Bidirectional LSTM
7. **Hyperparameter Tuning** — Random search (10 trials)
8. **Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter
jupyter notebook main.ipynb
```

## Dataset

- `dataset/data-mengandung.csv` — 886 Indonesian food products with ingredient text and safe/unsafe labels
- `dataset/*.jpg` — Product packaging images for OCR

## Model Output

Trained models are saved to `models/` directory:
- `bilstm_model.keras`
- `word2vec.model`
- `tokenizer.pkl`
- `label_encoder.pkl`

Copy these files to `bu-dian-ml-service/models/` for deployment.
