# Optimasi GPU — RTX 3080 Ti & Multi-GPU

Dokumen ini merangkum status optimasi GPU setelah audit, termasuk perilaku
multi-GPU untuk BiLSTM dan IndoBERT.

Target utama: **NVIDIA RTX 3080 Ti** (Ampere, compute capability 8.6,
12 GB VRAM, Tensor Core FP16/TF32). Untuk 2 kartu, komunikasi antar-GPU
memakai PCIe (tanpa NVLink), jadi scaling tidak linear.

---

## 1. Ringkasan Status

| Komponen | Single GPU | Multi-GPU | Optimasi utama |
|---|---|---|---|
| BiLSTM + Word2Vec | Ya | Ya, `MirroredStrategy` otomatis | FP16, `tf.data`, prefetch, batch scaling, `steps_per_execution` |
| IndoBERT final | Ya | Ya, lewat `accelerate launch` (DDP) | FP16, TF32, fused AdamW, gradient checkpointing, auto batch size, pin memory |
| HPO BiLSTM | Ya | Ya, satu trial memakai semua GPU | MirroredStrategy di dalam setiap trial |
| HPO IndoBERT | Ya | Paralel per-GPU (bukan DDP) | Beberapa proses `tune_bert.py`, masing-masing satu GPU |

---

## 2. BiLSTM — Optimasi yang Sudah Aktif

Default `BiLSTMConfig`:

| Field | Default | Efek |
|---|---|---|
| `mixed_precision` | `mixed_float16` | Tensor Core FP16 di 3080 Ti; fallback ke `float32` bila tidak ada GPU |
| `multi_gpu` | `True` | `MirroredStrategy` dipakai bila terlihat >1 GPU |
| `auto_scale_batch_size` | `True` | `batch_size` dianggap **per GPU**; batch global = batch × jumlah GPU |
| `use_tf_data` | `True` | Dataset `tf.data` + `prefetch(AUTOTUNE)` |
| `prefetch` | `True` | Overlap CPU/GPU |
| `memory_growth` | `True` | TF tidak mengambil seluruh 12 GB VRAM di awal |
| `steps_per_execution` | `1` | Opsional; set `10` jika versi Keras mendukung argumen ini di `model.fit()` |
| `deterministic` | `True` | Deterministic ops; jika driver menolak, set `False` |

Catatan:
- Output layer tetap `float32` agar BCE stabil.
- Metric training memakai implementasi custom (`reduce_sum`) agar tidak
  memicu error `UnsortedSegmentSum` di GPU.
- ROC-AUC/AP dihitung setelah training dengan scikit-learn.
- FP16/TF32 dapat mengubah angka metrik sedikit dibanding FP32. Untuk paper,
  tetapkan satu presisi dan laporkan; jangan mencampur hasil FP32 dan FP16
  dalam satu tabel perbandingan.

### Single GPU (RTX 3080 Ti)

```bash
python scripts/train_bilstm.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --mixed-precision mixed_float16 \
  --batch-size 32 --epochs 30
```

### Multi-GPU (2× RTX 3080 Ti)

Cukup satu proses; `MirroredStrategy` memakai semua GPU yang terlihat:

```bash
# Linux/macOS
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_bilstm.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --batch-size 32 --mixed-precision mixed_float16
```

```powershell
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES="0,1"
python scripts/train_bilstm.py --csv-input "..\old\ocr_output\data-mengandung.csv" `
  --batch-size 32 --mixed-precision mixed_float16
```

`--batch-size 32` berarti 32 per GPU → batch global 64.

### HPO BiLSTM

Setiap trial memakai semua GPU yang terlihat. HPO tetap sequential:

```bash
python scripts/tune_bilstm.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --preset standard --trials 30 --output-dir artifacts/hpo/bilstm_2gpu
```

---

## 3. IndoBERT — Optimasi yang Sudah Aktif

Default `BertConfig` yang relevan:

| Field | Default | Efek |
|---|---|---|
| `precision` | `fp32` | Set `fp16` untuk 3080 Ti |
| `tf32` | `True` | Matmul FP32 memakai TF32 di Ampere |
| `optimizer` | `adamw_torch` | Pilih `adamw_torch_fused` untuk CUDA |
| `dataloader_pin_memory` | `True` | Transfer CPU→GPU lebih cepat |
| `auto_find_batch_size` | `False` | Set `True` agar Trainer menurunkan batch saat OOM |
| `gradient_checkpointing` | `False` | Set `True` untuk menghemat VRAM (lebih lambat) |
| `ddp_find_unused_parameters` | `False` | Lebih cepat untuk BERT |
| `torch_compile` | `False` | Opsional di Linux; sering belum stabil di Windows |
| `group_by_length` | `False` | Set `True` untuk mengurangi padding |

### Single GPU

```bash
python scripts/train_bert.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --precision fp16 --batch-size 16 --max-len 256
```

### Multi-GPU (2× RTX 3080 Ti)

Gunakan `accelerate`:

```bash
accelerate launch --multi_gpu --num_processes 2 \
  scripts/train_bert.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --precision fp16 --batch-size 8 --max-len 256
```

- `batch-size` adalah **per GPU**. Dengan 2 GPU dan batch 8, batch global 16.
- Di Windows, DDP biasanya memakai backend `gloo`. Jika `accelerate` memilih
  NCCL dan gagal, set `BertConfig.ddp_backend="gloo"` atau konfigurasi
  accelerate ke `gloo`.
- Hanya rank 0 yang menulis artefak (`metrics.json`, `probs_*.csv`, model).
- Jangan menjalankan `tune_bert.py` lewat `accelerate launch`; HPO BERT
  sequential.

### HPO BERT paralel per GPU

Jalankan dua proses terpisah, satu GPU masing-masing:

```bash
# terminal 1
CUDA_VISIBLE_DEVICES=0 python scripts/tune_bert.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --preset standard --trials 15 --output-dir artifacts/hpo/bert_gpu0

# terminal 2
CUDA_VISIBLE_DEVICES=1 python scripts/tune_bert.py \
  --csv-input ../old/ocr_output/data-mengandung.csv \
  --preset standard --trials 15 --output-dir artifacts/hpo/bert_gpu1
```

Dengan cara ini dua trial berjalan paralel di dua GPU tanpa saling berebut
VRAM.

---

## 4. Panduan VRAM 12 GB (RTX 3080 Ti)

| Model | Konfigurasi aman | Catatan |
|---|---|---|
| BiLSTM | batch 32–64 per GPU, `max_len` 120–160 | Model kecil; VRAM bukan bottleneck utama |
| IndoBERT base | `max_len` 256, batch 16 per GPU, FP16 | Baseline yang disarankan |
| IndoBERT base | `max_len` 128, batch 32 per GPU, FP16 | Jika OOM pada 256 |
| IndoBERT base | `max_len` 384–512, batch 8, gradient checkpointing | Untuk teks sangat panjang |
| IndoBERT large | `max_len` 128–256, batch 4–8, gradient checkpointing | Mungkin tetap OOM di 12 GB |

Jika OOM:
1. Turunkan `batch_size` / `per_device_train_batch_size`.
2. Turunkan `max_len`.
3. Aktifkan `gradient_checkpointing`.
4. Aktifkan `auto_find_batch_size`.
5. Naikkan `gradient_accumulation_steps` untuk mempertahankan batch efektif.

---

## 5. Multi-GPU: Apa yang Realistis?

- **BiLSTM**: MirroredStrategy membagi batch global; scaling bagus karena
  model kecil, tetapi komunikasi PCIe bisa menjadi bottleneck jika batch
  per GPU terlalu kecil. Naikkan batch per GPU saat memakai 2 GPU.
- **IndoBERT**: DDP via `accelerate`; scaling biasanya 1.6–1.9× untuk 2 GPU
  pada 3080 Ti, tergantung batch size dan panjang token.
- **HPO**: jangan memakai DDP untuk HPO. Untuk BiLSTM, satu trial memakai
  semua GPU. Untuk BERT, jalankan beberapa proses HPO paralel per GPU.

---

## 6. Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `Deterministic GPU implementation ... not available` | Op/metric non-deterministik | Versi ini sudah memakai metric custom; jika tetap muncul set `BiLSTMConfig(deterministic=False)` |
| `CUDA out of memory` | Batch/seq terlalu besar | Turunkan batch/`max_len`, aktifkan gradient checkpointing/auto batch |
| Multi-GPU tidak lebih cepat | Batch per GPU terlalu kecil / PCIe | Naikkan batch per GPU, kurangi overhead data loader |
| DDP gagal di Windows | NCCL tidak tersedia | Gunakan backend `gloo` |
| GPU hanya terpakai 1 | `CUDA_VISIBLE_DEVICES` salah | Cek dengan `nvidia-smi` dan `torch.cuda.device_count()` |
| Memory growth error | GPU sudah diinisialisasi | Restart kernel/proses, panggil `configure_gpu` paling awal |
| Error pada `steps_per_execution` | Versi Keras tidak punya argumen itu | Biarkan `1`; pipeline otomatis mengabaikan jika tidak didukung |

---

## 7. Verifikasi Cepat

Di Python:

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("Jumlah GPU:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

Untuk TensorFlow:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
print(tf.keras.mixed_precision.global_policy())
```

Output yang diharapkan pada 3080 Ti: satu atau dua device GPU, policy
`mixed_float16` setelah training dimulai.
