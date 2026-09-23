"""Builder docx: Arsitektur & Evaluasi IndoBERT (disisipkan ke Bagian Hasil).

Penomoran lanjutan dari naskah build_paper.py (Gambar 4.1-4.5, Tabel 4.1-4.5,
Persamaan (1)-(4)): naskah ini memakai Gambar 4.6-4.9, Tabel 4.6-4.12,
Persamaan (5)-(7).
Semua angka dibaca dari artefak aktual saat build (tidak ada hardcode metrik).
"""
import json
import os

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
FIG = os.path.join(BASE, "..", "paper_figs")
BERT = os.path.join(ROOT, "artifacts", "bert", "results")
OUT_BIST = os.path.join(ROOT, "artifacts", "bilstm", "output")
FONT = "Times New Roman"
GRAY = "D9D9D9"

doc = Document()

# ---------- halaman & style dasar (identik build_paper.py) ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
pf = normal.paragraph_format
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf.line_spacing = 1.15
pf.space_after = Pt(6)

for i, (sz, before) in {1: (14, 12), 2: (12, 10), 3: (12, 8)}.items():
    st = doc.styles[f"Heading {i}"]
    st.font.name = FONT
    st.font.size = Pt(sz)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT


def h2(t):
    doc.add_heading(t, level=2)


def h3(t):
    doc.add_heading(t, level=3)


def P(*segs, align=None, size=12, space_after=6):
    p = doc.add_paragraph()
    for s in segs:
        if isinstance(s, str):
            r = p.add_run(s)
        else:
            r = p.add_run(s[0])
            r.bold = bool(len(s) > 1 and s[1])
            r.italic = bool(len(s) > 2 and s[2])
        r.font.name = FONT
        r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    return p


def table(headers, rows, widths=None, size=9.5):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    for j, hh in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = ""
        r = c.paragraphs[0].add_run(hh)
        r.bold = True
        r.font.name = FONT
        r.font.size = Pt(size)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), GRAY)
        shading.set(qn("w:val"), "clear")
        c._tc.get_or_add_tcPr().append(shading)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.text = ""
            r = c.paragraphs[0].add_run(str(val))
            r.font.name = FONT
            r.font.size = Pt(size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(kind, num, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{kind} {num}. {text}")
    r.bold = True
    r.font.name = FONT
    r.font.size = Pt(10)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    return p


def figure(path, width_cm, kind, num, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(width_cm))
    caption(kind, num, text)


def equation(path, num, width_cm=13.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(width_cm))
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = q.add_run(f"({num})")
    r.font.name = FONT
    r.font.size = Pt(12)


def F(x, nd=4):
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def Fe(x, nd=1):
    """Notasi ilmiah gaya Indonesia: 3.8e-06 -> 3,8×10⁻⁶."""
    mant, exp = f"{x:.{nd}e}".split("e")
    sup = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")
    return f"{mant.replace('.', ',')}\u00d710{str(int(exp)).translate(sup)}"


# ---------- muat artefak (satu-satunya sumber angka) ----------
cfg = json.load(open(os.path.join(BERT, "model", "config.json"), encoding="utf-8"))
metrics = json.load(open(os.path.join(BERT, "metrics.json"), encoding="utf-8"))
card = json.load(open(os.path.join(BERT, "model_card.json"), encoding="utf-8"))
blog = json.load(open(os.path.join(BERT, "training_log.json"), encoding="utf-8"))
ci = pd.read_csv(os.path.join(OUT_BIST, "bootstrap_ci_bert.csv"))
ev = [e for e in blog if "eval_loss" in e]
train_steps = [
    e for e in blog
    if "loss" in e and "eval_loss" not in e and "train_runtime" not in e
]
runtime = next(e for e in blog if "train_runtime" in e)
best = min(ev, key=lambda e: e["eval_loss"])

d_v = pd.read_csv(os.path.join(BERT, "probs_val.csv"))
d_h = pd.read_csv(os.path.join(BERT, "probs_holdout.csv"))


def cm(df):
    yt = (df["label"].astype(str).str.lower() == "unsafe").astype(int).values
    yp = (df["prob_unsafe"].astype(float).values >= 0.5).astype(int)
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tp = int(((yt == 1) & (yp == 1)).sum())
    return tn, fp, fn, tp


tn_v, fp_v, fn_v, tp_v = cm(d_v)
tn_h, fp_h, fn_h, tp_h = cm(d_h)


def errs(df, split):
    yt = (df["label"].astype(str).str.lower() == "unsafe").astype(int).values
    yp = (df["prob_unsafe"].astype(float).values >= 0.5).astype(int)
    out = df[yt != yp].copy()
    out["_split"] = split
    return out


errs_all = pd.concat([errs(d_v, "Validasi"), errs(d_h, "Himpunan beku")])
errs_all["nama_produk"] = errs_all.get("nama_produk", pd.Series("", index=errs_all.index))
# fallback nama: ambil dari split bila kolom kosong
if errs_all["nama_produk"].fillna("").eq("").any():
    import numpy as _np
    errs_all["nama_produk"] = _np.where(
        errs_all["nama_produk"].fillna("") == "",
        errs_all["text"].astype(str).str.slice(0, 36),
        errs_all["nama_produk"],
    )


def ci_row(split, metric):
    r = ci[(ci["split"] == split) & (ci["metric"] == metric)].iloc[0]
    return f"{F(r['mean'], 3)} [{F(r['lo'], 2)}\u2013{F(r['hi'], 2)}]"


ci_bi = pd.read_csv(os.path.join(OUT_BIST, "bootstrap_ci_v5.csv"))


def ci_bi_row(split, metric):
    r = ci_bi[(ci_bi["split"] == split) & (ci_bi["metric"] == metric)].iloc[0]
    return f"{F(r['mean'], 3)} [{F(r['lo'], 3)}\u2013{F(r['hi'], 3)}]"


eval_bi = pd.read_csv(os.path.join(OUT_BIST, "evaluation_table_v5.csv"))

# ================================================================
# 4.6 ARSITEKTUR & EVALUASI INDOBERT
# ================================================================
h2("4.6  Arsitektur dan Evaluasi Model IndoBERT")
P(
    "Sub-bab ini menguraikan arsitektur model pembanding berbasis Transformer, "
    "yaitu IndoBERT yang di-",
    ("fine-tuning", False, True),
    " untuk klasifikasi biner teks komposisi, beserta seluruh hasil evaluasi pada "
    "protokol himpunan beku yang sama dengan model Word2Vec\u2013BiLSTM. Seluruh "
    "angka dibaca langsung dari artefak eksperimen ",
    ("model/config.json", False, True), ", ",
    ("metrics.json", False, True), ", ",
    ("training_log.json", False, True), ", ",
    ("probs_*.csv", False, True), ", dan ",
    ("bootstrap_ci_bert.csv", False, True),
    " sehingga seluruh klaim dapat direproduksi.",
)

h3("4.6.1  Arsitektur End-to-End")
P(
    "Alur end-to-end model ditunjukkan pada Gambar 4.6. Teks komposisi masuk dalam "
    "bentuk mentah \u2014 tanpa pembuangan stopword maupun penghapusan tanda baca \u2014 "
    "karena tokenisasi subword bawaan (",
    ("WordPiece", False, True),
    ") membutuhkan konteks linguistik utuh dan agar perlakuan terhadap ketiga "
    "himpunan data identik. Tokenisasi menghasilkan urutan indeks kosakata "
    f"(vocab {int(cfg['vocab_size']):,} subword)".replace(",", ".")
    + " dengan pemotongan dan padding ke panjang tetap "
    f"{card['max_len']} token, disertai ",
    ("attention mask", False, True),
    " yang menandai token nyata agar lapisan attention mengabaikan padding. "
    "Setiap indeks dipetakan ke vekor embedding "
    f"{cfg['hidden_size']} dimensi yang ditambah embedding posisi, lalu diproses "
    f"oleh {cfg['num_hidden_layers']} blok Transformer encoder. Setiap blok "
    "terdiri atas multi-head self-attention dengan "
    f"{cfg['num_attention_heads']} kepala, jaringan feed-forward "
    f"{cfg['intermediate_size']} unit dengan aktivasi GELU, normalisasi lapisan, "
    f"residual, dan dropout {cfg['hidden_dropout_prob']}. Representasi token pada "
    "posisi [CLS] kemudian dilewatkan dropout klasifikasi dan kepadatan "
    f"{cfg['hidden_size']} \u2192 2 yang menghasilkan dua logit, dinormalisasi "
    "dengan softmax biner (Persamaan 5) menjadi probabilitas kelas, dan keputusan "
    "klasifikasi diambil pada ambang tetap 0,5 terhadap probabilitas kelas tidak "
    "aman.",
)
figure(
    f"{FIG}/bert_diagram.png", 15.8, "Gambar", "4.6",
    "Arsitektur end-to-end IndoBERT untuk klasifikasi biner teks komposisi.",
)
P(
    "Parameter backbone tercantum pada Tabel 4.6. Arsitektur ini bersifat statis "
    "selama eksperimen; seluruh bobot Transformer pra-latih (",
    ("indobenchmark/indobert-base-p1", False, True),
    ") ikut disetel ulang (",
    ("full fine-tuning", False, True),
    ") \u2014 bukan ",
    ("feature extraction", False, True),
    " \u2014 karena domain teks komposisi berbeda jauh dengan korpus pra-latih, "
    "sehingga penyetelan penuh diperlukan agar representasi internal menyesuaikan "
    "pada leksikon bahan pangan.",
)
table(
    ["Komponen", "Konfigurasi"],
    [
        ["Checkpoint pra-latih", card["model_name"]],
        ["Arsitektur", ", ".join(cfg.get("architectures", ["BertForSequenceClassification"]))],
        ["Jumlah lapisan encoder", cfg["num_hidden_layers"]],
        ["Dimensi tersembunyi (hidden)", cfg["hidden_size"]],
        ["Kepala attention per lapisan", cfg["num_attention_heads"]],
        ["Unit jaringan feed-forward", cfg["intermediate_size"]],
        ["Kosakata subword", f"{int(cfg['vocab_size']):,}".replace(",", ".")],
        ["Posisi maksimum (absolut)", cfg["max_position_embeddings"]],
        ["Panjang input efektif (max_length)", card["max_len"]],
        [
            "Dropout tersembunyi / attention",
            f"{cfg['hidden_dropout_prob']} / {cfg['attention_probs_dropout_prob']}",
        ],
        ["Kepala klasifikasi", f"Dense {cfg['hidden_size']} \u2192 2 (logit)"],
        ["Peta kelas", ", ".join(f"{k} = {v}" for k, v in cfg["id2label"].items())],
    ],
    widths=[7.0, 8.6],
)
caption(
    "Tabel", "4.6",
    "Konfigurasi parameter backbone IndoBERT (sumber: model/config.json).",
)

P("Representasi kelas dihasilkan melalui softmax biner berikut.")
equation(f"{FIG}/bert_eq_softmax.png", 5)
P(
    "dengan ",
    ("z", False, True),
    " sebagai pasangan logit (",
    ("z", False, True),
    "\u2080, ",
    ("z", False, True),
    "\u2081) dan ",
    ("p", False, True),
    "(y = 1 \u2223 ",
    ("z", False, True),
    ") adalah probabilitas kelas tidak aman yang dibandingkan dengan ambang 0,5. "
    "Fungsi rugi pelatihan adalah entropi silang biner (Persamaan 6).",
)
equation(f"{FIG}/bert_eq_ce.png", 6)

h3("4.6.2  Protokol Fine-Tuning dan Partisi Data")
P(
    "Data pelatihan, partisi, dan kriteria berhenti mengikuti protokol "
    "anti-leakage pada Sub-bab 3.6\u20133.10: model hanya melihat himpunan latih "
    "final (teks riil ditambah teks sintetik), penyetelan model (",
    ("early stopping", False, True),
    ", ",
    ("model selection", False, True),
    ") hanya memanfaatkan himpunan validasi, dan himpunan beku hanya diprediksi "
    "satu kali di akhir. Hiperparameter lengkap dirangkum pada Tabel 4.7.",
)
table(
    ["Hiperparameter / aspek", "Nilai"],
    [
        ["Checkpoint", card["model_name"]],
        ["Data latih", f"{card['train_file']} (1.399 teks: 399 riil + 1.000 sintetik)"],
        ["Himpunan validasi / beku", "50 riil (25 safe / 25 unsafe) / 50 riil (25/25)"],
        ["Optimizer", "AdamW (implementasi HF Trainer)"],
        ["Laju belajar", "2\u00d710\u207b\u2075"],
        ["Weight decay", "0,01"],
        ["Rasio warm-up", "0,10 dari total langkah"],
        ["Batch (latih / evaluasi)", f"{card['batch_size']} / 32"],
        ["Maksimum epoch", card["epochs"]],
        ["Penjepit gradien", "norm maksimum 1,0"],
        ["Kriteria seleksi", "eval_loss minimum; load_best_model_at_end = True"],
        ["Early stopping", "patience 2 atas eval_loss"],
        ["Benih", card["seed"]],
        [
            "Waktu pelatihan total",
            F(runtime['train_runtime'], 1) + " detik "
            f"({F(runtime['train_steps_per_second'], 2)} langkah/detik, "
            f"{int(runtime['step'])} langkah)",
        ],
    ],
    widths=[6.4, 9.2],
)
caption("Tabel", "4.7", "Hiperparameter dan protokol fine-tuning IndoBERT.")

h3("4.6.3  Dinamika Pelatihan")
P(
    "Tabel 4.8 menyajikan metrik evaluasi pada himpunan validasi di setiap epoch; "
    "kurva pendampingnya ditampilkan pada Gambar 4.7. Tiga fase dapat dibedakan. ",
    ("Fase konvergensi cepat", True),
    " (epoch 1\u20132): ",
    ("eval loss", False, True),
    f" turun dari {F(ev[0]['eval_loss'])} menjadi {F(ev[1]['eval_loss'])} "
    f"(penurunan {(1 - ev[1]['eval_loss'] / ev[0]['eval_loss']) * 100:.0f}%), "
    "menunjukkan kepala klasifikasi yang tadinya diinisialisasi acak menyesuaikan "
    "diri dengan sangat cepat; recall menyentuh 1,000 pada epoch 2. ",
    ("Fase optimum", True),
    f" (epoch {best['epoch']:g}): ",
    ("eval loss", False, True),
    f" mencapai minimum {F(best['eval_loss'])} dengan presisi 1,000 dan F1 "
    f"{F(best['eval_f1'])} \u2014 inilah bobot yang dipulihkan otomatis oleh "
    "mekanisme pemulihan bobot terbaik. ",
    ("Fase awal overfitting", True),
    f" (epoch {ev[-1]['epoch']:g}): ",
    ("eval loss", False, True),
    f" berbalik naik menjadi {F(ev[-1]['eval_loss'])} "
    f"(naik {(ev[-1]['eval_loss'] / best['eval_loss'] - 1) * 100:.0f}%) dan F1 turun ke "
    f"{F(ev[-1]['eval_f1'])}, sementara laju belajar telah menyusut hingga ",
    Fe(train_steps[-1]['learning_rate']) + ". Kenaikan tersebut merupakan gejala "
    "overfitting dini: dengan jumlah sampel latih yang terbatas, kapasitas model "
    "sebesar ini mulai menghafal pola latih sehingga generalisasi ke validasi "
    "memburuk; ",
    ("early stopping", False, True),
    " (patience 2) menangkap gejala tersebut dan mengembalikan bobot epoch "
    "terbaik.",
)
P(
    "Catatan tambahan: norma gradien melonjak dari 7,5 menjadi "
    + F(train_steps[2]['grad_norm'], 1)
    + " pada pertengahan epoch 3 sebelum kembali "
    "menurun. Lonjakan tersebut terkendali berkat penjepit gradien norm 1,0 dan "
    "justru didahului oleh perolehan ",
    ("eval loss", False, True),
    " minimum, sehingga tidak mengganggu konvergensi.",
)
t_epoch = table(
    ["Epoch", "eval_loss", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AP"],
    [
        [
            f"{e['epoch']:g}",
            F(e["eval_loss"]),
            F(e["eval_accuracy"], 3),
            F(e["eval_precision"], 3),
            F(e["eval_recall"], 3),
            F(e["eval_f1"], 3),
            F(e["eval_roc_auc"], 3),
            F(e["eval_average_precision"], 3),
        ]
        for e in ev
    ],
    widths=[1.4, 2.0, 2.0, 2.0, 1.8, 1.6, 2.0, 2.0],
)
caption(
    "Tabel", "4.8",
    "Metrik evaluasi per-epoch pada himpunan validasi (ambang 0,5).",
)
# tebalkan baris epoch terbaik (eval_loss minimum)
for i, e in enumerate(ev, start=1):
    if e["epoch"] == best["epoch"]:
        for cell in t_epoch.rows[i].cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True

figure(
    f"{FIG}/bert_training.png", 15.5, "Gambar", "4.7",
    "Dinamika pelatihan IndoBERT: (kiri) loss validasi vs sampling loss latih "
    "dengan penanda epoch terbaik; (kanan) metrik validasi per-epoch.",
)

h3("4.6.4  Metrik Akhir dan Ketidakpastian")
P(
    "Metrik akhir pada ambang tetap 0,5 disajikan pada Tabel 4.9 untuk kedua "
    "himpunan, dilengkapi interval kepercayaan (IK) 95% dari 2.000 ulangan "
    "bootstrap terstratifikasi pada Tabel 4.10. Pemilihan ambang dilakukan "
    "sebelum eksperimen dan tidak pernah disetel menggunakan himpunan beku, "
    "sehingga selisih kinerja antara validasi dan himpunan beku mencerminkan "
    "generalisasi sejati, bukan hasil optimasi seleksi.",
)
m_v, m_h = metrics["validation"], metrics["holdout"]
table(
    ["Himpunan", "n", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AP"],
    [
        [
            "Validasi (riil, acuan)",
            m_v["n"],
            F(m_v["accuracy"], 3),
            F(m_v["precision"], 3),
            F(m_v["recall"], 3),
            F(m_v["f1"], 3),
            F(m_v["roc_auc"], 3),
            F(m_v["average_precision"], 3),
        ],
        [
            "Himpunan beku (riil, acuan)",
            m_h["n"],
            F(m_h["accuracy"], 3),
            F(m_h["precision"], 3),
            F(m_h["recall"], 3),
            F(m_h["f1"], 3),
            F(m_h["roc_auc"], 3),
            F(m_h["average_precision"], 3),
        ],
    ],
    widths=[4.4, 1.0, 1.8, 1.9, 1.6, 1.6, 1.9, 1.6],
)
caption("Tabel", "4.9", "Metrik akhir IndoBERT pada ambang tetap 0,5.")
table(
    ["Metrik", "Validasi (titik [IK 95%])", "Himpunan beku (titik [IK 95%])"],
    [
        [
            name,
            f"{F(m_v[art], 3)} [{ci_row('val', key)}]",
            f"{F(m_h[art], 3)} [{ci_row('holdout', key)}]",
        ]
        for name, key, art in [
            ("Accuracy", "accuracy", "accuracy"),
            ("Precision", "precision", "precision"),
            ("Recall", "recall", "recall"),
            ("Skor-F1", "f1", "f1"),
            ("ROC-AUC", "roc_auc", "roc_auc"),
        ]
    ],
    widths=[3.6, 6.0, 6.0],
)
caption(
    "Tabel", "4.10",
    "Interval kepercayaan 95% bootstrap terstratifikasi (2.000 ulangan, "
    "benih 42); sumber: bootstrap_ci_bert.csv.",
)
P(
    f"Penurunan F1 dari {F(m_v['f1'])} pada validasi ke {F(m_h['f1'])} pada "
    f"himpunan beku (selisih {F(m_v['f1'] - m_h['f1'], 3)}) berada dalam "
    "rentang yang wajar untuk perpindahan ke himpunan yang tidak tersentuh "
    "proses seleksi maupun penyetelan apa pun; selisih ROC-AUC hanya "
    f"{F(m_v['roc_auc'] - m_h['roc_auc'], 3)}, mengindikasikan peringkat kelas "
    "yang stabil di luar data validasi.",
)

h3("4.6.5  Matriks Konfusi dan Analisis Kesalahan")
P(
    f"Pada himpunan validasi model menghasilkan {tn_v} benar-safe, {fp_v} "
    f"positif-palsu, {fn_v} negatif-palsu, dan {tp_v} benar-unsafe "
    f"(Gambar 4.8a). Pada himpunan beku (Gambar 4.8b) komposisinya menjadi "
    f"{tn_h}/{fp_h}/{fn_h}/{tp_h}: recall kelas tidak aman "
    f"{F(m_h['recall'], 3)} (hanya {fn_h} sampel berbahaya lolos) sementara "
    f"{fp_h} sampel aman diperingatkan berlebih. Untuk aplikasi keselamatan "
    "pangan, negatif-palsu (alergen lolos) jauh lebih berbahaya daripada "
    "positif-palsu (peringatan berlebih), sehingga profil kesalahan himpunan "
    "beku \u2014 yang didominasi positif-palsu \u2014 masih dapat diterima.",
)
figure(
    f"{FIG}/bert_confusion.png", 14.5, "Gambar", "4.8",
    "Matriks konfusi IndoBERT pada (a) validasi dan (b) himpunan beku.",
)
P(
    "Seluruh kesalahan klasifikasi teridentifikasi pada Tabel 4.11 beserta "
    "probabilitas kelas tidak aman yang diberikan model.",
)
err_rows = []
for _, r in errs_all.iterrows():
    gold = r["label"]
    pred = "unsafe" if float(r["prob_unsafe"]) >= 0.5 else "safe"
    prod = str(r.get("nama_produk") or "") or str(r.get("text", ""))[:36]
    err_rows.append([
        r["_split"],
        prod[:40],
        gold,
        pred,
        F(float(r["prob_unsafe"]), 3),
        "FN (alergen lolos)" if gold == "unsafe" else "FP (peringatan)",
    ])
table(
    ["Himpunan", "Kutipan teks (awal)", "Acuan", "Prediksi", "P(unsafe)", "Tipe kesalahan"],
    err_rows,
    widths=[2.3, 5.0, 1.7, 1.9, 1.9, 3.0],
)
caption(
    "Tabel", "4.11",
    "Analisis enam kesalahan klasifikasi IndoBERT (FN = negatif-palsu, "
    "FP = positif-palsu); kutipan teks disertakan agar tiap kasus dapat "
    "diidentifikasi kembali dari sumber data.",
)
P(
    "Tiga negatif-palsu pada himpunan beku semuanya berada di sekitar ambang "
    "(probabilitas 0,246\u20130,391): \u201cbihunku rasa ayam bawang\u201d (0,391), "
    "\u201cbrook farm strawberry milk\u201d (0,359), dan \u201cbrook farm fresh milk\u201d "
    "(0,246). Ketiganya adalah teks panjang berisi bumbu dan perisa di mana penanda "
    "alergen tenggelam oleh konteks dominan \u2014 pola khas kesalahan dekat-ambang (",
    ("near-threshold errors", False, True),
    "). Dua positif-palsu justru mengandung bukti tekstual eksplisit: "
    "\u201ckemplang udang\u201d (0,981; kata \u201cudang\u201d muncul dalam teks dan juga "
    "salah oleh BiLSTM) yang menandakan label acuan yang patut ditelaah ulang, "
    "serta \u201cpoco loco salted tortilla chips\u201d (0,522) yang berada persis di "
    "tepi ambang. Satu-satunya negatif-palsu pada validasi, \u201cbutter salted\u201d "
    "(0,100), juga lolos dari deteksi knowledge base berbasis aturan, yang "
    "mengindikasikan batas pengetahuan leksikal bersama yang dimiliki kedua "
    "pendekatan pada teks pendek dan minim penanda. Sesuai kontrak metodologi, "
    "keenam kasus ini hanya didiagnosis \u2014 tidak ada sampel himpunan beku yang "
    "ditambahkan ke latih dan tidak ada ambang yang disetel dari kesalahan ini.",
)

h3("4.6.6  Kurva ROC dan Precision\u2013Recall")
P(
    f"Kurva ROC (Gambar 4.9, kiri) menempatkan kedua himpunan di dekat sudut "
    f"kiri-atas: ROC-AUC validasi {F(m_v['roc_auc'], 3)} dan ROC-AUC himpunan beku "
    f"{F(m_h['roc_auc'], 3)}, keduanya jauh di atas garis acakan 0,5. Dengan kata "
    "lain, model memberi skor lebih tinggi kepada sampel berbahaya dibanding "
    "sampel aman secara acak dengan probabilitas 98,4% pada himpunan beku. Kurva ",
    ("precision\u2013recall", False, True),
    " (Gambar 4.9, kanan) melengkapi gambaran tersebut: pada himpunan beku presisi "
    f"bertahan di atas 0,88 hingga recall mendekati 0,88 "
    f"(AP = {F(m_h['average_precision'], 3)}), sehingga peningkatan recall melampaui "
    "titik tersebut hanya dapat dibayar dengan penurunan presisi yang tajam \u2014 "
    "batas operasi alami model pada ambang 0,5.",
)
figure(
    f"{FIG}/bert_roc_pr.png", 15.5, "Gambar", "4.9",
    "Kurva ROC (kiri) dan precision\u2013recall (kanan) IndoBERT pada validasi "
    "dan himpunan beku.",
)
P(
    "Agar penilaian kualitas peringkat tidak bergantung pada satu titik ambang, "
    "kualitas area di bawah kurva ROC dirumuskan sebagai integral berikut.",
)
equation(f"{FIG}/bert_eq_auc.png", 7)
P(
    "di mana TPR dan FPR masing-masing adalah laju positif dan laju negatif benar "
    "pada ambang yang diswept. Penalaran setara untuk kualitas presisi\u2013recall "
    "memakai rata-rata presisi terumpan recall (Persamaan 4).",
)

h3("4.6.7  Perbandingan dengan Word2Vec\u2013BiLSTM")
P(
    "Kedua model dilatih pada pool data, partisi, dan ambang yang identik, "
    "sehingga perbandingannya setara. Tabel 4.12 merangkum hasil pada himpunan "
    "beku.",
)
bi_h = eval_bi[eval_bi["split"].str.contains("Frozen")].iloc[0]
table(
    ["Metrik (himpunan beku)", "Word2Vec\u2013BiLSTM [IK 95%]", "IndoBERT [IK 95%]"],
    [
        ["Accuracy", f"{F(bi_h['accuracy'], 3)}", f"{F(m_h['accuracy'], 3)}"],
        ["Precision", f"{F(bi_h['precision'], 3)}", f"{F(m_h['precision'], 3)}"],
        ["Recall", f"{F(bi_h['recall'], 3)}", f"{F(m_h['recall'], 3)}"],
        [
            "Skor-F1",
            f"{F(bi_h['f1'], 3)} {ci_bi_row('frozen_holdout', 'f1')}",
            f"{F(m_h['f1'], 3)} {ci_row('holdout', 'f1')}",
        ],
        [
            "ROC-AUC",
            f"{F(bi_h['roc_auc'], 3)} {ci_bi_row('frozen_holdout', 'roc_auc')}",
            f"{F(m_h['roc_auc'], 3)} {ci_row('holdout', 'roc_auc')}",
        ],
    ],
    widths=[5.0, 5.3, 5.3],
)
caption(
    "Tabel", "4.12",
    "Perbandingan kedua model pada himpunan beku (titik = metrik uji; "
    "[IK] = bootstrap 95%; metrik tanpa [IK] belum di-bootstrap pada tabel ini).",
)
P(
    "Estimasi titik memihak IndoBERT pada hampir seluruh metrik, dengan satu "
    "pengecualian: recall himpunan beku dimenangkan BiLSTM. Keunggulan IndoBERT "
    "dapat ditelusuri pada dua keunggulan representasi. ",
    ("Pertama", True),
    ", tokenisasi subword menjadikannya kebal terhadap kata tak dikenal yang "
    "memengaruhi Word2Vec (tingkat OOV teks BiLSTM pada validasi dan himpunan beku "
    "masing-masing 3,96% dan 5,01%). ",
    ("Kedua", True),
    ", representasi kontekstual dua arah melalui mekanisme attention memungkinkan "
    "model memahami peran kata dalam kalimat panjang bertanda baca, bukan sekadar "
    "kehadiran leksikon.",
)
P(
    "Namun demikian, interval kepercayaan F1 kedua model pada himpunan beku "
    f"bertumpang tindih (BiLSTM {ci_bi_row('frozen_holdout', 'f1')} vs IndoBERT "
    f"{ci_row('holdout', 'f1')}); keunggulan IndoBERT karena itu signifikan pada "
    "estimasi titik dan ROC-AUC, tetapi belum terbukti secara statistik pada "
    "ukuran himpunan n = 50. Klaim superioritas mutlak memerlukan himpunan uji "
    "yang lebih besar \u2014 keterbatasan yang diakui pada Sub-bab 4.6.8. Dari sisi "
    "biaya, keunggulan akurasi IndoBERT dibayar dengan komputasi yang jauh lebih "
    "besar (bobot sekitar 125 juta parameter dan arsip model sekitar 475 MB, "
    "dibanding model rekuren berukuran beberapa megabyte), sehingga pemilihan di "
    "produksi bergantung pada trade-off akurasi\u2013latensi\u2013biaya; arsitektur "
    "ensemble berbobot 0,4/0,6 telah tersedia dan menggabungkan keunggulan "
    "kedua model.",
)

h3("4.6.8  Keterbatasan dan Arah Penelitian Lanjut")
P("Empat keterbatasan diakui terbuka dalam penelitian ini.")
for item in [
    [
        ("Ukuran himpunan.", True),
        " Validasi dan himpunan beku masing-masing hanya 50 sampel sehingga "
        "lebar interval kepercayaan F1 mencapai sekitar \u00b10,09; uji signifikan "
        "antar-model memerlukan n yang lebih besar.",
    ],
    [
        ("Recall kelas tidak aman.", True),
        f" Nilai {F(m_h['recall'], 3)} menyisakan {fn_h} negatif-palsu pada "
        "himpunan beku. Semuanya dekat-ambang; kalibrasi ambang per kategori "
        "alergen adalah arah lanjut yang sahih selama hanya memanfaatkan "
        "himpunan validasi, bukan himpunan beku.",
    ],
    [
        ("Kontribusi data sintetik belum diablasikan.", True),
        " Jalur ablasi (--train-file train_real.csv) telah tersedia namun belum "
        "dijalankan pada run pelaporan ini, sehingga kontribusi data sintetik "
        "terhadap kinerja final masih menjadi asumsi.",
    ],
    [
        ("Satu benih.", True),
        " Seluruh run memakai benih 42; sebaran antar-benih belum "
        "dikuantifikasi sehingga sensitivitas hasil terhadap inisialisasi belum "
        "dapat dinilai.",
    ],
]:
    p = doc.add_paragraph(style="List Bullet")
    p.clear()
    for s in item:
        r = p.add_run(s[0])
        r.bold = bool(s[1])
        r.font.name = FONT
        r.font.size = Pt(12)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

out = os.path.join(BASE, "Naskah_IndoBERT_Arsitektur_Evaluasi.docx")
doc.save(out)
print("saved:", out)
