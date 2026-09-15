"""Builder naskah docx: Metodologi (Bag. 3) + Hasil & Pembahasan (Bag. 4). Bahasa Indonesia."""
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "..", "paper_figs")
FONT = "Times New Roman"
GRAY = "D9D9D9"

doc = Document()

# ---------- halaman & style dasar ----------
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


def _add_field(par, instr):
    run = par.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    f2 = OxmlElement("w:instrText"); f2.set(qn("xml:space"), "preserve"); f2.text = instr
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(f2); run._r.append(f3)


def add_toc():
    p = doc.add_paragraph()
    _add_field(p, 'TOC \\o "1-3" \\h \\z \\u')
    doc.add_paragraph()


def add_page_numbers():
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_field(fp, "PAGE")


def h1(t):
    doc.add_heading(t, level=1)


def h2(t):
    doc.add_heading(t, level=2)


def h3(t):
    doc.add_heading(t, level=3)


def P(*segs, align=None, size=12, space_after=6):
    """Paragraf dari segmen (teks, bold, italic, superscript). String polos = normal."""
    p = doc.add_paragraph()
    for s in segs:
        if isinstance(s, str):
            r = p.add_run(s)
        else:
            txt = s[0]
            r = p.add_run(txt)
            r.bold = bool(len(s) > 1 and s[1])
            r.italic = bool(len(s) > 2 and s[2])
            if len(s) > 3 and s[3]:
                r.font.superscript = True
        r.font.name = FONT
        r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    return p


def bullets(items, numbered=False):
    for it in items:
        p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
        p.clear()
        if isinstance(it, str):
            it = [it]
        for s in it:
            if isinstance(s, str):
                r = p.add_run(s)
            else:
                r = p.add_run(s[0]); r.bold = bool(s[1])
            r.font.name = FONT
            r.font.size = Pt(12)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def table(headers, rows, widths=None, size=10):
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
        r.bold = True; r.font.name = FONT; r.font.size = Pt(size)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), GRAY); shading.set(qn("w:val"), "clear")
        c._tc.get_or_add_tcPr().append(shading)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.text = ""
            r = c.paragraphs[0].add_run(str(val))
            r.font.name = FONT; r.font.size = Pt(size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(kind, num, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{kind} {num}. {text}")
    r.bold = True; r.font.name = FONT; r.font.size = Pt(10)
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
    r.font.name = FONT; r.font.size = Pt(12)


def F(x, nd=4):
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


# ================= JUDUL =================
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Deteksi Alergen pada Komposisi Produk Pangan Kemasan Menggunakan "
              "Word2Vec–BiLSTM dan Fine-Tuning IndoBERT dengan Protokol Evaluasi Anti-Leakage")
r.bold = True; r.font.name = FONT; r.font.size = Pt(14)
doc.add_paragraph()
add_toc()
add_page_numbers()

# ================= 3. METODOLOGI =================
h1("3.  METODOLOGI PENELITIAN")
P("Penelitian ini dirancang sebagai penelitian eksperimental komparatif untuk menjawab dua pertanyaan utama. ",
  ("Pertama", True), ", seberapa akurat model Word2Vec–BiLSTM yang dilatih dengan protokol anti-leakage dalam mengklasifikasikan teks komposisi produk pangan sebagai aman (",
  ("safe", False, True), ") atau tidak aman (", ("unsafe", False, True),
  ") bagi penderita alergi. ", ("Kedua", True),
  ", bagaimana kinerjanya dibandingkan model Transformer berbahasa Indonesia (IndoBERT) yang di-",
  ("fine-tuning", False, True),
  " pada partisi data yang identik. Seluruh tahapan dirancang agar dapat direproduksi dan diaudit: kode, konfigurasi, manifes partisi, serta riwayat pelatihan diabadikan sebagai artefak eksperimen.")
P("Diagram alir tahapan penelitian ditunjukkan pada Gambar 3.1.")
figure(f"{FIG}/fig_diagram_metodologi.png", 15.5, "Gambar", "3.1",
       "Diagram alir tahapan penelitian, dari studi literatur hingga evaluasi dan audit.")

h2("3.1  Studi Literatur")
P("Tahap pertama adalah studi literatur untuk memetakan tiga hal. ",
  ("Pertama", True), ", landasan representasi teks untuk klasifikasi komposisi pangan, mencakup ",
  "model embedding prediktif Word2Vec dengan arsitektur ",
  ("skip-gram", False, True), " dan jaringan saraf rekuren dua arah (",
  ("bidirectional long short-term memory", False, True), ", BiLSTM) untuk pemodelan sekuensi. ",
  ("Kedua", True), ", praktik ",
  ("fine-tuning", False, True), " model bahasa Transformer pra-latih untuk klasifikasi teks, "
  "khususnya varian berbahasa Indonesia. ", ("Ketiga", True),
  ", metodologi evaluasi yang sahih untuk data kecil dan tidak seimbang, mencakup pemisahan himpunan beku "
  "(", ("frozen holdout", False, True), "), kurva ROC dan ",
  ("precision–recall", False, True), ", serta interval kepercayaan bootstrap. Hasil studi literatur menjadi dasar "
  "pemilihan arsitektur, protokol partisi data, dan protokol evaluasi pada sub-bab berikutnya.")

h2("3.2  Pengumpulan Data")
P("Data dikumpulkan dari tiga sumber yang saling melengkapi. Sumber ",
  ("pertama", True), " adalah foto kemasan produk pangan yang beredar di ritel Indonesia, yang merepresentasikan "
  "kondisi pengambilan data dunia nyata (variasi pencahayaan, sudut, dan tata letak label). Sumber ",
  ("kedua", True), " adalah teks komposisi yang tertera pada kemasan, diperoleh melalui ekstraksi ",
  ("optical character recognition", False, True), " (OCR) dan kurasi digital, sehingga setiap sampel berupa pasangan nama produk "
  "dan teks komposisi. Sumber ", ("ketiga", True), " adalah pelabelan acuan manual (",
  ("gold label", False, True), ") oleh anotator yang menetapkan status aman atau tidak aman berdasarkan kandungan alergen "
  "pada komposisi. Sesuai kaidah penulisan metodologi, rincian jumlah sampel, distribusi label, dan contoh data dilaporkan "
  "pada Bab Hasil (Sub-bab 4.1), bukan di sini.")

h2("3.3  Pra-pemrosesan Data")
P("Pra-pemrosesan dilakukan dalam dua jalur, yaitu jalur citra dan jalur teks. Pada jalur citra, setiap foto kemasan "
  "melalui penapisan bilateral untuk meredam derau, pengambangan adaptif Otsu, normalisasi latar, dan operasi morfologi "
  "penutup sebelum diumpankan ke mesin OCR. Daerah komposisi dilokalisasi melalui jangkar kata kunci (misalnya ",
  ("“komposisi”", False, True), ", ", ("“composition”", False, True), ", ", ("“ingredients”", False, True),
  ") dengan pola penghentian pada bagian label lain seperti informasi nilai gizi, sehingga teks di luar komposisi tidak terbawa. "
  "Pada jalur teks, normalisasi yang diterapkan meliputi pelipatan huruf kecil, penghilangan karakter non-alfanumerik, dan "
  "pemadatan spasi. Secara sadar ", ("tidak", True), " dilakukan pembuangan stopword, karena token fungsi bahasa Indonesia "
  "dapat membawa konteks yang relevan bagi tokenizer dan model embedding. Teks kosong dibuang dan didokumentasikan.")

h2("3.4  Pembangunan Knowledge Base Alergen")
P("Sebagai pembanding diagnostik dibangun knowledge base (KB) berbasis aturan yang mencakup delapan kategori alergen kanonis, "
  "yaitu gluten, susu (", ("dairy", False, True), "), telur (", ("egg", False, True), "), kedelai (",
  ("soy", False, True), "), ikan (", ("fish", False, True), "), krustasea (", ("crustacean", False, True),
  "), wijen (", ("sesame", False, True), "), dan kacang pohon (", ("tree nut", False, True),
  "). Setiap kategori didefinisikan melalui daftar istilah kanonis dwibahasa (Indonesia–Inggris) beserta sinonimnya. "
  "Pencocokan dilakukan dalam tiga lapis: (i) pencocokan eksak dan sinonim berbasis batas kata, (ii) pencocokan fuzzy "
  "konservatif untuk toleransi galat OCR, dan (iii) penangkapan bukti non-kanonis (misalnya sulfit) serta pola peringatan "
  "(", ("warning patterns", False, True), " seperti frasaولة “mengandung alergen” atau “diproduksi dengan peralatan yang juga memproses …”). "
  "Setiap keputusan KB disertai tingkat keyakinan, kategori yang terpukul, istilah yang cocok, dan metode buktinya. "
  "Ditegaskan bahwa KB berkedudukan sebagai ", ("bukti diagnostik, bukan kebenaran acuan", True),
  "; KB tidak pernah menggantikan gold label dalam pelatihan maupun evaluasi.")
table(["Kategori", "Contoh istilah kanonis"],
      [["Gluten", "tepung terigu, gandum, gluten, barley"],
       ["Dairy", "susu, susu bubuk, whey, keju, mentega"],
       ["Egg", "telur, telur bubuk, albumen"],
       ["Soy", "kedelai, soya, lesitin kedelai"],
       ["Fish", "ikan, tuna, salmon, sarden, teri"],
       ["Crustacean", "udang, kepiting, lobster, krustasea"],
       ["Sesame", "wijen, biji wijen"],
       ["Tree Nut", "almond, kacang mete/mede, kenari, pistachio"]],
      widths=[4.5, 11.0])
caption("Tabel", "3.1", "Delapan kategori alergen kanonis beserta contoh istilahnya.")

h2("3.5  Pelabelan Acuan")
P("Setiap sampel teks komposisi diberi satu label biner acuan: ", ("safe", False, True),
  " (kode 0) atau ", ("unsafe", False, True), " (kode 1). Label acuan diperlakukan sebagai satu-satunya ",
  ("ground truth", False, True), " untuk pelatihan dan evaluasi; label otomatis KB hanya dipakai untuk audit kualitas KB "
  "dan untuk validasi konsistensi data sintetik. Baris berlabel di luar kosakata biner tersebut dibuang.")

h2("3.6  Partisi Data Anti-Leakage")
P("Seluruh sampel riil dipartisi tepat satu kali menjadi tiga himpunan yang saling lepas: himpunan latih, himpunan validasi, "
  "dan himpunan beku (", ("frozen holdout", False, True), "). Dua mekanisme anti-leakage diterapkan secara simultan melalui struktur ",
  ("union-find", False, True), ": (i) grup produk — nama produk yang ternormalisasi sama wajib berada pada split yang sama, dan "
  "(ii) teks ternormalisasi yang identik secara eksak (hash SHA-256) wajib berada pada split yang sama. Himpunan beku dikunci "
  "berdasarkan daftar nama produk, diperluas ke komponen penuhnya, dan setiap konflik komponen menggagalkan eksperimen alih-alih "
  "diubah diam-diam. Himpunan validasi dipilih secara deterministik dengan komposisi kelas eksak dari sisa data pengembangan. "
  "Himpunan beku tidak pernah memasuki ", ("model.fit()", False, True), ", penyetelan hiperparameter, maupun penentuan ambang. "
  "Hasil partisi diabadikan dalam manifes split beserta audit ketiadaan irisan grup dan teks antar-split.")

h2("3.7  Pembangkitan Data Sintetik")
P("Untuk memperkaya himpunan latih yang kecil, data sintetik dibangkitkan ", ("setelah", True),
  " partisi data riil selesai dan ", ("hanya", True), " dari himpunan latih riil, sehingga validasi dan himpunan beku steril "
  "dari proses ini (kontrak sumber ", ("real_train_only", False, True), "). Frasa umum dipelajari dari bagian teks latih yang lolos KB "
  "sebagai aman, sedangkan frasa alergen dipelajari per kategori dari latih riil dengan daftar cadangan kanonis agar kedelapan "
  "kategori selalu terwakili. Sampel aman dibangun dari kombinasi frasa umum dan diperiksa ulang oleh KB; sampel tidak aman dipaksa "
  "mengandung minimal satu alergen kanonis dan diperiksa ulang pula. Setiap ketidakcocokan antara label sintetik dan keputusan KB "
  "menggagalkan pembangkitan. Himpunan latih final adalah gabungan latih riil dan latih sintetik; validasi dan himpunan beku tetap "
  "murni sampel riil berlabel acuan.")

h2("3.8  Representasi Fitur Word2Vec")
P("Representasi kata dipelajari dengan Word2Vec arsitektur ", ("skip-gram", False, True),
  " ", ("hanya pada teks latih final", True), " (latih riil ditambah sintetik); validasi dan himpunan beku hanya ditransformasi. "
  "Model ", ("skip-gram", False, True), " memaksimalkan peluang kata-kata konteks di sekitar tiap kata target pada jendela konteks, "
  "sehingga kata berkonteks serupa memperoleh vektor berdekatan. Matriks embedding dibangun dengan memetakan indeks tokenizer "
  "ke vektor Word2Vec, sedangkan kata tak terlihat diinisialisasi acak dan baris padding dinolkan. Tingkat kata tak dikenal (",
  ("out-of-vocabulary", False, True), ", OOV) diaudit per split.")

h2("3.9  Arsitektur Model")
h3("3.9.1  Word2Vec–BiLSTM")
P("Arsitektur BiLSTM tersusun atas lapisan embedding (bobot Word2Vec, dapat dilatih, dengan masking token padding), dua lapisan ",
  ("bidirectional LSTM", False, True), " bertumpuk (128 dan 64 unit) yang masing-masing diikuti dropout, satu lapisan padat ReLU 64 unit "
  "dengan dropout, serta satu neuron sigmoid yang mengeluarkan peluang kelas tidak aman (Gambar 3.2). Sel LSTM memelihara status sel "
  "melalui gerbang lupa, masuk, dan keluar sebagaimana Persamaan (1), sedangkan BiLSTM menggabungkan status arah maju dan mundur "
  "sebagaimana Persamaan (2).")
figure(f"{FIG}/fig_arsitektur_bilstm.png", 15.5, "Gambar", "3.2", "Arsitektur Word2Vec–BiLSTM yang digunakan.")
equation(f"{FIG}/eq_lstm.png", 1)
equation(f"{FIG}/eq_bilstm.png", 2)

h3("3.9.2  Fine-Tuning IndoBERT")
P("Sebagai pembanding digunakan IndoBERT (model bahasa Transformer pra-latih berbahasa Indonesia) yang di-",
  ("fine-tuning", False, True), " untuk klasifikasi sekuensi dua kelas dengan kepala softmax dua neuron. Teks dimasukkan apa adanya "
  "kepada tokenizer subword bawaan tanpa pembuangan stopword, agar konteks linguistik utuh. Pelatihan memakai data gabungan yang sama "
  "persis dengan BiLSTM (latih riil ditambah sintetik) pada belahan validasi dan himpunan beku yang identik, sehingga perbandingan "
  "bersifat apel-vs-apel.")

h2("3.10  Protokol Pelatihan")
P("Kedua model dilatih dengan fungsi rugi ", ("binary cross-entropy", False, True), " (Persamaan 3) dan ambang keputusan tetap 0,5 "
  "yang tidak pernah disetel. Hiperparameter BiLSTM ditetapkan tunggal (tanpa pencarian acak): batch 16, maksimum 20 epoch, optimizer "
  "Adam dengan laju belajar 1×10", ("-4", False, False, True), ", gradient clipping, ",
  ("early stopping", False, True), " berbasis ", ("validation loss", False, True),
  " dengan pengembalian bobot terbaik, serta reduksi laju belajar saat plateau. Fine-tuning IndoBERT memakai laju belajar 2×10",
  ("-5", False, False, True), ", batch 16, panjang maksimum 256, maksimum 4 epoch, dan ", ("early stopping", False, True),
  " patience 2. Reproduksibilitas dikunci melalui benih global 42, operasi deterministik, dan penetapan ulang benih sebelum setiap tahap; "
  "seluruh konfigurasi, manifes eksperimen, dan riwayat pelatihan diabadikan.")
equation(f"{FIG}/eq_bce.png", 3)
table(["Hiperparameter", "BiLSTM", "IndoBERT"],
      [["Data latih", "Riil + sintetik (pool sama)", "Riil + sintetik (pool sama)"],
       ["Batch / epoch maks.", "16 / 20", "16 / 4"],
       ["Laju belajar", "1×10−4 (Adam)", "2×10−5"],
       ["Kriteria berhenti", "Early stopping (patience 4, pantau val_loss)", "Early stopping (patience 2, pantau eval_loss)"],
       ["Ambang keputusan", "Tetap 0,5", "Tetap 0,5"],
       ["Benih", "42 (deterministik)", "42"]],
      widths=[5.0, 5.2, 5.2])
caption("Tabel", "3.2", "Ringkasan protokol pelatihan kedua model.")

h2("3.11  Protokol Evaluasi")
P("Evaluasi memakai akurasi, presisi, recall, dan skor-F1 kelas tidak aman, dilengkapi ROC-AUC dan ",
  ("average precision", False, True), " dari kurva ", ("precision–recall", False, True),
  ". Definisi presisi, recall, dan F1 diberikan pada Persamaan (4); ROC-AUC dibaca sebagai peluang model memberi skor lebih tinggi "
  "pada sampel positif acak dibanding sampel negatif acak, sehingga 0,5 berarti setara acakan. Matriks konfusi dilaporkan untuk "
  "membedah kesalahan per kelas. Ketidakpastian dilaporkan sebagai interval kepercayaan 95% dari 2.000 ulangan bootstrap terstratifikasi. "
  "Analisis kesalahan dilakukan melalui triase tiga arah (acuan vs KB vs model) yang membedakan kesalahan murni model dari kasus yang "
  "bahkan KB pun salah, tanpa menyetel apa pun dari himpunan beku.")
equation(f"{FIG}/eq_f1.png", 4)

# ================= 4. HASIL DAN PEMBAHASAN =================
h1("4.  HASIL DAN PEMBAHASAN")

h2("4.1  Pengumpulan Data")
P("Pengumpulan data menghasilkan 499 teks komposisi produk pangan kemasan yang beredar di ritel Indonesia, terdiri atas 251 sampel ",
  ("unsafe", False, True), " (50,3%) dan 248 sampel ", ("safe", False, True),
  " (49,7%), sehingga kedua kelas praktis seimbang (Gambar 4.1a). Sumber data terdiri atas tiga komponen: (i) 114 foto kemasan produk "
  "(arsip Dataset-Gambar.zip) yang merepresentasikan kondisi akuisisi dunia nyata, (ii) teks komposisi hasil ekstraksi OCR yang dikurasi "
  "bersama varian digitalnya (berkas data.csv dan data-mengandung.csv), serta (iii) pelabelan acuan manual per produk. Sebaran label acuan "
  "per split disajikan pada Gambar 4.1b, sedangkan tiga contoh sampel ditampilkan pada Tabel 4.1.")
figure(f"{FIG}/fig_distribusi_label.png", 15.0, "Gambar", "4.1",
       "Sebaran label acuan: (a) keseluruhan data riil (n=499); (b) per split (latih n=399, validasi n=50, holdout n=50).")
table(["Produk", "Label", "Kutipan teks komposisi"],
      [["alba cheese", "unsafe", "susu, garam, rennet dari yeast kluyveromyces lactis kultur starter (streptococcus thermophilus, …). Mengandung alergen …"],
       ["almond cookies", "unsafe", "Tepung Terigu, Gula Kristal, Air, Kacang Almond (8%), Mentega Tawar, Mentega, Garam, Susu Evaporasi …"],
       ["bihunku rasa soto", "safe", "Bihun: Pati Jagung, Beras, Pati Tapioka, Pengemulsi Nabati, … Minyak Bumbu: Minyak Nabati, … Cabe Bubuk"]],
      widths=[3.6, 2.0, 9.9])
caption("Tabel", "4.1", "Contoh sampel data: nama produk, label acuan, dan kutipan teks komposisi.")
P("Dua hal patut digarisbawahi dari Tabel 4.1. ", ("Pertama", True),
  ", teks komposisi bersifat panjang, berisik, dan multibahasa (misalnya istilah Latin kultur starter), sehingga representasi berbasis "
  "kata utuh menghadapi kosakata jarang. ", ("Kedua", True),
  ", penanda alergen sering muncul implisit (misalnya “Mentega” sebagai turunan susu) atau dalam peringatan pemrosesan, sehingga aturan "
  "kata-kunci naif tidak mencukupi — temuan ini memotivasi pembangunan KB pada Sub-bab 4.2 dan pemodelan sekuensial pada Sub-bab 4.4–4.5.")

h2("4.2  Pra-pemrosesan dan Audit Knowledge Base")
P("Audit KB terhadap 499 label acuan menghasilkan akurasi 0,733, presisi kelas ", ("unsafe", False, True),
  " 0,663, recall 0,956, dan F1 0,783 (Tabel 4.2). Polanya jelas: KB hampir tidak pernah melewatkan sampel tidak aman (recall tinggi), "
  "tetapi sering menandai sampel aman sebagai tidak aman (presisi rendah). Perilaku ini konsisten dengan desain KB yang mengutamakan ",
  ("peringatan dini", False, True), " — frasa seperti “diproduksi dengan peralatan yang juga memproses …” atau kandungan non-kanonis "
  "seperti sulfit langsung memicu status tidak aman. Konsekuensinya, KB tepat sebagai bukti diagnostik dan validator data sintetik, "
  "tetapi tidak layak sebagai kebenaran acuan; keputusan ini dipertahankan di seluruh eksperimen.")
table(["Metrik (kelas unsafe)", "Nilai"],
      [["Akurasi", "0,733"], ["Presisi", "0,663"], ["Recall", "0,956"], ["Skor-F1", "0,783"], ["n", "499"]],
      widths=[7.5, 8.0])
caption("Tabel", "4.2", "Kinerja KB berbasis aturan terhadap label acuan (diagnostik, bukan acuan).")
P("Audit tingkat token memperkuat kebutuhan pra-pemrosesan yang hati-hati. Pada teks latih riil, rata-rata panjang sekuens adalah "
  "34,3 token (median 28; persentil-95 sebesar 75), sehingga batas max_len=120 menyisakan sekitar 71% padding — fakta yang mendasari "
  "keputusan masking pada arsitektur. Sebanyak 37,5% kosakata latih riil adalah hapax legomena (muncul sekali), sekalipun hanya mencakup "
  "2,5% kemunculan token; 132 token berupa digit murni (kadar persen dan takaran) memecah kosakata. Temuan ini menjadi dasar eksperimen "
  "ablasi pra-pemrosesan, yang menyimpulkan bahwa masking padding membantu, sedangkan pelipatan digit justru menghilangkan sinyal kadar "
  "yang diskriminatif.")

h2("4.3  Partisi Data dan Audit Leakage")
P("Partisi deterministik menghasilkan latih riil 399 sampel (198 safe, 201 unsafe), validasi 50 sampel (25/25), dan himpunan beku 50 sampel "
  "(25/25); ditambah 1.000 sampel sintetik (500/500) sehingga pool latih final berukuran 1.399 (Tabel 4.3). Seluruh audit ketiadaan irisan "
  "grup produk maupun teks eksak antar-split berstatus PASS, termasuk audit pool latih final terhadap validasi dan himpunan beku. "
  "Himpunan beku dikunci berdasarkan daftar nama produk dan tidak pernah memasuki pelatihan maupun penyetelan dalam bentuk apa pun.")
table(["Himpunan", "n", "Safe", "Unsafe", "Sumber"],
      [["Latih riil", "399", "198", "201", "Label acuan"],
       ["Latih sintetik", "1.000", "500", "500", "Generator KB (sumber latih-riil saja)"],
       ["Validasi", "50", "25", "25", "Label acuan"],
       ["Himpunan beku", "50", "25", "25", "Label acuan (terkunci)"]],
      widths=[4.0, 2.2, 2.2, 2.2, 4.9])
caption("Tabel", "4.3", "Komposisi partisi data final.")

h2("4.4  Dinamika Pelatihan BiLSTM")
P("Riwayat pelatihan BiLSTM selama maksimum 20 epoch ditunjukkan pada Gambar 4.2. Tiga fase terlihat jelas. ",
  ("Fase pertama", True), " (epoch 1–5) adalah fase belajar cepat: akurasi latih melonjak dari 0,479 menjadi 0,711 dan ",
  ("training loss", False, True), " turun dari 0,706 menjadi 0,587, sementara ", ("validation loss", False, True),
  " anjlok dari 0,643 menjadi 0,335 — model menemukan struktur utama data. ", ("Fase kedua", True),
  " (epoch 6–16) adalah fase pemurnian: kedua kurva loss menurun monoton dan beriringan hingga titik terbaik pada epoch 16 "
  "(", ("validation loss", False, True), " 0,182; akurasi latih 0,924; akurasi validasi 0,92), tanpa pelebaran kesenjangan latih–validasi. ",
  ("Fase ketiga", True), " (epoch 17–20) menunjukkan gejala awal ", ("overfitting", False, True),
  ": ", ("training loss", False, True), " terus turun hingga 0,134, tetapi ", ("validation loss", False, True),
  " berbalik naik (0,217; 0,292; 0,336; 0,345). Mekanisme ", ("early stopping", False, True),
  " menangkap gejala ini dengan tepat dan mengembalikan bobot epoch 16, sehingga model final bebas dari overfitting lanjut.")
figure(f"{FIG}/fig_training_bilstm.png", 15.5, "Gambar", "4.2",
       "Riwayat pelatihan BiLSTM: (kiri) akurasi dan F1 latih vs validasi; (kanan) loss latih vs validasi. Garis vertikal menandai epoch terbaik (16).")
P("Dua catatan metodologis penting. ", ("Pertama", True),
  ", fluktuasi kecil kurva validasi (misalnya akurasi validasi berosilasi antara 0,92 dan 0,94) bukanlah overfitting, melainkan kuantisasi "
  "sampel kecil: dengan n=50, satu sampel bernilai 2%. ", ("Kedua", True),
  ", pada epoch 1 akurasi validasi (0,94) tampak lebih tinggi daripada akurasi latih (0,48); hal ini wajar karena regularisasi dropout aktif "
  "hanya saat latih dan himpunan latih mengandung sampel sintetik yang lebih sulit, bukan indikasi kebocoran — audit leakage pada Sub-bab 4.3 "
  "telah membuktikan ketiadaan irisan.")

h2("4.5  Dinamika Pelatihan IndoBERT")
P("Fine-tuning IndoBERT berlangsung selama maksimum 4 epoch (±171 detik) dengan riwayat pada Gambar 4.3. Epoch 1 mencatat ",
  ("training loss", False, True), " 1,271 — nilai tinggi yang wajar karena kepala klasifikasi diinisialisasi acak — dengan ",
  ("eval loss", False, True), " 0,615 dan F1 validasi 0,941. Epoch 2–3 menunjukkan perbaikan tajam (", ("eval loss", False, True),
  " 0,210 lalu 0,144; F1 validasi 0,980 lalu 0,980; ROC-AUC 0,998 lalu 0,995), dan epoch 3 menjadi titik terbaik. Epoch 4 kembali "
  "menunjukkan gejala overfitting awal (", ("eval loss", False, True), " naik ke 0,287; F1 turun ke 0,960), sehingga ",
  ("early stopping", False, True), " mengembalikan bobot epoch 3. Norma gradien yang melonjak pada epoch 3 (26,3 dari 7,5) sempat teramati "
  "tanpa merusak konvergensi; penjepit gradien 1,0 dipasang sebagai pengaman.")
figure(f"{FIG}/fig_training_bert.png", 15.5, "Gambar", "4.3",
       "Riwayat fine-tuning IndoBERT: (kiri) train loss vs eval loss; (kanan) F1 dan ROC-AUC validasi. Garis vertikal menandai epoch terbaik (3).")

h2("4.6  Hasil Evaluasi")
P("Tabel 4.4 dan 4.5 merangkum kinerja kedua model pada ambang tetap 0,5 beserta interval kepercayaan 95% dari 2.000 ulangan bootstrap "
  "terstratifikasi. Pada validasi, IndoBERT unggul di semua metrik (F1 0,980; IK 0,936–1,000; AUC 0,995) atas BiLSTM (F1 0,921; IK 0,840–0,980; "
  "AUC 0,977). Pada himpunan beku — ukuran yang bersih dari bias seleksi — IndoBERT mencatat F1 0,899 (IK 0,809–0,980; AUC 0,984), sedangkan "
  "BiLSTM mencatat F1 0,842 (IK 0,764–0,926; AUC 0,907).")
table(["Model", "Split", "Akurasi", "Presisi", "Recall", "F1 [IK 95%]", "ROC-AUC [IK 95%]"],
      [["BiLSTM", "Validasi (50)", "0,920", "0,920", "0,920", "0,921 [0,840–0,980]", "0,977 [0,930–1,000]"],
       ["BiLSTM", "Beku (50)", "0,820", "0,750", "0,960", "0,842 [0,764–0,926]", "0,907 [0,808–0,987]"],
       ["IndoBERT", "Validasi (50)", "0,980", "1,000", "0,960", "0,980 [0,936–1,000]", "0,995 [0,981–1,000]"],
       ["IndoBERT", "Beku (50)", "0,900", "0,917", "0,880", "0,899 [0,809–0,980]", "0,984 [0,955–1,000]"]],
      widths=[2.2, 2.6, 1.8, 1.8, 1.8, 3.2, 3.2], size=9)
caption("Tabel", "4.4", "Ringkasan evaluasi kedua model pada ambang tetap 0,5 (IK = interval kepercayaan bootstrap 95%, 2.000 ulangan).")

h2("4.7  Analisis Kurva ROC dan Precision–Recall")
P("Kurva ROC dan ", ("precision–recall", False, True), " ditunjukkan pada Gambar 4.4. Keempat kurva ROC menempel di sudut kiri atas dengan "
  "AUC 0,907–0,995, artinya kedua model memisahkan kelas dengan kuat dan jauh di atas garis acak (AUC 0,5). Pada himpunan beku, kurva BiLSTM "
  "tertinggal dari kurva validasinya pada daerah ", ("false positive rate", False, True),
  " rendah — cerminan delapan positif-palsu yang dibahas pada Sub-bab 4.9 — sedangkan kurva IndoBERT validasi dan beku hampir berhimpit, "
  "menandakan generalisasi peringkat yang stabil. Kurva ", ("precision–recall", False, True),
  " mempertegas hal yang sama: presisi IndoBERT bertahan di atas 0,9 pada hampir seluruh rentang recall, sementara presisi BiLSTM pada "
  "himpunan beku meluruh setelah recall 0,9 akibat tumpukan positif-palsu tersebut.")
figure(f"{FIG}/fig_roc_pr.png", 15.5, "Gambar", "4.4",
       "Kurva ROC (kiri) dan precision–recall (kanan) kedua model pada validasi dan himpunan beku.")

h2("4.8  Matriks Konfusi")
P("Matriks konfusi keempat evaluasi ditampilkan pada Gambar 4.5. Pada himpunan beku, BiLSTM menghasilkan 17 benar-safe, 8 positif-palsu, "
  "1 negatif-palsu, dan 24 benar-unsafe; artinya dari 25 sampel aman, sepertiganya salah diklasifikasikan, sementara hampir semua sampel "
  "tidak aman tertangkap (recall 0,96). IndoBERT pada himpunan yang sama menghasilkan 23 benar-safe, 2 positif-palsu, 3 negatif-palsu, dan "
  "22 benar-unsafe — profil kesalahan lebih seimbang. Pada validasi, kedua model nyaris sempurna (BiLSTM: 23/2/2/23; IndoBERT: 25/0/1/24). "
  "Asimetri ini penting secara substantif: untuk aplikasi keselamatan pangan, negatif-palsu (alergen lolos) jauh lebih berbahaya daripada "
  "positif-palsu (peringatan berlebih), sehingga recall tinggi BiLSTM (0,96) tetap bernilai meskipun presisinya lebih rendah.")
figure(f"{FIG}/fig_confusion.png", 13.5, "Gambar", "4.5",
       "Matriks konfusi: (a) BiLSTM validasi, (b) BiLSTM himpunan beku, (c) IndoBERT validasi, (d) IndoBERT himpunan beku.")

h2("4.9  Analisis Kesalahan")
P("Triase tiga arah (acuan vs KB vs model) memisahkan kesalahan murni model dari kasus yang bahkan KB pun salah. Satu-satunya negatif-palsu "
  "BiLSTM adalah “alba cheese” (skor 0,074) — ironisnya teksnya secara eksplisit menyebut “susu” dan KB pun menandainya tidak aman; ini murni "
  "kegagalan representasi yang menjadi kandidat investigasi lanjutan. Dari delapan positif-palsu BiLSTM, lima disepakati oleh KB (misalnya "
  "“basreng super” yang mengandung frasa “Ikan Segar”, dan “rumput laut mama suka” dengan skor 0,993): teks-teks ini memang mengandung bukti "
  "alergen tekstual sehingga label safe-nya patut ditinjau ulang — temuan ini dikembalikan sebagai antrean telaah label, bukan sebagai dasar "
  "penyetelan model. Tiga positif-palsu murni-model (“bihunku rasa soto” 0,998; “kentang home” 0,705; “fukumi beras porang” 0,509) menunjukkan "
  "model masih terkecoh oleh komposisi nabati yang panjang dan kompleks.")
P("Pola kesalahan IndoBERT berbeda dan saling melengkapi: tiga negatif-palsu himpunan beku (“bihunku rasa ayam bawang” 0,391; “brook farm fresh "
  "milk” 0,246; “brook farm strawberry milk” 0,359) seluruhnya adalah produk susu/minuman yang teksnya pendek atau didominasi perisa, sedangkan "
  "dua positif-palsunya (“kemplang udang” 0,981 — juga salah oleh BiLSTM — dan “poco loco salted tortilla chips” 0,522) melibatkan istilah "
  "ambiguitas lintas-kategori. Satu-satunya kesalahan validasi IndoBERT (“butter salted”, skor 0,100) sekaligus merupakan kasus KB-miss, "
  "menunjukkan batas pengetahuan leksikal yang dimiliki bersama oleh pendekatan aturan dan statistik pada frasa tak baku.")

h2("4.10  Perbandingan BiLSTM dan IndoBERT")
P("Perbandingan apel-vs-apel dimungkinkan karena kedua model memakai pool latih, validasi, himpunan beku, dan ambang yang identik. Estimasi titik "
  "memihak IndoBERT di hampir semua metrik (misalnya F1 beku 0,899 vs 0,842; AUC 0,984 vs 0,907), kecuali recall himpunan beku yang dimenangkan "
  "BiLSTM (0,96 vs 0,88). Namun interval kepercayaan 95% keduanya bertumpang-tindih pada himpunan beku (F1: 0,764–0,926 vs 0,809–0,980), sehingga "
  "klaim superioritas statistik memerlukan himpunan uji yang lebih besar — kejujuran ini kami nyatakan eksplisit agar tidak terjadi overklaim. "
  "Secara praktis, IndoBERT unggul karena representasi kontekstual dan tokenisasi subword-nya tahan terhadap kosakata jarang dan galat OCR yang "
  "melumpuhkan embedding kata-utuh Word2Vec (OOV validasi 3,96%; holdout 5,01%), dengan harga komputasi dan ketergantungan pra-latih yang jauh "
  "lebih besar.")
table(["Aspek", "BiLSTM", "IndoBERT"],
      [["F1 beku", "0,842", "0,899"],
       ["AUC beku", "0,907", "0,984"],
       ["Recall beku", "0,960 (unggul)", "0,880"],
       ["Ketahanan OOV/galat OCR", "Terbatas (kata utuh)", "Kuat (subword + konteks)"],
       ["Biaya", "Ringan (CPU)", "Berat (pra-latih + GPU disarankan)"]],
      widths=[5.0, 5.2, 5.2], size=10)
caption("Tabel", "4.5", "Ringkasan perbandingan kedua model.")

h2("4.11  Pembahasan")
P("Kontribusi utama penelitian ini bersifat ganda. ", ("Pertama", True),
  ", dari sisi metodologis, protokol anti-leakage (partisi union-find dua lapis, himpunan beku terkunci, sintetikbersumber latih-riil, ambang tetap) "
  "membuktikan bahwa evaluasi yang jujur dimungkinkan pada data kecil tanpa mengorbankan validitas — seluruh audit berstatus lolos dan setiap "
  "angka dilaporkan beserta ketidakpastiannya. ", ("Kedua", True),
  ", dari sisi empiris, kedua model melampaui baseline aturan KB (F1 0,783) dengan margin besar, dan sistem yang dihasilkan telah dioperasikan "
  "sebagai layanan deteksi (antarmuka teks dan citra) sehingga manfaatnya langsung teruji di luar laboratorium.")
P("Keterbatasan yang diakui secara terbuka adalah sebagai berikut. ", ("Pertama", True),
  ", teks latih berasal dari ekstraksi OCR sehingga galat baca dapat merambat; kasus “butter salted” menunjukkan kegagalan bersama aturan dan "
  "statistik pada frasa tak baku. ", ("Kedua", True),
  ", data sintetik bersifat template dan kosakatanya merupakan subset kosakata riil, sehingga kontribusinya lebih pada regularisasi daripada "
  "pengetahuan baru. ", ("Ketiga", True),
  ", satu ambang global 0,5 belum tentu optimal untuk tiap kategori alergen; analisis ambang per kategori disarankan sebagai riset lanjutan. ",
  ("Keempat", True), ", cakupan produk terbatas pada ritel Indonesia dan delapan kategori kanonis. Riset lanjutan yang disarankan mencakup kalibrasi "
  "probabilitas, arsitektur multimodal citra-teks, serta uji signifikansi pada himpunan beku yang lebih besar.")

h1("DAFTAR PUSTAKA")
for _ref in [
    "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT.",
    "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1723–1757.",
    "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
    "Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. Proceedings of ICLR.",
    "Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems, 26.",
    "Peraturan Badan Pengawas Obat dan Makanan Republik Indonesia Nomor 31 Tahun 2018 tentang Label Pangan Olahan.",
    "Prechelt, L. (1998). Early stopping — but when? Neural Networks: Tricks of the Trade. Springer.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3).",
    "Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11), 2673–2681.",
    "Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427–437.",
    "Codex Alimentarius Commission. (1985, rev. terakhir). General Standard for the Labelling of Prepackaged Foods (CXS 1-1985). FAO/WHO.",
    "Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Wilie, B., et al. (2020). IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding. Proceedings of AACL.",
]:
    _p = doc.add_paragraph(style="List Bullet")
    _p.clear()
    _r = _p.add_run(_ref)
    _r.font.name = FONT
    _r.font.size = Pt(11)
    _p.paragraph_format.left_indent = Cm(1.0)
    _p.paragraph_format.first_line_indent = Cm(-1.0)
    _p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

doc.save(os.path.join(BASE, "Naskah_Metodologi_Hasil_Pembahasan.docx"))
print("saved")

