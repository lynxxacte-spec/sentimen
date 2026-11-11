"""
app.py - Sistem Analisis Sentimen Pengguna Aplikasi Blibli (Streamlit)

Menu: Beranda | Analisis Komentar | Preprocessing | Riwayat & Excel | Visualisasi | Evaluasi Algoritma
Tidak ada menu upload dataset, upload model, maupun klasifikasi tanpa model.
Komentar baru yang diketik pengguna diproses oleh algoritma Multinomial Naive Bayes yang sudah dilatih,
dilengkapi aturan pendukung kelas netral (klasifikasi.py).

File yang harus ada di folder yang sama:
  - app.py
  - model_naive_bayes_blibli_final.pkl     (hasil notebook)
  - hasil_penelitian.json                  (ringkasan hasil 1.060 ulasan untuk menu Visualisasi)
  - klasifikasi.py                         (aturan pendukung kelas netral)
"""
from pathlib import Path
from datetime import datetime
from io import BytesIO
import json
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from klasifikasi import putusan
from openpyxl.styles import Alignment, Font, PatternFill

st.set_page_config(page_title="Analisis Sentimen Blibli", page_icon="📊", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model_naive_bayes_blibli_final.pkl"
HASIL_PATH = BASE / "hasil_penelitian.json"
WARNA = {"Positif": "#2E7D32", "Netral": "#F9A825", "Negatif": "#C62828"}

# ------------------------------------------------------------------ preprocessing (sama dengan notebook)
STOPWORDS_ID = set("""
yang dan di ke dari untuk dengan pada dalam ini itu ada adalah akan
atau juga karena jadi saya kamu kita mereka nya sebagai agar bisa
sudah sangat lebih hanya tentang oleh dapat menjadi suatu para serta
namun tetapi kalau jika saat setelah sebelum terhadap antara tanpa
hingga sampai masih sedang pernah harus mau ingin lagi terlalu banyak
buat lah pun si
""".split())
# Kata negasi sengaja tidak dihapus agar makna kalimat tidak hilang.
NEGATION_WORDS = {
    "tidak", "tak", "bukan", "jangan", "belum", "kurang",
    "ga", "gak", "nggak", "gk", "enggak", "ngga", "ndak",
    "not", "no", "nor", "never", "without",
}
STOPWORDS_ID = STOPWORDS_ID - NEGATION_WORDS


def preprocessing(teks):
    """Preprocessing yang sama dengan saat pelatihan algoritma."""
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#", "", teks)
    teks = re.sub(r"[^a-z0-9\s]", " ", teks)
    teks = re.sub(r"\b\d+\b", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    tokens = teks.split()
    tokens = [k for k in tokens if k not in STOPWORDS_ID and len(k) > 1]
    return " ".join(tokens)


def tahapan_preprocessing(komentar):
    """Hasil pada setiap tahap (untuk menu Preprocessing)."""
    folding = str(komentar).lower()
    t = re.sub(r"http\S+|www\S+", " ", folding)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    cleaning = re.sub(r"\s+", " ", t).strip()
    tokens = cleaning.split()
    akhir = [k for k in tokens if k not in STOPWORDS_ID and len(k) > 1]
    return {
        "Komentar Asli": komentar,
        "Case Folding": folding,
        "Cleaning": cleaning,
        "Tokenisasi": "[" + ", ".join(tokens) + "]",
        "Stopword Removal (Hasil Akhir)": " ".join(akhir),
        "Jumlah Token": f"{len(tokens)} → {len(akhir)}",
    }


# ------------------------------------------------------------------ algoritma & data
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "File 'model_naive_bayes_blibli_final.pkl' tidak ditemukan. "
            "Simpan di folder yang sama dengan file aplikasi ini."
        )
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    if not {"Negatif", "Netral", "Positif"}.issubset(set(model.classes_)):
        raise ValueError("Kelas algoritma tidak sesuai: harus Positif, Netral, dan Negatif.")
    return model


@st.cache_data
def load_hasil():
    if not HASIL_PATH.exists():
        return None
    return json.loads(HASIL_PATH.read_text(encoding="utf-8"))


def predict_comments(model, comments):
    cleaned = [preprocessing(c) for c in comments]
    valid = [i for i, t in enumerate(cleaned) if t.strip()]
    labels, probs, dasar = {}, {}, {}
    if valid:
        texts = [cleaned[i] for i in valid]
        for pos, lab, pr in zip(valid, model.predict(texts), model.predict_proba(texts)):
            labels[pos] = str(lab).capitalize()
            probs[pos] = {c: float(v) for c, v in zip(model.classes_, pr)}
            labels[pos], dasar[pos] = putusan(labels[pos], cleaned[pos])
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for i, (komentar, bersih) in enumerate(zip(comments, cleaned)):
        if not bersih.strip():
            rows.append({
                "Waktu Analisis": waktu, "Komentar": komentar, "Teks Setelah Preprocessing": bersih,
                "Prediksi Sentimen": "Teks tidak cukup informatif",
                "Probabilitas Tertinggi (%)": None, "Probabilitas Positif (%)": None,
                "Probabilitas Netral (%)": None, "Probabilitas Negatif (%)": None,
                "Dasar Keputusan": "-",
            })
            continue
        p = probs[i]
        if dasar[i] != "Algoritma Naive Bayes":
            # label ditentukan aturan: probabilitas Naive Bayes tidak ditampilkan agar tidak membingungkan
            rows.append({
                "Waktu Analisis": waktu, "Komentar": komentar, "Teks Setelah Preprocessing": bersih,
                "Prediksi Sentimen": labels[i],
                "Probabilitas Tertinggi (%)": None, "Probabilitas Positif (%)": None,
                "Probabilitas Netral (%)": None, "Probabilitas Negatif (%)": None,
                "Dasar Keputusan": dasar[i],
            })
            continue
        rows.append({
            "Waktu Analisis": waktu, "Komentar": komentar, "Teks Setelah Preprocessing": bersih,
            "Prediksi Sentimen": labels[i],
            "Probabilitas Tertinggi (%)": round(max(p.values()) * 100, 2),
            "Probabilitas Positif (%)": round(p.get("Positif", 0.0) * 100, 2),
            "Probabilitas Netral (%)": round(p.get("Netral", 0.0) * 100, 2),
            "Probabilitas Negatif (%)": round(p.get("Negatif", 0.0) * 100, 2),
            "Dasar Keputusan": dasar[i],
        })
    return rows


def to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Hasil Analisis", index=False)
        ws = writer.sheets["Hasil Analisis"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for col in ws.columns:
            lebar = min(max(max(len(str(c.value or "")) for c in col) + 2, 12), 45)
            ws.column_dimensions[col[0].column_letter].width = lebar
        ws.row_dimensions[1].height = 30
    buf.seek(0)
    return buf.getvalue()


KOLOM_CFG = {
    "Waktu Analisis": st.column_config.TextColumn("Waktu Analisis", width="medium"),
    "Komentar": st.column_config.TextColumn("Komentar", width="large"),
    "Teks Setelah Preprocessing": st.column_config.TextColumn("Teks Setelah Preprocessing", width="large"),
    "Prediksi Sentimen": st.column_config.TextColumn("Prediksi Sentimen", width="small"),
    "Probabilitas Tertinggi (%)": st.column_config.NumberColumn("Prob. Tertinggi (%)", width="small"),
    "Probabilitas Positif (%)": st.column_config.NumberColumn("Prob. Positif (%)", width="small"),
    "Probabilitas Netral (%)": st.column_config.NumberColumn("Prob. Netral (%)", width="small"),
    "Probabilitas Negatif (%)": st.column_config.NumberColumn("Prob. Negatif (%)", width="small"),
    "Dasar Keputusan": st.column_config.TextColumn("Dasar Keputusan", width="large"),
}


def tampil_grafik(wadah, fig):
    """Tampilkan grafik dengan ukuran alami (tidak melar memenuhi layar)."""
    try:
        wadah.pyplot(fig, width="content")
    except TypeError:                      # Streamlit versi lama
        wadah.pyplot(fig, use_container_width=False)


def grafik_batang(nilai, judul):
    """nilai: dict {Positif: n, Netral: n, Negatif: n}"""
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=100)
    label = ["Positif", "Netral", "Negatif"]
    angka = [int(nilai.get(k, 0)) for k in label]
    batang = ax.bar(label, angka, color=[WARNA[k] for k in label])
    for b in batang:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(b.get_height())), ha="center", va="bottom")
    ax.set_title(judul, fontsize=11)
    ax.set_ylabel("Jumlah")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ halaman
if "history" not in st.session_state:
    st.session_state.history = []

st.title("📊 Analisis Sentimen Pengguna Aplikasi Blibli")
st.caption("Text Mining • TF-IDF kata dan karakter • Algoritma Multinomial Naive Bayes + aturan kelas netral")

try:
    model = load_model()
except Exception as error:
    st.error(str(error))
    st.info("Jika pesannya soal versi scikit-learn atau pickle: pastikan requirements.txt yang diunggah ke GitHub "
            "adalah requirements.txt dari hasil notebook (versi scikit-learn harus sama dengan saat model dibuat).")
    st.stop()

tab_beranda, tab_analisis, tab_pre, tab_riwayat, tab_visual, tab_eval = st.tabs(
    ["Beranda", "Analisis Komentar", "Preprocessing", "Riwayat & Excel", "Visualisasi", "Evaluasi Algoritma"]
)

# ---------------- Beranda
with tab_beranda:
    st.subheader("Tentang Aplikasi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algoritma", "Multinomial Naive Bayes")
    c2.metric("Jenis Analisis", "Sentimen (3 kelas)")
    c3.metric("Sumber Data", "Google Play Store")
    st.write(
        "Aplikasi ini memproses **komentar baru** pengguna aplikasi Blibli. Setiap komentar melalui tahap "
        "pra-pemrosesan, diubah menjadi fitur TF-IDF kata dan karakter, lalu diklasifikasikan oleh algoritma "
        "Multinomial Naive Bayes menjadi sentimen **positif**, **netral**, atau **negatif**. "
        "Karena kelas netral sulit dikenali algoritma saja, hasil algoritma dilengkapi aturan sederhana "
        "(kata bernada sedang, opini campuran, atau kalimat tanpa kata opini) yang ditampilkan pada kolom **Dasar Keputusan**. "
        "Algoritma dilatih pada 600 ulasan berlabel manual dari 1.060 ulasan yang dikumpulkan."
    )
    st.markdown("**Alur penggunaan**")
    st.markdown(
        "1. Buka menu **Analisis Komentar**, lalu ketik satu atau beberapa komentar (satu komentar per baris).\n"
        "2. Tekan **Analisis Sentimen** untuk melihat prediksi dan probabilitas tiap kelas.\n"
        "3. Buka menu **Preprocessing** untuk melihat hasil setiap tahap pembersihan teks.\n"
        "4. Buka **Riwayat & Excel** untuk melihat seluruh hasil sesi ini dan mengunduhnya dalam format Excel.\n"
        "5. Buka **Visualisasi** untuk melihat grafik hasil analisis dan hasil penelitian.\n"
        "6. Buka **Evaluasi Algoritma** untuk melihat akurasi dan kinerja algoritma pada data uji."
    )

# ---------------- Analisis Komentar
with tab_analisis:
    st.subheader("Masukkan Komentar")
    st.caption("Satu komentar per baris. Anda dapat memasukkan satu atau beberapa komentar sekaligus.")
    with st.form("form_analisis_komentar", clear_on_submit=True):
        comment_text = st.text_area(
            "Komentar pengguna",
            placeholder=("Contoh: Aplikasinya bagus dan mudah digunakan.\n"
                         "Contoh: Pesanan saya terlambat dan layanan pelanggan lambat."),
            height=180,
        )
        submitted = st.form_submit_button("Analisis Sentimen", type="primary", width="stretch")
    if submitted:
        comments = [x.strip() for x in comment_text.splitlines() if x.strip()]
        if not comments:
            st.warning("Masukkan setidaknya satu komentar terlebih dahulu.")
        else:
            with st.spinner("Memproses komentar menggunakan algoritma Naive Bayes..."):
                baru = predict_comments(model, comments)
                st.session_state.history.extend(baru)
            st.success(f"{len(baru)} komentar berhasil dianalisis.")
            st.dataframe(pd.DataFrame(baru), width="stretch", hide_index=True, column_config=KOLOM_CFG)
            st.caption("Probabilitas hanya ditampilkan bila label ditentukan langsung oleh algoritma Naive Bayes. "
                       "Jika label ditentukan aturan pendukung, probabilitas dikosongkan dan alasannya tertulis di kolom Dasar Keputusan.")

# ---------------- Preprocessing
with tab_pre:
    st.subheader("Tahapan Preprocessing")
    st.caption("Lihat hasil setiap tahap pembersihan teks: case folding, cleaning, tokenisasi, dan stopword removal. "
               "Kata negasi seperti “tidak”, “gak”, dan “bukan” sengaja dipertahankan.")
    teks_pre = st.text_area("Komentar yang ingin dilihat tahapannya (satu komentar per baris)", height=140,
                            key="teks_pre", placeholder="Contoh: Penarikan saldo Blibli Pay butuh waktu sampai 3 hari kerja. TIDAK BISA REAL TIME")
    if st.button("Jalankan Preprocessing"):
        baris = [x.strip() for x in teks_pre.splitlines() if x.strip()]
        if not baris:
            st.warning("Masukkan setidaknya satu komentar terlebih dahulu.")
        else:
            for no, b in enumerate(baris, start=1):
                st.markdown(f"**Komentar {no}**")
                hasil_t = tahapan_preprocessing(b)
                st.dataframe(
                    pd.DataFrame({"Tahap": list(hasil_t.keys()), "Hasil": list(hasil_t.values())}),
                    width="stretch", hide_index=True,
                    column_config={"Tahap": st.column_config.TextColumn(width="medium"),
                                   "Hasil": st.column_config.TextColumn(width="large")},
                )

# ---------------- Riwayat & Excel
with tab_riwayat:
    st.subheader("Riwayat Hasil Analisis")
    if st.session_state.history:
        hdf = pd.DataFrame(st.session_state.history)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Komentar", len(hdf))
        c2.metric("Positif", int((hdf["Prediksi Sentimen"] == "Positif").sum()))
        c3.metric("Netral", int((hdf["Prediksi Sentimen"] == "Netral").sum()))
        c4.metric("Negatif", int((hdf["Prediksi Sentimen"] == "Negatif").sum()))
        st.dataframe(hdf, width="stretch", hide_index=True, column_config=KOLOM_CFG)
        st.download_button(
            "Unduh Hasil ke Excel", data=to_excel_bytes(hdf),
            file_name="HASIL_ANALISIS_KOMENTAR_BLIBLI.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
        )
        if st.button("Hapus Riwayat Analisis"):
            st.session_state.history = []
            st.rerun()
        st.caption("Riwayat hanya tersimpan selama sesi berjalan dan hilang saat halaman dimuat ulang atau ditutup.")
    else:
        st.info("Belum ada komentar yang dianalisis. Masukkan komentar pada menu Analisis Komentar.")

# ---------------- Visualisasi
with tab_visual:
    st.subheader("Visualisasi Hasil Sentimen")
    st.markdown("**1. Komentar yang dianalisis pada sesi ini**")
    if st.session_state.history:
        hdf = pd.DataFrame(st.session_state.history)
        hitung = hdf["Prediksi Sentimen"].value_counts().to_dict()
        kol, _ = st.columns(2)
        tampil_grafik(kol, grafik_batang(hitung, "Distribusi Sentimen Komentar yang Dianalisis"))
    else:
        st.info("Belum ada komentar yang dianalisis. Grafik akan muncul setelah kamu menganalisis komentar.")

    hasil = load_hasil()
    st.markdown("**2. Hasil penelitian pada 1.060 ulasan**")
    if hasil is None:
        st.warning("File 'hasil_penelitian.json' tidak ditemukan, sehingga hasil penelitian tidak dapat ditampilkan.")
    else:
        kiri, kanan = st.columns(2)
        with kiri:
            tampil_grafik(kiri, grafik_batang(hasil["distribusi"], "Distribusi Sentimen 1.060 Ulasan"))
            st.caption("Gabungan 600 label manual dan 460 prediksi algoritma.")
        with kanan:
            adf = pd.DataFrame(hasil["aspek"])
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            x = np.arange(len(adf)); w = 0.26
            ax.bar(x - w, adf["positif"], w, label="Positif", color=WARNA["Positif"])
            ax.bar(x, adf["netral"], w, label="Netral", color=WARNA["Netral"])
            ax.bar(x + w, adf["negatif"], w, label="Negatif", color=WARNA["Negatif"])
            ax.set_xticks(x); ax.set_xticklabels(adf["aspek"], rotation=15, fontsize=8)
            ax.set_ylabel("Persentase (%)"); ax.legend(fontsize=8)
            ax.set_title("Distribusi Sentimen per Aspek Layanan", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            tampil_grafik(kanan, fig)
        st.markdown("**Sepuluh kata yang paling sering muncul**")
        kk = hasil["kata"]
        st.dataframe(pd.DataFrame({
            "Kata Positif": [k for k, _ in kk["Positif"]], "Frekuensi Positif": [v for _, v in kk["Positif"]],
            "Kata Netral": [k for k, _ in kk["Netral"]], "Frekuensi Netral": [v for _, v in kk["Netral"]],
            "Kata Negatif": [k for k, _ in kk["Negatif"]], "Frekuensi Negatif": [v for _, v in kk["Negatif"]],
        }), width="stretch", hide_index=True)

# ---------------- Evaluasi Algoritma
with tab_eval:
    st.subheader("Evaluasi Algoritma pada Data Uji")
    st.caption("Angka di bawah berasal dari pengujian terpisah pada 120 ulasan berlabel manual. "
               "Akurasi ini bukan jaminan bahwa setiap komentar baru akan diprediksi benar.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Akurasi", "77,50%")
    c2.metric("Macro F1-score", "66,57%")
    c3.metric("Akurasi Nested 10-Fold CV", "72,50% ± 4,03%")
    st.dataframe(pd.DataFrame([
        {"Kelas": "Positif", "Precision": "87,50%", "Recall": "94,23%", "F1-score": "90,74%", "Data uji": 52},
        {"Kelas": "Netral", "Precision": "45,45%", "Recall": "20,83%", "F1-score": "28,57%", "Data uji": 24},
        {"Kelas": "Negatif", "Precision": "73,58%", "Recall": "88,64%", "F1-score": "80,41%", "Data uji": 44},
    ]), width="stretch", hide_index=True)
    st.write("**Confusion matrix**")
    st.dataframe(pd.DataFrame(
        [[49, 2, 1], [6, 5, 13], [1, 4, 39]],
        index=["Aktual Positif", "Aktual Netral", "Aktual Negatif"],
        columns=["Prediksi Positif", "Prediksi Netral", "Prediksi Negatif"]), width="stretch")
    st.warning("Kelas netral masih menjadi kelemahan utama algoritma. Dari 24 ulasan netral pada data uji, "
               "hanya 5 yang berhasil dikenali dengan benar.")

    st.markdown("---")
    st.subheader("Sistem Lengkap: Naive Bayes + Aturan Kelas Netral")
    st.caption("Aplikasi memakai algoritma Naive Bayes yang dilengkapi aturan pendukung (klasifikasi.py). "
               "Aturan memakai kamus kata sederhana dan tidak dilatih dari data, sehingga model .pkl tidak berubah.")
    e1, e2 = st.columns(2)
    e1.metric("Akurasi data uji (120 ulasan)", "69,17%")
    e2.metric("Macro F1-score data uji", "60,33%")
    st.dataframe(pd.DataFrame([
        {"Kelas": "Positif", "Precision": "81,82%", "Recall": "86,54%", "F1-score": "84,11%", "Data uji": 52},
        {"Kelas": "Netral", "Precision": "25,00%", "Recall": "20,83%", "F1-score": "22,73%", "Data uji": 24},
        {"Kelas": "Negatif", "Precision": "73,33%", "Recall": "75,00%", "F1-score": "74,16%", "Data uji": 44},
    ]), width="stretch", hide_index=True)
    st.dataframe(pd.DataFrame(
        [[45, 7, 0], [7, 5, 12], [3, 8, 33]],
        index=["Aktual Positif", "Aktual Netral", "Aktual Negatif"],
        columns=["Prediksi Positif", "Prediksi Netral", "Prediksi Negatif"]), width="stretch")
    st.info("Pada 10-fold cross-validation (600 ulasan berlabel, parameter tetap), algoritma saja menghasilkan akurasi 73,3% "
            "dan macro F1 63,8%; dengan aturan netral menjadi akurasi 71,8% dan macro F1 64,7%. "
            "Label netral pada data manual belum sepenuhnya konsisten, sehingga angka pada data berlabel tidak "
            "sepenuhnya mencerminkan ketepatan pada kalimat baru.")
