import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

MODEL_PATH = "flower_cnn_model2.keras"
IMG_SIZE = 180

CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

FLOWER_INFO = {
    "daisy": {
        "name": "Daisy",
        "desc": (
            "Daisy merupakan bunga dengan kelopak putih dan bagian tengah berwarna kuning. "
            "Bunga ini memiliki bentuk sederhana dan sering ditemukan di taman atau padang rumput."
        ),
    },
    "dandelion": {
        "name": "Dandelion",
        "desc": (
            "Dandelion merupakan bunga kecil yang umumnya berwarna kuning. "
            "Bunga ini memiliki kelopak halus dan sering tumbuh liar di area terbuka."
        ),
    },
    "rose": {
        "name": "Rose",
        "desc": (
            "Rose atau mawar memiliki kelopak berlapis dan rapat. "
            "Bunga ini dikenal karena bentuknya yang indah serta memiliki banyak variasi warna."
        ),
    },
    "sunflower": {
        "name": "Sunflower",
        "desc": (
            "Sunflower atau bunga matahari memiliki kelopak besar berwarna kuning dan bagian tengah yang gelap. "
            "Bunga ini memiliki bentuk yang sangat khas."
        ),
    },
    "tulip": {
        "name": "Tulip",
        "desc": (
            "Tulip memiliki bentuk kelopak seperti cangkir dengan warna yang beragam. "
            "Bunga ini mudah dikenali dari bentuk kelopaknya yang tegak dan rapi."
        ),
    },
}

st.set_page_config(
    page_title="Klasifikasi Jenis Bunga CNN",
    layout="wide",
)

st.markdown(
    """
<style>
html {
    scroll-behavior: smooth;
}

.stApp {
    background: #f6faf4;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.nav-box {
    background: white;
    padding: 16px;
    border-radius: 18px;
    border: 1px solid #dfe8dc;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}

.nav-link {
    display: block;
    text-decoration: none !important;
    color: #163d2a !important;
    font-weight: 700;
    padding: 10px 12px;
    border-radius: 12px;
    margin-bottom: 8px;
    background: #f4faf2;
    border: 1px solid #e4eee1;
}

.nav-link:hover {
    background: #dfeedd;
}

.hero {
    background: linear-gradient(135deg, #123d28, #2f734b);
    color: white;
    padding: 56px 50px;
    border-radius: 30px;
    box-shadow: 0 20px 50px rgba(18,61,40,0.28);
    margin-bottom: 28px;
}

.hero-title {
    font-size: 46px;
    font-weight: 900;
    line-height: 1.15;
    margin-bottom: 14px;
}

.hero-subtitle {
    font-size: 18px;
    line-height: 1.75;
    color: #e9f7ec;
    max-width: 900px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    padding: 9px 15px;
    border-radius: 999px;
    font-weight: 700;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.25);
}

.section {
    margin-top: 32px;
    margin-bottom: 28px;
}

.section-title {
    font-size: 32px;
    font-weight: 900;
    color: #163d2a;
    margin-bottom: 10px;
}

.section-subtitle {
    font-size: 16px;
    color: #5e6f63;
    line-height: 1.7;
    margin-bottom: 22px;
}

.card {
    background: white;
    padding: 26px;
    border-radius: 24px;
    border: 1px solid #dfe8dc;
    box-shadow: 0 10px 28px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

.small-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    border: 1px solid #dfe8dc;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    height: 100%;
}

.card-title {
    font-size: 22px;
    font-weight: 850;
    color: #163d2a;
    margin-bottom: 10px;
}

.text {
    font-size: 16px;
    color: #465a4c;
    line-height: 1.75;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 24px;
}

.metric-card {
    background: rgba(255,255,255,0.95);
    padding: 22px;
    border-radius: 22px;
    text-align: center;
    border: 1px solid #dfe8dc;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

.metric-value {
    font-size: 32px;
    font-weight: 900;
    color: #163d2a;
}

.metric-label {
    font-size: 14px;
    color: #68786c;
    margin-top: 6px;
}

.upload-box {
    background: white;
    padding: 28px;
    border-radius: 26px;
    border: 1px solid #dfe8dc;
    box-shadow: 0 12px 34px rgba(0,0,0,0.08);
}

.result-panel {
    background: linear-gradient(135deg, #123d28, #2f734b);
    color: white;
    padding: 34px;
    border-radius: 26px;
    box-shadow: 0 16px 42px rgba(18,61,40,0.28);
    text-align: center;
}

.result-label {
    font-size: 16px;
    opacity: 0.9;
}

.result-class {
    font-size: 48px;
    font-weight: 900;
    margin-top: 6px;
    text-transform: capitalize;
}

.result-confidence {
    font-size: 30px;
    font-weight: 850;
    margin-top: 6px;
}

.prob-item {
    background: #f8fbf7;
    padding: 16px 18px;
    border-radius: 18px;
    margin-bottom: 15px;
    border: 1px solid #e2ede0;
}

.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.prob-name {
    font-size: 17px;
    font-weight: 850;
    color: #163d2a;
}

.prob-rank {
    font-size: 12px;
    color: #68786c;
    margin-top: 3px;
}

.prob-percent {
    font-size: 20px;
    font-weight: 900;
    color: #163d2a;
}

.prob-track {
    width: 100%;
    height: 14px;
    background: #dfe8dc;
    border-radius: 999px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    background: linear-gradient(90deg, #143d2a, #45a35d);
    border-radius: 999px;
}

.prediction-note {
    background: #f3faf1;
    border-left: 5px solid #2f734b;
    padding: 18px;
    border-radius: 16px;
    color: #3c5142;
    line-height: 1.7;
}

.footer {
    background: #123d28;
    color: #e9f7ec;
    padding: 26px;
    border-radius: 24px;
    text-align: center;
    margin-top: 35px;
}

div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    background: #163d2a;
    color: white;
    font-weight: 850;
    font-size: 17px;
    border: none;
}

div.stButton > button:hover {
    background: #2f734b;
    color: white;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 34px;
    }

    .metric-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.array(resized) / 255.0
    array = np.expand_dims(array, axis=0)
    return image, array


def predict_image(model, image_array):
    prediction = model.predict(image_array)
    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction) * 100)
    return prediction[0], predicted_class, confidence


def probability_bar(label, value, rank):
    badge = "Prediksi Utama" if rank == 1 else f"Rank {rank}"

    return f"""
    <div class="prob-item">
        <div class="prob-header">
            <div>
                <div class="prob-name">{label}</div>
                <div class="prob-rank">{badge}</div>
            </div>
            <div class="prob-percent">{value:.2f}%</div>
        </div>
        <div class="prob-track">
            <div class="prob-fill" style="width: {value}%;"></div>
        </div>
    </div>
    """


if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.markdown('<div class="nav-box">', unsafe_allow_html=True)
    st.markdown("### Navigasi")
    st.markdown('<a class="nav-link" href="#home">Home</a>', unsafe_allow_html=True)
    st.markdown(
        '<a class="nav-link" href="#about">Tentang Website</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a class="nav-link" href="#classes">Jenis Bunga</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a class="nav-link" href="#classification">Klasifikasi</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a class="nav-link" href="#developer">Pembuat</a>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div id="home"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
    <div class="hero-tag">Computer Vision Project</div>
    <div class="hero-title">Klasifikasi Jenis Bunga Menggunakan Convolutional Neural Network</div>
    <div class="hero-subtitle">
        Website ini digunakan untuk mengklasifikasikan gambar bunga secara otomatis menggunakan model CNN.
        Pengguna dapat mengunggah gambar bunga, kemudian sistem akan menampilkan hasil prediksi,
        confidence, dan persentase probabilitas dari setiap kelas bunga.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">5</div>
        <div class="metric-label">Kelas Bunga</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">CNN</div>
        <div class="metric-label">Metode Model</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">180</div>
        <div class="metric-label">Ukuran Input</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">Web</div>
        <div class="metric-label">Implementasi</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tentang Website</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section-subtitle">
Website ini dibuat sebagai implementasi proyek Computer Vision untuk mengenali jenis bunga berdasarkan citra digital.
Sistem menggunakan model Convolutional Neural Network yang telah dilatih menggunakan dataset Flowers Recognition dari Kaggle.
</div>
""",
    unsafe_allow_html=True,
)

about_col1, about_col2 = st.columns([1, 1])

with about_col1:
    st.markdown(
        """
    <div class="card">
        <div class="card-title">Cara Kerja Sistem</div>
        <div class="text">
            Gambar bunga yang diunggah akan diubah ukurannya menjadi 150 x 150 piksel.
            Setelah itu nilai piksel dinormalisasi agar dapat diproses oleh model CNN.
            Model kemudian menghasilkan probabilitas untuk setiap kelas bunga dan memilih kelas
            dengan nilai probabilitas tertinggi sebagai hasil prediksi.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with about_col2:
    st.markdown(
        """
    <div class="card">
        <div class="card-title">Alur Proses</div>
        <div class="text">
            Dataset Bunga → Preprocessing → Training CNN → Model Keras → Upload Gambar → Prediksi → Hasil Klasifikasi
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div id="classes"></div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Jenis Bunga yang Diklasifikasikan</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-subtitle">
Model CNN pada website ini dapat mengenali lima jenis bunga. Setiap kelas memiliki karakteristik visual yang berbeda.
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)

for col, key in zip([c1, c2, c3, c4, c5], CLASS_NAMES):
    with col:
        st.markdown(
            f"""
        <div class="small-card">
            <div class="card-title">{FLOWER_INFO[key]["name"]}</div>
            <div class="text">{FLOWER_INFO[key]["desc"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div id="classification"></div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Klasifikasi Gambar</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section-subtitle">
Unggah gambar bunga dengan format JPG, JPEG, atau PNG. Setelah gambar diunggah, klik tombol klasifikasi untuk melihat hasil prediksi model.
</div>
""",
    unsafe_allow_html=True,
)

try:
    model = load_cnn_model()
except Exception as e:
    st.error(
        "Model gagal dimuat. Pastikan file flower_cnn_model.keras berada satu folder dengan app.py."
    )
    st.code(str(e))
    st.stop()

upload_col, preview_col = st.columns([0.9, 1.1])

with upload_col:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload gambar bunga",
        type=["jpg", "jpeg", "png", "webp"],
    )

    predict_button = st.button("Proses Klasifikasi")
    st.markdown("</div>", unsafe_allow_html=True)

with preview_col:
    if uploaded_file is not None:
        original_image, image_array = preprocess_image(uploaded_file)
        st.image(
            original_image,
            caption="Preview gambar yang diunggah",
            use_container_width=True,
        )
    else:
        st.info("Belum ada gambar yang diunggah.")

if uploaded_file is not None and predict_button:
    with st.spinner("Model sedang melakukan klasifikasi gambar..."):
        probabilities, predicted_class, confidence = predict_image(model, image_array)

    result_col, prob_col = st.columns([0.9, 1.1])

    with result_col:
        st.markdown(
            f"""
        <div class="result-panel">
            <div class="result-label">Hasil Prediksi</div>
            <div class="result-class">{FLOWER_INFO[predicted_class]["name"]}</div>
            <div class="result-confidence">{confidence:.2f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">Penjelasan Hasil</div>
            <div class="prediction-note">
                Model memprediksi gambar sebagai <b>{FLOWER_INFO[predicted_class]["name"]}</b>
                dengan tingkat keyakinan sebesar <b>{confidence:.2f}%</b>.
                {FLOWER_INFO[predicted_class]["desc"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with prob_col:
            sorted_indices = np.argsort(probabilities)[::-1]
            prob_html = """
            <div class="card">
        <div class="card-title">Probabilitas Setiap Kelas</div>
        <div class="text" style="margin-bottom:18px;">
            Berikut adalah persentase keyakinan model terhadap masing-masing jenis bunga,
            diurutkan dari probabilitas tertinggi ke terendah.
        </div>
    """
    

    for rank, idx in enumerate(sorted_indices, start=1):
        class_key = CLASS_NAMES[idx]
        prob_html += probability_bar(
            FLOWER_INFO[class_key]["name"],
            float(probabilities[idx] * 100),
            rank,
        )

    prob_html += "</div>"

    st.markdown(prob_html, unsafe_allow_html=True)


    

    st.session_state.history.append(
        {
            "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Nama File": uploaded_file.name,
            "Prediksi": FLOWER_INFO[predicted_class]["name"],
            "Confidence": f"{confidence:.2f}%",
        }
    )

if len(st.session_state.history) > 0:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Riwayat Prediksi</div>', unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Riwayat Prediksi",
        data=csv,
        file_name="riwayat_prediksi_bunga.csv",
        mime="text/csv",
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div id="developer"></div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Pembuat Website</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="card">
    <div class="text">
        <b>Nama:</b> Gerry Almidi<br>
        <b>NIM:</b> 2255301065<br>
        <b>Program Studi:</b> Teknik Informatika<br>
        <b>Institusi:</b> Politeknik Caltex Riau<br>
        <b>Judul Project:</b> Klasifikasi Jenis Bunga Menggunakan Convolutional Neural Network<br>
        <b>Platform:</b> Streamlit Community Cloud
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="footer">
    Klasifikasi Jenis Bunga Menggunakan CNN | Gerry Almidi | Politeknik Caltex Riau | 2026
</div>
""",
    unsafe_allow_html=True,
)
