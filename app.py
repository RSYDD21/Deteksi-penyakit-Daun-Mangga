import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# =========================
# Load Model dari Google Drive
# =========================
@st.cache_resource
def load_model():
    # Google Drive file id
    file_id = "1bOh9d5IA94IqzdAyb3kGMZqJ9KCYTCh6"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Simpan sementara
    output = "model.h5"

    # Download jika belum ada
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # Load model
    model = tf.keras.models.load_model(output)
    return model

model = load_model()

# =========================
# UI Streamlit Dasar
# =========================
st.title("🍃 Deteksi Penyakit Daun Mangga")

st.write("""
Upload gambar daun mangga untuk dideteksi penyakitnya.
""")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence  = np.max(prediction)

    st.markdown(f"**Prediksi Kelas:** {class_idx}")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
