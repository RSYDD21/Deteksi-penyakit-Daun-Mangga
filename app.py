import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")   # atau model.keras
    return model

model = load_model()

st.title("🍃 Mango Leaf Disease Detection")
st.write("Model EfficientNetB7 — Klasifikasi penyakit daun mangga")

uploaded = st.file_uploader("Upload gambar daun mangga", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar yang diupload", width=300)

    # Preprocessing
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    class_index = np.argmax(pred)

    st.success(f"Prediksi: Kelas {class_index}")