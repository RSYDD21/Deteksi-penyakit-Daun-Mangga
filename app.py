import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import os

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

IMG_SIZE = (224, 224)

# =========================
# Load Model TFLite
# =========================
@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# =========================
# UI
# =========================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Mangga",
    page_icon="🍃"
)

st.title("🍃 Deteksi Penyakit Daun Mangga")
st.write("Upload gambar daun mangga untuk mendeteksi jenis penyakit.")

file = st.file_uploader("Upload gambar daun mangga", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_container_width=True)

    # Preprocessing
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])

    idx = int(np.argmax(preds))
    conf = float(np.max(preds)) * 100

    st.success(f"🦠 Penyakit: **{labels[idx]}**")
    st.info(f"📊 Confidence: **{conf:.2f}%**")
