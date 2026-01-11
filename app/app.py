import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# -------------------
# Config
# -------------------
MODEL_PATH = "model/location_model.keras"
CLASS_PATH = "model/classes.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Location Classifier", page_icon="🌍")

# -------------------
# Load model + labels
# -------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, "r") as f:
        class_map = json.load(f)
    labels = {v: k for k, v in class_map.items()}
    return model, labels

model, labels = load_model_and_labels()

# -------------------
# UI
# -------------------
st.title("🌍 Location Image Classifier")
st.write("Upload an image and the model will predict the type of location.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.success(f"Prediction: **{labels[class_id]}**  \nConfidence: **{confidence:.2f}**")
