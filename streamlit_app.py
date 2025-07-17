import streamlit as st
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from PIL import Image
from utils.predict import preprocess_and_predict
from tensorflow.keras.models import load_model

# Load model and label encoder
model = load_model("model/plant_disease_cnn_model_final.h5")
labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌿 Plant Disease Classification")
st.write("Upload a plant leaf image and get the predicted disease label.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        image = Image.open(uploaded_file)
        pred_label, probabilities = preprocess_and_predict(image, model, labels)
        st.success(f"🩺 Predicted Class: **{pred_label}**")

        st.subheader("🔢 Prediction Probabilities:")
        for label, prob in zip(labels, probabilities[0]):
            st.write(f"- **{label}**: {prob:.4f}")

        st.subheader("📊 Probability Distribution:")
        df = pd.DataFrame({
            "Class": labels,
            "Probability": probabilities[0].tolist()
        })
        st.bar_chart(df.set_index("Class"))



