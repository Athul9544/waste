# Create a streamlit app file
app_code = ""
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model('/content/drive/MyDrive/sebrs_model.h5')

def classify_waste(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize
    predictions = model.predict(img)
    class_labels = ['Recyclable', 'Hazardous', 'Non-Recyclable']
    return class_labels[np.argmax(predictions)]

st.title("SEBRS - Smart E-Bio Recycler System")

uploaded_file = st.file_uploader( type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    category = classify_waste(image_path)
    st.image(uploaded_file, caption=f"Classified as: {category}")


with open("app.py", "w") as f:
    f.write(app_code)
