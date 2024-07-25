import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown

@st.cache_resource
def download_model():
    url = 'https://drive.google.com/uc?id=1LKNyZ1Q-HUUysHQnog1d_7Znt8_vzfBC'
    output = 'agricultural_pests_model.h5'
    gdown.download(url, output, quiet=False)
    return load_model(output)

model = download_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image

def predict_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.vgg19.decode_predictions(prediction, top=3)[0]
    return decoded_predictions

st.title("Image Classification with VGG19 Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predictions = predict_image(image, model)

    st.write("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i+1}: {label} ({score*100:.2f}%)")
