import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import gdown

# Function to download the model from Google Drive
@st.cache(allow_output_mutation=True)
def load_model():
    url = 'https://drive.google.com/uc?id=1whdEzMezUmvfBA8UtsQ4eeao2VPbBHc9'
    output = 'best_vgg19.h5'
    gdown.download(url, output, quiet=False)
    model = tf.keras.models.load_model(output)
    return model

# Load the model
model = load_model()

# Streamlit app
st.title("Image Classification with VGG19 Model")
st.write("Upload an image to classify")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.vgg19.decode_predictions(predictions, top=3)[0]

    st.write("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}: {label} ({score*100:.2f}%)")
