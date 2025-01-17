import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown

# Function to download the model from Google Drive
@st.cache_resource
def download_and_load_model():
    url = 'https://drive.google.com/uc?id=1LKNyZ1Q-HUUysHQnog1d_7Znt8_vzfBC'
    output = 'agricultural_pests_model.h5'
    gdown.download(url, output, quiet=False)
    model = load_model(output)
    return model

model = download_and_load_model()

def preprocess_image(image):
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

st.title("Beans Pest Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predictions = predict_image(image, model)

    # Classes based on the provided image
    classes = ['weevil', 'wasp', 'snail', 'slug', 'moth', 'grasshopper', 'earwig', 'earthworms', 'caterpillar', 'beetle', 'bees', 'ants']
    predicted_class = classes[np.argmax(predictions)]
    st.write(f'Prediction: Pest Detected')
