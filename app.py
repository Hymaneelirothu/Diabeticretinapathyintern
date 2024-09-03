import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# Function to download the model from Google Drive
def download_file_from_google_drive(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        raise Exception("Failed to download the file")

# Google Drive file download link
google_drive_link = 'https://drive.google.com/uc?id=1xWPWU4YaqTQlFVqcvfM7XL-oigWG7KUX&export=download'

# Path to save the downloaded model
model_path = 'best_model.keras'

# Download the file
if not os.path.exists(model_path):  # Only download if the file doesn't exist
    download_file_from_google_drive(google_drive_link, model_path)

# Load the model
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("Diabetic Retinopathy Detection App")
st.write("This app predicts the level of diabetic retinopathy from retinal images.")

# Upload an image
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for both models
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    # Duplicate the input for both EfficientNet and ConvNeXt (since both expect the same input)
    model_inputs = [processed_image, processed_image]

    # Make prediction
    prediction = model.predict(model_inputs)
    prediction = np.squeeze(prediction)  # Remove batch dimension

    # Interpret the results
    labels = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']
    st.write("Prediction results:")
    for label, prob in zip(labels, prediction):
        st.write(f"{label}: {prob:.2%}")

    # Display the highest probability
    predicted_label = labels[np.argmax(prediction)]
    st.subheader(f"Predicted: {predicted_label}")
