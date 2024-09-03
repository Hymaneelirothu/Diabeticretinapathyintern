import os
import gdown
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Google Drive file ID and model path
file_id = '1xWPWU4YaqTQlFVqcvfM7XL-oigWG7KUX'
model_path = 'best_model.keras'

# Google Drive download URL
download_url = f'https://drive.google.com/uc?id={file_id}'

# Download the model from Google Drive if it doesn't exist locally
if not os.path.exists(model_path):
    gdown.download(download_url, model_path, quiet=False)

# Check if the file was downloaded correctly
if not os.path.exists(model_path):
    st.error(f"Failed to download the model from Google Drive: {model_path}")
else:
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

        # Preprocess the image for the model
        processed_image = preprocess_image(image, target_size=(224, 224))

        # Make prediction
        prediction = model.predict(processed_image)
        prediction = np.squeeze(prediction)  # Remove batch dimension

        # Interpret the results
        labels = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']
        st.write("Prediction results:")
        for label, prob in zip(labels, prediction):
            st.write(f"{label}: {prob:.2%}")

        # Display the highest probability
        predicted_label = labels[np.argmax(prediction)]
        st.subheader(f"Predicted: {predicted_label}")
