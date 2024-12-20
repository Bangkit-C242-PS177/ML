import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
models_path = os.getenv('MODELS_PATH')

# Load your trained model
model = tf.keras.models.load_model(models_path + 'skin_type_model.keras')

def load_and_preprocess_image(img):
    try:
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize((224, 224))  # Resize to the input shape expected by the model
        img_array = img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess the image
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_image(model, img_array):
    if img_array is not None:
        predictions = model.predict(img_array)
        return predictions
    else:
        return None

st.title('Skin Type Classification')
st.header('Upload an image to classify skin type')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        img_array = load_and_preprocess_image(image)
        predictions = predict_image(model, img_array)
        
        if predictions is not None:
            st.write(predictions)
            st.write(f"Predictions: Oily: {predictions[0][0]:.2f}, Normal: {predictions[0][1]:.2f}, Dry: {predictions[0][2]:.2f}")
        else:
            st.write("Prediction failed. Please try uploading a different image.")
    except Exception as e:
        st.error(f"Error loading image: {e}")
