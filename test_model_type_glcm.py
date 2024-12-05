import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# Load your trained model
model = tf.keras.models.load_model('D:/capstone_models/skin_type_model.h5')

def extract_glcm_features(image):
    try:
        # Convert image to grayscale
        gray_image = image.convert('L')  # Convert to grayscale
        gray_array = np.array(gray_image)

        # Compute GLCM
        glcm = graycomatrix(gray_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extract GLCM properties
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # Return as a NumPy array
        return np.array([contrast, dissimilarity, homogeneity, energy, correlation], dtype=np.float32)
    except Exception as e:
        st.error(f"Error calculating GLCM features: {e}")
        return None


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

def predict_image(model, img_array, glcm_features):
    if img_array is not None and glcm_features is not None:
        predictions = model.predict([img_array, np.expand_dims(glcm_features, axis=0)])  # Add batch dimension to GLCM
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
        st.write("Classifying...")

        # Preprocess image for model input
        img_array = load_and_preprocess_image(image)

        # Extract GLCM features
        glcm_features = extract_glcm_features(image)

        # Predict using the model
        if img_array is not None and glcm_features is not None:
            predictions = predict_image(model, img_array, glcm_features)

            # Display predictions
            st.write(f"Predictions: Oily: {predictions[0][0]:.2f}, Normal: {predictions[0][1]:.2f}, Dry: {predictions[0][2]:.2f}")
        else:
            st.write("Failed to process image or calculate GLCM features.")
    except Exception as e:
        st.error(f"Error processing input: {e}")