import streamlit as st
import joblib
import pandas as pd
import numpy as np
from keras.models import load_model

# Load the pre-trained model and scaler
model = load_model('wine_model.keras', compile=False)
scaler = joblib.load('scaler.pkl')

# App title
st.title('🍷 Wine Quality Prediction')
st.write('Enter the features of the wine to predict its quality.')

# Feature names
feature_names = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# Create input fields with unique keys
inputs = [
    st.number_input(
        f"{feature}", min_value=0.0, step=0.1, key=f"input_{feature}"
    )
    for feature in feature_names
]

# Prediction button
if st.button('Predict'):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([inputs], columns=feature_names)

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display the prediction
    st.success(f'🍇 Predicted Wine Quality: **{np.argmax(prediction) + 1}**')  # Assuming quality is 1-10
