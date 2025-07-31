import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load model and scaler
model = load_model('wine_model.keras')
scaler = joblib.load('scaler.pkl')

# App title
st.title('🍷 Wine Quality Prediction')
st.write('Enter the features of the wine to predict its quality.')

# Feature statistics based on your dataset
feature_stats = [
    'alcohol',
    'malic_acid',
    'ash',
    'alcalinity_of_ash',
    'magnesium',
    'total_phenols',
    'flavanoids',
    'nonflavanoid_phenols',
    'proanthocyanins',
    'color_intensity',
    'hue',
    'od280/od315_of_diluted_wines',
    'proline'
]

# Create input fields dynamically
inputs = [
    st.number_input("Alcohol", min_value=11.0, max_value=15.0, value=13.0, step=0.1, key="input_alcohol"),
    st.number_input("Malic Acid", min_value=0.7, max_value=5.8, value=2.3, step=0.1, key="input_malic_acid"),
    st.number_input("Ash", min_value=1.3, max_value=3.3, value=2.4, step=0.1, key="input_ash"),
    st.number_input("Alcalinity of Ash", min_value=10.0, max_value=30.0, value=19.5, step=0.1, key="input_alcalinity_of_ash"),
    st.number_input("Magnesium", min_value=70.0, max_value=165.0, value=99.0, step=1.0, key="input_magnesium"),
    st.number_input("Total Phenols", min_value=0.9, max_value=3.9, value=2.3, step=0.1, key="input_total_phenols"),
    st.number_input("Flavanoids", min_value=0.3, max_value=5.0, value=2.0, step=0.1, key="input_flavanoids"),
    st.number_input("Nonflavanoid Phenols", min_value=0.1, max_value=1.0, value=0.3, step=0.05, key="input_nonflavanoid_phenols"),
    st.number_input("Proanthocyanins", min_value=0.3, max_value=4.0, value=1.6, step=0.1, key="input_proanthocyanins"),
    st.number_input("Color Intensity", min_value=1.0, max_value=13.0, value=5.0, step=0.1, key="input_color_intensity"),
    st.number_input("Hue", min_value=0.4, max_value=1.7, value=1.0, step=0.05, key="input_hue"),
    st.number_input("od280/od315_of_diluted_wines", min_value=1.2, max_value=4.0, value=3.0, step=0.1, key="input_od280/od315_of_diluted_wines"),
    st.number_input("Proline", min_value=270.0, max_value=1700.0, value=746.0, step=10.0, key="input_proline")
]

# Prediction button
if st.button('Predict'):
    # Prepare input as DataFrame
    input_df = pd.DataFrame([inputs], columns=feature_stats)

    # Scale
    scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled)

    # Show result
    st.success(f'🍇 Predicted Wine Quality: **{np.argmax(prediction) + 1}**')