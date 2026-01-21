import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷")

# Title
st.title("🍷 Wine Cultivar Origin Prediction")
st.write("Enter the chemical properties of the wine to predict its cultivar.")

# --- Load Model with Error Handling (Feedback: Fail Gracefully) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/wine_cultivar_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: The model file 'wine_cultivar_model.pkl' was not found. Please check your directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading the model: {e}")
        st.stop()

pipeline = load_model()

# --- User Inputs ---
st.sidebar.header("Wine Chemical Properties")

def user_input_features():
    # These ranges are based on the Wine dataset statistics
    alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 13.0)
    malic_acid = st.sidebar.slider('Malic Acid', 0.7, 6.0, 2.0)
    total_phenols = st.sidebar.slider('Total Phenols', 0.9, 4.0, 2.5)
    flavanoids = st.sidebar.slider('Flavanoids', 0.3, 5.1, 2.0)
    color_intensity = st.sidebar.slider('Color Intensity', 1.3, 13.0, 5.0)
    hue = st.sidebar.slider('Hue', 0.5, 1.8, 1.0)

    data = {
        'alcohol': alcohol,
        'malic_acid': malic_acid,
        'total_phenols': total_phenols,
        'flavanoids': flavanoids,
        'color_intensity': color_intensity,
        'hue': hue
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display User Input
st.subheader("Your Input Parameters")
st.write(input_df)

# --- Prediction with Error Handling ---
if st.button("Predict Cultivar"):
    try:
        # The Pipeline handles the scaling automatically!
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)

        # Map prediction to readable name
        cultivar_map = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
        result = cultivar_map.get(prediction[0], "Unknown")

        st.success(f"Predicted Origin: **{result}**")
        
        # Optional: Show confidence
        st.subheader("Prediction Confidence")
        st.bar_chart(prediction_proba[0])

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")