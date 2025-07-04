# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os

# Let's create a User interface
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

# Error handling for model loading
try:
    # Check if model files exist
    if not os.path.exists("pollution_model.pkl") or not os.path.exists("model_columns.pkl"):
        st.error("Model files not found! Please ensure 'pollution_model.pkl' and 'model_columns.pkl' are in the correct directory.")
        st.stop()
    
    # Load the model and structure
    model = joblib.load("pollution_model.pkl")
    model_cols = joblib.load("model_columns.pkl")
    
    # Validate model columns
    if not isinstance(model_cols, list):
        st.error("Invalid model columns format. Expected a list of column names.")
        st.stop()

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# Prediction function
def predict_pollutants(year, station_id):
    try:
        # Prepare the input
        input_df = pd.DataFrame({'year': [year], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model cols
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        
        return dict(zip(pollutants, predicted_pollutants))
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# To encode and then predict
if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        with st.spinner('Making prediction...'):
            predicted_values = predict_pollutants(year_input, station_id)
            
            if predicted_values is not None:
                st.subheader(f"Predicted pollutant levels for station '{station_id}' in {year_input}:")
                for pollutant, value in predicted_values.items():
                    st.metric(label=pollutant, value=f"{value:.2f}")