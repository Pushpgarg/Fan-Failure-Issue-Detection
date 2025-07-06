import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========== Configs ==========
failure_cols = [
    "Overheating", "Misalignment", "Bearing_Failure", "Imbalance", "Electrical_Issue"
]

# Load models and scalers
scalers = {}
models = {}

for failure in failure_cols:
    label = failure.replace(" ", "_")
    scalers[failure] = joblib.load(f"scalers/scaler_{label}.pkl")
    models[failure] = joblib.load(f"trained_models/model_{label}.pkl")

# ========== Input UI ==========
st.title("🛠️ Equipment Failure Predictor")
st.markdown("Enter the values below to predict possible failures.")

# Categorical options
equipment_types = ['Axial', 'Exhaust']
locations = ['Plant A', 'Plant B', 'Plant C']

data_input = {}

# Binary categorical inputs
st.subheader("🔘 Categorical Inputs")
data_input['Equipment Type'] = st.selectbox("Equipment Type", equipment_types)
data_input['Location'] = st.selectbox("Location", locations)

# Numeric inputs
st.subheader("📈 Sensor & Operational Data")
data_input['Temperature (°C)'] = st.number_input("Temperature (°C)", value=60.0)
data_input['Pressure (bar)'] = st.number_input("Pressure (bar)", value=1.5)
data_input['Vibration_X'] = st.number_input("Vibration_X", value=0.4)
data_input['Vibration_Y'] = st.number_input("Vibration_Y", value=0.4)
data_input['Vibration_Z'] = st.number_input("Vibration_Z", value=0.4)
data_input['Humidity (%)'] = st.number_input("Humidity (%)", value=5.0)
data_input['RPM'] = st.number_input("RPM", value=70.0)
data_input['Sound (dB)'] = st.number_input("Sound (dB)", value=65.0)
data_input['Current (A)'] = st.number_input("Current (A)", value=5.0)

# ========== Prediction ========== 
if st.button("🔍 Predict Failures"):
    input_df = pd.DataFrame([data_input])

    results = {}

    for failure in failure_cols:
        label = failure.replace(" ", "_")

        # Load X_test sample to get correct column structure (assumes same preprocessing order)
        sample_X = pd.read_csv(f"split_data/{label}/X_train.csv")
        cat_cols = ['Equipment Type', 'Location']
        num_cols = [col for col in sample_X.columns if col not in cat_cols]

        # One-hot encode manually
        input_encoded = pd.get_dummies(input_df)
        sample_columns = sample_X.columns
        input_encoded = input_encoded.reindex(columns=sample_columns, fill_value=0)

        # Scale using corresponding scaler
        scaler = scalers[failure]
        input_scaled = scaler.transform(input_encoded)

        # Predict using model
        model = models[failure]
        prediction = model.predict(input_scaled)[0]

        results[failure] = "❌ Failure" if prediction == 1 else "✅ Normal"

    st.subheader("📊 Prediction Results")
    for k, v in results.items():
        st.write(f"**{k}:** {v}")
