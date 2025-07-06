import streamlit as st
import pandas as pd
import numpy as np
import joblib


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

# Load encoder
encoder = joblib.load("encoder.pkl")

# ========== Input UI ==========
st.title("🛠️ Equipment Failure Predictor")
st.markdown("Enter the values below to predict possible failures.")

# Categorical options (3 values for Equipment Type and 3 for Location)
equipment_types = ['Axial', 'Exhaust', 'Centrifugal']
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
# Add derived feature manually
data_input['vibration_total'] = np.sqrt(
    data_input['Vibration_X']**2 + 
    data_input['Vibration_Y']**2 + 
    data_input['Vibration_Z']**2
)
# ========== Prediction ==========
if st.button("🔍 Predict Failures"):
    input_df = pd.DataFrame([data_input])

    results = {}

    for failure in failure_cols:
        label = failure.replace(" ", "_")

        # ➕ Step 1: Apply encoder (same as training)
        input_encoded = encoder.transform(input_df)

        # ➕ Step 2: Remove 'cat__' and 'remainder__' from feature names
        raw_cols = encoder.get_feature_names_out()
        clean_cols = [col.replace("cat__", "").replace("remainder__", "") for col in raw_cols]
        input_encoded_df = pd.DataFrame(input_encoded, columns=clean_cols)

        # ✅ Step 3: Fix column order to match training data
        sample_X = pd.read_csv(f"split_data/{label}/X_train.csv")
        input_encoded_df = input_encoded_df[sample_X.columns]

        # ✅ Step 4: Apply scaler
        scaler = scalers[failure]
        input_scaled = scaler.transform(input_encoded_df)

        # ✅ Step 5: Predict
        model = models[failure]
        prediction = model.predict(input_scaled)[0]

        results[failure] = "❌ Failure" if prediction == 1 else "✅ Normal"

    # Show results
    st.subheader("📊 Prediction Results")
    for k, v in results.items():
        st.write(f"**{k}:** {v}")
