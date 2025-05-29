import streamlit as st
import pandas as pd
import pickle

# Load saved model, scaler, label encoders, and selected features
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

st.write("Available Label Encoders:", list(label_encoders.keys()))

# Identify categorical and numeric columns based on loaded encoders and selected features
categorical_cols = [col for col in selected_features if col in label_encoders]
numeric_cols = [col for col in selected_features if col not in categorical_cols]

def user_input_features():
    st.header("Enter Client Information")
    inputs = {}

    # Numeric inputs
    for col in numeric_cols:
        inputs[col] = st.number_input(f"{col.replace('_', ' ').title()}", value=0.0)

    # Categorical inputs
    for col in categorical_cols:
        options = label_encoders[col].classes_.tolist()
        inputs[col] = st.selectbox(f"{col.replace('_', ' ').title()}", options)

    return pd.DataFrame([inputs])

# Get user input dataframe
input_df = user_input_features()

# Encode categorical columns using loaded label encoders
for col in categorical_cols:
    if col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Scale numeric columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Ensure columns order matches what the model expects
missing_cols = [col for col in selected_features if col not in input_df.columns]
if missing_cols:
    st.error(f"Missing required input columns: {missing_cols}")
else:
    input_df = input_df[selected_features]

    # Make prediction
    prediction = lr_model.predict(input_df)
    prediction_proba = lr_model.predict_proba(input_df)

    # Display results
    st.subheader("Prediction Result")
    st.write(f"Prediction (0 = No Deposit, 1 = Deposit): **{prediction[0]}**")
    st.write(f"Probability of Deposit: **{prediction_proba[0][1]:.2f}**")
