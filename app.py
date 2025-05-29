import streamlit as st
import pandas as pd
import pickle

# Hard-coded features
selected_features = [
    'age', 'job', 'education', 'balance_log',
    'housing', 'loan', 'day', 'month', 'duration_sqrt'
    , 'pdays_sqrt', 'previous_yeojohnson', 'poutcome'
]

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numeric_cols = ['age', 'balance_log', 'day', 'duration_sqrt', 'campaign_yeojohnson', 'pdays_sqrt', 'previous_yeojohnson']

# Load saved model, scaler and label encoders
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

def user_input_features():
    inputs = {}
    # For numeric inputs, provide a number input box
    for col in numeric_cols:
        inputs[col] = st.number_input(f"Enter {col}", value=0.0)

    # For categorical inputs, dropdown with possible classes from label encoders
    for col in categorical_cols:
        options = label_encoders[col].classes_.tolist()
        inputs[col] = st.selectbox(f"Select {col}", options)

    # Return DataFrame with one row
    return pd.DataFrame([inputs])

input_df = user_input_features()

# Apply label encoding to categorical columns
for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Ensure columns order
input_df = input_df[selected_features]

# Predict
prediction = lr_model.predict(input_df)
prediction_proba = lr_model.predict_proba(input_df)

st.write(f"Prediction (0=No Deposit, 1=Deposit): {prediction[0]}")
st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")
