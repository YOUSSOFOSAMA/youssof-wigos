import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PowerTransformer

# Load saved objects
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

with open('campaign_yeojohnson_transformer.pkl', 'rb') as f:
    campaign_transformer = pickle.load(f)

# Determine which numeric and categorical features are actually used in the model
categorical_cols = [col for col in selected_features if col in label_encoders]
numeric_cols = [col for col in selected_features if col not in categorical_cols]

# Final numeric columns to scale
final_numeric_cols = [col for col in selected_features if col in numeric_cols]

# UI
st.title("💼 Bank Deposit Prediction App")
st.markdown("""
This app predicts whether a client will **subscribe to a bank term deposit** based on their personal and banking information.
""")

# Input form
st.header("📋 Enter Client Information")
user_input = {}

# Input for numeric fields
for col in numeric_cols:
    if col == 'age':
        user_input[col] = np.sqrt(st.number_input("Age", min_value=18, max_value=100, value=30))
    elif col == 'balance_log':
        balance = st.number_input("Balance", value=0)
        shift = abs(balance) + 1 if balance < 0 else 0
        user_input[col] = np.log1p(balance + shift)
    elif col == 'day':
        user_input[col] = st.number_input("Day of Month (Contact)", min_value=1, max_value=31, value=15)
    elif col == 'duration_sqrt':
        duration = st.number_input("Duration (Call Duration in Seconds)", value=100)
        user_input[col] = np.sqrt(duration)
    elif col == 'campaign_yeojohnson':
        campaign = st.number_input("Number of Contacts During Campaign", value=1)
        user_input[col] = campaign_transformer.transform([[campaign]])[0][0]
    elif col == 'pdays_sqrt':
        pdays = st.number_input("Days Since Last Contact (999 if never contacted)", value=999)
        pdays = np.nan if pdays in [999, -1] else pdays
        user_input[col] = np.sqrt(pdays) if pdays is not None else 0
    elif col == 'previous_yeojohnson':
        previous = st.number_input("Number of Previous Contacts", value=0)
        # Use PowerTransformer directly, instead of re-fitting it every time
        previous_pt = PowerTransformer(method='yeo-johnson')
        user_input[col] = previous_pt.fit_transform([[previous]])[0][0]
    else:
        user_input[col] = st.number_input(col.replace('_', ' ').title(), value=0.0)

# Input for categorical fields
for col in categorical_cols:
    options = label_encoders[col].classes_.tolist()
    selection = st.selectbox(f"{col.replace('_', ' ').title()}", options)
    encoded = label_encoders[col].transform([selection])[0]
    user_input[col] = encoded

# Prediction
if st.button("🔮 Predict"):
    # Create DataFrame
    df_input = pd.DataFrame([user_input])

    # Apply scaling to final numeric features only
    df_input[final_numeric_cols] = scaler.transform(df_input[final_numeric_cols])

    # Filter final selected features
    df_input = df_input[selected_features]

    # Predict
    prediction = lr_model.predict(df_input)[0]
    probability = lr_model.predict_proba(df_input)[0][1]

    # Output
    if prediction == 1:
        st.success(f"✅ The model predicts the client is **likely to subscribe** to the deposit. (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ The model predicts the client is **not likely to subscribe** to the deposit. (Confidence: {1 - probability:.2%})")
