import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and preprocessing objects
with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

with open("pt_campaign.pkl", "rb") as f:
    pt_campaign = pickle.load(f)

with open("pt_previous.pkl", "rb") as f:
    pt_previous = pickle.load(f)

# Streamlit form inputs
st.title("Bank Term Deposit Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ['admin.', 'technician', 'services', 'management', 'retired',
                           'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
                           'self-employed', 'student', 'unknown'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['secondary', 'tertiary', 'primary', 'unknown'])
default = st.selectbox("Credit in Default?", ['no', 'yes'])
housing = st.selectbox("Housing Loan?", ['no', 'yes'])
loan = st.selectbox("Personal Loan?", ['no', 'yes'])
contact = st.selectbox("Contact Communication", ['cellular', 'telephone'])
month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=100)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact", min_value=-1, value=-1)
previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])

# Predict button
if st.button("Predict"):
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign_yeojohnson': campaign,  # Will be transformed
        'pdays': pdays,
        'previous_yeojohnson': previous,  # Will be transformed
        'poutcome': poutcome
    }

    input_df = pd.DataFrame([input_dict])

    # Apply label encoders to categorical features
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Apply PowerTransformers
    input_df['campaign_yeojohnson'] = pt_campaign.transform([[input_dict['campaign_yeojohnson']]])[0][0]
    input_df['previous_yeojohnson'] = pt_previous.transform([[input_dict['previous_yeojohnson']]])[0][0]

    # Select only features used in training
    input_df = input_df[selected_features]

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.success(f"Client is likely to subscribe to a term deposit. Probability: {prediction_proba:.2%}")
    else:
        st.warning(f"Client is unlikely to subscribe. Probability: {prediction_proba:.2%}")
