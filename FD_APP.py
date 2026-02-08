import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


model = joblib.load('fraud_detection_pipeline.pkl')


model.predict

st.title('Fraud Detection App')

st.markdown('Please enter the transaction details to predict whether it is fraudulent or not.')

st.divider()

transaction_type = st.selectbox('Transaction Type', ['PAYMENT', 'TRANSFER', 'CASH OUT', 'DEPOSIT'])

amount = st.number_input('Amount', min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input('Old Balance (Sender)', min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input('New Balance (Sender)', min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Old Balance (Receiver)', min_value=0.0, value=0.0)
newbalanceDest = st.number_input('New Balance (Receiver)', min_value=0.0, value=0.0)

if st.button('Predict'):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    prediction = model.predict(input_data)[0]

    st.subheader(f'Prediction Result: {int(prediction)}')

    if prediction == 1:
        st.error('The transaction is predicted to be fraudulent.')
    else:
        st.success('The transaction is predicted to be legitimate.')



import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "fraud_detection_pipeline.pkl")

model = joblib.load(model_path)
