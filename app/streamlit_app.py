import streamlit as st
import joblib
import pandas as pd

model = joblib.load('src/model_pipeline.joblib')

st.title("Telco Customer Churn Predictor")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
MonthlyCharges = st.slider("Monthly Charges", 10, 150, 70)
TotalCharges = st.slider("Total Charges", 0, 8000, 2000)

input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'Contract': [Contract],
    'DeviceProtection': [DeviceProtection],   # <-- Added here!
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
})

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.success("Prediction: " + ("Customer Will Churn 😬" if prediction == 1 else "Customer Will Stay 😊"))
    st.write(f"Churn Probability: {probability:.2%}")
