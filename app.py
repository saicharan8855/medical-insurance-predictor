import streamlit as st
import joblib
import numpy as np

model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Medical Insurance Cost Predictor")

age = st.number_input("Age", min_value=1, max_value=100)
bmi = st.number_input("BMI", min_value=1.0, max_value=60.0)
children = st.number_input("Children", min_value=0, max_value=10)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

sex_enc = 1 if sex == "male" else 0
smoker_enc = 1 if smoker == "yes" else 0
region_enc = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}[region]

if st.button("Predict"):
    input_data = np.array([[age, bmi, children, sex_enc, smoker_enc, region_enc]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")