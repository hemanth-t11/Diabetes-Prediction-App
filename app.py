import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load(open('diabetes_model.pkl', 'rb'))
scaler = joblib.load(open('diabetes_scaler.pkl', 'rb'))

st.title(" Diabetes Prediction App")
st.markdown("Enter the patient's health metrics below:")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 50, 200)
bp = st.number_input("Blood Pressure", 30, 150)
skin = st.number_input("Skin Thickness", 0, 99)
insulin = st.number_input("Insulin", 0, 846)
bmi = st.number_input("BMI", 10.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)

    if prediction[0] == 1:
        st.error(" High risk: Likely diabetic.")
    else:
        st.success(" Low risk: Not diabetic.")
