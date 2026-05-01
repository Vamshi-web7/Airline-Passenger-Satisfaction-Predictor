import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Airline Passenger Satisfaction Predictor")

age = st.number_input("Age", 18, 100)

flight_distance = st.number_input("Flight Distance")

wifi_service = st.slider("Wifi Service Rating", 1, 5)

if st.button("Predict"):

    features = np.array([[age, flight_distance, wifi_service]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Satisfied Passenger")
    else:
        st.error("Not Satisfied")
