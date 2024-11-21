
import streamlit as st
import pandas as pd
import joblib

st.title("Weather Prediction App")

model = joblib.load("../models/weather_model.pkl")

st.sidebar.header("Input Features")
year = st.sidebar.number_input("Year", min_value=2020, max_value=2030, value=2023)
month = st.sidebar.slider("Month", min_value=1, max_value=12, value=1)
day = st.sidebar.slider("Day", min_value=1, max_value=31, value=1)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0)

if st.button("Predict Temperature"):
    input_data = pd.DataFrame([[year, month, day, humidity, rainfall]], 
                              columns=['Year', 'Month', 'Day', 'Humidity', 'Rainfall'])
    prediction = model.predict(input_data)
    st.write(f"Predicted Temperature: {prediction[0]:.2f}°C")
