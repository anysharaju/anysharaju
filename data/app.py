import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="AI JalRakshak - Water Quality Predictor")

st.title("💧 AI JalRakshak")
st.subheader("Predict whether water is Safe or Unsafe")

# Load model
model = pickle.load(open("water_model.pkl", "rb"))

# Input fields
pH = st.slider("pH Level", 0.0, 14.0, 7.0)
do = st.slider("Dissolved Oxygen (mg/L)", 0.0, 10.0, 6.0)
cond = st.slider("Conductivity (µS/cm)", 100, 1000, 400)
bod = st.slider("Biological Oxygen Demand (mg/L)", 0.0, 10.0, 3.0)
nit = st.slider("Nitrogen Content (mg/L)", 0.0, 10.0, 2.0)

# Predict
if st.button("Predict Quality"):
    features = np.array([[pH, do, cond, bod, nit]])
    result = model.predict(features)[0]

    if result == "Safe":
        st.success("💧 The water is SAFE to use!")
    else:
        st.error("⚠️ The water is UNSAFE. Needs treatment.")
