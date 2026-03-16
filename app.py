import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# Page configuration
# =========================
st.set_page_config(page_title="Thermal Stress Predictor", layout="centered")

st.title("Thermal Stress Predictor")
st.write("Predict thermal stresses at Top, Middle and Bottom of a Concrete Slab")

# =========================
# Load model
# =========================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model or scaler file not found. Keep model.pkl and scaler.pkl in the same folder.")
    st.stop()

# =========================
# Input parameters
# =========================
st.subheader("Material Parameters")

fa_percent = st.slider(
    "Fly Ash Replacement (%)",
    min_value=0,
    max_value=100,
    value=0
)

thickness = st.number_input(
    "Slab Thickness (mm)",
    min_value=100.0,
    value=300.0
)

time_hr = st.slider(
    "Time (hours)",
    min_value=0,
    max_value=24,
    value=1
)

# =========================
# Temperature inputs
# =========================
st.subheader("Temperature Inputs")

t_top = st.number_input(
    "Top Temperature (°C)",
    min_value=0.0,
    max_value=80.0,
    value=35.0
)

t_mid = st.number_input(
    "Middle Temperature (°C)",
    min_value=0.0,
    max_value=80.0,
    value=30.0
)

t_bot = st.number_input(
    "Bottom Temperature (°C)",
    min_value=0.0,
    max_value=80.0,
    value=25.0
)

# =========================
# Prediction
# =========================
if st.button("Predict Thermal Stress"):

    input_data = pd.DataFrame(
        [[fa_percent, thickness, time_hr, t_top, t_mid, t_bot]],
        columns=["FA_ %", "Thickness_mm", "Time_hr", "T_top", "T_mid", "T_bot"]
    )

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]

    top_stress, mid_stress, bot_stress = prediction

    # =========================
    # Show results
    # =========================
    st.subheader("Predicted Thermal Stress (MPa)")

    col1, col2, col3 = st.columns(3)

    col1.metric("Top Stress", f"{top_stress:.4f}")
    col2.metric("Middle Stress", f"{mid_stress:.4f}")
    col3.metric("Bottom Stress", f"{bot_stress:.4f}")

    # =========================
    # Non-linear stress graph
    # =========================

    depth_points = np.array([0, thickness/2, thickness])
    stress_points = np.array([top_stress, mid_stress, bot_stress])

    coeff = np.polyfit(depth_points, stress_points, 2)
    poly = np.poly1d(coeff)

    depth_curve = np.linspace(0, thickness, 100)
    stress_curve = poly(depth_curve)

    fig, ax = plt.subplots()

    ax.plot(stress_curve, depth_curve, linewidth=3)
    ax.scatter(stress_points, depth_points)

    ax.set_xlabel("Stress (MPa)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title("Thermal Stress Distribution")

    # Depth increases from 0 → slab thickness
    ax.set_ylim(thickness, 0)

    st.pyplot(fig)

    st.success("Prediction completed successfully")
