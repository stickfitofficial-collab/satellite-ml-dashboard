import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================

# PAGE CONFIG

# ============================================================

st.set_page_config(
page_title="Project Debris",
layout="wide"
)

# ============================================================

# LOAD MODEL

# ============================================================

model = joblib.load("xgb_model.pkl")

# ============================================================

# TITLE

# ============================================================

st.title("🚀 Project Debris")
st.subheader("AI-Based Space Collision Risk Monitoring")

# ============================================================

# SIDEBAR

# ============================================================

st.sidebar.header("Conjunction Parameters")

relative_speed = st.sidebar.number_input(
"Relative Speed",
value=14000.0
)

time_to_tca = st.sidebar.number_input(
"Time To TCA",
value=2.0
)

F10 = st.sidebar.number_input(
"F10 Solar Flux",
value=80.0
)

SSN = st.sidebar.number_input(
"Sunspot Number",
value=20.0
)

AP = st.sidebar.number_input(
"Geomagnetic AP",
value=15.0
)

relative_position_r = st.sidebar.number_input(
"Relative Position R",
value=-20.0
)

relative_position_t = st.sidebar.number_input(
"Relative Position T",
value=15.0
)

relative_position_n = st.sidebar.number_input(
"Relative Position N",
value=10.0
)

t_cov = st.sidebar.number_input(
"Target Covariance",
value=1e7
)

c_cov = st.sidebar.number_input(
"Chaser Covariance",
value=1e12
)

# ============================================================

# FEATURE ENGINEERING

# ============================================================

kinetic_proxy = relative_speed ** 2

solar_drag_interaction = F10 * AP

urgency_score = 1 / (time_to_tca + 1e-6)

# ============================================================

# INPUT DATA

# ============================================================

data = {
"relative_speed": [relative_speed],
"time_to_tca": [time_to_tca],
"F10": [F10],
"SSN": [SSN],
"AP": [AP],
"relative_position_r": [relative_position_r],
"relative_position_t": [relative_position_t],
"relative_position_n": [relative_position_n],
"t_position_covariance_det": [t_cov],
"c_position_covariance_det": [c_cov],
"kinetic_proxy": [kinetic_proxy],
"solar_drag_interaction": [solar_drag_interaction],
"urgency_score": [urgency_score]
}

input_data = pd.DataFrame(data)

# ============================================================

# PREDICT

# ============================================================

probability = model.predict_proba(input_data)[0][1]

# ============================================================

# RISK LEVEL

# ============================================================

if probability < 0.20:
risk_level = "LOW"
color = "green"

elif probability < 0.50:
risk_level = "MEDIUM"
color = "orange"

elif probability < 0.80:
risk_level = "HIGH"
color = "red"

else:
risk_level = "CRITICAL"
color = "darkred"

# ============================================================

# ALERT ACTION

# ============================================================

if probability >= 0.95:
action = "IMMEDIATE MANUAL REVIEW"

elif probability >= 0.80:
action = "HIGH PRIORITY ALERT"

elif probability >= 0.50:
action = "MONITOR CLOSELY"

else:
action = "NO ACTION"

# ============================================================

# DISPLAY

# ============================================================

st.metric(
"Collision Probability",
f"{probability:.4f}"
)

st.markdown(
f""" <h2 style='color:{color};'>
Risk Level: {risk_level} </h2>
""",
unsafe_allow_html=True
)

st.subheader("Recommended Action")

st.write(action)

st.subheader("Input Features")

st.dataframe(input_data)

st.success("System Operational")
