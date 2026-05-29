

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Project Debris",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🚀 Project Debris")
st.subheader("AI-Based Space Collision Risk Monitoring")

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
    return joblib.load(model_path)

model = load_model()


# ============================================================
# SIDEBAR INPUTS
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
    value=1e7,
    format="%e"
)

c_cov = st.sidebar.number_input(
    "Chaser Covariance",
    value=1e12,
    format="%e"
)
# ============================================================
# PREDICTION
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
# DISPLAY RESULTS
# ============================================================

st.markdown("---")

st.metric(
    "Collision Probability",
    f"{probability:.4f}"
)

st.markdown(
    f"""
    ## Risk Level:
    <span style='color:{color};font-size:32px'>
    {risk_level}
    </span>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    ## Recommended Action:
    **{action}**
    """
)

# ============================================================
# EVENT DETAILS
# ============================================================

st.markdown("---")

st.subheader("Event Details")

st.dataframe(input_data)

# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown("---")

st.success("System Operational")
