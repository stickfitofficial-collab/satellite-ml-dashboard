

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

relative_speed = st.sidebar.slider(
    "Relative Speed",
    0,
    20000,
    14000
)

time_to_tca = st.sidebar.slider(
    "Time To TCA",
    0.1,
    10.0,
    2.0
)

F10 = st.sidebar.slider(
    "F10 Solar Flux",
    0,
    300,
    80
)

SSN = st.sidebar.slider(
    "Sunspot Number",
    0,
    300,
    20
)

AP = st.sidebar.slider(
    "Geomagnetic AP",
    0,
    100,
    15
)

relative_position_r = st.sidebar.slider(
    "Relative Position R",
    -500,
    500,
    -20
)

relative_position_t = st.sidebar.slider(
    "Relative Position T",
    -500,
    500,
    15
)

relative_position_n = st.sidebar.slider(
    "Relative Position N",
    -500,
    500,
    10
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
# FEATURE ENGINEERING
# ============================================================

kinetic_proxy = relative_speed ** 2

solar_drag_interaction = F10 * AP

urgency_score = 1 / (time_to_tca + 1e-6)

# ============================================================
# INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

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
})

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
