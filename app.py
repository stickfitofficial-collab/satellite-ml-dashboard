
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Debris Collision Monitor",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
div[data-testid="stMetricValue"] { font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# ── Feature names ───────────────────────────────────────────────────────────
FEATURES = [
    "relative_speed", "time_to_tca", "F10", "SSN", "AP",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "t_position_covariance_det", "c_position_covariance_det",
    "kinetic_proxy", "solar_drag_interaction", "urgency_score"
]

def classify(prob):
    if   prob >= 0.80: return "CRITICAL", "Maneuver immediately",               "#dc3545"
    elif prob >= 0.50: return "HIGH",     "Monitor closely / prepare maneuver",  "#fd7e14"
    elif prob >= 0.25: return "MEDIUM",   "Continue monitoring",                 "#0d6efd"
    else:              return "LOW",      "Routine tracking",                    "#198754"

# ── Simulated batch events using real model ─────────────────────────────────
@st.cache_data(ttl=60)
def simulate_events(n=200):
    np.random.seed(42)
    data = {
        "relative_speed":            np.random.uniform(0.1, 15.0, n),
        "time_to_tca":               np.random.uniform(0.1, 72.0, n),
        "F10":                       np.random.uniform(70, 250, n),
        "SSN":                       np.random.uniform(0, 200, n),
        "AP":                        np.random.uniform(0, 100, n),
        "relative_position_r":       np.random.uniform(-500, 500, n),
        "relative_position_t":       np.random.uniform(-500, 500, n),
        "relative_position_n":       np.random.uniform(-500, 500, n),
        "t_position_covariance_det": np.random.uniform(0.001, 10.0, n),
        "c_position_covariance_det": np.random.uniform(0.001, 10.0, n),
        "kinetic_proxy":             np.random.uniform(0.1, 100.0, n),
        "solar_drag_interaction":    np.random.uniform(0.0, 1.0, n),
        "urgency_score":             np.random.uniform(0.0, 1.0, n),
    }
    df = pd.DataFrame(data)
    probs = model.predict_proba(df[FEATURES])[:, 1]
    df["Probability"] = np.round(probs, 3)
    df["Risk Level"]  = df["Probability"].apply(lambda p: classify(p)[0])
    df["Action"]      = df["Probability"].apply(lambda p: classify(p)[1])
    df["Event ID"]    = [f"EVT-{i+1:03d}" for i in range(n)]
    df["TCA Time"]    = [
        (datetime.now() + timedelta(hours=h)).strftime("%H:%M %d-%b")
        for h in df["time_to_tca"]
    ]
    return df

df = simulate_events()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ Debris Monitor")
    st.caption(f"Model: XGBClassifier · 300 trees")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    st.subheader("🔍 Filters")
    levels = st.multiselect(
        "Risk Level",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )
    prob_min = st.slider("Min Probability", 0.0, 1.0, 0.0, 0.01)
    tca_max  = st.slider("Max TCA (hours)", 1, 72, 72)

    st.divider()
    st.subheader("📂 Pages")
    page = st.radio(
        label="",
        options=[
            "🔴 Live Alerts",
            "🔍 Event Details",
            "🧪 Manual Prediction",
            "📊 Analytics"
        ],
        index=0
    )

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Apply filters ────────────────────────────────────────────────────────────
filtered = df[
    df["Risk Level"].isin(levels) &
    (df["Probability"] >= prob_min) &
    (df["time_to_tca"] <= tca_max)
].sort_values("Probability", ascending=False)

icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🔵", "LOW": "🟢"}

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Alerts
# ════════════════════════════════════════════════════════════════════════════
if page == "🔴 Live Alerts":
    st.title("🔴 Live Conjunction Alerts")
    st.caption(f"Showing {len(filtered)} events · Last updated {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    # Metric row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events",   len(filtered))
    c2.metric("🔴 Critical",    len(filtered[filtered["Risk Level"] == "CRITICAL"]))
    c3.metric("🟠 High",        len(filtered[filtered["Risk Level"] == "HIGH"]))
    c4.metric("🔵 Medium",      len(filtered[filtered["Risk Level"] == "MEDIUM"]))
    c5.metric("🟢 Low",         len(filtered[filtered["Risk Level"] == "LOW"]))
    st.divider()

    # Critical
    critical = filtered[filtered["Risk Level"] == "CRITICAL"]
    if not critical.empty:
        st.error(f"⚠️ {len(critical)} CRITICAL conjunction(s) — immediate action required")
        for _, row in critical.iterrows():
            with st.expander(
                f"🔴 {row['Event ID']}  |  P = {row['Probability']:.3f}  |  TCA in {row['time_to_tca']:.1f}h",
                expanded=True
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Probability",    row["Probability"])
                col2.metric("Rel. Speed",     f"{row['relative_speed']:.2f} km/s")
                col3.metric("TCA",            row["TCA Time"])
                col4.metric("Urgency Score",  f"{row['urgency_score']:.2f}")
                st.error(f"Action: {row['Action']}")

    # High
    high = filtered[filtered["Risk Level"] == "HIGH"]
    if not high.empty:
        st.warning(f"⚡ {len(high)} HIGH risk conjunction(s)")
        for _, row in high.head(5).iterrows():
            with st.expander(
                f"🟠 {row['Event ID']}  |  P = {row['Probability']:.3f}  |  TCA in {row['time_to_tca']:.1f}h"
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Probability",   row["Probability"])
                col2.metric("Rel. Speed",    f"{row['relative_speed']:.2f} km/s")
                col3.metric("TCA",           row["TCA Time"])
                col4.metric("Urgency Score", f"{row['urgency_score']:.2f}")
                st.warning(f"Action: {row['Action']}")

    # Medium
    medium = filtered[filtered["Risk Level"] == "MEDIUM"]
    if not medium.empty:
        st.info(f"🔵 {len(medium)} MEDIUM risk conjunction(s)")

    # Low
    low = filtered[filtered["Risk Level"] == "LOW"]
    if not low.empty:
        st.success(f"🟢 {len(low)} LOW risk / safe conjunction(s)")

    st.divider()
    st.subheader("📋 All Events Table")
    disp = filtered.copy()
    disp["Risk Level"] = disp["Risk Level"].map(lambda x: f"{icons[x]} {x}")
    st.dataframe(
        disp[[
            "Event ID", "Risk Level", "Probability",
            "relative_speed", "time_to_tca",
            "urgency_score", "TCA Time", "Action"
        ]].rename(columns={
            "relative_speed": "Rel. Speed (km/s)",
            "time_to_tca":    "TCA (hrs)",
            "urgency_score":  "Urgency"
        }),
        use_container_width=True,
        hide_index=True
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Event Details
# ════════════════════════════════════════════════════════════════════════════
if page == "🔍 Event Details":
    st.title("🔍 Event Details")
    st.divider()

    sel_id = st.selectbox("Select Event ID", filtered["Event ID"].tolist())
    row    = filtered[filtered["Event ID"] == sel_id].iloc[0]
    level, action, _ = classify(row["Probability"])
    fn = {"CRITICAL": st.error, "HIGH": st.warning,
          "MEDIUM": st.info,    "LOW":  st.success}[level]

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Conjunction Summary")
        fn(f"Risk Level: {icons[level]} {level}")

        m1, m2 = st.columns(2)
        m1.metric("Collision Probability", f"{row['Probability']:.3f}")
        m2.metric("TCA Time",              row["TCA Time"])

        m3, m4 = st.columns(2)
        m3.metric("Relative Speed",  f"{row['relative_speed']:.2f} km/s")
        m4.metric("Urgency Score",   f"{row['urgency_score']:.3f}")

        m5, m6 = st.columns(2)
        m5.metric("Kinetic Proxy",   f"{row['kinetic_proxy']:.2f}")
        m6.metric("TCA (hours)",     f"{row['time_to_tca']:.1f}")

        st.divider()
        st.subheader("🌞 Space Weather")
        sw1, sw2, sw3 = st.columns(3)
        sw1.metric("F10 Solar Flux", f"{row['F10']:.1f}")
        sw2.metric("SSN Sunspots",   f"{row['SSN']:.0f}")
        sw3.metric("AP Geomag.",     f"{row['AP']:.1f}")

        st.divider()
        st.subheader("📐 Relative Position")
        p1, p2, p3 = st.columns(3)
        p1.metric("R (radial)",     f"{row['relative_position_r']:.1f} m")
        p2.metric("T (tangential)", f"{row['relative_position_t']:.1f} m")
        p3.metric("N (normal)",     f"{row['relative_position_n']:.1f} m")

        st.divider()
        fn(f"📋 Recommended Action: {action}")

    with col_r:
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(float(row["Probability"]), 3),
            gauge={
                "axis": {"range": [0, 1]},
                "bar":  {"color": "crimson"},
                "steps": [
                    {"range": [0.00, 0.25], "color": "#d4edda"},
                    {"range": [0.25, 0.50], "color": "#cce5ff"},
                    {"range": [0.50, 0.80], "color": "#fff3cd"},
                    {"range": [0.80, 1.00], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 0.80
                }
            },
            title={"text": "Collision Probability"}
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # All 13 feature values for this event
        st.subheader("📊 All Feature Values")
        feat_df = pd.DataFrame({
            "Feature": FEATURES,
            "Value":   [round(float(row[f]), 4) for f in FEATURES]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Manual Prediction
# ════════════════════════════════════════════════════════════════════════════
if page == "🧪 Manual Prediction":
    st.title("🧪 Manual Conjunction Prediction")
    st.caption("Enter feature values manually — your XGBoost model predicts in real time")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🚀 Orbital")
        relative_speed            = st.number_input("Relative Speed (km/s)",         0.0,  20.0,  7.5,  0.1)
        time_to_tca               = st.number_input("Time to TCA (hours)",            0.0,  72.0, 12.0,  0.5)
        relative_position_r       = st.number_input("Position R — Radial (m)",    -1000.0,1000.0, 50.0,  1.0)
        relative_position_t       = st.number_input("Position T — Tangential (m)",-1000.0,1000.0,-30.0,  1.0)
        relative_position_n       = st.number_input("Position N — Normal (m)",    -1000.0,1000.0, 10.0,  1.0)

    with col2:
        st.subheader("🌞 Space Weather")
        F10                       = st.number_input("F10 Solar Flux",               60.0, 300.0,150.0,  1.0)
        SSN                       = st.number_input("Sunspot Number (SSN)",           0.0, 250.0, 80.0,  1.0)
        AP                        = st.number_input("AP Geomagnetic Index",           0.0, 100.0, 15.0,  0.5)
        solar_drag_interaction    = st.number_input("Solar Drag Interaction",         0.0,   1.0,  0.3,  0.01)

    with col3:
        st.subheader("📐 Uncertainty & Risk")
        t_position_covariance_det = st.number_input("Target Covariance Det",          0.0,  20.0,  2.5,  0.1)
        c_position_covariance_det = st.number_input("Chaser Covariance Det",          0.0,  20.0,  1.8,  0.1)
        kinetic_proxy             = st.number_input("Kinetic Proxy",                  0.0, 200.0, 45.0,  1.0)
        urgency_score             = st.number_input("Urgency Score",                  0.0,   1.0,  0.6,  0.01)

    st.divider()
    if st.button("🔮 Predict Risk Now", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "relative_speed":            relative_speed,
            "time_to_tca":               time_to_tca,
            "F10":                       F10,
            "SSN":                       SSN,
            "AP":                        AP,
            "relative_position_r":       relative_position_r,
            "relative_position_t":       relative_position_t,
            "relative_position_n":       relative_position_n,
            "t_position_covariance_det": t_position_covariance_det,
            "c_position_covariance_det": c_position_covariance_det,
            "kinetic_proxy":             kinetic_proxy,
            "solar_drag_interaction":    solar_drag_interaction,
            "urgency_score":             urgency_score,
        }])
        prob = float(model.predict_proba(input_df[FEATURES])[:, 1][0])
        level, action, color = classify(prob)
        fn = {"CRITICAL": st.error, "HIGH": st.warning,
              "MEDIUM": st.info,    "LOW":  st.success}[level]

        r1, r2, r3 = st.columns(3)
        r1.metric("Collision Probability", f"{prob:.4f}")
        r2.metric("Risk Level",            f"{icons[level]} {level}")
        r3.metric("Recommended Action",    action)
        fn(f"{'⚠️' if level in ['CRITICAL','HIGH'] else '✅'} {level} — {action}")

        # Mini gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob, 4),
            gauge={
                "axis": {"range": [0, 1]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0.00, 0.25], "color": "#d4edda"},
                    {"range": [0.25, 0.50], "color": "#cce5ff"},
                    {"range": [0.50, 0.80], "color": "#fff3cd"},
                    {"range": [0.80, 1.00], "color": "#f8d7da"},
                ],
            },
            title={"text": "Predicted Probability"}
        ))
        fig.update_layout(height=260, margin=dict(t=40, b=10, l=60, r=60))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Analytics
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Analytics":
    st.title("📊 Analytics")
    st.caption("Distribution and trend analysis across all 200 simulated events")
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        counts = df["Risk Level"].value_counts().reindex(
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0
        ).reset_index()
        counts.columns = ["Risk Level", "Count"]
        fig = px.bar(
            counts, x="Risk Level", y="Count", color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#dc3545", "HIGH": "#fd7e14",
                "MEDIUM":   "#0d6efd", "LOW":  "#198754"
            },
            title="Alert Count by Risk Level"
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.histogram(
            df, x="Probability", nbins=40,
            color_discrete_sequence=["#0d6efd"],
            title="Probability Distribution (all events)"
        )
        fig.add_vline(x=0.25, line_dash="dash", line_color="blue",   annotation_text="Medium")
        fig.add_vline(x=0.50, line_dash="dash", line_color="orange", annotation_text="High")
        fig.add_vline(x=0.80, line_dash="dash", line_color="red",    annotation_text="Critical")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig = px.scatter(
            df, x="relative_speed", y="Probability",
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#dc3545", "HIGH": "#fd7e14",
                "MEDIUM":   "#0d6efd", "LOW":  "#198754"
            },
            opacity=0.6,
            title="Relative Speed vs Collision Probability"
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        fig = px.scatter(
            df, x="time_to_tca", y="Probability",
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#dc3545", "HIGH": "#fd7e14",
                "MEDIUM":   "#0d6efd", "LOW":  "#198754"
            },
            opacity=0.6,
            title="Time to TCA vs Collision Probability"
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        fig = px.box(
            df, x="Risk Level", y="relative_speed",
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#dc3545", "HIGH": "#fd7e14",
                "MEDIUM":   "#0d6efd", "LOW":  "#198754"
            },
            category_orders={"Risk Level": ["CRITICAL","HIGH","MEDIUM","LOW"]},
            title="Speed Distribution by Risk Level"
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        fig = px.box(
            df, x="Risk Level", y="urgency_score",
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#dc3545", "HIGH": "#fd7e14",
                "MEDIUM":   "#0d6efd", "LOW":  "#198754"
            },
            category_orders={"Risk Level": ["CRITICAL","HIGH","MEDIUM","LOW"]},
            title="Urgency Score Distribution by Risk Level"
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
