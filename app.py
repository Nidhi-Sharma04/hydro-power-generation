"""
app.py — Hydropower Energy Output Forecaster
Run with:  streamlit run app.py
Requires:  hydropower_model.sav  and  scaler.sav  in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💧 Hydropower Forecast",
    page_icon="💧",
    layout="centered",
)

# ─── Load Model & Scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("hydropower_model.sav")
    return model

model = load_artifacts()

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("💧 Hydropower Energy Output Forecast")
st.markdown(
    "Adjust the environmental and operational parameters below "
    "to predict how much energy (in **MW**) the plant will generate."
)
st.divider()

# ─── Sidebar — Input Parameters ──────────────────────────────────────────────
st.sidebar.header("⚙️ Input Parameters")

rainfall           = st.sidebar.slider("Rainfall (mm)",                  0.0,  280.0, 120.0, 1.0)
temperature        = st.sidebar.slider("Temperature (°C)",                4.0,   48.0,  25.0, 0.5)
humidity           = st.sidebar.slider("Humidity (%)",                   20.0,  100.0,  70.0, 1.0)
river_flow         = st.sidebar.slider("River Flow (m³/s)",              50.0,  700.0, 300.0, 5.0)
reservoir_level    = st.sidebar.slider("Reservoir Level (%)",            10.0,  100.0,  65.0, 1.0)
upstream_inflow    = st.sidebar.slider("Upstream Inflow (m³/s)",          0.0,  750.0, 300.0, 5.0)
sediment_load      = st.sidebar.slider("Sediment Load",                  10.0,   80.0,  40.0, 0.5)
turbine_efficiency = st.sidebar.slider("Turbine Efficiency (0–1)",        0.67,   0.84,  0.77, 0.001,
                                        format="%.3f")

# ─── Single Prediction ───────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "rainfall_mm"              : rainfall,
    "temperature_c"            : temperature,
    "humidity_percent"         : humidity,
    "river_flow_m3s"           : river_flow,
    "reservoir_level_percent"  : reservoir_level,
    "upstream_inflow_m3s"      : upstream_inflow,
    "sediment_load"            : sediment_load,
    "turbine_efficiency"       : turbine_efficiency,
}])

scaled_input  = input_data
prediction    = model.predict(scaled_input)[0]

# Clamp to a realistic min of 0
prediction = max(prediction, 0.0)

# ─── Result Display ──────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Current Input Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}).style.format("{:.3f}"),
                 use_container_width=True)

with col2:
    st.subheader("⚡ Predicted Output")
    st.metric(label="Hydropower Output", value=f"{prediction:.2f} MW")

    # Simple colour indicator
    if prediction < 5:
        st.error("🔴 Very Low Output")
    elif prediction < 15:
        st.warning("🟡 Moderate Output")
    else:
        st.success("🟢 High Output")

st.divider()

# ─── Multi-Step Forecast ─────────────────────────────────────────────────────
st.subheader("📈 Multi-Step Forecast (River Flow Sweep)")
st.markdown(
    "See how the predicted output changes as **River Flow** varies "
    "from low to high, with all other parameters held constant."
)

flow_range     = np.linspace(50, 700, 60)
forecast_rows  = []
for flow in flow_range:
    row = input_data.copy()
    row["river_flow_m3s"] = flow
    scaled_row = row
    pred       = max(model.predict(scaled_row)[0], 0.0)
    forecast_rows.append({"River Flow (m³/s)": flow, "Predicted Output (MW)": pred})

forecast_df = pd.DataFrame(forecast_rows)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(forecast_df["River Flow (m³/s)"], forecast_df["Predicted Output (MW)"],
        color="steelblue", linewidth=2.5)
ax.fill_between(forecast_df["River Flow (m³/s)"],
                forecast_df["Predicted Output (MW)"] * 0.92,
                forecast_df["Predicted Output (MW)"] * 1.08,
                alpha=0.2, color="steelblue", label="±8% uncertainty band")
ax.axvline(river_flow, color="red", linestyle="--", linewidth=1.5, label=f"Current Flow: {river_flow} m³/s")
ax.set_xlabel("River Flow (m³/s)", fontsize=11)
ax.set_ylabel("Predicted Output (MW)", fontsize=11)
ax.set_title("Hydropower Output vs River Flow", fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

# ─── Download Forecast ───────────────────────────────────────────────────────
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️  Download Forecast as CSV",
    data=csv,
    file_name="hydropower_forecast.csv",
    mime="text/csv",
)

st.divider()
st.caption("Model: Pipeline (StandardScaler + best estimator) trained on hydropower_dataset.csv • hydropower_model.sav")
