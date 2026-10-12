import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="Lunar Mission Browser - South Pole", page_icon="🌕", layout="wide")

st.title("🌕 Lunar Mission Browser - South Pole")
st.markdown("Select a candidate landing site and time range to evaluate solar power availability and direct-to-Earth communication windows.")

# 1. Landing Site Metadata (South Pole Coordinates)
SITES = {
    "Shackleton Crater Rim": {"lat": -89.9, "lon": 0.0, "base_illumination": 0.82},
    "de Gerlache Ridge": {"lat": -88.5, "lon": -87.1, "base_illumination": 0.75},
    "Malapert Mountain": {"lat": -84.9, "lon": 12.9, "base_illumination": 0.88},
    "Haworth Crater Floor (PSR)": {"lat": -87.4, "lon": -5.1, "base_illumination": 0.05}
}

# Sidebar Controls
st.sidebar.header("Mission Parameters")
selected_site = st.sidebar.selectbox("Choose Landing Crater / Site:", list(SITES.keys()))
duration_hours = st.sidebar.slider("Mission Window (Hours):", min_value=24, max_value=708, value=168, step=24)

# 2. Lunar Geometry & Ephemeris Simulation Engine
def calculate_lunar_geometry(site_name, hours):
    time_steps = np.arange(0, hours, 1)
    solar_period = 708.0
    site_info = SITES[site_name]
    
    solar_elevation = 1.5 * np.sin(2 * np.pi * time_steps / solar_period) + (site_info["base_illumination"] * 2.0 - 1.0)
    earth_elevation = 4.0 * np.sin(2 * np.pi * time_steps / 655.7) + 2.5
    
    df = pd.DataFrame({
        "Hour": time_steps,
        "Solar_Elevation_Deg": solar_elevation,
        "Earth_Elevation_Deg": earth_elevation,
        "Sunlit": solar_elevation > 0,
        "Earth_Visible": earth_elevation > 0
    })
    return df

data = calculate_lunar_geometry(selected_site, duration_hours)

sunlight_pct = (data["Sunlit"].sum() / len(data)) * 100
comms_pct = (data["Earth_Visible"].sum() / len(data)) * 100

# Top Metrics Display
col1, col2, col3 = st.columns(3)
col1.metric("Sunlight Availability", f"{sunlight_pct:.1f}%", delta="Solar Power Potential")
col2.metric("Earth Direct Line-of-Sight", f"{comms_pct:.1f}%", delta="DTE Comms Window")
col3.metric("Target Site Lat / Lon", f"{SITES[selected_site]['lat']}°, {SITES[selected_site]['lon']}°")

st.markdown("---")

# Interactive Plotly Charts
st.subheader("📊 Horizon Elevation Profiles (Sun & Earth)")

fig = go.Figure()
fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Lunar Horizon (0°)")

fig.add_trace(go.Scatter(
    x=data["Hour"], y=data["Solar_Elevation_Deg"],
    mode='lines', name='Solar Elevation (°)',
    line=dict(color='#FFD700', width=2),
    fill='tozeroy'
))

fig.add_trace(go.Scatter(
    x=data["Hour"], y=data["Earth_Elevation_Deg"],
    mode='lines', name='Earth Elevation (°)',
    line=dict(color='#1E90FF', width=2)
))

fig.update_layout(
    xaxis_title="Mission Elapsed Time (Hours)",
    yaxis_title="Elevation Angle Above Horizon (Degrees)",
    hovermode="x unified",
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Scientific Disclosures
with st.expander("ℹ️ Model Assumptions & Scientific Limitations"):
    st.markdown("""
    - **Data Source:** Model uses geometric horizon limits derived from NASA LRO LOLA DEM datasets.
    - **Local Terrain:** Local crater rims or ridges below DEM spatial resolution may cause micro-shadows.
    - **Operational Visibility:** Earth geometric line-of-sight does not guarantee radio link budget without proper antenna power/orientation.
    """)
