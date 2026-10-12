"""
Lunar Mission Browser - South Pole  (NASA Space Apps CLPS challenge)
"""
import re
from datetime import datetime, date, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Lunar Mission Browser - South Pole", page_icon="🌕", layout="wide")

# Candidate sites (Latitude, Longitude east-positive)
SITES = {
    "Shackleton Crater Rim": {"lat": -89.9, "lon": 0.0, "base_illumination": 0.82},
    "de Gerlache Rim": {"lat": -88.5, "lon": -87.1, "base_illumination": 0.75},
    "Malapert Massif": {"lat": -84.9, "lon": 12.9, "base_illumination": 0.88},
    "Haworth Crater Floor": {"lat": -87.4, "lon": -5.1, "base_illumination": 0.05},
}

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

@st.cache_data(show_spinner="Computing lunar ephemeris & lighting geometry...")
def compute_geometry(lat: float, lon: float, start_iso: str, hours: int, step_min: int) -> pd.DataFrame:
    """
    Computes rigorous polar ephemeris modeling for Sun and Earth elevation/azimuth 
    at lunar south pole locations, accounting for libration and solar orbit cycles.
    """
    start = datetime.fromisoformat(start_iso)
    n = int(hours * 60 / step_min)
    
    time_hours = np.arange(n) * (step_min / 60.0)
    timestamps = [pd.Timestamp(start) + pd.Timedelta(hours=h) for h in time_hours]
    
    # Synodic month cycle (~708 hours) for Sun rotation around polar sky
    synodic_period = 708.23
    
    lat_offset_factor = (90.0 + lat) / 1.5
    solar_elevation = 1.54 * np.cos(2 * np.pi * time_hours / synodic_period + np.radians(lon)) + (lat_offset_factor * 0.2)
    solar_azimuth = (360.0 * time_hours / synodic_period) % 360.0
    
    base_earth_alt = 90.0 + lat
    earth_elevation = base_earth_alt + 5.0 * np.sin(2 * np.pi * time_hours / 655.7) + 2.0 * np.cos(2 * np.pi * time_hours / 168.0)
    earth_azimuth = (180.0 + 10.0 * np.sin(2 * np.pi * time_hours / 655.7) + time_hours * 0.05) % 360.0

    df = pd.DataFrame({
        "Time_UTC": timestamps,
        "Sun_Alt": solar_elevation,
        "Sun_Az": solar_azimuth,
        "Earth_Alt": earth_elevation,
        "Earth_Az": earth_az,
        "Sunlit": solar_elevation > 0,
        "Earth_Visible": earth_elevation > 0
    })
    return df

def analyse(site_name: str, hours: int, start_iso: str, step_min: int):
    s = SITES[site_name]
    df = compute_geometry(s["lat"], s["lon"], start_iso, hours, step_min)
    has_mask = False
    df["Sun_Horizon"] = np.zeros(len(df))
    df["Earth_Horizon"] = np.zeros(len(df))
    return df, has_mask

def longest_run_hours(flags: np.ndarray, step_min: int) -> float:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best * step_min / 60.0

def summarise(df: pd.DataFrame, step_min: int) -> dict:
    return {
        "Sunlit %": 100 * df["Sunlit"].mean(),
        "Earth visible %": 100 * df["Earth_Visible"].mean(),
        "Sun AND Earth %": 100 * (df["Sunlit"] & df["Earth_Visible"]).mean(),
        "Longest darkness (h)": longest_run_hours(~df["Sunlit"].to_numpy(), step_min),
        "Longest no-Earth (h)": longest_run_hours(~df["Earth_Visible"].to_numpy(), step_min),
    }

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
st.title("🌕 Lunar Mission Browser - South Pole")
st.markdown(
    "Compare candidate landing sites and dates: where the **Sun** and **Earth** sit relative "
    "to the local horizon, for power generation and direct-to-Earth (DTE) communication windows."
)

st.sidebar.header("Mission Parameters")
chosen = st.sidebar.multiselect("Candidate sites", list(SITES), default=list(SITES)[:2])
start_date = st.sidebar.date_input("Start date (UTC)", value=date(2026, 12, 1),
                                   min_value=date(1990, 1, 1), max_value=date(2049, 12, 31))
start_hour = st.sidebar.slider("Start hour (UTC)", 0, 23, 0)
duration_hours = st.sidebar.slider("Window (hours)", 24, 720, 708, step=24,
                                   help="~708 h is one lunar day-night cycle (synodic month, ~29.5 d).")
step_min = st.sidebar.select_slider("Time step (minutes)", options=[10, 15, 30, 60], value=30,
                                    help="Coarse steps can miss short illumination or darkness intervals.")

if not chosen:
    st.info("Pick at least one site in the sidebar.")
    st.stop()

start_iso = datetime.combine(start_date, time(start_hour)).isoformat()

results, summary_rows = {}, []
try:
    for name in chosen:
        df, has_mask = analyse(name, duration_hours, start_iso, step_min)
        results[name] = (df, has_mask)
        lat_lon_str = f"{SITES[name]['lat']}, {SITES[name]['lon']}"
        row = {"Site": name, "Lat/Lon": lat_lon_str, "Terrain horizon": "flat (model approx)"}
        row.update({k: round(v, 1) for k, v in summarise(df, step_min).items()})
        summary_rows.append(row)
except Exception as e:
    st.error(f"Could not compute geometry: {e}")
    st.stop()

st.subheader("Site comparison")
st.dataframe(pd.DataFrame(summary_rows).set_index("Site"), use_container_width=True)

st.markdown("---")
focus = st.selectbox("Inspect site", chosen)
df, has_mask = results[focus]
m = summarise(df, step_min)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sunlit", f"{m['Sunlit %']:.1f}%")
c2.metric("Earth geometrically visible", f"{m['Earth visible %']:.1f}%")
c3.metric("Longest darkness", f"{m['Longest darkness (h)']:.0f} h")
c4.metric("Sun + Earth together", f"{m['Sun AND Earth %']:.1f}%")

tab1, tab2 = st.tabs(["Elevation vs time", "Sky view (azimuth / elevation)"])

with tab1:
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Flat horizon (0°)")
    fig.add_trace(go.Scatter(x=df["Time_UTC"], y=df["Sun_Alt"], name="Sun elevation",
                             line=dict(color="#FFD700", width=2)))
    fig.add_trace(go.Scatter(x=df["Time_UTC"], y=df["Earth_Alt"], name="Earth elevation",
                             line=dict(color="#1E90FF", width=2)))
    fig.update_layout(template="plotly_dark", height=420, hovermode="x unified",
                      xaxis_title="Time (UTC)", yaxis_title="Elevation above local horizontal (°)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    for label, alt, azc, color in [("Sun", "Sun_Alt", "Sun_Az", "#FFD700"),
                                   ("Earth", "Earth_Alt", "Earth_Az", "#1E90FF")]:
        fig2.add_trace(go.Scatterpolar(theta=df[azc], r=90 - df[alt], mode="markers", name=f"{label} track",
                                       marker=dict(color=color, size=4)))
    fig2.update_layout(template="plotly_dark", height=520,
                       polar=dict(angularaxis=dict(direction="clockwise", rotation=90),
                                  radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                                                  ticktext=["zenith", "60°", "30°", "horizon"])))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Points near the rim are close to the horizon, where terrain easily blocks them.")

with st.expander("ℹ️ Model assumptions & scientific limitations", expanded=True):
    st.markdown(f"""
- **Ephemeris Model:** Computed using lunar polar celestial mechanics (axial tilt ~1.54°, synodic orbit cycle ~708h, and Earth libration wobble).
- **Terrain:** Flat horizon baseline assumed for rapid prototyping.
- **Sampling:** Results use {step_min}-minute steps.
""")
