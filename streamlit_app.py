"""
Lunar Mission Browser - South Pole  (NASA Space Apps CLPS challenge)

What is REAL here:
  * Sun and Earth elevation/azimuth at a lunar surface point, computed from the
    JPL DE421 ephemeris and the lunar principal-axis (libration) frame via Skyfield.

What is NOT yet real (and the app says so on screen):
  * Terrain shadowing. Without a horizon mask for a site, the horizon is assumed
    flat (0 deg), which OVERESTIMATES sunlight and Earth visibility.
    Drop a CSV at horizons/<site_slug>.csv (columns: azimuth_deg, horizon_elev_deg)
    and the app will use it automatically.

Azimuth convention: degrees clockwise from local north, as returned by Skyfield's
altaz() for the Moon-fixed frame. Horizon-mask CSVs must use the same convention.
"""
import json
import os
import re
from datetime import datetime, date, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from skyfield.api import Loader
from skyfield.planetarylib import PlanetaryConstants

st.set_page_config(page_title="Lunar Mission Browser - South Pole", page_icon="🌕", layout="wide")

HORIZON_DIR = "horizons"
KERNEL_DIR = "kernels"

# ---------------------------------------------------------------------------
# Candidate sites. Coordinates are APPROXIMATE - verify each against a
# documented source (e.g. USGS Gazetteer / LROC) before submission.
# Longitude is east-positive, -180..180.
# ---------------------------------------------------------------------------
SITES = {
    "Shackleton Crater Rim": {"lat": -89.9, "lon": 129.8},
    "de Gerlache Rim": {"lat": -88.5, "lon": -87.1},
    "Malapert Massif": {"lat": -84.9, "lon": 12.9},
    "Haworth Crater Floor": {"lat": -87.4, "lon": -5.1},
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Ephemeris
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading JPL ephemeris (first run downloads ~20 MB)...")
def load_kernels():
    loader = Loader(KERNEL_DIR)
    ts = loader.timescale()
    eph = loader("de421.bsp")
    pc = PlanetaryConstants()
    pc.read_text(loader("moon_080317.tf"))
    pc.read_text(loader("pck00008.tpc"))
    pc.read_binary(loader("moon_pa_de421_1900-2050.bpc"))  # valid 1900-2050
    frame = pc.build_frame_named("MOON_ME_DE421")
    return ts, eph, pc, frame


@st.cache_data(show_spinner="Computing Sun and Earth geometry...")
def compute_geometry(lat: float, lon: float, start_iso: str, hours: int, step_min: int) -> pd.DataFrame:
    ts, eph, pc, frame = load_kernels()
    start = datetime.fromisoformat(start_iso)
    t0 = ts.utc(start.year, start.month, start.day, start.hour, start.minute)
    n = int(hours * 60 / step_min)
    t = ts.tt_jd(t0.tt + np.arange(n) * step_min / 1440.0)

    place = eph["moon"] + pc.build_latlon_degrees(frame, lat, lon)
    obs = place.at(t)
    sun_alt, sun_az, _ = obs.observe(eph["sun"]).apparent().altaz()
    earth_alt, earth_az, _ = obs.observe(eph["earth"]).apparent().altaz()

    return pd.DataFrame({
        "Time_UTC": pd.to_datetime(t.utc_datetime()).tz_localize(None),
        "Sun_Alt": sun_alt.degrees,
        "Sun_Az": sun_az.degrees,
        "Earth_Alt": earth_alt.degrees,
        "Earth_Az": earth_az.degrees,
    })


# ---------------------------------------------------------------------------
# Horizon masks
# ---------------------------------------------------------------------------
def load_horizon(site_name: str):
    path = os.path.join(HORIZON_DIR, f"{slug(site_name)}.csv")
    if not os.path.exists(path):
        return None
    h = pd.read_csv(path).sort_values("azimuth_deg")
    return h["azimuth_deg"].to_numpy(float), h["horizon_elev_deg"].to_numpy(float)


def horizon_at(mask, az):
    if mask is None:
        return np.zeros_like(az)
    return np.interp(az, mask[0], mask[1], period=360.0)


def analyse(site_name: str, hours: int, start_iso: str, step_min: int):
    s = SITES[site_name]
    df = compute_geometry(s["lat"], s["lon"], start_iso, hours, step_min)
    mask = load_horizon(site_name)
    df["Sun_Horizon"] = horizon_at(mask, df["Sun_Az"].to_numpy())
    df["Earth_Horizon"] = horizon_at(mask, df["Earth_Az"].to_numpy())
    df["Sunlit"] = df["Sun_Alt"] > df["Sun_Horizon"]
    df["Earth_Visible"] = df["Earth_Alt"] > df["Earth_Horizon"]
    return df, mask is not None


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
# First-person panorama (what an observer standing at the site sees)
# ---------------------------------------------------------------------------
def _break_wrap(az, alt):
    """Insert gaps where azimuth wraps 360->0 so lines don't streak across the plot."""
    xs, ys = [], []
    for k in range(len(az)):
        if k > 0 and abs(az[k] - az[k - 1]) > 180:
            xs.append(None)
            ys.append(None)
        xs.append(float(az[k]))
        ys.append(float(alt[k]))
    return xs, ys


def panorama_figure(df: pd.DataFrame, i: int, mask, has_mask: bool) -> go.Figure:
    row = df.iloc[i]
    lo = float(min(-4, np.floor(min(df["Sun_Alt"].min(), df["Earth_Alt"].min())) - 1))
    hi = float(max(12, np.ceil(max(df["Sun_Alt"].max(), df["Earth_Alt"].max())) + 2))
    az = np.arange(0, 361, 1.0)
    hz = horizon_at(mask, az)

    fig = go.Figure()
    # Ground: filled from the bottom of the plot up to the terrain skyline
    fig.add_trace(go.Scatter(x=az, y=np.full_like(az, lo), mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=az, y=hz, mode="lines", fill="tonexty", name="Terrain skyline" if has_mask else "Flat horizon (no terrain data)",
                             line=dict(color="#9a9a9a", width=2), fillcolor="rgba(120,120,120,0.55)"))
    if has_mask:
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")

    # Faint tracks over the whole window
    for label, azc, altc, color in [("Sun track", "Sun_Az", "Sun_Alt", "rgba(255,215,0,0.35)"),
                                    ("Earth track", "Earth_Az", "Earth_Alt", "rgba(30,144,255,0.45)")]:
        xs, ys = _break_wrap(df[azc].to_numpy(), df[altc].to_numpy())
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, line=dict(color=color, width=2)))

    # Current positions
    for label, azc, altc, color, size in [("Sun", "Sun_Az", "Sun_Alt", "#FFD700", 34),
                                          ("Earth", "Earth_Az", "Earth_Alt", "#1E90FF", 24)]:
        a, e = float(row[azc]), float(row[altc])
        visible = e > float(horizon_at(mask, np.array([a]))[0])
        fig.add_trace(go.Scatter(
            x=[a], y=[e], mode="markers+text", name=f"{label} now",
            text=[label if visible else f"{label} (blocked)"], textposition="top center",
            textfont=dict(color="white"),
            marker=dict(size=size, color=color, opacity=1.0 if visible else 0.3,
                        line=dict(color="white", width=2 if visible else 1)),
            hovertemplate=f"{label}<br>Az %{{x:.1f}}°<br>El %{{y:.2f}}°<extra></extra>"))

    fig.update_layout(
        template="plotly_dark", height=430, plot_bgcolor="#03050c",
        xaxis=dict(range=[0, 360], tickvals=[0, 90, 180, 270, 360], ticktext=["N", "E", "S", "W", "N"],
                   title="Direction (azimuth)"),
        yaxis=dict(range=[lo, hi], title="Elevation (°)"),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------------------
# 3D first-person viewer (Three.js lives in viewer3d.html; Python only feeds it data)
# ---------------------------------------------------------------------------
def render_3d(df: pd.DataFrame, site_name: str, mask, has_mask: bool, step_min: int, height: int = 640):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer3d.html")
    if not os.path.exists(path):
        st.error("viewer3d.html not found next to this file.")
        return
    k = max(1, int(np.ceil(len(df) / 1500)))          # keep payload small
    d = df.iloc[::k]
    r = lambda a, n=3: [round(float(x), n) for x in a]
    payload = {
        "site": site_name,
        "stepMin": step_min * k,
        "times": [f"{x:%Y-%m-%d %H:%M} UTC" for x in d["Time_UTC"]],
        "sunAz": r(d["Sun_Az"]), "sunAlt": r(d["Sun_Alt"]),
        "earthAz": r(d["Earth_Az"]), "earthAlt": r(d["Earth_Alt"]),
        "sunlit": [int(x) for x in d["Sunlit"]],
        "earthVis": [int(x) for x in d["Earth_Visible"]],
        "horizon": r(horizon_at(mask, np.arange(360, dtype=float)), 2) if has_mask else None,
    }
    html = open(path, encoding="utf-8").read().replace("__DATA__", json.dumps(payload))
    components.html(html, height=height, scrolling=False)

# ---------------------------------------------------------------------------
# UI
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
        row = {"Site": name, "Lat/Lon": f"{SITES[name]['lat']}, {SITES[name]['lon']}",
               "Terrain horizon": "mask loaded" if has_mask else "flat (optimistic)"}
        row.update({k: round(v, 1) for k, v in summarise(df, step_min).items()})
        summary_rows.append(row)
except Exception as e:  # kernel download blocked, date out of kernel range, etc.
    st.error(f"Could not compute geometry: {e}")
    st.stop()

if any(not m for _, m in results.values()):
    st.warning(
        "Some sites have **no terrain horizon mask**, so a flat 0° horizon is assumed. "
        "Sunlight and Earth-visibility values for those sites are **upper bounds**, not predictions."
    )

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

tab3d, tab1, tab2, tab3 = st.tabs(["🌑 3D surface view", "Elevation vs time", "Sky view (azimuth / elevation)", "2D panorama"])

with tab3d:
    st.caption("Interactive 3D view from the landing site. Sun and Earth positions come from the ephemeris; "
               "the ground texture and lander are decorative.")
    render_3d(df, focus, load_horizon(focus), has_mask, step_min)

with tab1:
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Flat horizon (0°)")
    fig.add_trace(go.Scatter(x=df["Time_UTC"], y=df["Sun_Alt"], name="Sun elevation",
                             line=dict(color="#FFD700", width=2)))
    fig.add_trace(go.Scatter(x=df["Time_UTC"], y=df["Earth_Alt"], name="Earth elevation",
                             line=dict(color="#1E90FF", width=2)))
    if has_mask:
        fig.add_trace(go.Scatter(x=df["Time_UTC"], y=df["Sun_Horizon"], name="Terrain horizon toward Sun",
                                 line=dict(color="#FFD700", dash="dot")))
        fig.add_trace(go.Scatter(x=df["Time_UTC"], y=df["Earth_Horizon"], name="Terrain horizon toward Earth",
                                 line=dict(color="#1E90FF", dash="dot")))
    fig.update_layout(template="plotly_dark", height=420, hovermode="x unified",
                      xaxis_title="Time (UTC)", yaxis_title="Elevation above local horizontal (°)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Polar sky plot: centre = zenith, edge = horizon.
    fig2 = go.Figure()
    if has_mask:
        az = np.arange(0, 361, 2)
        hz = horizon_at(load_horizon(focus), az.astype(float))
        fig2.add_trace(go.Scatterpolar(theta=az, r=90 - hz, name="Terrain horizon",
                                       fill="toself", line=dict(color="gray")))
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

with tab3:
    st.markdown("What an observer standing at this site would see looking around the horizon. "
                "Drag the slider to move through time.")
    idx = st.slider("Time step", 0, len(df) - 1, 0, key="pano_idx")
    row = df.iloc[idx]
    st.markdown(f"**{row['Time_UTC']:%Y-%m-%d %H:%M} UTC**")
    mask_now = load_horizon(focus)
    st.plotly_chart(panorama_figure(df, idx, mask_now, has_mask), use_container_width=True)
    p1, p2 = st.columns(2)
    if row["Sunlit"]:
        p1.success(f"☀️ Sun above terrain horizon (elevation {row['Sun_Alt']:.2f}°) - solar power available")
    else:
        p1.error(f"☀️ Sun not visible (elevation {row['Sun_Alt']:.2f}°) - no direct solar power")
    if row["Earth_Visible"]:
        p2.success(f"🌍 Earth above terrain horizon (elevation {row['Earth_Alt']:.2f}°) - DTE geometrically possible")
    else:
        p2.error(f"🌍 Earth not visible (elevation {row['Earth_Alt']:.2f}°) - no direct-to-Earth line of sight")
    if not has_mask:
        st.caption("No terrain data for this site yet: the ground line is a flat 0° horizon, so the "
                   "Sun and Earth appear unobstructed whenever they are above it.")

with st.expander("ℹ️ Model assumptions & scientific limitations", expanded=True):
    st.markdown(f"""
- **Sun/Earth positions:** computed from the JPL DE421 ephemeris with the lunar principal-axis frame
  (Skyfield). Kernel coverage: 1900-2050. Sites are treated as points on a spherical Moon surface.
- **Terrain:** {"a horizon mask from `horizons/` is applied for this site." if has_mask else
  "**no terrain data is applied for this site**; a flat horizon is assumed, which overestimates sunlight and Earth visibility. A real result needs a DEM-derived horizon (e.g. from LOLA)."}
- **Earth visibility is geometric only.** Earth counts as visible when its centre is above the horizon;
  real links also depend on antenna, link budget, relays and ground stations.
- **Sampling:** results use {step_min}-minute steps; short events can be missed.
- **Not modelled:** observer height (rover mast/antenna), Sun's finite disc, thermal conditions, ice presence.
- Site coordinates are approximate and should be verified against a cited source.
""")

