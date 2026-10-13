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

# ---------------------------------------------------------------------------
# 3D first-person viewer (Embedded directly to prevent missing file errors on Cloud)
# ---------------------------------------------------------------------------
def render_3d(df: pd.DataFrame, site_name: str, mask, has_mask: bool, step_min: int, height: int = 640):
    k = max(1, int(np.ceil(len(df) / 1500)))        # keep payload small
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

    html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html,body{margin:0;height:100%;background:#000;overflow:hidden;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#fff}
  #wrap{position:relative;width:100%;height:100%;touch-action:none;cursor:grab}
  #wrap:active{cursor:grabbing}
  canvas{display:block}
  .glass{background:rgba(10,14,28,.55);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.12);border-radius:12px}
  #hud{position:absolute;left:12px;top:12px;padding:10px 14px;font-size:13px;line-height:1.5;pointer-events:none}
  #hud b{font-size:15px}
  .chip{display:inline-block;margin:4px 6px 0 0;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
  .on{background:rgba(46,204,113,.25);color:#7dffb0;border:1px solid #2ecc71}
  .off{background:rgba(231,76,60,.25);color:#ff9d93;border:1px solid #e74c3c}
  #warn{position:absolute;right:12px;top:12px;max-width:260px;padding:8px 12px;font-size:12px;color:#ffd27d;display:none}
  #ctrl{position:absolute;left:12px;right:12px;bottom:12px;padding:8px 12px;display:flex;gap:10px;align-items:center;font-size:13px}
  #ctrl input[type=range]{flex:1;accent-color:#ffd54a}
  button,select{background:rgba(255,255,255,.1);color:#fff;border:1px solid rgba(255,255,255,.25);border-radius:8px;padding:5px 10px;font-size:13px;cursor:pointer}
  button:hover{background:rgba(255,255,255,.2)}
  #time{min-width:150px;font-variant-numeric:tabular-nums}
  .lbl{position:absolute;transform:translate(-50%,-50%);font-size:12px;font-weight:700;color:rgba(255,255,255,.85);text-shadow:0 0 4px #000;pointer-events:none}
  #foot{position:absolute;right:14px;bottom:62px;font-size:11px;opacity:.6;pointer-events:none}
</style>
</head>
<body>
<div id="wrap">
  <div id="hud" class="glass"></div>
  <div id="warn" class="glass"></div>
  <div id="ctrl" class="glass">
    <button id="play">⏸</button>
    <input id="slider" type="range" min="0" max="1" step="any" value="0">
    <span id="time"></span>
    <select id="speed">
      <option value="1">1 h/s</option><option value="6" selected>6 h/s</option>
      <option value="24">1 day/s</option><option value="72">3 days/s</option>
    </select>
    <button id="faceSun">☀ Sun</button><button id="faceEarth">🌍 Earth</button>
  </div>
  <div id="foot">Drag to look · scroll to zoom · Sun &amp; Earth at true angular size</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D = __DATA__;
const N = D.times.length, R = Math.PI / 180;
const wrap = document.getElementById('wrap');

function dirv(az, alt){const a=az*R,e=alt*R;return new THREE.Vector3(Math.cos(e)*Math.sin(a),Math.sin(e),-Math.cos(e)*Math.cos(a));}
function toAzAlt(v){return {az:(Math.atan2(v.x,-v.z)/R+360)%360, alt:Math.asin(Math.max(-1,Math.min(1,v.y)))/R};}

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMappingExposure = 0.9;
wrap.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 12000);
camera.position.set(0, 1.8, 0);
camera.rotation.order = 'YXZ';

function rnd(seed){let s=seed;return()=>{s=(s*16807)%2147483647;return(s-1)/2147483646;};}
function regolithTextures(){
  const S=1024, r=rnd(42);
  const c=document.createElement('canvas');c.width=c.height=S;const g=c.getContext('2d');
  g.fillStyle='#6e6c69';g.fillRect(0,0,S,S);
  for(let i=0;i<60000;i++){const v=80+r()*50|0;g.fillStyle=`rgba(${v},${v},${v-2},0.3)`;g.fillRect(r()*S,r()*S,1+r()*2,1+r()*2);}
  const b=document.createElement('canvas');b.width=b.height=S;const h=b.getContext('2d');
  h.fillStyle='#808080';h.fillRect(0,0,S,S);
  for(let i=0;i<260;i++){
    const x=r()*S,y=r()*S,rad=6+Math.pow(r(),3)*90;
    for(const ctx of [g,h]){
      const gr=ctx.createRadialGradient(x,y,rad*0.2,x,y,rad);
      if(ctx===h){gr.addColorStop(0,'rgba(105,105,105,0.8)');gr.addColorStop(0.8,'rgba(115,115,115,0.5)');gr.addColorStop(0.92,'rgba(150,150,150,0.7)');gr.addColorStop(1,'rgba(128,128,128,0)');}
      else{gr.addColorStop(0,'rgba(60,60,60,0.35)');gr.addColorStop(0.85,'rgba(80,80,80,0.2)');gr.addColorStop(0.93,'rgba(125,125,125,0.3)');gr.addColorStop(1,'rgba(110,110,110,0)');}
      ctx.fillStyle=gr;ctx.beginPath();ctx.arc(x,y,rad,0,7);ctx.fill();
    }
  }
  const mk=cv=>{const t=new THREE.CanvasTexture(cv);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(45,45);t.anisotropy=8;return t;};
  return {map:mk(c),bump:mk(b)};
}
function earthTexture(){
  const W=1024,H=512,r=rnd(7);
  const c=document.createElement('canvas');c.width=W;c.height=H;const g=c.getContext('2d');
  const o=g.createLinearGradient(0,0,0,H);o.addColorStop(0,'#1d4f91');o.addColorStop(.5,'#1b6bb5');o.addColorStop(1,'#1d4f91');
  g.fillStyle=o;g.fillRect(0,0,W,H);
  for(let i=0;i<70;i++){const x=r()*W,y=H*0.15+r()*H*0.7,rad=15+r()*55;
    g.fillStyle=r()<0.6?'#3f7d3a':'#b59b62';g.globalAlpha=.9;
    for(let k=0;k<7;k++){g.beginPath();g.arc(x+(r()-.5)*rad*1.5,y+(r()-.5)*rad,rad*(.4+r()*.5),0,7);g.fill();}}
  g.globalAlpha=1;g.fillStyle='#eef4f8';g.fillRect(0,0,W,22);g.fillRect(0,H-30,W,30);
  for(let i=0;i<110;i++){g.fillStyle=`rgba(255,255,255,${.18+r()*.3})`;g.beginPath();g.ellipse(r()*W,r()*H,20+r()*70,5+r()*16,0,0,7);g.fill();}
  return new THREE.CanvasTexture(c);
}
function glowTexture(stops){
  const c=document.createElement('canvas');c.width=c.height=256;const g=c.getContext('2d');
  const gr=g.createRadialGradient(128,128,0,128,128,128);stops.forEach(s=>gr.addColorStop(s[0],s[1]));
  g.fillStyle=gr;g.fillRect(0,0,256,256);return new THREE.CanvasTexture(c);
}

const tex = regolithTextures();
const ground = new THREE.Mesh(new THREE.CircleGeometry(6000,96),
  new THREE.MeshStandardMaterial({color:0x8a8a8a,map:tex.map,bumpMap:tex.bump,bumpScale:0.35,roughness:1,metalness:0}));
ground.rotation.x=-Math.PI/2; ground.receiveShadow=true; scene.add(ground);

const WALL=800;
if (D.horizon){
  const pos=[],idx=[];
  for(let i=0;i<=360;i++){
    const az=i*R, h=D.horizon[i%360]*R, top=WALL*Math.tan(h);
    const x=Math.sin(az)*WALL, z=-Math.cos(az)*WALL;
    pos.push(x,-80,z, x,top,z);
    if(i<360){const a=i*2;idx.push(a,a+1,a+2, a+1,a+3,a+2);}
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setIndex(idx);geo.computeVertexNormals();
  const wall=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0x4a4846,roughness:1,side:THREE.DoubleSide}));
  wall.receiveShadow=false;scene.add(wall);
  const lp=[];for(let i=0;i<=360;i++){const az=i*R;lp.push(Math.sin(az)*(WALL-1),WALL*Math.tan(D.horizon[i%360]*R),-Math.cos(az)*(WALL-1));}
  const lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.Float32BufferAttribute(lp,3));
  scene.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.35})));
}

(function(){
  const r=rnd(99),n=3500,p=new Float32Array(n*3),c=new Float32Array(n*3);
  for(let i=0;i<n;i++){
    const u=r()*2-1,t=r()*6.2832,s=Math.sqrt(1-u*u),b=.35+r()*.65;
    p.set([5000*s*Math.cos(t),5000*u,5000*s*Math.sin(t)],i*3);c.set([b,b,b*(.9+r()*.1)],i*3);
  }
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(p,3));g.setAttribute('color',new THREE.BufferAttribute(c,3));
  scene.add(new THREE.Points(g,new THREE.PointsMaterial({size:1.8,sizeAttenuation:false,vertexColors:true,toneMapped:false})));
})();

scene.add(new THREE.HemisphereLight(0x25324f,0x0d0d12,0.35));
const sunLight=new THREE.DirectionalLight(0xffffff,2.4);
sunLight.castShadow=true;sunLight.shadow.mapSize.set(4096,4096);
const sc=sunLight.shadow.camera;sc.left=-220;sc.right=220;sc.top=220;sc.bottom=-220;sc.near=1;sc.far=1200;
sunLight.shadow.bias=-0.0004;sunLight.shadow.normalBias=0.4;
scene.add(sunLight,sunLight.target);

const DIST=900;
const sunMesh=new THREE.Mesh(new THREE.SphereGeometry(DIST*Math.tan(0.2665*R),32,16),new THREE.MeshBasicMaterial({color:0xfff4d0,toneMapped:false}));
const sunGlow=new THREE.Sprite(new THREE.SpriteMaterial({map:glowTexture([[0,'rgba(255,240,200,1)'],[.12,'rgba(255,220,140,.7)'],[.4,'rgba(255,170,60,.18)'],[1,'rgba(255,150,40,0)']]),blending:THREE.AdditiveBlending,depthWrite:false,toneMapped:false}));
sunGlow.scale.set(130,130,1);
scene.add(sunMesh,sunGlow);

const earthR=DIST*Math.tan(0.95*R);
const sunUniform={value:new THREE.Vector3(1,0,0)};
const earth=new THREE.Mesh(new THREE.SphereGeometry(earthR,64,32),new THREE.ShaderMaterial({
  uniforms:{map:{value:earthTexture()},uSun:sunUniform},
  vertexShader:'varying vec2 vUv;varying vec3 vN;void main(){vUv=uv;vN=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
  fragmentShader:'uniform sampler2D map;uniform vec3 uSun;varying vec2 vUv;varying vec3 vN;void main(){vec3 c=texture2D(map,vUv).rgb;float l=max(dot(normalize(vN),normalize(uSun)),0.0);float t=smoothstep(0.0,0.25,l);gl_FragColor=vec4(c*(0.025+1.1*l),1.0);}'}));
const atmos=new THREE.Mesh(new THREE.SphereGeometry(earthR*1.07,48,24),new THREE.ShaderMaterial({
  uniforms:{uSun:sunUniform},transparent:true,side:THREE.BackSide,blending:THREE.AdditiveBlending,depthWrite:false,
  vertexShader:'varying vec3 vV;varying vec3 vW;void main(){vV=normalize(normalMatrix*normal);vW=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
  fragmentShader:'uniform vec3 uSun;varying vec3 vV;varying vec3 vW;void main(){float f=pow(0.75+dot(vV,vec3(0.,0.,1.)),3.0);float lit=smoothstep(-0.3,0.4,dot(vW,normalize(uSun)));gl_FragColor=vec4(0.35,0.62,1.0,1.0)*f*lit*0.9;}'}));
scene.add(earth,atmos);

(function(){
  const g=new THREE.Group(),m=new THREE.MeshStandardMaterial({color:0xd8d2c0,metalness:.6,roughness:.45}),
        gold=new THREE.MeshStandardMaterial({color:0xc9a227,metalness:.8,roughness:.35});
  const body=new THREE.Mesh(new THREE.CylinderGeometry(1.3,1.5,1.4,8),gold);body.position.y=2;
  const top=new THREE.Mesh(new THREE.CylinderGeometry(.9,1.3,.7,8),m);top.position.y=3.05;
  const dish=new THREE.Mesh(new THREE.SphereGeometry(.7,16,8,0,6.283,0,1.2),m);dish.position.set(.4,3.7,0);dish.rotation.z=-.5;
  g.add(body,top,dish);
  for(let i=0;i<4;i++){const a=i*Math.PI/2+Math.PI/4;
    const leg=new THREE.Mesh(new THREE.CylinderGeometry(.06,.06,2.4,6),m);
    leg.position.set(Math.cos(a)*1.5,1,Math.sin(a)*1.5);leg.rotation.z=Math.cos(a)*.5;leg.rotation.x=-Math.sin(a)*.5;
    const foot=new THREE.Mesh(new THREE.CylinderGeometry(.35,.4,.08,12),m);foot.position.set(Math.cos(a)*2.1,.04,Math.sin(a)*2.1);
    g.add(leg,foot);}
  g.traverse(o=>{if(o.isMesh)o.castShadow=true;});
  const d=dirv(25,0).multiplyScalar(16);g.position.set(d.x,0,d.z);scene.add(g);
})();

const labels=[['N',0],['NE',45],['E',90],['SE',135],['S',180],['SW',225],['W',270],['NW',315]].map(([t,az])=>{
  const el=document.createElement('div');el.className='lbl';el.textContent=t;wrap.appendChild(el);return{el,v:dirv(az,0).multiplyScalar(400)};});

let t=0,playing=true,yaw=D.sunAz[0],pitch=6,fov=60,last=performance.now();
const $=id=>document.getElementById(id);
const slider=$('slider');slider.max=N-1;
const stepH=D.stepMin/60;
if(!D.horizon){$('warn').style.display='block';$('warn').innerHTML='⚠ No terrain data for this site: flat horizon assumed. Sun/Earth visibility is an <b>upper bound</b>.';}

function sample(az,alt,t){const i=Math.floor(t),j=Math.min(i+1,N-1),f=t-i;
  return dirv(az[i],alt[i]).lerp(dirv(az[j],alt[j]),f).normalize();}

function update(dt){
  if(playing){t+=dt*Number($('speed').value)/stepH;if(t>=N-1)t=0;slider.value=t;}
  const i=Math.min(Math.floor(t+0.5),N-1);
  const sv=sample(D.sunAz,D.sunAlt,t),ev=sample(D.earthAz,D.earthAlt,t);
  sunMesh.position.copy(sv).multiplyScalar(DIST).add(camera.position);
  sunGlow.position.copy(sunMesh.position);
  earth.position.copy(ev).multiplyScalar(DIST).add(camera.position);atmos.position.copy(earth.position);
  earth.rotation.y=t*stepH*15*R;
  sunUniform.value.copy(sv);
  sunLight.position.copy(sv).multiplyScalar(500);sunLight.target.position.set(0,0,0);
  const target=D.sunlit[i]?2.4:0;sunLight.intensity+=(target-sunLight.intensity)*Math.min(1,dt*8);
  const a=toAzAlt(sv),b=toAzAlt(ev);
  $('time').textContent=D.times[i];$('hud').innerHTML=`<b>${D.site}</b><br>Sun &nbsp;az ${a.az.toFixed(1)}° &nbsp;el ${a.alt.toFixed(2)}°<br>Earth az ${b.az.toFixed(1)}° &nbsp;el ${b.alt.toFixed(2)}°<br>`+
    `<span class="chip ${D.sunlit[i]?'on':'off'}">☀ ${D.sunlit[i]?'Solar power available':'No direct sunlight'}</span>`+
    `<span class="chip ${D.earthVis[i]?'on':'off'}">🌍 ${D.earthVis[i]?'DTE line-of-sight':'No Earth line-of-sight'}</span>`;
}

function resize(){const w=wrap.clientWidth||innerWidth,h=wrap.clientHeight||innerHeight;renderer.setSize(w,h,false);
  renderer.domElement.style.width=w+'px';renderer.domElement.style.height=h+'px';camera.aspect=w/h;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe(wrap);resize();

function frame(now){
  const dt=Math.min(0.1,(now-last)/1000);last=now;update(dt);
  camera.rotation.y=-yaw*R;camera.rotation.x=pitch*R;camera.fov=fov;camera.updateProjectionMatrix();
  renderer.render(scene,camera);
  const w=wrap.clientWidth,h=wrap.clientHeight;
  labels.forEach(l=>{const p=l.v.clone().add(camera.position).project(camera);
    if(p.z<1&&Math.abs(p.x)<1.05){l.el.style.display='block';l.el.style.left=((p.x+1)/2*w)+'px';l.el.style.top=((1-p.y)/2*h)+'px';}else l.el.style.display='none';});
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

let drag=null;
wrap.addEventListener('pointerdown',e=>{if(e.target.closest('#ctrl'))return;drag={x:e.clientX,y:e.clientY};wrap.setPointerCapture(e.pointerId);});
wrap.addEventListener('pointermove',e=>{if(!drag)return;const k=fov/60*0.12;yaw-=(e.clientX-drag.x)*k;pitch=Math.max(-30,Math.min(80,pitch+(e.clientY-drag.y)*k));drag={x:e.clientX,y:e.clientY};});
wrap.addEventListener('pointerup',()=>drag=null);
wrap.addEventListener('wheel',e=>{e.preventDefault();fov=Math.max(15,Math.min(90,fov+e.deltaY*0.03));},{passive:false});
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'⏸':'▶';};
slider.oninput=()=>{t=Number(slider.value);};
$('faceSun').onclick=()=>{const a=toAzAlt(sample(D.sunAz,D.sunAlt,t));yaw=a.az;pitch=Math.max(0,a.alt)+3;};$('faceEarth').onclick=()=>{const a=toAzAlt(sample(D.earthAz,D.earthAlt,t));yaw=a.az;pitch=Math.max(0,a.alt)+3;};
</script>
</body>
</html>
"""

    html = html_template.replace("__DATA__", json.dumps(payload))
    components.html(html, height=height, scrolling=False)
