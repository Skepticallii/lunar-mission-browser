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
import base64
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
TERRAIN_DIR = "terrain"
KERNEL_DIR = "kernels"
EARTH_RADIUS_KM = 6378.137

# Earth is ~1.9 deg wide in the lunar sky (radius ~0.95 deg), so "visible" needs a definition.
EARTH_RULES = {
    "Any part of Earth's disc (NASA definition; used by Model 2)": 1.0,
    "Centre of Earth above horizon": 0.0,
    "Whole Earth disc above horizon": -1.0,
}

# ---------------------------------------------------------------------------
# Candidate sites and coordinates as given in the team's site_comparison.csv.
# Longitude is east-positive, -180..180.
# ---------------------------------------------------------------------------
SITES = {
    "LM1 - Shackleton Rim B": {"id": "LM1", "lat": -89.5244, "lon": 56.30993, "elev": -27.784672, "terrain": "lm1"},
    "Site06 - Nobile Rim 1": {"id": "Site06", "lat": -85.4381, "lon": 37.36667, "elev": 5946.128418, "terrain": "site06"},
    "Site20 - Leibnitz Beta Plateau": {"id": "Site20", "lat": -85.4266, "lon": 31.74276, "elev": 6398.174805, "terrain": "site20"},
    "Site23 - Malapert Massif": {"id": "Site23", "lat": -85.9948, "lon": -0.23578, "elev": 4783.629395, "terrain": "site23"},
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
def compute_geometry_n(lat: float, lon: float, elev_m: float, start_iso: str, n: int, step_min: float) -> pd.DataFrame:
    ts, eph, pc, frame = load_kernels()
    start = datetime.fromisoformat(start_iso)
    t0 = ts.utc(start.year, start.month, start.day, start.hour, start.minute, start.second)
    t = ts.tt_jd(t0.tt + np.arange(n) * step_min / 1440.0)

    place = eph["moon"] + pc.build_latlon_degrees(frame, lat, lon, elevation_m=elev_m)
    obs = place.at(t)
    sun_alt, sun_az, _ = obs.observe(eph["sun"]).apparent().altaz()
    earth_alt, earth_az, earth_dist = obs.observe(eph["earth"]).apparent().altaz()
    radius = np.degrees(np.arcsin(np.minimum(1.0, EARTH_RADIUS_KM / earth_dist.km)))

    return pd.DataFrame({
        "Time_UTC": pd.to_datetime(t.utc_datetime()).tz_localize(None),
        "Sun_Alt": sun_alt.degrees,
        "Sun_Az": sun_az.degrees,
        "Earth_Alt": earth_alt.degrees,
        "Earth_Az": earth_az.degrees,
        "Earth_Radius": radius,
    })


def compute_geometry(lat: float, lon: float, start_iso: str, hours: int, step_min: int, elev_m: float = 0.0) -> pd.DataFrame:
    return compute_geometry_n(lat, lon, elev_m, start_iso, int(hours * 60 / step_min), step_min)


# ---------------------------------------------------------------------------
# Horizon masks
# ---------------------------------------------------------------------------
HORIZON_SEARCH = [HORIZON_DIR, "data", "data/dte", "Model_1_Solar_Availability/outputs", "Model_2_DTE/outputs/dte",
                  "Model_2_DTE/outputs", "Model_3_Terrain/outputs", "Model_3_Terrain", "LOLA_Data", "."]


def find_file(fname: str):
    for d in HORIZON_SEARCH:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            return path
    return None


@st.cache_data
def _profile_ids(path: str):
    return set(pd.read_csv(path, usecols=["site_id"])["site_id"].unique())


def horizon_path(site_name: str):
    """Where this site's horizon comes from: the combined horizon_profiles.csv, else <id>_horizon.csv."""
    sid = SITES[site_name]["id"]
    combined = find_file("horizon_profiles.csv")
    if combined and sid in _profile_ids(combined):
        return combined
    return find_file(f"{sid.lower()}_horizon.csv")


def load_horizon(site_name: str):
    """Return (azimuth_deg, horizon_elev_deg) arrays, or None if the site has no mask."""
    path = horizon_path(site_name)
    if path is None:
        return None
    h = pd.read_csv(path)
    if "site_id" in h.columns:
        h = h[h["site_id"] == SITES[site_name]["id"]]
    col = "horizon_elevation_deg" if "horizon_elevation_deg" in h.columns else "horizon_elev_deg"
    h = h.sort_values("azimuth_deg")
    az, el = h["azimuth_deg"].to_numpy(float), h[col].to_numpy(float)
    if len(az) < 8 or not np.all(np.isfinite(el)):
        raise ValueError(f"{path}: horizon data must be finite with at least 8 azimuths")
    return az, el


def horizon_at(mask, az):
    if mask is None:
        return np.zeros_like(az)
    return np.interp(az, mask[0], mask[1], period=360.0)


def analyse_df(df: pd.DataFrame, site_name: str, earth_rule: str):
    mask = load_horizon(site_name)
    df["Sun_Horizon"] = horizon_at(mask, df["Sun_Az"].to_numpy())
    df["Earth_Horizon"] = horizon_at(mask, df["Earth_Az"].to_numpy())
    df["Sunlit"] = df["Sun_Alt"] > df["Sun_Horizon"]
    k = EARTH_RULES[earth_rule]                       # +1 any part of the disc, 0 centre, -1 whole disc
    df["Earth_Visible"] = (df["Earth_Alt"] + k * df["Earth_Radius"]) > df["Earth_Horizon"]
    return df, mask is not None


def analyse(site_name: str, hours: int, start_iso: str, step_min: int, earth_rule: str):
    s = SITES[site_name]
    df = compute_geometry(s["lat"], s["lon"], start_iso, hours, step_min, s.get("elev", 0.0))
    return analyse_df(df, site_name, earth_rule)


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
# Models 2 and 3: the team's CSV outputs
# ---------------------------------------------------------------------------
@st.cache_data
def load_dte(site_id: str):
    """Model 2 timeline for one site (<ID>_dte_timeline.csv), or None."""
    path = find_file(f"{site_id}_dte_timeline.csv")
    if path is None:
        return None
    d = pd.read_csv(path)
    t = d["timestamp_utc"].astype(str).str.strip().str.replace(r"\s*UTC$", "", regex=True).str.rstrip("Z").str.replace("T", " ")
    d["Time_UTC"] = pd.to_datetime(t)
    return d


@st.cache_data
def load_model3():
    path = find_file("model3_site_comparison.csv")
    return pd.read_csv(path).set_index("site_id") if path else None


def runs_table(times, flags, step_min: float, labels=("visible", "blackout")) -> pd.DataFrame:
    """Consecutive runs of True/False as a table of windows."""
    flags = np.asarray(flags).astype(bool)
    if len(flags) == 0:
        return pd.DataFrame(columns=["state", "start_utc", "end_utc", "hours"])
    change = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(flags)]
    times = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    step = pd.Timedelta(minutes=step_min)
    return pd.DataFrame({
        "state": [labels[0] if flags[i] else labels[1] for i in starts],
        "start_utc": [times[i] for i in starts],
        "end_utc": [times[j - 1] + step for j in ends],
        "hours": (ends - starts) * step_min / 60.0,
    })


@st.cache_data(show_spinner="Building the Model 1-3 scorecard...")
def build_scorecard(earth_rule: str):
    """One row per site: Model 1 (sun), Model 2 (DTE) and Model 3 (terrain) on a common time grid."""
    m3, rows, window = load_model3(), [], ""
    for name, s in SITES.items():
        dte = load_dte(s["id"])
        if dte is not None and len(dte) > 2:
            n, step = len(dte), float((dte["Time_UTC"].iloc[1] - dte["Time_UTC"].iloc[0]).total_seconds() / 60)
            start, source = dte["Time_UTC"].iloc[0].isoformat(), "Model 2 CSV (disc rule)"
        else:
            n, step, start = 12824, 10.0, "2026-10-15T00:00:00"
            source = "live (" + earth_rule.split(" (")[0].lower() + ")"
        g = compute_geometry_n(s["lat"], s["lon"], s.get("elev", 0.0), start, n, step)
        mask = load_horizon(name)
        sun = (g["Sun_Alt"] > horizon_at(mask, g["Sun_Az"].to_numpy())).to_numpy()
        if dte is not None and len(dte) > 2:
            earth = dte["earth_visible"].astype(bool).to_numpy()
        else:
            k = EARTH_RULES[earth_rule]
            earth = ((g["Earth_Alt"] + k * g["Earth_Radius"]) > horizon_at(mask, g["Earth_Az"].to_numpy())).to_numpy()
        window = f"{g['Time_UTC'].iloc[0]:%Y-%m-%d} to {g['Time_UTC'].iloc[-1]:%Y-%m-%d} UTC, {step:g}-minute steps ({n} samples)"
        row = {
            "Site": name,
            "Sunlit %": round(100 * sun.mean(), 1),
            "Longest sunlight (h)": round(longest_run_hours(sun, step), 1),
            "Longest darkness (h)": round(longest_run_hours(~sun, step), 1),
            "Earth visible %": round(100 * earth.mean(), 1),
            "Longest DTE blackout (h)": round(longest_run_hours(~earth, step), 1),
            "Sun AND Earth %": round(100 * (sun & earth).mean(), 1),
            "Earth source": source,
            "Horizon": "mask" if mask is not None else "flat (optimistic)",
        }
        if m3 is not None and s["id"] in m3.index:
            r = m3.loc[s["id"]]
            row.update({"Slope <= 5° (%)": round(float(r["pct_slope_le_5deg"]), 1),
                        "Slope <= 10° (%)": round(float(r["pct_slope_le_10deg"]), 1),
                        "Flat circle <= 10° (m)": round(float(r["largest_flat_circle_le_10deg_diameter_m"]), 0),
                        "Relief 1 km (m)": round(float(r["relief_m"]), 0),
                        "Terrain score (provisional)": round(float(r["terrain_score_provisional"]), 1),
                        "Terrain rank (provisional)": int(r["terrain_rank_provisional"])})
        rows.append(row)
    return pd.DataFrame(rows), window


# ---------------------------------------------------------------------------
# Terrain products (made by preprocessing/make_terrain_products.py)
# ---------------------------------------------------------------------------
@st.cache_data
def load_terrain(name: str):
    npz = os.path.join(TERRAIN_DIR, f"{name}_terrain.npz")
    js = os.path.join(TERRAIN_DIR, f"{name}_terrain.json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        return None
    d = np.load(npz)
    with open(js) as f:
        meta = json.load(f)
    h = d["height_m"].astype("float32")
    h = np.nan_to_num(h, nan=float(np.nanmean(h)))
    lon = np.radians(meta["lon"])
    derived = "north_xy" not in meta
    # Local north/east as unit vectors in the DEM's map axes (x right, y up).
    # Standard south-polar stereographic: north=(sin lon, cos lon), east=(cos lon, -sin lon).
    north = meta.get("north_xy", [float(np.sin(lon)), float(np.cos(lon))])
    east = meta.get("east_xy", [float(np.cos(lon)), float(-np.sin(lon))])
    return {"height": h, "slope": d["slope_deg"].astype("float32"), "meta": meta,
            "north": north, "east": east, "orientation_derived": derived}


def hillshade(h: np.ndarray, mpp: float, az_deg: float = 315.0, alt_deg: float = 35.0) -> np.ndarray:
    gy, gx = np.gradient(h.astype("float64"), mpp)       # rows run downward, so dz/dy(up) = -gy
    n = np.dstack([-gx, gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    light = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    return np.clip(n @ light, 0, 1)

def terrain_skyline(tr) -> list:
    """Skyline (deg) from the terrain cutout alone. Only sees out to the cutout edge (~2.5 km)."""
    h, mpp = tr["height"], tr["meta"]["metres_per_pixel"]
    rows, cols = h.shape
    east, north = tr["east"], tr["north"]
    ds = np.arange(mpp, min(rows, cols) / 2 * mpp - mpp, mpp)
    z0 = float(h[rows // 2, cols // 2]) + 1.8
    out = []
    for az in range(360):
        a = np.radians(az)
        e, n = np.sin(a) * ds, np.cos(a) * ds
        dx, dy = e * east[0] + n * north[0], e * east[1] + n * north[1]
        c = np.rint(dx / mpp + cols / 2).astype(int)
        r = np.rint(-dy / mpp + rows / 2).astype(int)
        ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        ang = np.degrees(np.arctan2(h[r[ok], c[ok]] - z0 - ds[ok] ** 2 / (2 * 1_737_400.0), ds[ok]))
        out.append(round(float(ang.max()), 2) if ang.size else 0.0)
    return out


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
def render_3d(df: pd.DataFrame, site_name: str, mask, has_mask: bool, step_min: int, tr=None, height: int = 640):
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
        "horizon": (r(horizon_at(mask, np.arange(360, dtype=float)), 2) if has_mask
                    else (terrain_skyline(tr) if tr is not None else None)),
        "horizonSource": "mask" if has_mask else ("terrain" if tr is not None else None),
        "terrain": None,
    }
    if tr is not None:
        hgt = tr["height"]
        scale = float(max(np.abs(hgt).max(), 1.0) / 32000.0)
        ints = np.round(hgt / scale).astype("<i2")
        payload["terrain"] = {"rows": int(hgt.shape[0]), "cols": int(hgt.shape[1]),
                              "mpp": float(tr["meta"]["metres_per_pixel"]), "scale": scale,
                              "north": tr["north"], "east": tr["east"],
                              "b64": base64.b64encode(ints.tobytes()).decode("ascii")}
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
start_date = st.sidebar.date_input("Start date (UTC)", value=date(2026, 10, 15),
                                   min_value=date(1990, 1, 1), max_value=date(2049, 12, 31))
start_hour = st.sidebar.slider("Start hour (UTC)", 0, 23, 0)
duration_hours = st.sidebar.slider("Window (hours)", 24, 2160, 2136, step=24,
                                   help="~708 h is one lunar day-night cycle (synodic month, ~29.5 d).")
step_min = st.sidebar.select_slider("Time step (minutes)", options=[10, 15, 30, 60], value=60,
                                    help="Coarse steps can miss short illumination or darkness intervals.")

earth_rule = st.sidebar.radio(
    "Earth visibility rule", list(EARTH_RULES), index=0,
    help="Earth is about 1.9 degrees wide in the lunar sky. Model 2 and NASA's Earth-visibility maps count Earth as visible "
         "when ANY part of its disc is above the terrain horizon, which is optimistic for real radio links. "
         "'Centre' and 'whole disc' are stricter.")

if not chosen:
    st.info("Pick at least one site in the sidebar.")
    st.stop()

start_iso = datetime.combine(start_date, time(start_hour)).isoformat()

results, summary_rows = {}, []
try:
    for name in chosen:
        df, has_mask = analyse(name, duration_hours, start_iso, step_min, earth_rule)
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
table = pd.DataFrame(summary_rows)
team_path = find_file("site_comparison.csv")
if team_path and start_iso == "2026-10-15T00:00:00" and duration_hours == 2136 and step_min == 60:
    team = pd.read_csv(team_path).set_index("site_id")
    ids = [SITES[n]["id"] for n in table["Site"]]
    table["Model 1 CSV sunlit %"] = [round(float(team.loc[i, "sun_visible_sample_percent"]), 1) if i in team.index else np.nan for i in ids]
    notes = ["Model 1 CSV sunlit % (SPICE, hourly) should agree with the app to within ~0.1 points."]
    if earth_rule == list(EARTH_RULES)[0]:
        table["Model 2 CSV Earth %"] = [round(100 * float(load_dte(i)["earth_visible"].mean()), 1) if load_dte(i) is not None else np.nan for i in ids]
        notes.append("Model 2 CSV Earth % uses the same disc rule and should agree to within ~0.2 points.")
    else:
        notes.append("Choose the 'any part of Earth's disc' rule in the sidebar to compare Earth % with Model 2.")
    st.caption(" ".join(notes))
st.dataframe(table.set_index("Site"), use_container_width=True)

st.markdown("---")
focus = st.selectbox("Inspect site", chosen)
df, has_mask = results[focus]
tr = load_terrain(SITES[focus]['terrain']) if 'terrain' in SITES[focus] else None
m = summarise(df, step_min)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sunlit", f"{m['Sunlit %']:.1f}%")
c2.metric("Earth geometrically visible", f"{m['Earth visible %']:.1f}%")
c3.metric("Longest darkness", f"{m['Longest darkness (h)']:.0f} h")
c4.metric("Sun + Earth together", f"{m['Sun AND Earth %']:.1f}%")

tab3d, tabS, tab1, tab2, tab3, tabD, tabT = st.tabs(["🌑 3D surface view", "📊 Scorecard (Models 1-3)", "Elevation vs time",
                                                    "Sky view (azimuth / elevation)", "2D panorama",
                                                    "📡 DTE windows (Model 2)", "⛰ Terrain suitability (Model 3)"])

with tab3d:
    if tr is not None:
        st.caption("Interactive 3D view from the landing site. Terrain relief is real LOLA data (10 m grid, "
                   f"{tr['height'].shape[1] * tr['meta']['metres_per_pixel'] / 1000:.1f} km square); beyond that the skyline "
                   "comes from the horizon mask. Sun/Earth positions come from the ephemeris; surface texture and the lander are decorative.")
    else:
        st.caption("Interactive 3D view from the landing site. Sun and Earth positions come from the ephemeris; "
                   "the ground is a flat decorative plane (no terrain product for this site yet).")
    want_terrain = "terrain" in SITES[focus]
    st.caption(" · ".join([
        f"**{focus}**",
        "terrain: ✅ loaded" if tr is not None else ("terrain: ❌ file not found in terrain/" if want_terrain
                                                      else "terrain: none for this site (decorative flat ground)"),
        f"horizon mask: ✅ {horizon_path(focus)}" if has_mask else "horizon mask: ❌ not found (flat 0° horizon)",
    ]))
    render_3d(df, focus, load_horizon(focus), has_mask, step_min, tr)

with tabS:
    sc, window_txt = build_scorecard(earth_rule)
    st.markdown("All three models side by side, for **every** candidate site, on one common time window.")
    st.dataframe(sc.set_index("Site"), use_container_width=True)
    best = [("Most sunlight", "Sunlit %", True), ("Most Earth visibility", "Earth visible %", True),
            ("Most time with Sun and Earth together", "Sun AND Earth %", True), ("Shortest DTE blackout", "Longest DTE blackout (h)", False)]
    if "Slope <= 10° (%)" in sc.columns:
        best.append(("Most gentle terrain (share of area <= 10°)", "Slope <= 10° (%)", True))
    lines = []
    for label, col, high in best:
        j = sc[col].idxmax() if high else sc[col].idxmin()
        lines.append(f"- **{label}:** {sc.loc[j, 'Site']} ({sc.loc[j, col]:g})")
    st.markdown("\n".join(lines))
    fig = go.Figure()
    for col in ["Sunlit %", "Earth visible %", "Sun AND Earth %"]:
        fig.add_bar(x=sc["Site"], y=sc[col], name=col)
    fig.update_layout(template="plotly_dark", barmode="group", height=360, yaxis_title="% of time", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Window: {window_txt}. Sun: live Skyfield with the team's horizon masks (Model 1). Earth: Model 2 timelines where available "
        "(rule: any part of Earth's disc). Terrain: Model 3, 1 km radius, 5 m cells. The terrain score and rank are PROVISIONAL - the "
        "scoring formula is not documented in the CSV - and no overall winner is computed here on purpose: the sites trade off "
        "sunlight, communications and terrain, and the weights are a mission decision. Horizon masks cover terrain out to ~9.5 km only.")

with tabD:
    s_ = SITES[focus]
    dte = load_dte(s_["id"])
    if dte is None:
        st.info(f"No Model 2 timeline found for {s_['id']}. Put `{s_['id']}_dte_timeline.csv` in `data/dte/`.")
    else:
        step_d = float((dte["Time_UTC"].iloc[1] - dte["Time_UTC"].iloc[0]).total_seconds() / 60)
        vis = dte["earth_visible"].astype(bool).to_numpy()
        runs = runs_table(dte["Time_UTC"], vis, step_d)
        black, seen = runs[runs["state"] == "blackout"], runs[runs["state"] == "visible"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Earth visible (Model 2)", f"{100 * vis.mean():.1f}%")
        d2.metric("Longest visible run", f"{seen['hours'].max() if len(seen) else 0:.0f} h")
        d3.metric("Longest blackout", f"{black['hours'].max() if len(black) else 0:.0f} h")
        d4.metric("Blackout periods", f"{len(black)}")
        figd = go.Figure()
        figd.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Earth's upper edge at the terrain horizon")
        figd.add_trace(go.Scatter(x=dte["Time_UTC"], y=dte["visibility_margin_deg"], mode="lines", name="Visibility margin",
                                  line=dict(color="#1E90FF", width=1.5), fill="tozeroy", fillcolor="rgba(30,144,255,0.15)"))
        figd.update_layout(template="plotly_dark", height=380, xaxis_title="Time (UTC)", margin=dict(t=20),
                           yaxis_title="Margin above terrain horizon (deg); >0 = some part of Earth visible")
        st.plotly_chart(figd, use_container_width=True)
        st.markdown("**Communication windows and blackouts**")
        show = runs.copy()
        show["hours"] = show["hours"].round(1)
        st.dataframe(show, use_container_width=True, height=300)
        st.download_button("Download windows as CSV", show.to_csv(index=False).encode("utf-8"), f"{s_['id']}_dte_windows.csv", "text/csv")
        with st.expander("Cross-check Model 2 against the live Skyfield calculation"):
            g = compute_geometry_n(s_["lat"], s_["lon"], s_.get("elev", 0.0), dte["Time_UTC"].iloc[0].isoformat(), len(dte), step_d)
            mk = load_horizon(focus)
            live = ((g["Earth_Alt"] + g["Earth_Radius"]) > horizon_at(mk, g["Earth_Az"].to_numpy())).to_numpy()
            x1, x2, x3 = st.columns(3)
            x1.metric("Flag agreement", f"{100 * (live == vis).mean():.2f}%")
            x2.metric("Mismatching samples", f"{int((live != vis).sum())}")
            x3.metric("Max elevation difference", f"{np.abs(g['Earth_Alt'].to_numpy() - dte['earth_center_elevation_deg'].to_numpy()).max():.4f}°")
            st.caption("The app recomputes Earth with Skyfield/DE421 and compares with Model 2's SPICE result. Elevation differences of a "
                       "few thousandths of a degree and no flag mismatches are expected; run `validate_dte.py` for the full check.")
        st.caption(f"Model 2: SPICE DE440, Earth measured from the site on the surface, {step_d:g}-minute steps. 'Visible' means any part of "
                   "Earth's disc (about 0.95° radius) is above the terrain horizon - NASA's definition, optimistic for real radio links, which "
                   "also need antenna pointing, link budget and a ground station. Use the sidebar rule to see stricter definitions elsewhere in the app.")

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

with tabT:
    m3 = load_model3()
    sid_ = SITES[focus]["id"]
    if m3 is not None and sid_ in m3.index:
        r3 = m3.loc[sid_]
        st.markdown("**Model 3 terrain screening** - 1 km radius around the site, 5 m cells")
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("Slope <= 5°", f"{r3['pct_slope_le_5deg']:.1f}%")
        t2.metric("Slope <= 10°", f"{r3['pct_slope_le_10deg']:.1f}%")
        t3.metric("Flat circle <= 5°", f"{r3['largest_flat_circle_le_5deg_diameter_m']:.0f} m")
        t4.metric("Flat circle <= 10°", f"{r3['largest_flat_circle_le_10deg_diameter_m']:.0f} m")
        t5.metric("Relief (1 km)", f"{r3['relief_m']:.0f} m")
        t6.metric("Score (rank), provisional", f"{r3['terrain_score_provisional']:.1f} ({int(r3['terrain_rank_provisional'])}/4)")
        st.caption("'Flat circle' is the diameter of the largest circle that fits entirely inside one connected area at or below that slope. "
                   "Score and rank are provisional: the scoring formula is not documented in the CSV, so do not quote the rank until the "
                   "weights are stated.")
    else:
        st.info("Model 3 CSV (model3_site_comparison.csv) not found - put it in `data/`.")
    st.markdown("---")
    st.markdown("**Slope map (LOLA cutout)**")
    if tr is None:
        st.info("No terrain product for this site yet. Run `preprocessing/make_terrain_products.py` for it, "
                "put the output in `terrain/`, and add `\"terrain\": \"<name>\"` to the site in `SITES`.")
    else:
        meta, hgt, slp = tr["meta"], tr["height"], tr["slope"]
        mpp = meta["metres_per_pixel"]
        rows, cols = hgt.shape
        thr = st.slider("Maximum acceptable slope (°)", 2, 25, 10,
                        help="Placeholder screening value. The real limit depends on the lander and must come from its specification.")
        yy, xx = np.mgrid[0:rows, 0:cols]
        dist = np.hypot(xx - cols / 2, yy - rows / 2) * mpp
        inside = dist <= (min(rows, cols) / 2) * mpp
        ok = (slp <= thr) & inside
        site_slope = float(slp[rows // 2 - 1: rows // 2 + 2, cols // 2 - 1: cols // 2 + 2].mean())
        near = ok[dist <= 500].mean() * 100 if (dist <= 500).any() else float("nan")
        nearest = float(dist[ok].min()) if ok.any() else float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Slope at site", f"{site_slope:.1f}°")
        m2.metric(f"Area ≤ {thr}° (whole cutout)", f"{100 * ok.sum() / inside.sum():.1f}%")
        m3.metric(f"Area ≤ {thr}° within 500 m", f"{near:.1f}%")
        m4.metric("Nearest cell ≤ threshold", f"{nearest:.0f} m" if np.isfinite(nearest) else "none")

        shade = hillshade(hgt, mpp)
        xs = (np.arange(cols) - cols / 2) * mpp
        ys = (rows / 2 - np.arange(rows)) * mpp
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=xs, y=ys[::-1], z=shade[::-1], colorscale="gray", showscale=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Heatmap(x=xs, y=ys[::-1], z=np.where(ok, 1.0, np.nan)[::-1],
                                 colorscale=[[0, "rgba(46,204,113,0.55)"], [1, "rgba(46,204,113,0.55)"]],
                                 showscale=False, name="slope ≤ threshold",
                                 customdata=slp[::-1], hovertemplate="slope %{customdata:.1f}°<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Site",
                                 marker=dict(symbol="x", size=12, color="cyan", line=dict(width=2))))
        n = np.array(tr["north"]) * 450
        fig.add_annotation(x=n[0], y=n[1], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="cyan",
                           text="N", font=dict(color="cyan", size=14))
        fig.update_layout(template="plotly_dark", height=560, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis=dict(title="Map x from site (m)", scaleanchor="y", constrain="domain"),
                          yaxis=dict(title="Map y from site (m)", constrain="domain"),
                          title="Green = slope at or below the threshold (hillshade underlay)")
        left, right = st.columns([3, 2])
        left.plotly_chart(fig, use_container_width=True)
        hist = go.Figure(go.Histogram(x=slp[inside].ravel(), nbinsx=50, marker_color="#c0c0c0"))
        hist.add_vline(x=thr, line_color="#2ecc71", annotation_text=f"{thr}°")
        hist.update_layout(template="plotly_dark", height=300, xaxis_title="Slope (°)", yaxis_title="Cells",
                           margin=dict(l=10, r=10, t=30, b=10), title="Slope distribution")
        right.plotly_chart(hist, use_container_width=True)
        stats = meta["slope_stats_native_resolution"]
        right.markdown("**Share of area by slope, at native 5 m resolution**  \n" +
                       "  \n".join(f"- {k.replace('area_below_', '≤ ').replace('deg_pct', '°')}: {v}%" for k, v in stats.items()))
        st.caption(
            f"Source: {meta['dem_file']}, {meta['source_metres_per_pixel']:g} m/px averaged to {mpp:g} m for display. "
            "Slope depends on DEM resolution, so quote the resolution with any slope number. This is a first screening, "
            "not a landing-safety assessment: it ignores boulders, craters smaller than the DEM resolution, and lander-specific limits. "
            "Percentages above use the averaged grid, so they read slightly gentler than the native-resolution table. "
            + ("Map axes follow the DEM projection (not north-up); the N arrow is derived from a standard polar-stereographic "
               "assumption - re-run make_terrain_products.py to store it from the DEM's own CRS." if tr["orientation_derived"]
               else "Map axes follow the DEM projection (not north-up); the N arrow comes from the DEM's CRS."))

with st.expander("ℹ️ Model assumptions & scientific limitations", expanded=True):
    st.markdown(f"""
- **Sun/Earth positions:** computed from the JPL DE421 ephemeris with the lunar principal-axis frame
  (Skyfield). Kernel coverage: 1900-2050. Sites are treated as points on a spherical Moon surface.
- **Terrain:** {"a horizon mask from `horizons/` is applied for this site." if has_mask else
  "**no terrain data is applied for this site**; a flat horizon is assumed, which overestimates sunlight and Earth visibility. A real result needs a DEM-derived horizon (e.g. from LOLA)."}
- **Earth visibility is geometric only.** Rule in use: **{earth_rule}**. Earth's disc is about 0.95° in radius, so the
  rule changes results by a few points. Real links also depend on antenna, link budget, relays and ground stations.
- **Sampling:** results use {step_min}-minute steps; short events can be missed.
- **Not modelled:** observer height (rover mast/antenna), Sun's finite disc, thermal conditions, ice presence.
- Site coordinates and elevations come from the team's CSV files. Horizon masks cover terrain out to about 9.5 km only.
""")
