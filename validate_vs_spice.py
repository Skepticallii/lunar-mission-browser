"""
Cross-check the app's Skyfield Sun geometry against the teammate's SPICE result.

Run from the project folder (needs internet the first time for Skyfield kernels):
    python validate_vs_spice.py

Inputs : LOLA_Data/lm1_horizon.csv      columns: azimuth_deg, horizon_elev_deg
         LOLA_Data/solar_timeline.csv   columns: time_utc, sunlight_visible   (from the SPICE script)
Output : agreement %, plus the time steps where the two methods disagree.

Two independent ephemeris implementations (SPICE DE440 vs Skyfield DE421) fed the same
horizon mask should agree on nearly every step. Disagreements should sit right at
sunrise/sunset edges (small elevation margin). Large margins would point to a real bug,
e.g. a different azimuth convention or longitude sign.
"""
import numpy as np
import pandas as pd
from skyfield.api import Loader
from skyfield.planetarylib import PlanetaryConstants

LAT, LON = -89.5244, 56.30993
HORIZON_CSV = "LOLA_Data/lm1_horizon.csv"
TIMELINE_CSV = "LOLA_Data/solar_timeline.csv"

# --- teammate's SPICE results ---
tl = pd.read_csv(TIMELINE_CSV)
tl["t"] = pd.to_datetime(tl["time_utc"].str.replace(" UTC", "", regex=False))
spice_flags = tl["sunlight_visible"].astype(str).str.lower().eq("true").to_numpy()

# --- horizon mask ---
hz = pd.read_csv(HORIZON_CSV).sort_values("azimuth_deg")

# --- Skyfield geometry for the same instants ---
loader = Loader("kernels")
ts = loader.timescale()
eph = loader("de421.bsp")
pc = PlanetaryConstants()
pc.read_text(loader("moon_080317.tf"))
pc.read_text(loader("pck00008.tpc"))
pc.read_binary(loader("moon_pa_de421_1900-2050.bpc"))
frame = pc.build_frame_named("MOON_ME_DE421")

t = ts.utc(tl["t"].dt.year.to_numpy(), tl["t"].dt.month.to_numpy(), tl["t"].dt.day.to_numpy(),
           tl["t"].dt.hour.to_numpy(), tl["t"].dt.minute.to_numpy(), tl["t"].dt.second.to_numpy())
site = eph["moon"] + pc.build_latlon_degrees(frame, LAT, LON)
alt, az, _ = site.at(t).observe(eph["sun"]).apparent().altaz()
horizon = np.interp(az.degrees, hz["azimuth_deg"], hz["horizon_elev_deg"], period=360.0)
margin = alt.degrees - horizon            # >0 means Sun above the terrain skyline
sky_flags = margin > 0

# --- compare ---
n = len(tl)
bad = sky_flags != spice_flags
print(f"Samples compared          : {n}")
print(f"SPICE sunlit %            : {100 * spice_flags.mean():.2f}")
print(f"Skyfield sunlit %         : {100 * sky_flags.mean():.2f}")
print(f"Agreement                 : {100 * (1 - bad.mean()):.2f}%  ({bad.sum()} mismatches)")
if bad.any():
    mm = np.abs(margin[bad])
    print(f"Mismatch |elevation margin|: median {np.median(mm):.4f}°, max {mm.max():.4f}°")
    print("(tiny margins = sunrise/sunset edge effects; margins of ~1° or more = real disagreement)\n")
    out = tl.loc[bad, ["time_utc"]].copy()
    out["spice"] = spice_flags[bad]
    out["skyfield"] = sky_flags[bad]
    out["margin_deg"] = margin[bad].round(4)
    print(out.head(30).to_string(index=False))
