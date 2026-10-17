
from pathlib import Path
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pyproj import CRS, Geod, Transformer
from rasterio.transform import rowcol

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).parent
DEM_PATH = ROOT / "LOLA_Data" / "LM1_final_adj_5mpp_surf.tif"

CSV_PATH = ROOT / "LOLA_Data" / "lm1_horizon.csv"
PLOT_PATH = ROOT / "LOLA_Data" / "lm1_horizon_preview.png"

# Provisional LM1 site coordinates
LATITUDE = -89.5244
LONGITUDE = 56.30993

MOON_RADIUS = 1_737_400.0  # metres
OBSERVER_HEIGHT = 2.0      # metres above terrain; provisional
STEP_METRES = 5.0
MAX_DISTANCE = 9500.0      # metres
AZIMUTH_STEP = 1            # degrees

# --------------------------------------------------
# 2. DEFINE THE CANDIDATE LUNAR PROJECTION
# --------------------------------------------------

geographic_crs = CRS.from_proj4(
    f"+proj=longlat +R={MOON_RADIUS} +type=crs"
)

projected_crs = CRS.from_proj4(
    f"+proj=stere +lat_0=-90 +lon_0=0 "
    f"+R={MOON_RADIUS} +units=m +type=crs"
)

transformer = Transformer.from_crs(
    geographic_crs,
    projected_crs,
    always_xy=True,
)

geod = Geod(a=MOON_RADIUS, b=MOON_RADIUS)

site_x, site_y = transformer.transform(
    LONGITUDE, LATITUDE
)

print("LM1 TERRAIN HORIZON MODEL")
print("-" * 40)
print("Candidate latitude:", LATITUDE)
print("Candidate longitude:", LONGITUDE)
print("Candidate projected site:", site_x, site_y)

# --------------------------------------------------
# 3. OPEN DEM AND VERIFY COVERAGE
# --------------------------------------------------

if not DEM_PATH.exists():
    raise FileNotFoundError(f"DEM not found: {DEM_PATH}")

with rasterio.open(DEM_PATH) as src:

    print("DEM dimensions:", src.width, "x", src.height)
    print("DEM bounds:", src.bounds)
    print("DEM affine transform:", src.transform)
    print("DEM CRS:", src.crs)

    if not (
        src.bounds.left <= site_x <= src.bounds.right
        and src.bounds.bottom <= site_y <= src.bounds.top
    ):
        raise ValueError(
            "Site is outside the DEM. No horizon was generated."
        )

    # Use Rasterio's affine transform for the observer pixel.
    row0, col0 = rowcol(src.transform, site_x, site_y)

    if not (
        0 <= row0 < src.height
        and 0 <= col0 < src.width
    ):
        raise ValueError("Observer pixel is outside the DEM.")

    elevation = src.read(1, masked=True)

    if np.ma.is_masked(elevation[row0, col0]):
        raise ValueError("Site elevation is missing.")

    z0 = float(elevation[row0, col0])

    print("Observer pixel:", row0, col0)
    print("Observer terrain elevation:", z0, "m")
    print("Observer height above terrain:", OBSERVER_HEIGHT, "m")

    distances = np.arange(
        STEP_METRES,
        MAX_DISTANCE + STEP_METRES / 2,
        STEP_METRES,
    )

    azimuths = np.arange(0, 360, AZIMUTH_STEP)
    horizon_angles = []

    # --------------------------------------------------
    # 4. CALCULATE HORIZON FOR EACH AZIMUTH
    # --------------------------------------------------

    observer_radius = (
        MOON_RADIUS + z0 + OBSERVER_HEIGHT
    )

    for azimuth in azimuths:

        # Follow a geodesic on the candidate lunar sphere.
        lon_start = np.full(distances.shape, LONGITUDE)
        lat_start = np.full(distances.shape, LATITUDE)
        az_start = np.full(distances.shape, float(azimuth))

        lon_points, lat_points, _ = geod.fwd(
            lon_start,
            lat_start,
            az_start,
            distances,
        )

        # Project sample coordinates into the DEM coordinate system.
        xs, ys = transformer.transform(
            lon_points, lat_points
        )

        # IMPORTANT: use the raster's actual affine transform.
        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        inside = (
            (rows >= 0)
            & (rows < src.height)
            & (cols >= 0)
            & (cols < src.width)
        )

        if not np.all(inside):
            raise ValueError(
                f"Azimuth {azimuth}° exits the DEM. "
                "No complete horizon profile was saved."
            )

        samples = elevation[rows, cols]
        heights = samples.filled(np.nan)

        if not np.all(np.isfinite(heights)):
            raise ValueError(
                f"Missing terrain along azimuth {azimuth}°. "
                "No complete horizon profile was saved."
            )

        # Spherical line-of-sight geometry.
        # Assumes DEM elevations are radial heights relative
        # to the same lunar reference sphere.
        target_radius = MOON_RADIUS + heights
        central_angle = distances / MOON_RADIUS

        vertical = (
            target_radius * np.cos(central_angle)
            - observer_radius
        )

        horizontal = (
            target_radius * np.sin(central_angle)
        )

        angles = np.degrees(
            np.arctan2(vertical, horizontal)
        )

        horizon_angles.append(float(np.max(angles)))

        if azimuth % 30 == 0:
            print(f"Processed azimuth {azimuth}°")

    # --------------------------------------------------
    # 5. SAVE HORIZON CSV
    # --------------------------------------------------

    with open(CSV_PATH, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "azimuth_deg",
            "horizon_elev_deg",
        ])

        for azimuth, angle in zip(
            azimuths, horizon_angles
        ):
            writer.writerow([
                int(azimuth),
                round(angle, 6),
            ])

    print("\nHorizon CSV saved:", CSV_PATH)
    print("Number of azimuths:", len(horizon_angles))
    print(f"Minimum horizon: {min(horizon_angles):.4f}°")
    print(f"Maximum horizon: {max(horizon_angles):.4f}°")
    print(f"Mean horizon: {np.mean(horizon_angles):.4f}°")

# --------------------------------------------------
# 6. CREATE POLAR PREVIEW
# --------------------------------------------------

theta = np.radians(azimuths)

fig, ax = plt.subplots(
    figsize=(8, 8),
    subplot_kw={"projection": "polar"},
)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.plot(theta, horizon_angles)
ax.set_title(
    "LM1 Shackleton Rim B — Provisional Terrain Horizon"
)

fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=160)
plt.close(fig)

print("\nHorizon preview saved:", PLOT_PATH)
print("Observer height assumption:", OBSERVER_HEIGHT, "m")
print("\nPrototype calculation finished.")
print(
    "Scientific validation of the DEM coordinate system and "
    "elevation datum is still required."
)
