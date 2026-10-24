from pathlib import Path
import csv
import json
import math
from datetime import datetime

import numpy as np
import spiceypy as spice


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "LOLA_Data"
KERNEL_DIR = ROOT / "SPICE_Kernels"

HORIZON_DIR = ROOT / "Model_2_DTE" / "outputs" / "horizons"
OUTPUT_DIR = ROOT / "Model_2_DTE" / "outputs" / "dte"

TIMELINE_PATH = DATA_DIR / "solar_timeline.csv"

MOON_RADIUS_M = 1_737_400.0
EARTH_RADIUS_KM = 6_378.137

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FOUR LANDING SITES
# ============================================================

SITES = [
    {
        "id": "LM1",
        "name": "Shackleton Rim B",
        "lat": -89.5244,
        "lon": 56.30993,
    },
    {
        "id": "Site06",
        "name": "Nobile Rim 1",
        "lat": -85.4381,
        "lon": 37.36667,
    },
    {
        "id": "Site20",
        "name": "Leibnitz Beta Plateau",
        "lat": -85.4266,
        "lon": 31.74276,
    },
    {
        "id": "Site23",
        "name": "Malapert Massif",
        "lat": -85.9948,
        "lon": -0.23578,
    },
]


# ============================================================
# 3. CSV AND TIMESTAMP HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    if not rows:
        raise ValueError(f"No data rows in {path}")

    return rows, columns


def find_timestamp_column(columns):
    candidates = [
        "time_utc",
        "timestamp_utc",
        "datetime_utc",
        "timestamp",
        "datetime",
        "time",
        "date",
    ]

    lookup = {column.strip().lower(): column for column in columns}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    if columns:
        return columns[0]

    raise ValueError("No timestamp column found in timeline CSV.")


def normalize_utc(value):
    """Convert ISO UTC timestamps to a SPICE-compatible time string."""
    value = value.strip()

    # Remove an existing UTC suffix or ISO Z suffix.
    if value.upper().endswith(" UTC"):
        value = value[:-4].strip()
    elif value.endswith("Z"):
        value = value[:-1]

    # SPICE accepts the space-separated form.
    value = value.replace("T", " ")

    return value + " UTC"


def parse_timestamp(value):
    normalized = normalize_utc(value)
    date_part = normalized[:-4].strip()

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(date_part, fmt)
        except ValueError:
            pass

    raise ValueError(f"Unsupported timeline timestamp: {value!r}")


# ============================================================
# 4. LOAD HORIZON PROFILES AND SITE ELEVATIONS
# ============================================================

def load_horizon(site_id):
    path = HORIZON_DIR / f"{site_id}_horizon.csv"
    rows, _ = load_csv(path)

    values = {}

    for row in rows:
        azimuth = int(float(row["azimuth_deg"]))
        angle = float(row["horizon_elevation_deg"])

        if azimuth in values:
            raise ValueError(
                f"{site_id}: duplicate azimuth {azimuth}"
            )

        values[azimuth] = angle

    if len(values) != 360:
        raise ValueError(
            f"{site_id}: expected 360 horizon angles; "
            f"found {len(values)}."
        )

    azimuths = np.arange(360, dtype=float)
    angles = np.array(
        [values[int(az)] for az in azimuths],
        dtype=float,
    )

    return azimuths, angles


def load_site_metadata():
    path = HORIZON_DIR / "site_metadata.json"

    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return {
        item["site_id"]: item
        for item in metadata
    }


# ============================================================
# 5. LOAD SPICE KERNELS
# ============================================================

def load_kernels():
    required = [
        "naif0012.tls",
        "pck00011.tpc",
        "moon_de440_250416.tf",
        "moon_pa_de440_200625.bpc",
        "de440s.bsp",
    ]

    for filename in required:
        path = KERNEL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing SPICE kernel: {path}"
            )

        spice.furnsh(str(path))
        print(f"Loaded kernel: {filename}")


# ============================================================
# 6. SITE POSITION AND LOCAL DIRECTIONS
# ============================================================

def site_geometry(latitude_deg, longitude_deg, elevation_m):
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)

    radius = MOON_RADIUS_M + elevation_m

    # Site position in a spherical Moon-fixed frame.
    site_position = np.array([
        radius * math.cos(latitude) * math.cos(longitude),
        radius * math.cos(latitude) * math.sin(longitude),
        radius * math.sin(latitude),
    ])

    east = np.array([
        -math.sin(longitude),
        math.cos(longitude),
        0.0,
    ])

    north = np.array([
        -math.sin(latitude) * math.cos(longitude),
        -math.sin(latitude) * math.sin(longitude),
        math.cos(latitude),
    ])

    up = np.array([
        math.cos(latitude) * math.cos(longitude),
        math.cos(latitude) * math.sin(longitude),
        math.sin(latitude),
    ])

    return site_position, east, north, up


# ============================================================
# 7. EARTH VISIBILITY AT ONE TIMESTAMP
# ============================================================

def calculate_visibility(
    et,
    site,
    elevation_m,
    horizon_azimuths,
    horizon_angles,
):
    # Earth centre relative to the Moon, in MOON_ME.
    earth_position, _ = spice.spkpos(
        "EARTH",
        et,
        "MOON_ME",
        "NONE",
        "MOON",
    )

    site_position, east, north, up = site_geometry(
        site["lat"],
        site["lon"],
        elevation_m,
    )

    earth_position_m = np.asarray(
        earth_position,
        dtype=float,
    ) * 1000.0

    direction = earth_position_m - site_position

    east_component = float(np.dot(direction, east))
    north_component = float(np.dot(direction, north))
    up_component = float(np.dot(direction, up))

    horizontal_distance = math.hypot(
        east_component,
        north_component,
    )

    distance_km = float(np.linalg.norm(direction)) / 1000.0

    azimuth = math.degrees(
        math.atan2(east_component, north_component)
    ) % 360.0

    centre_elevation = math.degrees(
        math.atan2(up_component, horizontal_distance)
    )

    earth_angular_radius = math.degrees(
        math.asin(
            min(1.0, EARTH_RADIUS_KM / distance_km)
        )
    )

    # ========================================================
    # FIXED TERRAIN-HORIZON INTERPOLATION
    # ========================================================

    # Interpolate the terrain horizon around the azimuth circle.
    azimuth_for_interpolation = azimuth % 360.0

    # Keep interpolation coordinates strictly ascending.
    # -1 degrees represents 359 degrees from the previous cycle.
    # 360 degrees represents 0 degrees from the next cycle.
    extended_azimuths = np.concatenate((
        [-1.0],
        horizon_azimuths,
        [360.0],
    ))

    extended_angles = np.concatenate((
        [horizon_angles[-1]],
        horizon_angles,
        [horizon_angles[0]],
    ))

    terrain_horizon = float(
        np.interp(
            azimuth_for_interpolation,
            extended_azimuths,
            extended_angles,
        )
    )

    # Positive margin means some part of Earth's apparent disc
    # is above the terrain horizon.
    visibility_margin = (
        centre_elevation
        + earth_angular_radius
        - terrain_horizon
    )

    visible = int(visibility_margin > 0.0)

    return {
        "earth_azimuth_deg": azimuth,
        "earth_center_elevation_deg": centre_elevation,
        "terrain_horizon_deg": terrain_horizon,
        "earth_angular_radius_deg": earth_angular_radius,
        "visibility_margin_deg": visibility_margin,
        "earth_visible": visible,
    }


# ============================================================
# 8. VISIBLE AND BLACKOUT RUNS
# ============================================================

def calculate_runs(flags):
    runs = []
    start = 0

    while start < len(flags):
        state = flags[start]
        end = start + 1

        while end < len(flags) and flags[end] == state:
            end += 1

        runs.append({
            "visible": state,
            "samples": end - start,
        })

        start = end

    return runs


# ============================================================
# 9. MAIN FOUR-SITE MODEL
# ============================================================

def main():
    timeline, columns = load_csv(TIMELINE_PATH)
    timestamp_column = find_timestamp_column(columns)

    timestamps = [
        parse_timestamp(row[timestamp_column])
        for row in timeline
    ]

    if len(timestamps) < 2:
        raise ValueError("Timeline must contain at least two rows.")

    intervals = [
        (timestamps[i] - timestamps[i - 1]).total_seconds()
        for i in range(1, len(timestamps))
    ]

    if any(interval <= 0 for interval in intervals):
        raise ValueError(
            "Timeline timestamps must be strictly increasing."
        )

    if not all(
        math.isclose(interval, intervals[0], abs_tol=0.01)
        for interval in intervals
    ):
        raise ValueError(
            "Timeline intervals are not uniform. "
            "Use interval-specific duration calculations."
        )

    interval_minutes = intervals[0] / 60.0

    print(f"Timeline rows: {len(timeline)}")
    print(f"Timestamp column: {timestamp_column}")
    print(f"First timestamp: {timeline[0][timestamp_column]}")
    print(f"Last timestamp: {timeline[-1][timestamp_column]}")
    print(f"Sampling interval: {interval_minutes:.2f} minutes")

    metadata = load_site_metadata()
    load_kernels()

    summary = []

    try:
        for site in SITES:
            site_id = site["id"]

            print()
            print("=" * 55)
            print(f'Processing {site_id}: {site["name"]}')

            if site_id not in metadata:
                raise ValueError(
                    f"Missing metadata for {site_id}"
                )

            elevation_m = float(
                metadata[site_id]["dem_elevation_at_site_m"]
            )

            horizon_azimuths, horizon_angles = load_horizon(site_id)

            output_path = (
                OUTPUT_DIR / f"{site_id}_dte_timeline.csv"
            )

            flags = []

            fieldnames = [
                "timestamp_utc",
                "site_id",
                "site_name",
                "earth_azimuth_deg",
                "earth_center_elevation_deg",
                "terrain_horizon_deg",
                "earth_angular_radius_deg",
                "visibility_margin_deg",
                "earth_visible",
            ]

            with output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
                writer.writeheader()

                for index, row in enumerate(timeline):
                    timestamp = row[timestamp_column]

                    # Convert the timestamp to a SPICE time.
                    et = spice.str2et(normalize_utc(timestamp))

                    result = calculate_visibility(
                        et,
                        site,
                        elevation_m,
                        horizon_azimuths,
                        horizon_angles,
                    )

                    flags.append(result["earth_visible"])

                    output_row = {
                        "timestamp_utc": timestamp,
                        "site_id": site_id,
                        "site_name": site["name"],
                    }

                    for key, value in result.items():
                        output_row[key] = (
                            round(value, 6)
                            if isinstance(value, float)
                            else value
                        )

                    writer.writerow(output_row)

                    if (index + 1) % 1000 == 0:
                        print(
                            f"Processed {index + 1}/"
                            f"{len(timeline)} timestamps"
                        )

            total_samples = len(flags)
            visible_samples = sum(flags)

            visibility_percent = (
                100.0 * visible_samples / total_samples
            )

            runs = calculate_runs(flags)

            longest_visible = max(
                (
                    run["samples"]
                    for run in runs
                    if run["visible"] == 1
                ),
                default=0,
            )

            longest_blackout = max(
                (
                    run["samples"]
                    for run in runs
                    if run["visible"] == 0
                ),
                default=0,
            )

            summary.append({
                "site_id": site_id,
                "site_name": site["name"],
                "timeline_samples": total_samples,
                "visible_samples": visible_samples,
                "visibility_percent": round(
                    visibility_percent, 4
                ),
                "longest_visible_run_samples": longest_visible,
                "longest_visible_run_minutes": round(
                    longest_visible * interval_minutes, 2
                ),
                "longest_blackout_samples": longest_blackout,
                "longest_blackout_minutes": round(
                    longest_blackout * interval_minutes, 2
                ),
                "sample_interval_minutes": interval_minutes,
                "timeline_start_utc": timeline[0][timestamp_column],
                "timeline_end_utc": timeline[-1][timestamp_column],
            })

            print(f"Saved timeline: {output_path}")
            print(f"Visibility: {visibility_percent:.2f}%")
            print(
                "Longest visible run: "
                f"{longest_visible * interval_minutes:.0f} minutes"
            )
            print(
                "Longest blackout: "
                f"{longest_blackout * interval_minutes:.0f} minutes"
            )

        summary_path = OUTPUT_DIR / "four_site_dte_summary.csv"

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(summary[0].keys()),
            )
            writer.writeheader()
            writer.writerows(summary)

        print()
        print("=" * 55)
        print("FOUR-SITE DTE RUN COMPLETED")
        print(f"Summary saved: {summary_path}")
        print(f"Output folder: {OUTPUT_DIR}")

    finally:
        spice.kclear()


if __name__ == "__main__":
    main()