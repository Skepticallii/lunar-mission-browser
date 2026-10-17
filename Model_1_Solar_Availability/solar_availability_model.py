
from pathlib import Path
import csv
import math
import spiceypy as spice

ROOT = Path(__file__).parent
KERNELS = ROOT / "SPICE_Kernels"
HORIZON_PATH = ROOT / "LOLA_Data" / "lm1_horizon.csv"

LATITUDE = -89.5244
LONGITUDE = 56.30993

START_UTC = "2026 OCT 15 00:00:00 UTC"
LUNAR_DAY_DAYS = 89.06118
STEP_MINUTES = 10
STEP_SECONDS = STEP_MINUTES * 60


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def get_horizon(azimuth, horizon):
    lower = math.floor(azimuth) % 360
    upper = (lower + 1) % 360
    fraction = azimuth - math.floor(azimuth)

    return (
        horizon[lower] * (1 - fraction)
        + horizon[upper] * fraction
    )


def format_time(et):
    return spice.et2utc(et, "ISOC", 0) + " UTC"


try:
    for filename in (
        "naif0012.tls",
        "de440s.bsp",
        "pck00011.tpc",
        "moon_pa_de440_200625.bpc",
        "moon_de440_250416.tf",
    ):
        spice.furnsh(str(KERNELS / filename))

    with open(HORIZON_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        horizon = {
            int(row["azimuth_deg"]):
                float(row["horizon_elev_deg"])
            for row in reader
        }

    if len(horizon) != 360:
        raise ValueError(
            "Expected 360 terrain-horizon directions."
        )

    lat = math.radians(LATITUDE)
    lon = math.radians(LONGITUDE)

    up = (
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    )

    east = (
        -math.sin(lon),
        math.cos(lon),
        0.0,
    )

    north = (
        -math.sin(lat) * math.cos(lon),
        -math.sin(lat) * math.sin(lon),
        math.cos(lat),
    )

    start_et = spice.str2et(START_UTC)
    total_steps = int(
        LUNAR_DAY_DAYS * 24 * 60 / STEP_MINUTES
    )

    sunlight = []

    print("Calculating preliminary sunlight availability...")
    print("Reference frame: MOON_ME")
    print("Start:", format_time(start_et))
    print("Sampling interval:", STEP_MINUTES, "minutes")
    print("Number of samples:", total_steps)

    for i in range(total_steps):
        et = start_et + i * STEP_SECONDS

        sun, _ = spice.spkpos(
            "SUN", et, "MOON_ME", "LT+S", "MOON"
        )

        distance = math.sqrt(dot(sun, sun))
        direction = tuple(
            component / distance for component in sun
        )

        up_component = dot(direction, up)
        east_component = dot(direction, east)
        north_component = dot(direction, north)

        elevation = math.degrees(
            math.atan2(
                up_component,
                math.hypot(east_component, north_component),
            )
        )

        azimuth = math.degrees(
            math.atan2(east_component, north_component)
        ) % 360

        terrain_angle = get_horizon(azimuth, horizon)
        sunlight.append(elevation > terrain_angle)

    intervals = []
    interval_start = 0

    for i in range(1, total_steps + 1):
        if (
            i == total_steps
            or sunlight[i] != sunlight[interval_start]
        ):
            duration = (i - interval_start) * STEP_MINUTES

            intervals.append({
                "sunlit": sunlight[interval_start],
                "start": start_et + interval_start * STEP_SECONDS,
                "end": start_et + i * STEP_SECONDS,
                "minutes": duration,
            })

            interval_start = i

    total_minutes = total_steps * STEP_MINUTES
    sunlight_minutes = sum(
        item["minutes"] for item in intervals
        if item["sunlit"]
    )

    sunlight_percent = 100 * sunlight_minutes / total_minutes

    longest_sunlight = max(
        (item["minutes"] for item in intervals if item["sunlit"]),
        default=0,
    )

    longest_darkness = max(
        (item["minutes"] for item in intervals if not item["sunlit"]),
        default=0,
    )

    print("\nPRELIMINARY RESULTS")
    print("-" * 40)
    print(f"Sunlight availability: {sunlight_percent:.2f}%")
    print(f"Longest sunlight: {longest_sunlight / 60:.2f} hours")
    print(f"Longest darkness: {longest_darkness / 60:.2f} hours")

    print("\nSUNLIGHT WINDOWS")
    print("-" * 40)

    for item in intervals:
        if item["sunlit"]:
            print(
                format_time(item["start"]),
                "to",
                format_time(item["end"]),
                f"({item['minutes'] / 60:.2f} hours)",
            )

    output_path = ROOT / "LOLA_Data" / "solar_timeline.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_utc", "sunlight_visible"])

        for i, is_sunlit in enumerate(sunlight):
            writer.writerow([
                format_time(start_et + i * STEP_SECONDS),
                is_sunlit,
            ])

    print("\nTimeline saved to:", output_path)
    print("Calculation completed.")
    print("IMPORTANT: Results are preliminary, not yet validated.")

finally:
    spice.kclear()
