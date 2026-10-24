
from pathlib import Path
import csv
import math
import spiceypy as spice

ROOT = Path(__file__).parent
KERNELS = ROOT / "SPICE_Kernels"
DATA_DIR = ROOT / "LOLA_Data"
RESULTS_DIR = ROOT / "generated_results"

START_UTC = "2026 OCT 15 00:00:00 UTC"
LUNAR_DAY_DAYS = 89.06118
STEP_MINUTES = 10
STEP_SECONDS = STEP_MINUTES * 60

MOON_RADIUS_KM = 1737.4
OBSERVER_HEIGHT_M = 2.0

SITES = [
    {
        "id": "LM1",
        "name": "Shackleton Rim B",
        "latitude": -89.5244,
        "longitude": 56.30993,
        "horizon_files": ["lm1_horizon.csv"],
    },
    {
        "id": "Site06",
        "name": "Nobile Rim 1",
        "latitude": -85.4381,
        "longitude": 37.36667,
        "horizon_files": [
            "site06_horizon.csv",
            "Site06_horizon.csv",
        ],
    },
    {
        "id": "Site20",
        "name": "Leibnitz Beta Plateau",
        "latitude": -85.4266,
        "longitude": 31.74276,
        "horizon_files": [
            "site20_horizon.csv",
            "Site20_horizon.csv",
        ],
    },
    {
        "id": "Site23",
        "name": "Malapert Massif",
        "latitude": -85.9948,
        "longitude": -0.23578,
        "horizon_files": [
            "site23_horizon.csv",
            "Site23_horizon.csv",
        ],
    },
]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def format_time(et):
    return spice.et2utc(et, "ISOC", 0) + " UTC"


def find_horizon_file(site):
    # Prefer a separate horizon file for the requested site.
    for filename in site["horizon_files"]:
        path = DATA_DIR / filename
        if path.exists():
            return path

    # Otherwise, try the combined four-site horizon file.
    combined = RESULTS_DIR / "horizon_profiles.csv"
    if combined.exists():
        return combined

    raise FileNotFoundError(
        f"No horizon data found for {site['id']} ({site['name']}). "
        f"Expected one of: {site['horizon_files']} or "
        f"{combined}. Generate the missing terrain profile first."
    )


def load_horizon(site):
    path = find_horizon_file(site)
    horizon = {}

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []

        # Supports either individual horizon CSV format.
        if "horizon_elev_deg" in fields:
            azimuth_column = "azimuth_deg"
            elevation_column = "horizon_elev_deg"
            site_column = None
        elif "horizon_elevation_deg" in fields:
            azimuth_column = "azimuth_deg"
            elevation_column = "horizon_elevation_deg"
            site_column = "site_id" if "site_id" in fields else None
        else:
            raise ValueError(
                f"Unrecognised horizon columns in {path}: {fields}"
            )

        for row in reader:
            if site_column:
                if row[site_column].strip().casefold() != (
                    site["id"].casefold()
                ):
                    continue

            azimuth = int(float(row[azimuth_column])) % 360
            elevation = float(row[elevation_column])

            if not math.isfinite(elevation):
                raise ValueError(
                    f"Non-finite horizon elevation in {path}"
                )

            horizon[azimuth] = elevation

    if len(horizon) != 360:
        raise ValueError(
            f"{site['id']}: expected 360 horizon directions, "
            f"found {len(horizon)} in {path}. "
            "Check that the file contains this site's profile."
        )

    print(f"Horizon for {site['id']}: {path}")
    return horizon


def get_horizon(azimuth, horizon):
    lower = math.floor(azimuth) % 360
    upper = (lower + 1) % 360
    fraction = azimuth - math.floor(azimuth)

    return (
        horizon[lower] * (1 - fraction)
        + horizon[upper] * fraction
    )


def make_local_basis(latitude, longitude):
    lat = math.radians(latitude)
    lon = math.radians(longitude)

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

    return up, east, north


def calculate_site(site, start_et, total_steps):
    print("\n" + "=" * 55)
    print(f"Processing {site['id']}: {site['name']}")

    horizon = load_horizon(site)

    up, east, north = make_local_basis(
        site["latitude"], site["longitude"]
    )

    # Approximate surface-observer position in MOON_ME, in km.
    radius_km = MOON_RADIUS_KM + OBSERVER_HEIGHT_M / 1000.0
    site_pos = [radius_km * component for component in up]

    sunlight = []

    for i in range(total_steps):
        et = start_et + i * STEP_SECONDS

        sun, _ = spice.spkpos(
            "SUN", et, "MOON_ME", "LT+S", "MOON"
        )

        # Corrected vector from the surface observer towards the Sun.
        sun_topo = [
            sun_component - site_component
            for sun_component, site_component
            in zip(sun, site_pos)
        ]

        distance = math.sqrt(dot(sun_topo, sun_topo))
        direction = tuple(
            component / distance for component in sun_topo
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
        item["minutes"]
        for item in intervals if item["sunlit"]
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

    print(f"Sunlight availability: {sunlight_percent:.2f}%")
    print(f"Longest sunlight: {longest_sunlight / 60:.2f} hours")
    print(f"Longest darkness: {longest_darkness / 60:.2f} hours")

    # Save an individual timeline for this site.
    output_path = DATA_DIR / f"solar_timeline_{site['id']}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_utc", "sunlight_visible"])

        for i, is_sunlit in enumerate(sunlight):
            writer.writerow([
                format_time(start_et + i * STEP_SECONDS),
                is_sunlit,
            ])

    print(f"Timeline saved: {output_path}")

    # Save the sunlight intervals for this site.
    windows_path = DATA_DIR / f"sunlight_windows_{site['id']}.csv"

    with open(windows_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "site_id", "site_name", "sunlight_visible",
            "start_utc", "end_utc", "duration_hours",
        ])

        for item in intervals:
            writer.writerow([
                site["id"],
                site["name"],
                item["sunlit"],
                format_time(item["start"]),
                format_time(item["end"]),
                round(item["minutes"] / 60, 3),
            ])

    print(f"Windows saved: {windows_path}")

    return {
        "site_id": site["id"],
        "site_name": site["name"],
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "samples": total_steps,
        "sunlight_percent": round(sunlight_percent, 4),
        "longest_sunlight_hours": round(longest_sunlight / 60, 3),
        "longest_darkness_hours": round(longest_darkness / 60, 3),
    }, sunlight


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in (
        "naif0012.tls",
        "de440s.bsp",
        "pck00011.tpc",
        "moon_pa_de440_200625.bpc",
        "moon_de440_250416.tf",
    ):
        kernel_path = KERNELS / filename
        if not kernel_path.exists():
            raise FileNotFoundError(f"Missing SPICE kernel: {kernel_path}")
        spice.furnsh(str(kernel_path))

    start_et = spice.str2et(START_UTC)
    total_steps = int(
        LUNAR_DAY_DAYS * 24 * 60 / STEP_MINUTES
    )

    print("PRELIMINARY FOUR-SITE SUNLIGHT MODEL")
    print("Frame: MOON_ME; surface-observer correction enabled")
    print("Start:", format_time(start_et))
    print(f"Sampling interval: {STEP_MINUTES} minutes")
    print(f"Samples per site: {total_steps}")

    summaries = []
    all_timelines = []

    for site in SITES:
        summary, sunlight = calculate_site(
            site, start_et, total_steps
        )
        summaries.append(summary)

        for i, is_sunlit in enumerate(sunlight):
            all_timelines.append([
                site["id"],
                site["name"],
                format_time(start_et + i * STEP_SECONDS),
                is_sunlit,
            ])

    # Combined four-site timeline.
    combined_path = DATA_DIR / "solar_timeline_all_sites.csv"

    with open(combined_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "site_id", "site_name", "time_utc", "sunlight_visible"
        ])
        writer.writerows(all_timelines)

    # Summary comparison for all four sites.
    summary_path = DATA_DIR / "solar_site_comparison.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + "=" * 55)
    print("FOUR-SITE SUMMARY")
    for summary in summaries:
        print(
            f"{summary['site_id']}: "
            f"{summary['sunlight_percent']:.2f}% sunlight"
        )

    print(f"\nCombined timeline: {combined_path}")
    print(f"Site summary: {summary_path}")
    print("Results are preliminary and require validation.")


try:
    main()
finally:
    spice.kclear()
