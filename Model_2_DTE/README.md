# Model 2 — Earth Visibility (DTE)

## Overview

Model 2 estimates when Earth is geometrically visible from four candidate lunar south-pole landing sites. It accounts for Earth's apparent angular radius and the local terrain horizon.

The model generates time-series visibility data and a summary comparing all four sites. These outputs are intended for integration into the Lunar South Pole Mission Planner website.

## Candidate Sites

- **LM1:** Shackleton Rim B
- **Site06:** Nobile Rim 1
- **Site20:** Leibnitz Beta Plateau
- **Site23:** Malapert Massif

## Repository Files

### Main Python script

- `four_site_dte_model.py` — calculates DTE visibility for all four sites and generates the output CSV files.

### Output data

Located in `outputs/dte/`:

- `four_site_dte_summary.csv` — comparison of visibility percentages, longest visible periods, and longest blackouts for all four sites.
- `LM1_dte_timeline.csv` — time-series visibility results for Shackleton Rim B.
- `Site06_dte_timeline.csv` — time-series visibility results for Nobile Rim 1.
- `Site20_dte_timeline.csv` — time-series visibility results for Leibnitz Beta Plateau.
- `Site23_dte_timeline.csv` — time-series visibility results for Malapert Massif.

## Timeline Details

- Sampling interval: 10 minutes
- Timeline start: 15 October 2026, 00:00 UTC
- Timeline end: 12 January 2027, 01:10 UTC
- Samples per site: 12,824

## Website Integration

The website can use `four_site_dte_summary.csv` to display a comparison of the candidate sites and the individual timeline CSVs to display visibility over time.

Important timeline columns include:

- `timestamp_utc` — observation timestamp
- `site_id` and `site_name` — site identification
- `earth_azimuth_deg` — Earth's azimuth
- `earth_center_elevation_deg` — Earth's centre elevation
- `terrain_horizon_deg` — local terrain horizon angle
- `earth_angular_radius_deg` — apparent angular radius of Earth
- `visibility_margin_deg` — calculated visibility margin
- `earth_visible` — visibility flag (`1` = visible, `0` = not visible)

## Interpretation and Limitations

The model estimates geometric visibility of Earth above the local terrain. Visibility does not guarantee a functioning radio link, adequate signal strength, or availability of a ground station.

The outputs have passed internal consistency checks. Independent scientific validation of coordinate frames, terrain data, and physical predictions remains separate work.

## Notes

The committed CSV files are precomputed outputs intended for website integration. Re-running the Python model requires its supporting horizon inputs and SPICE kernels, which are not included in this minimal website-integration upload.
