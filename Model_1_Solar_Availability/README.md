# Model 1 — Solar Availability

This model estimates solar visibility at the provisional LM1 / Shackleton Rim B lunar south-pole site using SPICE Sun-position calculations and a terrain horizon derived from LOLA elevation data.

## Outputs

- `lm1_horizon.csv` — terrain horizon elevation angles for 360 azimuth directions.
- `solar_timeline.csv` — simulated sunlight visibility at 10-minute intervals.

## Preliminary results

- Estimated sunlight availability: 55.96%.
- Longest simulated sunlight interval: approximately 399.67 hours.
- Longest simulated darkness interval: approximately 319.17 hours.

## Limitations

These results are preliminary and have not been fully scientifically validated. Site-coordinate alignment, terrain horizon distance sensitivity, and other modelling assumptions require further verification.

The model estimates geometric sunlight visibility; it does not independently establish actual lander power output.

## Source data and dependencies

The model uses a LOLA terrain elevation raster and SPICE kernels. Source-data provenance, kernel versions, dependencies, and instructions for reproducing the outputs should be documented before the model is treated as a reproducible scientific result.
