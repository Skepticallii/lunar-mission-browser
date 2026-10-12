# lunar-mission-browser
NASA Space Apps Challenge - Lunar South Pole Mission Planner.
Put one CSV per site here, named <site_slug>.csv, e.g. shackleton_crater_rim.csv
Columns: azimuth_deg,horizon_elev_deg
Azimuth: degrees clockwise from local north (same convention as Skyfield altaz()).
Derive these from a LOLA DEM (PDS LOLA archive) by ray-marching outward from the site
in each azimuth and recording the maximum elevation angle of the terrain.
