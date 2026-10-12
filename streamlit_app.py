@st.cache_data(show_spinner="Computing lunar ephemeris & lighting geometry...")
def compute_geometry(lat: float, lon: float, start_iso: str, hours: int, step_min: int) -> pd.DataFrame:
    start = datetime.fromisoformat(start_iso)
    n = int(hours * 60 / step_min)
    
    time_hours = np.arange(n) * (step_min / 60.0)
    timestamps = [pd.Timestamp(start) + pd.Timedelta(hours=h) for h in time_hours]
    
    synodic_period = 708.23
    
    lat_offset_factor = (90.0 + lat) / 1.5
    solar_elevation = 1.54 * np.cos(2 * np.pi * time_hours / synodic_period + np.radians(lon)) + (lat_offset_factor * 0.2)
    solar_azimuth = (360.0 * time_hours / synodic_period) % 360.0
    
    base_earth_alt = 90.0 + lat
    earth_elevation = base_earth_alt + 5.0 * np.sin(2 * np.pi * time_hours / 655.7) + 2.0 * np.cos(2 * np.pi * time_hours / 168.0)
    earth_azimuth = (180.0 + 10.0 * np.sin(2 * np.pi * time_hours / 655.7) + time_hours * 0.05) % 360.0

    df = pd.DataFrame({
        "Time_UTC": timestamps,
        "Sun_Alt": solar_elevation,
        "Sun_Az": solar_azimuth,
        "Earth_Alt": earth_elevation,
        "Earth_Az": earth_azimuth,  # Fixed variable name here!
        "Sunlit": solar_elevation > 0,
        "Earth_Visible": earth_elevation > 0
    })
    return df
