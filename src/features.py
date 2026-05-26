'''
features.py

this file is responsible for feature engenieering and EPA AQI computation

responsibilites:
    Define the EPA AQI table for all six pollutants
    compute the EPA 0 - 500 AQI 
    assemble feature rows
    handle cold start condition

Out of Scope:
    dont perform API call, file read/write and database write.

EPA AQI Formula Reference:
    https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf

'''

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

#=====================================================================
# ------ EPA AQI Breakpoint tables -------
#=====================================================================

# each table is a list of tuple, the structure is as follow:
# (C_low, C_high, AQI_low, AQI_high)

PM25_BREAKPOINTS = [
    (0.0,   12.0,  0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0,    54,   0,   50),
    (55,   154,  51,  100),
    (155,  254,  101, 150),
    (255,  354,  151, 200),
    (355,  424,  201, 300),
    (425,  504,  301, 400),
    (505,  604,  401, 500),
]

O3_BREAKPOINTS = [
    (0,    54,   0,   50),
    (55,   70,   51,  100),
    (71,   85,   101, 150),
    (86,   105,  151, 200),
    (106,  200,  201, 300),
    (201,  404,  301, 400),   
    (405,  604,  401, 500),   
]

CO_BREAKPOINTS = [
    (0.0,  4.4,  0,   50),
    (4.5,  9.4,  51,  100),
    (9.5,  12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

NO2_BREAKPOINTS = [
    (0,    53,   0,   50),
    (54,   100,  51,  100),
    (101,  360,  101, 150),
    (361,  649,  151, 200),
    (650,  1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

SO2_BREAKPOINTS = [
    (0,   35,   0,   50),
    (36,  75,   51,  100),
    (76,  185,  101, 150),
    (186, 304,  151, 200),
    (305, 604,  201, 300),
    (605, 804,  301, 400),
    (805, 1004, 401, 500),
]

# open-meteo returns pollutants O3, NO2, SO2, CO in µg/m³, 
# but EPA breakpoints uses uses ppb and ppm.

# defining molecular weights of each pollutants (used for conversion)
O3_MW = 48.0
NO2_MW = 46.0
SO2_MW = 64.0
CO_MW = 28.0

#=====================================================================
#----- FUNCTION 1 : CALCULATE AQI USING EPA BREAKPOINT TABLES --------
#=====================================================================

def calculate_aqi_for_pollutant(concentration, breakpoints):
    ''' 
    Compute the AQI for single pollutant using EPA linear interpolation
    '''

    if concentration is None:
        logger.warning("Recieved None concentration - cannot compute AQI")
        return None
    
    if concentration < 0:
        logger.warning("Recieved negative concentration (%.4f) - physically invalid", concentration)
        return None
    
    for (c_low, c_high, aqi_low, aqi_high) in breakpoints:
        if c_low <= concentration <= c_high:

            # ------- Applying EPA Linear Interpolation Formula -----------------
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low

            return round(aqi)
        
    logger.warning("concentration %.4f exceed all EPA breakpoint ranges - capping AQI at 500", concentration)
    return 500

#=====================================================================
#------------- FUNCTION 2 : Perform Unit Conversion & ----------------
#---------------- Compute EPA AQI for all Pollutants ----------------- 
#=====================================================================

def compute_epa_aqi(pollutants):
    '''
    compute the EPQ AQI for all the pollutants

    Applies unit converstion to the required pollutants
    '''
    def safe_key(key):
        ''' savely extracting the pollutant concentration, logging if the key is missing'''
        value = pollutants.get(key)

        if value is None:
            logging.warning("Pollutant '%s' is missing from API response", key)
        
        return value

    # etraxting values for each pollutants:
    pm2_5_raw = safe_key("pm2_5")
    pm10_raw = safe_key("pm10")
    o3_raw = safe_key("o3")
    co_raw = safe_key("co")
    no2_raw = safe_key("no2")
    so2_raw = safe_key("so2")

    # performing unit conversion all except pm2.5 and pm10 from µg/m³ to ppb and ppm(CO)
    # Formula: ppb = (µg/m³ × 24.45) / molecular_weight
    MOLAR_VOLUME = 24.45

    o3_ppb  = (o3_raw  * MOLAR_VOLUME / O3_MW)  if o3_raw  is not None else None
    no2_ppb = (no2_raw * MOLAR_VOLUME / NO2_MW) if no2_raw is not None else None
    so2_ppb = (so2_raw * MOLAR_VOLUME / SO2_MW) if so2_raw is not None else None
    co_ppm  = (co_raw  * MOLAR_VOLUME / CO_MW / 1000) if co_raw  is not None else None

    logger.debug(
        "converted concentration - O3: %.2f ppb, NO2: %.2f ppb, SO2: %.2f ppb, CO: %.4f ppm",
        o3_ppb or 0,
        no2_ppb or 0,
        so2_ppb or 0,
        co_ppm or 0
    )

    # computing AQI for all pollutants:
    per_pollutant_aqi = {
        "pm2_5": calculate_aqi_for_pollutant(pm2_5_raw, PM25_BREAKPOINTS),
        "pm10":  calculate_aqi_for_pollutant(pm10_raw,  PM10_BREAKPOINTS),
        "o3":    calculate_aqi_for_pollutant(o3_ppb,    O3_BREAKPOINTS),
        "co":    calculate_aqi_for_pollutant(co_ppm,    CO_BREAKPOINTS),
        "no2":   calculate_aqi_for_pollutant(no2_ppb,   NO2_BREAKPOINTS),
        "so2":   calculate_aqi_for_pollutant(so2_ppb,   SO2_BREAKPOINTS),
    }

    logger.debug("per-pollutant AQI values: %s", per_pollutant_aqi)

    # filtering out the failed ones and finding the max
    valid_aqi = {
        pollutant: aqi

        for pollutant, aqi in per_pollutant_aqi.items()
        if aqi is not None
    }

    if not valid_aqi:
        logger.error(
            "All pollutant AQI computations failed. "
            "Cannot produce a valid AQI for this row."
        )
        return {"aqi": None, "dominant_pollutant": None}
    
    dominant_pollutant = max(valid_aqi, key=valid_aqi.get)
    final_aqi = valid_aqi[dominant_pollutant]

    logger.info(
        "Final EPA AQI: %d — Dominant pollutant: %s",
        final_aqi, dominant_pollutant
    )

    return {
        "aqi": final_aqi,
        "dominant_pollutant": dominant_pollutant
    }

#=====================================================================
#-------------------- FUNCTION 3 : CATEGORIZE EPA ------=-------------
#=====================================================================

def get_epa_category(aqi):
    if aqi is None: return None
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

#=====================================================================
#---------------------- FEATURE ENGENIEERING -------------------------
#=====================================================================

def engineer_features(raw_pollution, raw_weather, historical_rows):
    '''
    This function is the single entry point for all feature engineering. 
    It delegates AQI computation, 
    extracts raw features, 
    computes time-based
    features, 
    handles derived features,
    and assembles a complete, flat feature row
    '''

    logger.info("Engineering features. historical rows avaible: %d", len(historical_rows))

    # extracting raw pollutants concentrations.
    pollutants = {
        "pm2_5": raw_pollution.get("pm2_5"),
        "pm10":  raw_pollution.get("pm10"),
        "no2":   raw_pollution.get("nitrogen_dioxide"),
        "o3":    raw_pollution.get("ozone"),
        "co":    raw_pollution.get("carbon_monoxide"),
        "so2":   raw_pollution.get("sulphur_dioxide"),
    }

    logger.debug("Extracted components: %s", pollutants)

    # computing EPA AQI:
    aqi_result = compute_epa_aqi(pollutants)
    current_aqi = aqi_result["aqi"]

    if current_aqi is None:
        logger.error("AQI computation returned None. skipping feature row.")
        return None
    
    # Extracting weather features:
    weather = {
        "temp":       raw_weather.get("temperature_2m"),
        "humidity":   raw_weather.get("relative_humidity_2m"),
        "pressure":   raw_weather.get("surface_pressure"),
        "wind_speed": raw_weather.get("wind_speed_10m"),
        "wind_deg":   raw_weather.get("wind_direction_10m"),
        "clouds":     raw_weather.get("cloud_cover")
    }

    logger.debug("Extracted weather features: %s", weather)

    # Computing time-based features
    try:
        # Look for the historical timestamp first (for backfills). Fallback to now() for live pipeline.
        obs_time_str = raw_weather.get("time")
        if obs_time_str:
            now = datetime.fromisoformat(obs_time_str).replace(tzinfo=timezone.utc)
        else:
            now = datetime.now(timezone.utc)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning("Failed to parse timestamp from weather data: %s. Falling back to current UTC time.", e)
        now = datetime.now(timezone.utc)

    is_weekend = 1 if now.weekday() >= 5 else 0

    time_features = {
        "timestamp":   now.isoformat(),
        "hour":        now.hour,
        "day_of_week": now.weekday(),   
        "month":       now.month,
        "is_weekend":  is_weekend
    }

    logger.debug("Time features: %s", time_features)

    # extracting wind and categorizing it
    wind_speed = weather["wind_speed"]

    if wind_speed is None:
        wind_speed_category = "unknown"
        logger.warning("Wind speed missing — category set to 'unknown'.")
    elif wind_speed <= 1.5:
        wind_speed_category = "calm"
    elif wind_speed <= 3.3:
        wind_speed_category = "moderate"
    else:
        wind_speed_category = "strong"

    # Computing derived features:
    current_pm2_5 = pollutants["pm2_5"]

    # edge-case 1 : when we dont have any record:
    if not historical_rows:
        logger.warning("No historical row avaable. Using defaults fro derived features.")
        aqi_change_rate = 0
        pm2_5_rolling_3h = current_pm2_5

    # edge-case 2 : when we have only one historical record:
    else:
        most_recent_row = historical_rows[-1]
        previous_aqi = most_recent_row.get("aqi")
        aqi_change_rate = (round(current_aqi - previous_aqi, 2) if previous_aqi is not None else 0)

    historical_pm2_5 = []

    for row in historical_rows[-2:]:
        value = row.get("pm2_5")
        if value is not None:
            historical_pm2_5.append(value)

    all_pm2_5_values = historical_pm2_5 + [current_pm2_5]

    pm2_5_rolling_3h = round(sum(all_pm2_5_values) /  len(all_pm2_5_values), 4)

    logger.info("Derived features — aqi_change_rate: %s, pm2_5_rolling_3h: %s", aqi_change_rate, pm2_5_rolling_3h    )

    # Assembling and returning the complete feature row:
    feature_row = {
        # Timestamp
        "timestamp":            time_features["timestamp"],

        # Target variable
        "aqi":                  current_aqi,
        "category":             get_epa_category(current_aqi),
        "dominant_pollutant":   aqi_result["dominant_pollutant"],

        # Raw pollutants
        "pm2_5":                pollutants["pm2_5"],
        "pm10":                 pollutants["pm10"],
        "no2":                  pollutants["no2"],
        "o3":                   pollutants["o3"],
        "co":                   pollutants["co"],
        "so2":                  pollutants["so2"],

        # Weather conditions
        "temp":                 weather["temp"],
        "humidity":             weather["humidity"],
        "pressure":             weather["pressure"],
        "wind_speed":           weather["wind_speed"],
        "wind_deg":             weather["wind_deg"],
        "clouds":               weather["clouds"],

        # Time-based features
        "hour":                 time_features["hour"],
        "day_of_week":          time_features["day_of_week"],
        "month":                time_features["month"],
        "is_weekend":           time_features["is_weekend"],

        # Derived features
        "aqi_change_rate":      aqi_change_rate,
        "pm2_5_rolling_3h":     pm2_5_rolling_3h,
        "wind_speed_category":  wind_speed_category,
    }

    logger.info(
        "Feature row assembled successfully. AQI: %d, Dominant: %s, "
        "Timestamp: %s",
        current_aqi,
        aqi_result["dominant_pollutant"],
        time_features["timestamp"]
    )

    return feature_row