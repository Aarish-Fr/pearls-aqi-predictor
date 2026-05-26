"""
Feature-Pipeline:

Orchestrates the hourly data collection lifescycle
- Loads credentials
- fetches raw data from open-meteo
- read recnet record from the Hopswork
- engineers features
- write new row to features group
"""

import logging
import os
import sys

import requests
import pandas as pd
from dotenv import load_dotenv

from features import engineer_features

#=====================================================================
# ------------------ Logging Configuration ------------------
#=====================================================================

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

#=====================================================================
# ------------------ Loading Credentials ------------------
#=====================================================================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
LAT = os.getenv("KARACHI_LAT")
LON = os.getenv("KARACHI_LON")

# Saftey Guard

_required = {
    "HOPSWORKS_API_KEY": HOPSWORKS_API_KEY,
    "HOPSWORKS_PROJECT_NAME": HOPSWORKS_PROJECT_NAME,
    "KARACHI_LAT": LAT,
    "KARACHI_LON": LON,
}

_missing = [key for key, val in _required.items() if not val]

if _missing:
    logger.error("Missing required enviroment variable: %s", _missing)
    sys.exit(1)

logger.info("Credentials loaded successfully")

#=====================================================================
# ------------- Function for air pollution fetching ------------------
#=====================================================================

def fetch_air_pollution() -> dict:
    '''
    fetches raw air pollution data from Open-Meteo
    Uses past_days=1 to guarantee we catch any hours missed by GitHub Actions.
    '''

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "UTC",
        "past_days": 1  
    }

    logger.info("Fetching air pollution arrays from Open-Meteo")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info("Open-Meteo air pollution data fetched successfully.")
        return data
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch pollutant data: %s", e)
        raise

#=====================================================================
# -------------- Function for weather data fetching ------------------
#=====================================================================   

def fetch_weather() -> dict:
    '''
    fetchs raw current weather data from openwweathermap api
    '''
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
        "past_days": 1
    }

    logger.info("Fetching weather arrays from Open-Meteo")
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info("Open-Meteo weather data fetched successfully.")
        return data
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch current weather data: %s", e)
        raise

#=====================================================================
# -------------- Hopswork - Running the Pipline ------------------
#=====================================================================       

def run_pipeline() -> None:
    '''
    Orchestrates the hourly features pipeline:
    '''

    import hopsworks
    import pandas as pd

    logger.info("=" * 60)
    logger.info("Pearls AQI Feature Pipeline — Run Started")
    logger.info("=" * 60)

    # connecting to hopsworks:
    logger.info("Connecting to hopswork's project: %s", HOPSWORKS_PROJECT_NAME)

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )

    fs = project.get_feature_store()
    logging.info("Connected to hopswork feature store successfully")

    # getting/creating feature group
    logger.info("Retrieving Feature group")
    feature_group = fs.get_or_create_feature_group(
        name = "aqi_features",
        version = 3, 
        primary_key = ["timestamp"],
        event_time = "timestamp",
        description = "Hourly AQI features for Karachi"
    )

    logger.info("Feature group retrived successfuly")

    # reading last three rows from hopsworks
    logger.info("Reading the last three rows from the feature group")

    try:
        history_df = feature_group.read()

        if history_df.empty:
            logging.warning("Feature group is empty. Intializing edge case")
            historical_rows = []

        else:
            history_df = history_df.sort_values("timestamp", ascending=False)
            historical_rows = history_df.head(3).to_dict(orient="records")
            logger.info("Historical data loded: %d rows retrieved", len(historical_rows))

    except Exception as e:
        logger.warning("Couldnot read history from the feature group: %s. prceeding with edge case", e)
        historical_rows = []

    # fetching the data
    raw_pollution = fetch_air_pollution()
    raw_weather = fetch_weather()

    # Find the most recent hour that actually has pollution data reported
    times = raw_pollution['hourly']['time']
    latest_valid_index = -1
    
    for i in range(len(times) - 1, -1, -1):
        if raw_pollution['hourly']['pm10'][i] is not None:
            latest_valid_index = i
            break
            
    if latest_valid_index == -1:
        logger.error("No valid pollution data found in the Open-Meteo response arrays.")
        sys.exit(1)

    # Extract the single values for that hour to pass to the feature engine
    current_pollution_dict = {
        "pm2_5": raw_pollution['hourly']['pm2_5'][latest_valid_index],
        "pm10": raw_pollution['hourly']['pm10'][latest_valid_index],
        "nitrogen_dioxide": raw_pollution['hourly']['nitrogen_dioxide'][latest_valid_index],
        "ozone": raw_pollution['hourly']['ozone'][latest_valid_index],
        "carbon_monoxide": raw_pollution['hourly']['carbon_monoxide'][latest_valid_index],
        "sulphur_dioxide": raw_pollution['hourly']['sulphur_dioxide'][latest_valid_index]
    }
    
    current_weather_dict = {
        "temperature_2m": raw_weather['hourly']['temperature_2m'][latest_valid_index],
        "relative_humidity_2m": raw_weather['hourly']['relative_humidity_2m'][latest_valid_index],
        "surface_pressure": raw_weather['hourly']['surface_pressure'][latest_valid_index],
        "wind_speed_10m": raw_weather['hourly']['wind_speed_10m'][latest_valid_index],
        "wind_direction_10m": raw_weather['hourly']['wind_direction_10m'][latest_valid_index],
        "cloud_cover": raw_weather['hourly']['cloud_cover'][latest_valid_index],
        "time": times[latest_valid_index] 
    }

    #engineering features:
    logger.info("Engineering features for timestamp: %s", times[latest_valid_index])
    feature_dict = engineer_features(current_pollution_dict, current_weather_dict, historical_rows)

    if feature_dict is None:
        logger.error("Feature engineering returned None. Aborting pipeline insertion.")
        sys.exit(1)
        
    logger.info(
        "Features engineered — Timestamp: %s | AQI: %s | Category: %s",
        feature_dict.get("timestamp"),
        feature_dict.get("aqi"),
        feature_dict.get("category"),
    )

    # Writing the new row in feature group
    logger.info("Writing new feature row to the feature store")
    feature_df = pd.DataFrame([feature_dict])
    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"], utc=True)    

    # Force all numeric columns to float64 to prevent Hopsworks bigint/double mismatch errors
    numeric_columns = [
        "aqi", "pm2_5", "pm10", "no2", "o3", "co", "so2", 
        "temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
        "aqi_change_rate", "pm2_5_rolling_3h"
    ]
    for col in numeric_columns:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype("float64")

    feature_group.insert(feature_df)

    logger.info("Feature row inserted successfully")

    logger.info("=" * 60)
    logger.info("Pearls AQI Feature Pipeline — Run Completed Successfully")
    logger.info("=" * 60)

#=====================================================================
# -------------------------- Main Function --------------------------
#===================================================================== 

def main() -> None:
    '''
    Top-level entry point. Calls run_pipeline()
    '''
    
    try:
        run_pipeline()
    except Exception as e:
        logger.error("=" * 60)
        logger.error("Pipeline run FAILED: %s", e)
        logger.error("=" * 60)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()