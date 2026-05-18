"""
Feature-Pipeline:

Orchestrates the hourly data collection lifescycle
- Loads credentials
- fetches raw data
- read recnet record from the Hopswork
- engineers features
- write new row to features group
"""

import logging
import os
import sys

import requests
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

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_AIR_URL = os.getenv("OPENWEATHER_AIR_URL")
OPENWEATHER_BASE_URL = os.getenv("OPENWEATHER_BASE_URL")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
LAT = os.getenv("KARACHI_LAT")
LON = os.getenv("KARACHI_LON")

# Saftey Guard

_required = {
    "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
    "OPENWEATHER_AIR_URL": OPENWEATHER_AIR_URL,
    "OPENWEATHER_BASE_URL": OPENWEATHER_BASE_URL,
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
    fetches raw air pollution data from the OpenWeatheMap API
    '''

    params = {
        'lat': LAT,
        'lon': LON,
        'appid': OPENWEATHER_API_KEY
    }

    logger.info("Fetching air pollution data from API")

    try:
        response = requests.get(OPENWEATHER_AIR_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info("Air pollution data fetch successfuly.")
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
    params = {
        "lat": LAT,
        "lon": LON,
        "appid": OPENWEATHER_API_KEY
    }

    logger.info("Fetching current weather data from OpenWeatherMap API")
    try:
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(
            "Weather data fetched successfully. Temp: %.1fK, Humidity: %s%%",
            data["main"]["temp"],
            data["main"]["humidity"]
        )
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
    logger.info("Retriving Feature group")
    feature_group = fs.get_or_create_feature_group(
        name = "aqi_features",
        version = 2,
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

    #engineering features:
    logger.info("Engineering features...")
    feature_dict = engineer_features(raw_pollution, raw_weather, historical_rows)

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
    feature_group.insert(feature_df)

    logger.info("feature row inserted successfuly")

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