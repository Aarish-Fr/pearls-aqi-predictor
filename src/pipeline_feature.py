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
import time

import requests
from dotenv import load_dotenv
from datetime import datetime, timezone

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

MONGO_URI = os.getenv("MONGO_URI")
LAT = os.getenv("KARACHI_LAT")
LON = os.getenv("KARACHI_LON")

# Saftey Guard

_required = {
    "MONGO_URI": MONGO_URI,
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
        "past_days": 1,       
        "forecast_days": 0  
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
        "past_days": 1,       
        "forecast_days": 0    
    }

    logger.info("Fetching weather arrays from Open-Meteo")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            logger.info("Open-Meteo weather data fetched successfully.")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(
                "Attempt %d/%d failed to fetch weather data: %s",
                attempt, max_retries, e
            )
            if attempt < max_retries:
                logger.info("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                logger.error("All %d attempts failed. Raising exception.", max_retries)
                raise

#=====================================================================
# -------------- MongoDB - Running the Pipeline ------------------
#=====================================================================       

def run_pipeline() -> None:
    '''
    Orchestrates the hourly features pipeline:
    '''
    import pymongo
    
    MONGO_URI = os.getenv("MONGO_URI")

    logger.info("=" * 60)
    logger.info("Pearls AQI Feature Pipeline (MongoDB) — Run Started")
    logger.info("=" * 60)

    # Connecting to MongoDB Atlas
    client = None

    try:
        logger.info("Connecting to MongoDB Atlas cluster")
        client = pymongo.MongoClient(MONGO_URI)
        db = client["aqi_project"]
        collection = db["features"]
        logger.info("Connected to MongoDB successfully")

        # Reading last three rows from MongoDB for rolling averages
        logger.info("Reading the last three rows from the features collection")

        try:
            # Fetch the 3 newest rows
            historical_cursor = collection.find().sort("timestamp", -1).limit(3)
            historical_rows = list(historical_cursor)[::-1]

            if not historical_rows:
                logger.warning("MongoDB collection is empty. Initializing edge case.")
            else:
                logger.info("Historical data loaded: %d rows retrieved", len(historical_rows))

        except Exception as e:
            logger.warning("Could not read history from MongoDB: %s. Proceeding with edge case.", e)
            historical_rows = []

        # fetching the data
        raw_pollution = fetch_air_pollution()
        raw_weather = fetch_weather()

        # Find the most recent hour that actually has pollution data reported
        existing_timestamps = set(
            doc["timestamp"] for doc in collection.find({}, {"timestamp": 1})
        )
        logger.info("Found %d existing timestamps in MongoDB.", len(existing_timestamps))
        # Loop through ALL hours in the API response
        times = raw_pollution['hourly']['time']
        inserted_count = 0
        for i, timestamp in enumerate(times):
            # Skip if pm10 is None (data not available for this hour)
            if raw_pollution['hourly']['pm10'][i] is None:
                continue
            # Normalize timestamp to match MongoDB format
            normalized_ts = timestamp + "+00:00" if "+" not in timestamp and "Z" not in timestamp else timestamp
            # Skip if already in MongoDB
            if normalized_ts in existing_timestamps:
                logger.debug("Timestamp %s already exists. Skipping.", normalized_ts)
                continue
            # Extract this hour's data
            current_pollution_dict = {
                "pm2_5":             raw_pollution['hourly']['pm2_5'][i],
                "pm10":              raw_pollution['hourly']['pm10'][i],
                "nitrogen_dioxide":  raw_pollution['hourly']['nitrogen_dioxide'][i],
                "ozone":             raw_pollution['hourly']['ozone'][i],
                "carbon_monoxide":   raw_pollution['hourly']['carbon_monoxide'][i],
                "sulphur_dioxide":   raw_pollution['hourly']['sulphur_dioxide'][i]
            }
            current_weather_dict = {
                "temperature_2m":       raw_weather['hourly']['temperature_2m'][i],
                "relative_humidity_2m": raw_weather['hourly']['relative_humidity_2m'][i],
                "surface_pressure":     raw_weather['hourly']['surface_pressure'][i],
                "wind_speed_10m":       raw_weather['hourly']['wind_speed_10m'][i],
                "wind_direction_10m":   raw_weather['hourly']['wind_direction_10m'][i],
                "cloud_cover":          raw_weather['hourly']['cloud_cover'][i],
                "time":                 timestamp
            }
            logger.info("Engineering features for timestamp: %s", timestamp)
            feature_dict = engineer_features(current_pollution_dict, current_weather_dict, historical_rows)
            if feature_dict is None:
                logger.warning("Feature engineering returned None for %s. Skipping.", timestamp)
                continue
            # Normalize and upsert into MongoDB
            if "+" not in feature_dict["timestamp"] and "Z" not in feature_dict["timestamp"]:
                feature_dict["timestamp"] = feature_dict["timestamp"] + "+00:00"
            collection.update_one(
                {"timestamp": feature_dict["timestamp"]},
                {"$set": feature_dict},
                upsert=True
            )
            inserted_count += 1
            logger.info("Inserted: %s | AQI: %s", feature_dict.get("timestamp"), feature_dict.get("aqi"))
        if inserted_count == 0:
            logger.info("No new hours to insert. API may still be lagging. Exiting cleanly.")
        else:
            logger.info("Successfully inserted %d new rows into MongoDB.", inserted_count)

        logger.info("=" * 60)
        logger.info("Pearls AQI Feature Pipeline — Run Completed Successfully")
        logger.info("=" * 60)
    
    finally:
        # close the MongoDB connection 
        if client is not None:
            client.close()
            logger.info("MongoDB connection closed")

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