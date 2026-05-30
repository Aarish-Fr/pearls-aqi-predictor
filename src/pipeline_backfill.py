"""
Backfill-Pipeline:

Fetches the last 90 days of historical data from Open-Meteo,
processes it chronologically to maintain stateful rolling features,
and bulk-inserts it into MongoDB.
"""

import logging
import os
import sys

import requests
from dotenv import load_dotenv
import pymongo
from pymongo.errors import BulkWriteError

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

_required = {
    "MONGO_URI": MONGO_URI,
    "KARACHI_LAT": LAT,
    "KARACHI_LON": LON,
}

_missing = [key for key, val in _required.items() if not val]

if _missing:
    logger.error("Missing required environment variable: %s", _missing)
    sys.exit(1)

#=====================================================================
# ------------- Historical Data Fetching ------------------
#=====================================================================

def fetch_historical_data(days=92):
    logger.info("Fetching %d days of historical data from Open-Meteo...", days)
    
    poll_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    poll_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "UTC",
        "past_days": days,
        "forecast_days": 0
    }
    
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
        "past_days": days,
        "forecast_days": 0
    }

    try:
        poll_resp = requests.get(poll_url, params=poll_params, timeout=15)
        poll_resp.raise_for_status()
        raw_pollution = poll_resp.json()
        
        weather_resp = requests.get(weather_url, params=weather_params, timeout=15)
        weather_resp.raise_for_status()
        raw_weather = weather_resp.json()
        
        logger.info("Historical data fetched successfully.")
        return raw_pollution, raw_weather
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch historical data: %s", e)
        sys.exit(1)

#=====================================================================
# -------------- Backfill Logic ------------------
#=====================================================================       

def run_backfill():
    logger.info("=" * 60)
    logger.info("Pearls AQI Backfill Pipeline (MongoDB) Started")
    logger.info("=" * 60)
    
    #Connecting to MongoDB Atlas
    client = None
    try:
        logger.info("Connecting to MongoDB Atlas...")
        client = pymongo.MongoClient(MONGO_URI)
        db = client["aqi_project"]
        collection = db["features"]
        logger.info("Connected to MongoDB successfully")

        #Fetching Data
        raw_pollution, raw_weather = fetch_historical_data(days=92)
        times = raw_pollution['hourly']['time']
        total_hours = len(times)
        
        logger.info("Processing %d hours of data chronologically...", total_hours)
        backfill_history = []
        skipped_count = 0

        #Processing data Chronologically
        for i in range(total_hours):

            if i > 0 and i % 500 == 0:
                logger.info(
                    "Progress: %d / %d hours processed | %d valid rows engineered so far | %d skipped",
                    i, total_hours, len(backfill_history), skipped_count
                )

            # Safeguard: Ensure the timestamps perfectly align before extracting
            if raw_pollution['hourly']['time'][i] != raw_weather['hourly']['time'][i]:
                logger.error("API Desync detected at index %d. Aborting to prevent corrupt data.", i)
                client.close()
                sys.exit(1)

            # Safeguard: Skip hours where pollution OR weather data is explicitly null
            if raw_pollution['hourly']['pm10'][i] is None or raw_weather['hourly']['temperature_2m'][i] is None:
                skipped_count += 1
                continue
                
            current_pollution_dict = {
                "pm2_5":             raw_pollution['hourly']['pm2_5'][i],
                "pm10":              raw_pollution['hourly']['pm10'][i],
                "nitrogen_dioxide":  raw_pollution['hourly']['nitrogen_dioxide'][i],
                "ozone":             raw_pollution['hourly']['ozone'][i],
                "carbon_monoxide":   raw_pollution['hourly']['carbon_monoxide'][i],
                "sulphur_dioxide":   raw_pollution['hourly']['sulphur_dioxide'][i],
            }
            
            current_weather_dict = {
                "temperature_2m":       raw_weather['hourly']['temperature_2m'][i],
                "relative_humidity_2m": raw_weather['hourly']['relative_humidity_2m'][i],
                "surface_pressure":     raw_weather['hourly']['surface_pressure'][i],
                "wind_speed_10m":       raw_weather['hourly']['wind_speed_10m'][i],
                "wind_direction_10m":   raw_weather['hourly']['wind_direction_10m'][i],
                "cloud_cover":          raw_weather['hourly']['cloud_cover'][i],
                "time":                 times[i],
            }

            historical_rows_to_pass = backfill_history[-3:] if len(backfill_history) > 0 else []
            
            logger.setLevel(logging.WARNING)
            feature_dict = engineer_features(current_pollution_dict, current_weather_dict, historical_rows_to_pass)
            logger.setLevel(logging.INFO)
            
            if feature_dict:
                # Ensuring proper UTC-aware ISO timestamp string for MongoDB sorting
                if "+" not in feature_dict["timestamp"] and "Z" not in feature_dict["timestamp"]:
                    feature_dict["timestamp"] = feature_dict["timestamp"] + "+00:00"
                
                backfill_history.append(feature_dict)

        logger.info(
            "Processing complete. %d valid rows engineered | %d hours skipped (null data).",
            len(backfill_history), skipped_count
        )

        # Upsert Loop (Heals Corrupted Data)
        logger.info("Initiating upsert to MongoDB to heal corrupted records...")
        upsert_count = 0
        
        for feature_dict in backfill_history:
            collection.update_one(
                {"timestamp": feature_dict["timestamp"]},
                {"$set": feature_dict},
                upsert=True
            )
            upsert_count += 1
            
        logger.info("Successfully upserted %d rows. Bad data overwritten.", upsert_count)

        logger.info("=" * 60)
        logger.info("Pearls AQI Backfill Pipeline — Run Completed Successfully")
        logger.info("=" * 60)
    
    finally:
        if client is not None:
            client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    try:
        run_backfill()
    except Exception as e:
        logger.error("Backfill failed: %s", e)
        sys.exit(1)