"""
Backfill-Pipeline:

Fetches historical data from Open-Meteo for a specified date range,
processes it chronologically to maintain stateful rolling features,
and bulk-upserts it into MongoDB.
"""

import logging
import os
import sys

import requests
from dotenv import load_dotenv
import pymongo
from pymongo.errors import BulkWriteError
from pymongo import UpdateOne 

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

def fetch_historical_data(start_date: str, end_date: str):
    """
    Fetches air quality and weather data for an arbitrary date range.
    """

    logger.info("Fetching historical data from %s to %s ...", start_date, end_date)
    
    poll_url    = "https://air-quality-api.open-meteo.com/v1/air-quality"
    poll_params = {
        "latitude":   LAT,
        "longitude":  LON,
        "hourly":     "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone":   "UTC",
        "start_date": start_date,                     
        "end_date":   end_date,                       
    }
    
    weather_url    = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude":   LAT,
        "longitude":  LON,
        "hourly":     "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone":   "UTC",
        "start_date": start_date,                     
        "end_date":   end_date,                       
    }

    try:
        poll_resp = requests.get(poll_url, params=poll_params, timeout=30)
        poll_resp.raise_for_status()
        raw_pollution = poll_resp.json()
        
        weather_resp = requests.get(weather_url, params=weather_params, timeout=30)
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

def run_backfill(start_date: str, end_date: str):
    logger.info("=" * 60)
    logger.info("Pearls AQI Backfill Pipeline (MongoDB) Started")
    logger.info("Date range: %s → %s", start_date, end_date)
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
        raw_pollution, raw_weather = fetch_historical_data(start_date, end_date)
        times = raw_pollution['hourly']['time']
        total_hours = len(times)
        
        logger.info("Processing %d hours of data chronologically...", total_hours)
        
        #Processing data Chronologically
        backfill_history = []
        skipped_count = 0

        for i in range(total_hours):

            if i > 0 and i % 500 == 0:
                logger.info(
                    "Progress: %d / %d hours processed | %d valid rows engineered so far | %d skipped",
                    i, total_hours, len(backfill_history), skipped_count
                )

            # Safeguard: Ensure the timestamps perfectly align before extracting
            if raw_pollution['hourly']['time'][i] != raw_weather['hourly']['time'][i]:
                logger.error("API Desync detected at index %d. Aborting.", i)
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

            historical_rows_to_pass = backfill_history[-3:] if backfill_history else []
            
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

        if not backfill_history:
            logger.warning("No valid rows produced. Nothing to upsert. Exiting.")
            return

        # Bulk Write to DB
        logger.info(
            "Initiating bulk upsert to MongoDB (%d rows) ...",
            len(backfill_history)
        )

        BATCH_SIZE   = 1000                          
        total_upserted = 0
        
        for batch_start in range(0, len(backfill_history), BATCH_SIZE):
            batch = backfill_history[batch_start : batch_start + BATCH_SIZE]

            operations = [
                UpdateOne(
                    {"timestamp": row["timestamp"]},
                    {"$set": row},
                    upsert=True
                )
                for row in batch
            ]

            result = collection.bulk_write(operations, ordered=False)
            total_upserted += result.upserted_count + result.modified_count

            logger.info(
                "Batch %d/%d complete — upserted: %d | modified: %d",
                (batch_start // BATCH_SIZE) + 1,
                -(-len(backfill_history) // BATCH_SIZE),
                result.upserted_count,
                result.modified_count
            )

        logger.info(
            "Bulk upsert complete. %d total rows written to MongoDB.",
            total_upserted
        )


        logger.info("=" * 60)
        logger.info("Pearls AQI Backfill Pipeline — Run Completed Successfully")
        logger.info("=" * 60)
    
    finally:
        if client is not None:
            client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    run_backfill(
        start_date = "2024-06-01",
        end_date   = "2026-05-06"
    )