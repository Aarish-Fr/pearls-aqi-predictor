'''
Backend:
Pearls AQI Predictor — backend.py
Handles: constants, MongoDB, model loading, feature building, predictions
'''

import json
import logging
import os
import tempfile
import joblib
import requests 
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient

#=====================================================================
# ------------------ Logging Configuration ------------------
#=====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pearls_aqi.backend")

#=====================================================================
# ------------------ Loading Credentials & Env ------------------
#=====================================================================
load_dotenv()

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
KARACHI_LAT = float(os.getenv("KARACHI_LAT", "24.8607"))
KARACHI_LON = float(os.getenv("KARACHI_LON", "67.0011"))
MONGO_URI = os.getenv("MONGO_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

EXPERIMENT_NAME  = "Default"
REQUIRED_ARTIFACTS = [
    "xgb_model.pkl",
    "ridge_model.pkl",
    "mlp_model.keras",
    "scaler.pkl",
    "feature_columns.json",
    "metrics.json"
]

#=====================================================================
# ------------------ AQI Constants & Mappings ------------------
#=====================================================================
AQI_BINS   = [0, 50, 100, 150, 200, 300, 500]
AQI_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

AQI_COLORS = {
    "Good":                           "#00e400",
    "Moderate":                       "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy":                      "#ff0000",
    "Very Unhealthy":                 "#8f3f97",
    "Hazardous":                      "#7e0023",
}

WEATHER_RENAME = {
    "temperature_2m":             "temperature",
    "relative_humidity_2m":       "humidity",
    "wind_speed_10m":             "wind_speed",
    "wind_direction_10m":         "wind_direction",
    "precipitation":              "precipitation",
    "surface_pressure":           "pressure",
    "shortwave_radiation":        "solar_radiation",
    "et0_fao_evapotranspiration": "evapotranspiration",
}

HEALTH_RECOMMENDATIONS = {
    "Good": (
        "Air quality is satisfactory. "
        "Enjoy outdoor activities freely."
    ),
    "Moderate": (
        "Air quality is acceptable. "
        "Unusually sensitive individuals should consider limiting "
        "prolonged outdoor exertion."
    ),
    "Unhealthy for Sensitive Groups": (
        "People with respiratory or heart conditions, the elderly, "
        "and children should limit prolonged outdoor exertion."
    ),
    "Unhealthy": (
        "Everyone may begin to experience health effects. "
        "Sensitive groups should avoid outdoor exertion. "
        "Others should limit prolonged outdoor activity."
    ),
    "Very Unhealthy": (
        "Health alert: everyone may experience more serious health effects. "
        "Avoid outdoor exertion. Keep windows closed."
    ),
    "Hazardous": (
        "Health warning of emergency conditions. "
        "The entire population is likely to be affected. "
        "Stay indoors, use air purifiers, wear N95 masks if you must go outside."
    ),
}

#=====================================================================
# ------------------ MLflow Fallback Chain ------------------
#=====================================================================
def _configure_mlflow():
    '''
    Point MLflow at DagsHub and return an authenticated client.
    '''
    import mlflow
    from mlflow.tracking import MlflowClient

    username = os.getenv("DAGSHUB_USERNAME")
    token    = os.getenv("DAGSHUB_TOKEN")
    
    tracking_uri = f"https://dagshub.com/{username}/pearls-aqi-predictor.mlflow"
    
    if username and token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    logger.info("MLflow configured. Tracking URI: %s", tracking_uri)
    
    return client

def _run_has_all_artifacts(client, run_id: str) -> bool:
    '''
    Return True only if every required artifact exists in this run.
    '''
    try:
        artifacts = [
            a.path
            for a in client.list_artifacts(run_id)
        ]

        missing = [f for f in REQUIRED_ARTIFACTS if f not in artifacts]
        
        if missing:
            logger.warning(
                "Run %s is missing artifacts: %s — skipping.", run_id, missing
            )

            return False
        
        return True
    
    except Exception as exc:
        logger.warning("Could not list artifacts for run %s: %s", run_id, exc)
        
        return False
    
def _find_best_run(client) -> str:
    '''
    Walk runs newest-first. Return the first run_id that is:
        1. Status == FINISHED
        2. Contains all 5 required artifact files
    Raises RuntimeError if no valid run is found.
    '''

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' not found on DagsHub."
        )
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],   
    )

    if not runs:
        raise RuntimeError("No MLflow runs found in experiment 'Default'.")
    
    for run in runs:
        run_id = run.info.run_id
        status = run.info.status
        
        if status != "FINISHED":
            logger.warning(
                "Run %s has status '%s' — skipping.", run_id, status
            )
            continue
            
        if _run_has_all_artifacts(client, run_id):
            logger.info("Valid run found: %s", run_id)
            return run_id
        
    raise RuntimeError(
        "No complete, finished MLflow run found. "
        "Check your DagsHub experiment for failed or incomplete runs."
    )

#=====================================================================
# ----------------------- Model Loader  -----------------------
#=====================================================================
def load_all_models() -> dict:
    '''
    Download all artifacts from the best valid MLflow run into a temp
    directory, load them into memory, and return a dict.
    '''

    logger.info("Starting model load sequence...")
    
    client = _configure_mlflow()
    run_id = _find_best_run(client)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Downloading artifacts from run %s into %s", run_id, tmp_dir)
        
        import mlflow.artifacts
        from keras.models import load_model

        for artifact_name in [
            "xgb_model.pkl", 
            "ridge_model.pkl", 
            "mlp_model.keras", 
            "scaler.pkl", 
            "feature_columns.json",
            "metrics.json"
        ]:
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{artifact_name}",
                dst_path=tmp_dir
            )
            logger.info("Downloaded: %s", artifact_name)
        
        # Load models from the temp directory
        xgb   = joblib.load(os.path.join(tmp_dir, "xgb_model.pkl"))
        ridge = joblib.load(os.path.join(tmp_dir, "ridge_model.pkl"))
        mlp   = load_model(os.path.join(tmp_dir, "mlp_model.keras"))
        scaler = joblib.load(os.path.join(tmp_dir, "scaler.pkl"))
        
        with open(os.path.join(tmp_dir, "feature_columns.json")) as f:
            feature_columns = json.load(f)

        with open(os.path.join(tmp_dir, "metrics.json")) as f:
            metrics = json.load(f)
    
    logger.info(
        "All models loaded successfully. Feature vector length: %d",
        len(feature_columns),
    )

    logger.info("Metrics loaded. Models evaluated: %s", list(metrics.keys()))

    
    
    return {
        "xgb":             xgb,
        "ridge":           ridge,
        "mlp":             mlp,
        "scaler":          scaler,
        "feature_columns": feature_columns,
        "metrics":         metrics,
        "run_id":          run_id,
    }

#=====================================================================
# ------------------ MongoDB Data Loader ------------------
#=====================================================================
MONGO_DB_NAME         = "aqi_project"   
MONGO_COLLECTION_NAME = "features"      

def _get_mongo_client() -> MongoClient:
    '''
    Create and return an authenticated MongoDB client.
    '''
    mongo_uri = os.getenv("MONGO_URI")
    
    if not mongo_uri:
        raise RuntimeError(
            "MONGO_URI is not set. Add it to your .env file or Hugging Face Secrets."
        )
    
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    
    # Force a real connection check — fails fast if URI is wrong
    client.admin.command("ping")
    logger.info("MongoDB connection established successfully.")
    
    return client

def fetch_latest_actuals(hours: int = 48) -> pd.DataFrame:
    '''
    Fetch the latest `hours` records from MongoDB, sorted newest-first.
    '''

    logger.info("Fetching latest %d records from MongoDB...", hours)
    
    client  = _get_mongo_client()
    db      = client[MONGO_DB_NAME]
    col     = db[MONGO_COLLECTION_NAME]
    cursor = (
        col.find({}, {"_id": 0})        
            .sort("timestamp", -1)       
            .limit(hours)
    )
    records = list(cursor)
    client.close()
    
    if not records:
        raise RuntimeError(
            "MongoDB collection is empty. "
            "Check that your ingestion pipeline has run successfully."
        )
    
    df = pd.DataFrame(records)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Fetched %d records. Range: %s → %s",
        len(df),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
    )
    return df

def fetch_latest_row() -> pd.Series:
    '''
    Fetch only the single most recent document from MongoDB.
    Used by the feature builder as the base row for inference.
    '''

    logger.info("Fetching latest single row from MongoDB for inference...")
    
    client = _get_mongo_client()
    db     = client[MONGO_DB_NAME]
    col    = db[MONGO_COLLECTION_NAME]
    record = col.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
    client.close()
    
    if record is None:
        raise RuntimeError("MongoDB collection is empty — cannot build feature vector.")
    
    row = pd.Series(record)
    row["timestamp"] = pd.to_datetime(row["timestamp"], utc=True)
    
    logger.info("Latest row timestamp: %s", row["timestamp"])
    
    return row

#=====================================================================
# ------------------ Open-Meteo Forecast Fetcher ------------------
#=====================================================================
import requests
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

OPEN_METEO_PARAMS = {
    "latitude":        KARACHI_LAT,
    "longitude":       KARACHI_LON,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
    ],
    # "forecast_days":   4,  # Removed in favor of dynamic start_date/end_date
    "timezone":        "Asia/Karachi",
}

def _fetch_open_meteo(latest_date_str: str) -> dict:
    '''
    Fetch 3-day forecast from Open-Meteo for Karachi.
    Returns a dict with keys: day1, day2, day3.
    Each value is a flat dict of weather features (daily means) for that day.
    '''

    logger.info("Fetching 3-day forecast from Open-Meteo starting from %s...", latest_date_str)
    
    import datetime
    start_dt = datetime.datetime.strptime(latest_date_str, "%Y-%m-%d")
    end_dt   = start_dt + datetime.timedelta(days=3)
    
    params = OPEN_METEO_PARAMS.copy()
    params["start_date"] = start_dt.strftime("%Y-%m-%d")
    params["end_date"]   = end_dt.strftime("%Y-%m-%d")
    
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    
    except requests.exceptions.Timeout:
        raise RuntimeError("Open-Meteo API timed out after 10 seconds.")
    
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open-Meteo API request failed: {exc}")
    
    try:
        hourly = data["hourly"]
        # Daily values — computed as 24-hour means 
        # Index 0 = today (already passed)
        forecast = {}
        
        for i, suffix in enumerate(["day1", "day2", "day3"], start=1):
            hour_start = i * 24
            hour_end   = hour_start + 24
            
            forecast[suffix] = {
                f"temp_{suffix}":       float(np.mean(hourly["temperature_2m"][hour_start:hour_end])),
                f"humidity_{suffix}":   float(np.mean(hourly["relative_humidity_2m"][hour_start:hour_end])),
                f"pressure_{suffix}":   float(np.mean(hourly["surface_pressure"][hour_start:hour_end])),
                f"wind_speed_{suffix}": float(np.mean(hourly["wind_speed_10m"][hour_start:hour_end])),
                f"wind_deg_{suffix}":   float(np.mean(hourly["wind_direction_10m"][hour_start:hour_end])),
                f"clouds_{suffix}":     float(np.mean(hourly["cloud_cover"][hour_start:hour_end])),
            }

        logger.info("Open-Meteo forecast fetched and averaged successfully.")
        
        return forecast
    
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Open-Meteo response structure: {exc}")

#=====================================================================
# ------------------ Feature Builder ------------------
#=====================================================================
def build_feature_vector(
    df_48h: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    '''
    Build a single-row inference feature vector aligned to feature_columns.
    '''
    
    logger.info("Building inference feature vector...")
    
    # Latest row as base
    latest = df_48h.iloc[-1].copy()
    
    features = {}
    # Direct MongoDB features 
    direct_cols = [
        "aqi", "aqi_change_rate", "clouds", "co", "humidity",
        "is_weekend", "no2", "o3", "pm10", "pm2_5",
        "pm2_5_rolling_3h", "pressure", "so2", "temp",
        "wind_deg", "wind_speed",
    ]

    for col in direct_cols:
        features[col] = float(latest[col])
    
    #  Cyclical encodings 
    hour        = int(latest["hour"])
    month       = int(latest["month"])
    day_of_week = int(latest["day_of_week"])
    features["hour_sin"]  = float(np.sin(2 * np.pi * hour  / 24))
    features["hour_cos"]  = float(np.cos(2 * np.pi * hour  / 24))
    features["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    features["month_cos"] = float(np.cos(2 * np.pi * month / 12))
    features["dow_sin"]   = float(np.sin(2 * np.pi * day_of_week / 7))
    features["dow_cos"]   = float(np.cos(2 * np.pi * day_of_week / 7))
    
    logger.debug(
        "Cyclical features — hour=%d sin=%.4f cos=%.4f",
        hour, features["hour_sin"], features["hour_cos"],
    )

    # On-the-fly windowing 
    last_24  = df_48h["aqi"].iloc[-24:]
    prev_24  = df_48h["aqi"].iloc[-48:-24]
    features["aqi_rolling_24h"]   = float(last_24.mean())
    features["prev_day_max_aqi"]  = float(last_24.max())
    features["prev_2day_max_aqi"] = float(prev_24.max())
    
    logger.debug(
        "Windowed features — rolling_24h=%.2f prev_day_max=%.2f prev_2day_max=%.2f",
        features["aqi_rolling_24h"],
        features["prev_day_max_aqi"],
        features["prev_2day_max_aqi"],
    )

    # Open-Meteo forecast features (aligned to latest MongoDB timestamp)
    latest_date_str = pd.to_datetime(latest["timestamp"]).strftime("%Y-%m-%d")
    forecast = _fetch_open_meteo(latest_date_str)
    
    for suffix in ["day1", "day2", "day3"]:
        features.update(forecast[suffix])
    
    missing = [c for c in feature_columns if c not in features]
    if missing:
        raise RuntimeError(
            f"Feature vector is missing {len(missing)} column(s): {missing}\n"
            "This means MongoDB schema or Open-Meteo response changed. "
            "Check your pipeline."
        )
    
    X = pd.DataFrame([features])[feature_columns]
    
    logger.info(
        "Feature vector built. Shape: %s. All %d columns present.",
        X.shape, len(feature_columns),
    )
    return X

#=====================================================================
# ------------------ Prediction Engine ------------------
#=====================================================================

def get_aqi_label(aqi_value: float) -> str:
    '''
    Map a numeric AQI value to its category label based on EPA bins.
    '''
    for i in range(len(AQI_BINS) - 1):
        if AQI_BINS[i] <= aqi_value < AQI_BINS[i + 1]:
            return AQI_LABELS[i]
        
    return AQI_LABELS[-1]  # Hazardous if above 300


def run_predictions(
    X: pd.DataFrame,
    models: dict,
) -> dict:
    '''
    Run all three models on the feature vector and return predictions.
    '''
    
    logger.info("Running predictions on feature vector...")

    xgb = models["xgb"]
    ridge = models["ridge"]
    mlp = models["mlp"]
    scaler = models["scaler"]

    # XGBoost — no scaling
    xgb_preds = xgb.predict(X)[0]   

    # Ridge and MLP — full matrix scaled
    X_scaled    = scaler.transform(X)
    ridge_preds = ridge.predict(X_scaled)[0]
    mlp_preds   = mlp.predict(X_scaled, verbose=0)[0]

    def _build_result(preds) -> dict:
        day1 = float(np.clip(preds[0], 0, 500))
        day2 = float(np.clip(preds[1], 0, 500))
        day3 = float(np.clip(preds[2], 0, 500))
        
        return {
            "day1":   day1,
            "day2":   day2,
            "day3":   day3,
            "labels": {
                "day1": get_aqi_label(day1),
                "day2": get_aqi_label(day2),
                "day3": get_aqi_label(day3),
            }
        }

    results = {
        "xgb":      _build_result(xgb_preds),
        "ridge":    _build_result(ridge_preds),
        "mlp":      _build_result(mlp_preds),
    }

    # Featured model = XGBoost (wins 2/3 horizons on MAE)
    results["featured"] = results["xgb"]

    logger.info(
        "Predictions complete — XGBoost: Day1=%.1f (%s) | Day2=%.1f (%s) | Day3=%.1f (%s)",
        results["xgb"]["day1"], results["xgb"]["labels"]["day1"],
        results["xgb"]["day2"], results["xgb"]["labels"]["day2"],
        results["xgb"]["day3"], results["xgb"]["labels"]["day3"],
    )

    return results
