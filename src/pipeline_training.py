"""
Training-Pipeline:

- Trains Ridge, Random Forest, and MLP models on historical AQI feature data.
- Logs metrics and saves artifacts to DagsHub via MLflow.
- Runs daily via GitHub Actions.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
import joblib
import tensorflow as tf
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import dagshub
import json as json_module

from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


#=====================================================================
# ------------------ Enviroment Setup ------------------
#=====================================================================

load_dotenv()

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    log.error("MONGO_URI not found in environment. Exiting.")
    sys.exit(1)

#=====================================================================
# ------------------ Fetching Features from MongoDB ------------------
#=====================================================================

def fetch_features() -> pd.DataFrame:
    """
    Connects to MongoDB Atlas
    fetches all documents from the features collection sorted chronologically (oldest first)
    returns a clean pandas DataFrame ready for feature construction.
    """

    client = None
    try:
        log.info("Connecting to MongoDB Atlas ...")
        client = MongoClient(MONGO_URI)

        db = client["aqi_project"]
        collection = db["features"]

        logging.info("Fetching all documents sorted by timestamp ASC...")
        cursor = collection.find().sort("timestamp", 1)
        documents = list(cursor)

        if not documents:
            log.error("No documents returned from MongoDB. Exiting.")
            sys.exit(1)

        log.info("Fetched %d documents from MongoDB.", len(documents))

        df = pd.DataFrame(documents)

        # dropping id column from the dataframe
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
            log.info("Dropped internal MongoDB '_id' column.")

        # - pipeline guard: Shape Verification
        expected_rows = 553
        expected_cols = 23

        if df.shape[1] != expected_cols:
            log.error("Column count mismatched. Expected %d, got %d. Exiting", expected_cols, df.shape[1])
            sys.exit(1)

        if df.shape[0] < expected_rows:
            log.warning("Row count is lower than expected. Expected %d rows, got %d. Continuing with available rows", expected_rows, df.shape[0])

        else:
            log.info("Shape verified: %d rows x %d columns", df.shape[0], df.shape[1])

        # - pipeline guard: Null value check
        nul_count = df.isnull().sum().sum()
        if nul_count > 0:
            log.error("Null values detected: %d total across %d columns. Exiting", nul_count, df.isnull().any().sum())
            sys.exit(1)
        
        log.info("Null check passed - zero null values confirmed")

        # - pipeline guard: Timestamp ordering check
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if not df["timestamp"].is_monotonic_increasing:
            log.error(
                "Timestamp column is not monotonically increasing. "
                "Row order cannot be trusted for chronological split. Exiting."
            )
            sys.exit(1)

        log.info("Timestamp ordering verified. Monotonically increasing.")
        log.info("Date range: %s → %s", df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

        log.info("====== Clean DataFrame ready ======")

        return df
    
    finally:
        if client:
            client.close()
            log.info("MongoDB connection closed.")

#=====================================================================
# -------------------- Constructing X AND y --------------------------
#=====================================================================

COLUMNS_TO_DROP = [
    "timestamp",
    "category",
    "dominant_pollutant",
    "wind_speed_category"
]

def construct_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Constructs the feature matrix X and target vector y from the raw DataFrame.

    y: next hour's AQI — created by shifting the aqi column back by one row.
    X: all columns except the four dropped columns defined in COLUMNS_TO_DROP.
    """

    log.info("Constructing X and y ...")

    # Cyclical Time encoding:
    log.info("Applying cyclical encoding to time features...")

    # Hour encoding (period = 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Month encoding (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Day of week encoding (period = 7)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Droping original raw time columns 
    df.drop(columns=["hour", "month", "day_of_week"], inplace=True)

    log.info("Cyclical encoding applied.")

    # ---- Multi-output label construction ----
    # Pass 1: compute daily max AQI lookup table
    daily_max = df.groupby(df["timestamp"].dt.normalize())["aqi"].max()
    log.info("Daily max lookup table built — %d unique calendar days.", len(daily_max))

    # Pass 2: for each row, look up the max AQI of the next 3 calendar days
    row_date = df["timestamp"].dt.normalize()
    df["y_day1"] = (row_date + pd.Timedelta(days=1)).map(daily_max)
    df["y_day2"] = (row_date + pd.Timedelta(days=2)).map(daily_max)
    df["y_day3"] = (row_date + pd.Timedelta(days=3)).map(daily_max)

    # Feature: previous calendar day's max AQI
    df["prev_day_max_aqi"] = (row_date - pd.Timedelta(days=1)).map(daily_max)
    # Feature: two calendar days ago max AQI
    df["prev_2day_max_aqi"] = (row_date - pd.Timedelta(days=2)).map(daily_max)
    # Feature: 24-hour rolling mean AQI (captures full-day context)
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=24).mean()

    # ---- Future weather context features ----
    weather_vars = ["temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"]
    
    for var in weather_vars:
        daily_mean = df.groupby(df["timestamp"].dt.normalize())[var].mean()
        
        df[f"{var}_day1"] = (row_date + pd.Timedelta(days=1)).map(daily_mean)
        df[f"{var}_day2"] = (row_date + pd.Timedelta(days=2)).map(daily_mean)
        df[f"{var}_day3"] = (row_date + pd.Timedelta(days=3)).map(daily_mean)
    
    log.info("Future weather context features added — %d new columns.", len(weather_vars) * 3)

    # Drop rows where any future label is missing (last ~3 calendar days)
    before_drop = len(df)
    df = df.dropna(subset=["y_day1", "y_day2", "y_day3", "prev_day_max_aqi", "prev_2day_max_aqi", "aqi_rolling_24h", "temp_day1", "temp_day2", "temp_day3"])
    after_drop = len(df)
    log.info(
        "Dropped %d rows with incomplete future labels. %d rows remaining.",
        before_drop - after_drop, after_drop
    )

    # Separate y (3-column DataFrame) and aligned timestamps
    y = df[["y_day1", "y_day2", "y_day3"]].reset_index(drop=True)
    timestamps = df["timestamp"].reset_index(drop=True)


    X = df.drop(columns=COLUMNS_TO_DROP + ["y_day1", "y_day2", "y_day3"]).reset_index(drop=True)

    # - pipeline guard: Shape verification
    expected_feature_count = 43
    if X.shape[1] != expected_feature_count:
        log.error("Feature count missmatched. Expected %d, got %d features. Check COLUMNS_TO_DROP. Exiting", expected_feature_count, X.shape[1])
        sys.exit(1)

    if X.shape[0] != y.shape[0]:
        log.error("X and y row count mismatched. X has %d rows, y has %d rows. Alignment is broken. Exiting", X.shape[0], y.shape[0])
        sys.exit(1)

    log.info("X shape: (%d, %d)  |  y shape: (%d, %d)", X.shape[0], X.shape[1], y.shape[0], y.shape[1])

    if y.shape[1] != 3:
        log.error("y should have 3 label columns, got %d. Exiting.", y.shape[1])
        sys.exit(1)


    # - pipeline guard: NaN check on y:
    nan_count = y.isnull().sum().sum()
    if nan_count > 0:
        log.error("y contains %d NaN values after label construction. Exiting.", nan_count)
        sys.exit(1)

    log.debug("NaN check passed: y contains zero NaN values.")

    # - pipeline guard: NaN check on X
    nan_count_X = X.isnull().sum().sum()
    if nan_count_X > 0:
        log.error("X contains %d NaN values. Check upstream feature pipeline. Exiting.", nan_count_X)
        sys.exit(1)

    log.debug("NaN check passed - X contains zero NaN values.")

    # Sanity Check:
        # Label sanity check:
    log.debug("Label sanity check:")
    log.debug("  Row 0 date: %s", timestamps.iloc[0].date())
    log.debug(
        "  y_day1=%.1f | y_day2=%.1f | y_day3=%.1f",
        y.iloc[0]["y_day1"], y.iloc[0]["y_day2"], y.iloc[0]["y_day3"]
    )
    log.debug(
        "Cyclical sanity check (row 0): hour_sin = %.4f | hour_cos = %.4f",
        X["hour_sin"].iloc[0],
        X["hour_cos"].iloc[0]
    )

    log.info("Data validation passed: Zero NaNs, labels aligned, cyclical encoding verified.")

    # saving feature column list:
    feature_columns = list(X.columns)
    
    os.makedirs("artifacts", exist_ok=True)
    
    with open("artifacts/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    log.info("Feature columns saved to artifacts/feature_columns.json")
    log.info("Features: %s", feature_columns)
    log.info("====== X and y ready ======")

    return X, y, timestamps


#=====================================================================
# ------------------- Train / Val / Test Split -----------------------
#=====================================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def split_data(
        X: pd.DataFrame,
        y: pd.DataFrame,
        timestamps: pd.Series
) -> tuple:
    """
    Splits X and y into train, validation, and test sets 
    using a chronological positional slice.
    """
    
    log.info("Intiating chronological train/val/test split ...")
    n = len(X)

    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Feature matrix splits
    X_train = X.iloc[:train_end]
    X_val   = X.iloc[train_end:val_end]
    X_test  = X.iloc[val_end:]

    # Target vector splits
    y_train = y.iloc[:train_end]
    y_val   = y.iloc[train_end:val_end]
    y_test  = y.iloc[val_end:]

    # Timestamp splits — for logging and date range verification only
    ts_train = timestamps.iloc[:train_end]
    ts_val   = timestamps.iloc[train_end:val_end]
    ts_test  = timestamps.iloc[val_end:]

    # reetting indecies
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    ts_train = ts_train.reset_index(drop=True)
    ts_val   = ts_val.reset_index(drop=True)
    ts_test  = ts_test.reset_index(drop=True)

    # - Guard: Row count must sum to total
    total_rows = len(X_train) + len(X_val) + len(X_test)
    if total_rows != n:
        log.error("Split row count mismatch. Expected %d, got %d rows. Exiting.", n, total_rows)
        sys.exit(1)

    # - Guard: X and y must have identical row counts in each split.
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test)
    ]:
        if len(X_split) != len(y_split):
            log.error(
                "%s set X/y misalignment. X has %d rows, y has %d rows. Exiting.", split_name, len(X_split), len(y_split)
            )
            sys.exit(1)

    # - Guard: No temporal overlap between splits
    if ts_train.iloc[-1] >= ts_val.iloc[0]:
        log.error(
            f"Temporal overlap detected between train and val sets. "
            f"Train ends: {ts_train.iloc[-1]} | "
            f"Val starts: {ts_val.iloc[0]}. Exiting."
        )
        sys.exit(1)

    if ts_val.iloc[-1] >= ts_test.iloc[0]:
        log.error(
            f"Temporal overlap detected between val and test sets. "
            f"Val ends: {ts_val.iloc[-1]} | "
            f"Test starts: {ts_test.iloc[0]}. Exiting."
        )
        sys.exit(1)

    # LOG Split summary:
    log.info("Split summary:")
    log.info(
        f"  Train : {len(X_train):>4} rows | "
        f"{ts_train.iloc[0].date()} → {ts_train.iloc[-1].date()}"
    )
    log.info(
        f"  Val   : {len(X_val):>4} rows | "
        f"{ts_val.iloc[0].date()} → {ts_val.iloc[-1].date()}"
    )
    log.info(
        f"  Test  : {len(X_test):>4} rows | "
        f"{ts_test.iloc[0].date()} → {ts_test.iloc[-1].date()}"
    )
    log.info(f"  Total : {total_rows} rows — no rows lost confirmed.")
    log.info("No temporal overlap detected between any splits.")
    log.info("======== train/val/test sets ready ========")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        ts_train, ts_val, ts_test
    )

#=====================================================================
# ------------------------- Feature Scaling --------------------------
#=====================================================================

SCALER_PATH = "scaler.pkl"

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple:
    """
    Fits a StandardScaler on the training set only, then transforms
    all three splits using the training statistics.
    """

    log.info("Initating Feature Scaling ...")

    # initalizing and fitting scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train)

    log.info(
        "StandardScaler fitted on training set only - %d rows, %d features.",
        X_train.shape[0],
        X_train.shape[1]
    )

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    log.info("All three splits transformed using training statistics.")

    # - pipeline guard: Scaled array shapes must match orignal
    for split_name, original, scaled in [
        ("train", X_train, X_train_scaled),
        ("val",   X_val,   X_val_scaled),
        ("test",  X_test,  X_test_scaled)
    ]:
        if original.shape != scaled.shape:
            log.error(
                "Shape mismatch after scaling in %s set. "
                "Original: %s | Scaled: %s. Exiting.",
                split_name,
                original.shape,
                scaled.shape
            )
            sys.exit(1)

    log.info("Shape integrity confirmed - all scaled arrays match originals.")

    # pipeline guard: No NaN values introduced by scaling
    for split_name, scaled in [
        ("train", X_train_scaled),
        ("val",   X_val_scaled),
        ("test",  X_test_scaled)
    ]:
        nan_count = np.isnan(scaled).sum()
        if nan_count > 0:
            log.error(
                "NaN values detected in scaled %s set - %d total. "
                "A feature column likely has zero variance. Exiting.",
                split_name,
                nan_count
            )
            sys.exit(1)

    log.info("NaN check passed - no NaN values in any scaled array.")

    # log leraned statistics
    feature_names = list(X_train.columns)
    check_features = ["aqi", "pressure", "is_weekend"]

    log.info("Scaler sanity check — learned statistics for key features:")

    for feature in check_features:
        if feature in feature_names:
            idx = feature_names.index(feature)
            mean = scaler.mean_[idx]
            std = scaler.scale_[idx]
            log.info(
                "  %-20s  mean = %6.2f  |  std = %6.2f",
                feature,
                mean,
                std
            )
    
    # Saving scalar locally:
    os.makedirs("artifacts", exist_ok=True)
    
    scaler_path = "artifacts/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    log.info("Scaler saved locally to '%s'.", scaler_path)

    log.info(
        "Ridge and MLP will use scaled arrays. "
    )
    log.info("======== scaled arrays ready ========")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

#=====================================================================
# -------------- Model One: Ridge Regression Training ----------------
#=====================================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def train_ridge(
    X_train_scaled: np.ndarray,
    y_train: pd.DataFrame,
    X_val_scaled: np.ndarray,
    y_val: pd.DataFrame
) -> tuple:
    """
    Tunes Ridge Regression alpha hyperparameter on the validation set,
    then trains the final model on the training set using the best alpha.
    """

    log.info("Initiating Ridge Regression Training ...")
    log.info("Candidate alphas: %s", RIDGE_ALPHAS)

    # Hyper-parameter tunning:
    best_alpha   = None
    best_val_mae = float("inf")  
    tuning_results = []

    for alpha in RIDGE_ALPHAS:
        # training on scaled training data
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        # evaluating on scaled validation data
        val_preds = model.predict(X_val_scaled)
        val_mae = mean_absolute_error(y_val, val_preds)

        tuning_results.append((alpha, val_mae))

        log.debug(
            "  alpha = %8.2f  |  val MAE = %.4f",
            alpha,
            val_mae
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha   = alpha

    log.info("Best alpha selected: %s — val MAE = %.4f", best_alpha, best_val_mae)

    # Training Final model with best Alpha:
    log.info("Training final Ridge model with best alpha = %s ...", best_alpha)

    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train_scaled, y_train)

    train_preds = ridge_model.predict(X_train_scaled)
    train_mae = mean_absolute_error(y_train, train_preds)

    log.info("Final Ridge model performance:")
    log.info("  Train MAE : %.4f", train_mae)
    log.info("  Val MAE   : %.4f", best_val_mae)
    log.info(
        "  Gap       : %.4f  (%s)",
        best_val_mae - train_mae,
        "acceptable" if (best_val_mae - train_mae) < 15 else "investigate — possible overfit"
    )

    # - pipeline guard:
    val_pred_std = np.std(val_preds)
    if val_pred_std < 1.0:
        log.warning(
            "Ridge validation predictions have very low std = %.4f. "
            "Model may be predicting near-constant values. "
            "Consider reducing alpha.",
            val_pred_std
        )
    else:
        log.info(
            "Prediction variance confirmed — val pred std = %.4f.",
            val_pred_std
        )

    log.info("======== Ridge model ready ========")

    return ridge_model, best_alpha, best_val_mae

#=====================================================================
# ---------------- Model Two: XGBoost Training -----------------
#=====================================================================

XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1]
}
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame
) -> tuple:
    """
    Tunes XGBoost hyperparameters on the validation set using grid search, 
    then trains the final model.
    """

    log.info("Initiating XGBoost training ...")

    total_combinations = len(XGB_PARAM_GRID["n_estimators"]) * len(XGB_PARAM_GRID["max_depth"]) * len(XGB_PARAM_GRID["learning_rate"])
    
    log.info("Grid search over %d combinations.", total_combinations)
    best_val_mae = float("inf")
    best_params = None
    best_model = None

    for n_est in XGB_PARAM_GRID["n_estimators"]:
        for depth in XGB_PARAM_GRID["max_depth"]:
            for lr in XGB_PARAM_GRID["learning_rate"]:
                
                base_xgb = xgb.XGBRegressor(
                    n_estimators = n_est,
                    max_depth = depth,
                    learning_rate = lr,
                    random_state = 42,
                    n_jobs = -1
                )
                model = MultiOutputRegressor(base_xgb)
                model.fit(X_train, y_train)
                val_preds = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_preds)
                log.debug(
                    "  n_est = %3d | depth = %d | lr = %.2f | val MAE = %.4f",
                    n_est, depth, lr, val_mae
                )
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_params = {
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "learning_rate": lr
                    }
                    best_model = model

    log.info("Best params selected: %s - val MAE = %.4f", best_params, best_val_mae)
    log.info("Training final XGBoost model...")
    final_base_xgb = xgb.XGBRegressor(
        n_estimators = best_params["n_estimators"],
        max_depth = best_params["max_depth"],
        learning_rate = best_params["learning_rate"],
        random_state = 42,
        n_jobs = -1
    )

    xgb_model = MultiOutputRegressor(final_base_xgb)
    xgb_model.fit(X_train, y_train)
    train_preds = xgb_model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_preds)

    log.info("Final XGBoost performance:")
    log.info("  Train MAE : %.4f", train_mae)
    log.info("  Val MAE   : %.4f", best_val_mae)
    log.info(
        "  Gap       : %.4f  (%s)",
        best_val_mae - train_mae,
        "acceptable" if (best_val_mae - train_mae) < 25 else "investigate — possible overfit"
    )
    log.info("======== XGBoost model ready ========")

    return xgb_model, best_params, best_val_mae

#=====================================================================
# ---------------- Model Three: Shallow Neural Network (MLP) ---------
#=====================================================================

MLP_CONFIGS = [
    {"hidden_1": 128, "hidden_2": 64,  "dropout": 0.2, "l2_reg": 0.001},
    {"hidden_1": 64,  "hidden_2": 32,  "dropout": 0.2, "l2_reg": 0.001},
    {"hidden_1": 256, "hidden_2": 128, "dropout": 0.3, "l2_reg": 0.0005}
]

def train_mlp(
    X_train_scaled: np.ndarray,
    y_train: pd.DataFrame,
    X_val_scaled: np.ndarray,
    y_val: pd.DataFrame
) -> tuple:
    """
    Evaluates two distinct Shallow Neural Network architectures on the 
    validation set using Early Stopping, then trains the final model 
    on the training set using the best configuration.
    """

    log.info("Initiating Shallow Neural Network (MLP) training ...")
    log.info("Candidate architectures: %d combinations.", len(MLP_CONFIGS))

    best_val_mae = float("inf")
    best_config  = None

    for config in MLP_CONFIGS:
        # Seeding for reproducibility
        tf.random.set_seed(42)

        # Defining Architecture:
        model = Sequential([
            Dense(
                config["hidden_1"], 
                activation = 'relu', 
                kernel_regularizer = l2(config["l2_reg"]), 
            ),
            Dropout(config["dropout"]),

            Dense(
                config["hidden_2"], 
                activation = 'relu', 
                kernel_regularizer = l2(config["l2_reg"])
            ),
            Dropout(config["dropout"]),

            Dense(3)
        ])

        model.compile(
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics = ['mean_absolute_error']
        )

        early_stopping = EarlyStopping(
            monitor = 'val_loss',
            patience = 30,
            restore_best_weights = True
        )

        # Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_data = (X_val_scaled, y_val),
            epochs = 500,
            batch_size = 64,
            callbacks = [early_stopping],
            verbose = 0
        )

        stopped_epoch = len(history.history['loss'])

        # evaluating:
        val_preds = model.predict(X_val_scaled, verbose=0)
        val_mae = mean_absolute_error(y_val, val_preds)

        log.info(
            "  hidden_1 = %2d | hidden_2 = %2d | stopped_epoch = %3d | val MAE = %.4f",
            config["hidden_1"],
            config["hidden_2"],
            stopped_epoch,
            val_mae
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_config = config

    log.info(
        "Best architecture selected: [%d, %d] - val MAE = %.4f",
        best_config["hidden_1"],
        best_config["hidden_2"],
        best_val_mae
    )

    # Training Final model with best architecture
    log.info(
        "Training final MLP - hidden_1 = %d | hidden_2 = %d ...",
        best_config["hidden_1"],
        best_config["hidden_2"]
    )

    # Reseting seed for final deterministic run
    tf.random.set_seed(42)

    mlp_model = Sequential([
        Dense(
            best_config["hidden_1"], 
            activation = 'relu', 
            kernel_regularizer = l2(best_config["l2_reg"]), 
        ),
        Dropout(best_config["dropout"]),
        
        Dense(
            best_config["hidden_2"], 
            activation = 'relu', 
            kernel_regularizer = l2(best_config["l2_reg"])
        ),
        Dropout(best_config["dropout"]),
        
        Dense(3)
    ])

    mlp_model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error'
    )

    final_early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 30,
        restore_best_weights = True
    )

    mlp_model.fit(
        X_train_scaled, y_train,
        validation_data = (X_val_scaled, y_val),
        epochs = 500,
        batch_size = 64,
        callbacks = [final_early_stopping],
        verbose = 0
    )

    # Training set performance
    train_preds = mlp_model.predict(X_train_scaled, verbose=0)
    train_mae = mean_absolute_error(y_train, train_preds)

    # Validation set performance
    final_val_preds = mlp_model.predict(X_val_scaled, verbose=0)
    final_val_mae = mean_absolute_error(y_val, final_val_preds)

    log.info("Final MLP performance:")
    log.info("  Train MAE : %.4f", train_mae)
    log.info("  Val MAE   : %.4f", final_val_mae)
    log.info(
        "  Gap       : %.4f  (%s)",
        final_val_mae - train_mae,
        "acceptable" if (final_val_mae - train_mae) < 15 else "investigate — possible overfit"
    )

    # - pipeline guard: 
    val_pred_std = np.std(final_val_preds)
    if val_pred_std < 1.0:
        log.warning(
            "MLP validation predictions have very low std = %.4f. "
            "Model may be predicting near-constant values. "
            "Consider reducing L2 penalty or checking scale.",
            val_pred_std
        )
    else:
        log.info(
            "Prediction variance confirmed — val pred std = %.4f.",
            val_pred_std
        )

    log.info("======== MLP model ready ========")

    return mlp_model, best_config, final_val_mae

#=====================================================================
# ----------------------- Models Evaluation --------------------------
#=====================================================================

def compute_metrics(y_true, y_pred, split_name: str) -> dict:
    """
    Computes RMSE, MAE, and R² for a given set of predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    log.debug("  %s — RMSE: %.4f | MAE: %.4f | R²: %.4f", split_name, rmse, mae, r2)

    return {"rmse": rmse, "mae": mae, "r2": r2}

def compute_persistence_baseline(
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> dict:
    """
    Persistence baseline for multi-output daily max prediction.
    Strategy: predict that each future day's max AQI equals the
    mean daily max AQI observed in the training set.
    This is the simplest defensible constant baseline for a
    calendar-day max target with no same-day context available.
    """
    baseline_preds = np.tile(
        y_train.mean().values,
        (len(y_test), 1)
    )
    return baseline_preds

def evaluate_all_models(
    ridge_model, xgb_model, mlp_model,
    X_train, X_val, X_test,
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test
) -> dict:
    """
    Evaluates all three models plus a persistence baseline on the test set.
    Prints a single, clean summary table for GitHub Actions logs.
    """

    # Generating predictions 
    ridge_preds = ridge_model.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test)
    mlp_preds = mlp_model.predict(X_test_scaled, verbose=0)
    baseline_preds = compute_persistence_baseline(y_train, y_test)
    day_names = ["y_day1", "y_day2", "y_day3"]
    all_metrics = {}
    
    # Compute all metrics silently
    for model_name, preds in [
        ("ridge", ridge_preds),
        ("xgb", xgb_preds),
        ("mlp", mlp_preds),
        ("persistence", baseline_preds)
    ]:
        model_metrics = {}
        
        for day_idx, day_name in enumerate(day_names):
            actual = y_test.iloc[:, day_idx].values
            predicted = preds[:, day_idx]
            model_metrics[day_name] = compute_metrics(actual, predicted, "")
        all_metrics[model_name] = model_metrics
    
    # Print Clean Summary Table
    log.info("=" * 60)
    log.info("FINAL EVALUATION METRICS (Test Set N=%d)", len(y_test))
    log.info("=" * 60)
    
    for day_name in day_names:
        baseline_mae = all_metrics["persistence"][day_name]["mae"]
        log.info("--- %s (Persistence Baseline MAE: %.2f) ---", day_name.upper(), baseline_mae)
        
        for model_name in ["ridge", "xgb", "mlp"]:
            m = all_metrics[model_name][day_name]
            delta = m["mae"] - baseline_mae
            
            log.info(
                "  %-8s | RMSE: %5.2f | MAE: %5.2f (%+5.2f) | R²: %5.3f",
                model_name.upper(), m["rmse"], m["mae"], delta, m["r2"]
            )
        log.info("")  # Empty line for spacing

    log.info("============================================================")
    
    return all_metrics

#=====================================================================
# -------------------- DagsHub - Saving Artifacts --------------------
#=====================================================================

def save_artifacts(
    ridge_model, xgb_model, mlp_model,
    scaler,
    ridge_best_alpha, xgb_best_params, mlp_best_config,
    all_metrics: dict,
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
):
    """
    Initialises a DagsHub MLflow run and logs:
    - Hyperparameters for all three models
    - All evaluation metrics (Day 1, Day 2, Day 3)
    - Model artifacts (ridge, xgb, mlp, scaler, feature_columns)
    """

    log.info("Saving artifacts to DagsHub via MLflow...")
    
    DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME")
    DAGSHUB_REPO     = os.environ.get("DAGSHUB_REPO")
    DAGSHUB_TOKEN    = os.environ.get("DAGSHUB_TOKEN")
    
    if not all([DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN]):
        log.error(
            "DagsHub credentials missing from environment. "
            "Ensure DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN are set. Exiting."
        )
        sys.exit(1)

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    dagshub.init(
        repo_owner = DAGSHUB_USERNAME,
        repo_name  = DAGSHUB_REPO,
        mlflow     = True
    )
    log.info("DagsHub connection initialised. Starting MLflow run...")

    with mlflow.start_run(run_name="pipeline_training") as run:

        # LOG HYPERPARAMETERS
        mlflow.log_param("ridge_best_alpha",      ridge_best_alpha)
        mlflow.log_param("xgb_n_estimators",      xgb_best_params["n_estimators"])
        mlflow.log_param("xgb_max_depth",         xgb_best_params["max_depth"])
        mlflow.log_param("xgb_learning_rate",     xgb_best_params["learning_rate"])
        mlflow.log_param("mlp_hidden_1",          mlp_best_config["hidden_1"])
        mlflow.log_param("mlp_hidden_2",          mlp_best_config["hidden_2"])
        mlflow.log_param("mlp_dropout",           mlp_best_config["dropout"])
        mlflow.log_param("mlp_l2_reg",            mlp_best_config["l2_reg"])
        mlflow.log_param("train_size",            len(X_train))
        mlflow.log_param("val_size",              len(X_val))
        mlflow.log_param("test_size",             len(X_test))
        mlflow.log_param("feature_count",         X_train.shape[1])
        log.info("Hyperparameters logged.")

        # LOG MULTI-OUTPUT METRICS (Day 1, Day 2, Day 3)
        day_names = ["y_day1", "y_day2", "y_day3"]
        model_names = ["ridge", "xgb", "mlp", "persistence"]
        for model_name in model_names:
            for day_name in day_names:
                metrics = all_metrics[model_name][day_name]
                mlflow.log_metric(f"{model_name}_{day_name}_rmse", metrics["rmse"])
                mlflow.log_metric(f"{model_name}_{day_name}_mae",  metrics["mae"])
                mlflow.log_metric(f"{model_name}_{day_name}_r2",   metrics["r2"])
        log.info("All metrics logged to MLflow.")
        
        ARTIFACTS_DIR = "artifacts"
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # SSaving and logging the artifacts
        # Ridge
        ridge_path = os.path.join(ARTIFACTS_DIR, "ridge_model.pkl")
        joblib.dump(ridge_model, ridge_path)
        mlflow.log_artifact(ridge_path)
        log.info("Ridge model saved and logged.")
        
        # XGBoost
        xgb_path = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
        joblib.dump(xgb_model, xgb_path)
        mlflow.log_artifact(xgb_path)
        log.info("XGBoost model saved and logged.")
        
        # MLP
        mlp_path = os.path.join(ARTIFACTS_DIR, "mlp_model.keras")
        mlp_model.save(mlp_path)
        mlflow.log_artifact(mlp_path)
        log.info("MLP model saved and logged.")
        
        # Scaler
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        log.info("Scaler logged.")
        
        # Feature columns
        feature_columns_path = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
        mlflow.log_artifact(feature_columns_path)
        log.info("Feature columns logged.")
        
        # Metrics JSON
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json_module.dump(all_metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)
        log.info("Metrics JSON saved and logged.")
        
        run_id = run.info.run_id
        log.info("MLflow run completed. Run ID: %s", run_id)
    
    log.info("======== all artifacts saved to DagsHub ========")


#=====================================================================
# -------------------------- Main Function --------------------------
#=====================================================================

if __name__ == "__main__":
    df = fetch_features()
    log.info("") 

    X, y, timestamps = construct_features(df)      
    log.info("") 

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        ts_train, ts_val, ts_test
    ) = split_data(X, y, timestamps)
    log.info("") 

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(X_train, X_val, X_test)
    log.info("") 

    ridge_model, ridge_best_alpha, ridge_val_mae = train_ridge(
        X_train_scaled, y_train,
        X_val_scaled,   y_val
    )
    log.info("") 

    xgb_model, xgb_best_params, xgb_val_mae = train_xgboost(
        X_train, y_train,
        X_val,   y_val
    )
    log.info("") 

    mlp_model, mlp_best_config, mlp_val_mae = train_mlp(
        X_train_scaled, y_train,
        X_val_scaled,   y_val
    )
    log.info("") 

    log.info("======== All Three Models Trained Successfully ========")
    log.info("") 

    all_metrics = evaluate_all_models(
        ridge_model, xgb_model, mlp_model,
        X_train, X_val, X_test,
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test                    
    )
    log.info("") 

    save_artifacts(
        ridge_model, xgb_model, mlp_model, 
        scaler,
        ridge_best_alpha, xgb_best_params, mlp_best_config, 
        all_metrics,
        X_train, X_val, X_test  
    )
    log.info("") 
    log.info("======== Pipeline complete. Exiting cleanly ========")
    sys.exit(0)