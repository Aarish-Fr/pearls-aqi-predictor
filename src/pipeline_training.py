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

from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


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

def construct_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Constructs the feature matrix X and target vector y from the raw DataFrame.

    y: next hour's AQI — created by shifting the aqi column back by one row.
    X: all columns except the four dropped columns defined in COLUMNS_TO_DROP.
    """

    log.info("Constructing X and y ...")

    # for y, shift aqi column back by one position
    y = df["aqi"].shift(-1)
    X = df.drop(columns = COLUMNS_TO_DROP)

    # dropping the last row:
    X = X.iloc[:-1]
    y = y.iloc[:-1]

    # - pipeline guard: Shape verification
    expected_feature_count = 19
    if X.shape[1] != expected_feature_count:
        log.error("Feature count missmatched. Expected %d, got %d features. Check COLUMNS_TO_DROP. Exiting", expected_feature_count, X.shape[1])
        sys.exit(1)

    if X.shape[0] != y.shape[0]:
        log.error("X and y row count mismatched. X has %d rows, y has %d rows. Alignment is broken. Exiting", X.shape[0], y.shape[0])
        sys.exit(1)

    log.info(f"X shape: {X.shape}  |  y shape: {y.shape}")

    # - pipeline guard: NaN check on y:
    nan_count = y.isnull().sum()
    if nan_count > 0:
        log.error("y contains %d NaN values after shift and drop. This should never happen. Exiting.", nan_count)
        sys.exit(1)

    log.info("NaN check passed: y contains zero NaN values.")

    # - pipeline guard: NaN check on X
    nan_count_X = X.isnull().sum().sum()
    if nan_count_X > 0:
        log.error("X contains %d NaN values. Check upstream feature pipeline. Exiting.", nan_count_X)
        sys.exit(1)

    log.info("NaN check passed - X contains zero NaN values.")

    # Sanity Check:
    log.info("Shift sanity check:")
    log.info(f"  df['aqi'].iloc[0] = {df['aqi'].iloc[0]}  →  y.iloc[0] = {y.iloc[0]}")
    log.info(f"  df['aqi'].iloc[1] = {df['aqi'].iloc[1]}  →  y.iloc[1] = {y.iloc[1]}")
    log.info(f"  df['aqi'].iloc[2] = {df['aqi'].iloc[2]}  →  y.iloc[2] = {y.iloc[2]}")

    # saving feature column list:
    feature_columns = list(X.columns)
    with open("feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    log.info("Feature columns saved to feature_columns.json")
    log.info("Features: %s", feature_columns)
    log.info("====== X and y ready ======")

    return X, y

#=====================================================================
# ------------------- Train / Val / Test Split -----------------------
#=====================================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def split_data(
        X: pd.DataFrame,
        y: pd.Series,
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
    joblib.dump(scaler, SCALER_PATH)
    log.info("Scaler saved locally to '%s'.", SCALER_PATH)
    log.info(
        "Ridge and MLP will use scaled arrays. "
        "Random Forest will use original unscaled arrays."
    )
    log.info("======== scaled arrays ready ========")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

#=====================================================================
# -------------- Model One: Ridge Regression Training ----------------
#=====================================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def train_ridge(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_val_scaled: np.ndarray,
    y_val: pd.Series
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

        log.info(
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
        "acceptable" if (best_val_mae - train_mae) < 10 else "investigate — possible overfit"
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
# ---------------- Model Two: Random Forest Training -----------------
#=====================================================================

RF_PARAM_GRID = {
    "n_estimators": [50, 100],           
    "max_depth":    [4, 5, 6],           
    "min_samples_leaf": [3, 5, 8],       
    "max_features": [1.0, "sqrt"]        
}

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> tuple:
    """
    Tunes Random Forest hyperparameters on the
    validation set using a grid search, then trains the final model on
    the training set.
    """

    log.info("Initiating Random Forest training ...")
    total_combinations = (
        len(RF_PARAM_GRID["n_estimators"]) * len(RF_PARAM_GRID["max_depth"]) * len(RF_PARAM_GRID["min_samples_leaf"]) *
        len(RF_PARAM_GRID["max_features"])
    )
    log.info("Grid search over %d combinations.", total_combinations)

    # Grid search
    best_val_mae = float("inf")
    best_params  = None
    best_model   = None

    for n_est in RF_PARAM_GRID["n_estimators"]:
        for depth in RF_PARAM_GRID["max_depth"]:
            for min_leaf in RF_PARAM_GRID["min_samples_leaf"]:
                for max_feat in RF_PARAM_GRID["max_features"]:
                    
                    model = RandomForestRegressor(
                        n_estimators     = n_est,
                        max_depth        = depth,
                        min_samples_leaf = min_leaf,
                        max_features     = max_feat,
                        random_state     = 42,    
                        n_jobs           = -1
                    ) 
                    model.fit(X_train, y_train)

                    val_preds = model.predict(X_val)
                    val_mae   = mean_absolute_error(y_val, val_preds)
                    
                    depth_str = str(depth) if depth is not None else "None"
                    feat_str  = str(max_feat)

                    log.info(
                        "  n_est = %3d | max_depth = %4s | min_leaf = %2d | max_feat = %4s | val MAE = %.4f",
                        n_est,
                        depth_str,
                        min_leaf,
                        feat_str,
                        val_mae
                    )

                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        best_params  = {
                            "n_estimators": n_est,
                            "max_depth":    depth,
                            "min_samples_leaf": min_leaf,
                            "max_features": max_feat
                        }
                        best_model = model

    log.info(
        "Best params selected: n_estimators = %d | max_depth = %s | min_leaf = %d | max_feat = %s - val MAE = %.4f",
        best_params["n_estimators"],
        str(best_params["max_depth"]),
        best_params["min_samples_leaf"],
        str(best_params["max_features"]),
        best_val_mae
    )

    # Training Final model with best params
    log.info(
        "Training final Random Forest - n_estimators = %d | max_depth = %s | min_leaf = %d | max_feat = %s ...",
        best_params["n_estimators"],
        str(best_params["max_depth"]),
        best_params["min_samples_leaf"],
        str(best_params["max_features"])
    )

    rf_model = RandomForestRegressor(
        n_estimators     = best_params["n_estimators"],
        max_depth        = best_params["max_depth"],
        min_samples_leaf = best_params["min_samples_leaf"],
        max_features     = best_params["max_features"],
        random_state     = 42,
        n_jobs           = -1
    )
    rf_model.fit(X_train, y_train)

    # training set performance:
    train_preds = rf_model.predict(X_train)
    train_mae   = mean_absolute_error(y_train, train_preds)

    log.info("Final Random Forest performance:")
    log.info("  Train MAE : %.4f", train_mae)
    log.info("  Val MAE   : %.4f", best_val_mae)
    log.info(
        "  Gap       : %.4f  (%s)",
        best_val_mae - train_mae,
        "acceptable" if (best_val_mae - train_mae) < 10 else "investigate — possible overfit"
    )

    # Feature Importance:
    feature_names = list(X_train.columns)
    importances = rf_model.feature_importances_
    importance_pairs = sorted(
        zip(feature_names, importances),
        key = lambda x: x[1],
        reverse = True
    )

    log.info("Top 5 feature importances:")
    for rank, (feature, score) in enumerate(importance_pairs[:5], start=1):
        log.info("  %d. %-22s  %.4f", rank, feature, score)

    # - pipeline guard: 
    val_pred_std = np.std(rf_model.predict(X_val))
    if val_pred_std < 1.0:
        log.warning(
            "Random Forest validation predictions have very low std = %.4f. "
            "Model may be predicting near-constant values. "
            "Consider reducing max_depth or n_estimators.",
            val_pred_std
        )
    else:
        log.info(
            "Prediction variance confirmed — val pred std = %.4f.",
            val_pred_std
        )

    log.info("======== Random Forest model ready ========")

    return rf_model, best_params, best_val_mae

#=====================================================================
# -------------------------- Main Function --------------------------
#=====================================================================

if __name__ == "__main__":
    df = fetch_features()
    X, y = construct_features(df)

    timestamps = df["timestamp"].iloc[:-1].reset_index(drop=True)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        ts_train, ts_val, ts_test
    ) = split_data(X, y, timestamps)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )  = scale_features(X_train, X_val, X_test)

    ridge_model, ridge_best_alpha, ridge_val_mae = train_ridge(
        X_train_scaled, y_train,
        X_val_scaled,   y_val
    )

    rf_model, rf_best_params, rf_val_mae = train_random_forest(
        X_train, y_train,
        X_val,   y_val
    )