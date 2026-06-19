"""
train_xgboost.py
=================
Model 2A — LANL Earthquake Prediction (Acoustic Signals)

Predicts time_to_failure from 7 hand-engineered statistical features
(mean, std, max, min, median, 1st/99th percentile) extracted from
150,000-sample acoustic windows, using XGBoost.

Run:
    python src/train_xgboost.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    LANL_FEATURES_PATH,
    LANL_TEST_SIZE,
    LANL_TRAIN_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SEGMENT_SIZE,
    XGB_PARAMS,
)
from preprocessing import build_segment_feature_table


def load_or_build_features() -> pd.DataFrame:
    """
    Build the segmented statistical-feature table from the raw LANL
    signal (chunked read, since train.csv is multi-GB), caching the
    result so repeated runs/notebook cells don't re-process raw data.
    """
    if os.path.exists(LANL_FEATURES_PATH):
        print(f"      -> Found cached features at {LANL_FEATURES_PATH}")
        return pd.read_csv(LANL_FEATURES_PATH)

    print("      -> No cache found, processing raw signal in chunks...")
    dtypes = {"acoustic_data": np.int16, "time_to_failure": np.float64}

    reader = pd.read_csv(
        LANL_TRAIN_PATH,
        dtype=dtypes,
        chunksize=SEGMENT_SIZE,
    )

    feature_rows = []
    leftover_acoustic = np.array([], dtype=np.int16)
    leftover_ttf = np.array([], dtype=np.float64)

    for chunk in reader:
        acoustic = np.concatenate([leftover_acoustic, chunk["acoustic_data"].values])
        ttf = np.concatenate([leftover_ttf, chunk["time_to_failure"].values])

        n_full = len(acoustic) // SEGMENT_SIZE
        usable = n_full * SEGMENT_SIZE

        if n_full > 0:
            segment_table = build_segment_feature_table(
                acoustic[:usable], ttf[:usable], segment_size=SEGMENT_SIZE
            )
            feature_rows.append(segment_table)

        leftover_acoustic = acoustic[usable:]
        leftover_ttf = ttf[usable:]

    features_df = pd.concat(feature_rows, ignore_index=True)
    os.makedirs(os.path.dirname(LANL_FEATURES_PATH), exist_ok=True)
    features_df.to_csv(LANL_FEATURES_PATH, index=False)
    print(f"      -> Cached {len(features_df):,} segments to {LANL_FEATURES_PATH}")

    return features_df


def main():
    # ----------------------------------------------------------------
    # 1. Build / load segmented statistical features
    # ----------------------------------------------------------------
    print("[1/5] Preparing segment-level statistical features...")
    df = load_or_build_features()

    feature_cols = ["mean", "std", "max", "min", "median", "p01", "p99"]
    X = df[feature_cols]
    y = df["time_to_failure"]

    # ----------------------------------------------------------------
    # 2. Train/test split
    # ----------------------------------------------------------------
    print("[2/5] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=LANL_TEST_SIZE, random_state=RANDOM_STATE
    )

    # ----------------------------------------------------------------
    # 3. Train XGBoost regressor
    # ----------------------------------------------------------------
    print(f"[3/5] Training XGBoost with params: {XGB_PARAMS}")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ----------------------------------------------------------------
    # 4. Evaluate
    # ----------------------------------------------------------------
    print("[4/5] Evaluating on test set...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"      -> MAE  : {mae:.3f} seconds")
    print(f"      -> RMSE : {rmse:.3f} seconds")
    print(f"      -> R2   : {r2:.4f}")

    # ----------------------------------------------------------------
    # 5. Save model + feature importance
    # ----------------------------------------------------------------
    print("[5/5] Saving model artifact...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_lanl.joblib")
    joblib.dump(model, model_path)
    print(f"      -> Saved to {model_path}")

    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importance)

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    main()
