"""
train_random_forest.py
=======================
Model 1 — USGS Significant Earthquakes (1965-2016)

Predicts Magnitude and Depth from Timestamp, Latitude, and Longitude
using a RandomForestRegressor, tuned via GridSearchCV over n_estimators.

Run:
    python src/train_random_forest.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS_DIR,
    RANDOM_STATE,
    RF_PARAM_GRID,
    USGS_FEATURE_COLUMNS,
    USGS_PROCESSED_PATH,
    USGS_RAW_PATH,
    USGS_TARGET_COLUMNS,
    USGS_TEST_SIZE,
)
from preprocessing import load_and_clean_usgs


def main():
    # ----------------------------------------------------------------
    # 1. Load & preprocess
    # ----------------------------------------------------------------
    print("[1/5] Loading and cleaning USGS dataset...")
    df = load_and_clean_usgs(USGS_RAW_PATH)
    os.makedirs(os.path.dirname(USGS_PROCESSED_PATH), exist_ok=True)
    df.to_csv(USGS_PROCESSED_PATH, index=False)
    print(f"      -> {len(df):,} clean rows | saved to {USGS_PROCESSED_PATH}")

    X = df[USGS_FEATURE_COLUMNS]
    y = df[USGS_TARGET_COLUMNS]

    # ----------------------------------------------------------------
    # 2. Train/test split (80/20)
    # ----------------------------------------------------------------
    print("[2/5] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=USGS_TEST_SIZE, random_state=RANDOM_STATE
    )

    # ----------------------------------------------------------------
    # 3. Hyperparameter tuning via GridSearchCV
    # ----------------------------------------------------------------
    print(f"[3/5] Running GridSearchCV over n_estimators={RF_PARAM_GRID['n_estimators']}...")
    base_model = RandomForestRegressor(random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=RF_PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"      -> Best n_estimators: {grid_search.best_params_['n_estimators']}")
    print(f"      -> Best CV R2 score:  {grid_search.best_score_:.4f}")

    # ----------------------------------------------------------------
    # 4. Evaluate on held-out test set
    # ----------------------------------------------------------------
    print("[4/5] Evaluating on test set...")
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"      -> R2 Score : {r2 * 100:.2f}%")
    print(f"      -> MAE      : {mae:.4f}")
    print(f"      -> RMSE     : {rmse:.4f}")

    # ----------------------------------------------------------------
    # 5. Persist the trained model
    # ----------------------------------------------------------------
    print("[5/5] Saving model artifact...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "random_forest_usgs.joblib")
    joblib.dump(best_model, model_path)
    print(f"      -> Saved to {model_path}")

    return best_model, {"r2": r2, "mae": mae, "rmse": rmse}


if __name__ == "__main__":
    main()
