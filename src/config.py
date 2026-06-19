"""
config.py
=========
Centralized configuration for the Earthquake Prediction via AI project.

Keeping paths, random seeds, and hyperparameter grids in one place means
every notebook/script (Random Forest, XGBoost, 1D-CNN) stays consistent
and reproducible.
"""

import os

# ----------------------------------------------------------------------
# Path configuration
# ----------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")

# Dataset 1: USGS Significant Earthquakes (1965-2016)
USGS_RAW_PATH = os.path.join(DATA_RAW_DIR, "database.csv")
USGS_PROCESSED_PATH = os.path.join(DATA_PROCESSED_DIR, "usgs_processed.csv")

# Dataset 2: LANL Earthquake Prediction (acoustic signals)
LANL_TRAIN_PATH = os.path.join(DATA_RAW_DIR, "train.csv")
LANL_FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "lanl_segment_features.csv")

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Dataset 1 — Random Forest (USGS) configuration
# ----------------------------------------------------------------------
USGS_FEATURE_COLUMNS = ["Timestamp", "Latitude", "Longitude"]
USGS_TARGET_COLUMNS = ["Magnitude", "Depth"]
USGS_TEST_SIZE = 0.20

RF_PARAM_GRID = {
    "n_estimators": [10, 20, 50, 100, 200, 500]
}

# ----------------------------------------------------------------------
# Dataset 2 — LANL signal segmentation configuration
# ----------------------------------------------------------------------
SEGMENT_SIZE = 150_000          # rows per acoustic segment
LANL_TEST_SIZE = 0.20

# XGBoost hyperparameters (statistical-feature pipeline)
XGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}

# 1D-CNN hyperparameters (raw-signal pipeline)
CNN_CONFIG = {
    "conv1_filters": 16,
    "conv1_kernel": 10,
    "pool1_size": 10,
    "conv2_filters": 32,
    "conv2_kernel": 10,
    "pool2_size": 10,
    "dense_units": 64,
    "dropout_rate": 0.3,
    "epochs": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "loss": "mae",
}
