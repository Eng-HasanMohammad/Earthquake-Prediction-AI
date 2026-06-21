"""Centralized configuration for the Earthquake Prediction AI System.

This module is the single source of truth for filesystem paths, random
seeds, model hyperparameters, and visualization settings used across the
three modeling pipelines:

    1. Random Forest  -- USGS Significant Earthquakes (1965-2016), regional
       magnitude/depth regression from geospatial-temporal features.
    2. XGBoost         -- LANL Earthquake Prediction, time-to-failure
       regression from hand-engineered statistical signal descriptors.
    3. 1D-CNN           -- LANL Earthquake Prediction, time-to-failure
       regression learned end-to-end from the raw acoustic waveform.

All paths are resolved *relative to this file's location* rather than the
current working directory, so the project runs identically whether it is
launched from the repo root, from ``src/``, or from ``notebooks/`` --
no manual path editing required after cloning.

Typical usage::

    from config import USGS_RAW_PATH, RANDOM_STATE, RF_PARAM_GRID
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Path configuration (relative, anchored to this file -- never to cwd)
# ---------------------------------------------------------------------------
# config.py lives at <repo_root>/src/config.py, so the repo root is one
# level up from this file's directory. This makes every derived path
# portable across machines and execution contexts (script, notebook, CI).
SRC_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: Final[str] = os.path.dirname(SRC_DIR)

DATA_RAW_DIR: Final[str] = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR: Final[str] = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR: Final[str] = os.path.join(ROOT_DIR, "models")
FIGURES_DIR: Final[str] = os.path.join(ROOT_DIR, "reports", "figures")
LOGS_DIR: Final[str] = os.path.join(ROOT_DIR, "logs")

# Dataset 1: USGS Significant Earthquakes (1965-2016)
# Source: Kaggle "Significant Earthquakes, 1965-2016" (database.csv)
USGS_RAW_PATH: Final[str] = os.path.join(DATA_RAW_DIR, "database.csv")
USGS_PROCESSED_PATH: Final[str] = os.path.join(
    DATA_PROCESSED_DIR, "usgs_processed.csv"
)

# Dataset 2: LANL Earthquake Prediction (acoustic signals)
# Source: Kaggle "LANL Earthquake Prediction" competition (train.csv)
LANL_TRAIN_PATH: Final[str] = os.path.join(DATA_RAW_DIR, "train.csv")
LANL_FEATURES_PATH: Final[str] = os.path.join(
    DATA_PROCESSED_DIR, "lanl_segment_features.csv"
)

# Ensure output directories exist as soon as the project is imported, so
# downstream scripts never fail on a missing folder mid-run.
for _dir in (DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: Final[int] = 42

# ---------------------------------------------------------------------------
# Dataset 1 -- Random Forest (USGS) configuration
# ---------------------------------------------------------------------------
USGS_FEATURE_COLUMNS: Final[list[str]] = ["Timestamp", "Latitude", "Longitude"]
USGS_TARGET_COLUMNS: Final[list[str]] = ["Magnitude", "Depth"]
USGS_TEST_SIZE: Final[float] = 0.20

RF_PARAM_GRID: Final[dict[str, list[int]]] = {
    "n_estimators": [10, 20, 50, 100, 200, 500],
}
RF_CV_FOLDS: Final[int] = 5
RF_SCORING: Final[str] = "r2"

# ---------------------------------------------------------------------------
# Dataset 2 -- LANL signal segmentation configuration
# ---------------------------------------------------------------------------
SEGMENT_SIZE: Final[int] = 150_000  # rows per acoustic segment
LANL_TEST_SIZE: Final[float] = 0.20
LANL_STAT_FEATURE_COLUMNS: Final[list[str]] = [
    "mean", "std", "max", "min", "median", "p01", "p99",
]

# XGBoost hyperparameters (statistical-feature pipeline)
XGB_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}

# 1D-CNN hyperparameters (raw-signal pipeline)
CNN_CONFIG: Final[dict[str, object]] = {
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

# ---------------------------------------------------------------------------
# Visualization configuration (shared "publication" theme)
# ---------------------------------------------------------------------------
PLOT_DPI: Final[int] = 300
PLOT_STYLE: Final[str] = "dark_background"
PLOT_PALETTE_SEQUENTIAL: Final[str] = "viridis"
PLOT_PALETTE_DIVERGING: Final[str] = "coolwarm"

# Centralized accent colors so every figure across the three notebooks
# shares one visual identity.
PLOT_COLORS: Final[dict[str, str]] = {
    "primary": "#3FA7D6",     # cool azure
    "secondary": "#F4A261",   # warm amber
    "accent": "#E63946",      # alert red (used for reference/identity lines)
    "tertiary": "#8338EC",    # violet (CNN series)
    "grid": "#3A3F4B",
    "text": "#E8E8E8",
}


@dataclass(frozen=True)
class FigureSpec:
    """Reusable figure sizing/resolution spec.

    Attributes:
        figsize: Width/height in inches as ``(w, h)``.
        dpi: Dots-per-inch for on-screen rendering and saved output.
    """

    figsize: tuple[float, float] = (10, 6)
    dpi: int = PLOT_DPI


DEFAULT_FIGURE: Final[FigureSpec] = FigureSpec()
WIDE_FIGURE: Final[FigureSpec] = FigureSpec(figsize=(14, 5))
SQUARE_FIGURE: Final[FigureSpec] = FigureSpec(figsize=(7, 7))
