"""Model 2A -- LANL Earthquake Prediction: XGBoost (Statistical Features).

Predicts `time_to_failure` from 7 hand-engineered statistical descriptors
(`mean`, `std`, `max`, `min`, `median`, `p01`, `p99`) extracted from
150,000-sample acoustic emission windows, using `XGBRegressor`.

Seismological framing: in stick-slip laboratory friction experiments,
the acoustic emission signal's volatility and tail behavior intensify
as the simulated fault approaches failure (micro-fracturing precedes
macroscopic slip). Gradient-boosted trees over these statistical
summaries provide a lightweight, fast-to-train alternative to learning
directly from the raw waveform (see `train_cnn.py`).

Run:
    python src/train_xgboost.py
"""

from __future__ import annotations

import logging
import os
from typing import Final

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    FIGURES_DIR,
    LANL_FEATURES_PATH,
    LANL_STAT_FEATURE_COLUMNS,
    LANL_TEST_SIZE,
    LANL_TRAIN_PATH,
    MODELS_DIR,
    PLOT_COLORS,
    PLOT_DPI,
    PLOT_STYLE,
    RANDOM_STATE,
    SEGMENT_SIZE,
    XGB_PARAMS,
)
from logging_utils import console, get_logger, metrics_table, section
from preprocessing import build_segment_feature_table

logger: logging.Logger = get_logger(__name__)

_READER_DTYPES: Final[dict[str, type]] = {"acoustic_data": np.int16, "time_to_failure": np.float64}


def _apply_plot_style() -> None:
    """Apply the shared dark, high-contrast publication theme to Matplotlib."""
    plt.style.use(PLOT_STYLE)
    sns.set_style("darkgrid", {"axes.facecolor": "#1b1f27", "grid.color": PLOT_COLORS["grid"]})
    plt.rcParams.update({
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "axes.edgecolor": PLOT_COLORS["text"],
        "axes.labelcolor": PLOT_COLORS["text"],
        "text.color": PLOT_COLORS["text"],
        "xtick.color": PLOT_COLORS["text"],
        "ytick.color": PLOT_COLORS["text"],
        "font.size": 11,
    })


def load_or_build_features() -> pd.DataFrame:
    """Build (or load a cached copy of) the segment-level statistical feature table.

    The raw LANL `train.csv` is several gigabytes, so it is streamed in
    `SEGMENT_SIZE`-aligned chunks rather than loaded into memory at once.
    Leftover rows that don't complete a full segment at a chunk boundary
    are carried over and concatenated with the next chunk, ensuring
    segments are never split across chunk reads. The resulting feature
    table is cached to disk so repeated runs skip re-processing the raw
    signal.

    Returns:
        A `pandas.DataFrame` with columns
        ``["mean", "std", "max", "min", "median", "p01", "p99",
        "time_to_failure"]``.

    Raises:
        FileNotFoundError: If no cached feature file exists and the raw
            LANL `train.csv` is also missing.
    """
    if os.path.exists(LANL_FEATURES_PATH):
        logger.info("Found cached segment features at '%s'.", LANL_FEATURES_PATH)
        return pd.read_csv(LANL_FEATURES_PATH)

    logger.info("No cache found -- streaming raw signal from '%s'.", LANL_TRAIN_PATH)
    try:
        reader = pd.read_csv(LANL_TRAIN_PATH, dtype=_READER_DTYPES, chunksize=SEGMENT_SIZE)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find LANL training data at '{LANL_TRAIN_PATH}'. "
            "Place the Kaggle 'LANL Earthquake Prediction' train.csv at "
            "this location (see config.LANL_TRAIN_PATH)."
        ) from exc

    feature_rows: list[pd.DataFrame] = []
    leftover_acoustic = np.array([], dtype=np.int16)
    leftover_ttf = np.array([], dtype=np.float64)

    for chunk in reader:
        acoustic = np.concatenate([leftover_acoustic, chunk["acoustic_data"].values])
        ttf = np.concatenate([leftover_ttf, chunk["time_to_failure"].values])

        n_full = len(acoustic) // SEGMENT_SIZE
        usable = n_full * SEGMENT_SIZE

        if n_full > 0:
            feature_rows.append(
                build_segment_feature_table(acoustic[:usable], ttf[:usable], segment_size=SEGMENT_SIZE)
            )

        leftover_acoustic = acoustic[usable:]
        leftover_ttf = ttf[usable:]

    if not feature_rows:
        raise ValueError(
            "No full segments were extracted from the raw signal -- the "
            "file may be shorter than a single SEGMENT_SIZE window."
        )

    features_df = pd.concat(feature_rows, ignore_index=True)
    os.makedirs(os.path.dirname(LANL_FEATURES_PATH), exist_ok=True)
    features_df.to_csv(LANL_FEATURES_PATH, index=False)
    logger.info("Cached %d segments to '%s'.", len(features_df), LANL_FEATURES_PATH)
    return features_df


def train_xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> xgb.XGBRegressor:
    """Fit an `XGBRegressor` with evaluation tracking on both train and test sets.

    Args:
        X_train: Training feature matrix (7 statistical descriptors).
        y_train: Training `time_to_failure` targets.
        X_test: Held-out feature matrix.
        y_test: Held-out `time_to_failure` targets.

    Returns:
        The fitted `xgb.XGBRegressor`, with `.evals_result()` populated
        for both the ``validation_0`` (train) and ``validation_1``
        (test) eval sets.
    """
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    return model


def evaluate_model(
    model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[np.ndarray, dict[str, float]]:
    """Score the fitted model on a held-out test set.

    Args:
        model: A fitted regressor exposing `.predict`.
        X_test: Held-out feature matrix.
        y_test: Held-out `time_to_failure` targets.

    Returns:
        A tuple ``(y_pred, metrics)`` with `metrics` containing
        ``mae``, ``rmse``, and ``r2``.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }
    return y_pred, metrics


def plot_training_curve(model: xgb.XGBRegressor, save_path: str) -> None:
    """Plot train/test RMSE per boosting round and save to disk.

    Args:
        model: A fitted `XGBRegressor` with populated `evals_result()`.
        save_path: Destination PNG path.
    """
    evals_result = model.evals_result()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(evals_result["validation_0"]["rmse"], label="Train RMSE",
            color=PLOT_COLORS["primary"], linewidth=2)
    ax.plot(evals_result["validation_1"]["rmse"], label="Test RMSE",
            color=PLOT_COLORS["secondary"], linewidth=2)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("RMSE")
    ax.set_title("XGBoost Training Curve", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curve to '%s'.", save_path)


def plot_prediction_scatter(y_test: pd.Series, y_pred: np.ndarray, save_path: str) -> None:
    """Plot actual-vs-predicted `time_to_failure` scatter and save to disk.

    Args:
        y_test: Ground-truth `time_to_failure` values.
        y_pred: Model predictions, aligned with `y_test`.
        save_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.45, s=14, color=PLOT_COLORS["secondary"], edgecolors="none")
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, "--", linewidth=1.8, color=PLOT_COLORS["accent"], label="Perfect prediction")
    ax.set_xlabel("Actual time_to_failure (s)")
    ax.set_ylabel("Predicted time_to_failure (s)")
    ax.set_title("XGBoost: Actual vs. Predicted Time-to-Failure", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction scatter plot to '%s'.", save_path)


def plot_feature_importance(model: xgb.XGBRegressor, feature_columns: list[str], save_path: str) -> pd.Series:
    """Plot and save a horizontal feature-importance bar chart.

    Args:
        model: A fitted `XGBRegressor` exposing `feature_importances_`.
        feature_columns: Feature names, aligned with `feature_importances_`.
        save_path: Destination PNG path.

    Returns:
        A `pandas.Series` of importances sorted descending, indexed by
        feature name.
    """
    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importance.values, y=importance.index, hue=importance.index,
                palette="flare", legend=False, edgecolor="none", ax=ax)
    ax.set_title("XGBoost Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot to '%s'.", save_path)
    return importance


def main() -> tuple[xgb.XGBRegressor, dict[str, float]]:
    """Run the full XGBoost training pipeline end-to-end.

    Returns:
        A tuple ``(model, metrics)`` with the fitted estimator and its
        held-out test metrics (`mae`, `rmse`, `r2`).

    Raises:
        FileNotFoundError: If neither a cached feature table nor the
            raw LANL dataset can be located.
    """
    _apply_plot_style()

    section("1 / 5  --  Prepare Segment-Level Statistical Features")
    try:
        df = load_or_build_features()
    except (FileNotFoundError, ValueError):
        logger.exception("Failed to prepare LANL segment features.")
        raise
    console.print(f"[green]Feature table ready[/green]: {df.shape[0]:,} segments x {df.shape[1]} columns.")

    X = df[LANL_STAT_FEATURE_COLUMNS]
    y = df["time_to_failure"]

    section("2 / 5  --  Train / Test Split (80 / 20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=LANL_TEST_SIZE, random_state=RANDOM_STATE
    )
    console.print(f"Train: [bold]{X_train.shape}[/bold]  |  Test: [bold]{X_test.shape}[/bold]")

    section("3 / 5  --  Train XGBoost")
    console.print(f"Hyperparameters: {XGB_PARAMS}")
    model = train_xgboost_model(X_train, y_train, X_test, y_test)
    plot_training_curve(model, os.path.join(FIGURES_DIR, "xgb_training_curve.png"))

    section("4 / 5  --  Evaluation on Held-Out Test Set")
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    console.print(metrics_table("XGBoost -- Test Set Metrics", metrics))

    plot_prediction_scatter(y_test, y_pred, os.path.join(FIGURES_DIR, "xgb_actual_vs_predicted.png"))
    importance = plot_feature_importance(
        model, LANL_STAT_FEATURE_COLUMNS, os.path.join(FIGURES_DIR, "xgb_feature_importance.png")
    )
    console.print(metrics_table("XGBoost -- Feature Importance", importance.to_dict()))

    section("5 / 5  --  Persist Model Artifact")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_lanl.joblib")
    joblib.dump(model, model_path)
    console.print(f"[bold green]Model saved[/bold green] -> {model_path}")

    return model, metrics


if __name__ == "__main__":
    main()
