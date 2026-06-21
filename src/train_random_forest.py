"""Model 1 -- USGS Significant Earthquakes (1965-2016): Random Forest.

Predicts `Magnitude` and `Depth` from `Timestamp`, `Latitude`, and
`Longitude` using a `RandomForestRegressor`, tuned via `GridSearchCV`
over `n_estimators`.

Seismological framing: earthquake magnitude and depth are governed by
complex, nonlinear tectonic processes (plate boundary type, fault
geometry, regional stress accumulation) that vary by location and,
more weakly, by time. A Random Forest is a strong tabular baseline here
because it captures nonlinear interactions between latitude, longitude,
and time without requiring an explicit geophysical model.

Run:
    python src/train_random_forest.py
"""

from __future__ import annotations

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from config import (
    FIGURES_DIR,
    MODELS_DIR,
    PLOT_COLORS,
    PLOT_DPI,
    PLOT_STYLE,
    RANDOM_STATE,
    RF_CV_FOLDS,
    RF_PARAM_GRID,
    RF_SCORING,
    USGS_FEATURE_COLUMNS,
    USGS_PROCESSED_PATH,
    USGS_RAW_PATH,
    USGS_TARGET_COLUMNS,
    USGS_TEST_SIZE,
)
from logging_utils import console, get_logger, metrics_table, section
from preprocessing import load_and_clean_usgs

logger: logging.Logger = get_logger(__name__)


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


def load_dataset() -> pd.DataFrame:
    """Load and persist the cleaned USGS dataset.

    Returns:
        The cleaned `pandas.DataFrame` (see `preprocessing.load_and_clean_usgs`).

    Raises:
        FileNotFoundError: If the raw USGS CSV is missing.
        KeyError: If the raw CSV is missing expected columns.
    """
    df = load_and_clean_usgs(USGS_RAW_PATH)
    os.makedirs(os.path.dirname(USGS_PROCESSED_PATH), exist_ok=True)
    df.to_csv(USGS_PROCESSED_PATH, index=False)
    logger.info("Saved processed dataset to '%s' (%d rows).", USGS_PROCESSED_PATH, len(df))
    return df


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> GridSearchCV:
    """Run `GridSearchCV` over `n_estimators` for a multi-output Random Forest.

    Args:
        X_train: Training features (`Timestamp`, `Latitude`, `Longitude`).
        y_train: Training targets (`Magnitude`, `Depth`).

    Returns:
        The fitted `GridSearchCV` object, exposing `.best_estimator_` and
        `.best_params_`.
    """
    base_model = RandomForestRegressor(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=RF_PARAM_GRID,
        cv=RF_CV_FOLDS,
        scoring=RF_SCORING,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """Score the fitted model on a held-out test set.

    Args:
        model: A fitted regressor exposing `.predict`.
        X_test: Held-out feature matrix.
        y_test: Held-out target matrix.

    Returns:
        A tuple ``(y_pred, metrics)`` where `metrics` contains the
        aggregate ``r2``, ``mae``, and ``rmse`` across both targets.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    return y_pred, metrics


def plot_grid_search_curve(grid_search: GridSearchCV, save_path: str) -> None:
    """Plot mean CV R2 vs. `n_estimators` with error bars, then save to disk.

    Args:
        grid_search: A fitted `GridSearchCV` object.
        save_path: Destination PNG path.
    """
    results = pd.DataFrame(grid_search.cv_results_)
    results = results[["param_n_estimators", "mean_test_score", "std_test_score"]]
    results = results.sort_values("param_n_estimators")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        results["param_n_estimators"], results["mean_test_score"],
        yerr=results["std_test_score"], marker="o", capsize=4,
        color=PLOT_COLORS["primary"], ecolor=PLOT_COLORS["secondary"],
        linewidth=2, markersize=7,
    )
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Mean CV $R^2$ Score")
    ax.set_title("GridSearchCV: $R^2$ vs. n_estimators", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved grid search curve to '%s'.", save_path)


def plot_prediction_scatter(
    y_test: pd.DataFrame, y_pred: np.ndarray, target_columns: list[str], save_path: str
) -> None:
    """Plot actual-vs-predicted scatter panels (one per target) and save to disk.

    Args:
        y_test: Ground-truth targets.
        y_pred: Model predictions, aligned with `y_test`.
        target_columns: Names of the target columns, in `y_pred` column order.
        save_path: Destination PNG path.
    """
    y_pred_df = pd.DataFrame(y_pred, columns=target_columns, index=y_test.index)
    fig, axes = plt.subplots(1, len(target_columns), figsize=(7 * len(target_columns), 6))
    if len(target_columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, target_columns):
        ax.scatter(
            y_test[col], y_pred_df[col], alpha=0.45, s=14,
            color=PLOT_COLORS["primary"], edgecolors="none",
        )
        lims = [y_test[col].min(), y_test[col].max()]
        ax.plot(lims, lims, "--", linewidth=1.8, color=PLOT_COLORS["accent"], label="Perfect prediction")
        ax.set_xlabel(f"Actual {col}")
        ax.set_ylabel(f"Predicted {col}")
        ax.set_title(f"Actual vs. Predicted {col}", fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction scatter plot to '%s'.", save_path)


def plot_feature_importance(
    model: RandomForestRegressor, feature_columns: list[str], save_path: str
) -> pd.Series:
    """Plot and save a horizontal feature-importance bar chart.

    Args:
        model: A fitted `RandomForestRegressor` exposing `feature_importances_`.
        feature_columns: Feature names, aligned with `feature_importances_`.
        save_path: Destination PNG path.

    Returns:
        A `pandas.Series` of importances sorted descending, indexed by
        feature name.
    """
    importances = pd.Series(
        model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="viridis", legend=False, edgecolor="none", ax=ax)
    ax.set_title("Random Forest Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot to '%s'.", save_path)
    return importances


def main() -> tuple[RandomForestRegressor, dict[str, float]]:
    """Run the full Random Forest training pipeline end-to-end.

    Returns:
        A tuple ``(best_model, metrics)`` with the fitted estimator and
        its held-out test metrics (`r2`, `mae`, `rmse`).

    Raises:
        FileNotFoundError: If the raw USGS dataset cannot be located.
    """
    _apply_plot_style()

    section("1 / 5  --  Load & Preprocess USGS Data")
    try:
        df = load_dataset()
    except (FileNotFoundError, KeyError):
        logger.exception("Failed to load the USGS dataset.")
        raise
    console.print(f"[green]Loaded[/green] {len(df):,} clean rows.")

    X = df[USGS_FEATURE_COLUMNS]
    y = df[USGS_TARGET_COLUMNS]

    section("2 / 5  --  Train / Test Split (80 / 20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=USGS_TEST_SIZE, random_state=RANDOM_STATE
    )
    console.print(f"Train: [bold]{X_train.shape}[/bold]  |  Test: [bold]{X_test.shape}[/bold]")

    section("3 / 5  --  Hyperparameter Tuning (GridSearchCV)")
    console.print(f"Searching n_estimators over {RF_PARAM_GRID['n_estimators']} ({RF_CV_FOLDS}-fold CV)...")
    grid_search = tune_random_forest(X_train, y_train)
    best_model = grid_search.best_estimator_
    console.print(
        f"[bold green]Best n_estimators:[/bold green] {grid_search.best_params_['n_estimators']}  "
        f"|  [bold green]Best CV R2:[/bold green] {grid_search.best_score_:.4f}"
    )
    plot_grid_search_curve(grid_search, os.path.join(FIGURES_DIR, "rf_grid_search_curve.png"))

    section("4 / 5  --  Evaluation on Held-Out Test Set")
    y_pred, metrics = evaluate_model(best_model, X_test, y_test)
    console.print(metrics_table("Random Forest -- Test Set Metrics", metrics))

    for col in USGS_TARGET_COLUMNS:
        y_pred_df = pd.DataFrame(y_pred, columns=USGS_TARGET_COLUMNS, index=y_test.index)
        col_r2 = r2_score(y_test[col], y_pred_df[col])
        col_mae = mean_absolute_error(y_test[col], y_pred_df[col])
        console.print(f"  [cyan]{col:>10s}[/cyan] -> R2: {col_r2:.4f} | MAE: {col_mae:.4f}")

    plot_prediction_scatter(
        y_test, y_pred, USGS_TARGET_COLUMNS,
        os.path.join(FIGURES_DIR, "rf_actual_vs_predicted.png"),
    )
    plot_feature_importance(
        best_model, USGS_FEATURE_COLUMNS,
        os.path.join(FIGURES_DIR, "rf_feature_importance.png"),
    )

    section("5 / 5  --  Persist Model Artifact")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "random_forest_usgs.joblib")
    joblib.dump(best_model, model_path)
    console.print(f"[bold green]Model saved[/bold green] -> {model_path}")

    return best_model, metrics


if __name__ == "__main__":
    main()
