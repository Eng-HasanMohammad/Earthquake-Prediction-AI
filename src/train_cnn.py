"""Model 2B -- LANL Earthquake Prediction: 1D Convolutional Neural Network.

Predicts `time_to_failure` directly from the raw 150,000-sample acoustic
emission waveform (reshaped into Conv1D-compatible tensors) -- no manual
statistical feature engineering, the network learns its own
representations.

Architecture::

    Conv1D(16, k=10, relu) -> MaxPool1D(10)
    Conv1D(32, k=10, relu) -> MaxPool1D(10)
    Flatten -> Dense(64, relu) -> Dropout(0.3) -> Dense(1, linear)

Seismological framing: where `train_xgboost.py` compresses each segment
into 7 hand-chosen statistics, this network instead consumes the
unabridged waveform, letting convolutional filters discover their own
precursor signatures (e.g. spike morphology, local frequency content)
that a fixed statistical summary might discard. The trained model is
persisted in the native Keras v3 format (`.keras`) rather than the
legacy HDF5 (`.h5`) format, which TensorFlow 2.x flags as deprecated
and which can fail to deserialize cleanly across Keras versions.

Run:
    python src/train_cnn.py
"""

from __future__ import annotations

import logging
import os
from typing import Any, Final

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback

from config import (
    CNN_CONFIG,
    FIGURES_DIR,
    LANL_TRAIN_PATH,
    LANL_TEST_SIZE,
    MODELS_DIR,
    PLOT_COLORS,
    PLOT_DPI,
    PLOT_STYLE,
    RANDOM_STATE,
    SEGMENT_SIZE,
)
from logging_utils import console, get_logger, metrics_table, section
from preprocessing import build_segment_tensor_table

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


def set_global_seeds(seed: int) -> None:
    """Seed NumPy and Keras/TensorFlow global RNGs for reproducible training.

    Args:
        seed: Integer seed applied to both NumPy and `keras.utils.set_random_seed`.
    """
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def load_raw_tensors() -> tuple[np.ndarray, np.ndarray]:
    """Stream the raw LANL signal and build `(n_segments, SEGMENT_SIZE, 1)` tensors.

    The raw signal is read in `SEGMENT_SIZE`-aligned chunks (the file is
    multi-gigabyte) with leftover rows carried across chunk boundaries so
    no segment is split. Unlike the XGBoost pipeline, no statistical
    aggregation happens here -- the full waveform of every segment is
    preserved for the CNN to consume directly.

    Returns:
        A tuple ``(X, y)`` where `X` has shape
        ``(n_segments, SEGMENT_SIZE, 1)`` and `y` has shape `(n_segments,)`.

    Raises:
        FileNotFoundError: If the raw LANL `train.csv` is missing.
        ValueError: If no full segments could be extracted.
    """
    try:
        reader = pd.read_csv(LANL_TRAIN_PATH, dtype=_READER_DTYPES, chunksize=SEGMENT_SIZE)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find LANL training data at '{LANL_TRAIN_PATH}'. "
            "Place the Kaggle 'LANL Earthquake Prediction' train.csv at "
            "this location (see config.LANL_TRAIN_PATH)."
        ) from exc

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    leftover_acoustic = np.array([], dtype=np.int16)
    leftover_ttf = np.array([], dtype=np.float64)

    for chunk in reader:
        acoustic = np.concatenate([leftover_acoustic, chunk["acoustic_data"].values])
        ttf = np.concatenate([leftover_ttf, chunk["time_to_failure"].values])

        n_full = len(acoustic) // SEGMENT_SIZE
        usable = n_full * SEGMENT_SIZE

        if n_full > 0:
            X_seg, y_seg = build_segment_tensor_table(
                acoustic[:usable], ttf[:usable], segment_size=SEGMENT_SIZE
            )
            X_chunks.append(X_seg)
            y_chunks.append(y_seg)

        leftover_acoustic = acoustic[usable:]
        leftover_ttf = ttf[usable:]

    if not X_chunks:
        raise ValueError(
            "No full segments were extracted from the raw signal -- the "
            "file may be shorter than a single SEGMENT_SIZE window."
        )

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    return X, y


def normalize_signal(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Z-score normalize acoustic amplitude using TRAIN-set statistics only.

    Computing the mean/std from the training split alone (and applying
    those same statistics to the test split) avoids test-set leakage --
    using test-set statistics would let information about held-out data
    influence the normalization applied during training.

    Args:
        X_train: Raw training tensor, shape `(n_train, SEGMENT_SIZE, 1)`.
        X_test: Raw test tensor, shape `(n_test, SEGMENT_SIZE, 1)`.

    Returns:
        A tuple ``(X_train_norm, X_test_norm, norm_stats)`` where
        `norm_stats` is ``{"mean": float, "std": float}`` -- persist
        this alongside the model for consistent inference later.
    """
    mean = float(X_train.mean())
    std = float(X_train.std())
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm, {"mean": mean, "std": std}


def build_cnn_model(input_shape: tuple[int, int], cfg: dict[str, Any]) -> keras.Model:
    """Construct and compile the 1D-CNN regression architecture.

    Args:
        input_shape: Shape of a single input sample, e.g. `(SEGMENT_SIZE, 1)`.
        cfg: Architecture/training hyperparameters (see `config.CNN_CONFIG`).

    Returns:
        A compiled `keras.Model` ready for `.fit()`.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(cfg["conv1_filters"], cfg["conv1_kernel"], activation="relu"),
        layers.MaxPooling1D(cfg["pool1_size"]),
        layers.Conv1D(cfg["conv2_filters"], cfg["conv2_kernel"], activation="relu"),
        layers.MaxPooling1D(cfg["pool2_size"]),
        layers.Flatten(),
        layers.Dense(cfg["dense_units"], activation="relu"),
        layers.Dropout(cfg["dropout_rate"]),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=cfg["optimizer"], loss=cfg["loss"], metrics=["mae"])
    return model


def evaluate_model(
    model: keras.Model, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Score the fitted model on a held-out test set.

    Args:
        model: A fitted `keras.Model` exposing `.predict`.
        X_test: Held-out, normalized feature tensor.
        y_test: Held-out `time_to_failure` targets.

    Returns:
        A tuple ``(y_pred, metrics)`` with `metrics` containing
        ``mae``, ``rmse``, and ``r2``.
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }
    return y_pred, metrics


def plot_sample_waveforms(X: np.ndarray, y: np.ndarray, save_path: str) -> None:
    """Plot two example raw acoustic waveforms with their time-to-failure labels.

    Args:
        X: Raw (pre-normalization) signal tensor, shape `(n, SEGMENT_SIZE, 1)`.
        y: Corresponding `time_to_failure` targets.
        save_path: Destination PNG path.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = [PLOT_COLORS["primary"], PLOT_COLORS["secondary"]]

    for ax, idx, color in zip(axes, [0, min(5, X.shape[0] - 1)], colors):
        ax.plot(X[idx, :, 0], linewidth=0.4, color=color)
        ax.set_title(f"Raw Acoustic Waveform -- Segment {idx} "
                     f"(time_to_failure \u2248 {y[idx]:.2f}s)", fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Sample index within segment")
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sample waveform plot to '%s'.", save_path)


def plot_prediction_scatter(y_test: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Plot actual-vs-predicted `time_to_failure` scatter and save to disk.

    Args:
        y_test: Ground-truth `time_to_failure` values.
        y_pred: Model predictions, aligned with `y_test`.
        save_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.45, s=14, color=PLOT_COLORS["tertiary"], edgecolors="none")
    lims = [float(y_test.min()), float(y_test.max())]
    ax.plot(lims, lims, "--", linewidth=1.8, color=PLOT_COLORS["accent"], label="Perfect prediction")
    ax.set_xlabel("Actual time_to_failure (s)")
    ax.set_ylabel("Predicted time_to_failure (s)")
    ax.set_title("1D-CNN: Actual vs. Predicted Time-to-Failure", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction scatter plot to '%s'.", save_path)


def plot_training_curve(history: keras.callbacks.History, save_path: str) -> None:
    """Plot train/validation MAE loss per epoch and save to disk.

    Args:
        history: The `History` object returned by `model.fit()`.
        save_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.history["loss"], label="Train MAE", color=PLOT_COLORS["primary"], linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation MAE", color=PLOT_COLORS["secondary"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss")
    ax.set_title("1D-CNN Training Curve", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curve to '%s'.", save_path)


def main() -> tuple[keras.Model, keras.callbacks.History, dict[str, float]]:
    """Run the full 1D-CNN training pipeline end-to-end.

    Returns:
        A tuple ``(model, history, metrics)`` with the trained model,
        its Keras training history, and held-out test metrics
        (`mae`, `rmse`, `r2`).

    Raises:
        FileNotFoundError: If the raw LANL dataset cannot be located.
    """
    _apply_plot_style()
    set_global_seeds(RANDOM_STATE)

    section("1 / 6  --  Load Raw Acoustic Segments as Tensors")
    try:
        X, y = load_raw_tensors()
    except (FileNotFoundError, ValueError):
        logger.exception("Failed to load raw LANL signal tensors.")
        raise
    console.print(f"[green]X shape:[/green] {X.shape}  |  [green]y shape:[/green] {y.shape}")
    plot_sample_waveforms(X, y, os.path.join(FIGURES_DIR, "cnn_sample_waveforms.png"))

    section("2 / 6  --  Train / Test Split (80 / 20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=LANL_TEST_SIZE, random_state=RANDOM_STATE
    )
    console.print(f"Train: [bold]{X_train.shape}[/bold]  |  Test: [bold]{X_test.shape}[/bold]")

    section("3 / 6  --  Normalize Signal Amplitude (Train-Set Z-Score)")
    X_train, X_test, norm_stats = normalize_signal(X_train, X_test)
    console.print(f"Normalization stats -> mean: {norm_stats['mean']:.4f}, std: {norm_stats['std']:.4f}")

    section("4 / 6  --  Build the 1D-CNN Architecture")
    model = build_cnn_model(input_shape=(SEGMENT_SIZE, 1), cfg=CNN_CONFIG)
    model.summary(print_fn=lambda line: console.print(f"[dim]{line}[/dim]"))

    section("5 / 6  --  Train the Model")
    console.print(
        f"Training for [bold]{CNN_CONFIG['epochs']}[/bold] epochs "
        f"(batch_size=[bold]{CNN_CONFIG['batch_size']}[/bold])..."
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=CNN_CONFIG["epochs"],
        batch_size=CNN_CONFIG["batch_size"],
        verbose=0,
        callbacks=[TqdmCallback(verbose=1)],
    )
    plot_training_curve(history, os.path.join(FIGURES_DIR, "cnn_training_curve.png"))

    section("6 / 6  --  Evaluation & Persistence")
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    console.print(metrics_table("1D-CNN -- Test Set Metrics", metrics))
    plot_prediction_scatter(y_test, y_pred, os.path.join(FIGURES_DIR, "cnn_actual_vs_predicted.png"))

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "cnn_1d_lanl.keras")
    model.save(model_path)
    console.print(f"[bold green]Model saved[/bold green] -> {model_path}")
    console.print(f"[bold yellow]Normalization stats (required at inference):[/bold yellow] {norm_stats}")

    return model, history, metrics


if __name__ == "__main__":
    main()
