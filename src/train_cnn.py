"""
train_cnn.py
============
Model 2B — LANL Earthquake Prediction (Acoustic Signals)

Predicts time_to_failure directly from raw 150,000-sample acoustic
windows (reshaped into Conv1D-compatible tensors) using a 1D
Convolutional Neural Network — no manual statistical feature
engineering required, the network learns its own representations.

Architecture:
    Conv1D(16, k=10, relu) -> MaxPool1D(10)
    Conv1D(32, k=10, relu) -> MaxPool1D(10)
    Flatten -> Dense(64, relu) -> Dropout(0.3) -> Dense(1, linear)

Run:
    python src/train_cnn.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CNN_CONFIG,
    LANL_TRAIN_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SEGMENT_SIZE,
    LANL_TEST_SIZE,
)
from preprocessing import build_segment_tensor_table


def load_raw_tensors():
    """
    Stream the raw LANL signal in 150k-row chunks and build the
    (n_segments, 150000, 1) tensor dataset consumed directly by the CNN.

    NOTE: Because each segment carries its full 150,000-length raw
    waveform (vs. 7 scalar features for XGBoost), this tensor set is
    memory-heavy. For very large datasets, consider a tf.data
    generator/pipeline instead of materializing the full array.
    """
    dtypes = {"acoustic_data": np.int16, "time_to_failure": np.float64}
    reader = pd.read_csv(LANL_TRAIN_PATH, dtype=dtypes, chunksize=SEGMENT_SIZE)

    X_chunks, y_chunks = [], []
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

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    return X, y


def normalize_signal(X_train, X_test):
    """Standardize acoustic amplitude using train-set statistics only."""
    mean = X_train.mean()
    std = X_train.std()
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm, {"mean": float(mean), "std": float(std)}


def build_cnn_model(input_shape, cfg: dict) -> keras.Model:
    """Construct the 1D-CNN regression architecture per the thesis spec."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(
                filters=cfg["conv1_filters"],
                kernel_size=cfg["conv1_kernel"],
                activation="relu",
            ),
            layers.MaxPooling1D(pool_size=cfg["pool1_size"]),
            layers.Conv1D(
                filters=cfg["conv2_filters"],
                kernel_size=cfg["conv2_kernel"],
                activation="relu",
            ),
            layers.MaxPooling1D(pool_size=cfg["pool2_size"]),
            layers.Flatten(),
            layers.Dense(cfg["dense_units"], activation="relu"),
            layers.Dropout(cfg["dropout_rate"]),
            layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(optimizer=cfg["optimizer"], loss=cfg["loss"], metrics=["mae"])
    return model


def main():
    # ----------------------------------------------------------------
    # 1. Load raw signal segments as 3D tensors
    # ----------------------------------------------------------------
    print("[1/6] Loading raw acoustic segments as tensors...")
    X, y = load_raw_tensors()
    print(f"      -> X shape: {X.shape} | y shape: {y.shape}")

    # ----------------------------------------------------------------
    # 2. Train/test split
    # ----------------------------------------------------------------
    print("[2/6] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=LANL_TEST_SIZE, random_state=RANDOM_STATE
    )

    # ----------------------------------------------------------------
    # 3. Normalize signal amplitude
    # ----------------------------------------------------------------
    print("[3/6] Normalizing signal amplitude (z-score, train stats)...")
    X_train, X_test, norm_stats = normalize_signal(X_train, X_test)

    # ----------------------------------------------------------------
    # 4. Build & train the 1D-CNN
    # ----------------------------------------------------------------
    print("[4/6] Building 1D-CNN architecture...")
    model = build_cnn_model(input_shape=(SEGMENT_SIZE, 1), cfg=CNN_CONFIG)
    model.summary()

    print(
        f"[5/6] Training for {CNN_CONFIG['epochs']} epochs "
        f"(batch_size={CNN_CONFIG['batch_size']})..."
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=CNN_CONFIG["epochs"],
        batch_size=CNN_CONFIG["batch_size"],
        verbose=1,
    )

    # ----------------------------------------------------------------
    # 5. Evaluate
    # ----------------------------------------------------------------
    print("[6/6] Evaluating on test set...")
    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"      -> MAE  : {mae:.3f} seconds")
    print(f"      -> RMSE : {rmse:.3f} seconds")
    print(f"      -> R2   : {r2:.4f}")

    # ----------------------------------------------------------------
    # 6. Save model
    # ----------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "cnn_1d_lanl.h5")
    model.save(model_path)
    print(f"      -> Saved to {model_path}")
    print(f"      -> Normalization stats (apply at inference): {norm_stats}")

    return model, history, {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    main()
