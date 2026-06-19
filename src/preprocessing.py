"""
preprocessing.py
================
Data-cleaning and feature-engineering utilities shared by the
USGS (Random Forest) and LANL (XGBoost / 1D-CNN) pipelines.
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Dataset 1: USGS Significant Earthquakes (1965-2016)
# ------------------------------------------------------------------
def load_and_clean_usgs(path: str) -> pd.DataFrame:
    """
    Load the raw USGS catalogue and engineer a unified Timestamp feature.

    The raw CSV stores `Date` (mm/dd/yyyy) and `Time` (HH:MM:SS) as
    separate string columns. A handful of rows use ISO-8601 datetime
    strings instead (a known quirk of this public dataset), so both
    formats are handled defensively.

    Returns a DataFrame filtered down to the columns the model actually
    consumes: Timestamp, Latitude, Longitude, Magnitude, Depth.
    """
    df = pd.read_csv(path)

    timestamps = []
    for date_str, time_str in zip(df["Date"], df["Time"]):
        try:
            dt = pd.to_datetime(
                f"{date_str} {time_str}", format="%m/%d/%Y %H:%M:%S"
            )
        except (ValueError, TypeError):
            # Fallback for rows already stored as ISO datetime strings
            try:
                dt = pd.to_datetime(date_str)
            except (ValueError, TypeError):
                dt = pd.NaT
        timestamps.append(dt)

    df["Timestamp"] = pd.Series(timestamps)

    # Convert to a numeric (Unix epoch, seconds) representation so it can
    # feed directly into a tree-based regressor.
    df["Timestamp"] = df["Timestamp"].apply(
        lambda x: x.timestamp() if pd.notnull(x) else np.nan
    )

    # Keep only the columns required for the modeling task
    keep_cols = ["Timestamp", "Latitude", "Longitude", "Magnitude", "Depth"]
    df = df[keep_cols].dropna().reset_index(drop=True)

    return df


# ------------------------------------------------------------------
# Dataset 2: LANL Earthquake Prediction (acoustic signals)
# ------------------------------------------------------------------
def extract_segment_features(segment: np.ndarray) -> dict:
    """
    Compute the 7 statistical descriptors used as XGBoost input features
    for a single 150,000-sample acoustic segment.

    Features: mean, std, max, min, median, 1st percentile, 99th percentile.
    """
    return {
        "mean": np.mean(segment),
        "std": np.std(segment),
        "max": np.max(segment),
        "min": np.min(segment),
        "median": np.median(segment),
        "p01": np.percentile(segment, 1),
        "p99": np.percentile(segment, 99),
    }


def build_segment_feature_table(
    acoustic_data: np.ndarray,
    time_to_failure: np.ndarray,
    segment_size: int = 150_000,
) -> pd.DataFrame:
    """
    Slide a non-overlapping window of `segment_size` across the raw LANL
    signal, extracting 7 statistical features per segment plus the
    `time_to_failure` target sampled at the END of each segment (this
    matches the original Kaggle competition's labeling convention).
    """
    n_segments = len(acoustic_data) // segment_size
    rows = []

    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size
        segment = acoustic_data[start:end]

        features = extract_segment_features(segment)
        features["time_to_failure"] = time_to_failure[end - 1]
        rows.append(features)

    return pd.DataFrame(rows)


def build_segment_tensor_table(
    acoustic_data: np.ndarray,
    time_to_failure: np.ndarray,
    segment_size: int = 150_000,
):
    """
    Slice the raw signal into (segment_size, 1) tensors for the 1D-CNN
    pipeline — no statistical aggregation, the network learns its own
    feature representations directly from the waveform.

    Returns
    -------
    X : np.ndarray of shape (n_segments, segment_size, 1)
    y : np.ndarray of shape (n_segments,)
    """
    n_segments = len(acoustic_data) // segment_size
    X = np.zeros((n_segments, segment_size, 1), dtype=np.float32)
    y = np.zeros(n_segments, dtype=np.float32)

    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size
        X[i, :, 0] = acoustic_data[start:end]
        y[i] = time_to_failure[end - 1]

    return X, y
