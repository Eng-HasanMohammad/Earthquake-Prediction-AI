"""Data-cleaning and feature-engineering utilities.

Shared by the USGS (Random Forest) and LANL (XGBoost / 1D-CNN) pipelines.
Two independent seismic data modalities are handled here:

    * USGS catalog records -- sparse, tabular, one row per earthquake
      event, with categorical date/time strings that must be normalized
      into a single numeric timestamp feature.
    * LANL acoustic emission signals -- a dense, continuous laboratory
      time series that must be windowed into fixed-length segments
      before it can be consumed by either a gradient-boosted tree
      ensemble (via statistical descriptors) or a 1D-CNN (via the raw
      waveform).
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from tqdm import tqdm

from logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)

# Columns retained for the USGS modeling task. Declared once here so the
# "contract" between `load_and_clean_usgs` and its callers is explicit.
_USGS_KEEP_COLUMNS: Final[list[str]] = [
    "Timestamp", "Latitude", "Longitude", "Magnitude", "Depth",
]


# ---------------------------------------------------------------------------
# Dataset 1: USGS Significant Earthquakes (1965-2016)
# ---------------------------------------------------------------------------
def _parse_event_timestamp(date_str: str, time_str: str) -> pd.Timestamp:
    """Parse a single USGS `Date`/`Time` pair into a `pandas.Timestamp`.

    The raw USGS catalog predominantly stores dates as ``mm/dd/yyyy`` and
    times as ``HH:MM:SS`` in separate columns, but a known data-quality
    quirk in this public CSV means a handful of rows instead store a
    single ISO-8601 datetime string in the `Date` column. Both formats
    are handled defensively so the function never raises on malformed
    input -- it degrades to `pandas.NaT`, which is then dropped
    downstream by `load_and_clean_usgs`.

    Args:
        date_str: Raw `Date` field, e.g. ``"01/02/1965"`` or an ISO
            datetime string.
        time_str: Raw `Time` field, e.g. ``"13:44:18"``.

    Returns:
        A parsed `pandas.Timestamp`, or `pandas.NaT` if parsing fails
        under both the primary and fallback strategies.
    """
    try:
        return pd.to_datetime(f"{date_str} {time_str}", format="%m/%d/%Y %H:%M:%S")
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_str)
        except (ValueError, TypeError):
            return pd.NaT


def load_and_clean_usgs(path: str) -> pd.DataFrame:
    """Load the raw USGS catalog and engineer a unified `Timestamp` feature.

    Seismological rationale: tree-based regressors cannot natively
    consume categorical date/time strings, but earthquake occurrence
    exhibits genuine temporal structure (aftershock sequences, regional
    seismic cycles). Collapsing `Date` + `Time` into a single Unix-epoch
    `Timestamp` preserves that ordinal temporal signal as a plain numeric
    feature a `RandomForestRegressor` can split on directly.

    Args:
        path: Filesystem path to the raw USGS CSV (the Kaggle
            "Significant Earthquakes, 1965-2016" `database.csv`).

    Returns:
        A `pandas.DataFrame` with exactly the columns
        ``["Timestamp", "Latitude", "Longitude", "Magnitude", "Depth"]``,
        with all rows containing unparseable timestamps or missing
        values dropped.

    Raises:
        FileNotFoundError: If `path` does not point to an existing file.
        KeyError: If the expected `Date`/`Time`/`Latitude`/`Longitude`/
            `Magnitude`/`Depth` columns are absent from the source CSV.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        logger.error("USGS raw file not found at '%s'.", path)
        raise FileNotFoundError(
            f"Could not find USGS dataset at '{path}'. Place the Kaggle "
            "'Significant Earthquakes, 1965-2016' database.csv at this "
            "location (see config.USGS_RAW_PATH)."
        ) from exc

    required_cols = {"Date", "Time", "Latitude", "Longitude", "Magnitude", "Depth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"USGS source CSV is missing expected column(s): {sorted(missing)}. "
            "Verify this is the official Kaggle 'database.csv' file."
        )

    logger.info("Parsing %d event timestamps...", len(df))
    timestamps = [
        _parse_event_timestamp(date_str, time_str)
        for date_str, time_str in tqdm(
            zip(df["Date"], df["Time"]), total=len(df), desc="Parsing timestamps", unit="row"
        )
    ]
    df["Timestamp"] = pd.Series(timestamps, index=df.index)

    # Convert to numeric Unix-epoch seconds so it can feed directly into
    # a tree-based regressor (which cannot consume datetime objects).
    df["Timestamp"] = df["Timestamp"].apply(
        lambda ts: ts.timestamp() if pd.notnull(ts) else np.nan
    )

    df = df[_USGS_KEEP_COLUMNS]
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(
            "Dropped %d/%d rows (%.2f%%) with unparseable or missing values.",
            n_dropped, n_before, 100 * n_dropped / n_before,
        )

    logger.info("Clean USGS dataset ready: %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Dataset 2: LANL Earthquake Prediction (acoustic signals)
# ---------------------------------------------------------------------------
def extract_segment_features(segment: np.ndarray) -> dict[str, float]:
    """Compute 7 statistical descriptors for one acoustic emission segment.

    Seismological rationale: in stick-slip laboratory experiments, the
    acoustic emission signal grows more volatile (higher variance, more
    extreme tail values) as the simulated fault approaches failure. Tail
    percentiles (`p01`, `p99`) and extrema (`min`, `max`) specifically
    capture the precursor micro-fracture "spikes" that a raw `mean`/`std`
    summary would smooth over -- this is the central feature-engineering
    insight that lets a gradient-boosted tree model approximate the
    raw-waveform information a deep network would otherwise need to
    learn on its own.

    Args:
        segment: 1-D array of raw `acoustic_data` amplitude samples for
            a single fixed-length window.

    Returns:
        A dict with keys ``mean``, ``std``, ``max``, ``min``, ``median``,
        ``p01``, ``p99``.

    Raises:
        ValueError: If `segment` is empty.
    """
    if segment.size == 0:
        raise ValueError("Cannot extract statistical features from an empty segment.")

    return {
        "mean": float(np.mean(segment)),
        "std": float(np.std(segment)),
        "max": float(np.max(segment)),
        "min": float(np.min(segment)),
        "median": float(np.median(segment)),
        "p01": float(np.percentile(segment, 1)),
        "p99": float(np.percentile(segment, 99)),
    }


def build_segment_feature_table(
    acoustic_data: np.ndarray,
    time_to_failure: np.ndarray,
    segment_size: int = 150_000,
) -> pd.DataFrame:
    """Slide a non-overlapping window across the raw signal, extracting features.

    For every `segment_size`-sample window, computes the 7 statistical
    descriptors from `extract_segment_features` plus the `time_to_failure`
    target sampled at the **end** of the segment -- matching the original
    LANL Kaggle competition's labeling convention, where the target
    represents time remaining until failure as of the segment's final
    sample.

    Args:
        acoustic_data: 1-D array of raw acoustic amplitude samples.
        time_to_failure: 1-D array of time-to-failure labels, aligned
            index-for-index with `acoustic_data`.
        segment_size: Number of samples per non-overlapping window.
            Defaults to 150,000 (the convention used throughout this
            project; see `config.SEGMENT_SIZE`).

    Returns:
        A `pandas.DataFrame` with one row per full segment and columns
        ``["mean", "std", "max", "min", "median", "p01", "p99",
        "time_to_failure"]``.

    Raises:
        ValueError: If `acoustic_data` and `time_to_failure` have
            mismatched lengths, or if `segment_size` exceeds the
            available signal length (zero full segments).
    """
    if len(acoustic_data) != len(time_to_failure):
        raise ValueError(
            f"acoustic_data (len={len(acoustic_data)}) and time_to_failure "
            f"(len={len(time_to_failure)}) must have equal length."
        )

    n_segments = len(acoustic_data) // segment_size
    if n_segments == 0:
        raise ValueError(
            f"Signal length ({len(acoustic_data)}) is shorter than "
            f"segment_size ({segment_size}); no full segments available."
        )

    rows: list[dict[str, float]] = []
    for i in tqdm(range(n_segments), desc="Extracting statistical features", unit="segment"):
        start, end = i * segment_size, (i + 1) * segment_size
        segment = acoustic_data[start:end]
        features = extract_segment_features(segment)
        features["time_to_failure"] = float(time_to_failure[end - 1])
        rows.append(features)

    return pd.DataFrame(rows)


def build_segment_tensor_table(
    acoustic_data: np.ndarray,
    time_to_failure: np.ndarray,
    segment_size: int = 150_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the raw signal into `(segment_size, 1)` tensors for the 1D-CNN.

    Unlike `build_segment_feature_table`, no statistical aggregation is
    performed -- the full raw waveform of each segment is preserved so a
    convolutional network can learn its own hierarchical feature
    representations (local waveform shape, spike morphology, frequency
    content) directly from the signal.

    Args:
        acoustic_data: 1-D array of raw acoustic amplitude samples.
        time_to_failure: 1-D array of time-to-failure labels, aligned
            index-for-index with `acoustic_data`.
        segment_size: Number of samples per non-overlapping window.
            Defaults to 150,000.

    Returns:
        A tuple ``(X, y)`` where `X` has shape
        ``(n_segments, segment_size, 1)`` (`float32`) and `y` has shape
        ``(n_segments,)`` (`float32`).

    Raises:
        ValueError: If `acoustic_data` and `time_to_failure` have
            mismatched lengths, or if no full segments are available.
    """
    if len(acoustic_data) != len(time_to_failure):
        raise ValueError(
            f"acoustic_data (len={len(acoustic_data)}) and time_to_failure "
            f"(len={len(time_to_failure)}) must have equal length."
        )

    n_segments = len(acoustic_data) // segment_size
    if n_segments == 0:
        raise ValueError(
            f"Signal length ({len(acoustic_data)}) is shorter than "
            f"segment_size ({segment_size}); no full segments available."
        )

    X = np.zeros((n_segments, segment_size, 1), dtype=np.float32)
    y = np.zeros(n_segments, dtype=np.float32)

    for i in tqdm(range(n_segments), desc="Building raw-signal tensors", unit="segment"):
        start, end = i * segment_size, (i + 1) * segment_size
        X[i, :, 0] = acoustic_data[start:end]
        y[i] = time_to_failure[end - 1]

    return X, y
