# AI-Powered Earthquake Prediction System

Two complementary modeling approaches to earthquake-related regression,
built on two distinct, high-profile seismological datasets:

| Pipeline | Dataset | Task | Model |
|---|---|---|---|
| `train_random_forest.py` | USGS Significant Earthquakes (1965–2016) | Predict `Magnitude` & `Depth` from location/time | `RandomForestRegressor` + `GridSearchCV` |
| `train_xgboost.py` | LANL Earthquake Prediction (lab acoustic signals) | Predict `time_to_failure` from statistical signal descriptors | `XGBRegressor` |
| `train_cnn.py` | LANL Earthquake Prediction (lab acoustic signals) | Predict `time_to_failure` directly from the raw waveform | 1D-CNN (`tensorflow.keras`) |

## Project structure

```
.
├── data/
│   ├── raw/              # place Kaggle CSVs here (not committed)
│   └── processed/        # auto-generated cleaned/cached data
├── models/                # auto-generated trained model artifacts
├── notebooks/              # exploratory notebooks (01, 02, 03)
├── reports/figures/        # auto-generated publication-style plots
└── src/
    ├── config.py          # all paths, seeds, hyperparameters
    ├── logging_utils.py   # shared rich-based logging/console helpers
    ├── preprocessing.py   # USGS + LANL data cleaning & feature engineering
    ├── train_random_forest.py
    ├── train_xgboost.py
    └── train_cnn.py
```

All paths in `config.py` are resolved relative to the repo root (derived
from `config.py`'s own file location), so every script runs correctly
regardless of your current working directory — no manual path edits
needed after cloning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the two datasets from Kaggle and place them at:

- `data/raw/database.csv` — [Significant Earthquakes, 1965-2016](https://www.kaggle.com/datasets/usgs/earthquake-database)
- `data/raw/train.csv` — [LANL Earthquake Prediction](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction)

> **Note:** `train.csv` is several gigabytes. All LANL pipelines stream
> it in chunks rather than loading it into memory at once, and cache
> the engineered XGBoost feature table to `data/processed/` so repeat
> runs are fast.

## Running

```bash
python src/train_random_forest.py
python src/train_xgboost.py
python src/train_cnn.py
```

Each script prints styled progress via `rich`, shows `tqdm` progress
bars during feature extraction / training, evaluates on a held-out
20% test split, saves publication-ready plots to `reports/figures/`,
and persists the trained model to `models/`.

## Notes on this refactor

- The CNN model is saved in the native Keras v3 format (`.keras`)
  rather than the legacy HDF5 (`.h5`) format used in the original
  prototype — `.h5` is deprecated in current TensorFlow/Keras and was
  found to fail on reload in this environment's Keras version.
- Logging uses Python's standard `logging` module routed through a
  shared `rich.logging.RichHandler`, replacing raw `print()` calls,
  while `rich.console.Console` renders section banners and metrics
  tables for a clean terminal experience.
