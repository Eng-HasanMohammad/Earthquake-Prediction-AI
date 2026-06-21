<div align="center">

# 🌋 AI-Powered Earthquake Prediction System

### Created by **Hasan Mohammad**

*Automatic Control & Computer Engineer — Faculty of Mechanical and Electrical Engineering, Homs University*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%20v3-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-3776AB?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Overview

Earthquakes sit at the intersection of two very different physical scales: the **geospatial, historical record** of seismic events recorded across the globe, and the **microscopic, laboratory-controlled** acoustic emissions of rock under shear stress. This repository treats both as regression problems and builds a complete, reproducible pipeline around each — from raw data to publication-ready evaluation plots.

Rather than forcing a single model onto two physically distinct problems, this project deliberately implements **three independent pipelines**, each matched to the structure of its data:

| Pipeline | Dataset | Task | Model |
|---|---|---|---|
| `train_random_forest.py` | USGS Significant Earthquakes (1965–2016) | Predict `Magnitude` & `Depth` from location/time | `RandomForestRegressor` + `GridSearchCV` |
| `train_xgboost.py` | LANL Earthquake Prediction (lab acoustic signals) | Predict `time_to_failure` from statistical signal descriptors | `XGBRegressor` |
| `train_cnn.py` | LANL Earthquake Prediction (lab acoustic signals) | Predict `time_to_failure` directly from the raw waveform | 1D-CNN (`tensorflow.keras`) |

This contrast is intentional: the **USGS** pipeline tests whether classical tabular ML can recover structure in real-world, sparsely-sampled tectonic data, while the two **LANL** pipelines compare a *feature-engineered* gradient-boosting approach against an *end-to-end* deep learning approach on the same raw acoustic signal — a direct, controlled comparison between domain-driven and representation-driven learning.

---

## 🌍 Scientific Context

**USGS Significant Earthquakes (1965–2016)** is a curated catalog of significant global seismic events, combining decades of geophysical monitoring into structured tabular records of location, depth, magnitude, and time. It is widely used as a benchmark for geospatial and time-aware regression.

**LANL Earthquake Prediction** originates from a Los Alamos National Laboratory experiment that mimics fault behavior at laboratory scale: acoustic emissions from a sheared rock sample are recorded continuously, with `time_to_failure` marking the countdown to the next labquake. The dataset's scientific premise is to test whether the path to failure is encoded in the acoustic signal — an open question in earthquake physics with direct implications for early-warning research.

---

## 🏗️ Repository Architecture

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

**Engineering principles behind the structure:**

- 🔧 **Single source of truth for configuration** — every path, random seed, and hyperparameter lives in `src/config.py`, resolved relative to the repository root via the file's own location. The pipelines run identically regardless of the working directory they're launched from, with zero manual path edits after cloning.
- 📦 **Memory-safe data handling** — `train.csv` (LANL) is several gigabytes; rather than loading it into memory, the pipelines stream it in chunks and cache engineered feature tables to `data/processed/`, so repeat runs are fast.
- 🪵 **Production-style logging** — `logging_utils.py` routes all output through Python's standard `logging` module and a shared `rich.logging.RichHandler`, replacing raw `print()` calls. `rich.console.Console` renders section banners and metrics tables for clean, readable terminal output.
- 🧱 **Modern artifact formats** — the CNN is saved in the native Keras v3 format (`.keras`) rather than the deprecated legacy HDF5 (`.h5`) format, ensuring reliable reloading on current TensorFlow/Keras versions.

---

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Data

Download the two datasets from Kaggle and place them at:

- `data/raw/database.csv` — [Significant Earthquakes, 1965–2016](https://www.kaggle.com/datasets/usgs/earthquake-database)
- `data/raw/train.csv` — [LANL Earthquake Prediction](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction)

> **Note:** `train.csv` is several gigabytes. All LANL pipelines stream it in chunks rather than loading it into memory at once, and cache the engineered XGBoost feature table to `data/processed/` so repeat runs are fast.

## 🚀 Running

```bash
python src/train_random_forest.py
python src/train_xgboost.py
python src/train_cnn.py
```

Each script prints styled progress via `rich`, shows `tqdm` progress bars during feature extraction / training, evaluates on a held-out 20% test split, saves publication-ready plots to `reports/figures/`, and persists the trained model to `models/`.

---

## 🔬 Notes on This Refactor

- The CNN model is saved in the native Keras v3 format (`.keras`) rather than the legacy HDF5 (`.h5`) format used in the original prototype — `.h5` is deprecated in current TensorFlow/Keras and was found to fail on reload in this environment's Keras version.
- Logging uses Python's standard `logging` module routed through a shared `rich.logging.RichHandler`, replacing raw `print()` calls, while `rich.console.Console` renders section banners and metrics tables for a clean terminal experience.

---

## 👤 Author

**Hasan Mohammad**
Automatic Control & Computer Engineer
Faculty of Mechanical and Electrical Engineering, Homs University

This project was built as part of a graduate-school portfolio in Machine Learning and AI-driven signal/geospatial analysis, in preparation for Master's studies in **Green Industrial Engineering** and **Control Engineering**.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
