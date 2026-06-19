<div align="center">

# 🌍 Earthquake Prediction via Artificial Intelligence

### *Decoding Seismic Signatures with Ensemble Learning & Deep Sequential Networks*

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0-007ACC?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
</p>

<p>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/status-completed-success?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/datasets-USGS%20%7C%20LANL-blueviolet?style=flat-square" alt="Datasets"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome"/>
</p>

<br/>

> *"Earthquakes are not random — they are the audible signature of stress accumulating in the Earth's crust. The challenge isn't the absence of a pattern; it's that the pattern is buried in extreme non-linearity."*

</div>

---

## 📖 Overview

Earthquakes are notoriously difficult to forecast because the underlying physical process — stress build-up and sudden rupture along a fault plane — is governed by **highly non-linear, chaotic dynamics**. Traditional statistical and physical models struggle to capture the subtle precursory signatures embedded in seismic and acoustic data.

This project frames earthquake prediction as **two complementary machine learning problems**:

1. **Geospatial-Temporal Characterization** — given *where* and *when* a significant earthquake historically occurred, can we predict its **magnitude** and **depth**?
2. **Precursor Signal Regression** — given raw **acoustic emission data** from a laboratory fault-friction experiment, can we predict the **time remaining until failure**?

By tackling both problems with three distinct AI paradigms — **ensemble tree-based learning** and **deep sequential learning** — this project benchmarks how classical feature engineering stacks up against end-to-end representation learning on real seismic data.

---

## 🧬 Methodology

This project follows a **dual-dataset, tri-model** experimental design:

```
                    ┌────────────────────────────┐
                    │   Earthquake Prediction AI  │
                    └──────────────┬─────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                          │
   ┌──────────▼───────────┐                 ┌────────────▼────────────┐
   │   Dataset 1: USGS     │                 │   Dataset 2: LANL        │
   │   Catalogue (1965-16) │                 │   Acoustic Signals       │
   └──────────┬───────────┘                 └────────────┬────────────┘
              │                                           │
   ┌──────────▼───────────┐              ┌────────────────┴────────────────┐
   │  Random Forest        │              │                                 │
   │  Regressor             │   ┌─────────▼─────────┐         ┌────────────▼───────────┐
   │  (Ensemble Learning)   │   │   XGBoost           │         │   1D-CNN                │
   └────────────────────────┘   │   (Statistical       │         │   (Deep Sequential       │
                                 │   Feature Ensemble)  │         │   Representation Learning)│
                                 └──────────────────────┘         └────────────────────────┘
```

### 1️⃣ Dataset 1 — USGS Significant Earthquakes (1965–2016)

| | |
|---|---|
| **Task** | Regression — predict `Magnitude` and `Depth` |
| **Inputs** | `Timestamp`, `Latitude`, `Longitude` |
| **Preprocessing** | Merge `Date` + `Time` into a unified Unix `Timestamp`; filter to essential columns |
| **Model** | `RandomForestRegressor` |
| **Tuning** | `GridSearchCV` over `n_estimators ∈ {10, 20, 50, 100, 200, 500}` |
| **Split** | 80% train / 20% test |

> Random Forests excel here because the magnitude-depth relationship with geospatial-temporal coordinates is **non-linear but low-dimensional** — an ensemble of decision trees captures regional seismic zone behavior without requiring explicit feature crosses.

### 2️⃣ Dataset 2 — LANL Earthquake Prediction (Acoustic Emission Signals)

This dataset originates from a **laboratory earthquake experiment**: a fault-friction apparatus generates continuous acoustic emission data, and the goal is to predict `time_to_failure` — the seconds remaining until the simulated fault slips.

Two fundamentally different AI paradigms are benchmarked on the *same* raw signal:

#### 🌲 Model 2A — XGBoost (Statistical Feature Engineering)

The raw signal is segmented into **non-overlapping windows of 150,000 samples**. For each window, **7 statistical descriptors** are extracted:

`mean` · `std` · `max` · `min` · `median` · `1st percentile` · `99th percentile`

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `learning_rate` | 0.1 |
| `max_depth` | 5 |

#### 🧠 Model 2B — 1D Convolutional Neural Network (Deep Sequential Learning)

Rather than hand-crafting features, the raw 150,000-sample segments are reshaped into 3D tensors and fed directly into a 1D-CNN, which learns its own hierarchical representations of the waveform.

```
Input (150000, 1)
   │
   ▼
Conv1D(16 filters, kernel=10, ReLU)
   │
   ▼
MaxPooling1D(pool_size=10)
   │
   ▼
Conv1D(32 filters, kernel=10, ReLU)
   │
   ▼
MaxPooling1D(pool_size=10)
   │
   ▼
Flatten()
   │
   ▼
Dense(64, ReLU)
   │
   ▼
Dropout(0.3)
   │
   ▼
Dense(1, Linear)  →  time_to_failure
```

| Training Config | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | MAE |
| Epochs | 10 |
| Batch Size | 16 |

---

## 📊 Results

<div align="center">

| Model | Dataset | Task | Metric | Score |
|:---|:---|:---|:---:|:---:|
| 🌲 **Random Forest** | USGS (1965–2016) | Magnitude & Depth Regression | R² Score | **87.5%** |
| 🚀 **XGBoost** | LANL Acoustic Signals | Time-to-Failure Regression | MAE | **2.013 s** |
| 🧠 **1D-CNN** | LANL Acoustic Signals | Time-to-Failure Regression | MAE | **2.005 s** |

</div>

### 🔍 Key Insight

> The **1D-CNN marginally outperforms XGBoost** (2.005s vs. 2.013s MAE) despite requiring **zero manual feature engineering** — the convolutional filters learn to detect precursory waveform patterns (amplitude spikes, volatility shifts) automatically. This demonstrates that **end-to-end deep representation learning is competitive with, and can slightly surpass, carefully engineered statistical features** for high-frequency seismic signal regression — at the cost of significantly higher training time and compute.

---

## 🗂️ Repository Structure

```
earthquake-prediction-ai/
├── data/
│   ├── raw/                       # Place downloaded datasets here (gitignored)
│   └── processed/                 # Cached/engineered feature tables (gitignored)
├── notebooks/
│   ├── 01_random_forest_usgs.ipynb
│   ├── 02_xgboost_lanl.ipynb
│   └── 03_cnn_lanl.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                  # Centralized paths, seeds & hyperparameters
│   ├── preprocessing.py           # Shared data-cleaning & feature engineering
│   ├── train_random_forest.py     # Model 1 — standalone training script
│   ├── train_xgboost.py           # Model 2A — standalone training script
│   └── train_cnn.py               # Model 2B — standalone training script
├── models/                        # Saved model artifacts (.joblib / .h5)
├── reports/
│   └── figures/                   # Exported plots & visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/earthquake-prediction-ai.git
cd earthquake-prediction-ai

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download the Datasets

| Dataset | Source | Place at |
|---|---|---|
| USGS Significant Earthquakes | [Kaggle ↗](https://www.kaggle.com/datasets/usgs/earthquake-database) | `data/raw/database.csv` |
| LANL Earthquake Prediction | [Kaggle Competition ↗](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction) | `data/raw/train.csv` |

### 3a. Run via Jupyter Notebooks (recommended for exploration)

```bash
jupyter lab notebooks/
```

Then open, in order:
1. `01_random_forest_usgs.ipynb`
2. `02_xgboost_lanl.ipynb`
3. `03_cnn_lanl.ipynb`

### 3b. Run via Standalone Scripts (recommended for reproducibility)

```bash
python src/train_random_forest.py
python src/train_xgboost.py
python src/train_cnn.py
```

Each script will preprocess data, train its model, print evaluation metrics, and save the trained artifact to `models/`.

---

## 🛠️ Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Wrangling** | Pandas, NumPy |
| **Classical ML** | Scikit-Learn (Random Forest, GridSearchCV) |
| **Gradient Boosting** | XGBoost |
| **Deep Learning** | TensorFlow / Keras (1D-CNN) |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib, HDF5 |

</div>

---

## 🎓 Academic Context

This repository is the recreated, open-source companion codebase for the graduation thesis **"Earthquake Prediction via Artificial Intelligence."** It demonstrates a complete applied ML workflow — from raw geophysical/acoustic data ingestion through feature engineering, model selection, hyperparameter optimization, and rigorous quantitative evaluation — across both **classical ensemble methods** and **modern deep sequential architectures**.

---

<div align="center">

### 📬 Contact & Citation

If you use this work in academic research, please cite the original thesis.
Questions, issues, and pull requests are welcome!

<br/>

**⭐ If this project helped you, consider giving it a star!**

</div>
