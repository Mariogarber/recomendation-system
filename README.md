# Recommendation System

A collaborative-filtering recommendation system that implements multiple rating-prediction algorithms, ensemble methods, and evaluation utilities. The project was built for a supervised rating-prediction task (explicit ratings 1–10) and includes both classical and Bayesian matrix-factorisation approaches.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Modules](#modules)
  - [Models](#models)
  - [Metrics](#metrics)
  - [Ensemble Methods](#ensemble-methods)
  - [Utilities](#utilities)
- [API Reference](#api-reference)
- [Notebooks](#notebooks)

---

## Project Overview

The system is designed to predict ratings a user would give to an item they have not yet rated. It supports:

- **Baseline models** – global/user/item means and Surprise-based BaselineOnly
- **k-Nearest Neighbours** – user-based and item-based KNN with multiple similarity metrics
- **Matrix Factorisation** – SGD-based latent factor model (PMF) and a PMF-based regressor
- **Non-negative Matrix Factorisation** – Surprise NMF wrapper
- **Bayesian PMF** – full Gibbs-sampling BPMF with Normal-Wishart priors
- **Bayesian Non-Negative MF** – variational Bayesian model with Dirichlet/Beta priors (NNBPMF)
- **Ensemble methods** – mean, median, weighted, stacking, and adaptive cold-start ensemble
- **Evaluation** – RMSE, MAE, NDCG, per-frequency-band analysis, dataset profiling

---

## Repository Structure

```
recomendation-system/
├── README.md
├── data/
│   ├── train.csv          # 390,351 user-item-rating interactions
│   └── test.csv           # 43,320 user-item pairs to predict
└── colaborative-filtering/
    ├── model/             # Core recommendation models
    │   ├── __init__.py
    │   ├── base.py        # Abstract BaseModel
    │   ├── baseline.py    # MeanBaseline, SurpriseBaselineOnlyModel, SurpriseKNNBaselineWrapper
    │   ├── PMF.py         # MatrixFactorization, PMFRegressor, SurpriseNMFModel
    │   ├── BPMF.py        # BayesianPMF
    │   └── NNBPMF.py      # BayesianNonNegativeMF
    ├── metric/            # Evaluation metrics
    │   ├── __init__.py
    │   ├── NDCG.py        # Normalised Discounted Cumulative Gain
    │   └── similarity.py  # WeightedUserSimilarity
    ├── ensemble/          # Ensemble methods
    │   ├── __init__.py
    │   ├── ensemble.py    # RatingEnsemble (mean/median/weighted/stacking)
    │   └── adaptative.py  # AdaptiveColdStartEnsemble
    ├── utils/             # Utilities
    │   ├── __init__.py
    │   ├── split.py       # FrequencyBandSplitter, FrequencyBandEvaluator
    │   ├── predict.py     # Predictor, ThresholdItemPredictor
    │   └── analysis.py    # DatasetProfiler, analyze_biases_with_counts
    ├── analysis.ipynb
    ├── recomendation-system.ipynb
    ├── hiperparameter.ipynb
    └── knn.ipynb
```

---

## Data

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `data/train.csv` | 390,351 | `user`, `item`, `rating` | Explicit ratings (1–10 scale) used for training |
| `data/test.csv` | 43,320 | `ID`, `user`, `item` | User-item pairs for which ratings must be predicted |

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib scikit-surprise joblib
```

Key packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations, array handling |
| `pandas` | DataFrame processing |
| `scipy` | Statistical distributions (Normal-Wishart sampling, Wishart prior) |
| `scikit-learn` | Regressors (Ridge, Lasso, RandomForest, SVR, ElasticNet), utilities |
| `scikit-surprise` | BaselineOnly, KNNBaseline, NMF algorithm wrappers |
| `joblib` | Model serialisation (`save`/`load`) |
| `matplotlib` | Training curve visualisation |

---

## Quick Start

```python
import pandas as pd
from colaborative-filtering.model.baseline import SurpriseBaselineOnlyModel
from colaborative-filtering.model.PMF import MatrixFactorization

# Load data
train = pd.read_csv("data/train.csv")   # columns: user, item, rating
test  = pd.read_csv("data/test.csv")    # columns: ID, user, item

# ── Baseline model ───────────────────────────────────────────────
model = SurpriseBaselineOnlyModel(rating_scale=(1, 10), clip_range=(1, 10))
model.fit(train)

print(model.predict(user=1, item=100))          # single prediction
print(model.rmse(train))                        # training RMSE
print(model.explain_prediction(user=1, item=100))

# ── Matrix Factorisation ─────────────────────────────────────────
mf = MatrixFactorization(n_factors=20, lr=0.01, reg=0.02, n_epochs=20,
                         clip_range=(1, 10), verbose=True)
mf.fit(train)
mf.plot_training()                              # visualise convergence

# ── Batch prediction for submission ─────────────────────────────
from colaborative-filtering.utils.predict import Predictor

predictor = Predictor(test_path="data/test.csv",
                      save_path="submissions/baseline.csv")
predictor.predict(model)
```

---

## Architecture

All models inherit from `BaseModel` (abstract base class), which enforces a consistent interface across every algorithm.

```
BaseModel  (abstract)
├── fit(df)               ← required
├── predict(user, item)   ← required
├── predict_df(df)        ← batch predictions
├── rmse(df)              ← evaluation
├── mae(df)               ← evaluation
├── save(filepath)        ← serialisation
└── load(filepath)        ← deserialisation

Concrete implementations:
    MeanBaseline
    SurpriseBaselineOnlyModel
    SurpriseKNNBaselineWrapper
    MatrixFactorization
    PMFRegressor
    SurpriseNMFModel
    BayesianPMF
    BayesianNonNegativeMF
    AdaptiveColdStartEnsemble   (also a BaseModel)
```

---

## Modules

### Models

Full documentation: [`docs/models.md`](docs/models.md)

| Class | File | Algorithm |
|-------|------|-----------|
| `MeanBaseline` | `baseline.py` | User/item mean average |
| `SurpriseBaselineOnlyModel` | `baseline.py` | Bias model: μ + b_u + b_i |
| `SurpriseKNNBaselineWrapper` | `baseline.py` | KNN with baseline similarity |
| `MatrixFactorization` | `PMF.py` | SGD matrix factorisation |
| `PMFRegressor` | `PMF.py` | MF embeddings + sklearn regressor |
| `SurpriseNMFModel` | `PMF.py` | Non-negative matrix factorisation |
| `BayesianPMF` | `BPMF.py` | Bayesian PMF (Gibbs sampling) |
| `BayesianNonNegativeMF` | `NNBPMF.py` | Variational Bayesian NMF |

### Metrics

Full documentation: [`docs/metrics.md`](docs/metrics.md)

| Component | File | Description |
|-----------|------|-------------|
| `dcg` / `idcg` / `ndcg` | `NDCG.py` | NDCG@k ranking metric |
| `WeightedUserSimilarity` | `similarity.py` | Pearson + shrinkage + significance + Jaccard similarity |

### Ensemble Methods

Full documentation: [`docs/ensemble.md`](docs/ensemble.md)

| Class | File | Description |
|-------|------|-------------|
| `RatingEnsemble` | `ensemble.py` | Mean, median, weighted, stacking ensemble |
| `AdaptiveColdStartEnsemble` | `adaptative.py` | Adaptive weighting based on evidence |

### Utilities

Full documentation: [`docs/utils.md`](docs/utils.md)

| Class / Function | File | Description |
|-----------------|------|-------------|
| `FrequencyBandSplitter` | `split.py` | Split data into low/mid/high frequency bands |
| `FrequencyBandEvaluator` | `split.py` | Evaluate models per frequency band |
| `Predictor` | `predict.py` | Generate and export predictions from test CSV |
| `ThresholdItemPredictor` | `predict.py` | Hybrid predictor switching model by item frequency |
| `DatasetProfiler` | `analysis.py` | Full dataset profiling and visualisation |
| `analyze_biases_with_counts` | `analysis.py` | Correlate model biases with entity frequency |

---

## API Reference

### BaseModel

```python
class BaseModel(ABC):
    def __init__(self, name=None, clip_range=None)
    def fit(self, df: pd.DataFrame)               # abstract
    def predict(self, user, item) -> float        # abstract
    def predict_df(self, df, round_predictions=False) -> pd.DataFrame
    def rmse(self, df) -> float
    def mae(self, df, round_predictions=False) -> float
    def save(self, filepath: str)
    @staticmethod
    def load(filepath: str) -> BaseModel
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str \| None` | Human-readable model name |
| `clip_range` | `tuple \| None` | `(min, max)` to clip all predictions |

---

### SurpriseBaselineOnlyModel

```python
model = SurpriseBaselineOnlyModel(
    bsl_options={"method": "als", "n_epochs": 10, "reg_u": 12, "reg_i": 5},
    rating_scale=(1, 10),
    clip_range=(1, 10),
)
model.fit(train_df)
pred = model.predict(user, item)
explanation = model.explain_prediction(user, item)
# {"global_mean": 7.2, "user_bias": -0.5, "item_bias": 0.3, "prediction": 7.0}
```

---

### MatrixFactorization

```python
mf = MatrixFactorization(
    n_factors=20,    # latent space dimension
    lr=0.01,         # SGD learning rate
    reg=0.02,        # L2 regularisation
    n_epochs=20,     # training epochs
    clip_range=(1, 10),
)
mf.fit(train_df)
mf.plot_training()  # plot RMSE per epoch
```

---

### BayesianPMF

```python
bpmf = BayesianPMF(
    n_factors=20,
    n_iters=100,    # total MCMC iterations
    burn_in=50,     # discard first N samples
    thin=2,         # keep every Nth sample
    rating_std=1.0,
    clip_range=(1, 10),
)
bpmf.fit(train_df)
pred = bpmf.predict(user, item)         # posterior mean prediction
emb  = bpmf.get_user_embedding(user)    # latent factor vector
```

---

### RatingEnsemble

```python
from colaborative-filtering.ensemble.ensemble import RatingEnsemble

ensemble = RatingEnsemble(
    models=[model1, model2, model3],
    strategy="stacking",    # "mean" | "median" | "weighted" | "stacking"
    clip_range=(1, 10),
)
ensemble.fit_stacking(val_df)           # fit meta-learner
pred = ensemble.predict(user, item)
```

---

### AdaptiveColdStartEnsemble

```python
from colaborative-filtering.ensemble.adaptative import AdaptiveColdStartEnsemble

ensemble = AdaptiveColdStartEnsemble(
    main_model=mf,
    shrink_user=10.0,
    shrink_item=10.0,
    partial_cold_start_penalty=0.35,
    full_cold_start_penalty=0.0,
    clip_range=(1, 10),
)
ensemble.fit(train_df)
pred = ensemble.predict(user, item)
```

---

### Predictor

```python
from colaborative-filtering.utils.predict import Predictor

predictor = Predictor(
    test_path="data/test.csv",
    save_path="submissions/my_model.csv",
    round_predictions=False,
)
predictor.predict(model)    # writes CSV with columns ID, rating
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `recomendation-system.ipynb` | Main experiments and model comparison |
| `analysis.ipynb` | Dataset exploration and profiling |
| `hiperparameter.ipynb` | Hyperparameter search |
| `knn.ipynb` | KNN-specific experiments |
