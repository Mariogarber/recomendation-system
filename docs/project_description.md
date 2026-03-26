# Project Description: Collaborative Filtering Recommendation System

## 1. Problem Statement

The goal of this project is to build a **rating-prediction system** for a collaborative-filtering task. Given a dataset of explicit user–item ratings (on a 1–10 integer scale), the system must learn to predict the rating that any user would assign to any item they have not yet rated. Formally, the task is a supervised regression problem:

> **Given** a training set of triples *(user, item, rating)*, **predict** the rating for each *(user, item)* pair in the held-out test set, minimising Root Mean Squared Error (RMSE).

The dataset is highly sparse (~98.25% unobserved entries), which makes cold-start handling—accurate prediction for users or items with few or no historical ratings—a central challenge.

---

## 2. Dataset

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `data/train.csv` | 390,351 | `user`, `item`, `rating` | Explicit ratings (1–10 scale) used for training |
| `data/test.csv` | 43,320 | `ID`, `user`, `item` | User–item pairs whose ratings must be predicted |

**Key statistics**

| Metric | Value |
|--------|-------|
| Unique users | ~6,040 |
| Unique items | ~3,706 |
| Rating scale | 1 – 10 (integers) |
| Matrix density | ~1.75% |
| Matrix sparsity | ~98.25% |

The extreme sparsity means that simple nearest-neighbour and matrix-factorisation approaches must be augmented with regularisation, Bayesian priors, and cold-start strategies to generalise well.

---

## 3. System Architecture

All models implement a shared **`BaseModel`** interface (abstract base class), which enforces a consistent API across every algorithm and allows ensemble methods to combine them interchangeably.

```
BaseModel  (abstract base class)
│
├── fit(df)                  ← train on a DataFrame of (user, item, rating)
├── predict(user, item)      ← return a single predicted rating
├── predict_df(df)           ← batch predictions over a DataFrame
├── rmse(df)                 ← Root Mean Squared Error
├── mae(df)                  ← Mean Absolute Error
├── save(filepath)           ← serialise model with joblib
└── load(filepath)           ← deserialise model

Concrete models
    MeanBaseline
    SurpriseBaselineOnlyModel
    SurpriseKNNBaselineWrapper
    MatrixFactorization
    PMFRegressor
    SurpriseNMFModel
    BayesianPMF
    BayesianNonNegativeMF
    AdaptiveColdStartEnsemble   (BaseModel + cold-start logic)
```

The codebase is divided into four packages:

| Package | Responsibility |
|---------|---------------|
| `model/` | Core recommendation algorithms (8 classes) |
| `metric/` | Evaluation metrics (NDCG, weighted similarity) |
| `ensemble/` | Ensemble and adaptive blending strategies |
| `utils/` | Dataset analysis, frequency-band splitting, submission generation |

---

## 4. Models

### 4.1 Baseline Models (`model/baseline.py`)

Baselines capture global, user-level, and item-level biases without learning latent factors.

**`MeanBaseline`**

Prediction is the arithmetic mean of the user's average rating and the item's average rating:

```
r̂(u, i) = (mean_u + mean_i) / 2
```

Falls back to the global mean for cold-start users or items.

**`SurpriseBaselineOnlyModel`**

An additive bias model fitted via Alternating Least Squares (ALS) or SGD (wraps Surprise's `BaselineOnly`):

```
r̂(u, i) = μ + b_u + b_i
```

where μ is the global mean, b_u is the user bias, and b_i is the item bias.  
Cold-start entities receive a bias of 0 (i.e., the prediction falls back to μ).

**`SurpriseKNNBaselineWrapper`**

Item-based (or user-based) k-nearest-neighbour model that uses the baseline-adjusted cosine similarity and computes a weighted average of neighbour ratings, also wrapping Surprise's `KNNBaseline`.

---

### 4.2 Matrix Factorisation (`model/PMF.py`)

**`MatrixFactorization`** — SGD Probabilistic Matrix Factorisation (PMF)

Decomposes the rating matrix into low-rank user and item embedding matrices P ∈ ℝ^(|U|×K) and Q ∈ ℝ^(|I|×K):

```
r̂(u, i) = μ + b_u + b_i + P_u · Q_i
```

Parameters are optimised by minimising the L2-regularised squared error with SGD over all observed ratings. The number of latent factors K, learning rate, regularisation coefficient, and number of epochs are configurable.

**`PMFRegressor`**

A two-stage model: first trains PMF to obtain user and item embeddings, then stacks those embeddings as features for a scikit-learn regressor (Ridge, RandomForest, SVR, Lasso, or ElasticNet).

**`SurpriseNMFModel`**

Non-negative Matrix Factorisation (NMF) wrapper around Surprise's implementation. All embedding values are constrained to be non-negative, giving factor dimensions a "topic strength" interpretation.

---

### 4.3 Bayesian PMF (`model/BPMF.py`)

**`BayesianPMF`** — full Bayesian treatment via Gibbs sampling with Normal-Wishart priors.

**Generative model:**

```
Λ_U ~ Wishart(W_0, ν_0)                   precision matrix prior for users
μ_U ~ Normal(μ_0, (β_0 Λ_U)^{−1})

u_k ~ Normal(μ_U, Λ_U^{−1})               user latent factors

Λ_V ~ Wishart(W_0, ν_0)                   precision matrix prior for items
μ_V ~ Normal(μ_0, (β_0 Λ_V)^{−1})

v_i ~ Normal(μ_V, Λ_V^{−1})               item latent factors

r(u,i) ~ Normal(u_u · v_i, σ_r²)          observed rating
```

The Gibbs sampler alternates between sampling the precision matrices and mean vectors from their Normal-Wishart posteriors, and then sampling user and item factors given the current hyperparameters. Predictions are taken as the posterior mean of all non-burn-in samples.

---

### 4.4 Bayesian Non-Negative MF (`model/NNBPMF.py`)

**`BayesianNonNegativeMF`** — variational Bayesian NMF with Dirichlet/Beta priors.

**Generative model:**

```
φ_u ~ Dirichlet(α, …, α)                  user preference over K topics
κ_{i,k} ~ Beta(β, β)                      item like-probability per topic
z_{u,i} ~ Categorical(φ_u)               latent topic assignment
ρ_{u,i} ~ Binomial(R, κ_{i, z_{u,i}})    implicit rating
r*_{u,i} = ρ_{u,i} / R                   normalised to [0, R]
```

Inference uses Variational EM (coordinate-ascent updates). The model supports prediction of the full probability distribution over ratings (`predict_proba_each_rating`) and interpretable explanations (`explain_prediction`) that expose the topic-level contributions.

---

### 4.5 Item-Based KNN (`model/KNN.py`)

**`ItemKNNModel`**

Computes item–item similarity from the training matrix and predicts a weighted average of K-nearest neighbour ratings. Supported distance metrics: cosine, Pearson correlation, Euclidean, and Manhattan. For cosine/correlation, similarity = 1 − distance; for Euclidean/Manhattan, similarity = 1 / (1 + distance).

---

### 4.6 SVD Wrappers (`model/SVD.py`)

- **`SurpriseSVDModel`**: Wraps Surprise's SVD (150 latent factors by default).
- **`SurpriseSVDppModel`**: Wraps Surprise's SVD++, which additionally incorporates implicit feedback from the set of items a user has rated.

---

## 5. Ensemble Methods

### 5.1 `RatingEnsemble` (`ensemble/ensemble.py`)

Combines predictions from any number of base models using one of four strategies:

| Strategy | Description |
|----------|-------------|
| `"mean"` | Simple arithmetic average of all model predictions |
| `"median"` | Median (more robust to outlier predictions) |
| `"weighted"` | Weighted average; weights set manually or learned from a validation set |
| `"stacking"` | Ridge meta-learner trained on base-model predictions as features |

Weights can be learnt by minimising validation RMSE (`fit_weights_from_errors`) or via a constrained optimiser (`fit_weights_optimized`).

### 5.2 `AdaptiveColdStartEnsemble` (`ensemble/adaptative.py`)

Inherits from `BaseModel` and blends a main collaborative model with a simpler baseline model using evidence-based weights:

```
weight_user = count_u / (count_u + shrink_u)
weight_item = count_i / (count_i + shrink_i)
w           = weight_user × weight_item          (capped and adjusted)

r̂(u, i) = w × main_pred + (1 − w) × baseline_pred
```

For partially cold-start pairs (one unseen entity) the weight is reduced by a configurable `partial_cold_start_penalty`; for fully cold-start pairs (both entities unseen) the weight is set to `full_cold_start_penalty`.

---

## 6. Evaluation Framework

### 6.1 Point-prediction metrics

Every model exposes:

- **RMSE** — Root Mean Squared Error (primary competition metric)
- **MAE** — Mean Absolute Error

### 6.2 Ranking metric

- **NDCG@k** (Normalised Discounted Cumulative Gain) — measures ranking quality of the top-k recommended items.

### 6.3 Frequency-band analysis (`utils/split.py`)

**`FrequencyBandSplitter`** partitions users or items into three frequency bands—*low* (cold-start), *mid*, and *high* (popular)—using either equal-quantile splits or manual thresholds. **`FrequencyBandEvaluator`** then computes RMSE and MAE separately per band, making it possible to diagnose whether a model degrades specifically on cold-start entities.

### 6.4 Dataset profiling (`utils/analysis.py`)

**`DatasetProfiler`** reports interaction counts, density/sparsity, Gini coefficient (concentration measure), rating distribution percentiles, and the fraction of users and items falling below common cold-start thresholds (< 5 / 10 / 20 interactions).

**`analyze_biases_with_counts`** correlates a fitted model's learned biases with entity interaction frequencies to detect systematic over- or under-prediction for rare vs. popular entities.

---

## 7. Utilities

| Class / Function | Purpose |
|-----------------|---------|
| `Predictor` | Reads the test CSV, calls `model.predict_df`, writes a submission CSV with columns `ID` and `rating`; falls back to 7.0 on any exception |
| `ThresholdItemPredictor` | Routes predictions through a `rare_model` for items below a count threshold and a `frequent_model` otherwise |
| `FrequencyBandSplitter` | Splits data into low / mid / high frequency bands (quantile or threshold mode) |
| `FrequencyBandEvaluator` | Evaluates any `BaseModel` per frequency band |
| `DatasetProfiler` | Full dataset statistics, histograms, and cold-start diagnostics |
| `analyze_biases_with_counts` | Correlates model biases with entity frequency |

---

## 8. Notebooks

| Notebook | Description |
|----------|-------------|
| `recomendation-system.ipynb` | End-to-end experiments: model training, comparison table, and final submission generation |
| `analysis.ipynb` | Exploratory data analysis and dataset profiling |
| `hiperparameter.ipynb` | Hyperparameter search for key models (PMF, BPMF, SVD) |
| `knn.ipynb` | KNN-specific experiments, similarity metric comparison |
| `notebook_svd_knn.ipynb` | Head-to-head comparison of SVD and KNN models |

---

## 9. Key Design Decisions

1. **Shared `BaseModel` interface** — enforces a uniform API (fit / predict / predict_df / rmse / mae / save / load) so that any model can be dropped into an ensemble or evaluation pipeline without modification.

2. **Cold-start aware design** — every model defines an explicit fallback strategy (global mean, bias only, prior-based prediction) so that predictions are always returned, even for unseen users or items.

3. **Frequency-band evaluation** — splitting evaluation by entity interaction frequency exposes where each model excels or struggles, guiding ensemble weighting.

4. **Adaptive ensemble** — the `AdaptiveColdStartEnsemble` automatically increases the weight of the simpler baseline for entities with limited evidence, smoothly blending from baseline (cold-start) to the main model (warm entities) with no hard threshold.

5. **Bayesian approaches for uncertainty** — `BayesianPMF` and `BayesianNonNegativeMF` provide principled uncertainty estimates and reduce overfitting through priors, which is especially beneficial in a sparse-data regime.

---

## 10. Summary

This project implements a comprehensive collaborative-filtering pipeline for explicit rating prediction. Starting from simple mean baselines, it progressively introduces bias models, matrix factorisation, Bayesian methods, nearest-neighbour approaches, and multi-model ensembles. A shared abstract base class, frequency-band evaluation, and an adaptive cold-start ensemble make the system modular, extensible, and robust to the challenges of real-world sparse rating data.
