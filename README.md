# Recommendation System

A rating-prediction project built in two stages on a Yelp-derived dataset: a collaborative filtering baseline and a full content-based ML pipeline with deep user encoders and a LightGBM routing strategy.

**Dataset**: 967k training interactions · 414k test predictions · 541k users · 30k businesses · density 0.000059 · 1–5 star ratings

---

## Architecture

The project is structured in two modules that build progressively:

```
recomendation-system/
├── colaborative-filtering/   Classical recommendation baseline
└── content-based/            Production-grade ML pipeline (main)
```

---

## Content-Based Pipeline

The core of this project. Predicts star ratings by learning structured business representations and deep user profiles, then routing predictions through band-specific models.

### Business Representation

876-dimensional feature vector built from structured metadata:

| Block | Dimensions | Source |
|-------|-----------|--------|
| Geographic | 84 | State, city (hashed), geo-cluster |
| Categories | 558 | Multi-hot bag-of-words |
| Attributes | 225 | Flattened key/value pairs |
| Hours | 4 | Compact numerical schedule |
| Priors | 5 | Train-derived aggregates only — no raw metadata leakage |

### User Representation

Two complementary profiles per user:

- **Manual profile** — centered-weighted average of rated business vectors
- **Deep encoder** — neural network trained to produce a dense user embedding from interaction history and safe social metadata

### LightGBM Router

Predictions are routed through three specialized branches based on user history depth:

| Band | Condition | Model |
|------|-----------|-------|
| Cold start | 0 interactions | LightGBM cold model |
| Short history | 6–20 interactions | LightGBM + prefix-deep features |
| Known user | >20 interactions | LightGBM known-user model |

A deep correction model (`known_user_deep_e2e`) is trained per band to refine predictions for users with interaction history.

### Results

| Metric | Value | Scope |
|--------|-------|-------|
| Validation MAE | **0.5999** | Known-user band (v3 router) |
| Leaderboard MAE | **0.6528** | Full test set |

### Engineering Highlights

- **Leakage audit** — raw metadata stats excluded; all priors recomputed from the train set only
- **Cold-start analysis** — 41% of test rows are new users; cold branch is a first-class concern
- **Iterative development** — router evolved from v1 → v3, improving known-user MAE by −0.003 through band-specific experts
- **Full experiment log** — architecture comparisons, ablation studies, and decision rationale documented in [`docs/experiments/`](docs/experiments/)

---

## Collaborative Filtering

Classical recommendation baseline implementing multiple algorithms on explicit ratings.

| Model | Algorithm |
|-------|-----------|
| `MeanBaseline` | Global / user / item means |
| `SurpriseBaselineOnlyModel` | Bias model: μ + b_u + b_i |
| `SurpriseKNNBaselineWrapper` | KNN with baseline-adjusted similarity |
| `MatrixFactorization` | SGD matrix factorisation (PMF) |
| `BayesianPMF` | Gibbs-sampling BPMF with Normal-Wishart priors |
| `BayesianNonNegativeMF` | Variational Bayesian NMF |
| `RatingEnsemble` | Mean / median / weighted / stacking |
| `AdaptiveColdStartEnsemble` | Evidence-weighted cold-start routing |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.12 |
| Deep learning | PyTorch |
| Gradient boosting | LightGBM |
| Classical ML | scikit-learn |
| Data | pandas, numpy |
| Package manager | uv |

---

## Installation

```bash
pip install uv
uv sync
```

> **Note**: scikit-surprise (collaborative filtering module) requires a C compiler. On Windows, install via conda:
> ```bash
> conda install -c conda-forge scikit-surprise
> ```

---

## Project Structure

```
recomendation-system/
├── data/                       # train.csv · test.csv
├── colaborative-filtering/     # Classical CF baseline
│   ├── model/                  # Baseline, KNN, PMF, BPMF, NMF models
│   ├── ensemble/               # Stacking and adaptive ensembles
│   ├── metric/                 # NDCG, weighted similarity
│   └── utils/                  # Frequency-band splitting, prediction, analysis
├── content-based/              # Deep learning pipeline
│   ├── model/                  # Deep user encoder, frozen regressor, two-tower
│   ├── pipelines/              # Training and prediction scripts
│   ├── representations/        # Business and user feature builders
│   ├── utils/                  # Feature engineering, IO, leakage audit
│   └── tests/                  # pytest suite
└── docs/                       # Architecture, experiments, training guides
```
