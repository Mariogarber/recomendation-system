# Design: LightGBM Regressor over Frozen Embeddings

- Date: 2026-04-10
- Status: implemented
- Context: content-based branch, competition submission improvement

## Implementation Update

This design is now implemented as a hybrid submission pipeline.

Implemented scripts:

- `content-based/train_gbm_regressor.py`
- `content-based/train_gbm_submission_model.py`
- `content-based/predict_gbm_submission.py`
- `content-based/blend_deep_gbm_submission.py`

Implemented final rule:

```text
if known_train_user:
    prediction = round_half_up((deep_star + gbm_star) / 2)
else:
    prediction = gbm_star
```

## Problem

The current pipeline (FrozenEmbeddingRegressor MLP over deep embeddings) achieves 0.93 MAE on temporal
validation but 1.0496 on the leaderboard. The gap is almost entirely explained by cold-start: ~41% of
test rows belong to new users, for whom the deep user embedding is a metadata-only fallback.
The current model was validated on a temporal split that is dominated by known users, so the cold-start
degradation was invisible during development.

The leaderboard leader is at 0.60 MAE. The goal is to close the gap significantly on the content-based
side before introducing a CF ensemble.

## Root Causes

1. Cold-start gap: 41% of test users are new. The deep regressor has no explicit mechanism to handle them.
2. Review context covariate shift: `useful`/`funny`/`cool` are non-zero in train but nearly zero in test.
3. Two-stage misalignment: the deep encoder optimizes a training-head objective that diverges from the
   downstream frozen regressor's task.

## Solution

Train a LightGBM regressor on a compact, cold-start-aware feature matrix derived from the existing frozen
deep embeddings and scalar priors. The implemented version adds synthetic cold-start rows during GBM
training so the model learns a real fallback for `is_new_user = 1`.

The deep frozen regressor is now used directly in the final blended submission for known users.

## Feature Matrix (~410 features)

### Group 1: Embedding interactions (~390 features)

Source: frozen 128-dim user + 128-dim business embeddings from `competition_embeddings_v3_iter04`.

- raw user embedding (128-dim) — zeros for new users (metadata-only path already handles this)
- raw business embedding (128-dim)
- element-wise product user * business (128-dim)
- cosine similarity, dot product, user L2 norm, business L2 norm, norm gap, history_log1p (6 scalars)

For new users, the user embedding is the metadata-only fallback already produced by the deep encoder.
The model learns from `is_new_user` and `history_count` to discount user-side features for these rows.

Implementation note:

- the shipped `gbm_features.py` treats new users against train-derived priors
- new-user rows mask the user embedding to zeros
- bundle membership is not used as the cold-start definition

### Group 2: Scalar prior / cold-start features (~10 features)

All computed exclusively from `train_reviews.csv` (no metadata leakage):

- `user_mean_rating`: mean rating given by the user in train (global mean for new users)
- `user_rating_std`: std of ratings given by the user in train (0 for new users)
- `history_count`: number of train reviews for the user (0 for new users)
- `history_log1p`: log1p of history_count
- `is_new_user`: binary flag (1 if user not in train)
- `business_mean_rating`: mean rating received by the business in train
- `business_rating_std`: std of ratings received by the business in train
- `business_review_count`: number of train reviews for the business
- `global_mean_rating`: scalar constant (same for all rows)
- `mean_rating_gap`: user_mean_rating - business_mean_rating

### Group 3: Review context (~10 features)

Available at test time from the review row itself:

- `useful_log1p`, `funny_log1p`, `cool_log1p` (near-zero at test time, kept for training signal)
- `recency_days`: days since train minimum timestamp
- `month_sin`, `month_cos`: cyclical month encoding
- `weekday_sin`, `weekday_cos`: cyclical weekday encoding
- `hour_sin`, `hour_cos`: cyclical hour encoding

## Training

**Model**: LightGBM with `objective="regression_l1"` (directly optimizes MAE).

**Key hyperparameters** (starting point, subject to tuning):
```
num_leaves = 127
learning_rate = 0.05
n_estimators = 1000  (with early stopping)
min_child_samples = 50
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.1
reg_lambda = 1.0
```

**Validation**: same temporal split as the current pipeline (`val_size=0.2`, `temporal_train_validation_split`).
Early stopping monitored on val MAE. This keeps results directly comparable to the deep regressor benchmarks.

**Baseline**: a Ridge regression on Group 2 scalar features only is always fit alongside the main model,
providing a floor for comparison (consistent with the existing pipeline's `fit_ridge_embedding_baseline`).

**Predictions**: clipped to `[1, 5]` as floats during validation. Rounded to integers only at submission time.

**Submission model**: retrained on 100% of `train_reviews` using the n_estimators found by early stopping.

## Cold-Start Routing

The implemented hybrid submission now uses a fixed rule:

```text
if known_train_user:
    prediction = round_half_up((gbm_star + deep_star) / 2)
else:
    prediction = gbm_star
```

## Implementation Structure

```
content-based/
├── utils/
│   └── gbm_features.py                  # Feature matrix builder (Groups 1-3)
├── train_gbm_regressor.py               # Train + validate LightGBM, saves artifact
├── train_gbm_submission_model.py        # Retrain on full train, fixed n_estimators
└── predict_gbm_submission.py            # Generate rounded submission CSV

content-based/artifacts/
└── gbm_regressor_v1/
    ├── validation_summary.json
    ├── band_metrics.csv
    ├── experiment_ranking.csv
    ├── feature_importance.csv
    └── gbm_submission_v1/
        ├── model.pkl
        └── submission.csv
```

`gbm_features.py` is the single source of truth for feature construction. Both training and inference call
it with the same arguments — no feature duplication between scripts.

Scripts mirror the existing frozen-regressor trio for consistency with the rest of the codebase.

Current selected validation artifact: `content-based/artifacts/gbm_regressor_v1_cs30`

Current final submission artifact: `content-based/artifacts/blended_submission_v1/submission.csv`

## Success Criteria

- validation winner selected and retrained on full train
- final hybrid submission exported with full test coverage
- cold-start routing delegated to GBM-only fallback on unseen users
- deliverable file exported with columns `review_id,stars`
