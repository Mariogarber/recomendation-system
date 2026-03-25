# Ensemble Methods Reference

All ensemble utilities live under `colaborative-filtering/ensemble/`.

---

## Table of Contents

- [RatingEnsemble](#ratingensemble)
- [AdaptiveColdStartEnsemble](#adaptivecoldstartensemble)
- [Choosing an Ensemble Strategy](#choosing-an-ensemble-strategy)

---

## RatingEnsemble

**File:** `ensemble/ensemble.py`

Combines predictions from a list of base models using one of four aggregation strategies.

### Supported Strategies

| Strategy | Description |
|----------|-------------|
| `"mean"` | Simple average of all base model predictions |
| `"median"` | Median of predictions (more robust to outliers) |
| `"weighted"` | Weighted average; weights can be set manually or learned |
| `"stacking"` | Trains a Ridge regression meta-learner on base model outputs |

### Constructor

```python
RatingEnsemble(
    models,              # list of BaseModel instances
    strategy="mean",     # "mean" | "median" | "weighted" | "stacking"
    weights=None,        # list[float] | None; manual weights for strategy="weighted"
    clip_range=None,     # (min, max) to clip final predictions
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list` | required | Base models (must implement `predict(user, item)`) |
| `strategy` | `str` | `"mean"` | Aggregation strategy |
| `weights` | `list[float] \| None` | `None` | Manual weights (auto-normalised to sum to 1) |
| `clip_range` | `tuple \| None` | `None` | Clip predictions to this range |

### Methods

#### `predict(user, item) → float`
Aggregate base model predictions for a single pair. `NaN` predictions are excluded before aggregation.

#### `predict_df(df) → pd.DataFrame`
Batch predictions. Appends a `'prediction'` column to the input DataFrame.

#### `fit_weights_from_errors(val_df) → self`
Learn weights from validation errors using softmax over inverse errors:

```
weight[k] = exp(-error[k]) / Σ exp(-error[j])
```

#### `fit_weights_optimized(val_df) → self`
Learn weights via constrained optimisation (minimise RMSE, subject to weights ≥ 0, Σ weights = 1).

#### `fit_stacking(val_df, alpha=1.0) → self`
Train a Ridge regression meta-learner on top of base model predictions.

```
meta_pred = Ridge.predict([pred_model_1, pred_model_2, ...])
```

#### `get_model_predictions(user, item) → dict`
Return individual predictions from each base model as `{model_name: prediction}`.

#### `get_weights() → dict`
Return the current weights as `{model_name: weight}`.

### Example

```python
from ensemble.ensemble import RatingEnsemble

ensemble = RatingEnsemble(
    models=[baseline_model, mf_model, bpmf_model],
    strategy="weighted",
    clip_range=(1, 10),
)

# Learn weights from validation data
ensemble.fit_weights_from_errors(val_df)
print(ensemble.get_weights())

# Stacking ensemble
ensemble_stack = RatingEnsemble(
    models=[baseline_model, mf_model, bpmf_model],
    strategy="stacking",
    clip_range=(1, 10),
)
ensemble_stack.fit_stacking(val_df)

pred = ensemble_stack.predict(user=1, item=42)
rmse = ensemble_stack.rmse(val_df)
```

---

## AdaptiveColdStartEnsemble

**File:** `ensemble/adaptative.py`

A hybrid ensemble that inherits from `BaseModel`. It blends a main collaborative model with a robust baseline using **adaptive weights** that depend on the amount of evidence available for each user and item.

### Motivation

Pure collaborative filtering degrades for users or items with few interactions (cold-start problem). This ensemble down-weights the main model prediction and up-weights the baseline when evidence is sparse.

### Architecture

```
AdaptiveColdStartEnsemble
├── main_model          ← any BaseModel (collaborative signal)
└── baseline            ← user/item/global means (robust fallback)

prediction = w * main_pred + (1 - w) * baseline_pred
```

### Adaptive Weight Computation

The blending weight `w` is computed from user and item interaction counts using shrinkage:

```
weight_user = count_user / (count_user + shrink_user)
weight_item = count_item / (count_item + shrink_item)

# Combined weight (geometric mean, capped at max_count_weight)
w = weight_user * weight_item
```

Additional penalties:
- **Partial cold-start**: one of user/item is unknown → weight multiplied by `(1 - partial_cold_start_penalty)`
- **Full cold-start**: both unknown → weight set to `full_cold_start_penalty`

### Constructor

```python
AdaptiveColdStartEnsemble(
    main_model: BaseModel,
    shrink_user=10.0,
    shrink_item=10.0,
    max_count_weight=20,
    partial_cold_start_penalty=0.35,
    full_cold_start_penalty=0.0,
    min_rating=None,
    max_rating=None,
    use_item_popularity_in_weight=True,
    fit_main_model=True,
    clip_range=None,
    name=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `main_model` | `BaseModel` | required | Collaborative model to blend |
| `shrink_user` | `float` | `10.0` | Shrinkage factor for user weight |
| `shrink_item` | `float` | `10.0` | Shrinkage factor for item weight |
| `max_count_weight` | `int` | `20` | Max count used in weight computation (caps at 1.0) |
| `partial_cold_start_penalty` | `float` | `0.35` | Penalty when one of user/item is unseen |
| `full_cold_start_penalty` | `float` | `0.0` | Weight for main model when both are unseen |
| `use_item_popularity_in_weight` | `bool` | `True` | Include item frequency in adaptive weight |
| `fit_main_model` | `bool` | `True` | If `False`, assume `main_model` is already trained |

### Methods

All `BaseModel` methods are available:

| Method | Description |
|--------|-------------|
| `fit(df)` | Train baseline stats and (optionally) the main model |
| `predict(user, item)` | Adaptive blend of main model and baseline |
| `predict_df(df)` | Batch adaptive predictions |
| `rmse(df)` / `mae(df)` | Standard evaluation metrics |

### Example

```python
from ensemble.adaptative import AdaptiveColdStartEnsemble
from model.PMF import MatrixFactorization

mf = MatrixFactorization(n_factors=20, n_epochs=20, clip_range=(1, 10))

ensemble = AdaptiveColdStartEnsemble(
    main_model=mf,
    shrink_user=10.0,
    shrink_item=10.0,
    partial_cold_start_penalty=0.35,
    full_cold_start_penalty=0.0,
    clip_range=(1, 10),
)

ensemble.fit(train_df)

pred = ensemble.predict(user=1, item=42)
val_rmse = ensemble.rmse(val_df)
```

### Weight Visualisation

For a given `(user, item)` pair:

```
count_user = 3   → weight_user = 3/(3+10) ≈ 0.23
count_item = 50  → weight_item = 50/(50+10) ≈ 0.83
combined_w = 0.23 * 0.83 ≈ 0.19

pred = 0.19 * main_pred + 0.81 * baseline_pred
```

This shows that a user with only 3 interactions heavily relies on the baseline.

---

## Choosing an Ensemble Strategy

| Strategy | Best For |
|----------|---------|
| `RatingEnsemble("mean")` | Quick baseline combination with low variance |
| `RatingEnsemble("median")` | Robust to outlier model predictions |
| `RatingEnsemble("weighted")` | When some models are clearly stronger than others |
| `RatingEnsemble("stacking")` | Maximum accuracy with sufficient validation data |
| `AdaptiveColdStartEnsemble` | Datasets with many cold-start users or items |
