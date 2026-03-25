# Models Reference

All models live under `colaborative-filtering/model/` and share a common base class.

---

## Table of Contents

- [BaseModel](#basemodel)
- [MeanBaseline](#meanbaseline)
- [SurpriseBaselineOnlyModel](#surprisebaselineonlymodel)
- [SurpriseKNNBaselineWrapper](#surpriseknnbaselinewrapper)
- [MatrixFactorization](#matrixfactorization)
- [PMFRegressor](#pmfregressor)
- [SurpriseNMFModel](#surprisenmfmodel)
- [BayesianPMF](#bayesianpmf)
- [BayesianNonNegativeMF](#bayesiannonnegativemf)
- [Cold-Start Handling Summary](#cold-start-handling-summary)

---

## BaseModel

**File:** `model/base.py`

Abstract base class that every model must extend. Provides a consistent API for training, predicting, evaluating, and serialising models.

### Constructor

```python
BaseModel(name=None, clip_range=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | class name | Human-readable identifier |
| `clip_range` | `tuple \| None` | `None` | `(min, max)` tuple to clip predictions |

### Abstract Methods

```python
def fit(self, df: pd.DataFrame)
```
Train the model on a DataFrame with columns `['user', 'item', 'rating']`.

```python
def predict(self, user, item) -> float
```
Return the predicted rating for a single `(user, item)` pair.

### Concrete Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict_df` | `(df, round_predictions=False) → pd.DataFrame` | Batch predictions; appends `'prediction'` column |
| `rmse` | `(df) → float` | Root Mean Squared Error on a labelled DataFrame |
| `mae` | `(df, round_predictions=False) → float` | Mean Absolute Error |
| `save` | `(filepath: str)` | Serialise model to disk via `joblib.dump` |
| `load` | `(filepath: str) → BaseModel` | Deserialise model from disk (static method) |

> **Note:** `predict_df` iterates row-by-row calling `predict(u, i)` for every pair, so it can be slow for large DataFrames. Consider overriding it in subclasses for vectorised predictions.

### Example

```python
model.save("models/my_model.pkl")
model2 = BaseModel.load("models/my_model.pkl")
```

---

## MeanBaseline

**File:** `model/baseline.py`

The simplest possible baseline. Predicts the average of the user mean and the item mean.

```
pred(u, i) = (mean_u + mean_i) / 2
```

Falls back to the global mean for unknown users or items.

### Constructor

```python
MeanBaseline(name=None, clip_range=None)
```

### Example

```python
from model.baseline import MeanBaseline

model = MeanBaseline()
model.fit(train_df)
pred = model.predict(user=1, item=42)
```

---

## SurpriseBaselineOnlyModel

**File:** `model/baseline.py`

Wraps the Surprise library's `BaselineOnly` algorithm. Learns user and item biases using ALS or SGD optimisation.

```
pred(u, i) = global_mean + user_bias[u] + item_bias[i]
```

For unknown users or items, the corresponding bias is treated as 0 (falls back towards the global mean).

### Constructor

```python
SurpriseBaselineOnlyModel(
    bsl_options=None,    # {"method": "als", "n_epochs": 10, "reg_u": 12, "reg_i": 5}
    verbose=False,
    rating_scale=None,   # e.g. (1, 10); inferred from data if None
    clip_range=None,
    name=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bsl_options` | `dict \| None` | ALS defaults | Surprise bias-fitting options |
| `rating_scale` | `tuple \| None` | inferred | `(min_rating, max_rating)` for Surprise Reader |
| `clip_range` | `tuple \| None` | `None` | Clip final predictions |

### Extra Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict_batch(df)` | `np.ndarray` | Vectorised batch prediction |
| `recommend(user, top_k=10, exclude_seen=True)` | `list[(item, score)]` | Top-k recommendations for a user |
| `get_user_bias(user)` | `float` | User bias from trained model |
| `get_item_bias(item)` | `float` | Item bias from trained model |
| `explain_prediction(user, item)` | `dict` | Decompose prediction into components |

### Example

```python
model = SurpriseBaselineOnlyModel(rating_scale=(1, 10), clip_range=(1, 10))
model.fit(train_df)

# Explain a prediction
explanation = model.explain_prediction(user=5, item=100)
# {"global_mean": 7.1, "user_bias": -0.4, "item_bias": 0.6, "prediction": 7.3}

# Get top recommendations
recs = model.recommend(user=5, top_k=5)
# [(item_id, score), ...]
```

---

## SurpriseKNNBaselineWrapper

**File:** `model/baseline.py`

Wraps Surprise's `KNNBaseline`. Supports both user-user and item-item collaborative filtering with a learned baseline correction.

### Constructor

```python
SurpriseKNNBaselineWrapper(
    k=40,
    min_k=1,
    user_based=False,                # True = user-user, False = item-item
    sim_name="pearson_baseline",     # "pearson_baseline" | "cosine" | "msd" | "pearson"
    shrinkage=100,
    bsl_options=None,
    min_rating=None,
    max_rating=None,
    unknown_strategy="global_mean",  # "global_mean" | "nan" | float | "surprise"
    clip_range=None,
    verbose=False,
    name=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `40` | Maximum number of neighbours |
| `min_k` | `int` | `1` | Minimum neighbours required for a prediction |
| `user_based` | `bool` | `False` | `True` for user-user, `False` for item-item |
| `sim_name` | `str` | `"pearson_baseline"` | Similarity metric |
| `shrinkage` | `int \| float` | `100` | Shrinkage regularisation for similarity |
| `unknown_strategy` | `str \| float` | `"global_mean"` | Fallback for unknown users/items |

#### `unknown_strategy` options

| Value | Behaviour |
|-------|-----------|
| `"global_mean"` | Use the training global mean |
| `"nan"` | Return `np.nan` |
| `float` value | Return that fixed value |
| `"surprise"` | Let Surprise handle the prediction |

### Example

```python
model = SurpriseKNNBaselineWrapper(
    k=40,
    user_based=False,
    sim_name="pearson_baseline",
    unknown_strategy="global_mean",
    clip_range=(1, 10),
)
model.fit(train_df)
pred = model.predict(user=1, item=42)
```

---

## MatrixFactorization

**File:** `model/PMF.py`

Explicit-rating matrix factorisation trained with Stochastic Gradient Descent.

```
pred(u, i) = global_mean + user_bias[u] + item_bias[i] + P[u] · Q[i]
```

where `P` is the user-embedding matrix and `Q` is the item-embedding matrix, both of dimension `n_factors`.

### Constructor

```python
MatrixFactorization(
    n_factors=20,
    lr=0.01,
    reg=0.02,
    n_epochs=20,
    use_bias=True,
    init_std=0.1,
    random_state=42,
    shuffle=True,
    clip_range=None,
    verbose=True,
    name=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_factors` | `int` | `20` | Dimension of latent space |
| `lr` | `float` | `0.01` | SGD learning rate |
| `reg` | `float` | `0.02` | L2 regularisation coefficient |
| `n_epochs` | `int` | `20` | Number of training epochs |
| `use_bias` | `bool` | `True` | Include global/user/item bias terms |
| `init_std` | `float` | `0.1` | Standard deviation for weight initialisation |
| `shuffle` | `bool` | `True` | Shuffle training pairs each epoch |

### Cold-Start Handling

| Scenario | Prediction |
|----------|-----------|
| User and item both unknown | `global_mean` |
| User unknown | `global_mean + item_bias[i]` |
| Item unknown | `global_mean + user_bias[u]` |
| Both known | Full MF prediction |

### Extra Methods

| Method | Description |
|--------|-------------|
| `plot_training()` | Plot RMSE over training epochs |

### Example

```python
mf = MatrixFactorization(n_factors=20, lr=0.01, reg=0.02, n_epochs=20,
                          clip_range=(1, 10), verbose=True)
mf.fit(train_df)
mf.plot_training()

pred = mf.predict(user=1, item=42)
rmse = mf.rmse(val_df)
```

---

## PMFRegressor

**File:** `model/PMF.py`

Uses the latent embeddings from a trained `MatrixFactorization` model as features for a downstream sklearn regressor.

```
features(u, i) = concat(P[u], Q[i])    ← 2 * n_factors dimensions
pred = regressor.predict(features)
```

Falls back to the underlying PMF for unknown users/items.

### Constructor

```python
PMFRegressor(
    pmf: MatrixFactorization,
    model: str | sklearn_regressor,  # "randomforest" | "svr" | "ridge" | "lasso" | "elasticnet"
    name=None,
    clip_range=None,
    n_jobs=16,
)
```

| `model` string | Sklearn class |
|----------------|--------------|
| `"randomforest"` | `RandomForestRegressor` |
| `"svr"` | `SVR` |
| `"ridge"` | `Ridge` |
| `"lasso"` | `Lasso` |
| `"elasticnet"` | `ElasticNet` |

### Example

```python
from model.PMF import MatrixFactorization, PMFRegressor

pmf = MatrixFactorization(n_factors=20, n_epochs=20)
pmf.fit(train_df)

regressor = PMFRegressor(pmf=pmf, model="randomforest", clip_range=(1, 10))
regressor.fit(train_df)
pred = regressor.predict(user=1, item=42)
```

---

## SurpriseNMFModel

**File:** `model/PMF.py`

Wraps Surprise's `NMF` (Non-negative Matrix Factorisation). All embeddings are constrained to be non-negative, making them naturally interpretable as topic strengths.

```
# Unbiased
pred(u, i) = P[u] · Q[i]

# Biased
pred(u, i) = global_mean + user_bias[u] + item_bias[i] + P[u] · Q[i]
```

### Constructor

```python
SurpriseNMFModel(
    n_factors=15,
    n_epochs=50,
    biased=True,
    reg_pu=0.06,
    reg_qi=0.06,
    reg_bu=0.02,
    reg_bi=0.02,
    lr_bu=0.005,
    lr_bi=0.005,
    init_low=0.0,
    init_high=1.0,
    random_state=42,
    verbose=True,
    rating_scale=(1, 10),
    clip_range=None,
    name=None,
)
```

### Extra Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict_batch(df)` | `np.ndarray` | Batch prediction |
| `recommend(user, top_k=10, exclude_seen=True)` | `list[(item, score)]` | Top-k recommendations |
| `get_user_embedding(user)` | `np.ndarray \| None` | User latent factor vector |
| `get_item_embedding(item)` | `np.ndarray \| None` | Item latent factor vector |
| `get_user_bias(user)` | `float` | User bias (0 if unbiased or unknown) |
| `get_item_bias(item)` | `float` | Item bias (0 if unbiased or unknown) |
| `explain_prediction(user, item)` | `dict` | Decompose prediction into components |

---

## BayesianPMF

**File:** `model/BPMF.py`

Bayesian Probabilistic Matrix Factorisation with Gibbs sampling. Places Normal-Wishart priors on user and item factor matrices.

### Generative Model

```
Λ_u ~ Wishart(W_0, ν_0)
μ_u ~ Normal(μ_0, (β_0 Λ_u)^{-1})
u_k ~ Normal(μ_u, Λ_u^{-1})    for each user k

(similarly for items)

r_{ui} ~ Normal(u_u · v_i, σ_r^2)
```

The posterior is sampled via Gibbs: hyperparameters → user factors → item factors.

### Constructor

```python
BayesianPMF(
    n_factors=20,
    n_iters=100,
    burn_in=50,
    thin=2,
    rating_std=1.0,
    clip_range=None,
    random_state=42,
    verbose=True,
    name=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_factors` | `int` | `20` | Dimension of latent factors |
| `n_iters` | `int` | `100` | Total Gibbs sampling iterations |
| `burn_in` | `int` | `50` | Iterations discarded before collecting samples |
| `thin` | `int` | `2` | Keep every `thin`-th sample |
| `rating_std` | `float` | `1.0` | Observation noise standard deviation |

### Extra Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_user_embedding(user)` | `np.ndarray \| None` | Posterior mean user factor |
| `get_item_embedding(item)` | `np.ndarray \| None` | Posterior mean item factor |
| `recommend(user, top_k=10, exclude_seen=True)` | `list[(item, score)]` | Top-k recommendations |

### Example

```python
from model.BPMF import BayesianPMF

bpmf = BayesianPMF(n_factors=20, n_iters=100, burn_in=50, clip_range=(1, 10))
bpmf.fit(train_df)
pred = bpmf.predict(user=1, item=42)
recs = bpmf.recommend(user=1, top_k=10)
```

---

## BayesianNonNegativeMF

**File:** `model/NNBPMF.py`

Variational Bayesian Non-Negative Matrix Factorisation based on the Hernando et al. model. Uses a Binomial likelihood with Dirichlet and Beta priors.

### Generative Model

```
φ_u   ~ Dirichlet(α, ..., α)          # user preference over K topics
κ_{i,k} ~ Beta(β, β)                   # item like-probability per topic
z_{ui} ~ Categorical(φ_u)             # topic assignment
ρ_{ui} ~ Binomial(R, κ_{i, z_{ui}})   # implicit rating
r*_{ui} = ρ_{ui} / R                  # normalised to [0, R]
```

Parameters are inferred via variational EM (coordinate-ascent updates).

### Constructor

```python
BayesianNonNegativeMF(
    n_factors=6,
    n_iters=100,
    alpha=0.3,         # Dirichlet concentration (must be in (0,1))
    beta=1.0,          # Beta prior parameter
    R=10,              # Rating scale upper bound
    init_noise=0.05,
    eps=1e-12,
    clip_range=(0.0, 10.0),
    random_state=42,
    verbose=True,
    store_history=False,
    name=None,
)
```

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| `alpha` | `(0, 1)` | Dirichlet prior concentration |
| `beta` | `> 0` | Beta prior parameter |
| `R` | positive integer | Rating scale (e.g. 10 for 1–10 scale) |
| `n_factors` | `> 0` | Number of latent topics K |

### Extra Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict_expected_rating(user, item)` | `float` | Posterior expectation |
| `predict_normalized(user, item)` | `float` | p_ui = Σ_k a_uk · b_ik |
| `predict_proba_each_rating(user, item)` | `np.ndarray` | Full Binomial PMF |
| `explain_prediction(user, item, top_k=3)` | `dict` | Top contributing factors |
| `analyze_case(user, item)` | `dict` | Comprehensive interpretability report |
| `get_user_posterior_summary(user)` | `dict` | Dirichlet posterior for user |
| `get_item_posterior_summary(item)` | `dict` | Beta posterior for item |

### Example

```python
from model.NNBPMF import BayesianNonNegativeMF

nnbpmf = BayesianNonNegativeMF(n_factors=6, n_iters=100, alpha=0.3,
                                clip_range=(1, 10))
nnbpmf.fit(train_df)

# Full probability distribution over ratings 0..R
proba = nnbpmf.predict_proba_each_rating(user=1, item=42)

# Interpretability
explanation = nnbpmf.explain_prediction(user=1, item=42)
```

---

## Cold-Start Handling Summary

| Model | Unknown User | Unknown Item | Both Unknown |
|-------|-------------|-------------|-------------|
| `MeanBaseline` | global mean | global mean | global mean |
| `SurpriseBaselineOnlyModel` | 0 user bias | 0 item bias | global mean |
| `SurpriseKNNBaselineWrapper` | configurable via `unknown_strategy` | configurable | configurable |
| `MatrixFactorization` | `global_mean + item_bias` | `global_mean + user_bias` | `global_mean` |
| `PMFRegressor` | delegates to PMF | delegates to PMF | delegates to PMF |
| `BayesianPMF` | prior-based | prior-based | prior-based |
| `BayesianNonNegativeMF` | prior `a_prior_` | prior `b_prior_` | global_prob |
| `AdaptiveColdStartEnsemble` | weighted baseline blend | weighted baseline blend | `full_cold_start_penalty` |
