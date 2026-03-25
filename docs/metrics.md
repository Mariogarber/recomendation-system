# Metrics Reference

All metric utilities live under `colaborative-filtering/metric/`.

---

## Table of Contents

- [NDCG](#ndcg)
- [WeightedUserSimilarity](#weightedusersimilarity)

---

## NDCG

**File:** `metric/NDCG.py`

Normalised Discounted Cumulative Gain (NDCG) is a ranking quality metric. It measures how well a ranked list orders items by relevance, discounting gains from lower-ranked positions logarithmically.

### Functions

```python
def dcg(relevance_scores: list[float]) -> float
```
Discounted Cumulative Gain for an ordered list of relevance scores.

```
DCG = Σ_{k=1}^{n} relevance[k] / log2(k + 1)
```

```python
def idcg(relevance_scores: list[float]) -> float
```
Ideal DCG — DCG of the scores sorted in descending order (the best possible ranking).

```python
def ndcg(relevance_scores: list[float]) -> float
```
Normalised DCG: `NDCG = DCG / IDCG`. Returns 0 if IDCG is 0.

### Example

```python
from metric.NDCG import ndcg

# Predicted ranking: ratings in the order the model ranked items
predicted_order = [5, 3, 4, 1, 2]
score = ndcg(predicted_order)
print(f"NDCG: {score:.4f}")
```

---

## WeightedUserSimilarity

**File:** `metric/similarity.py`

A custom user-user similarity measure that combines multiple components into a single robust similarity score. Designed for k-NN collaborative filtering, particularly in sparse or noisy datasets.

### Components

| Component | Description |
|-----------|-------------|
| **Pearson correlation** | Computed on centred ratings (subtracting user mean) over co-rated items |
| **Shrinkage regularisation** | Bayesian correction that down-weights similarities based on few co-rated items: `sim *= n / (n + shrinkage)` |
| **Significance weighting** | Caps confidence at `beta_significance` co-rated items: `min(n / β, 1)` |
| **Jaccard similarity** | Set overlap of rated items: `|A ∩ B| / |A ∪ B|` |

### Constructor

```python
WeightedUserSimilarity(
    alpha_jaccard=0.3,
    beta_significance=50,
    shrinkage=10,
    min_common=2,
    use_abs=True,
    jaccard_mode="mix",   # "mix" | "multiply" | "none"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha_jaccard` | `float` | `0.3` | Weight of Jaccard component in `"mix"` mode |
| `beta_significance` | `int` | `50` | Number of co-rated items for full significance weight |
| `shrinkage` | `int` | `10` | Bayesian shrinkage factor |
| `min_common` | `int` | `2` | Minimum co-rated items to compute similarity |
| `use_abs` | `bool` | `True` | Use absolute value of Pearson correlation |
| `jaccard_mode` | `str` | `"mix"` | How to combine Pearson with Jaccard |

### `jaccard_mode` options

| Mode | Formula |
|------|---------|
| `"mix"` | `(1 - α) * pearson + α * jaccard` |
| `"multiply"` | `pearson * jaccard` |
| `"none"` | `pearson` only |

### Methods

```python
def fit(self, df: pd.DataFrame) -> self
```
Fit on a DataFrame with columns `['user', 'item', 'rating']`. Builds the user-item pivot matrix and user mean cache.

```python
def compute_similarity_matrix(self) -> pd.DataFrame
```
Compute the full O(n²) user-user similarity matrix. Returns a square DataFrame indexed and columned by user IDs.

```python
def predict(self, user_id, item_id, k=20) -> float
```
Predict the rating for `(user_id, item_id)` using weighted k-NN:

```
pred(u, i) = mean_u + Σ_{v in N_k(u)} sim(u,v) * (r_{v,i} - mean_v) / Σ sim(u,v)
```

Returns the user mean as fallback if no neighbours have rated item `i`.

### Example

```python
from metric.similarity import WeightedUserSimilarity

sim = WeightedUserSimilarity(
    alpha_jaccard=0.3,
    beta_significance=50,
    shrinkage=10,
    min_common=2,
    jaccard_mode="mix",
)
sim.fit(train_df)

# Full similarity matrix (may be large)
sim_matrix = sim.compute_similarity_matrix()

# k-NN prediction
pred = sim.predict(user_id=1, item_id=42, k=20)
```

---

## Built-in Evaluation Metrics (BaseModel)

In addition to the metric utilities above, all models that inherit from `BaseModel` expose two evaluation methods directly:

```python
model.rmse(df)                   # Root Mean Squared Error
model.mae(df, round_predictions=False)   # Mean Absolute Error
```

Both methods ignore rows where the prediction is `NaN`.

### Example

```python
train_rmse = model.rmse(train_df)
val_rmse   = model.rmse(val_df)
val_mae    = model.mae(val_df)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Val   RMSE: {val_rmse:.4f}")
print(f"Val   MAE:  {val_mae:.4f}")
```
