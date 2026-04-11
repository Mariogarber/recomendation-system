# Utilities Reference

> Documento legacy. La referencia actual vive en:
> - [docs/reference/collaborative-filtering-utils.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/reference/collaborative-filtering-utils.md)

All utility classes and functions live under `colaborative-filtering/utils/`.

---

## Table of Contents

- [FrequencyBandSplitter](#frequencybandsplitter)
- [FrequencyBandEvaluator](#frequencybandevaluator)
- [Predictor](#predictor)
- [ThresholdItemPredictor](#thresholditempredictor)
- [DatasetProfiler](#datasetprofiler)
- [analyze_biases_with_counts](#analyze_biases_with_counts)

---

## FrequencyBandSplitter

**File:** `utils/split.py`

Splits a DataFrame of interactions into three subsets — `"low"`, `"mid"`, and `"high"` — based on how frequently users or items appear in training data. This is useful for evaluating model performance per frequency band and diagnosing cold-start behaviour.

### Constructor

```python
FrequencyBandSplitter(
    entity="item",                  # "user" | "item"
    mode="quantile",                # "quantile" | "threshold"
    low_threshold=None,             # only for mode="threshold"
    high_threshold=None,            # only for mode="threshold"
    include_unknown_as="low",       # "low" | "mid" | "high" | None
    user_col="user",
    item_col="item",
    rating_col="rating",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity` | `str` | `"item"` | Which entity to measure frequency for |
| `mode` | `str` | `"quantile"` | Split by quantile (auto) or manual thresholds |
| `low_threshold` | `int \| None` | `None` | Frequency ≤ this value → `"low"` band |
| `high_threshold` | `int \| None` | `None` | Frequency ≥ this value → `"high"` band |
| `include_unknown_as` | `str \| None` | `"low"` | Band assignment for unseen entities; `None` excludes them |

### Methods

```python
def fit(self, df: pd.DataFrame) -> self
```
Learn frequency counts from the training DataFrame.

```python
def transform(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]
```
Apply learned band boundaries to a (test/validation) DataFrame. Returns `{"low": df_low, "mid": df_mid, "high": df_high}`.

```python
def fit_transform(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]
```
Convenience method: `fit` + `transform` in one call.

```python
def summary(self) -> list[BandSummary]
```
Return statistics per band: number of rows, unique entities, min/max/mean count.

```python
def get_entity_frequencies(self) -> pd.Series
```
Return the raw frequency counts learned during `fit`.

### Example

```python
from utils.split import FrequencyBandSplitter

splitter = FrequencyBandSplitter(entity="item", mode="quantile")
splitter.fit(train_df)
splits = splitter.transform(val_df)

df_low  = splits["low"]   # items with few interactions
df_mid  = splits["mid"]
df_high = splits["high"]  # popular items

# Manual thresholds
splitter2 = FrequencyBandSplitter(
    entity="item",
    mode="threshold",
    low_threshold=5,
    high_threshold=50,
)
splits2 = splitter2.fit_transform(train_df)
```

---

## FrequencyBandEvaluator

**File:** `utils/split.py`

Evaluates a model (or any callable) on each frequency band produced by a `FrequencyBandSplitter`. Supports standard RMSE/MAE and custom metrics.

### Example

```python
from utils.split import FrequencyBandSplitter, FrequencyBandEvaluator

splitter = FrequencyBandSplitter(entity="item", mode="quantile")
splitter.fit(train_df)

evaluator = FrequencyBandEvaluator(splitter=splitter)
results = evaluator.evaluate(model=my_model, df=val_df)

# results is a dict: {"low": {"rmse": ..., "mae": ...}, "mid": {...}, "high": {...}}
for band, metrics in results.items():
    print(f"{band}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
```

---

## Predictor

**File:** `utils/predict.py`

Loads the test CSV, runs a trained model over all user-item pairs, and saves the predictions as a submission CSV.

### Constructor

```python
Predictor(
    test_path: str,
    save_path: str,
    round_predictions: bool = False,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `test_path` | `str` | Path to test CSV (columns: `ID`, `user`, `item`) |
| `save_path` | `str` | Output path for predictions CSV (columns: `ID`, `rating`) |
| `round_predictions` | `bool` | If `True`, round predictions to nearest integer |

### Methods

```python
def load_test_from_csv(self) -> tuple[pd.DataFrame, np.ndarray]
```
Load test file. Returns `(test_df, ids_array)`.

```python
def compute_predictions(self, model: BaseModel, test_df: pd.DataFrame) -> np.ndarray
```
Generate predictions for all rows in `test_df`. Falls back to `7.0` on any exception.

```python
def predict(self, model: BaseModel) -> pd.DataFrame
```
End-to-end: load, predict, save, and return the solution DataFrame.

### Attributes

| Attribute | Description |
|-----------|-------------|
| `errors` | Count of rows where prediction failed (exception raised) |

### Example

```python
from utils.predict import Predictor

predictor = Predictor(
    test_path="data/test.csv",
    save_path="submissions/mf_model.csv",
    round_predictions=False,
)
solution = predictor.predict(model)
print(f"Errors: {predictor.errors}")
```

---

## ThresholdItemPredictor

**File:** `utils/predict.py`

Extends `Predictor` with a hybrid model selection strategy: chooses between a `rare_model` and a `frequent_model` based on item frequency in the training set.

```
if count(item) < threshold  → rare_model.predict(user, item)
if count(item) >= threshold → frequent_model.predict(user, item)
```

### Constructor

```python
ThresholdItemPredictor(
    test_path: str,
    save_path: str,
    rare_model: BaseModel,
    frequent_model: BaseModel,
    threshold: int,
    train_df: pd.DataFrame | None = None,
    item_counts: dict | pd.Series | None = None,
    round_predictions: bool = False,
    unknown_item_policy: str = "rare",   # "rare" | "frequent" | float
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `rare_model` | `BaseModel` | Model used for items with few interactions |
| `frequent_model` | `BaseModel` | Model used for popular items |
| `threshold` | `int` | Item count cutoff |
| `train_df` | `pd.DataFrame \| None` | Used to compute item counts if `item_counts` not provided |
| `item_counts` | `dict \| pd.Series \| None` | Pre-computed item frequency counts |
| `unknown_item_policy` | `str \| float` | Policy for items not seen in training |

### Example

```python
from utils.predict import ThresholdItemPredictor

predictor = ThresholdItemPredictor(
    test_path="data/test.csv",
    save_path="submissions/hybrid.csv",
    rare_model=baseline_model,
    frequent_model=mf_model,
    threshold=10,
    train_df=train_df,
    unknown_item_policy="rare",
)
predictor.predict(None)  # model argument is ignored; models are set internally
```

---

## DatasetProfiler

**File:** `utils/analysis.py`

Comprehensive profiling tool for recommendation datasets. Analyses user and item activity distributions, computes concentration metrics, and generates diagnostic reports.

### Constructor

```python
DatasetProfiler(
    df: pd.DataFrame,
    user_col: str = "user",
    item_col: str = "item",
    rating_col: str = "rating",
)
```

### Report Sections

The profiler computes the following when `build_report()` is called:

| Section | Metrics |
|---------|--------|
| **Dataset summary** | Total interactions, unique users, unique items, density, sparsity |
| **User profile** | Min/max/mean/median interactions per user, percentiles (p25, p75, p90, p95, p99), Gini coefficient |
| **Item profile** | Same statistics for items |
| **Head concentration** | Fraction of interactions accounted for by top 1%, 5%, 10% of items |
| **Cold-start diagnostics** | Estimated percentage of users/items with fewer than 5, 10, 20 interactions |

### Methods

```python
def build_report(self) -> dict
```
Compute and return the full profiling report as a dictionary.

```python
def print_report(self) -> None
```
Print a formatted summary to stdout.

```python
def save_report(self, path: str) -> None
```
Export the report as a JSON file.

```python
def plot_support_curves(self) -> None
```
Plot cumulative coverage vs. minimum interaction threshold for users and items.

```python
def plot_count_distributions(self) -> None
```
Plot histograms of user and item interaction counts.

### Example

```python
from utils.analysis import DatasetProfiler
import pandas as pd

train = pd.read_csv("data/train.csv")
profiler = DatasetProfiler(train)

profiler.print_report()
profiler.plot_count_distributions()
profiler.plot_support_curves()
profiler.save_report("reports/dataset_profile.json")
```

### Sample Report Output

```
Dataset Summary
───────────────
Total interactions : 390,351
Unique users       : 6,040
Unique items       : 3,706
Density            : 1.75%
Sparsity           : 98.25%

User Profile
────────────
Mean interactions  : 64.6
Median             : 44
p90                : 155
Gini coefficient   : 0.47

Item Profile
────────────
Mean interactions  : 105.3
Median             : 68
Head 1%            : 11.2% of all interactions
Head 10%           : 54.8% of all interactions
```

---

## analyze_biases_with_counts

**File:** `utils/analysis.py`

A standalone function that correlates a model's learned biases with entity interaction frequency, helping diagnose systematic over/under-prediction for rare or popular entities.

### Signature

```python
def analyze_biases_with_counts(
    model,
    train_df: pd.DataFrame,
    entity: str = "item",      # "user" | "item"
    top_n: int = 20,
) -> pd.DataFrame
```

### Returns

A DataFrame with one row per entity, containing:

| Column | Description |
|--------|-------------|
| `entity_id` | User or item identifier |
| `count` | Number of interactions in training data |
| `bias` | Learned model bias for this entity |
| `abs_bias` | Absolute value of bias |

Sorted by `abs_bias` descending (most extreme biases first).

### Example

```python
from utils.analysis import analyze_biases_with_counts
from model.baseline import SurpriseBaselineOnlyModel

model = SurpriseBaselineOnlyModel(rating_scale=(1, 10))
model.fit(train_df)

bias_df = analyze_biases_with_counts(model, train_df, entity="item", top_n=20)
print(bias_df.head())
# Shows items with the largest biases and their interaction counts
# Helps identify rare items that are consistently over/under-predicted
```
