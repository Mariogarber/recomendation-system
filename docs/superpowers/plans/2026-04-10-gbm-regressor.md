# LightGBM Regressor over Frozen Embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a LightGBM regressor over frozen deep embeddings + scalar priors, with implicit cold-start handling, producing a competition submission that beats the current 1.0496 leaderboard MAE.

**Architecture:** A feature builder (`gbm_features.py`) assembles a ~410-feature matrix per (user, business) pair from three groups: embedding interactions (128 × 3 + 6 scalars), scalar priors from `train_reviews`, and review context. LightGBM with MAE objective trains on this matrix. New users get zero user-embeddings and `is_new_user=1` — no rows are filtered. Three scripts mirror the existing frozen-regressor pipeline: validate, retrain on full data, predict submission.

**Tech Stack:** Python 3.12, LightGBM ≥ 4.0, NumPy, Pandas, scikit-learn, joblib, pytest

---

## Implementation Status

- Implemented validation scripts:
  - `content-based/train_gbm_regressor.py`
  - `content-based/train_gbm_submission_model.py`
  - `content-based/predict_gbm_submission.py`
  - `content-based/blend_deep_gbm_submission.py`
- Implemented validation winner:
  - `content-based/artifacts/gbm_regressor_v1_cs30`
- Implemented full-train GBM artifact:
  - `content-based/artifacts/gbm_submission_v1`
- Implemented final hybrid submission:
  - `content-based/artifacts/blended_submission_v1/submission.csv`

Actual blend rule:

```text
if known_train_user:
    prediction = round_half_up((deep_star + gbm_star) / 2)
else:
    prediction = gbm_star
```

Actual selected validation result:

- `val_mae_gbm_raw = 1.0054`
- `blend_validation.blend_mae = 0.9457`
- `synthetic_cold_start_fraction = 0.3`

This original plan is now completed in practice and superseded by the implemented hybrid submission path.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `lightgbm>=4.0.0` dependency |
| `content-based/conftest.py` | Create | pytest sys.path setup |
| `content-based/tests/__init__.py` | Create | Makes tests a package |
| `content-based/tests/test_gbm_features.py` | Create | Unit tests for `gbm_features` |
| `content-based/utils/gbm_features.py` | Create | `ScalarPriors`, `compute_scalar_priors`, `fit_review_context_scaler`, `build_gbm_feature_matrix` |
| `content-based/train_gbm_regressor.py` | Create | Train + validate LightGBM, save artifacts |
| `content-based/train_gbm_submission_model.py` | Create | Retrain on 100% of train data, fixed n_estimators |
| `content-based/predict_gbm_submission.py` | Create | Load model, predict test set, write submission CSV |

---

## Task 1: Add lightgbm and install

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add lightgbm to dependencies**

Edit `pyproject.toml` so the `dependencies` list reads:

```toml
dependencies = [
    "ipykernel>=7.2.0",
    "lightgbm>=4.0.0",
    "pandas>=3.0.2",
    "scikit-learn>=1.8.0",
    "torch>=2.5.1",
]
```

- [ ] **Step 2: Install**

```bash
uv add lightgbm
```

Expected: `uv.lock` updated, no errors.

- [ ] **Step 3: Verify**

```bash
python -c "import lightgbm; print(lightgbm.__version__)"
```

Expected: version string like `4.x.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add lightgbm dependency"
```

---

## Task 2: Set up pytest infrastructure

**Files:**
- Create: `content-based/conftest.py`
- Create: `content-based/tests/__init__.py`

- [ ] **Step 1: Create conftest.py in content-based/**

```python
# content-based/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

- [ ] **Step 2: Create tests package**

Create `content-based/tests/__init__.py` as an empty file.

- [ ] **Step 3: Verify pytest runs**

```bash
cd content-based && python -m pytest tests/ -v
```

Expected: `no tests ran` with exit code 0 or 5.

- [ ] **Step 4: Commit**

```bash
git add content-based/conftest.py content-based/tests/__init__.py
git commit -m "test: add pytest infrastructure for content-based"
```

---

## Task 3: `compute_scalar_priors` — TDD

**Files:**
- Create: `content-based/utils/gbm_features.py` (initial version)
- Create: `content-based/tests/test_gbm_features.py`

- [ ] **Step 1: Write failing tests**

Create `content-based/tests/test_gbm_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from utils.gbm_features import ScalarPriors, compute_scalar_priors


def _make_train_reviews() -> pd.DataFrame:
    """Minimal train reviews in raw format (user_id, business_id, stars)."""
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u3"],
        "business_id": ["b1", "b2", "b1", "b3"],
        "stars": [4.0, 2.0, 5.0, 3.0],
    })


def test_global_mean():
    priors = compute_scalar_priors(_make_train_reviews())
    assert abs(priors.global_mean - 3.5) < 1e-5


def test_user_mean():
    priors = compute_scalar_priors(_make_train_reviews())
    # u1: rated b1=4 and b2=2 → mean 3.0
    assert abs(priors.user_mean["u1"] - 3.0) < 1e-5
    # u2: rated b1=5 → mean 5.0
    assert abs(priors.user_mean["u2"] - 5.0) < 1e-5


def test_business_mean():
    priors = compute_scalar_priors(_make_train_reviews())
    # b1: received 4 and 5 → mean 4.5
    assert abs(priors.business_mean["b1"] - 4.5) < 1e-5


def test_user_count():
    priors = compute_scalar_priors(_make_train_reviews())
    assert priors.user_count["u1"] == 2
    assert priors.user_count["u2"] == 1


def test_single_review_std_is_zero():
    priors = compute_scalar_priors(_make_train_reviews())
    assert priors.user_std["u2"] == 0.0


def test_accepts_canonical_format():
    """compute_scalar_priors must also accept user/item/rating columns."""
    canonical = pd.DataFrame({
        "user": ["u1", "u2"],
        "item": ["b1", "b1"],
        "rating": [4.0, 2.0],
    })
    priors = compute_scalar_priors(canonical)
    assert abs(priors.global_mean - 3.0) < 1e-5
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd content-based && python -m pytest tests/test_gbm_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'utils.gbm_features'`.

- [ ] **Step 3: Create `content-based/utils/gbm_features.py` with ScalarPriors and compute_scalar_priors**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ScalarPriors:
    global_mean: float
    user_mean: dict[str, float]
    user_std: dict[str, float]
    user_count: dict[str, int]
    business_mean: dict[str, float]
    business_std: dict[str, float]
    business_count: dict[str, int]


def compute_scalar_priors(train_reviews: pd.DataFrame) -> ScalarPriors:
    """
    Compute user and business prior statistics from a reviews DataFrame.

    Accepts either raw format (user_id, business_id, stars) or canonical
    format (user, item, rating). All values are derived from the provided
    DataFrame only — no external metadata.
    """
    df = train_reviews
    user_col = "user_id" if "user_id" in df.columns else "user"
    item_col = "business_id" if "business_id" in df.columns else "item"
    rating_col = "stars" if "stars" in df.columns else "rating"

    global_mean = float(df[rating_col].mean())

    user_stats = df.groupby(user_col)[rating_col].agg(["mean", "std", "count"])
    business_stats = df.groupby(item_col)[rating_col].agg(["mean", "std", "count"])
    user_stats["std"] = user_stats["std"].fillna(0.0)
    business_stats["std"] = business_stats["std"].fillna(0.0)

    return ScalarPriors(
        global_mean=global_mean,
        user_mean=user_stats["mean"].to_dict(),
        user_std=user_stats["std"].to_dict(),
        user_count=user_stats["count"].astype(int).to_dict(),
        business_mean=business_stats["mean"].to_dict(),
        business_std=business_stats["std"].to_dict(),
        business_count=business_stats["count"].astype(int).to_dict(),
    )
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd content-based && python -m pytest tests/test_gbm_features.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add content-based/utils/gbm_features.py content-based/tests/test_gbm_features.py
git commit -m "feat: add ScalarPriors and compute_scalar_priors"
```

---

## Task 4: `build_gbm_feature_matrix` and `fit_review_context_scaler` — TDD

**Files:**
- Modify: `content-based/utils/gbm_features.py`
- Modify: `content-based/tests/test_gbm_features.py`

- [ ] **Step 1: Append failing tests to test_gbm_features.py**

Add the following to the end of `content-based/tests/test_gbm_features.py`:

```python
from pathlib import Path
from utils.frozen_embedding_regression import FrozenEmbeddingBundle
from utils.gbm_features import build_gbm_feature_matrix, fit_review_context_scaler


def _make_bundle() -> FrozenEmbeddingBundle:
    user_ids = pd.Series(["u1", "u2"])
    business_ids = pd.Series(["b1", "b2"])
    user_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    business_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    user_table = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "history_count_train": [2.0, 1.0],
        "history_band": ["2-5", "1"],
    })
    return FrozenEmbeddingBundle(
        root=Path("."),
        user_ids=user_ids,
        business_ids=business_ids,
        user_embeddings=user_embeddings,
        business_embeddings=business_embeddings,
        user_table=user_table,
        summary={},
    )


def _make_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "user": ["u1", "u2", "u_new"],
        "item": ["b1", "b2", "b1"],
        "timestamp": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-09-01"]),
        "useful": [0.0, 1.0, 0.0],
        "funny": [0.0, 0.0, 0.0],
        "cool": [0.0, 0.0, 0.0],
    })


def _build_matrix():
    bundle = _make_bundle()
    priors = compute_scalar_priors(_make_train_reviews())
    frame = _make_frame()
    min_ts, means, stds = fit_review_context_scaler(frame)
    return build_gbm_feature_matrix(
        frame, bundle, priors,
        review_context_min_timestamp=min_ts,
        review_context_means=means,
        review_context_stds=stds,
    )


def test_matrix_keeps_all_rows():
    matrix, names = _build_matrix()
    assert matrix.shape[0] == 3  # new user is NOT filtered out


def test_matrix_columns_match_feature_names():
    matrix, names = _build_matrix()
    assert matrix.shape[1] == len(names)


def test_new_user_embedding_is_zero():
    matrix, names = _build_matrix()
    user_emb_cols = [i for i, n in enumerate(names) if n.startswith("user_emb_")]
    # row 2 is the new user
    assert np.all(matrix[2, user_emb_cols] == 0.0)


def test_new_user_flag_is_set():
    matrix, names = _build_matrix()
    is_new_col = names.index("is_new_user")
    assert matrix[2, is_new_col] == 1.0
    assert matrix[0, is_new_col] == 0.0


def test_known_user_embedding_is_nonzero():
    matrix, names = _build_matrix()
    user_emb_cols = [i for i, n in enumerate(names) if n.startswith("user_emb_")]
    # row 0 is u1 with embedding [1, 0]
    assert matrix[0, user_emb_cols[0]] == pytest.approx(1.0)


def test_new_user_prior_falls_back_to_global_mean():
    bundle = _make_bundle()
    priors = compute_scalar_priors(_make_train_reviews())
    frame = _make_frame()
    min_ts, means, stds = fit_review_context_scaler(frame)
    matrix, names = build_gbm_feature_matrix(
        frame, bundle, priors,
        review_context_min_timestamp=min_ts,
        review_context_means=means,
        review_context_stds=stds,
    )
    user_mean_col = names.index("user_mean_rating")
    # u_new is not in priors, so should fall back to global_mean (3.5)
    assert matrix[2, user_mean_col] == pytest.approx(priors.global_mean)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd content-based && python -m pytest tests/test_gbm_features.py::test_matrix_keeps_all_rows -v
```

Expected: `ImportError` — `build_gbm_feature_matrix` not defined yet.

- [ ] **Step 3: Append the remaining functions to `content-based/utils/gbm_features.py`**

```python
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frozen_embedding_regression import FrozenEmbeddingBundle


REVIEW_CONTEXT_FEATURE_NAMES = [
    "useful_log1p",
    "funny_log1p",
    "cool_log1p",
    "recency_days",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "hour_sin",
    "hour_cos",
]


def _build_review_context_matrix(frame: pd.DataFrame, min_timestamp: pd.Timestamp) -> np.ndarray:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    useful = np.log1p(frame["useful"].to_numpy(dtype=np.float32))
    funny = np.log1p(frame["funny"].to_numpy(dtype=np.float32))
    cool = np.log1p(frame["cool"].to_numpy(dtype=np.float32))
    recency = ((ts - min_timestamp).dt.total_seconds() / 86400.0).to_numpy(dtype=np.float32)
    month_angle = 2.0 * np.pi * (ts.dt.month.fillna(1).to_numpy(dtype=np.float32) - 1.0) / 12.0
    weekday_angle = 2.0 * np.pi * ts.dt.weekday.fillna(0).to_numpy(dtype=np.float32) / 7.0
    hour_angle = 2.0 * np.pi * ts.dt.hour.fillna(0).to_numpy(dtype=np.float32) / 24.0
    return np.column_stack([
        useful, funny, cool, recency,
        np.sin(month_angle), np.cos(month_angle),
        np.sin(weekday_angle), np.cos(weekday_angle),
        np.sin(hour_angle), np.cos(hour_angle),
    ]).astype(np.float32)


def fit_review_context_scaler(
    train_frame: pd.DataFrame,
) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """
    Fit a z-score scaler for review context features on the training frame.

    Returns (min_timestamp, means, stds). Apply the same values to eval and
    test frames using _build_review_context_matrix to avoid leakage.
    """
    min_timestamp = pd.to_datetime(train_frame["timestamp"], errors="coerce").min()
    raw = _build_review_context_matrix(train_frame, min_timestamp)
    means = raw.mean(axis=0).astype(np.float32)
    stds = raw.std(axis=0).astype(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    return min_timestamp, means, stds


def build_gbm_feature_matrix(
    frame: pd.DataFrame,
    bundle: "FrozenEmbeddingBundle",
    priors: ScalarPriors,
    *,
    review_context_min_timestamp: pd.Timestamp,
    review_context_means: np.ndarray,
    review_context_stds: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the GBM feature matrix for a frame with columns:
    user, item, timestamp, useful, funny, cool.

    All rows are kept. New users (not in bundle) get zero user embeddings
    and is_new_user=1. New businesses get zero embeddings and priors default
    to global mean.

    Returns (matrix of shape [n_rows, n_features], list of feature names).
    """
    n = len(frame)
    embedding_dim = bundle.user_embeddings.shape[1]

    user_to_idx = {uid: i for i, uid in enumerate(bundle.user_ids.to_numpy())}
    biz_to_idx = {bid: i for i, bid in enumerate(bundle.business_ids.to_numpy())}

    user_indices = np.array([user_to_idx.get(u, -1) for u in frame["user"]], dtype=np.int64)
    biz_indices = np.array([biz_to_idx.get(b, -1) for b in frame["item"]], dtype=np.int64)

    is_new_user = (user_indices == -1).astype(np.float32)

    user_emb = np.zeros((n, embedding_dim), dtype=np.float32)
    known_user = user_indices >= 0
    if known_user.any():
        user_emb[known_user] = bundle.user_embeddings[user_indices[known_user]]

    biz_emb = np.zeros((n, embedding_dim), dtype=np.float32)
    known_biz = biz_indices >= 0
    if known_biz.any():
        biz_emb[known_biz] = bundle.business_embeddings[biz_indices[known_biz]]

    product = user_emb * biz_emb
    dot = np.einsum("ij,ij->i", user_emb, biz_emb).astype(np.float32)
    user_norm = np.linalg.norm(user_emb, axis=1).astype(np.float32)
    biz_norm = np.linalg.norm(biz_emb, axis=1).astype(np.float32)
    cosine = (dot / np.maximum(user_norm * biz_norm, 1e-8)).astype(np.float32)
    norm_gap = np.abs(user_norm - biz_norm)

    history_lookup: dict[str, float] = {}
    if "user_id" in bundle.user_table.columns and "history_count_train" in bundle.user_table.columns:
        history_lookup = dict(zip(
            bundle.user_table["user_id"].to_numpy(),
            bundle.user_table["history_count_train"].astype(float).to_numpy(),
        ))
    history_count = np.array([history_lookup.get(u, 0.0) for u in frame["user"]], dtype=np.float32)
    history_log1p = np.log1p(history_count)

    users = frame["user"].to_numpy()
    items = frame["item"].to_numpy()
    user_mean = np.array([priors.user_mean.get(u, priors.global_mean) for u in users], dtype=np.float32)
    user_std = np.array([priors.user_std.get(u, 0.0) for u in users], dtype=np.float32)
    biz_mean = np.array([priors.business_mean.get(b, priors.global_mean) for b in items], dtype=np.float32)
    biz_std = np.array([priors.business_std.get(b, 0.0) for b in items], dtype=np.float32)
    biz_count = np.array([priors.business_count.get(b, 0) for b in items], dtype=np.float32)
    global_mean_arr = np.full(n, priors.global_mean, dtype=np.float32)
    mean_rating_gap = user_mean - biz_mean

    rc_raw = _build_review_context_matrix(frame, review_context_min_timestamp)
    rc_scaled = ((rc_raw - review_context_means) / review_context_stds).astype(np.float32)

    matrix = np.column_stack([
        user_emb,               # 128
        biz_emb,                # 128
        product,                # 128
        cosine[:, None],
        dot[:, None],
        user_norm[:, None],
        biz_norm[:, None],
        norm_gap[:, None],
        history_log1p[:, None],
        is_new_user[:, None],
        user_mean[:, None],
        user_std[:, None],
        history_count[:, None],
        biz_mean[:, None],
        biz_std[:, None],
        biz_count[:, None],
        global_mean_arr[:, None],
        mean_rating_gap[:, None],
        rc_scaled,              # 10
    ]).astype(np.float32)

    feature_names = (
        [f"user_emb_{i}" for i in range(embedding_dim)]
        + [f"biz_emb_{i}" for i in range(embedding_dim)]
        + [f"prod_emb_{i}" for i in range(embedding_dim)]
        + [
            "cosine", "dot", "user_norm", "biz_norm", "norm_gap", "history_log1p",
            "is_new_user", "user_mean_rating", "user_rating_std", "history_count",
            "biz_mean_rating", "biz_rating_std", "biz_review_count",
            "global_mean", "mean_rating_gap",
        ]
        + REVIEW_CONTEXT_FEATURE_NAMES
    )

    return matrix, feature_names
```

- [ ] **Step 4: Run all tests**

```bash
cd content-based && python -m pytest tests/test_gbm_features.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add content-based/utils/gbm_features.py content-based/tests/test_gbm_features.py
git commit -m "feat: add build_gbm_feature_matrix with cold-start handling"
```

---

## Task 5: `train_gbm_regressor.py`

**Files:**
- Create: `content-based/train_gbm_regressor.py`

- [ ] **Step 1: Create the training and validation script**

Create `content-based/train_gbm_regressor.py`:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.frozen_embedding_regression import (
    attach_embedding_indices,
    build_review_interaction_frame,
    compute_band_metrics,
    fit_ridge_embedding_baseline,
    load_frozen_embedding_bundle,
    rmse,
)
from utils.gbm_features import (
    ScalarPriors,
    build_gbm_feature_matrix,
    compute_scalar_priors,
    fit_review_context_scaler,
)
from utils.io import get_default_data_dir, load_train_reviews
from utils.split import temporal_train_validation_split


@dataclass(slots=True)
class GBMRegressorConfig:
    num_leaves: int = 127
    learning_rate: float = 0.05
    n_estimators: int = 1000
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    temporal_val_size: float = 0.2
    early_stopping_rounds: int = 50
    random_seed: int = 42


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _attach_history_band(frame: pd.DataFrame, bundle) -> pd.DataFrame:
    """Add history_band column from the bundle user table. New users get '0'."""
    out = frame.copy()
    if "history_band" not in out.columns:
        band_lookup = {}
        if "user_id" in bundle.user_table.columns and "history_band" in bundle.user_table.columns:
            band_lookup = dict(zip(
                bundle.user_table["user_id"].to_numpy(),
                bundle.user_table["history_band"].to_numpy(),
            ))
        out["history_band"] = out["user"].map(band_lookup).fillna("0")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM regressor over frozen embeddings.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "competition_embeddings_v3_iter03",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "gbm_regressor_v1",
    )
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--temporal-val-size", type=float, default=0.2)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GBMRegressorConfig(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        temporal_val_size=args.temporal_val_size,
        early_stopping_rounds=args.early_stopping_rounds,
        random_seed=args.seed,
    )
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    _save_json(save_root / "config.json", asdict(config))

    train_reviews = load_train_reviews(args.data_dir)
    interactions = build_review_interaction_frame(train_reviews)
    train_split, val_split = temporal_train_validation_split(
        interactions, val_size=config.temporal_val_size, timestamp_col="timestamp"
    )

    # Priors from train split only — no val leakage
    priors = compute_scalar_priors(train_split)
    bundle = load_frozen_embedding_bundle(args.embedding_root)

    min_ts, rc_means, rc_stds = fit_review_context_scaler(train_split)

    x_train, feature_names = build_gbm_feature_matrix(
        train_split, bundle, priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_train = train_split["rating"].to_numpy(dtype=np.float32)

    x_val, _ = build_gbm_feature_matrix(
        val_split, bundle, priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_val = val_split["rating"].to_numpy(dtype=np.float32)

    lgb_train = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
    lgb_val = lgb.Dataset(x_val, label=y_val, feature_name=feature_names, reference=lgb_train)

    params = {
        "objective": "regression_l1",
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "verbose": -1,
        "seed": config.random_seed,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.n_estimators,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    best_n = int(booster.best_iteration)
    val_pred = np.clip(
        booster.predict(x_val, num_iteration=best_n).astype(np.float32), 1.0, 5.0
    )
    val_mae = float(np.mean(np.abs(y_val - val_pred)))
    val_rmse = rmse(y_val, val_pred)

    val_eval = _attach_history_band(val_split, bundle)
    val_eval["pred"] = val_pred
    band_metrics = compute_band_metrics(val_eval[["rating", "history_band", "pred"]].copy())

    # Ridge baseline using only known-user rows (for fair comparison)
    train_indexed, _ = attach_embedding_indices(train_split, bundle)
    val_indexed, _ = attach_embedding_indices(val_split, bundle)
    ridge_summary, _ = fit_ridge_embedding_baseline(
        bundle=bundle, train_frame=train_indexed, val_frame=val_indexed,
    )

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    joblib.dump(booster, save_root / "model.pkl")
    _save_json(save_root / "review_context_scaler.json", {
        "min_timestamp": min_ts.isoformat(),
        "means": rc_means.tolist(),
        "stds": rc_stds.tolist(),
    })
    _save_json(save_root / "scalar_priors.json", {
        "global_mean": priors.global_mean,
        "user_mean": priors.user_mean,
        "user_std": priors.user_std,
        "user_count": priors.user_count,
        "business_mean": priors.business_mean,
        "business_std": priors.business_std,
        "business_count": priors.business_count,
    })
    band_metrics.to_csv(save_root / "band_metrics.csv", index=False)
    importance.to_csv(save_root / "feature_importance.csv", index=False)

    summary = {
        "objective": "rating_regression",
        "model_type": "lightgbm_l1",
        "embedding_root": str(args.embedding_root),
        "best_n_estimators": best_n,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "ridge_baseline_mae": ridge_summary["mae"],
        "n_train_rows": int(len(train_split)),
        "n_val_rows": int(len(val_split)),
        "n_features": len(feature_names),
        "band_metrics": band_metrics.to_dict(orient="records"),
        "config": asdict(config),
    }
    _save_json(save_root / "validation_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "band_metrics"}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with small n_estimators**

```bash
cd content-based && python train_gbm_regressor.py \
  --n-estimators 50 \
  --early-stopping-rounds 10 \
  --save-root artifacts/gbm_regressor_smoke
```

Expected: runs to completion, prints JSON summary with `val_mae` and `ridge_baseline_mae`.

- [ ] **Step 3: Verify artifacts**

```bash
ls artifacts/gbm_regressor_smoke/
```

Expected: `model.pkl`, `validation_summary.json`, `band_metrics.csv`, `feature_importance.csv`, `review_context_scaler.json`, `scalar_priors.json`, `config.json`.

- [ ] **Step 4: Commit**

```bash
git add content-based/train_gbm_regressor.py
git commit -m "feat: add train_gbm_regressor with LightGBM MAE objective"
```

---

## Task 6: `train_gbm_submission_model.py`

**Files:**
- Create: `content-based/train_gbm_submission_model.py`

- [ ] **Step 1: Create the full-train retraining script**

Create `content-based/train_gbm_submission_model.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.frozen_embedding_regression import (
    build_review_interaction_frame,
    load_frozen_embedding_bundle,
)
from utils.gbm_features import (
    ScalarPriors,
    build_gbm_feature_matrix,
    compute_scalar_priors,
    fit_review_context_scaler,
)
from utils.io import get_default_data_dir, load_train_reviews


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_source = Path(__file__).resolve().parent / "artifacts" / "gbm_regressor_v1"
    parser = argparse.ArgumentParser(
        description="Retrain GBM on all train data for competition submission."
    )
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--source-run", type=Path, default=default_source)
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "gbm_submission_v1",
    )
    parser.add_argument(
        "--fixed-n-estimators",
        type=int,
        default=None,
        help="Override n_estimators. Defaults to best_n_estimators from validation run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_summary = _load_json(args.source_run / "validation_summary.json")
    source_config = source_summary["config"]
    fixed_n = int(args.fixed_n_estimators or source_summary["best_n_estimators"])
    embedding_root = Path(source_summary["embedding_root"])

    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    bundle = load_frozen_embedding_bundle(embedding_root)
    train_reviews = load_train_reviews(args.data_dir)
    train_frame = build_review_interaction_frame(train_reviews)

    # Priors from 100% of train_reviews — correct for final submission
    priors = compute_scalar_priors(train_reviews)
    min_ts, rc_means, rc_stds = fit_review_context_scaler(train_frame)

    x_train, feature_names = build_gbm_feature_matrix(
        train_frame, bundle, priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_train = train_frame["rating"].to_numpy(dtype=np.float32)

    params = {
        "objective": "regression_l1",
        "num_leaves": int(source_config["num_leaves"]),
        "learning_rate": float(source_config["learning_rate"]),
        "min_child_samples": int(source_config["min_child_samples"]),
        "subsample": float(source_config["subsample"]),
        "colsample_bytree": float(source_config["colsample_bytree"]),
        "reg_alpha": float(source_config["reg_alpha"]),
        "reg_lambda": float(source_config["reg_lambda"]),
        "verbose": -1,
        "seed": int(source_config["random_seed"]),
    }
    lgb_train = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
    booster = lgb.train(params, lgb_train, num_boost_round=fixed_n)

    joblib.dump(booster, save_root / "model.pkl")
    _save_json(save_root / "review_context_scaler.json", {
        "min_timestamp": min_ts.isoformat(),
        "means": rc_means.tolist(),
        "stds": rc_stds.tolist(),
    })
    _save_json(save_root / "scalar_priors.json", {
        "global_mean": priors.global_mean,
        "user_mean": priors.user_mean,
        "user_std": priors.user_std,
        "user_count": priors.user_count,
        "business_mean": priors.business_mean,
        "business_std": priors.business_std,
        "business_count": priors.business_count,
    })
    _save_json(save_root / "embedding_root.json", {"embedding_root": str(embedding_root)})

    summary = {
        "objective": "rating_regression",
        "model_type": "lightgbm_l1",
        "training_mode": "full_train_for_competition_submission",
        "source_run": str(args.source_run),
        "embedding_root": str(embedding_root),
        "fixed_n_estimators": fixed_n,
        "n_train_rows": int(len(train_frame)),
        "n_features": len(feature_names),
    }
    _save_json(save_root / "train_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
cd content-based && python train_gbm_submission_model.py \
  --source-run artifacts/gbm_regressor_smoke \
  --fixed-n-estimators 10 \
  --save-root artifacts/gbm_submission_smoke
```

Expected: completes, prints JSON summary.

- [ ] **Step 3: Verify artifacts**

```bash
ls artifacts/gbm_submission_smoke/
```

Expected: `model.pkl`, `train_summary.json`, `review_context_scaler.json`, `scalar_priors.json`, `embedding_root.json`.

- [ ] **Step 4: Commit**

```bash
git add content-based/train_gbm_submission_model.py
git commit -m "feat: add train_gbm_submission_model for full-train retraining"
```

---

## Task 7: `predict_gbm_submission.py`

**Files:**
- Create: `content-based/predict_gbm_submission.py`

- [ ] **Step 1: Create the prediction script**

Create `content-based/predict_gbm_submission.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.frozen_embedding_regression import (
    build_review_context_only_frame,
    load_frozen_embedding_bundle,
)
from utils.gbm_features import (
    ScalarPriors,
    build_gbm_feature_matrix,
)
from utils.io import get_default_data_dir, load_test_reviews


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _priors_from_json(data: dict[str, Any]) -> ScalarPriors:
    return ScalarPriors(
        global_mean=float(data["global_mean"]),
        user_mean=data["user_mean"],
        user_std=data["user_std"],
        user_count={k: int(v) for k, v in data["user_count"].items()},
        business_mean=data["business_mean"],
        business_std=data["business_std"],
        business_count={k: int(v) for k, v in data["business_count"].items()},
    )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "artifacts" / "gbm_submission_v1"
    parser = argparse.ArgumentParser(
        description="Generate competition submission from trained GBM model."
    )
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--artifact-root", type=Path, default=default_root)
    parser.add_argument("--save-path", type=Path, default=default_root / "submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root

    booster = joblib.load(root / "model.pkl")
    priors = _priors_from_json(_load_json(root / "scalar_priors.json"))
    scaler = _load_json(root / "review_context_scaler.json")
    embedding_root = _load_json(root / "embedding_root.json")["embedding_root"]

    bundle = load_frozen_embedding_bundle(embedding_root)
    min_timestamp = pd.Timestamp(scaler["min_timestamp"])
    rc_means = np.array(scaler["means"], dtype=np.float32)
    rc_stds = np.array(scaler["stds"], dtype=np.float32)

    test_reviews = load_test_reviews(args.data_dir)
    test_frame = build_review_context_only_frame(test_reviews)

    x_test, _ = build_gbm_feature_matrix(
        test_frame, bundle, priors,
        review_context_min_timestamp=min_timestamp,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )

    raw_pred = np.clip(booster.predict(x_test).astype(np.float32), 1.0, 5.0)
    rounded_pred = np.rint(raw_pred).clip(1, 5).astype(np.int32)

    submission = pd.DataFrame({
        "ids": test_frame["review_id"].astype(str).to_numpy(),
        "prediction": rounded_pred,
    })
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.save_path, index=False)

    payload = {
        "artifact_root": str(root),
        "embedding_root": embedding_root,
        "save_path": str(args.save_path),
        "n_rows": int(len(submission)),
        "prediction_min": int(rounded_pred.min()),
        "prediction_max": int(rounded_pred.max()),
        "prediction_mean_raw": float(raw_pred.mean()),
        "prediction_mean_rounded": float(rounded_pred.mean()),
    }
    _save_json(root / "submission_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
cd content-based && python predict_gbm_submission.py \
  --artifact-root artifacts/gbm_submission_smoke \
  --save-path artifacts/gbm_submission_smoke/submission.csv
```

Expected: submission.csv created, JSON payload printed.

- [ ] **Step 3: Verify submission format**

```bash
cd content-based && python -c "
import pandas as pd
df = pd.read_csv('artifacts/gbm_submission_smoke/submission.csv')
print('shape:', df.shape)
print(df.head())
print('pred range:', df['prediction'].min(), '-', df['prediction'].max())
assert df['prediction'].between(1, 5).all()
assert df['ids'].notna().all()
print('OK')
"
```

Expected: correct shape, predictions in [1, 5], no missing IDs.

- [ ] **Step 4: Commit**

```bash
git add content-based/predict_gbm_submission.py
git commit -m "feat: add predict_gbm_submission for competition CSV generation"
```

---

## Task 8: Full production run and submission

- [ ] **Step 1: Train with full hyperparameters**

```bash
cd content-based && python train_gbm_regressor.py \
  --embedding-root artifacts/competition_embeddings_v3_iter03 \
  --save-root artifacts/gbm_regressor_v1 \
  --num-leaves 127 \
  --learning-rate 0.05 \
  --n-estimators 1000 \
  --min-child-samples 50 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.1 \
  --reg-lambda 1.0 \
  --early-stopping-rounds 50
```

Expected: runs with early stopping. Check `val_mae < 0.93` (must beat the deep frozen regressor).
Also check `val_mae < ridge_baseline_mae` (must beat the ridge baseline).

- [ ] **Step 2: Inspect band metrics**

```bash
cd content-based && python -c "
import pandas as pd
df = pd.read_csv('artifacts/gbm_regressor_v1/band_metrics.csv')
print(df.to_string())
"
```

Key check: history bands `0` and `1` should have lower MAE than in the deep regressor's band_metrics.
Those bands represent cold-start users — improvement there confirms the core hypothesis.

- [ ] **Step 3: Inspect top features**

```bash
cd content-based && python -c "
import pandas as pd
df = pd.read_csv('artifacts/gbm_regressor_v1/feature_importance.csv')
print(df.head(20).to_string())
"
```

Expected: `biz_mean_rating`, `user_mean_rating`, `mean_rating_gap`, `is_new_user` should rank high.
If embedding features dominate completely, consider re-running with lower colsample_bytree (0.6).

- [ ] **Step 4: Retrain on full train data**

```bash
cd content-based && python train_gbm_submission_model.py \
  --source-run artifacts/gbm_regressor_v1 \
  --save-root artifacts/gbm_submission_v1
```

- [ ] **Step 5: Generate and validate submission**

```bash
cd content-based && python predict_gbm_submission.py \
  --artifact-root artifacts/gbm_submission_v1 \
  --save-path artifacts/gbm_submission_v1/submission.csv
```

```bash
cd content-based && python -c "
import pandas as pd
df = pd.read_csv('artifacts/gbm_submission_v1/submission.csv')
assert df.shape[0] == len(open('../data/test_reviews.csv').readlines()) - 1, 'row count mismatch'
assert df['prediction'].between(1, 5).all(), 'predictions out of range'
assert df['ids'].notna().all(), 'missing ids'
print('shape:', df.shape)
print(df['prediction'].value_counts().sort_index())
print('OK — ready to submit')
"
```

- [ ] **Step 6: Commit artifacts**

```bash
git add content-based/artifacts/gbm_regressor_v1/validation_summary.json \
        content-based/artifacts/gbm_regressor_v1/band_metrics.csv \
        content-based/artifacts/gbm_regressor_v1/feature_importance.csv \
        content-based/artifacts/gbm_regressor_v1/config.json
git commit -m "feat: GBM regressor v1 validation results"
```
