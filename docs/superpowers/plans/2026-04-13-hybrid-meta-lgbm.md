# Hybrid Meta-LightGBM Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-layer hybrid stack (CF bias model + content-based deep router) under a LightGBM meta-learner to break through the 0.65 leaderboard ceiling and reach 0.60 MAE.

**Architecture:** Train a simple user-item bias CF model on the Yelp content-based data (968k reviews, string IDs, 1-5 stars). Use its predictions alongside the existing deep router predictions as features for a meta-LightGBM trained on the validation set. Cold-start users (history_band=0) bypass the meta-model and use the CB incumbent prediction directly.

**Tech Stack:** pandas, numpy, scikit-learn (no new deps), lightgbm

---

## Context

- **CB val predictions** already saved: `content-based/artifacts/known_user_deep_router_v2_eval_v3/known_user_deep_validation_predictions.csv`
  - 64,727 rows, bands 1/2-5/6-20/>20
  - Key columns: `review_id`, `user`, `item`, `rating`, `history_band`, `history_count`, `history_rating_std`, `deep_prediction_raw`, `incumbent_prediction_raw`
- **CB test predictions** (rounded): `content-based/artifacts/known_user_deep_router_v2_eval_v3/submission.csv` — only has `review_id, stars` (integer). Need raw floats.
- **Existing CF models** (in `models/`) use different dataset (integer IDs, 1-10 scale). NOT usable.
- **CF model to build**: trained on `content-based/data/train_reviews.csv` using Yelp string IDs.
- **Val split**: `temporal_train_validation_split(train_reviews, val_size=0.2, timestamp_col="date")` — same deterministic split used in deep router training.
- **All commands** run from repo root with `uv run python content-based/<script>.py`.

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `content-based/predict_known_user_deep_router_submission.py` | Save `raw_predictions.csv` alongside submission |
| Create | `content-based/train_cf_for_meta.py` | Train CF bias model, generate CF predictions for val+test |
| Create | `content-based/train_meta_lgbm.py` | Build meta-features, train LightGBM, generate submission |

---

## Task 1: Modify CB prediction script to save raw scores

**Files:**
- Modify: `content-based/predict_known_user_deep_router_submission.py:248-254`

The existing script saves only rounded `stars`. We need raw float predictions for the meta-learner.

- [ ] **Step 1.1: Read the current save block**

Open `content-based/predict_known_user_deep_router_submission.py` and find the block around line 248:

```python
    submission = pd.DataFrame(
        {
            "review_id": final_test["review_id"].astype(str),
            "stars": _round_half_up(final_test["final_prediction_raw"].to_numpy(dtype=np.float32)).clip(1, 5).astype(np.int32),
        }
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=False)
```

- [ ] **Step 1.2: Add raw predictions save after `submission.to_csv`**

Insert immediately after `submission.to_csv(save_path, index=False)`:

```python
    raw_pred_path = save_path.parent / "raw_predictions.csv"
    final_test[["review_id", "history_band", "incumbent_prediction_raw", "final_prediction_raw", "final_router_branch"]].rename(
        columns={"final_prediction_raw": "cb_prediction_raw"}
    ).to_csv(raw_pred_path, index=False)
```

- [ ] **Step 1.3: Run the prediction script to regenerate submission + raw predictions**

```bash
cd content-based
uv run python predict_known_user_deep_router_submission.py \
    --artifact-root artifacts/known_user_deep_router_v2_eval_v3
```

Expected output: a JSON summary printed to stdout, e.g.:
```
{
  "n_rows": 414765,
  ...
}
```

- [ ] **Step 1.4: Verify raw_predictions.csv was created**

```bash
head -3 content-based/artifacts/known_user_deep_router_v2_eval_v3/raw_predictions.csv
wc -l content-based/artifacts/known_user_deep_router_v2_eval_v3/raw_predictions.csv
```

Expected: header + 414765 rows, columns `review_id,history_band,incumbent_prediction_raw,cb_prediction_raw,final_router_branch`.

- [ ] **Step 1.5: Commit**

```bash
git add content-based/predict_known_user_deep_router_submission.py
git -c commit.gpgsign=false commit -m "feat: save raw_predictions.csv alongside submission in deep router"
```

---

## Task 2: Train CF bias model and generate predictions

**Files:**
- Create: `content-based/train_cf_for_meta.py`

Trains a user-item bias model (`r_hat = global_mean + user_bias + item_bias`) on the temporal train split. Generates predictions for the 64,727 known-user val rows and all 414,765 test rows.

- [ ] **Step 2.1: Write `content-based/train_cf_for_meta.py`**

```python
"""
Train a simple CF bias model on the Yelp content-based training data and
generate predictions for the validation known-user rows and test rows.

Outputs to artifacts/cf_meta_model_v1/:
  - cf_model.joblib          : trained CFBiasModel
  - cf_val_predictions.csv   : review_id, cf_prediction  (64k known-user val rows)
  - cf_test_predictions.csv  : review_id, cf_prediction  (414k test rows)
  - train_summary.json       : val MAE of CF model alone
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.split import temporal_train_validation_split

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
CB_VAL_PRED_PATH = ARTIFACTS_DIR / "known_user_deep_router_v2_eval_v3" / "known_user_deep_validation_predictions.csv"
OUT_DIR = ARTIFACTS_DIR / "cf_meta_model_v1"


class CFBiasModel:
    """Global mean + per-user bias + per-item bias collaborative filter."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.user_bias_: dict[str, float] = {}
        self.item_bias_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "CFBiasModel":
        """df must have columns: user_id, business_id, stars."""
        self.global_mean_ = float(df["stars"].mean())
        user_means = df.groupby("user_id")["stars"].mean()
        self.user_bias_ = (user_means - self.global_mean_).to_dict()
        item_means = df.groupby("business_id")["stars"].mean()
        self.item_bias_ = (item_means - self.global_mean_).to_dict()
        return self

    def predict_row(self, user_id: str, business_id: str) -> float:
        b_u = self.user_bias_.get(user_id, 0.0)
        b_i = self.item_bias_.get(business_id, 0.0)
        return float(np.clip(self.global_mean_ + b_u + b_i, 1.0, 5.0))

    def predict_df(self, df: pd.DataFrame, user_col: str = "user_id", item_col: str = "business_id") -> np.ndarray:
        return np.array(
            [self.predict_row(u, i) for u, i in zip(df[user_col], df[item_col])],
            dtype=np.float32,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train_reviews.csv ...")
    train_reviews = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)

    print("Applying temporal split (val_size=0.2, timestamp_col='date') ...")
    train_split, val_split = temporal_train_validation_split(
        train_reviews, val_size=0.2, timestamp_col="date"
    )
    print(f"  train_split: {len(train_split):,} rows | val_split: {len(val_split):,} rows")

    print("Training CF bias model on train_split ...")
    model = CFBiasModel().fit(train_split)
    print(f"  global_mean = {model.global_mean_:.4f}")
    print(f"  known users = {len(model.user_bias_):,} | known items = {len(model.item_bias_):,}")

    print("Saving model ...")
    joblib.dump(model, OUT_DIR / "cf_model.joblib")

    # --- Val predictions (known-user rows only) ---
    print("Loading CB val predictions ...")
    cb_val = pd.read_csv(CB_VAL_PRED_PATH, low_memory=False)
    # cb_val has 'user' (user_id string) and 'item' (business_id string)
    print(f"  CB val rows: {len(cb_val):,}")

    print("Generating CF val predictions ...")
    cb_val["cf_prediction"] = model.predict_df(cb_val, user_col="user", item_col="item")
    cf_val_out = cb_val[["review_id", "cf_prediction"]].copy()
    cf_val_out.to_csv(OUT_DIR / "cf_val_predictions.csv", index=False)

    val_cf_mae = float(np.mean(np.abs(cb_val["rating"].to_numpy(dtype=np.float32) - cb_val["cf_prediction"].to_numpy(dtype=np.float32))))
    val_cb_mae = float(np.mean(np.abs(cb_val["rating"].to_numpy(dtype=np.float32) - cb_val["deep_prediction_raw"].to_numpy(dtype=np.float32))))
    print(f"  CF val MAE (known users): {val_cf_mae:.6f}")
    print(f"  CB val MAE (known users): {val_cb_mae:.6f}")

    # --- Test predictions ---
    print("Loading test_reviews.csv ...")
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)
    print(f"  Test rows: {len(test_reviews):,}")

    print("Generating CF test predictions ...")
    test_reviews["cf_prediction"] = model.predict_df(test_reviews, user_col="user_id", item_col="business_id")
    cf_test_out = test_reviews[["review_id", "cf_prediction"]].copy()
    cf_test_out.to_csv(OUT_DIR / "cf_test_predictions.csv", index=False)

    summary = {
        "train_split_rows": int(len(train_split)),
        "val_split_rows": int(len(val_split)),
        "known_users": int(len(model.user_bias_)),
        "known_items": int(len(model.item_bias_)),
        "global_mean": float(model.global_mean_),
        "val_cf_mae_known_users": val_cf_mae,
        "val_cb_mae_known_users": val_cb_mae,
        "test_rows": int(len(test_reviews)),
    }
    (OUT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done. Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run the script**

```bash
cd content-based
uv run python train_cf_for_meta.py
```

Expected output (approximate):
```
Loading train_reviews.csv ...
Applying temporal split (val_size=0.2, timestamp_col='date') ...
  train_split: 774,227 rows | val_split: 193,557 rows
Training CF bias model on train_split ...
  global_mean = 3.7xxx
  known users = 432,xxx | known items = 30,0xx
Saving model ...
...
  CF val MAE (known users): 0.7xxx
  CB val MAE (known users): 0.6xxx
...
Done.
```

- [ ] **Step 2.3: Verify outputs**

```bash
ls content-based/artifacts/cf_meta_model_v1/
head -3 content-based/artifacts/cf_meta_model_v1/cf_val_predictions.csv
wc -l content-based/artifacts/cf_meta_model_v1/cf_val_predictions.csv
wc -l content-based/artifacts/cf_meta_model_v1/cf_test_predictions.csv
```

Expected: 4 files, cf_val has 64,727 rows + header, cf_test has 414,765 rows + header.

- [ ] **Step 2.4: Commit**

```bash
git add content-based/train_cf_for_meta.py
git -c commit.gpgsign=false commit -m "feat: train CF bias model and generate meta predictions"
```

---

## Task 3: Train meta-LightGBM and generate submission

**Files:**
- Create: `content-based/train_meta_lgbm.py`

Joins CB and CF val predictions, builds meta-features, trains LightGBM on val set, applies to test. Cold-start rows (history_band=0) use CB prediction directly.

- [ ] **Step 3.1: Write `content-based/train_meta_lgbm.py`**

```python
"""
Train a meta-LightGBM stacker on top of CB (deep router) and CF (bias model) predictions.

Inputs:
  - content-based/artifacts/known_user_deep_router_v2_eval_v3/known_user_deep_validation_predictions.csv
  - content-based/artifacts/known_user_deep_router_v2_eval_v3/raw_predictions.csv
  - content-based/artifacts/cf_meta_model_v1/cf_val_predictions.csv
  - content-based/artifacts/cf_meta_model_v1/cf_test_predictions.csv
  - content-based/data/train_reviews.csv  (for test user stats)

Outputs to artifacts/meta_lgbm_hybrid_v1/:
  - meta_model.joblib    : trained LightGBM booster
  - submission.csv       : review_id, stars
  - validation_mae.json  : meta val MAE vs CB alone
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
CB_VAL_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "known_user_deep_validation_predictions.csv"
CB_TEST_RAW_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "raw_predictions.csv"
CF_VAL_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_val_predictions.csv"
CF_TEST_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_test_predictions.csv"
OUT_DIR = ARTIFACTS / "meta_lgbm_hybrid_v1"

HISTORY_BAND_ORDER = {"0": 0, "1": 1, "2-5": 2, "6-20": 6, ">20": 20}

META_FEATURES = [
    "cf_prediction",
    "cb_prediction_raw",
    "history_band_enc",
    "history_count",
    "history_rating_std",
    "cf_cb_diff",
]


def encode_band(band_series: pd.Series) -> pd.Series:
    return band_series.map(HISTORY_BAND_ORDER).fillna(0).astype(np.float32)


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def build_val_meta_frame(cb_val: pd.DataFrame, cf_val: pd.DataFrame) -> pd.DataFrame:
    df = cb_val.merge(cf_val, on="review_id", how="inner")
    df["cb_prediction_raw"] = df["deep_prediction_raw"].astype(np.float32)
    df["history_band_enc"] = encode_band(df["history_band"].astype(str))
    df["history_count"] = df["history_count"].astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].astype(np.float32)
    df["cf_prediction"] = df["cf_prediction"].astype(np.float32)
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]
    return df


def compute_test_user_stats(train_reviews: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user history stats from full training data (context for test)."""
    stats = train_reviews.groupby("user_id").agg(
        history_count=("stars", "count"),
        history_rating_std=("stars", "std"),
    ).reset_index()
    stats["history_rating_std"] = stats["history_rating_std"].fillna(0.0).astype(np.float32)
    stats["history_count"] = stats["history_count"].astype(np.float32)
    return stats


def build_test_meta_frame(
    cb_test_raw: pd.DataFrame,
    cf_test: pd.DataFrame,
    user_stats: pd.DataFrame,
    test_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (known_df, cold_df) split by history_band."""
    df = cb_test_raw.merge(cf_test, on="review_id", how="left")
    # Join user_id from test_reviews for stat lookup
    user_lookup = test_reviews[["review_id", "user_id"]].copy()
    df = df.merge(user_lookup, on="review_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["cf_prediction"] = df["cf_prediction"].fillna(df["cb_prediction_raw"]).astype(np.float32)
    df["history_band_enc"] = encode_band(df["history_band"].astype(str))
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]

    cold_mask = df["history_band"].astype(str) == "0"
    return df[~cold_mask].copy(), df[cold_mask].copy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CB val predictions ...")
    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    print(f"  {len(cb_val):,} rows")

    print("Loading CF val predictions ...")
    cf_val = pd.read_csv(CF_VAL_PATH, low_memory=False)

    print("Building val meta-feature frame ...")
    val_df = build_val_meta_frame(cb_val, cf_val)
    print(f"  Meta val rows: {len(val_df):,}")

    X = val_df[META_FEATURES].to_numpy(dtype=np.float32)
    y = val_df["rating"].to_numpy(dtype=np.float32)

    # Inner split for early stopping: 90% train_meta / 10% early_stop
    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)

    print("Training meta-LightGBM ...")
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=META_FEATURES)
    deval = lgb.Dataset(X_es, label=y_es, reference=dtrain)

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "num_leaves": 16,
        "min_data_in_leaf": 50,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "verbose": -1,
    }
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=True), lgb.log_evaluation(period=20)]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[deval],
        callbacks=callbacks,
    )

    val_preds_meta = booster.predict(X, num_iteration=booster.best_iteration)
    val_mae_meta = float(np.mean(np.abs(y - val_preds_meta.astype(np.float32))))
    val_mae_cb = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(dtype=np.float32))))
    val_mae_cf = float(np.mean(np.abs(y - val_df["cf_prediction"].to_numpy(dtype=np.float32))))
    print(f"  Val MAE — CB alone: {val_mae_cb:.6f} | CF alone: {val_mae_cf:.6f} | Meta: {val_mae_meta:.6f}")

    joblib.dump(booster, OUT_DIR / "meta_model.joblib")

    # --- Test predictions ---
    print("Loading CB test raw predictions ...")
    cb_test_raw = pd.read_csv(CB_TEST_RAW_PATH, low_memory=False)

    print("Loading CF test predictions ...")
    cf_test = pd.read_csv(CF_TEST_PATH, low_memory=False)

    print("Loading train_reviews.csv for test user stats ...")
    train_reviews = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    user_stats = compute_test_user_stats(train_reviews)

    print("Loading test_reviews.csv ...")
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)

    print("Building test meta-feature frame ...")
    known_test, cold_test = build_test_meta_frame(cb_test_raw, cf_test, user_stats, test_reviews)
    print(f"  Known test rows: {len(known_test):,} | Cold test rows: {len(cold_test):,}")

    # Meta predictions for known users
    X_test = known_test[META_FEATURES].to_numpy(dtype=np.float32)
    known_test = known_test.copy()
    known_test["final_prediction_raw"] = booster.predict(X_test, num_iteration=booster.best_iteration).astype(np.float32)

    # Cold users: use CB incumbent directly
    cold_test = cold_test.copy()
    cold_test["final_prediction_raw"] = cold_test["cb_prediction_raw"].astype(np.float32)

    all_test = pd.concat([known_test[["review_id", "final_prediction_raw"]], cold_test[["review_id", "final_prediction_raw"]]], ignore_index=True)
    all_test["stars"] = _round_half_up(np.clip(all_test["final_prediction_raw"].to_numpy(dtype=np.float32), 1.0, 5.0))

    submission = all_test[["review_id", "stars"]].copy()
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"Submission saved: {len(submission):,} rows")
    print(f"  stars distribution:\n{submission['stars'].value_counts().sort_index().to_string()}")

    summary = {
        "val_mae_cb_alone": val_mae_cb,
        "val_mae_cf_alone": val_mae_cf,
        "val_mae_meta": val_mae_meta,
        "val_delta_vs_cb": val_mae_meta - val_mae_cb,
        "meta_features": META_FEATURES,
        "lgbm_best_iteration": int(booster.best_iteration),
        "known_test_rows": int(len(known_test)),
        "cold_test_rows": int(len(cold_test)),
        "total_test_rows": int(len(submission)),
    }
    (OUT_DIR / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run the meta-LightGBM training**

```bash
cd content-based
uv run python train_meta_lgbm.py
```

Expected output (approximate):
```
...
  Val MAE — CB alone: 0.6694 | CF alone: 0.7xxx | Meta: 0.64xx
Submission saved: 414,765 rows
```

The meta MAE should be lower than the CB-alone MAE (0.6694 on known users).

- [ ] **Step 3.3: Verify the submission**

```bash
head -5 content-based/artifacts/meta_lgbm_hybrid_v1/submission.csv
wc -l content-based/artifacts/meta_lgbm_hybrid_v1/submission.csv
cat content-based/artifacts/meta_lgbm_hybrid_v1/validation_mae.json
```

Expected: 414,765 rows + header, stars in range 1-5, `val_delta_vs_cb` negative (improvement).

- [ ] **Step 3.4: If `val_delta_vs_cb >= 0` (no improvement), diagnose**

Run this snippet to check feature importances:
```python
import joblib
booster = joblib.load("content-based/artifacts/meta_lgbm_hybrid_v1/meta_model.joblib")
import lightgbm as lgb
print(dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain"))))
```

If `cf_prediction` importance is near zero, the CF model adds no signal. In this case: increase CF model complexity (add per-category item biases) or blend CB alone with a 0.95 weight to CB.

- [ ] **Step 3.5: Commit**

```bash
git add content-based/train_meta_lgbm.py
git -c commit.gpgsign=false commit -m "feat: meta-LightGBM hybrid stacker (CF + CB deep router)"
```

---

## Task 4: Compare results and decide on submission

- [ ] **Step 4.1: Print comparison table**

```bash
python3 -c "
import json, pathlib
v3 = json.loads(pathlib.Path('content-based/artifacts/known_user_deep_router_v2_eval_v3/validation_summary.json').read_text())
meta = json.loads(pathlib.Path('content-based/artifacts/meta_lgbm_hybrid_v1/validation_mae.json').read_text())
print('CB v3 overall val MAE:', v3.get('final_overall_mae'))
print('Meta CB-only val MAE (known users):', meta['val_mae_cb_alone'])
print('Meta CF-only val MAE (known users):', meta['val_mae_cf_alone'])
print('Meta stacked val MAE (known users):', meta['val_mae_meta'])
print('Delta vs CB (known users):', meta['val_delta_vs_cb'])
"
```

- [ ] **Step 4.2: Submit if `val_delta_vs_cb < 0`**

If the meta MAE is lower than CB alone on the known-user val rows, submit `content-based/artifacts/meta_lgbm_hybrid_v1/submission.csv` to the leaderboard.

- [ ] **Step 4.3: Update experiment registry**

Add a row to `docs/experiments/registry.md`:

```markdown
| [`meta_lgbm_hybrid_v1`](../content-based/artifacts/meta_lgbm_hybrid_v1) | `candidate` | hybrid CF+CB meta stacker | meta-LightGBM on CF bias + CB deep router |
```

- [ ] **Step 4.4: Commit**

```bash
git add docs/experiments/registry.md
git -c commit.gpgsign=false commit -m "docs: register meta_lgbm_hybrid_v1 candidate"
```
