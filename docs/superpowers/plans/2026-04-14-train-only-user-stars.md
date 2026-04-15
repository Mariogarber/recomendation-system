# Train-Only User Stars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `user_average_stars` (from `usuarios.csv`, Yelp all-time aggregate that may encode test-period reviews) with a train-split-derived per-user mean in the LightGBM router.

**Architecture:** Compute per-user mean from `train_split` (validation branch) or full `train_reviews` (submission branch), overwrite `users_df["average_stars"]` before it reaches `_prepare_users()`, fall back to global mean for cold users. A new helper `build_train_user_stars` lives in `utils/lgbm_raw_features.py`.

**Tech Stack:** Python, pandas, LightGBM, pytest

---

### Task 1: Add `build_train_user_stars` + tests

**Files:**
- Modify: `content-based/utils/lgbm_raw_features.py`
- Modify: `content-based/tests/test_lgbm_raw_features.py`

- [ ] **Step 1: Write failing tests**

Add to `content-based/tests/test_lgbm_raw_features.py`:

```python
from content_based.utils.lgbm_raw_features import build_train_user_stars

def test_build_train_user_stars_known_user():
    train = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "stars": [3.0, 5.0, 2.0]})
    result = build_train_user_stars(train, global_mean=3.5)
    assert abs(result["u1"] - 4.0) < 1e-5
    assert abs(result["u2"] - 2.0) < 1e-5

def test_build_train_user_stars_cold_user_gets_global_mean():
    train = pd.DataFrame({"user_id": ["u1"], "stars": [4.0]})
    result = build_train_user_stars(train, global_mean=3.5)
    assert "u_cold" not in result

def test_build_train_user_stars_returns_dict():
    train = pd.DataFrame({"user_id": ["u1"], "stars": [4.0]})
    result = build_train_user_stars(train, global_mean=3.5)
    assert isinstance(result, dict)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd content-based && python -m pytest tests/test_lgbm_raw_features.py::test_build_train_user_stars_known_user -v`
Expected: ImportError or AttributeError (function not yet defined)

- [ ] **Step 3: Add `build_train_user_stars` to `utils/lgbm_raw_features.py`**

```python
def build_train_user_stars(train_reviews_df: pd.DataFrame, global_mean: float) -> dict:
    """Compute per-user mean stars from train_reviews only (no leakage)."""
    user_col = "user_id" if "user_id" in train_reviews_df.columns else "user"
    rating_col = "stars" if "stars" in train_reviews_df.columns else "rating"
    return (
        train_reviews_df.groupby(user_col)[rating_col]
        .mean()
        .astype(float)
        .to_dict()
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd content-based && python -m pytest tests/test_lgbm_raw_features.py::test_build_train_user_stars_known_user tests/test_lgbm_raw_features.py::test_build_train_user_stars_cold_user_gets_global_mean tests/test_lgbm_raw_features.py::test_build_train_user_stars_returns_dict -v`
Expected: 3 PASSED

---

### Task 2: Patch `train_lgbm_raw_router.py`

**Files:**
- Modify: `content-based/train_lgbm_raw_router.py`

- [ ] **Step 1: Add import**

Add `build_train_user_stars` to the import from `utils.lgbm_raw_features`.

- [ ] **Step 2: Add validation-branch replacement block**

After `train_split`/`val_split` are created, before validation feature building:

```python
_train_global_mean = float(train_split["stars"].mean())
_train_user_stars = build_train_user_stars(train_split, global_mean=_train_global_mean)
users_df = users_df.copy()
users_df["average_stars"] = (
    users_df["user_id"].map(_train_user_stars).fillna(_train_global_mean).astype(float)
)
```

- [ ] **Step 3: Add submission-branch replacement block**

Before the submission call that uses full `train_reviews`:

```python
_sub_global_mean = float(train_reviews["stars"].mean())
_sub_user_stars = build_train_user_stars(train_reviews, global_mean=_sub_global_mean)
users_df_sub = load_users(args.data_dir).copy()
users_df_sub["average_stars"] = (
    users_df_sub["user_id"].map(_sub_user_stars).fillna(_sub_global_mean).astype(float)
)
```

Then pass `users_df_sub` to all submission-branch feature builders.

- [ ] **Step 4: Change default save root**

Change default `--save-root` from `lgbm_raw_router_prefix_deep_v1` to `lgbm_train_stars_v1`.

---

### Task 3: Confirm submission section scope

- [ ] Read the full submission section of `train_lgbm_raw_router.py` to verify all feature builder calls use `users_df_sub` and none accidentally reuse `users_df`.

---

### Task 4: Train

- [ ] Run: `cd content-based && python train_lgbm_raw_router.py`
- [ ] Confirm artifacts saved to `artifacts/lgbm_train_stars_v1/`
- [ ] Note local MAE (expected: higher than 0.6265 — leaky feature removed)

---

### Task 5: Confirm submission file

- [ ] Check `artifacts/lgbm_train_stars_v1/submission.csv` exists
- [ ] Spot-check: head, shape, value range 1–5

---

### Task 6: Update docs

**Files:**
- Modify: `docs/status/current-state.md`
- Modify: `docs/proposals/content-based-next-ideas.md`
- Modify: `docs/experiments/registry.md`

- [ ] Add `lgbm_train_stars_v1` to the experiments registry with local MAE and hypothesis
- [ ] Update `current-state.md` with the experiment result and note leakage hypothesis
- [ ] Update `content-based-next-ideas.md`: if local MAE rose, note that and await leaderboard
