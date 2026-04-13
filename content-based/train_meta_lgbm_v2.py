"""
Meta-LightGBM v2 — fixes from v1:

  1. Direction 2 fix: round deep_prediction_raw to integers before using as
     cb_prediction_raw, so the CB feature distribution matches test.

  2. More meta-features available for both val and test:
       - user_mean_rating    (user bias from training history)
       - business_mean_rating (item bias from training history)
       - business_review_count (item popularity)

  3. Tighter regularization: num_leaves=8, min_data_in_leaf=100,
     lambda_l1=0.5 to reduce overfitting to the 64k val set.

Outputs to artifacts/meta_lgbm_hybrid_v2/:
  - meta_model_v2.joblib
  - submission.csv
  - validation_mae.json
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

from utils.lgbm_raw_features import history_band_from_count
from utils.split import temporal_train_validation_split

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

CB_VAL_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "known_user_deep_validation_predictions.csv"
CB_TEST_SUBMISSION_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "submission.csv"
CF_VAL_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_val_predictions.csv"
CF_TEST_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_test_predictions.csv"
OUT_DIR = ARTIFACTS / "meta_lgbm_hybrid_v2"

HISTORY_BAND_ORDER: dict[str, int] = {"0": 0, "1": 1, "2-5": 2, "6-20": 6, ">20": 20}

META_FEATURES = [
    "cf_prediction",
    "cb_prediction_raw",       # rounded integer (1-5) — matches test format
    "history_band_enc",
    "history_count",
    "history_rating_std",
    "user_mean_rating",        # user bias from training history
    "business_mean_rating",    # item bias from training history
    "business_review_count",   # item popularity in training
    "cf_cb_diff",
]


def encode_band(s: pd.Series) -> pd.Series:
    return s.astype(str).map(HISTORY_BAND_ORDER).fillna(0).astype(np.float32)


def _round_half_up(v: np.ndarray) -> np.ndarray:
    return np.floor(v + 0.5).astype(np.int32)


def compute_train_stats(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (user_stats, item_stats) computed from train_df.
    user_stats columns : user_id, user_mean_rating, history_count, history_rating_std
    item_stats columns : business_id, business_mean_rating, business_review_count
    """
    user_stats = (
        train_df.groupby("user_id")["stars"]
        .agg(user_mean_rating="mean", history_count="count", history_rating_std="std")
        .reset_index()
    )
    user_stats["history_rating_std"] = user_stats["history_rating_std"].fillna(0.0).astype(np.float32)
    user_stats["history_count"] = user_stats["history_count"].astype(np.float32)
    user_stats["user_mean_rating"] = user_stats["user_mean_rating"].astype(np.float32)

    item_stats = (
        train_df.groupby("business_id")["stars"]
        .agg(business_mean_rating="mean", business_review_count="count")
        .reset_index()
    )
    item_stats["business_mean_rating"] = item_stats["business_mean_rating"].astype(np.float32)
    item_stats["business_review_count"] = item_stats["business_review_count"].astype(np.float32)

    return user_stats, item_stats


def build_val_meta_frame(
    cb_val: pd.DataFrame,
    cf_val: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    df = cb_val.merge(cf_val, on="review_id", how="inner")

    # Only pull the new columns from user/item stats (cb_val already has history_count, history_rating_std)
    user_mean = user_stats[["user_id", "user_mean_rating"]]
    item_mean = item_stats[["business_id", "business_mean_rating", "business_review_count"]]

    df = df.merge(user_mean, left_on="user", right_on="user_id", how="left")
    df = df.merge(item_mean, left_on="item", right_on="business_id", how="left")

    # Direction 2 fix: round CB prediction to integer (matches test format)
    df["cb_prediction_raw"] = df["deep_prediction_raw"].round().clip(1, 5).astype(np.float32)

    df["history_band_enc"] = encode_band(df["history_band"])
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_mean_rating"] = df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_review_count"] = df["business_review_count"].fillna(0.0).astype(np.float32)
    df["cf_prediction"] = df["cf_prediction"].astype(np.float32)
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]
    return df


def build_test_meta_frame(
    cb_test_sub: pd.DataFrame,
    cf_test: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    test_reviews: pd.DataFrame,
    global_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = test_reviews[["review_id", "user_id", "business_id"]].copy()
    df = cb_test_sub.merge(lookup, on="review_id", how="left")
    df = df.merge(cf_test, on="review_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    df = df.merge(item_stats, on="business_id", how="left")

    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_mean_rating"] = df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_review_count"] = df["business_review_count"].fillna(0.0).astype(np.float32)
    df["history_band"] = df["history_count"].apply(lambda c: history_band_from_count(int(c)))
    df["history_band_enc"] = encode_band(df["history_band"])
    df["cb_prediction_raw"] = df["stars"].astype(np.float32)
    df["cf_prediction"] = df["cf_prediction"].fillna(df["cb_prediction_raw"]).astype(np.float32)
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]

    cold_mask = df["history_band"].astype(str) == "0"
    return df[~cold_mask].copy(), df[cold_mask].copy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute training stats from the same train_split used to train CB
    print("Loading train_reviews.csv and computing split stats ...")
    train_reviews_full = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    train_split, _ = temporal_train_validation_split(
        train_reviews_full, val_size=0.2, timestamp_col="date"
    )
    global_mean = float(train_split["stars"].mean())
    print(f"  train_split: {len(train_split):,} rows | global_mean={global_mean:.4f}")

    # For val: use train_split stats (matches what CB model saw)
    user_stats_split, item_stats_split = compute_train_stats(train_split)

    # For test: use full training data stats (CB model at test time uses all of train)
    user_stats_full, item_stats_full = compute_train_stats(train_reviews_full)
    global_mean_full = float(train_reviews_full["stars"].mean())

    # ------------------------------------------------------------------ #
    # Val meta-frame                                                       #
    # ------------------------------------------------------------------ #
    print("Loading CB val and CF val predictions ...")
    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    cf_val = pd.read_csv(CF_VAL_PATH, low_memory=False)

    print("Building val meta-feature frame ...")
    val_df = build_val_meta_frame(cb_val, cf_val, user_stats_split, item_stats_split, global_mean)
    print(f"  Val rows: {len(val_df):,} | features: {META_FEATURES}")

    X = val_df[META_FEATURES].to_numpy(dtype=np.float32)
    y = val_df["rating"].to_numpy(dtype=np.float32)

    # Baselines
    mae_cb_raw = float(np.mean(np.abs(y - val_df["deep_prediction_raw"].to_numpy(dtype=np.float32))))
    mae_cb_round = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(dtype=np.float32))))
    mae_cf = float(np.mean(np.abs(y - val_df["cf_prediction"].to_numpy(dtype=np.float32))))
    print(f"  Val MAE CB raw     : {mae_cb_raw:.6f}")
    print(f"  Val MAE CB rounded : {mae_cb_round:.6f}  (correct baseline, matches test)")
    print(f"  Val MAE CF bias    : {mae_cf:.6f}")

    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)

    # ------------------------------------------------------------------ #
    # Train meta-LightGBM (tighter regularization than v1)                #
    # ------------------------------------------------------------------ #
    print("\nTraining meta-LightGBM v2 ...")
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=META_FEATURES)
    deval = lgb.Dataset(X_es, label=y_es, reference=dtrain)

    params: dict = {
        "objective": "regression_l1",
        "metric": "mae",
        "num_leaves": 8,            # v1 used 16 — less capacity, more generalization
        "min_data_in_leaf": 100,    # v1 used 50
        "learning_rate": 0.05,
        "lambda_l1": 0.5,           # L1 regularization
        "lambda_l2": 0.5,           # L2 regularization
        "verbose": -1,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=25),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[deval],
        callbacks=callbacks,
    )

    val_preds = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    mae_meta = float(np.mean(np.abs(y - val_preds)))
    print(f"\n  Val MAE meta v2    : {mae_meta:.6f}")
    print(f"  Delta vs CB rounded: {mae_meta - mae_cb_round:+.6f}")

    importances = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain").tolist()))
    print(f"  Feature importances (gain): {importances}")

    joblib.dump(booster, OUT_DIR / "meta_model_v2.joblib")

    # ------------------------------------------------------------------ #
    # Test predictions                                                     #
    # ------------------------------------------------------------------ #
    print("\nLoading test inputs ...")
    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    cf_test = pd.read_csv(CF_TEST_PATH, low_memory=False)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)

    print("Building test meta-feature frame ...")
    known_test, cold_test = build_test_meta_frame(
        cb_test_sub, cf_test, user_stats_full, item_stats_full, test_reviews, global_mean_full
    )
    print(f"  Known test: {len(known_test):,} | Cold test: {len(cold_test):,}")

    X_test = known_test[META_FEATURES].to_numpy(dtype=np.float32)
    known_test = known_test.copy()
    known_test["final_prediction_raw"] = booster.predict(X_test, num_iteration=booster.best_iteration).astype(np.float32)

    cold_test = cold_test.copy()
    cold_test["final_prediction_raw"] = cold_test["cb_prediction_raw"].astype(np.float32)

    all_test = pd.concat(
        [known_test[["review_id", "final_prediction_raw"]], cold_test[["review_id", "final_prediction_raw"]]],
        ignore_index=True,
    )
    all_test["stars"] = _round_half_up(np.clip(all_test["final_prediction_raw"].to_numpy(dtype=np.float32), 1.0, 5.0))

    submission = all_test[["review_id", "stars"]].copy()
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"\nSubmission saved: {len(submission):,} rows")
    print(f"Stars distribution:\n{submission['stars'].value_counts().sort_index().to_string()}")

    summary = {
        "val_mae_cb_raw": mae_cb_raw,
        "val_mae_cb_rounded": mae_cb_round,
        "val_mae_cf": mae_cf,
        "val_mae_meta_v2": mae_meta,
        "val_delta_vs_cb_rounded": mae_meta - mae_cb_round,
        "meta_features": META_FEATURES,
        "lgbm_best_iteration": int(booster.best_iteration),
        "lgbm_feature_importances_gain": importances,
        "known_test_rows": int(len(known_test)),
        "cold_test_rows": int(len(cold_test)),
        "total_test_rows": int(len(submission)),
    }
    (OUT_DIR / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
