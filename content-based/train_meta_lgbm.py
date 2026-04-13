"""
Train a meta-LightGBM stacker on top of CB (deep router) and CF (bias model) predictions.

Val set inputs (64k known-user rows):
  - artifacts/known_user_deep_router_v2_eval_v3/known_user_deep_validation_predictions.csv
  - artifacts/cf_meta_model_v1/cf_val_predictions.csv

Test set inputs (414k rows):
  - artifacts/known_user_deep_router_v2_eval_v3/submission.csv   (rounded stars 1-5)
  - artifacts/cf_meta_model_v1/cf_test_predictions.csv
  - data/train_reviews.csv  (for test user stats + history_band)
  - data/test_reviews.csv   (for user_id / business_id lookup)

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

from utils.lgbm_raw_features import history_band_from_count

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

CB_VAL_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "known_user_deep_validation_predictions.csv"
CB_TEST_SUBMISSION_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "submission.csv"
CF_VAL_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_val_predictions.csv"
CF_TEST_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_test_predictions.csv"
OUT_DIR = ARTIFACTS / "meta_lgbm_hybrid_v1"

HISTORY_BAND_ORDER: dict[str, int] = {"0": 0, "1": 1, "2-5": 2, "6-20": 6, ">20": 20}

META_FEATURES = [
    "cf_prediction",
    "cb_prediction_raw",
    "history_band_enc",
    "history_count",
    "history_rating_std",
    "cf_cb_diff",
]


def encode_band(band_series: pd.Series) -> pd.Series:
    return band_series.astype(str).map(HISTORY_BAND_ORDER).fillna(0).astype(np.float32)


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def build_val_meta_frame(cb_val: pd.DataFrame, cf_val: pd.DataFrame) -> pd.DataFrame:
    df = cb_val.merge(cf_val, on="review_id", how="inner")
    df = df.copy()
    df["cb_prediction_raw"] = df["deep_prediction_raw"].astype(np.float32)
    df["history_band_enc"] = encode_band(df["history_band"])
    df["history_count"] = df["history_count"].astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["cf_prediction"] = df["cf_prediction"].astype(np.float32)
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]
    return df


def compute_user_stats_from_train(train_reviews: pd.DataFrame) -> pd.DataFrame:
    """Per-user history stats computed from full training data (context for test rows)."""
    stats = (
        train_reviews.groupby("user_id")
        .agg(
            history_count=("stars", "count"),
            history_rating_std=("stars", "std"),
        )
        .reset_index()
    )
    stats["history_count"] = stats["history_count"].astype(np.float32)
    stats["history_rating_std"] = stats["history_rating_std"].fillna(0.0).astype(np.float32)
    return stats


def build_test_meta_frame(
    cb_test_submission: pd.DataFrame,
    cf_test: pd.DataFrame,
    user_stats: pd.DataFrame,
    test_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (known_df, cold_df).

    cb_test_submission has: review_id, stars (rounded integer)
    history_band is derived from train user counts.
    """
    # Join user_id from test_reviews
    user_lookup = test_reviews[["review_id", "user_id", "business_id"]].copy()
    df = cb_test_submission.merge(user_lookup, on="review_id", how="left")

    # Join CF predictions
    df = df.merge(cf_test, on="review_id", how="left")

    # Join user stats
    df = df.merge(user_stats, on="user_id", how="left")
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)

    # Derive history_band from count
    df["history_band"] = df["history_count"].apply(lambda c: history_band_from_count(int(c)))

    # CB prediction: use stars as float (rounded, range 1-5)
    df["cb_prediction_raw"] = df["stars"].astype(np.float32)

    # CF prediction: fall back to CB if missing
    df["cf_prediction"] = df["cf_prediction"].fillna(df["cb_prediction_raw"]).astype(np.float32)

    # Encode features
    df["history_band_enc"] = encode_band(df["history_band"])
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]

    cold_mask = df["history_band"].astype(str) == "0"
    return df[~cold_mask].copy(), df[cold_mask].copy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Build val meta-feature frame                                        #
    # ------------------------------------------------------------------ #
    print("Loading CB val predictions ...")
    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    print(f"  {len(cb_val):,} rows")

    print("Loading CF val predictions ...")
    cf_val = pd.read_csv(CF_VAL_PATH, low_memory=False)

    print("Building val meta-feature frame ...")
    val_df = build_val_meta_frame(cb_val, cf_val)
    print(f"  Meta val rows: {len(val_df):,}")
    print(f"  Bands: {val_df['history_band'].value_counts().to_dict()}")

    X = val_df[META_FEATURES].to_numpy(dtype=np.float32)
    y = val_df["rating"].to_numpy(dtype=np.float32)

    # Inner 90/10 split for early stopping
    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)

    # ------------------------------------------------------------------ #
    #  Train meta-LightGBM                                                 #
    # ------------------------------------------------------------------ #
    print("Training meta-LightGBM ...")
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=META_FEATURES)
    deval = lgb.Dataset(X_es, label=y_es, reference=dtrain)

    params: dict = {
        "objective": "regression_l1",
        "metric": "mae",
        "num_leaves": 16,
        "min_data_in_leaf": 50,
        "learning_rate": 0.05,
        "verbose": -1,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=True),
        lgb.log_evaluation(period=20),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[deval],
        callbacks=callbacks,
    )

    val_preds_meta = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    val_mae_meta = float(np.mean(np.abs(y - val_preds_meta)))
    val_mae_cb = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(dtype=np.float32))))
    val_mae_cf = float(np.mean(np.abs(y - val_df["cf_prediction"].to_numpy(dtype=np.float32))))

    print(f"\nVal MAE  CB alone : {val_mae_cb:.6f}")
    print(f"Val MAE  CF alone : {val_mae_cf:.6f}")
    print(f"Val MAE  Meta     : {val_mae_meta:.6f}")
    print(f"Delta vs CB       : {val_mae_meta - val_mae_cb:+.6f}")

    # Feature importances
    importances = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain").tolist()))
    print(f"Feature importances (gain): {importances}")

    joblib.dump(booster, OUT_DIR / "meta_model.joblib")

    # ------------------------------------------------------------------ #
    #  Build test meta-feature frame                                       #
    # ------------------------------------------------------------------ #
    print("\nLoading CB test submission ...")
    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    print(f"  {len(cb_test_sub):,} rows")

    print("Loading CF test predictions ...")
    cf_test = pd.read_csv(CF_TEST_PATH, low_memory=False)

    print("Loading train_reviews.csv for user stats ...")
    train_reviews = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    user_stats = compute_user_stats_from_train(train_reviews)

    print("Loading test_reviews.csv ...")
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)

    print("Building test meta-feature frame ...")
    known_test, cold_test = build_test_meta_frame(cb_test_sub, cf_test, user_stats, test_reviews)
    print(f"  Known test rows: {len(known_test):,} | Cold test rows: {len(cold_test):,}")

    # ------------------------------------------------------------------ #
    #  Generate test predictions                                           #
    # ------------------------------------------------------------------ #
    X_test = known_test[META_FEATURES].to_numpy(dtype=np.float32)
    known_preds = booster.predict(X_test, num_iteration=booster.best_iteration).astype(np.float32)
    known_test = known_test.copy()
    known_test["final_prediction_raw"] = known_preds

    # Cold users: use CB incumbent (rounded stars) directly
    cold_test = cold_test.copy()
    cold_test["final_prediction_raw"] = cold_test["cb_prediction_raw"].astype(np.float32)

    all_test = pd.concat(
        [
            known_test[["review_id", "final_prediction_raw"]],
            cold_test[["review_id", "final_prediction_raw"]],
        ],
        ignore_index=True,
    )
    all_test["stars"] = _round_half_up(
        np.clip(all_test["final_prediction_raw"].to_numpy(dtype=np.float32), 1.0, 5.0)
    )

    submission = all_test[["review_id", "stars"]].copy()
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"\nSubmission saved: {len(submission):,} rows")
    print(f"Stars distribution:\n{submission['stars'].value_counts().sort_index().to_string()}")

    summary = {
        "val_mae_cb_alone": val_mae_cb,
        "val_mae_cf_alone": val_mae_cf,
        "val_mae_meta": val_mae_meta,
        "val_delta_vs_cb": val_mae_meta - val_mae_cb,
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
