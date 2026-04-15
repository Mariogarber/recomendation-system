"""
Train a meta-LightGBM stacker on top of CB (deep router) predictions.

Version 1: Stacks CB + CF predictions. Light regularization.
Version 2: Stacks CB + CF + user/item bias features. Tighter regularization.
Version 3: CB + user/item bias only (no CF — CF hurts). Relaxed regularization.
           This is the recommended default.

Select with --version 1|2|3 (default: 3).

Outputs to artifacts/meta_lgbm_hybrid_v{version}/:
  - meta_model_v{version}.joblib
  - submission.csv
  - validation_mae.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.lgbm_raw_features import history_band_from_count
from utils.split import temporal_train_validation_split

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CB_VAL_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "known_user_deep_validation_predictions.csv"
CB_TEST_SUBMISSION_PATH = ARTIFACTS / "known_user_deep_router_v2_eval_v3" / "submission.csv"
CF_VAL_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_val_predictions.csv"
CF_TEST_PATH = ARTIFACTS / "cf_meta_model_v1" / "cf_test_predictions.csv"

HISTORY_BAND_ORDER: dict[str, int] = {"0": 0, "1": 1, "2-5": 2, "6-20": 6, ">20": 20}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def encode_band(s: pd.Series) -> pd.Series:
    return s.astype(str).map(HISTORY_BAND_ORDER).fillna(0).astype(np.float32)


def _round_half_up(v: np.ndarray) -> np.ndarray:
    return np.floor(v + 0.5).astype(np.int32)


def compute_train_stats(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _finalize_submission(known_test: pd.DataFrame, cold_test: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    all_test = pd.concat(
        [known_test[["review_id", "final_prediction_raw"]], cold_test[["review_id", "final_prediction_raw"]]],
        ignore_index=True,
    )
    all_test["stars"] = _round_half_up(np.clip(all_test["final_prediction_raw"].to_numpy(np.float32), 1.0, 5.0))
    submission = all_test[["review_id", "stars"]].copy()
    submission.to_csv(out_dir / "submission.csv", index=False)
    print(f"Submission saved: {len(submission):,} rows")
    print(f"Stars distribution:\n{submission['stars'].value_counts().sort_index().to_string()}")
    return submission


def _train_lgbm(X_tr, y_tr, X_es, y_es, features, params, num_boost_round, early_stopping_rounds, log_period):
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    deval = lgb.Dataset(X_es, label=y_es, reference=dtrain)
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=log_period),
    ]
    return lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[deval], callbacks=callbacks)


# ---------------------------------------------------------------------------
# Version 1
# ---------------------------------------------------------------------------

META_FEATURES_V1 = [
    "cf_prediction", "cb_prediction_raw", "history_band_enc",
    "history_count", "history_rating_std", "cf_cb_diff",
]

PARAMS_V1: dict = {
    "objective": "regression_l1", "metric": "mae",
    "num_leaves": 16, "min_data_in_leaf": 50,
    "learning_rate": 0.05, "verbose": -1,
}


def _build_val_v1(cb_val, cf_val):
    df = cb_val.merge(cf_val, on="review_id", how="inner").copy()
    df["cb_prediction_raw"] = df["deep_prediction_raw"].astype(np.float32)
    df["history_band_enc"] = encode_band(df["history_band"])
    df["history_count"] = df["history_count"].astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["cf_prediction"] = df["cf_prediction"].astype(np.float32)
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]
    return df


def _build_test_v1(cb_test_sub, cf_test, user_stats, test_reviews):
    lookup = test_reviews[["review_id", "user_id", "business_id"]].copy()
    df = cb_test_sub.merge(lookup, on="review_id", how="left").merge(cf_test, on="review_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["history_band"] = df["history_count"].apply(lambda c: history_band_from_count(int(c)))
    df["cb_prediction_raw"] = df["stars"].astype(np.float32)
    df["cf_prediction"] = df["cf_prediction"].fillna(df["cb_prediction_raw"]).astype(np.float32)
    df["history_band_enc"] = encode_band(df["history_band"])
    df["cf_cb_diff"] = df["cf_prediction"] - df["cb_prediction_raw"]
    cold_mask = df["history_band"].astype(str) == "0"
    return df[~cold_mask].copy(), df[cold_mask].copy()


def run_v1() -> None:
    out_dir = ARTIFACTS / "meta_lgbm_hybrid_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    cf_val = pd.read_csv(CF_VAL_PATH, low_memory=False)
    val_df = _build_val_v1(cb_val, cf_val)
    print(f"  Meta val rows: {len(val_df):,}  bands: {val_df['history_band'].value_counts().to_dict()}")

    X = val_df[META_FEATURES_V1].to_numpy(np.float32)
    y = val_df["rating"].to_numpy(np.float32)
    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)

    booster = _train_lgbm(X_tr, y_tr, X_es, y_es, META_FEATURES_V1, PARAMS_V1, 200, 20, 20)

    val_preds = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    mae_meta = float(np.mean(np.abs(y - val_preds)))
    mae_cb = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(np.float32))))
    mae_cf = float(np.mean(np.abs(y - val_df["cf_prediction"].to_numpy(np.float32))))
    print(f"Val MAE CB: {mae_cb:.6f}  CF: {mae_cf:.6f}  Meta: {mae_meta:.6f}  Delta: {mae_meta - mae_cb:+.6f}")
    joblib.dump(booster, out_dir / "meta_model.joblib")

    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    cf_test = pd.read_csv(CF_TEST_PATH, low_memory=False)
    train_reviews = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    user_stats, _ = compute_train_stats(train_reviews)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)

    known_test, cold_test = _build_test_v1(cb_test_sub, cf_test, user_stats, test_reviews)
    known_test = known_test.copy()
    known_test["final_prediction_raw"] = booster.predict(
        known_test[META_FEATURES_V1].to_numpy(np.float32), num_iteration=booster.best_iteration
    ).astype(np.float32)
    cold_test = cold_test.copy()
    cold_test["final_prediction_raw"] = cold_test["cb_prediction_raw"].astype(np.float32)
    submission = _finalize_submission(known_test, cold_test, out_dir)

    importances = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain").tolist()))
    summary = {
        "val_mae_cb_alone": mae_cb, "val_mae_cf_alone": mae_cf, "val_mae_meta": mae_meta,
        "val_delta_vs_cb": mae_meta - mae_cb, "meta_features": META_FEATURES_V1,
        "lgbm_best_iteration": int(booster.best_iteration),
        "lgbm_feature_importances_gain": importances,
        "known_test_rows": int(len(known_test)), "cold_test_rows": int(len(cold_test)),
        "total_test_rows": int(len(submission)),
    }
    (out_dir / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Version 2
# ---------------------------------------------------------------------------

META_FEATURES_V2 = [
    "cf_prediction", "cb_prediction_raw", "history_band_enc",
    "history_count", "history_rating_std",
    "user_mean_rating", "business_mean_rating", "business_review_count", "cf_cb_diff",
]

PARAMS_V2: dict = {
    "objective": "regression_l1", "metric": "mae",
    "num_leaves": 8, "min_data_in_leaf": 100,
    "learning_rate": 0.05, "lambda_l1": 0.5, "lambda_l2": 0.5, "verbose": -1,
}


def _build_val_v2(cb_val, cf_val, user_stats, item_stats, global_mean):
    df = cb_val.merge(cf_val, on="review_id", how="inner")
    df = df.merge(user_stats[["user_id", "user_mean_rating"]], left_on="user", right_on="user_id", how="left")
    df = df.merge(item_stats[["business_id", "business_mean_rating", "business_review_count"]], left_on="item", right_on="business_id", how="left")
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


def _build_test_v2(cb_test_sub, cf_test, user_stats, item_stats, test_reviews, global_mean):
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


def run_v2() -> None:
    out_dir = ARTIFACTS / "meta_lgbm_hybrid_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_reviews_full = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    train_split, _ = temporal_train_validation_split(train_reviews_full, val_size=0.2, timestamp_col="date")
    global_mean = float(train_split["stars"].mean())
    user_stats_split, item_stats_split = compute_train_stats(train_split)
    user_stats_full, item_stats_full = compute_train_stats(train_reviews_full)
    global_mean_full = float(train_reviews_full["stars"].mean())

    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    cf_val = pd.read_csv(CF_VAL_PATH, low_memory=False)
    val_df = _build_val_v2(cb_val, cf_val, user_stats_split, item_stats_split, global_mean)
    print(f"  Val rows: {len(val_df):,}")

    X = val_df[META_FEATURES_V2].to_numpy(np.float32)
    y = val_df["rating"].to_numpy(np.float32)
    mae_cb_round = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(np.float32))))
    mae_cf = float(np.mean(np.abs(y - val_df["cf_prediction"].to_numpy(np.float32))))
    print(f"  Val MAE CB rounded: {mae_cb_round:.6f}  CF: {mae_cf:.6f}")

    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)
    booster = _train_lgbm(X_tr, y_tr, X_es, y_es, META_FEATURES_V2, PARAMS_V2, 500, 30, 25)

    val_preds = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    mae_meta = float(np.mean(np.abs(y - val_preds)))
    print(f"  Val MAE meta v2: {mae_meta:.6f}  Delta: {mae_meta - mae_cb_round:+.6f}")
    joblib.dump(booster, out_dir / "meta_model_v2.joblib")

    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    cf_test = pd.read_csv(CF_TEST_PATH, low_memory=False)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)
    known_test, cold_test = _build_test_v2(cb_test_sub, cf_test, user_stats_full, item_stats_full, test_reviews, global_mean_full)
    known_test = known_test.copy()
    known_test["final_prediction_raw"] = booster.predict(
        known_test[META_FEATURES_V2].to_numpy(np.float32), num_iteration=booster.best_iteration
    ).astype(np.float32)
    cold_test = cold_test.copy()
    cold_test["final_prediction_raw"] = cold_test["cb_prediction_raw"].astype(np.float32)
    submission = _finalize_submission(known_test, cold_test, out_dir)

    importances = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain").tolist()))
    summary = {
        "val_mae_cb_rounded": mae_cb_round, "val_mae_cf": mae_cf, "val_mae_meta_v2": mae_meta,
        "val_delta_vs_cb_rounded": mae_meta - mae_cb_round, "meta_features": META_FEATURES_V2,
        "lgbm_best_iteration": int(booster.best_iteration),
        "lgbm_feature_importances_gain": importances,
        "known_test_rows": int(len(known_test)), "cold_test_rows": int(len(cold_test)),
        "total_test_rows": int(len(submission)),
    }
    (out_dir / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Version 3 (recommended)
# ---------------------------------------------------------------------------

META_FEATURES_V3 = [
    "cb_prediction_raw", "history_band_enc", "history_count", "history_rating_std",
    "user_mean_rating", "business_mean_rating", "business_review_count",
]

PARAMS_V3: dict = {
    "objective": "regression_l1", "metric": "mae",
    "num_leaves": 16, "min_data_in_leaf": 50,
    "learning_rate": 0.03, "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1,
}


def _build_val_v3(cb_val, user_stats, item_stats, global_mean):
    df = cb_val.copy()
    df = df.merge(user_stats[["user_id", "user_mean_rating"]], left_on="user", right_on="user_id", how="left")
    df = df.merge(item_stats[["business_id", "business_mean_rating", "business_review_count"]], left_on="item", right_on="business_id", how="left")
    df["cb_prediction_raw"] = df["deep_prediction_raw"].round().clip(1, 5).astype(np.float32)
    df["history_band_enc"] = encode_band(df["history_band"])
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_mean_rating"] = df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_review_count"] = df["business_review_count"].fillna(0.0).astype(np.float32)
    return df


def _build_test_v3(cb_test_sub, user_stats, item_stats, test_reviews, global_mean):
    lookup = test_reviews[["review_id", "user_id", "business_id", "history_count", "history_rating_std"]].copy()
    df = cb_test_sub.merge(lookup, on="review_id", how="left")
    df = df.merge(user_stats[["user_id", "user_mean_rating"]], on="user_id", how="left")
    df = df.merge(item_stats, on="business_id", how="left")
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_mean_rating"] = df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_review_count"] = df["business_review_count"].fillna(0.0).astype(np.float32)
    df["history_band"] = df["history_count"].apply(lambda c: history_band_from_count(int(c)))
    df["history_band_enc"] = encode_band(df["history_band"])
    df["cb_prediction_raw"] = df["stars"].astype(np.float32)
    cold_mask = df["history_band"].astype(str) == "0"
    return df[~cold_mask].copy(), df[cold_mask].copy()


def run_v3() -> None:
    out_dir = ARTIFACTS / "meta_lgbm_hybrid_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_reviews_full = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    train_split, _ = temporal_train_validation_split(train_reviews_full, val_size=0.2, timestamp_col="date")
    global_mean = float(train_split["stars"].mean())
    user_stats_split, item_stats_split = compute_train_stats(train_split)
    user_stats_full, item_stats_full = compute_train_stats(train_reviews_full)
    global_mean_full = float(train_reviews_full["stars"].mean())

    user_history_full = (
        train_reviews_full.groupby("user_id")["stars"]
        .agg(history_count="count", history_rating_std="std")
        .reset_index()
    )
    user_history_full["history_count"] = user_history_full["history_count"].astype(np.float32)
    user_history_full["history_rating_std"] = user_history_full["history_rating_std"].fillna(0.0).astype(np.float32)

    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    val_df = _build_val_v3(cb_val, user_stats_split, item_stats_split, global_mean)
    print(f"  Val rows: {len(val_df):,}")

    X = val_df[META_FEATURES_V3].to_numpy(np.float32)
    y = val_df["rating"].to_numpy(np.float32)
    mae_cb_raw = float(np.mean(np.abs(y - val_df["deep_prediction_raw"].to_numpy(np.float32))))
    mae_cb_round = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(np.float32))))
    print(f"  Val MAE CB raw: {mae_cb_raw:.6f}  CB rounded: {mae_cb_round:.6f}")

    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)
    booster = _train_lgbm(X_tr, y_tr, X_es, y_es, META_FEATURES_V3, PARAMS_V3, 1000, 50, 50)

    val_preds = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    mae_meta = float(np.mean(np.abs(y - val_preds)))
    print(f"  Val MAE meta v3: {mae_meta:.6f}  Delta: {mae_meta - mae_cb_round:+.6f}")
    importances = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain").tolist()))
    print(f"  Feature importances: {importances}")
    joblib.dump(booster, out_dir / "meta_model_v3.joblib")

    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)
    test_reviews = test_reviews.merge(
        user_history_full[["user_id", "history_count", "history_rating_std"]], on="user_id", how="left"
    )
    test_reviews["history_count"] = test_reviews["history_count"].fillna(0.0).astype(np.float32)
    test_reviews["history_rating_std"] = test_reviews["history_rating_std"].fillna(0.0).astype(np.float32)

    known_test, cold_test = _build_test_v3(cb_test_sub, user_stats_full, item_stats_full, test_reviews, global_mean_full)

    # Re-join history stats for known_test
    history_lookup = test_reviews[["review_id", "history_count", "history_rating_std"]].copy()
    known_test = known_test.drop(columns=["history_count", "history_rating_std"], errors="ignore")
    known_test = known_test.merge(history_lookup, on="review_id", how="left")
    known_test["history_count"] = known_test["history_count"].fillna(0.0).astype(np.float32)
    known_test["history_rating_std"] = known_test["history_rating_std"].fillna(0.0).astype(np.float32)

    known_test = known_test.copy()
    known_test["final_prediction_raw"] = booster.predict(
        known_test[META_FEATURES_V3].to_numpy(np.float32), num_iteration=booster.best_iteration
    ).astype(np.float32)
    cold_test = cold_test.copy()
    cold_test["final_prediction_raw"] = cold_test["cb_prediction_raw"].astype(np.float32)
    submission = _finalize_submission(known_test, cold_test, out_dir)

    summary = {
        "val_mae_cb_raw": mae_cb_raw, "val_mae_cb_rounded": mae_cb_round, "val_mae_meta_v3": mae_meta,
        "val_delta_vs_cb_rounded": mae_meta - mae_cb_round, "meta_features": META_FEATURES_V3,
        "lgbm_best_iteration": int(booster.best_iteration),
        "lgbm_feature_importances_gain": importances,
        "known_test_rows": int(len(known_test)), "cold_test_rows": int(len(cold_test)),
        "total_test_rows": int(len(submission)),
    }
    (out_dir / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Version 4 (recommended) — 19 features: v3 + engagement + user metadata + interactions
# ---------------------------------------------------------------------------
# New vs v3:
#   Group A (free from CB val CSV)  : history_positive_share, history_negative_share,
#                                      cb_pred_fractional_part, correction_hat
#   Group B (test_reviews join)     : review_useful_log1p, review_funny_log1p,
#                                      review_month, review_weekday
#   Group C (usuarios.csv join)     : user_tenure_years, user_elite_any,
#                                      user_total_votes_log1p
#   Group D (computed interactions) : user_business_bias_gap,
#                                      history_count_log1p, business_review_count_log1p
# ---------------------------------------------------------------------------

META_FEATURES_V4 = [
    # core (kept from v3, log-transformed counts)
    "cb_prediction_raw",
    "history_band_enc",
    "history_count_log1p",
    "history_rating_std",
    "user_mean_rating",
    "business_mean_rating",
    "business_review_count_log1p",
    # Group A: already in the CB validation CSV
    "history_positive_share",
    "history_negative_share",
    "cb_pred_fractional_part",
    "correction_hat",
    # Group B: review engagement votes (available in test_reviews.csv, non-leaky)
    "review_useful_log1p",
    "review_funny_log1p",
    "review_month",
    "review_weekday",
    # Group C: user metadata from usuarios.csv (static, non-leaky)
    "user_tenure_years",
    "user_elite_any",
    "user_total_votes_log1p",
    # Group D: derived interactions
    "user_business_bias_gap",
]

PARAMS_V4: dict = {
    "objective": "regression_l1",
    "metric": "mae",
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}


def _load_user_meta(users_path: Path) -> pd.DataFrame:
    """Load user metadata needed for Group C features."""
    users = pd.read_csv(
        users_path,
        usecols=["user_id", "yelping_since", "elite", "useful", "funny", "cool"],
        low_memory=False,
    )
    ref_date = pd.Timestamp("2026-01-01")
    users["user_tenure_years"] = (
        (ref_date - pd.to_datetime(users["yelping_since"], errors="coerce")).dt.days / 365.25
    ).clip(lower=0.0).astype(np.float32)
    users["user_elite_any"] = (users["elite"].fillna("").astype(str).str.strip() != "").astype(np.float32)
    users["user_total_votes_log1p"] = np.log1p(
        users[["useful", "funny", "cool"]].fillna(0.0).sum(axis=1)
    ).astype(np.float32)
    return users[["user_id", "user_tenure_years", "user_elite_any", "user_total_votes_log1p"]]


def _attach_review_engagement(df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Join useful/funny/cool + date features from a reviews dataframe onto df (keyed by review_id)."""
    eng = reviews_df[["review_id", "useful", "funny", "cool", "date"]].copy()
    eng["review_useful_log1p"] = np.log1p(eng["useful"].fillna(0.0)).astype(np.float32)
    eng["review_funny_log1p"] = np.log1p(eng["funny"].fillna(0.0)).astype(np.float32)
    dates = pd.to_datetime(eng["date"], errors="coerce")
    eng["review_month"] = dates.dt.month.fillna(0).astype(np.float32)
    eng["review_weekday"] = dates.dt.weekday.fillna(0).astype(np.float32)
    eng = eng[["review_id", "review_useful_log1p", "review_funny_log1p", "review_month", "review_weekday"]]
    return df.merge(eng, on="review_id", how="left")


def _add_v4_derived(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    """Add Group A (from CB val fields) + Group D (interaction) features."""
    # Group A
    df["history_positive_share"] = df["history_positive_share"].fillna(0.0).astype(np.float32)
    df["history_negative_share"] = df["history_negative_share"].fillna(0.0).astype(np.float32)
    # correction_hat: may not exist in test path (comes from deep router val CSV only)
    if "correction_hat" in df.columns:
        df["correction_hat"] = df["correction_hat"].fillna(0.0).astype(np.float32)
    else:
        df["correction_hat"] = np.float32(0.0)
    # fractional part — how far CB rounded prediction is from its raw value
    df["cb_pred_fractional_part"] = np.abs(
        df["deep_prediction_raw"].fillna(df["cb_prediction_raw"]).astype(np.float32)
        - df["cb_prediction_raw"].astype(np.float32)
    ).astype(np.float32)
    # Group D
    df["history_count_log1p"] = np.log1p(df["history_count"].fillna(0.0)).astype(np.float32)
    df["business_review_count_log1p"] = np.log1p(df["business_review_count"].fillna(0.0)).astype(np.float32)
    df["user_business_bias_gap"] = (
        df["user_mean_rating"].fillna(global_mean).astype(np.float32)
        - df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    ).astype(np.float32)
    # fill Group B fallbacks (0 if join missed)
    for col in ["review_useful_log1p", "review_funny_log1p", "review_month", "review_weekday"]:
        if col not in df.columns:
            df[col] = np.float32(0.0)
        else:
            df[col] = df[col].fillna(0.0).astype(np.float32)
    # fill Group C fallbacks
    for col in ["user_tenure_years", "user_elite_any", "user_total_votes_log1p"]:
        if col not in df.columns:
            df[col] = np.float32(0.0)
        else:
            df[col] = df[col].fillna(0.0).astype(np.float32)
    return df


def _build_val_v4(
    cb_val: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_meta: pd.DataFrame,
    train_reviews_split: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    df = cb_val.copy()
    # v3 base joins
    df = df.merge(user_stats[["user_id", "user_mean_rating"]], left_on="user", right_on="user_id", how="left")
    df = df.merge(item_stats[["business_id", "business_mean_rating", "business_review_count"]], left_on="item", right_on="business_id", how="left")
    df["cb_prediction_raw"] = df["deep_prediction_raw"].round().clip(1, 5).astype(np.float32)
    df["history_band_enc"] = encode_band(df["history_band"])
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_mean_rating"] = df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_review_count"] = df["business_review_count"].fillna(0.0).astype(np.float32)
    # Group C: user meta
    df = df.merge(user_meta, left_on="user", right_on="user_id", how="left")
    # Group B: engagement from train_reviews (val rows are training rows)
    df = _attach_review_engagement(df, train_reviews_split)
    # Group A + D
    df = _add_v4_derived(df, global_mean)
    return df


def _build_test_v4(
    cb_test_sub: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_meta: pd.DataFrame,
    test_reviews: pd.DataFrame,
    global_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = test_reviews[["review_id", "user_id", "business_id", "history_count", "history_rating_std"]].copy()
    df = cb_test_sub.merge(lookup, on="review_id", how="left")
    # Only join user_mean_rating — history_count/history_rating_std already come from lookup
    df = df.merge(user_stats[["user_id", "user_mean_rating"]], on="user_id", how="left")
    df = df.merge(item_stats[["business_id", "business_mean_rating", "business_review_count"]], on="business_id", how="left")
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_mean_rating"] = df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_review_count"] = df["business_review_count"].fillna(0.0).astype(np.float32)
    df["history_band"] = df["history_count"].apply(lambda c: history_band_from_count(int(c)))
    df["history_band_enc"] = encode_band(df["history_band"])
    df["cb_prediction_raw"] = df["stars"].astype(np.float32)
    # deep_prediction_raw not in test path — use cb_prediction_raw as proxy for fractional part calc
    df["deep_prediction_raw"] = df["cb_prediction_raw"]
    df["history_positive_share"] = np.float32(0.0)  # not available at test time
    df["history_negative_share"] = np.float32(0.0)
    # Group C: user meta
    df = df.merge(user_meta, on="user_id", how="left")
    # Group B: engagement from test_reviews
    df = _attach_review_engagement(df, test_reviews)
    # Group A + D
    df = _add_v4_derived(df, global_mean)
    cold_mask = df["history_band"].astype(str) == "0"
    return df[~cold_mask].copy(), df[cold_mask].copy()


def run_v4() -> None:
    out_dir = ARTIFACTS / "meta_lgbm_hybrid_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_reviews_full = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    train_split, _ = temporal_train_validation_split(train_reviews_full, val_size=0.2, timestamp_col="date")
    global_mean = float(train_split["stars"].mean())
    user_stats_split, item_stats_split = compute_train_stats(train_split)
    user_stats_full, item_stats_full = compute_train_stats(train_reviews_full)
    global_mean_full = float(train_reviews_full["stars"].mean())

    user_history_full = (
        train_reviews_full.groupby("user_id")["stars"]
        .agg(history_count="count", history_rating_std="std")
        .reset_index()
    )
    user_history_full["history_count"] = user_history_full["history_count"].astype(np.float32)
    user_history_full["history_rating_std"] = user_history_full["history_rating_std"].fillna(0.0).astype(np.float32)

    print("Loading user metadata (usuarios.csv) ...")
    user_meta = _load_user_meta(DATA_DIR / "usuarios.csv")

    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    # val reviews are in the temporal val split of train_reviews — load for engagement
    val_review_ids = set(cb_val["review_id"].tolist())
    # Pull engagement from the full train_reviews file for matching review_ids
    train_reviews_engagement = train_reviews_full[train_reviews_full["review_id"].isin(val_review_ids)].copy()

    print("Building val meta-feature frame (v4) ...")
    val_df = _build_val_v4(cb_val, user_stats_split, item_stats_split, user_meta, train_reviews_engagement, global_mean)
    print(f"  Val rows: {len(val_df):,}  features: {META_FEATURES_V4}")

    X = val_df[META_FEATURES_V4].to_numpy(np.float32)
    y = val_df["rating"].to_numpy(np.float32)
    mae_cb_raw = float(np.mean(np.abs(y - val_df["deep_prediction_raw"].to_numpy(np.float32))))
    mae_cb_round = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(np.float32))))
    print(f"  Val MAE CB raw: {mae_cb_raw:.6f}  CB rounded: {mae_cb_round:.6f}")

    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)
    booster = _train_lgbm(X_tr, y_tr, X_es, y_es, META_FEATURES_V4, PARAMS_V4, 1000, 50, 50)

    val_preds = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    mae_meta = float(np.mean(np.abs(y - val_preds)))
    print(f"  Val MAE meta v4: {mae_meta:.6f}  Delta vs CB rounded: {mae_meta - mae_cb_round:+.6f}")
    importances = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain").tolist()))
    print(f"  Feature importances (gain): {importances}")
    joblib.dump(booster, out_dir / "meta_model_v4.joblib")

    print("\nBuilding test meta-feature frame (v4) ...")
    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)
    test_reviews = test_reviews.merge(
        user_history_full[["user_id", "history_count", "history_rating_std"]], on="user_id", how="left"
    )
    test_reviews["history_count"] = test_reviews["history_count"].fillna(0.0).astype(np.float32)
    test_reviews["history_rating_std"] = test_reviews["history_rating_std"].fillna(0.0).astype(np.float32)

    known_test, cold_test = _build_test_v4(
        cb_test_sub, user_stats_full, item_stats_full, user_meta, test_reviews, global_mean_full
    )
    # Re-join history stats for known_test (dropped in _build_test_v4 merge)
    history_lookup = test_reviews[["review_id", "history_count", "history_rating_std"]].copy()
    known_test = known_test.drop(columns=["history_count", "history_rating_std"], errors="ignore")
    known_test = known_test.merge(history_lookup, on="review_id", how="left")
    known_test["history_count"] = known_test["history_count"].fillna(0.0).astype(np.float32)
    known_test["history_rating_std"] = known_test["history_rating_std"].fillna(0.0).astype(np.float32)
    known_test["history_count_log1p"] = np.log1p(known_test["history_count"]).astype(np.float32)
    print(f"  Known test: {len(known_test):,}  Cold test: {len(cold_test):,}")

    known_test = known_test.copy()
    known_test["final_prediction_raw"] = booster.predict(
        known_test[META_FEATURES_V4].to_numpy(np.float32), num_iteration=booster.best_iteration
    ).astype(np.float32)
    cold_test = cold_test.copy()
    cold_test["final_prediction_raw"] = cold_test["cb_prediction_raw"].astype(np.float32)
    submission = _finalize_submission(known_test, cold_test, out_dir)

    summary = {
        "val_mae_cb_raw": mae_cb_raw,
        "val_mae_cb_rounded": mae_cb_round,
        "val_mae_meta_v4": mae_meta,
        "val_delta_vs_cb_rounded": mae_meta - mae_cb_round,
        "meta_features": META_FEATURES_V4,
        "lgbm_best_iteration": int(booster.best_iteration),
        "lgbm_feature_importances_gain": importances,
        "known_test_rows": int(len(known_test)),
        "cold_test_rows": int(len(cold_test)),
        "total_test_rows": int(len(submission)),
    }
    (out_dir / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Version 5 — meta covers ALL bands including cold (band 0)
# ---------------------------------------------------------------------------
# Key changes vs v4:
#   - Training set includes cold-user val rows (derived from val_split directly)
#   - Test scoring: meta applied to ALL 414k test rows, no cold bypass
#   - New feature: business_bayesian_prior (Bayesian shrinkage toward global mean, k=5)
#   - Removed: review_funny_log1p, review_month, review_weekday (near-zero gain in v4)
# ---------------------------------------------------------------------------

BAYESIAN_K = 5.0  # pseudo-count for Bayesian business prior

META_FEATURES_V5 = [
    "cb_prediction_raw",
    "history_band_enc",
    "history_count_log1p",
    "history_rating_std",
    "user_mean_rating",
    "business_mean_rating",
    "business_review_count_log1p",
    "business_bayesian_prior",      # new: Bayesian shrinkage toward global mean
    "history_positive_share",
    "history_negative_share",
    "cb_pred_fractional_part",
    "correction_hat",
    "review_useful_log1p",          # kept: modest gain, free at test time
    "user_tenure_years",
    "user_elite_any",
    "user_total_votes_log1p",
    "user_business_bias_gap",
]

PARAMS_V5: dict = {
    "objective": "regression_l1",
    "metric": "mae",
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}


def _add_v5_derived(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    """Add all derived features for v5 (superset of v4, plus bayesian_prior)."""
    df["history_positive_share"] = df["history_positive_share"].fillna(0.0).astype(np.float32)
    df["history_negative_share"] = df["history_negative_share"].fillna(0.0).astype(np.float32)
    if "correction_hat" in df.columns:
        df["correction_hat"] = df["correction_hat"].fillna(0.0).astype(np.float32)
    else:
        df["correction_hat"] = np.float32(0.0)
    df["cb_pred_fractional_part"] = np.abs(
        df["deep_prediction_raw"].fillna(df["cb_prediction_raw"]).astype(np.float32)
        - df["cb_prediction_raw"].astype(np.float32)
    ).astype(np.float32)
    df["history_count_log1p"] = np.log1p(df["history_count"].fillna(0.0)).astype(np.float32)
    brc = df["business_review_count"].fillna(0.0).astype(np.float64)
    bm = df["business_mean_rating"].fillna(global_mean).astype(np.float64)
    df["business_review_count_log1p"] = np.log1p(brc).astype(np.float32)
    df["business_bayesian_prior"] = (
        (brc * bm + BAYESIAN_K * global_mean) / (brc + BAYESIAN_K)
    ).clip(1.0, 5.0).astype(np.float32)
    df["user_business_bias_gap"] = (
        df["user_mean_rating"].fillna(global_mean).astype(np.float32)
        - df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    ).astype(np.float32)
    for col in ["review_useful_log1p"]:
        df[col] = df[col].fillna(0.0).astype(np.float32) if col in df.columns else np.float32(0.0)
    for col in ["user_tenure_years", "user_elite_any", "user_total_votes_log1p"]:
        df[col] = df[col].fillna(0.0).astype(np.float32) if col in df.columns else np.float32(0.0)
    return df


def _build_cold_val_rows(
    val_split: pd.DataFrame,
    train_user_ids: set,
    item_stats: pd.DataFrame,
    user_meta: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    """Build meta-feature rows for cold users (band 0) from the temporal val split."""
    cold = val_split[~val_split["user_id"].isin(train_user_ids)].copy()
    cold = cold.merge(item_stats[["business_id", "business_mean_rating", "business_review_count"]], on="business_id", how="left")
    cold["business_mean_rating"] = cold["business_mean_rating"].fillna(global_mean).astype(np.float32)
    cold["business_review_count"] = cold["business_review_count"].fillna(0.0).astype(np.float32)

    # Bayesian prior as CB prediction for cold users
    brc = cold["business_review_count"].astype(np.float64)
    bm = cold["business_mean_rating"].astype(np.float64)
    cold["deep_prediction_raw"] = ((brc * bm + BAYESIAN_K * global_mean) / (brc + BAYESIAN_K)).clip(1.0, 5.0).astype(np.float32)
    cold["cb_prediction_raw"] = cold["deep_prediction_raw"].round().clip(1, 5).astype(np.float32)

    cold["history_band"] = "0"
    cold["history_band_enc"] = np.float32(0.0)
    cold["history_count"] = np.float32(0.0)
    cold["history_rating_std"] = np.float32(0.0)
    cold["history_positive_share"] = np.float32(0.0)
    cold["history_negative_share"] = np.float32(0.0)
    cold["correction_hat"] = np.float32(0.0)
    cold["user_mean_rating"] = np.float32(global_mean)

    cold = cold.merge(user_meta, on="user_id", how="left")

    # engagement from val reviews (same source as train_reviews)
    eng_cold = cold[["review_id", "useful", "funny", "cool", "date"]].copy()
    eng_cold["review_useful_log1p"] = np.log1p(eng_cold["useful"].fillna(0.0)).astype(np.float32)
    cold = cold.merge(eng_cold[["review_id", "review_useful_log1p"]], on="review_id", how="left")

    cold["rating"] = cold["stars"].astype(np.float32)
    return cold


def _build_val_v5(
    cb_val: pd.DataFrame,
    val_split: pd.DataFrame,
    train_user_ids: set,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_meta: pd.DataFrame,
    train_reviews_full: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    # --- known users: same as v4 ---
    known = cb_val.copy()
    known = known.merge(user_stats[["user_id", "user_mean_rating"]], left_on="user", right_on="user_id", how="left")
    known = known.merge(item_stats[["business_id", "business_mean_rating", "business_review_count"]], left_on="item", right_on="business_id", how="left")
    known["cb_prediction_raw"] = known["deep_prediction_raw"].round().clip(1, 5).astype(np.float32)
    known["history_band_enc"] = encode_band(known["history_band"])
    known["history_count"] = known["history_count"].fillna(0.0).astype(np.float32)
    known["history_rating_std"] = known["history_rating_std"].fillna(0.0).astype(np.float32)
    known["user_mean_rating"] = known["user_mean_rating"].fillna(global_mean).astype(np.float32)
    known["business_mean_rating"] = known["business_mean_rating"].fillna(global_mean).astype(np.float32)
    known["business_review_count"] = known["business_review_count"].fillna(0.0).astype(np.float32)
    known = known.merge(user_meta, left_on="user", right_on="user_id", how="left")
    val_review_ids = set(known["review_id"].tolist())
    engagement = train_reviews_full[train_reviews_full["review_id"].isin(val_review_ids)].copy()
    engagement["review_useful_log1p"] = np.log1p(engagement["useful"].fillna(0.0)).astype(np.float32)
    known = known.merge(engagement[["review_id", "review_useful_log1p"]], on="review_id", how="left")
    known["rating"] = known["rating"].astype(np.float32)
    known = _add_v5_derived(known, global_mean)

    # --- cold users from val_split ---
    cold = _build_cold_val_rows(val_split, train_user_ids, item_stats, user_meta, global_mean)
    cold = _add_v5_derived(cold, global_mean)

    shared_cols = META_FEATURES_V5 + ["rating", "review_id"]
    return pd.concat([known[[c for c in shared_cols if c in known.columns]],
                      cold[[c for c in shared_cols if c in cold.columns]]], ignore_index=True)


def _build_all_test_v5(
    cb_test_sub: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_meta: pd.DataFrame,
    test_reviews: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    """Build meta feature frame for ALL test rows (cold + known). No bypass."""
    lookup = test_reviews[["review_id", "user_id", "business_id", "history_count", "history_rating_std", "useful", "funny", "cool", "date"]].copy()
    df = cb_test_sub.merge(lookup, on="review_id", how="left")
    df = df.merge(user_stats[["user_id", "user_mean_rating"]], on="user_id", how="left")
    df = df.merge(item_stats[["business_id", "business_mean_rating", "business_review_count"]], on="business_id", how="left")
    df["history_count"] = df["history_count"].fillna(0.0).astype(np.float32)
    df["history_rating_std"] = df["history_rating_std"].fillna(0.0).astype(np.float32)
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_mean_rating"] = df["business_mean_rating"].fillna(global_mean).astype(np.float32)
    df["business_review_count"] = df["business_review_count"].fillna(0.0).astype(np.float32)
    df["history_band"] = df["history_count"].apply(lambda c: history_band_from_count(int(c)))
    df["history_band_enc"] = encode_band(df["history_band"])

    # CB prediction: for known rows use stars from submission; for cold rows compute Bayesian prior
    is_cold = df["history_band"].astype(str) == "0"
    brc = df["business_review_count"].astype(np.float64)
    bm = df["business_mean_rating"].astype(np.float64)
    bayesian = ((brc * bm + BAYESIAN_K * global_mean) / (brc + BAYESIAN_K)).clip(1.0, 5.0)
    df["deep_prediction_raw"] = np.where(is_cold, bayesian, df["stars"].astype(np.float64)).astype(np.float32)
    df["cb_prediction_raw"] = df["deep_prediction_raw"].round().clip(1, 5).astype(np.float32)

    df["history_positive_share"] = np.float32(0.0)
    df["history_negative_share"] = np.float32(0.0)
    df["correction_hat"] = np.float32(0.0)

    df = df.merge(user_meta, on="user_id", how="left")
    df["review_useful_log1p"] = np.log1p(df["useful"].fillna(0.0)).astype(np.float32)
    df = _add_v5_derived(df, global_mean)
    return df


def run_v5() -> None:
    out_dir = ARTIFACTS / "meta_lgbm_hybrid_v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_reviews_full = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    train_split, val_split = temporal_train_validation_split(train_reviews_full, val_size=0.2, timestamp_col="date")
    global_mean = float(train_split["stars"].mean())
    global_mean_full = float(train_reviews_full["stars"].mean())
    print(f"  train_split: {len(train_split):,}  val_split: {len(val_split):,}  global_mean: {global_mean:.4f}")

    user_stats_split, item_stats_split = compute_train_stats(train_split)
    user_stats_full, item_stats_full = compute_train_stats(train_reviews_full)
    train_user_ids = set(train_split["user_id"].unique())

    user_history_full = (
        train_reviews_full.groupby("user_id")["stars"]
        .agg(history_count="count", history_rating_std="std")
        .reset_index()
    )
    user_history_full["history_count"] = user_history_full["history_count"].astype(np.float32)
    user_history_full["history_rating_std"] = user_history_full["history_rating_std"].fillna(0.0).astype(np.float32)

    print("Loading user metadata ...")
    user_meta = _load_user_meta(DATA_DIR / "usuarios.csv")

    print("Building val meta-feature frame (v5, all bands) ...")
    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    val_df = _build_val_v5(
        cb_val, val_split, train_user_ids,
        user_stats_split, item_stats_split, user_meta,
        train_reviews_full, global_mean,
    )
    cold_val_count = (val_df["history_band_enc"] == 0).sum()
    known_val_count = (val_df["history_band_enc"] != 0).sum()
    print(f"  Total val rows: {len(val_df):,}  (known: {known_val_count:,}  cold: {cold_val_count:,})")

    X = val_df[META_FEATURES_V5].to_numpy(np.float32)
    y = val_df["rating"].to_numpy(np.float32)
    mae_cb_round = float(np.mean(np.abs(y - val_df["cb_prediction_raw"].to_numpy(np.float32))))
    print(f"  Val MAE CB rounded (all bands): {mae_cb_round:.6f}")

    X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.1, random_state=42)
    booster = _train_lgbm(X_tr, y_tr, X_es, y_es, META_FEATURES_V5, PARAMS_V5, 3000, 100, 100)

    val_preds = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    mae_meta = float(np.mean(np.abs(y - val_preds)))
    # also report MAE split by band for diagnostics
    per_band_mae: dict = {}
    for band_label, mask_fn in [("cold(0)", lambda df: df["history_band_enc"] == 0),
                                  ("known(1+)", lambda df: df["history_band_enc"] != 0)]:
        mask = mask_fn(val_df).to_numpy()
        if mask.sum() > 0:
            band_mae = float(np.mean(np.abs(y[mask] - val_preds[mask])))
            band_cb = float(np.mean(np.abs(y[mask] - val_df["cb_prediction_raw"].to_numpy(np.float32)[mask])))
            per_band_mae[band_label] = {"meta_mae": band_mae, "cb_mae": band_cb, "delta": band_mae - band_cb}
            print(f"  [{band_label}] meta MAE: {band_mae:.6f}  CB rounded: {band_cb:.6f}  delta: {band_mae - band_cb:+.6f}")
    print(f"  Val MAE meta v5 (all bands): {mae_meta:.6f}  Delta vs CB: {mae_meta - mae_cb_round:+.6f}")

    importances = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain").tolist()))
    print(f"  Feature importances (gain): {importances}")
    joblib.dump(booster, out_dir / "meta_model_v5.joblib")

    print("\nBuilding test meta-feature frame (v5, all bands) ...")
    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)
    test_reviews = test_reviews.merge(
        user_history_full[["user_id", "history_count", "history_rating_std"]], on="user_id", how="left"
    )
    test_reviews["history_count"] = test_reviews["history_count"].fillna(0.0).astype(np.float32)
    test_reviews["history_rating_std"] = test_reviews["history_rating_std"].fillna(0.0).astype(np.float32)

    all_test = _build_all_test_v5(cb_test_sub, user_stats_full, item_stats_full, user_meta, test_reviews, global_mean_full)
    print(f"  Test rows (all bands): {len(all_test):,}")

    all_test = all_test.copy()
    all_test["final_prediction_raw"] = booster.predict(
        all_test[META_FEATURES_V5].to_numpy(np.float32), num_iteration=booster.best_iteration
    ).astype(np.float32)
    all_test["stars"] = _round_half_up(np.clip(all_test["final_prediction_raw"].to_numpy(np.float32), 1.0, 5.0))
    submission = all_test[["review_id", "stars"]].copy()
    submission.to_csv(out_dir / "submission.csv", index=False)
    print(f"Submission saved: {len(submission):,} rows")
    print(f"Stars distribution:\n{submission['stars'].value_counts().sort_index().to_string()}")

    summary = {
        "val_mae_cb_rounded_all_bands": mae_cb_round,
        "val_mae_meta_v5": mae_meta,
        "val_delta_vs_cb_rounded": mae_meta - mae_cb_round,
        "val_known_rows": int(known_val_count),
        "val_cold_rows": int(cold_val_count),
        "val_per_band_mae": per_band_mae,
        "meta_features": META_FEATURES_V5,
        "lgbm_best_iteration": int(booster.best_iteration),
        "lgbm_feature_importances_gain": importances,
        "total_test_rows": int(len(submission)),
    }
    (out_dir / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Version 6 — two dedicated models: known-user meta (v4 retrained) + cold-user meta
# ---------------------------------------------------------------------------
# Strategy:
#   - v5 joint training hurt known users (+0.037 MAE) because cold rows (67% of training
#     data) dominated tree splits toward predicting business priors for everyone.
#   - Solution: train two completely separate LightGBM models routed by history_count==0.
#     * Known model: trained only on known val rows (64k) — equivalent to v4 but with
#       the cleaner v5-style feature set (drops near-zero features from v4).
#     * Cold model: trained only on cold val rows (128k) — uses business-only features
#       (history features are all-zero and add noise for cold users).
# ---------------------------------------------------------------------------

# Features for known users (have history — full set minus near-zero v4 features)
META_FEATURES_V6_KNOWN = [
    "cb_prediction_raw",
    "history_band_enc",
    "history_count_log1p",
    "history_rating_std",
    "user_mean_rating",
    "business_mean_rating",
    "business_review_count_log1p",
    "business_bayesian_prior",
    "history_positive_share",
    "history_negative_share",
    "cb_pred_fractional_part",
    "correction_hat",
    "review_useful_log1p",
    "user_tenure_years",
    "user_elite_any",
    "user_total_votes_log1p",
    "user_business_bias_gap",
]

# Features for cold users (no history — only business signals + user meta from profile)
# Drop history-derived features (all zero = no signal, only noise)
META_FEATURES_V6_COLD = [
    "business_mean_rating",
    "business_review_count_log1p",
    "business_bayesian_prior",
    "user_tenure_years",
    "user_elite_any",
    "user_total_votes_log1p",
    "review_useful_log1p",
]

PARAMS_V6: dict = {
    "objective": "regression_l1",
    "metric": "mae",
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "learning_rate": 0.03,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}

# Fewer leaves for cold model (simpler signal, avoid overfitting on business features)
PARAMS_V6_COLD: dict = {
    "objective": "regression_l1",
    "metric": "mae",
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbose": -1,
}


def run_v6() -> None:
    """Train two separate meta-models: one for known users, one for cold users."""
    out_dir = ARTIFACTS / "meta_lgbm_hybrid_v6"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_reviews_full = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    train_split, val_split = temporal_train_validation_split(train_reviews_full, val_size=0.2, timestamp_col="date")
    global_mean = float(train_split["stars"].mean())
    global_mean_full = float(train_reviews_full["stars"].mean())
    print(f"  train_split: {len(train_split):,}  val_split: {len(val_split):,}  global_mean: {global_mean:.4f}")

    user_stats_split, item_stats_split = compute_train_stats(train_split)
    user_stats_full, item_stats_full = compute_train_stats(train_reviews_full)
    train_user_ids = set(train_split["user_id"].unique())

    user_history_full = (
        train_reviews_full.groupby("user_id")["stars"]
        .agg(history_count="count", history_rating_std="std")
        .reset_index()
    )
    user_history_full["history_count"] = user_history_full["history_count"].astype(np.float32)
    user_history_full["history_rating_std"] = user_history_full["history_rating_std"].fillna(0.0).astype(np.float32)

    print("Loading user metadata ...")
    user_meta = _load_user_meta(DATA_DIR / "usuarios.csv")

    # ---- Build val frames ----
    print("\nBuilding val meta-feature frame (known users) ...")
    cb_val = pd.read_csv(CB_VAL_PATH, low_memory=False)
    val_df = _build_val_v5(
        cb_val, val_split, train_user_ids,
        user_stats_split, item_stats_split, user_meta,
        train_reviews_full, global_mean,
    )
    known_val = val_df[val_df["history_band_enc"] != 0].copy()
    cold_val = val_df[val_df["history_band_enc"] == 0].copy()
    print(f"  Known val rows: {len(known_val):,}  Cold val rows: {len(cold_val):,}")

    # ---- Train known-user model ----
    print("\nTraining known-user model ...")
    Xk = known_val[META_FEATURES_V6_KNOWN].to_numpy(np.float32)
    yk = known_val["rating"].to_numpy(np.float32)
    mae_cb_known = float(np.mean(np.abs(yk - known_val["cb_prediction_raw"].to_numpy(np.float32))))
    print(f"  CB rounded MAE (known): {mae_cb_known:.6f}")
    Xk_tr, Xk_es, yk_tr, yk_es = train_test_split(Xk, yk, test_size=0.1, random_state=42)
    booster_known = _train_lgbm(Xk_tr, yk_tr, Xk_es, yk_es, META_FEATURES_V6_KNOWN, PARAMS_V6, 2000, 100, 200)
    known_preds = booster_known.predict(Xk, num_iteration=booster_known.best_iteration).astype(np.float32)
    mae_known = float(np.mean(np.abs(yk - known_preds)))
    print(f"  Known val MAE: {mae_known:.6f}  delta vs CB: {mae_known - mae_cb_known:+.6f}")
    importances_known = dict(zip(booster_known.feature_name(), booster_known.feature_importance(importance_type="gain").tolist()))
    joblib.dump(booster_known, out_dir / "meta_model_v6_known.joblib")

    # ---- Train cold-user model ----
    print("\nTraining cold-user model ...")
    Xc = cold_val[META_FEATURES_V6_COLD].to_numpy(np.float32)
    yc = cold_val["rating"].to_numpy(np.float32)
    mae_cb_cold = float(np.mean(np.abs(yc - cold_val["cb_prediction_raw"].to_numpy(np.float32))))
    print(f"  CB rounded MAE (cold): {mae_cb_cold:.6f}")
    Xc_tr, Xc_es, yc_tr, yc_es = train_test_split(Xc, yc, test_size=0.1, random_state=42)
    booster_cold = _train_lgbm(Xc_tr, yc_tr, Xc_es, yc_es, META_FEATURES_V6_COLD, PARAMS_V6_COLD, 2000, 150, 200)
    cold_preds = booster_cold.predict(Xc, num_iteration=booster_cold.best_iteration).astype(np.float32)
    mae_cold = float(np.mean(np.abs(yc - cold_preds)))
    print(f"  Cold val MAE: {mae_cold:.6f}  delta vs CB: {mae_cold - mae_cb_cold:+.6f}")
    importances_cold = dict(zip(booster_cold.feature_name(), booster_cold.feature_importance(importance_type="gain").tolist()))
    joblib.dump(booster_cold, out_dir / "meta_model_v6_cold.joblib")

    # ---- Combined val MAE ----
    all_val_n = len(known_val) + len(cold_val)
    combined_mae = (len(known_val) * mae_known + len(cold_val) * mae_cold) / all_val_n
    combined_cb = (len(known_val) * mae_cb_known + len(cold_val) * mae_cb_cold) / all_val_n
    print(f"\nCombined val MAE: {combined_mae:.6f}  CB: {combined_cb:.6f}  delta: {combined_mae - combined_cb:+.6f}")

    # ---- Build test frames ----
    print("\nBuilding test meta-feature frame (v6, all bands) ...")
    cb_test_sub = pd.read_csv(CB_TEST_SUBMISSION_PATH, low_memory=False)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)
    test_reviews = test_reviews.merge(
        user_history_full[["user_id", "history_count", "history_rating_std"]], on="user_id", how="left"
    )
    test_reviews["history_count"] = test_reviews["history_count"].fillna(0.0).astype(np.float32)
    test_reviews["history_rating_std"] = test_reviews["history_rating_std"].fillna(0.0).astype(np.float32)

    all_test = _build_all_test_v5(cb_test_sub, user_stats_full, item_stats_full, user_meta, test_reviews, global_mean_full)
    print(f"  Test rows total: {len(all_test):,}")

    is_cold_test = all_test["history_band_enc"] == 0
    known_test = all_test[~is_cold_test].copy()
    cold_test = all_test[is_cold_test].copy()
    print(f"  Known: {len(known_test):,}  Cold: {len(cold_test):,}")

    known_test["final_prediction_raw"] = booster_known.predict(
        known_test[META_FEATURES_V6_KNOWN].to_numpy(np.float32), num_iteration=booster_known.best_iteration
    ).astype(np.float32)
    cold_test["final_prediction_raw"] = booster_cold.predict(
        cold_test[META_FEATURES_V6_COLD].to_numpy(np.float32), num_iteration=booster_cold.best_iteration
    ).astype(np.float32)

    combined_test = pd.concat([known_test[["review_id", "final_prediction_raw"]],
                                cold_test[["review_id", "final_prediction_raw"]]], ignore_index=True)
    combined_test["stars"] = _round_half_up(np.clip(combined_test["final_prediction_raw"].to_numpy(np.float32), 1.0, 5.0))
    submission = combined_test[["review_id", "stars"]].sort_values("review_id").reset_index(drop=True)
    submission.to_csv(out_dir / "submission.csv", index=False)
    print(f"Submission saved: {len(submission):,} rows")
    print(f"Stars distribution:\n{submission['stars'].value_counts().sort_index().to_string()}")

    summary = {
        "strategy": "two dedicated models: known-user (v6_known) + cold-user (v6_cold)",
        "val_mae_known": mae_known,
        "val_mae_cold": mae_cold,
        "val_mae_combined": combined_mae,
        "val_cb_mae_known": mae_cb_known,
        "val_cb_mae_cold": mae_cb_cold,
        "val_cb_mae_combined": combined_cb,
        "val_delta_combined": combined_mae - combined_cb,
        "val_known_rows": int(len(known_val)),
        "val_cold_rows": int(len(cold_val)),
        "lgbm_best_iteration_known": int(booster_known.best_iteration),
        "lgbm_best_iteration_cold": int(booster_cold.best_iteration),
        "meta_features_known": META_FEATURES_V6_KNOWN,
        "meta_features_cold": META_FEATURES_V6_COLD,
        "lgbm_importances_known": importances_known,
        "lgbm_importances_cold": importances_cold,
        "known_test_rows": int(len(known_test)),
        "cold_test_rows": int(len(cold_test)),
        "total_test_rows": int(len(submission)),
    }
    (out_dir / "validation_mae.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a meta-LightGBM stacker on CB predictions."
    )
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=6,
        help="Stacker version: 1=CB+CF, 2=CB+CF+bias, 3=CB+bias only, 4=v3+engagement+user_meta, 5=all-bands joint (deprecated), 6=two dedicated models known+cold (recommended). Default: 6.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Running meta-LightGBM version {args.version} ...")
    if args.version == 1:
        run_v1()
    elif args.version == 2:
        run_v2()
    elif args.version == 3:
        run_v3()
    elif args.version == 4:
        run_v4()
    elif args.version == 5:
        run_v5()
    else:
        run_v6()


if __name__ == "__main__":
    main()
