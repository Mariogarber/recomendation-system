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
