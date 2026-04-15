"""
Train a collaborative filtering model on the Yelp content-based training data.

Version 1: Global mean + per-user bias + per-item bias (CFBiasModel).
Version 2: Bias model + TruncatedSVD on bias residuals (CFSVDModel).

Select with --version 1|2 (default: 2).

Outputs to artifacts/cf_meta_model_v{version}/:
  v1:
    - cf_model.joblib, cf_val_predictions.csv, cf_test_predictions.csv, train_summary.json
  v2:
    - cf_model_v2.joblib, cf_val_predictions_v2.csv, cf_test_predictions_v2.csv, train_summary_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from utils.split import temporal_train_validation_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
CB_VAL_PRED_PATH = ARTIFACTS_DIR / "known_user_deep_router_v2_eval_v3" / "known_user_deep_validation_predictions.csv"

N_COMPONENTS = 50
CLIP_LO, CLIP_HI = 1.0, 5.0


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class CFBiasModel:
    """Global mean + per-user bias + per-item bias collaborative filter."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.user_bias_: dict[str, float] = {}
        self.item_bias_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "CFBiasModel":
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


class CFSVDModel:
    """
    Bias model + low-rank SVD on residuals.
    r_hat(u, i) = clip(mu + b_u + b_i + U[u] @ Vt[:, i], 1, 5)
    Falls back to mu + b_u + b_i when user or item is unknown.
    """

    def __init__(self, n_components: int = N_COMPONENTS, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.global_mean_: float = 0.0
        self.user_bias_: dict[str, float] = {}
        self.item_bias_: dict[str, float] = {}
        self.user2idx_: dict[str, int] = {}
        self.item2idx_: dict[str, int] = {}
        self.U_: np.ndarray | None = None
        self.Vt_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "CFSVDModel":
        self.global_mean_ = float(df["stars"].mean())
        user_means = df.groupby("user_id")["stars"].mean()
        self.user_bias_ = (user_means - self.global_mean_).to_dict()
        item_means = df.groupby("business_id")["stars"].mean()
        self.item_bias_ = (item_means - self.global_mean_).to_dict()

        users = df["user_id"].unique()
        items = df["business_id"].unique()
        self.user2idx_ = {u: i for i, u in enumerate(users)}
        self.item2idx_ = {i: j for j, i in enumerate(items)}

        residuals = (
            df["stars"].values
            - self.global_mean_
            - df["user_id"].map(self.user_bias_).fillna(0.0).values
            - df["business_id"].map(self.item_bias_).fillna(0.0).values
        ).astype(np.float32)

        u_idx = df["user_id"].map(self.user2idx_).values.astype(np.int32)
        i_idx = df["business_id"].map(self.item2idx_).values.astype(np.int32)
        R = csr_matrix(
            (residuals, (u_idx, i_idx)),
            shape=(len(users), len(items)),
            dtype=np.float32,
        )

        print(f"  Fitting TruncatedSVD (k={self.n_components}) on {R.shape}, nnz={R.nnz:,} ...")
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state, n_iter=5)
        self.U_ = svd.fit_transform(R).astype(np.float32)
        self.Vt_ = svd.components_.astype(np.float32)
        explained = svd.explained_variance_ratio_.sum()
        print(f"  SVD explained variance ratio (top {self.n_components}): {explained:.4f}")
        return self

    def predict_batch(self, user_ids: pd.Series, business_ids: pd.Series) -> np.ndarray:
        u_idx = np.array([self.user2idx_.get(u, -1) for u in user_ids], dtype=np.int32)
        i_idx = np.array([self.item2idx_.get(i, -1) for i in business_ids], dtype=np.int32)
        preds = np.array(
            [self.global_mean_ + self.user_bias_.get(u, 0.0) + self.item_bias_.get(i, 0.0)
             for u, i in zip(user_ids, business_ids)],
            dtype=np.float32,
        )
        known = (u_idx >= 0) & (i_idx >= 0)
        if known.sum() > 0 and self.U_ is not None and self.Vt_ is not None:
            U_sub = self.U_[u_idx[known]]
            Vt_sub = self.Vt_[:, i_idx[known]].T
            preds[known] += np.sum(U_sub * Vt_sub, axis=1).astype(np.float32)
        return np.clip(preds, CLIP_LO, CLIP_HI)


# ---------------------------------------------------------------------------
# Version-specific run logic
# ---------------------------------------------------------------------------

def run_v1(train_reviews: pd.DataFrame, cb_val: pd.DataFrame, test_reviews: pd.DataFrame) -> None:
    out_dir = ARTIFACTS_DIR / "cf_meta_model_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Applying temporal split ...")
    train_split, val_split = temporal_train_validation_split(
        train_reviews, val_size=0.2, timestamp_col="date"
    )
    print(f"  train_split: {len(train_split):,} | val_split: {len(val_split):,}")

    print("Training CF bias model ...")
    model = CFBiasModel().fit(train_split)
    print(f"  global_mean={model.global_mean_:.4f}  users={len(model.user_bias_):,}  items={len(model.item_bias_):,}")
    joblib.dump(model, out_dir / "cf_model.joblib")

    print("Generating CF val predictions ...")
    cb_val = cb_val.copy()
    cb_val["cf_prediction"] = model.predict_df(cb_val, user_col="user", item_col="item")
    cb_val[["review_id", "cf_prediction"]].to_csv(out_dir / "cf_val_predictions.csv", index=False)

    val_cf_mae = float(np.mean(np.abs(cb_val["rating"].to_numpy(np.float32) - cb_val["cf_prediction"].to_numpy(np.float32))))
    val_cb_mae = float(np.mean(np.abs(cb_val["rating"].to_numpy(np.float32) - cb_val["deep_prediction_raw"].to_numpy(np.float32))))
    print(f"  CF val MAE (known users): {val_cf_mae:.6f}")
    print(f"  CB val MAE (known users): {val_cb_mae:.6f}")

    print("Generating CF test predictions ...")
    test_reviews = test_reviews.copy()
    test_reviews["cf_prediction"] = model.predict_df(test_reviews, user_col="user_id", item_col="business_id")
    test_reviews[["review_id", "cf_prediction"]].to_csv(out_dir / "cf_test_predictions.csv", index=False)

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
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done v1. Summary:", json.dumps(summary, indent=2))


def run_v2(train_reviews: pd.DataFrame, cb_val: pd.DataFrame, test_reviews: pd.DataFrame) -> None:
    out_dir = ARTIFACTS_DIR / "cf_meta_model_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Applying temporal split ...")
    train_split, val_split = temporal_train_validation_split(
        train_reviews, val_size=0.2, timestamp_col="date"
    )
    print(f"  train_split: {len(train_split):,} | val_split: {len(val_split):,}")

    print("Training CF SVD model ...")
    model = CFSVDModel(n_components=N_COMPONENTS).fit(train_split)
    print(f"  global_mean={model.global_mean_:.4f}  users={len(model.user2idx_):,}  items={len(model.item2idx_):,}")
    joblib.dump(model, out_dir / "cf_model_v2.joblib")

    print("Generating CF v2 val predictions ...")
    cf_v2_val = model.predict_batch(cb_val["user"], cb_val["item"])

    cf_v1_path = ARTIFACTS_DIR / "cf_meta_model_v1" / "cf_val_predictions.csv"
    y_val = cb_val["rating"].values.astype(np.float32)
    mae_v2 = float(np.mean(np.abs(y_val - cf_v2_val)))
    print(f"  CF v2 bias+SVD  MAE : {mae_v2:.6f}")
    if cf_v1_path.exists():
        cf_v1_val = pd.read_csv(cf_v1_path)["cf_prediction"].values
        mae_v1 = float(np.mean(np.abs(y_val - cf_v1_val)))
        print(f"  CF v1 bias-only MAE : {mae_v1:.6f}")
    pd.DataFrame({"review_id": cb_val["review_id"], "cf_prediction": cf_v2_val}).to_csv(
        out_dir / "cf_val_predictions_v2.csv", index=False
    )

    print("Generating CF v2 test predictions ...")
    cf_v2_test = model.predict_batch(test_reviews["user_id"], test_reviews["business_id"])
    pd.DataFrame({"review_id": test_reviews["review_id"], "cf_prediction": cf_v2_test}).to_csv(
        out_dir / "cf_test_predictions_v2.csv", index=False
    )

    summary = {
        "n_components": N_COMPONENTS,
        "global_mean": float(model.global_mean_),
        "known_users": len(model.user2idx_),
        "known_items": len(model.item2idx_),
        "val_cf_v2_mae": mae_v2,
        "test_rows": len(test_reviews),
    }
    (out_dir / "train_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print("Done v2. Summary:", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a collaborative filtering model (v1: bias-only, v2: bias+SVD).")
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Model version: 1 = bias-only (CFBiasModel), 2 = bias+SVD (CFSVDModel). Default: 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data (version={args.version}) ...")
    train_reviews = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)
    cb_val = pd.read_csv(CB_VAL_PRED_PATH, low_memory=False)
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)
    print(f"  train: {len(train_reviews):,}  cb_val: {len(cb_val):,}  test: {len(test_reviews):,}")

    if args.version == 1:
        run_v1(train_reviews, cb_val, test_reviews)
    else:
        run_v2(train_reviews, cb_val, test_reviews)


if __name__ == "__main__":
    main()
