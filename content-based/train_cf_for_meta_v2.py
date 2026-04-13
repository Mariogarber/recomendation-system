"""
Train a bias + TruncatedSVD collaborative filter on the Yelp content-based
training data and generate predictions for val known-user rows and test rows.

Improvement over v1:
  - Adds latent factors via TruncatedSVD on the bias-residual matrix
  - Captures user-item interaction patterns beyond simple means

Outputs to artifacts/cf_meta_model_v2/:
  - cf_model_v2.joblib          : trained CFSVDModel
  - cf_val_predictions_v2.csv   : review_id, cf_prediction
  - cf_test_predictions_v2.csv  : review_id, cf_prediction
  - train_summary_v2.json       : val MAE comparison v1 vs v2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.split import temporal_train_validation_split

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
CB_VAL_PRED_PATH = ARTIFACTS_DIR / "known_user_deep_router_v2_eval_v3" / "known_user_deep_validation_predictions.csv"
OUT_DIR = ARTIFACTS_DIR / "cf_meta_model_v2"

N_COMPONENTS = 50
CLIP_LO, CLIP_HI = 1.0, 5.0


class CFSVDModel:
    """
    Bias model + low-rank SVD on residuals.
    r_hat(u, i) = clip(mu + b_u + b_i + U[u] @ Vt[:, i], 1, 5)
    Falls back to mu + b_u + b_i when user or item is unknown.
    """

    def __init__(self, n_components: int = 50, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        # learned
        self.global_mean_: float = 0.0
        self.user_bias_: dict[str, float] = {}
        self.item_bias_: dict[str, float] = {}
        self.user2idx_: dict[str, int] = {}
        self.item2idx_: dict[str, int] = {}
        self.U_: np.ndarray | None = None   # shape (n_users, n_components)
        self.Vt_: np.ndarray | None = None  # shape (n_components, n_items)

    def fit(self, df: pd.DataFrame) -> "CFSVDModel":
        """df must have columns: user_id, business_id, stars."""
        self.global_mean_ = float(df["stars"].mean())

        user_means = df.groupby("user_id")["stars"].mean()
        self.user_bias_ = (user_means - self.global_mean_).to_dict()

        item_means = df.groupby("business_id")["stars"].mean()
        self.item_bias_ = (item_means - self.global_mean_).to_dict()

        # Build index maps
        users = df["user_id"].unique()
        items = df["business_id"].unique()
        self.user2idx_ = {u: i for i, u in enumerate(users)}
        self.item2idx_ = {i: j for j, i in enumerate(items)}

        # Bias-residual: r - mu - b_u - b_i
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

        print(f"  Fitting TruncatedSVD (k={self.n_components}) on {R.shape} matrix, nnz={R.nnz:,} ...")
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state, n_iter=5)
        self.U_ = svd.fit_transform(R).astype(np.float32)            # (n_users, k)
        self.Vt_ = svd.components_.astype(np.float32)                # (k, n_items)
        explained = svd.explained_variance_ratio_.sum()
        print(f"  SVD explained variance ratio (top {self.n_components}): {explained:.4f}")
        return self

    def predict_batch(self, user_ids: pd.Series, business_ids: pd.Series) -> np.ndarray:
        u_idx = np.array([self.user2idx_.get(u, -1) for u in user_ids], dtype=np.int32)
        i_idx = np.array([self.item2idx_.get(i, -1) for i in business_ids], dtype=np.int32)

        # Start with bias prediction
        preds = np.array(
            [self.global_mean_ + self.user_bias_.get(u, 0.0) + self.item_bias_.get(i, 0.0)
             for u, i in zip(user_ids, business_ids)],
            dtype=np.float32,
        )

        # Add SVD latent factor where both user and item are known
        known = (u_idx >= 0) & (i_idx >= 0)
        if known.sum() > 0 and self.U_ is not None and self.Vt_ is not None:
            U_sub = self.U_[u_idx[known]]          # (n_known, k)
            Vt_sub = self.Vt_[:, i_idx[known]].T   # (n_known, k)
            preds[known] += np.sum(U_sub * Vt_sub, axis=1).astype(np.float32)

        return np.clip(preds, CLIP_LO, CLIP_HI)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train_reviews.csv ...")
    train_reviews = pd.read_csv(DATA_DIR / "train_reviews.csv", low_memory=False)

    print("Applying temporal split (val_size=0.2) ...")
    train_split, val_split = temporal_train_validation_split(
        train_reviews, val_size=0.2, timestamp_col="date"
    )
    print(f"  train_split: {len(train_split):,} | val_split: {len(val_split):,}")

    print("Training CF SVD model ...")
    model = CFSVDModel(n_components=N_COMPONENTS).fit(train_split)
    print(f"  global_mean={model.global_mean_:.4f}  users={len(model.user2idx_):,}  items={len(model.item2idx_):,}")

    print("Saving model ...")
    joblib.dump(model, OUT_DIR / "cf_model_v2.joblib")

    # --- Val predictions ---
    print("Loading CB val predictions ...")
    cb_val = pd.read_csv(CB_VAL_PRED_PATH, low_memory=False)

    print("Generating CF v2 val predictions ...")
    cf_v2_val = model.predict_batch(cb_val["user"], cb_val["item"])
    cf_v1_val = pd.read_csv(ARTIFACTS_DIR / "cf_meta_model_v1" / "cf_val_predictions.csv")["cf_prediction"].values

    y_val = cb_val["rating"].values.astype(np.float32)
    mae_v1 = float(np.mean(np.abs(y_val - cf_v1_val)))
    mae_v2 = float(np.mean(np.abs(y_val - cf_v2_val)))
    print(f"  CF v1 bias-only MAE : {mae_v1:.6f}")
    print(f"  CF v2 bias+SVD  MAE : {mae_v2:.6f}")

    out_val = pd.DataFrame({"review_id": cb_val["review_id"], "cf_prediction": cf_v2_val})
    out_val.to_csv(OUT_DIR / "cf_val_predictions_v2.csv", index=False)

    # --- Test predictions ---
    print("Loading test_reviews.csv ...")
    test_reviews = pd.read_csv(DATA_DIR / "test_reviews.csv", low_memory=False)

    print("Generating CF v2 test predictions ...")
    cf_v2_test = model.predict_batch(test_reviews["user_id"], test_reviews["business_id"])
    out_test = pd.DataFrame({"review_id": test_reviews["review_id"], "cf_prediction": cf_v2_test})
    out_test.to_csv(OUT_DIR / "cf_test_predictions_v2.csv", index=False)

    summary = {
        "n_components": N_COMPONENTS,
        "global_mean": float(model.global_mean_),
        "known_users": len(model.user2idx_),
        "known_items": len(model.item2idx_),
        "val_cf_v1_mae": mae_v1,
        "val_cf_v2_mae": mae_v2,
        "val_cf_improvement": mae_v1 - mae_v2,
        "test_rows": len(test_reviews),
    }
    (OUT_DIR / "train_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print("Done. Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
