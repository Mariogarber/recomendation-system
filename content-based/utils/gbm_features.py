from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .frozen_embedding_regression import FrozenEmbeddingBundle


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
    forced_new_user_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the GBM feature matrix for a frame with columns:
    user, item, timestamp, useful, funny, cool.

    All rows are kept. New users are defined against the provided train-derived
    priors, not against bundle membership. This matters because a leaky bundle
    may already contain metadata-only embeddings for test users.

    Rows marked as new-user naturally or through forced_new_user_mask mask the
    user-side features so the downstream model can learn a true cold-start
    fallback.

    Returns (matrix of shape [n_rows, n_features], list of feature names).
    """
    n = len(frame)
    embedding_dim = bundle.user_embeddings.shape[1]
    users = frame["user"].to_numpy()
    items = frame["item"].to_numpy()

    user_to_idx = {uid: i for i, uid in enumerate(bundle.user_ids.to_numpy())}
    biz_to_idx = {bid: i for i, bid in enumerate(bundle.business_ids.to_numpy())}

    user_indices = np.array([user_to_idx.get(u, -1) for u in users], dtype=np.int64)
    biz_indices = np.array([biz_to_idx.get(b, -1) for b in items], dtype=np.int64)

    known_train_user = np.array([u in priors.user_count for u in users], dtype=bool)
    if forced_new_user_mask is not None:
        forced_new_user_mask = np.asarray(forced_new_user_mask, dtype=bool)
        if forced_new_user_mask.shape != (n,):
            raise ValueError(
                "forced_new_user_mask must have shape (n_rows,), "
                f"received {forced_new_user_mask.shape!r} for {n} rows."
            )
        known_train_user = known_train_user & ~forced_new_user_mask
    is_new_user = (~known_train_user).astype(np.float32)

    user_emb = np.zeros((n, embedding_dim), dtype=np.float32)
    known_user = (user_indices >= 0) & known_train_user
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

    history_count = np.array(
        [priors.user_count.get(u, 0) if is_known else 0 for u, is_known in zip(users, known_train_user)],
        dtype=np.float32,
    )
    history_log1p = np.log1p(history_count)

    user_mean = np.array(
        [priors.user_mean.get(u, priors.global_mean) if is_known else priors.global_mean for u, is_known in zip(users, known_train_user)],
        dtype=np.float32,
    )
    user_std = np.array(
        [priors.user_std.get(u, 0.0) if is_known else 0.0 for u, is_known in zip(users, known_train_user)],
        dtype=np.float32,
    )
    biz_mean = np.array([priors.business_mean.get(b, priors.global_mean) for b in items], dtype=np.float32)
    biz_std = np.array([priors.business_std.get(b, 0.0) for b in items], dtype=np.float32)
    biz_count = np.array([priors.business_count.get(b, 0) for b in items], dtype=np.float32)
    global_mean_arr = np.full(n, priors.global_mean, dtype=np.float32)
    mean_rating_gap = user_mean - biz_mean

    rc_raw = _build_review_context_matrix(frame, review_context_min_timestamp)
    rc_scaled = ((rc_raw - review_context_means) / review_context_stds).astype(np.float32)

    matrix = np.column_stack([
        user_emb,               # embedding_dim
        biz_emb,                # embedding_dim
        product,                # embedding_dim
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
