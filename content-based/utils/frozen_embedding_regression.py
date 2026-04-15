from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import canonicalize_reviews


TABULAR_SUFFIXES = (".csv", ".parquet")


class FrozenEmbeddingArtifactError(RuntimeError):
    pass


@dataclass(slots=True)
class FrozenEmbeddingBundle:
    root: Path
    user_ids: pd.Series
    business_ids: pd.Series
    user_embeddings: np.ndarray
    business_embeddings: np.ndarray
    user_table: pd.DataFrame
    summary: dict[str, Any]


def read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise FrozenEmbeddingArtifactError(f"Unsupported tabular extension: {path}")


def resolve_tabular_path(directory: Path, stem: str) -> Path | None:
    for suffix in TABULAR_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_dense_npz(path: Path) -> np.ndarray:
    return np.load(path)["embeddings"].astype(np.float32)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_frozen_embedding_bundle(root: str | Path) -> FrozenEmbeddingBundle:
    root = Path(root)
    deep_dir = root / "user_deep_repr"
    if not deep_dir.exists():
        raise FrozenEmbeddingArtifactError(f"Missing user_deep_repr under {root}")

    user_ids_path = resolve_tabular_path(deep_dir, "user_deep_ids")
    business_ids_path = resolve_tabular_path(deep_dir, "business_deep_ids")
    user_table_path = resolve_tabular_path(deep_dir, "user_deep_clean_table")
    summary_path = deep_dir / "user_deep_summary.json"
    user_embeddings_path = deep_dir / "user_deep_features.npz"
    business_embeddings_path = deep_dir / "business_deep_features.npz"

    missing = [
        label
        for label, path in {
            "user_ids": user_ids_path,
            "business_ids": business_ids_path,
            "user_table": user_table_path,
            "summary": summary_path if summary_path.exists() else None,
            "user_embeddings": user_embeddings_path if user_embeddings_path.exists() else None,
            "business_embeddings": business_embeddings_path if business_embeddings_path.exists() else None,
        }.items()
        if path is None
    ]
    if missing:
        raise FrozenEmbeddingArtifactError(f"Incomplete embedding bundle under {root}: missing {', '.join(missing)}")

    user_ids_df = read_tabular_file(user_ids_path)
    business_ids_df = read_tabular_file(business_ids_path)
    user_table = read_tabular_file(user_table_path)

    if "user_id" not in user_ids_df.columns or "user_id" not in user_table.columns:
        raise FrozenEmbeddingArtifactError(f"Bundle under {root} is missing user_id columns.")
    if "business_id" not in business_ids_df.columns:
        raise FrozenEmbeddingArtifactError(f"Bundle under {root} is missing business_id columns.")

    return FrozenEmbeddingBundle(
        root=root,
        user_ids=user_ids_df["user_id"].reset_index(drop=True),
        business_ids=business_ids_df["business_id"].reset_index(drop=True),
        user_embeddings=load_dense_npz(user_embeddings_path),
        business_embeddings=load_dense_npz(business_embeddings_path),
        user_table=user_table,
        summary=load_json(summary_path),
    )


def build_review_interaction_frame(reviews: pd.DataFrame) -> pd.DataFrame:
    canonical = canonicalize_reviews(reviews)
    required_cols = {"user", "item", "rating", "timestamp"}
    missing = required_cols - set(canonical.columns)
    if missing:
        raise ValueError(f"Review table is missing columns: {', '.join(sorted(missing))}")

    frame = canonical.copy()
    for column in ("review_id", "useful", "funny", "cool"):
        if column not in frame.columns:
            frame[column] = np.nan if column == "review_id" else 0.0
    frame["useful"] = pd.to_numeric(frame["useful"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["funny"] = pd.to_numeric(frame["funny"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["cool"] = pd.to_numeric(frame["cool"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return frame[["review_id", "user", "item", "rating", "timestamp", "useful", "funny", "cool"]].dropna(
        subset=["user", "item", "rating", "timestamp"]
    ).reset_index(drop=True)


def build_review_context_only_frame(reviews: pd.DataFrame) -> pd.DataFrame:
    canonical = canonicalize_reviews(reviews)
    if "timestamp" not in canonical.columns:
        raise ValueError("Review table is missing timestamp/date.")
    frame = canonical.copy()
    for column in ("review_id", "useful", "funny", "cool"):
        if column not in frame.columns:
            frame[column] = np.nan if column == "review_id" else 0.0
    frame["useful"] = pd.to_numeric(frame["useful"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["funny"] = pd.to_numeric(frame["funny"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["cool"] = pd.to_numeric(frame["cool"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return frame[["review_id", "user", "item", "timestamp", "useful", "funny", "cool"]].dropna(
        subset=["user", "item", "timestamp"]
    ).reset_index(drop=True)


def summarize_embedding_join(frame: pd.DataFrame, bundle: FrozenEmbeddingBundle) -> dict[str, Any]:
    user_index = set(bundle.user_ids.to_numpy())
    business_index = set(bundle.business_ids.to_numpy())
    known_user = frame["user"].isin(user_index)
    known_business = frame["item"].isin(business_index)
    return {
        "total_rows": int(len(frame)),
        "known_user_rows": int(known_user.sum()),
        "known_business_rows": int(known_business.sum()),
        "known_both_rows": int((known_user & known_business).sum()),
        "missing_user_rows": int((~known_user).sum()),
        "missing_business_rows": int((~known_business).sum()),
        "missing_either_rows": int((~(known_user & known_business)).sum()),
    }


def attach_embedding_indices(frame: pd.DataFrame, bundle: FrozenEmbeddingBundle) -> tuple[pd.DataFrame, dict[str, Any]]:
    join_summary = summarize_embedding_join(frame, bundle)
    user_index = pd.Series(np.arange(len(bundle.user_ids), dtype=np.int32), index=bundle.user_ids.to_numpy())
    business_index = pd.Series(np.arange(len(bundle.business_ids), dtype=np.int32), index=bundle.business_ids.to_numpy())
    history_lookup = pd.Series(
        bundle.user_table["history_count_train"].to_numpy(dtype=np.float32),
        index=bundle.user_table["user_id"].to_numpy(),
    )
    band_lookup = pd.Series(
        bundle.user_table["history_band"].to_numpy(),
        index=bundle.user_table["user_id"].to_numpy(),
    )

    filtered = frame[frame["user"].isin(user_index.index) & frame["item"].isin(business_index.index)].copy()
    filtered["user_idx"] = filtered["user"].map(user_index).astype(np.int32)
    filtered["business_idx"] = filtered["item"].map(business_index).astype(np.int32)
    filtered["history_count_train"] = filtered["user"].map(history_lookup).fillna(0.0).astype(np.float32)
    filtered["history_band"] = filtered["user"].map(band_lookup).fillna("Unknown").astype(str)
    filtered["history_log1p"] = np.log1p(filtered["history_count_train"].to_numpy(dtype=np.float32))
    filtered = filtered.reset_index(drop=True)
    join_summary["kept_rows"] = int(len(filtered))
    return filtered, join_summary


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_preference_auc(eval_df: pd.DataFrame) -> tuple[float, int]:
    aucs: list[float] = []
    usable = 0
    for _, group in eval_df.groupby("user", sort=False):
        pos = group.loc[group["rating"] >= 4.0, "pred"].to_numpy(dtype=np.float32)[:20]
        neg = group.loc[group["rating"] <= 2.0, "pred"].to_numpy(dtype=np.float32)[:20]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        aucs.append(float((diff > 0).mean() + 0.5 * (diff == 0).mean()))
        usable += 1
    return (float(np.mean(aucs)) if aucs else float("nan"), usable)


def compute_band_metrics(frame: pd.DataFrame, prediction_col: str = "pred") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for history_band in ["0", "1", "2-5", "6-20", ">20", "Unknown"]:
        subset = frame[frame["history_band"] == history_band]
        if subset.empty:
            continue
        rows.append(
            {
                "history_band": history_band,
                "mae": float(mean_absolute_error(subset["rating"], subset[prediction_col])),
                "rmse": rmse(
                    subset["rating"].to_numpy(dtype=np.float32),
                    subset[prediction_col].to_numpy(dtype=np.float32),
                ),
                "n_samples": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def compute_dense_pair_features(
    user_matrix: np.ndarray,
    item_matrix: np.ndarray,
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    history_log1p: np.ndarray,
) -> np.ndarray:
    user_repr = user_matrix[user_idx]
    item_repr = item_matrix[item_idx]
    dot = np.einsum("ij,ij->i", user_repr, item_repr)
    user_norm = np.linalg.norm(user_repr, axis=1)
    item_norm = np.linalg.norm(item_repr, axis=1)
    cosine = dot / np.maximum(user_norm * item_norm, 1e-8)
    return np.column_stack(
        [cosine, dot, user_norm, item_norm, np.abs(user_norm - item_norm), history_log1p]
    ).astype(np.float32)


def fit_ridge_embedding_baseline(
    *,
    bundle: FrozenEmbeddingBundle,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    x_train = compute_dense_pair_features(
        bundle.user_embeddings,
        bundle.business_embeddings,
        train_frame["user_idx"].to_numpy(dtype=np.int32),
        train_frame["business_idx"].to_numpy(dtype=np.int32),
        train_frame["history_log1p"].to_numpy(dtype=np.float32),
    )
    x_val = compute_dense_pair_features(
        bundle.user_embeddings,
        bundle.business_embeddings,
        val_frame["user_idx"].to_numpy(dtype=np.int32),
        val_frame["business_idx"].to_numpy(dtype=np.int32),
        val_frame["history_log1p"].to_numpy(dtype=np.float32),
    )
    y_train = train_frame["rating"].to_numpy(dtype=np.float32)
    y_val = val_frame["rating"].to_numpy(dtype=np.float32)

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    prediction = np.clip(model.predict(x_val).astype(np.float32), 1.0, 5.0)

    eval_frame = val_frame[["review_id", "user", "item", "rating", "timestamp", "history_band"]].copy()
    eval_frame["pred"] = prediction
    pairwise_auc, auc_users = compute_preference_auc(eval_frame)
    summary = {
        "model_type": "ridge_baseline",
        "mae": float(mean_absolute_error(y_val, prediction)),
        "rmse": rmse(y_val, prediction),
        "pairwise_auc": pairwise_auc,
        "pairwise_auc_users": int(auc_users),
        "n_train_rows": int(len(train_frame)),
        "n_val_rows": int(len(val_frame)),
        "feature_names": [
            "cosine",
            "dot",
            "user_norm",
            "business_norm",
            "norm_gap",
            "history_log1p",
        ],
    }
    return summary, eval_frame


def build_review_context_features(
    *,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    include_review_context: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not include_review_context:
        return (
            np.zeros((len(train_frame), 0), dtype=np.float32),
            np.zeros((len(eval_frame), 0), dtype=np.float32),
            {
                "enabled": False,
                "feature_names": [],
                "transforms": [],
            },
        )

    feature_names = [
        "useful_log1p",
        "funny_log1p",
        "cool_log1p",
        "recency_days_z",
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
        "hour_sin",
        "hour_cos",
    ]
    min_timestamp = train_frame["timestamp"].min()

    def _matrix(frame: pd.DataFrame) -> np.ndarray:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        useful = np.log1p(frame["useful"].to_numpy(dtype=np.float32))
        funny = np.log1p(frame["funny"].to_numpy(dtype=np.float32))
        cool = np.log1p(frame["cool"].to_numpy(dtype=np.float32))
        recency_days = ((ts - min_timestamp).dt.total_seconds() / 86400.0).to_numpy(dtype=np.float32)
        month_angle = 2.0 * np.pi * (ts.dt.month.fillna(1).to_numpy(dtype=np.float32) - 1.0) / 12.0
        weekday_angle = 2.0 * np.pi * ts.dt.weekday.fillna(0).to_numpy(dtype=np.float32) / 7.0
        hour_angle = 2.0 * np.pi * ts.dt.hour.fillna(0).to_numpy(dtype=np.float32) / 24.0
        return np.column_stack(
            [
                useful,
                funny,
                cool,
                recency_days,
                np.sin(month_angle),
                np.cos(month_angle),
                np.sin(weekday_angle),
                np.cos(weekday_angle),
                np.sin(hour_angle),
                np.cos(hour_angle),
            ]
        ).astype(np.float32)

    train_matrix = _matrix(train_frame)
    eval_matrix = _matrix(eval_frame)

    means = train_matrix.mean(axis=0)
    stds = train_matrix.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    train_scaled = ((train_matrix - means) / stds).astype(np.float32)
    eval_scaled = ((eval_matrix - means) / stds).astype(np.float32)

    return (
        train_scaled,
        eval_scaled,
        {
            "enabled": True,
            "feature_names": feature_names,
            "transforms": [
                "log1p on useful/funny/cool",
                "recency_days derived from train split minimum timestamp",
                "cyclical sin/cos encoding for month, weekday, and hour",
                "z-score standardization fit on train split only",
            ],
            "train_means": {name: float(value) for name, value in zip(feature_names, means, strict=False)},
            "train_stds": {name: float(value) for name, value in zip(feature_names, stds, strict=False)},
            "train_min_timestamp": pd.Timestamp(min_timestamp).isoformat(),
        },
    )


def transform_review_context_features(
    frame: pd.DataFrame,
    *,
    context_summary: dict[str, Any],
) -> np.ndarray:
    if not context_summary.get("enabled", False):
        return np.zeros((len(frame), 0), dtype=np.float32)

    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    min_timestamp = pd.Timestamp(context_summary["train_min_timestamp"])
    useful = np.log1p(pd.to_numeric(frame["useful"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32))
    funny = np.log1p(pd.to_numeric(frame["funny"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32))
    cool = np.log1p(pd.to_numeric(frame["cool"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32))
    recency_days = ((ts - min_timestamp).dt.total_seconds() / 86400.0).to_numpy(dtype=np.float32)
    month_angle = 2.0 * np.pi * (ts.dt.month.fillna(1).to_numpy(dtype=np.float32) - 1.0) / 12.0
    weekday_angle = 2.0 * np.pi * ts.dt.weekday.fillna(0).to_numpy(dtype=np.float32) / 7.0
    hour_angle = 2.0 * np.pi * ts.dt.hour.fillna(0).to_numpy(dtype=np.float32) / 24.0
    matrix = np.column_stack(
        [
            useful,
            funny,
            cool,
            recency_days,
            np.sin(month_angle),
            np.cos(month_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            np.sin(hour_angle),
            np.cos(hour_angle),
        ]
    ).astype(np.float32)

    means = np.array(
        [context_summary["train_means"][name] for name in context_summary["feature_names"]],
        dtype=np.float32,
    )
    stds = np.array(
        [context_summary["train_stds"][name] for name in context_summary["feature_names"]],
        dtype=np.float32,
    )
    stds = np.where(stds < 1e-6, 1.0, stds)
    return ((matrix - means) / stds).astype(np.float32)
