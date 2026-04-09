from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from .audit import compare_user_metadata_with_train, summarize_user_comparison
from .business_representation import BusinessRepresentationBundle
from .io import canonicalize_reviews


AggregationMode = Literal["mean", "rating", "centered", "recency"]


@dataclass(slots=True)
class UserRepresentationConfig:
    aggregation_mode: AggregationMode = "centered"
    business_view: str = "content"
    business_blocks: list[str] | None = None
    include_metadata: bool = True
    recency_half_life_days: float = 180.0


@dataclass(slots=True)
class UserRepresentationBundle:
    user_ids: pd.Series
    clean_user_table: pd.DataFrame
    profile_matrix: sparse.csr_matrix
    metadata_matrix: sparse.csr_matrix
    full_user_matrix: sparse.csr_matrix
    profile_summary: dict[str, Any]
    feature_metadata: pd.DataFrame
    user_metadata_audit_summary: dict[str, Any]
    user_metadata_audit_details: pd.DataFrame
    profile_feature_names: list[str]
    metadata_feature_names: list[str]
    full_feature_names: list[str]

    def get_matrix(self, view: str = "full", blocks: list[str] | None = None) -> sparse.csr_matrix:
        if blocks is not None:
            mask = self.feature_metadata["block_name"].isin(blocks)
            indices = self.feature_metadata.loc[mask, "feature_index"].to_numpy(dtype=int)
            return self.full_user_matrix[:, indices]

        if view == "profile":
            return self.profile_matrix
        if view == "metadata":
            return self.metadata_matrix
        if view == "full":
            return self.full_user_matrix
        raise ValueError("view must be one of: 'profile', 'metadata', 'full'")

    def save(self, save_dir: str | Path) -> dict[str, str]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, str] = {}

        user_ids_path = save_dir / "user_ids.csv"
        self.user_ids.to_frame(name="user_id").to_csv(user_ids_path, index=False)
        saved_paths["user_ids"] = str(user_ids_path)

        sparse.save_npz(save_dir / "user_profile_features.npz", self.profile_matrix)
        sparse.save_npz(save_dir / "user_metadata_features.npz", self.metadata_matrix)
        sparse.save_npz(save_dir / "user_full_features.npz", self.full_user_matrix)
        saved_paths["profile_matrix"] = str(save_dir / "user_profile_features.npz")
        saved_paths["metadata_matrix"] = str(save_dir / "user_metadata_features.npz")
        saved_paths["full_matrix"] = str(save_dir / "user_full_features.npz")

        feature_names_path = save_dir / "user_feature_names.json"
        feature_names_path.write_text(
            json.dumps(
                {
                    "profile_features": self.profile_feature_names,
                    "metadata_features": self.metadata_feature_names,
                    "full_features": self.full_feature_names,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        saved_paths["feature_names"] = str(feature_names_path)

        feature_metadata_path = save_dir / "user_feature_metadata.csv"
        self.feature_metadata.to_csv(feature_metadata_path, index=False)
        saved_paths["feature_metadata"] = str(feature_metadata_path)

        summary_path = save_dir / "user_profile_summary.json"
        summary_path.write_text(
            json.dumps(self.profile_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        saved_paths["profile_summary"] = str(summary_path)

        audit_summary_path = save_dir / "user_metadata_audit_summary.json"
        audit_summary_path.write_text(
            json.dumps(self.user_metadata_audit_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        saved_paths["metadata_audit_summary"] = str(audit_summary_path)

        audit_details_path = save_dir / "user_metadata_audit_details.csv"
        self.user_metadata_audit_details.to_csv(audit_details_path, index=False)
        saved_paths["metadata_audit_details"] = str(audit_details_path)

        clean_table_path = _save_clean_table(self.clean_user_table, save_dir / "clean_user_table")
        saved_paths["clean_user_table"] = str(clean_table_path)

        return saved_paths


class UserRepresentationBuilder:
    def __init__(self, config: UserRepresentationConfig | None = None) -> None:
        self.config = config or UserRepresentationConfig()

    def fit_transform(
        self,
        train_reviews: pd.DataFrame,
        business_bundle: BusinessRepresentationBundle,
        users_df: pd.DataFrame | None = None,
        target_user_ids: pd.Series | list[str] | np.ndarray | None = None,
    ) -> UserRepresentationBundle:
        canonical = canonicalize_reviews(train_reviews)
        if {"user", "item", "rating"} - set(canonical.columns):
            raise ValueError("train_reviews must contain user, item, and rating columns")

        business_matrix, business_feature_names = self._select_business_view(business_bundle)
        business_index = pd.Series(
            np.arange(len(business_bundle.business_ids), dtype=np.int32),
            index=business_bundle.business_ids.to_numpy(),
        )

        interactions = canonical[canonical["item"].isin(business_index.index)].copy()
        if interactions.empty:
            raise ValueError("No train interactions match the provided business representation.")

        interactions["item_idx"] = interactions["item"].map(business_index).astype(np.int32)
        user_codes, unique_users = pd.factorize(interactions["user"], sort=False)
        interactions["user_idx"] = user_codes.astype(np.int32)
        unique_user_series = pd.Series(unique_users, name="user_id")

        user_stats = self._build_user_stats(interactions, len(unique_users))
        weights, fallback_summary = self._compute_weights(interactions, user_stats)

        weight_matrix, denominators = self._build_weight_matrix(
            user_idx=interactions["user_idx"].to_numpy(dtype=np.int32),
            item_idx=interactions["item_idx"].to_numpy(dtype=np.int32),
            weights=weights,
            n_users=len(unique_users),
            n_items=business_matrix.shape[0],
            mode=self.config.aggregation_mode,
        )
        profile_matrix = sparse.diags(1.0 / denominators).dot(weight_matrix).dot(business_matrix).tocsr().astype(np.float32)

        output_user_series = _resolve_target_user_series(
            base_user_ids=unique_user_series,
            target_user_ids=target_user_ids,
        )
        if len(output_user_series) != len(unique_user_series) or not output_user_series.equals(unique_user_series):
            profile_matrix = _expand_sparse_matrix_to_target_users(
                matrix=profile_matrix,
                source_user_ids=unique_user_series,
                target_user_ids=output_user_series,
            )
            user_stats = _expand_user_stats_to_target_users(
                user_stats=user_stats,
                source_user_ids=unique_user_series,
                target_user_ids=output_user_series,
            )
        else:
            output_user_series = unique_user_series.copy()

        metadata_matrix, clean_metadata_table, metadata_feature_names = self._build_metadata_block(
            output_user_series,
            users_df,
            train_reviews,
        )

        full_user_matrix = sparse.hstack([profile_matrix, metadata_matrix], format="csr").astype(np.float32)
        profile_feature_names = [f"profile__{name}" for name in business_feature_names]
        full_feature_names = profile_feature_names + metadata_feature_names
        feature_metadata = self._build_feature_metadata(profile_feature_names, metadata_feature_names)

        clean_user_table = self._build_clean_user_table(
            output_user_series,
            user_stats,
            fallback_summary,
            clean_metadata_table,
        )

        profile_summary = self._build_profile_summary(
            unique_user_series=output_user_series,
            profile_matrix=profile_matrix,
            metadata_matrix=metadata_matrix,
            full_user_matrix=full_user_matrix,
            business_feature_names=business_feature_names,
            fallback_summary=fallback_summary,
            user_stats=user_stats,
            metadata_feature_names=metadata_feature_names,
        )

        audit_details = compare_user_metadata_with_train(
            users_df if users_df is not None else pd.DataFrame({"user_id": output_user_series}),
            train_reviews,
        ) if users_df is not None else pd.DataFrame({"user_id": output_user_series})
        audit_summary = summarize_user_comparison(audit_details) if users_df is not None else {
            "total_rows": int(len(output_user_series)),
            "rows_seen_in_train": int(len(output_user_series)),
            "rows_unseen_in_train": 0,
        }

        self._run_validation_checks(
            user_ids=output_user_series,
            profile_matrix=profile_matrix,
            metadata_matrix=metadata_matrix,
            full_user_matrix=full_user_matrix,
            feature_metadata=feature_metadata,
            business_feature_names=business_feature_names,
        )

        return UserRepresentationBundle(
            user_ids=output_user_series,
            clean_user_table=clean_user_table,
            profile_matrix=profile_matrix,
            metadata_matrix=metadata_matrix,
            full_user_matrix=full_user_matrix,
            profile_summary=profile_summary,
            feature_metadata=feature_metadata,
            user_metadata_audit_summary=audit_summary,
            user_metadata_audit_details=audit_details,
            profile_feature_names=profile_feature_names,
            metadata_feature_names=metadata_feature_names,
            full_feature_names=full_feature_names,
        )

    def _select_business_view(
        self,
        business_bundle: BusinessRepresentationBundle,
    ) -> tuple[sparse.csr_matrix, list[str]]:
        matrix = business_bundle.get_matrix(
            view=self.config.business_view,
            blocks=self.config.business_blocks,
        ).tocsr()

        if self.config.business_blocks is not None:
            mask = business_bundle.feature_metadata["block_name"].isin(self.config.business_blocks)
            feature_names = business_bundle.feature_metadata.loc[mask, "feature_name"].tolist()
        elif self.config.business_view == "content":
            feature_names = business_bundle.content_feature_names
        elif self.config.business_view == "prior":
            feature_names = business_bundle.prior_feature_names
        elif self.config.business_view == "full":
            feature_names = business_bundle.full_feature_names
        else:
            raise ValueError("Unsupported business_view")

        return matrix, feature_names

    def _build_user_stats(self, interactions: pd.DataFrame, n_users: int) -> pd.DataFrame:
        grouped = interactions.groupby("user_idx", sort=False)
        stats = grouped["rating"].agg(["count", "mean"]).rename(columns={"count": "history_count", "mean": "user_mean_rating"})
        stats = stats.reindex(range(n_users))
        if "timestamp" in interactions.columns:
            stats["last_timestamp"] = grouped["timestamp"].max().reindex(range(n_users))
        else:
            stats["last_timestamp"] = pd.NaT
        stats["history_band"] = stats["history_count"].apply(_history_band)
        return stats

    def _compute_weights(
        self,
        interactions: pd.DataFrame,
        user_stats: pd.DataFrame,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ratings = interactions["rating"].to_numpy(dtype=np.float32)
        user_idx = interactions["user_idx"].to_numpy(dtype=np.int32)
        history_count = user_stats["history_count"].to_numpy(dtype=np.int32)
        history_per_interaction = history_count[user_idx]
        fallback_single_review = history_per_interaction <= 1

        mode = self.config.aggregation_mode
        if mode == "mean":
            weights = np.ones(len(interactions), dtype=np.float32)
        elif mode == "rating":
            weights = ratings.astype(np.float32)
        elif mode == "centered":
            mean_per_interaction = user_stats["user_mean_rating"].to_numpy(dtype=np.float32)[user_idx]
            weights = ratings - mean_per_interaction
            weights[fallback_single_review] = 1.0
            abs_sum = np.bincount(user_idx, weights=np.abs(weights), minlength=len(user_stats))
            zero_abs_mask = abs_sum == 0
            if zero_abs_mask.any():
                weights[zero_abs_mask[user_idx]] = 1.0
            fallback_zero_abs = int(zero_abs_mask.sum())
        elif mode == "recency":
            if "timestamp" not in interactions.columns:
                weights = np.ones(len(interactions), dtype=np.float32)
            else:
                timestamps = pd.to_datetime(interactions["timestamp"], errors="coerce")
                last_timestamps = pd.to_datetime(user_stats["last_timestamp"], errors="coerce")
                age_days = (
                    last_timestamps.iloc[user_idx].reset_index(drop=True) - timestamps.reset_index(drop=True)
                ).dt.total_seconds() / 86400.0
                age_days = age_days.fillna(0.0).clip(lower=0.0)
                weights = np.exp(-age_days.to_numpy(dtype=np.float32) / float(self.config.recency_half_life_days)).astype(np.float32)
        else:
            raise ValueError("Unsupported aggregation_mode")

        summary = {
            "aggregation_mode": mode,
            "fallback_single_review_count": int(fallback_single_review.sum()) if mode == "centered" else 0,
            "fallback_zero_abs_count": fallback_zero_abs if mode == "centered" else 0,
        }
        return weights, summary

    def _build_weight_matrix(
        self,
        *,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        weights: np.ndarray,
        n_users: int,
        n_items: int,
        mode: AggregationMode,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        weight_matrix = sparse.coo_matrix(
            (weights, (user_idx, item_idx)),
            shape=(n_users, n_items),
            dtype=np.float32,
        ).tocsr()

        if mode == "centered":
            denominators = np.asarray(np.abs(weight_matrix).sum(axis=1)).ravel()
        else:
            denominators = np.asarray(weight_matrix.sum(axis=1)).ravel()

        denominators = denominators.astype(np.float32)
        denominators[denominators == 0.0] = 1.0
        return weight_matrix, denominators

    def _build_metadata_block(
        self,
        unique_users: pd.Series,
        users_df: pd.DataFrame | None,
        train_reviews: pd.DataFrame,
    ) -> tuple[sparse.csr_matrix, pd.DataFrame, list[str]]:
        return build_safe_user_metadata_block(
            unique_users=unique_users,
            users_df=users_df,
            train_reviews=train_reviews,
            include_metadata=self.config.include_metadata,
        )

    def _build_feature_metadata(
        self,
        profile_feature_names: list[str],
        metadata_feature_names: list[str],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        feature_index = 0
        for feature_name in profile_feature_names:
            rows.append(
                {
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "block_name": "profile",
                    "source": "train_reviews + business_representation.content",
                    "requires_audit": False,
                    "default_rule": "computed from weighted aggregation of rated business vectors",
                }
            )
            feature_index += 1

        for feature_name in metadata_feature_names:
            rows.append(
                {
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "block_name": "metadata",
                    "source": "user_metadata_safe_direct",
                    "requires_audit": False,
                    "default_rule": "missing metadata->filled and normalized",
                }
            )
            feature_index += 1

        return pd.DataFrame(rows)

    def _build_clean_user_table(
        self,
        unique_users: pd.Series,
        user_stats: pd.DataFrame,
        fallback_summary: dict[str, Any],
        clean_metadata_table: pd.DataFrame,
    ) -> pd.DataFrame:
        table = pd.DataFrame(
            {
                "user_id": unique_users,
                "history_count": user_stats["history_count"].to_numpy(dtype=np.int32),
                "user_mean_rating": user_stats["user_mean_rating"].to_numpy(dtype=np.float32),
                "history_band": user_stats["history_band"].astype(str).to_numpy(),
                "aggregation_mode": self.config.aggregation_mode,
                "fallback_single_review_count_global": fallback_summary["fallback_single_review_count"],
                "fallback_zero_abs_count_global": fallback_summary["fallback_zero_abs_count"],
            }
        )
        if not clean_metadata_table.empty:
            table = table.merge(clean_metadata_table, on="user_id", how="left")
        return table

    def _build_profile_summary(
        self,
        *,
        unique_user_series: pd.Series,
        profile_matrix: sparse.csr_matrix,
        metadata_matrix: sparse.csr_matrix,
        full_user_matrix: sparse.csr_matrix,
        business_feature_names: list[str],
        fallback_summary: dict[str, Any],
        user_stats: pd.DataFrame,
        metadata_feature_names: list[str],
    ) -> dict[str, Any]:
        history_counts = user_stats["history_count"].fillna(0).astype(int)
        band_counts = user_stats["history_band"].value_counts(dropna=False).to_dict()
        return {
            "n_users_with_profile": int(len(unique_user_series)),
            "profile_shape": [int(profile_matrix.shape[0]), int(profile_matrix.shape[1])],
            "metadata_shape": [int(metadata_matrix.shape[0]), int(metadata_matrix.shape[1])],
            "full_user_shape": [int(full_user_matrix.shape[0]), int(full_user_matrix.shape[1])],
            "aggregation_mode": self.config.aggregation_mode,
            "business_view": self.config.business_view,
            "business_blocks": self.config.business_blocks,
            "include_metadata": bool(self.config.include_metadata),
            "profile_feature_count": int(len(business_feature_names)),
            "metadata_feature_count": int(len(metadata_feature_names)),
            "fallback_summary": fallback_summary,
            "history_count_summary": {
                "min": int(history_counts.min()),
                "median": float(history_counts.median()),
                "p90": float(history_counts.quantile(0.9)),
                "p99": float(history_counts.quantile(0.99)),
                "max": int(history_counts.max()),
            },
            "history_band_counts": {str(key): int(value) for key, value in band_counts.items()},
            "one_review_user_count": int((history_counts == 1).sum()),
        }

    def _run_validation_checks(
        self,
        *,
        user_ids: pd.Series,
        profile_matrix: sparse.csr_matrix,
        metadata_matrix: sparse.csr_matrix,
        full_user_matrix: sparse.csr_matrix,
        feature_metadata: pd.DataFrame,
        business_feature_names: list[str],
    ) -> None:
        if user_ids.duplicated().any():
            raise ValueError("user_ids must be unique in the user representation bundle")
        if profile_matrix.shape[0] != len(user_ids):
            raise ValueError("profile_matrix row count must match number of users")
        if metadata_matrix.shape[0] != len(user_ids):
            raise ValueError("metadata_matrix row count must match number of users")
        if full_user_matrix.shape[0] != len(user_ids):
            raise ValueError("full_user_matrix row count must match number of users")
        if profile_matrix.shape[1] + metadata_matrix.shape[1] != full_user_matrix.shape[1]:
            raise ValueError("profile_matrix + metadata_matrix width does not match full_user_matrix width")
        if feature_metadata["feature_index"].nunique() != len(feature_metadata):
            raise ValueError("feature_index must be unique")
        if feature_metadata["feature_index"].max() != len(feature_metadata) - 1:
            raise ValueError("feature_index must be contiguous")
        if self.config.business_view == "content" and any(name.startswith("prior__") for name in business_feature_names):
            raise ValueError("content user profiles must not include business priors")


def _history_band(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return ">20"


def _parse_elite_years_count(value: Any) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() == "none":
        return 0.0
    years = [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
    return float(len(years))


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    if std == 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - mean) / std).astype(np.float32)


def _save_clean_table(df: pd.DataFrame, filepath_without_suffix: Path) -> Path:
    parquet_path = filepath_without_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError, ValueError):
        csv_path = filepath_without_suffix.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def build_safe_user_metadata_block(
    unique_users: pd.Series,
    users_df: pd.DataFrame | None,
    train_reviews: pd.DataFrame,
    *,
    include_metadata: bool = True,
) -> tuple[sparse.csr_matrix, pd.DataFrame, list[str]]:
    if not include_metadata:
        empty = sparse.csr_matrix((len(unique_users), 0), dtype=np.float32)
        return empty, pd.DataFrame({"user_id": unique_users}), []

    metadata = pd.DataFrame({"user_id": unique_users})
    if users_df is not None and "user_id" in users_df.columns:
        user_metadata = users_df.drop_duplicates(subset=["user_id"]).copy()
        keep_columns = ["user_id", "yelping_since", "useful", "funny", "cool", "fans", "elite"]
        keep_columns.extend([column for column in user_metadata.columns if column.startswith("compliment_")])
        keep_columns = [column for column in keep_columns if column in user_metadata.columns]
        metadata = metadata.merge(user_metadata[keep_columns], on="user_id", how="left")

    train_end = pd.to_datetime(train_reviews["date"], errors="coerce").max()
    yelping_since = pd.to_datetime(metadata.get("yelping_since"), errors="coerce")
    tenure_days = (train_end - yelping_since).dt.total_seconds() / 86400.0
    nanmedian_tenure = float(np.nanmedian(tenure_days)) if len(metadata) else 0.0
    metadata["metadata__tenure_days"] = tenure_days.fillna(nanmedian_tenure if np.isfinite(nanmedian_tenure) else 0.0)
    metadata["metadata__elite_years_count"] = metadata.get("elite", pd.Series(index=metadata.index)).apply(_parse_elite_years_count)
    metadata["metadata__elite_any"] = (metadata["metadata__elite_years_count"] > 0).astype(np.float32)

    numeric_base: list[str] = []
    for column in ["useful", "funny", "cool", "fans"]:
        if column in metadata.columns:
            transformed = np.log1p(pd.to_numeric(metadata[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
            normalized = _safe_zscore(transformed)
            feature_name = f"metadata__{column}_log1p_z"
            metadata[feature_name] = normalized
            numeric_base.append(feature_name)

    compliment_columns = [column for column in metadata.columns if column.startswith("compliment_")]
    metadata_feature_names = ["metadata__tenure_days_z", "metadata__elite_years_count_z", "metadata__elite_any"]
    metadata["metadata__tenure_days_z"] = _safe_zscore(metadata["metadata__tenure_days"].to_numpy(dtype=np.float32))
    metadata["metadata__elite_years_count_z"] = _safe_zscore(metadata["metadata__elite_years_count"].to_numpy(dtype=np.float32))
    metadata_feature_names.extend(numeric_base)

    for column in compliment_columns:
        transformed = np.log1p(pd.to_numeric(metadata[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        feature_name = f"metadata__{column}_log1p_z"
        metadata[feature_name] = _safe_zscore(transformed)
        metadata_feature_names.append(feature_name)

    matrix_values = metadata[metadata_feature_names].fillna(0.0).to_numpy(dtype=np.float32)
    metadata_matrix = sparse.csr_matrix(matrix_values)
    clean_table = metadata[[
        "user_id",
        "metadata__tenure_days",
        "metadata__elite_years_count",
        "metadata__elite_any",
        *numeric_base,
        *[name for name in metadata_feature_names if name.startswith("metadata__compliment_")],
    ]].copy()
    return metadata_matrix, clean_table, metadata_feature_names


def _resolve_target_user_series(
    *,
    base_user_ids: pd.Series,
    target_user_ids: pd.Series | list[str] | np.ndarray | None,
) -> pd.Series:
    if target_user_ids is None:
        return base_user_ids.reset_index(drop=True).rename("user_id")

    raw_series = pd.Series(target_user_ids, dtype="object").dropna()
    if raw_series.empty:
        return base_user_ids.reset_index(drop=True).rename("user_id")

    resolved = raw_series.drop_duplicates().reset_index(drop=True)
    resolved.name = "user_id"
    return resolved


def _expand_sparse_matrix_to_target_users(
    *,
    matrix: sparse.csr_matrix,
    source_user_ids: pd.Series,
    target_user_ids: pd.Series,
) -> sparse.csr_matrix:
    if len(source_user_ids) == len(target_user_ids) and source_user_ids.reset_index(drop=True).equals(target_user_ids.reset_index(drop=True)):
        return matrix

    source_index = pd.Series(
        np.arange(len(source_user_ids), dtype=np.int32),
        index=source_user_ids.to_numpy(),
    )
    mapped = target_user_ids.map(source_index)
    known_mask = mapped.notna().to_numpy()
    if not known_mask.any():
        return sparse.csr_matrix((len(target_user_ids), matrix.shape[1]), dtype=matrix.dtype)

    known_positions = np.flatnonzero(known_mask)
    sliced = matrix[mapped[known_mask].to_numpy(dtype=np.int32)].tocoo()
    remapped_rows = known_positions[sliced.row]
    expanded = sparse.coo_matrix(
        (sliced.data, (remapped_rows, sliced.col)),
        shape=(len(target_user_ids), matrix.shape[1]),
        dtype=matrix.dtype,
    )
    return expanded.tocsr()


def _expand_user_stats_to_target_users(
    *,
    user_stats: pd.DataFrame,
    source_user_ids: pd.Series,
    target_user_ids: pd.Series,
) -> pd.DataFrame:
    if len(source_user_ids) == len(target_user_ids) and source_user_ids.reset_index(drop=True).equals(target_user_ids.reset_index(drop=True)):
        return user_stats.reset_index(drop=True)

    expanded = user_stats.copy()
    expanded.index = pd.Index(source_user_ids.to_numpy(), name="user_id")
    expanded = expanded.reindex(target_user_ids.to_numpy())
    expanded["history_count"] = expanded["history_count"].fillna(0).astype(np.int32)
    expanded["user_mean_rating"] = expanded["user_mean_rating"].fillna(0.0).astype(np.float32)
    if "last_timestamp" in expanded.columns:
        expanded["last_timestamp"] = pd.to_datetime(expanded["last_timestamp"], errors="coerce")
    expanded["history_band"] = [
        "0" if int(count) == 0 else _history_band(int(count))
        for count in expanded["history_count"].to_numpy(dtype=np.int32)
    ]
    return expanded.reset_index(drop=True)
