from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .deep_user_embeddings import _build_fixed_context_arrays, _build_prefix_training_arrays
from .frozen_embedding_regression import FrozenEmbeddingBundle, load_frozen_embedding_bundle
from .io import canonicalize_reviews
from .lgbm_raw_features import history_band_from_count


DEFAULT_KNOWN_PREFIX_TARGET_BANDS = ("1", "2-5", "6-20")
_PREFIX_VECTOR_BLOCKS = (
    "known_prefix_candidate_emb",
    "known_prefix_history_mean_emb",
    "known_prefix_history_recency_emb",
    "known_prefix_history_attn_emb",
)
_PREFIX_SCALAR_FEATURES = (
    "known_prefix_candidate_embedding_known",
    "known_prefix_history_count",
    "known_prefix_history_count_log1p",
    "known_prefix_count_is_2",
    "known_prefix_count_is_3",
    "known_prefix_count_is_4",
    "known_prefix_count_is_5",
    "known_prefix_history_rating_mean",
    "known_prefix_history_rating_std",
    "known_prefix_history_rating_min",
    "known_prefix_history_rating_max",
    "known_prefix_history_rating_range",
    "known_prefix_history_last_rating",
    "known_prefix_history_positive_share",
    "known_prefix_history_negative_share",
    "known_prefix_history_similarity_max",
    "known_prefix_history_similarity_mean",
    "known_prefix_last_item_similarity",
    "known_prefix_history_similarity_mean_x_count",
    "known_prefix_history_similarity_max_x_count",
    "known_prefix_last_item_similarity_x_count",
    "known_prefix_history_rating_std_x_count",
    "known_prefix_history_similarity_mean_x_rating_std",
    "known_prefix_last_item_l2",
    "known_prefix_mean_cosine",
    "known_prefix_mean_dot",
    "known_prefix_mean_l2",
    "known_prefix_recency_cosine",
    "known_prefix_recency_dot",
    "known_prefix_recency_l2",
    "known_prefix_attn_cosine",
    "known_prefix_attn_dot",
    "known_prefix_attn_l2",
)


def load_known_prefix_embedding_bundle(root: str | Path) -> FrozenEmbeddingBundle:
    return load_frozen_embedding_bundle(root)


def get_known_prefix_feature_names(embedding_dim: int) -> list[str]:
    feature_names: list[str] = []
    for prefix in _PREFIX_VECTOR_BLOCKS:
        feature_names.extend(f"{prefix}_{index:03d}" for index in range(embedding_dim))
    feature_names.extend(_PREFIX_SCALAR_FEATURES)
    return feature_names


def parse_known_prefix_target_bands(raw_value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if raw_value is None:
        return DEFAULT_KNOWN_PREFIX_TARGET_BANDS
    if isinstance(raw_value, tuple):
        values = list(raw_value)
    elif isinstance(raw_value, list):
        values = raw_value
    else:
        values = [token.strip() for token in str(raw_value).split(",")]
    allowed = {"1", "2-5", "6-20", ">20"}
    normalized = tuple(value for value in values if value in allowed)
    if not normalized:
        raise ValueError("known_prefix_target_bands must include one or more of: 1, 2-5, 6-20, >20")
    return normalized


def mask_known_prefix_bands(history_band: pd.Series | np.ndarray | list[str], target_bands: tuple[str, ...]) -> np.ndarray:
    target_set = set(target_bands)
    values = np.asarray(history_band, dtype=object)
    return np.array([str(value) in target_set for value in values], dtype=bool)


def resolve_router_branches(
    *,
    user_known_mask: np.ndarray,
    history_band: pd.Series | np.ndarray | list[str],
    enabled_known_prefix_bands: tuple[str, ...],
) -> np.ndarray:
    history_values = np.asarray(history_band, dtype=object)
    prefix_mask = user_known_mask & mask_known_prefix_bands(history_values, enabled_known_prefix_bands)
    return np.where(~user_known_mask, "cold_model", np.where(prefix_mask, "known_prefix_deep_model", "known_model"))


def build_known_prefix_train_frame(
    reviews_df: pd.DataFrame,
    bundle: FrozenEmbeddingBundle,
    *,
    max_history_len: int,
    target_history_bands: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    target_bands = parse_known_prefix_target_bands(target_history_bands)
    prepared_reviews, business_index, user_index = _prepare_reviews_with_indices(reviews_df, bundle)
    arrays = _build_prefix_training_arrays(interactions=prepared_reviews, max_history_len=max_history_len)
    ordered = _sorted_target_rows(prepared_reviews)
    ordered["known_prefix_exact_history_count"] = ordered.groupby("user", sort=False).cumcount().astype(np.int32)
    return _build_prefix_feature_frame(
        ordered_targets=ordered,
        bundle=bundle,
        history_item_idx=arrays["history_item_idx"],
        history_ratings=arrays["history_ratings"],
        exact_history_count=ordered["known_prefix_exact_history_count"].to_numpy(dtype=np.int32),
        target_bands=target_bands,
    )


def build_known_prefix_eval_frame(
    target_reviews: pd.DataFrame,
    context_reviews: pd.DataFrame,
    bundle: FrozenEmbeddingBundle,
    *,
    max_history_len: int,
    target_history_bands: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    target_bands = parse_known_prefix_target_bands(target_history_bands)
    prepared_target, prepared_context = _prepare_target_and_context_reviews(target_reviews, context_reviews, bundle)
    arrays = _build_fixed_context_arrays(
        target_interactions=prepared_target,
        context_interactions=prepared_context,
        max_history_len=max_history_len,
    )
    ordered_targets = _sorted_target_rows(prepared_target)
    context_counts = prepared_context.groupby("user").size().to_dict()
    exact_history_count = ordered_targets["user"].map(context_counts).fillna(0).astype(np.int32).to_numpy()
    ordered_targets["known_prefix_exact_history_count"] = exact_history_count
    return _build_prefix_feature_frame(
        ordered_targets=ordered_targets,
        bundle=bundle,
        history_item_idx=arrays["history_item_idx"],
        history_ratings=arrays["history_ratings"],
        exact_history_count=exact_history_count,
        target_bands=target_bands,
    )


def _prepare_target_and_context_reviews(
    target_reviews: pd.DataFrame,
    context_reviews: pd.DataFrame,
    bundle: FrozenEmbeddingBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_target = canonicalize_reviews(target_reviews)
    canonical_context = canonicalize_reviews(context_reviews)
    all_users = pd.Index(
        pd.concat(
            [
                canonical_target.get("user", pd.Series(dtype=object)),
                canonical_context.get("user", pd.Series(dtype=object)),
            ],
            ignore_index=True,
        ).dropna().drop_duplicates()
    )
    user_index = pd.Series(np.arange(len(all_users), dtype=np.int32), index=all_users.to_numpy())
    business_index = pd.Series(np.arange(len(bundle.business_ids), dtype=np.int32), index=bundle.business_ids.to_numpy())
    prepared_target = _prepare_review_frame(target_reviews, business_index=business_index, user_index=user_index)
    prepared_context = _prepare_review_frame(context_reviews, business_index=business_index, user_index=user_index)
    return prepared_target, prepared_context


def _prepare_reviews_with_indices(
    reviews_df: pd.DataFrame,
    bundle: FrozenEmbeddingBundle,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    canonical = canonicalize_reviews(reviews_df)
    all_users = pd.Index(canonical.get("user", pd.Series(dtype=object)).dropna().drop_duplicates())
    user_index = pd.Series(np.arange(len(all_users), dtype=np.int32), index=all_users.to_numpy())
    business_index = pd.Series(np.arange(len(bundle.business_ids), dtype=np.int32), index=bundle.business_ids.to_numpy())
    prepared = _prepare_review_frame(reviews_df, business_index=business_index, user_index=user_index)
    return prepared, business_index, user_index


def _prepare_review_frame(
    reviews_df: pd.DataFrame,
    *,
    business_index: pd.Series,
    user_index: pd.Series,
) -> pd.DataFrame:
    frame = canonicalize_reviews(reviews_df)
    if "review_id" not in frame.columns:
        frame["review_id"] = np.arange(len(frame), dtype=np.int64)
    if "rating" not in frame.columns:
        frame["rating"] = np.nan
    if "timestamp" not in frame.columns:
        raise ValueError("Review table is missing timestamp/date.")
    frame["item_idx"] = frame["item"].map(business_index).fillna(-1).astype(np.int32)
    frame["user_idx"] = frame["user"].map(user_index).fillna(-1).astype(np.int32)
    return frame[["review_id", "user", "item", "rating", "timestamp", "item_idx", "user_idx"]].copy()


def _sorted_target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["user", "timestamp", "item"], kind="stable").reset_index(drop=True).copy()


def _build_prefix_feature_frame(
    *,
    ordered_targets: pd.DataFrame,
    bundle: FrozenEmbeddingBundle,
    history_item_idx: np.ndarray,
    history_ratings: np.ndarray,
    exact_history_count: np.ndarray,
    target_bands: tuple[str, ...],
) -> pd.DataFrame:
    history_band = np.array([history_band_from_count(int(value)) for value in exact_history_count], dtype=object)
    target_mask = mask_known_prefix_bands(history_band, target_bands)
    if not target_mask.any():
        empty = ordered_targets.iloc[0:0][["review_id", "user", "item", "rating", "timestamp"]].copy()
        empty["known_prefix_history_band"] = pd.Series(dtype="string")
        for feature_name in get_known_prefix_feature_names(bundle.business_embeddings.shape[1]):
            empty[feature_name] = pd.Series(dtype=np.float32)
        return empty

    feature_names = get_known_prefix_feature_names(bundle.business_embeddings.shape[1])
    filtered_targets = ordered_targets.loc[target_mask, ["review_id", "user", "item", "rating", "timestamp"]].reset_index(drop=True)
    filtered_targets["known_prefix_history_band"] = pd.Series(history_band[target_mask], dtype="string")
    feature_matrix = _compute_prefix_feature_matrix(
        bundle=bundle,
        candidate_item_idx=ordered_targets.loc[target_mask, "item_idx"].to_numpy(dtype=np.int32),
        history_item_idx=history_item_idx[target_mask],
        history_ratings=history_ratings[target_mask],
        exact_history_count=exact_history_count[target_mask],
    )
    feature_frame = pd.DataFrame(feature_matrix, columns=feature_names, copy=False)
    for column in feature_frame.columns:
        feature_frame[column] = feature_frame[column].astype(np.float32)
    return pd.concat([filtered_targets, feature_frame], axis=1)


def _compute_prefix_feature_matrix(
    *,
    bundle: FrozenEmbeddingBundle,
    candidate_item_idx: np.ndarray,
    history_item_idx: np.ndarray,
    history_ratings: np.ndarray,
    exact_history_count: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    business_embeddings = bundle.business_embeddings.astype(np.float32, copy=False)
    embedding_dim = business_embeddings.shape[1]
    feature_names = get_known_prefix_feature_names(embedding_dim)
    feature_matrix = np.zeros((len(candidate_item_idx), len(feature_names)), dtype=np.float32)
    if len(candidate_item_idx) == 0:
        return feature_matrix

    max_history_len = history_item_idx.shape[1]
    base_positions = np.arange(1, max_history_len + 1, dtype=np.float32)
    for start in range(0, len(candidate_item_idx), chunk_size):
        stop = min(start + chunk_size, len(candidate_item_idx))
        batch_candidate_idx = candidate_item_idx[start:stop]
        batch_history_idx = history_item_idx[start:stop]
        batch_history_ratings = history_ratings[start:stop]
        batch_exact_count = exact_history_count[start:stop].astype(np.float32, copy=False)

        candidate_known = batch_candidate_idx >= 0
        candidate_safe_idx = np.where(candidate_known, batch_candidate_idx, 0).astype(np.int32, copy=False)
        candidate_emb = business_embeddings[candidate_safe_idx].copy()
        candidate_emb[~candidate_known] = 0.0

        history_valid = batch_history_idx >= 0
        history_safe_idx = np.where(history_valid, batch_history_idx, 0).astype(np.int32, copy=False)
        history_emb = business_embeddings[history_safe_idx]
        history_emb = history_emb * history_valid[..., None].astype(np.float32)

        valid_count = history_valid.sum(axis=1).astype(np.float32)
        safe_count = np.maximum(valid_count, 1.0)
        rating_mask = history_valid.astype(np.float32)
        rating_sum = (batch_history_ratings * rating_mask).sum(axis=1)
        rating_mean = rating_sum / safe_count
        centered = (batch_history_ratings - rating_mean[:, None]) * rating_mask
        rating_std = np.sqrt((centered * centered).sum(axis=1) / safe_count)
        rating_min = np.where(history_valid, batch_history_ratings, np.inf).min(axis=1)
        rating_max = np.where(history_valid, batch_history_ratings, -np.inf).max(axis=1)
        rating_min = np.where(valid_count > 0, rating_min, 0.0)
        rating_max = np.where(valid_count > 0, rating_max, 0.0)
        positive_share = ((batch_history_ratings >= 4.0).astype(np.float32) * rating_mask).sum(axis=1) / safe_count
        negative_share = ((batch_history_ratings <= 2.0).astype(np.float32) * rating_mask).sum(axis=1) / safe_count

        history_mean = history_emb.sum(axis=1) / safe_count[:, None]
        recency_weights = rating_mask * base_positions[None, :]
        recency_denominator = np.maximum(recency_weights.sum(axis=1, keepdims=True), 1.0)
        history_recency = (history_emb * recency_weights[..., None]).sum(axis=1) / recency_denominator

        attention_scores = np.einsum("bhd,bd->bh", history_emb, candidate_emb, optimize=True)
        attention_scores = np.where(history_valid, attention_scores, -1e9)
        attention_max = attention_scores.max(axis=1, keepdims=True)
        attention_exp = np.exp(attention_scores - attention_max) * rating_mask
        attention_denominator = attention_exp.sum(axis=1, keepdims=True)
        uniform_attention = rating_mask / safe_count[:, None]
        attention_weights = np.where(attention_denominator > 0.0, attention_exp / np.maximum(attention_denominator, 1e-8), uniform_attention)
        history_attn = np.einsum("bh,bhd->bd", attention_weights, history_emb, optimize=True)

        candidate_norm = np.linalg.norm(candidate_emb, axis=1)
        history_item_dot = np.einsum("bhd,bd->bh", history_emb, candidate_emb, optimize=True)
        history_item_norm = np.linalg.norm(history_emb, axis=2)
        history_item_cos = history_item_dot / np.maximum(history_item_norm * candidate_norm[:, None], 1e-8)
        history_item_cos = np.where(history_valid, history_item_cos, -1.0)
        similarity_max = np.where(valid_count > 0, history_item_cos.max(axis=1), 0.0)
        similarity_mean = (np.where(history_valid, history_item_cos, 0.0).sum(axis=1) / safe_count)

        last_valid_idx = np.maximum(valid_count.astype(np.int32) - 1, 0)
        last_item_emb = history_emb[np.arange(stop - start), last_valid_idx]
        last_item_emb = np.where(valid_count[:, None] > 0, last_item_emb, 0.0)
        last_rating = batch_history_ratings[np.arange(stop - start), last_valid_idx]
        last_rating = np.where(valid_count > 0, last_rating, 0.0)
        last_item_dot = np.einsum("bd,bd->b", last_item_emb, candidate_emb, optimize=True)
        last_item_norm = np.linalg.norm(last_item_emb, axis=1)
        last_item_similarity = last_item_dot / np.maximum(last_item_norm * candidate_norm, 1e-8)
        last_item_similarity = np.where(valid_count > 0, last_item_similarity, 0.0)
        last_item_l2 = np.linalg.norm(last_item_emb - candidate_emb, axis=1)
        last_item_l2 = np.where(valid_count > 0, last_item_l2, 0.0)

        mean_dot = np.einsum("bd,bd->b", history_mean, candidate_emb, optimize=True)
        mean_norm = np.linalg.norm(history_mean, axis=1)
        mean_cos = mean_dot / np.maximum(mean_norm * candidate_norm, 1e-8)
        mean_l2 = np.linalg.norm(history_mean - candidate_emb, axis=1)

        recency_dot = np.einsum("bd,bd->b", history_recency, candidate_emb, optimize=True)
        recency_norm = np.linalg.norm(history_recency, axis=1)
        recency_cos = recency_dot / np.maximum(recency_norm * candidate_norm, 1e-8)
        recency_l2 = np.linalg.norm(history_recency - candidate_emb, axis=1)

        attn_dot = np.einsum("bd,bd->b", history_attn, candidate_emb, optimize=True)
        attn_norm = np.linalg.norm(history_attn, axis=1)
        attn_cos = attn_dot / np.maximum(attn_norm * candidate_norm, 1e-8)
        attn_l2 = np.linalg.norm(history_attn - candidate_emb, axis=1)

        scalar_block = np.column_stack(
            [
                candidate_known.astype(np.float32),
                batch_exact_count,
                np.log1p(batch_exact_count),
                (batch_exact_count == 2.0).astype(np.float32),
                (batch_exact_count == 3.0).astype(np.float32),
                (batch_exact_count == 4.0).astype(np.float32),
                (batch_exact_count == 5.0).astype(np.float32),
                rating_mean.astype(np.float32),
                rating_std.astype(np.float32),
                rating_min.astype(np.float32),
                rating_max.astype(np.float32),
                (rating_max - rating_min).astype(np.float32),
                last_rating.astype(np.float32),
                positive_share.astype(np.float32),
                negative_share.astype(np.float32),
                similarity_max.astype(np.float32),
                similarity_mean.astype(np.float32),
                last_item_similarity.astype(np.float32),
                (similarity_mean * batch_exact_count).astype(np.float32),
                (similarity_max * batch_exact_count).astype(np.float32),
                (last_item_similarity * batch_exact_count).astype(np.float32),
                (rating_std * batch_exact_count).astype(np.float32),
                (similarity_mean * rating_std).astype(np.float32),
                last_item_l2.astype(np.float32),
                mean_cos.astype(np.float32),
                mean_dot.astype(np.float32),
                mean_l2.astype(np.float32),
                recency_cos.astype(np.float32),
                recency_dot.astype(np.float32),
                recency_l2.astype(np.float32),
                attn_cos.astype(np.float32),
                attn_dot.astype(np.float32),
                attn_l2.astype(np.float32),
            ]
        )

        feature_block = np.concatenate(
            [
                candidate_emb.astype(np.float32),
                history_mean.astype(np.float32),
                history_recency.astype(np.float32),
                history_attn.astype(np.float32),
                scalar_block.astype(np.float32),
            ],
            axis=1,
        )
        feature_matrix[start:stop] = feature_block
    return feature_matrix
