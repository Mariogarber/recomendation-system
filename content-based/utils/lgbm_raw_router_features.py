from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from .business_features import parse_categories
from .lgbm_raw_features import (
    CATEGORICAL_COLUMNS as BASE_CATEGORICAL_COLUMNS,
    RAW_CORE_FEATURE_SET,
    RawLGBMFeatureSpec,
    _normalize_text,
    _prepare_businesses,
    _prepare_reviews,
    _prepare_users,
    build_raw_feature_frame,
    fit_raw_feature_spec,
)


ROUTER_ARCHETYPE_NUMERIC_COLUMNS = [
    "user_average_stars",
    "user_review_count_log1p",
    "user_tenure_years",
    "user_total_votes_log1p",
    "user_engagement_log1p",
    "user_friends_log1p",
    "user_elite_years_count",
    "user_compliment_log1p_total",
    "user_compliment_nonzero_count",
]

ROUTER_CATEGORICAL_COLUMNS = [
    *BASE_CATEGORICAL_COLUMNS,
    "user_archetype_id",
    "user_activity_bucket",
    "user_reputation_bucket",
    "user_tenure_bucket",
    "business_city_top",
    "business_primary_category_family",
    "business_star_bin",
]

ROUTER_EXCLUDED_TRAINING_COLUMNS = [
    "user_known_in_train",
    "business_known_in_train",
]


@dataclass(slots=True)
class RouterRawFeatureSpec:
    base_spec: RawLGBMFeatureSpec
    feature_columns: list[str]
    known_feature_columns: list[str]
    cold_feature_columns: list[str]
    known_prefix_feature_columns: list[str]
    categorical_columns: list[str]
    categorical_levels: dict[str, list[str]]
    archetype_numeric_columns: list[str]
    archetype_impute_values: dict[str, float]
    archetype_scale_values: dict[str, float]
    archetype_cluster_centers: np.ndarray
    archetype_profiles: pd.DataFrame
    archetype_overall_table: pd.DataFrame
    archetype_state_table: pd.DataFrame
    archetype_city_table: pd.DataFrame
    archetype_star_bin_table: pd.DataFrame
    archetype_open_table: pd.DataFrame
    archetype_category_table: pd.DataFrame
    business_train_stats_table: pd.DataFrame
    top_city_values: list[str]
    top_category_values: list[str]
    manifest: dict[str, Any]
    config: dict[str, Any]
    known_prefix_embedding_root: str
    known_prefix_max_history_len: int
    known_prefix_target_bands: list[str]
    enabled_known_prefix_bands: list[str]
    routing_policy: dict[str, Any]
    business_embedding_table: pd.DataFrame | None = field(default=None)
    # columns: business_id, cf_item_bias — item bias from a pre-trained CF model
    cf_item_bias_table: pd.DataFrame | None = field(default=None)
    # columns: reviewer_user_id, business_id, stars — compact train reviews for friend lookup
    friend_lookup_table: pd.DataFrame | None = field(default=None)

    def __setstate__(self, state) -> None:
        """Handle backward-compatible deserialization when new slots are added after pickling.

        Python's default pickle state for slotted classes is (dict_state, slots_dict).
        We intercept this to set defaults for any slots that didn't exist when the object was saved.
        """
        # Standard pickle format for slotted objects: (None, {slot: value})
        if isinstance(state, tuple):
            _, slots_state = state
        else:
            # Fallback: state is a plain dict (some custom pickle paths)
            slots_state = state or {}
        for key, value in (slots_state or {}).items():
            try:
                object.__setattr__(self, key, value)
            except AttributeError:
                pass
        # Backward-compat defaults for slots added after initial release
        if not hasattr(self, "business_train_stats_table"):
            object.__setattr__(
                self,
                "business_train_stats_table",
                pd.DataFrame(columns=["business_id", "business_train_mean_rating", "business_train_rating_std"]),
            )
        if not hasattr(self, "business_embedding_table"):
            object.__setattr__(self, "business_embedding_table", None)
        if not hasattr(self, "cf_item_bias_table"):
            object.__setattr__(self, "cf_item_bias_table", None)
        if not hasattr(self, "friend_lookup_table"):
            object.__setattr__(self, "friend_lookup_table", None)


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(skipna=True))
    return value if np.isfinite(value) and value > 1e-6 else 1.0


def _bucket_user_activity(review_count: float) -> str:
    if not np.isfinite(review_count) or review_count <= 0:
        return "0"
    if review_count == 1:
        return "1"
    if review_count <= 5:
        return "2-5"
    if review_count <= 20:
        return "6-20"
    if review_count <= 100:
        return "21-100"
    return ">100"


def _bucket_user_reputation(average_stars: float) -> str:
    if not np.isfinite(average_stars):
        return "__missing__"
    if average_stars < 2.5:
        return "1.0-2.5"
    if average_stars < 3.5:
        return "2.5-3.5"
    if average_stars < 4.0:
        return "3.5-4.0"
    if average_stars < 4.5:
        return "4.0-4.5"
    return "4.5-5.0"


def _bucket_user_tenure(tenure_years: float) -> str:
    if not np.isfinite(tenure_years):
        return "__missing__"
    if tenure_years < 1.0:
        return "<1y"
    if tenure_years < 3.0:
        return "1-3y"
    if tenure_years < 6.0:
        return "3-6y"
    return ">6y"


def _bucket_business_star(stars: float) -> str:
    if not np.isfinite(stars):
        return "__missing__"
    if stars < 2.5:
        return "1.0-2.5"
    if stars < 3.5:
        return "2.5-3.5"
    if stars < 4.0:
        return "3.5-4.0"
    if stars < 4.5:
        return "4.0-4.5"
    return "4.5-5.0"


def _first_matching_category(categories: list[str], allowed: set[str]) -> str:
    for category in categories:
        category_text = _normalize_text(category, default="__other__")
        if category_text in allowed:
            return category_text
    return "__other__"


def _predict_nearest_archetype(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.zeros(len(x), dtype=np.int32)
    diff = x[:, None, :] - centers[None, :, :]
    distances = np.square(diff).sum(axis=2)
    return distances.argmin(axis=1).astype(np.int32)


def _predict_archetype_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return the squared Euclidean distance to the nearest archetype centroid (normalized units)."""
    if centers.size == 0:
        return np.zeros(len(x), dtype=np.float32)
    diff = x[:, None, :] - centers[None, :, :]
    distances = np.square(diff).sum(axis=2)
    return distances.min(axis=1).astype(np.float32)


def _build_friend_lookup_table(
    train_reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact (reviewer_user_id, business_id, stars) table for friend-business lookup."""
    reviews = train_reviews[["user_id", "business_id", "stars"]].copy()
    reviews.columns = ["reviewer_user_id", "business_id", "stars"]
    reviews["stars"] = reviews["stars"].astype(np.float32)
    return reviews.reset_index(drop=True)


def _compute_friend_business_features(
    frame: pd.DataFrame,
    prepared_users: pd.DataFrame,
    friend_lookup_table: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    """For each (user, business) row, compute mean and count of friend ratings for that business.

    Uses the friend strings in prepared_users['user_friends'] (comma-separated user IDs) and the
    compact train review table stored in the spec.  Falls back to global_mean when no friend has
    reviewed the business.
    """
    # Build business -> {reviewer_user_id: stars} dict from lookup table (vectorised)
    _bids = friend_lookup_table["business_id"].astype(str).tolist()
    _uids = friend_lookup_table["reviewer_user_id"].astype(str).tolist()
    _stars = friend_lookup_table["stars"].astype(float).tolist()
    biz_reviewers: dict[str, dict[str, float]] = {}
    for bid, uid, s in zip(_bids, _uids, _stars):
        if bid not in biz_reviewers:
            biz_reviewers[bid] = {}
        biz_reviewers[bid][uid] = s

    # Build user -> frozenset(friend_ids) dict only for users appearing in this frame
    frame_user_ids = set(frame["user"].astype(str).unique())
    user_friends_lookup: dict[str, frozenset[str]] = {}
    pu = prepared_users.set_index("user_id")
    for uid in frame_user_ids:
        if uid in pu.index:
            raw = pu.at[uid, "user_friends"]
            if raw and not (isinstance(raw, float) and pd.isna(raw)):
                parts = {f.strip() for f in str(raw).split(",") if f.strip()}
                user_friends_lookup[uid] = frozenset(parts)
        if uid not in user_friends_lookup:
            user_friends_lookup[uid] = frozenset()

    # Vectorised row computation
    users_arr = frame["user"].astype(str).tolist()
    items_arr = frame["item"].astype(str).tolist()
    n = len(frame)
    friend_means = np.full(n, global_mean, dtype=np.float32)
    friend_counts = np.zeros(n, dtype=np.float32)

    for i, (uid, bid) in enumerate(zip(users_arr, items_arr)):
        friends = user_friends_lookup.get(uid, frozenset())
        if friends and bid in biz_reviewers:
            ratings = [biz_reviewers[bid][f] for f in friends if f in biz_reviewers[bid]]
            if ratings:
                friend_means[i] = float(np.mean(ratings))
                friend_counts[i] = float(len(ratings))

    result = pd.DataFrame(index=frame.index)
    result["friend_business_mean"] = friend_means
    result["friend_business_count_log1p"] = np.log1p(friend_counts).astype(np.float32)
    result["friend_business_bias"] = (friend_means - global_mean).astype(np.float32)
    return result


def _assign_user_archetypes(
    prepared_users: pd.DataFrame,
    *,
    numeric_columns: list[str],
    impute_values: dict[str, float],
    scale_values: dict[str, float],
    cluster_centers: np.ndarray,
) -> pd.DataFrame:
    users = prepared_users.copy()
    numeric_frame = users[numeric_columns].copy()
    completeness = numeric_frame.notna().mean(axis=1).astype(np.float32)
    sparse_flag = completeness < 0.75

    impute_series = pd.Series(impute_values)
    scale_series = pd.Series(scale_values)
    normalized = ((numeric_frame.fillna(impute_series) - impute_series) / scale_series).to_numpy(dtype=np.float32)
    labels = _predict_nearest_archetype(normalized, cluster_centers)

    users["user_metadata_completeness"] = completeness
    users["user_metadata_sparse_flag"] = sparse_flag.astype(np.float32)
    users["user_archetype_distance"] = _predict_archetype_distances(normalized, cluster_centers)
    users["user_archetype_id"] = np.where(
        sparse_flag.to_numpy(),
        "__metadata_sparse__",
        pd.Series(labels, index=users.index).map(lambda label: f"archetype_{int(label):03d}").to_numpy(),
    )
    users["user_activity_bucket"] = users["user_review_count"].map(_bucket_user_activity).astype("string")
    users["user_reputation_bucket"] = users["user_average_stars"].map(_bucket_user_reputation).astype("string")
    users["user_tenure_bucket"] = users["user_tenure_years"].map(_bucket_user_tenure).astype("string")
    return users


def _fit_user_archetypes(
    prepared_users: pd.DataFrame,
    *,
    numeric_columns: list[str],
    n_clusters: int,
    random_seed: int,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, pd.DataFrame]:
    numeric_frame = prepared_users[numeric_columns].copy()
    impute_values = {
        column: float(numeric_frame[column].mean(skipna=True))
        if np.isfinite(float(numeric_frame[column].mean(skipna=True)))
        else 0.0
        for column in numeric_columns
    }
    scale_values = {column: _safe_std(numeric_frame[column]) for column in numeric_columns}
    impute_series = pd.Series(impute_values)
    scale_series = pd.Series(scale_values)
    normalized = ((numeric_frame.fillna(impute_series) - impute_series) / scale_series).to_numpy(dtype=np.float32)

    if len(normalized) == 0:
        centers = np.zeros((0, len(numeric_columns)), dtype=np.float32)
    else:
        effective_clusters = int(min(max(4, n_clusters), len(normalized)))
        kmeans = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=random_seed,
            batch_size=4096,
            n_init=10,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        kmeans.fit(normalized)
        centers = kmeans.cluster_centers_.astype(np.float32)

    users_with_archetypes = _assign_user_archetypes(
        prepared_users,
        numeric_columns=numeric_columns,
        impute_values=impute_values,
        scale_values=scale_values,
        cluster_centers=centers,
    )

    archetype_profiles = (
        users_with_archetypes.groupby("user_archetype_id", dropna=False)
        .agg(
            n_users=("user_id", "count"),
            metadata_completeness_mean=("user_metadata_completeness", "mean"),
            metadata_sparse_rate=("user_metadata_sparse_flag", "mean"),
            user_average_stars_mean=("user_average_stars", "mean"),
            user_review_count_log1p_mean=("user_review_count_log1p", "mean"),
            user_tenure_years_mean=("user_tenure_years", "mean"),
            user_total_votes_log1p_mean=("user_total_votes_log1p", "mean"),
            user_engagement_log1p_mean=("user_engagement_log1p", "mean"),
            user_friends_log1p_mean=("user_friends_log1p", "mean"),
            user_elite_years_count_mean=("user_elite_years_count", "mean"),
            user_compliment_log1p_total_mean=("user_compliment_log1p_total", "mean"),
            user_compliment_nonzero_count_mean=("user_compliment_nonzero_count", "mean"),
        )
        .reset_index()
        .sort_values(["user_archetype_id"], kind="stable")
        .reset_index(drop=True)
    )
    return impute_values, scale_values, centers, archetype_profiles


def _prepare_business_facets(
    prepared_businesses: pd.DataFrame,
    train_reviews: pd.DataFrame,
    *,
    max_top_cities: int,
    max_top_categories: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    businesses = prepared_businesses.copy()
    if "business_category_tokens" not in businesses.columns:
        businesses["business_category_tokens"] = businesses["business_categories"].apply(parse_categories)
    train_prepared = _prepare_reviews(train_reviews)
    train_join = train_prepared.merge(
        businesses[["business_id", "business_city", "business_category_tokens"]],
        left_on="item",
        right_on="business_id",
        how="left",
    )

    top_city_values = (
        train_join["business_city"].fillna("__missing__").astype(str).value_counts().head(max_top_cities).index.tolist()
    )
    exploded_categories = train_join["business_category_tokens"].explode().dropna().astype(str)
    top_category_values = exploded_categories.value_counts().head(max_top_categories).index.tolist()
    allowed_categories = set(top_category_values)
    allowed_cities = set(top_city_values)

    businesses["business_city_top"] = businesses["business_city"].map(
        lambda value: value if value in allowed_cities else "__other__"
    )
    businesses["business_primary_category_family"] = businesses["business_category_tokens"].apply(
        lambda values: _first_matching_category(values if isinstance(values, list) else [], allowed_categories)
    )
    businesses["business_star_bin"] = businesses["business_stars"].map(_bucket_business_star)
    return businesses, top_city_values, top_category_values


def _smoothed_group_stats(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    prefix: str,
    global_mean: float,
    alpha: float,
) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)["rating"].agg(["sum", "count"]).reset_index()
    grouped[f"{prefix}_support_count"] = grouped["count"].astype(np.float32)
    grouped[f"{prefix}_mean"] = ((grouped["sum"] + alpha * global_mean) / (grouped["count"] + alpha)).astype(np.float32)
    return grouped.drop(columns=["sum", "count"])


def _build_archetype_priors(
    train_reviews: pd.DataFrame,
    *,
    prepared_users: pd.DataFrame,
    prepared_businesses: pd.DataFrame,
    global_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _prepare_reviews(train_reviews)
    train = train.merge(
        prepared_users[["user_id", "user_archetype_id"]],
        left_on="user",
        right_on="user_id",
        how="left",
    )
    train = train.merge(
        prepared_businesses[
            [
                "business_id",
                "business_state",
                "business_city_top",
                "business_star_bin",
                "business_is_open",
                "business_primary_category_family",
            ]
        ],
        left_on="item",
        right_on="business_id",
        how="left",
    )
    train["user_archetype_id"] = train["user_archetype_id"].fillna("__metadata_sparse__")
    train["business_state"] = train["business_state"].fillna("__missing__")
    train["business_city_top"] = train["business_city_top"].fillna("__other__")
    train["business_star_bin"] = train["business_star_bin"].fillna("__missing__")
    train["business_primary_category_family"] = train["business_primary_category_family"].fillna("__other__")
    train["business_is_open"] = pd.to_numeric(train["business_is_open"], errors="coerce").fillna(0.0).astype(np.int32)

    overall = _smoothed_group_stats(
        train,
        group_cols=["user_archetype_id"],
        prefix="archetype_train",
        global_mean=global_mean,
        alpha=20.0,
    )
    state = _smoothed_group_stats(
        train,
        group_cols=["user_archetype_id", "business_state"],
        prefix="archetype_state",
        global_mean=global_mean,
        alpha=50.0,
    )
    city = _smoothed_group_stats(
        train,
        group_cols=["user_archetype_id", "business_city_top"],
        prefix="archetype_city",
        global_mean=global_mean,
        alpha=50.0,
    )
    star_bin = _smoothed_group_stats(
        train,
        group_cols=["user_archetype_id", "business_star_bin"],
        prefix="archetype_star_bin",
        global_mean=global_mean,
        alpha=50.0,
    )
    open_table = _smoothed_group_stats(
        train,
        group_cols=["user_archetype_id", "business_is_open"],
        prefix="archetype_open",
        global_mean=global_mean,
        alpha=50.0,
    )
    category = _smoothed_group_stats(
        train,
        group_cols=["user_archetype_id", "business_primary_category_family"],
        prefix="archetype_category",
        global_mean=global_mean,
        alpha=50.0,
    )
    return overall, state, city, star_bin, open_table, category


def _build_router_manifest() -> dict[str, Any]:
    return {
        "used_base_variables": [
            "user_average_stars",
            "business_stars",
            "user_minus_global_mean",
            "business_minus_global_mean",
            "user_business_metadata_gap",
            "user_review_count",
            "user_review_count_log1p",
            "user_engagement_log1p",
            "user_total_votes",
            "user_total_votes_log1p",
            "user_compliment_total",
            "user_compliment_log1p_total",
            "user_tenure_days",
            "user_tenure_years",
            "business_review_count",
            "business_review_count_log1p",
            "business_rating_per_review",
            "business_attributes_count",
            "business_attribute_true_count",
            "business_attribute_false_count",
            "business_attribute_string_count",
            "business_weekly_open_minutes",
            "review_total_votes",
            "review_useful",
            "review_funny",
            "review_cool",
            "review_days_since_train_start",
            "review_days_since_train_end",
        ],
        "archetype_seed_variables": ROUTER_ARCHETYPE_NUMERIC_COLUMNS,
        "generated_archetype_variables": [
            "user_archetype_id",
            "user_metadata_completeness",
            "user_metadata_sparse_flag",
            "user_activity_bucket",
            "user_reputation_bucket",
            "user_tenure_bucket",
            "business_city_top",
            "business_primary_category_family",
            "business_star_bin",
            "archetype_train_mean",
            "archetype_train_support_count",
            "archetype_train_bias",
            "archetype_state_mean",
            "archetype_state_support_count",
            "archetype_city_mean",
            "archetype_city_support_count",
            "archetype_star_bin_mean",
            "archetype_star_bin_support_count",
            "archetype_open_mean",
            "archetype_open_support_count",
            "archetype_category_mean",
            "archetype_category_support_count",
            "archetype_business_star_gap",
            "archetype_state_gap",
            "archetype_category_gap",
            "business_train_mean_rating",
            "business_train_rating_std",
            "business_train_vs_yelp_gap",
        ],
        "discarded_variables": {
            "excluded_for_router_training": ROUTER_EXCLUDED_TRAINING_COLUMNS,
            "not_modeled_identifiers": ["review_id", "user_id", "business_id", "user", "item"],
            "discarded_feature_families": [
                "raw_priors user_train_*",
                "raw_priors business_train_*",
                "raw_priors city/state/postal train priors",
                "deep user embeddings",
                "deep business embeddings",
                "raw free-text fields from attributes/categories/hours/friends/elite",
            ],
        },
    }


def fit_router_feature_spec(
    train_reviews: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    *,
    feature_set: str = RAW_CORE_FEATURE_SET,
    n_user_archetypes: int = 64,
    max_top_cities: int = 100,
    max_top_categories: int = 32,
    random_seed: int = 42,
    include_biz_train_stats: bool = True,
    business_embedding_table: pd.DataFrame | None = None,
    cf_item_bias_table: pd.DataFrame | None = None,
    include_friend_features: bool = True,
) -> RouterRawFeatureSpec:
    base_spec = fit_raw_feature_spec(
        train_reviews,
        users_df,
        businesses_df,
        feature_set=feature_set,
    )
    prepared_users = _prepare_users(users_df, train_max_timestamp=base_spec.train_max_timestamp)
    prepared_businesses = _prepare_businesses(businesses_df)

    (
        archetype_impute_values,
        archetype_scale_values,
        archetype_cluster_centers,
        archetype_profiles,
    ) = _fit_user_archetypes(
        prepared_users,
        numeric_columns=ROUTER_ARCHETYPE_NUMERIC_COLUMNS,
        n_clusters=n_user_archetypes,
        random_seed=random_seed,
    )
    prepared_users = _assign_user_archetypes(
        prepared_users,
        numeric_columns=ROUTER_ARCHETYPE_NUMERIC_COLUMNS,
        impute_values=archetype_impute_values,
        scale_values=archetype_scale_values,
        cluster_centers=archetype_cluster_centers,
    )
    prepared_businesses, top_city_values, top_category_values = _prepare_business_facets(
        prepared_businesses,
        train_reviews,
        max_top_cities=max_top_cities,
        max_top_categories=max_top_categories,
    )
    (
        archetype_overall_table,
        archetype_state_table,
        archetype_city_table,
        archetype_star_bin_table,
        archetype_open_table,
        archetype_category_table,
    ) = _build_archetype_priors(
        train_reviews,
        prepared_users=prepared_users,
        prepared_businesses=prepared_businesses,
        global_mean=base_spec.global_mean,
    )

    # Business training statistics — derived from train_reviews only (no leakage)
    if include_biz_train_stats:
        _biz_stats = (
            train_reviews.groupby("business_id")["stars"]
            .agg(business_train_mean_rating="mean", business_train_rating_std="std")
            .reset_index()
        )
        _biz_stats["business_train_mean_rating"] = _biz_stats["business_train_mean_rating"].fillna(base_spec.global_mean).astype(np.float32)
        _biz_stats["business_train_rating_std"] = _biz_stats["business_train_rating_std"].fillna(0.0).astype(np.float32)
    else:
        _biz_stats = pd.DataFrame(columns=["business_id", "business_train_mean_rating", "business_train_rating_std"])

    # Build friend-business lookup table from train reviews (compact: reviewer_user_id, business_id, stars)
    _friend_lookup = _build_friend_lookup_table(train_reviews) if include_friend_features else None

    temp_spec = RouterRawFeatureSpec(
        base_spec=base_spec,
        feature_columns=[],
        known_feature_columns=[],
        cold_feature_columns=[],
        known_prefix_feature_columns=[],
        categorical_columns=ROUTER_CATEGORICAL_COLUMNS.copy(),
        categorical_levels={},
        archetype_numeric_columns=ROUTER_ARCHETYPE_NUMERIC_COLUMNS.copy(),
        archetype_impute_values=archetype_impute_values,
        archetype_scale_values=archetype_scale_values,
        archetype_cluster_centers=archetype_cluster_centers,
        archetype_profiles=archetype_profiles,
        archetype_overall_table=archetype_overall_table,
        archetype_state_table=archetype_state_table,
        archetype_city_table=archetype_city_table,
        archetype_star_bin_table=archetype_star_bin_table,
        archetype_open_table=archetype_open_table,
        archetype_category_table=archetype_category_table,
        business_train_stats_table=_biz_stats,
        top_city_values=top_city_values,
        top_category_values=top_category_values,
        manifest=_build_router_manifest(),
        business_embedding_table=business_embedding_table,
        cf_item_bias_table=cf_item_bias_table,
        friend_lookup_table=_friend_lookup,
        config={
            "feature_set": feature_set,
            "n_user_archetypes": int(min(max(4, n_user_archetypes), max(4, len(prepared_users)))),
            "max_top_cities": max_top_cities,
            "max_top_categories": max_top_categories,
            "random_seed": random_seed,
        },
        known_prefix_embedding_root="",
        known_prefix_max_history_len=20,
        known_prefix_target_bands=[],
        enabled_known_prefix_bands=[],
        routing_policy={},
    )
    train_frame = build_router_feature_frame(train_reviews, users_df, businesses_df, temp_spec)


    metadata_columns = {"review_id", "user", "item", "rating", "review_date"}
    feature_columns = [
        column
        for column in train_frame.columns
        if column not in metadata_columns
        and (
            pd.api.types.is_numeric_dtype(train_frame[column])
            or isinstance(train_frame[column].dtype, pd.CategoricalDtype)
        )
    ]
    known_feature_columns = [column for column in feature_columns if column not in ROUTER_EXCLUDED_TRAINING_COLUMNS]
    cold_feature_columns = [column for column in feature_columns if column not in ROUTER_EXCLUDED_TRAINING_COLUMNS]
    categorical_levels = {
        column: train_frame[column].astype("string").fillna("__missing__").drop_duplicates().tolist()
        for column in ROUTER_CATEGORICAL_COLUMNS
        if column in train_frame.columns
    }

    return RouterRawFeatureSpec(
        base_spec=base_spec,
        feature_columns=feature_columns,
        known_feature_columns=known_feature_columns,
        cold_feature_columns=cold_feature_columns,
        known_prefix_feature_columns=[],
        categorical_columns=ROUTER_CATEGORICAL_COLUMNS.copy(),
        categorical_levels=categorical_levels,
        archetype_numeric_columns=ROUTER_ARCHETYPE_NUMERIC_COLUMNS.copy(),
        archetype_impute_values=archetype_impute_values,
        archetype_scale_values=archetype_scale_values,
        archetype_cluster_centers=archetype_cluster_centers,
        archetype_profiles=archetype_profiles,
        archetype_overall_table=archetype_overall_table,
        archetype_state_table=archetype_state_table,
        archetype_city_table=archetype_city_table,
        archetype_star_bin_table=archetype_star_bin_table,
        archetype_open_table=archetype_open_table,
        archetype_category_table=archetype_category_table,
        business_train_stats_table=_biz_stats,
        top_city_values=top_city_values,
        top_category_values=top_category_values,
        manifest=_build_router_manifest(),
        config=temp_spec.config,
        known_prefix_embedding_root="",
        known_prefix_max_history_len=20,
        known_prefix_target_bands=[],
        enabled_known_prefix_bands=[],
        routing_policy={},
        business_embedding_table=business_embedding_table,
        cf_item_bias_table=cf_item_bias_table,
        friend_lookup_table=_friend_lookup,
    )


def build_router_feature_frame(
    reviews_df: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    spec: RouterRawFeatureSpec,
) -> pd.DataFrame:
    frame = build_raw_feature_frame(reviews_df, users_df, businesses_df, spec.base_spec).copy()
    users = _prepare_users(users_df, train_max_timestamp=spec.base_spec.train_max_timestamp)
    users = _assign_user_archetypes(
        users,
        numeric_columns=spec.archetype_numeric_columns,
        impute_values=spec.archetype_impute_values,
        scale_values=spec.archetype_scale_values,
        cluster_centers=spec.archetype_cluster_centers,
    )
    businesses = _prepare_businesses(businesses_df)
    if "business_category_tokens" not in businesses.columns:
        businesses["business_category_tokens"] = businesses["business_categories"].apply(parse_categories)
    allowed_cities = set(spec.top_city_values)
    allowed_categories = set(spec.top_category_values)
    businesses["business_city_top"] = businesses["business_city"].map(
        lambda value: value if value in allowed_cities else "__other__"
    )
    businesses["business_primary_category_family"] = businesses["business_category_tokens"].apply(
        lambda values: _first_matching_category(values if isinstance(values, list) else [], allowed_categories)
    )
    businesses["business_star_bin"] = businesses["business_stars"].map(_bucket_business_star)

    frame = frame.merge(
        users[
            [
                "user_id",
                "user_archetype_id",
                "user_metadata_completeness",
                "user_metadata_sparse_flag",
                "user_archetype_distance",
                "user_activity_bucket",
                "user_reputation_bucket",
                "user_tenure_bucket",
            ]
        ],
        left_on="user",
        right_on="user_id",
        how="left",
    ).drop(columns=["user_id"], errors="ignore")
    frame = frame.merge(
        businesses[
            [
                "business_id",
                "business_city_top",
                "business_primary_category_family",
                "business_star_bin",
            ]
        ],
        left_on="item",
        right_on="business_id",
        how="left",
    ).drop(columns=["business_id"], errors="ignore")

    global_mean = spec.base_spec.global_mean
    frame["user_archetype_id"] = frame["user_archetype_id"].fillna("__metadata_sparse__")
    frame["user_metadata_completeness"] = frame["user_metadata_completeness"].fillna(0.0).astype(np.float32)
    frame["user_metadata_sparse_flag"] = frame["user_metadata_sparse_flag"].fillna(1.0).astype(np.float32)
    frame["user_archetype_distance"] = frame["user_archetype_distance"].fillna(0.0).astype(np.float32)
    frame["user_activity_bucket"] = frame["user_activity_bucket"].fillna("__missing__")
    frame["user_reputation_bucket"] = frame["user_reputation_bucket"].fillna("__missing__")
    frame["user_tenure_bucket"] = frame["user_tenure_bucket"].fillna("__missing__")
    frame["business_city_top"] = frame["business_city_top"].fillna("__other__")
    frame["business_primary_category_family"] = frame["business_primary_category_family"].fillna("__other__")
    frame["business_star_bin"] = frame["business_star_bin"].fillna("__missing__")
    frame["business_is_open"] = pd.to_numeric(frame["business_is_open"], errors="coerce").fillna(0.0).astype(np.int32)

    frame = frame.merge(spec.archetype_overall_table, on="user_archetype_id", how="left")
    frame = frame.merge(spec.archetype_state_table, on=["user_archetype_id", "business_state"], how="left")
    frame = frame.merge(spec.archetype_city_table, on=["user_archetype_id", "business_city_top"], how="left")
    frame = frame.merge(spec.archetype_star_bin_table, on=["user_archetype_id", "business_star_bin"], how="left")
    frame = frame.merge(spec.archetype_open_table, on=["user_archetype_id", "business_is_open"], how="left")
    frame = frame.merge(
        spec.archetype_category_table,
        on=["user_archetype_id", "business_primary_category_family"],
        how="left",
    )

    for prefix in [
        "archetype_train",
        "archetype_state",
        "archetype_city",
        "archetype_star_bin",
        "archetype_open",
        "archetype_category",
    ]:
        frame[f"{prefix}_support_count"] = frame[f"{prefix}_support_count"].fillna(0.0).astype(np.float32)
        frame[f"{prefix}_mean"] = frame[f"{prefix}_mean"].fillna(global_mean).astype(np.float32)

    frame["archetype_train_bias"] = (frame["archetype_train_mean"] - global_mean).astype(np.float32)
    frame["archetype_business_star_gap"] = (frame["archetype_train_mean"] - frame["business_stars"]).astype(np.float32)
    frame["archetype_state_gap"] = (frame["archetype_state_mean"] - frame["business_stars"]).astype(np.float32)
    frame["archetype_category_gap"] = (frame["archetype_category_mean"] - frame["business_stars"]).astype(np.float32)

    # Merge pre-computed business content embeddings (PCA-reduced, optional)
    if spec.business_embedding_table is not None and len(spec.business_embedding_table) > 0:
        emb_cols = [c for c in spec.business_embedding_table.columns if c != "business_id"]
        frame = frame.merge(
            spec.business_embedding_table,
            left_on="item",
            right_on="business_id",
            how="left",
        ).drop(columns=["business_id"], errors="ignore")
        for col in emb_cols:
            frame[col] = frame[col].fillna(0.0).astype(np.float32)

    # Merge business training statistics (mean rating + rating std from train_reviews)
    if spec.business_train_stats_table is not None and len(spec.business_train_stats_table) > 0:
        frame = frame.merge(
            spec.business_train_stats_table[["business_id", "business_train_mean_rating", "business_train_rating_std"]],
            left_on="item",
            right_on="business_id",
            how="left",
        ).drop(columns=["business_id"], errors="ignore")
        frame["business_train_mean_rating"] = frame["business_train_mean_rating"].fillna(global_mean).astype(np.float32)
        frame["business_train_rating_std"] = frame["business_train_rating_std"].fillna(0.0).astype(np.float32)
        # Gap between training crowd consensus and Yelp all-time rating
        frame["business_train_vs_yelp_gap"] = (
            frame["business_train_mean_rating"] - frame["business_stars"]
        ).astype(np.float32)

    # CF item bias — regularised item quality estimate from a pre-trained CF model
    if spec.cf_item_bias_table is not None and len(spec.cf_item_bias_table) > 0:
        frame = frame.merge(
            spec.cf_item_bias_table[["business_id", "cf_item_bias"]],
            left_on="item",
            right_on="business_id",
            how="left",
        ).drop(columns=["business_id"], errors="ignore")
        frame["cf_item_bias"] = frame["cf_item_bias"].fillna(0.0).astype(np.float32)
        frame["cf_item_prediction"] = (global_mean + frame["cf_item_bias"]).astype(np.float32)

    # Friend-business features — mean rating of user's friends for the target business
    if spec.friend_lookup_table is not None and len(spec.friend_lookup_table) > 0:
        friend_features = _compute_friend_business_features(
            frame,
            prepared_users=users,
            friend_lookup_table=spec.friend_lookup_table,
            global_mean=global_mean,
        )
        for col in friend_features.columns:
            frame[col] = friend_features[col].values

    for column in spec.categorical_columns:
        if column not in frame.columns:
            continue
        if spec.categorical_levels:
            frame[column] = pd.Categorical(
                frame[column].astype("string").fillna("__missing__"),
                categories=spec.categorical_levels.get(column, ["__missing__"]),
            )
        else:
            frame[column] = pd.Categorical(frame[column].astype("string").fillna("__missing__"))

    metadata_columns = {"review_id", "user", "item", "rating", "review_date"}
    if spec.feature_columns:
        feature_columns = [column for column in spec.feature_columns if column in frame.columns]
    else:
        feature_columns = [
            column
            for column in frame.columns
            if column not in metadata_columns
            and (
                pd.api.types.is_numeric_dtype(frame[column])
                or isinstance(frame[column].dtype, pd.CategoricalDtype)
            )
        ]
    return frame[["review_id", "user", "item", "rating", "review_date", *feature_columns]].copy()
