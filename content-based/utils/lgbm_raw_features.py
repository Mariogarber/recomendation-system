from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from .business_features import extract_hours_features, parse_attributes, parse_categories
from .io import canonicalize_reviews


RAW_CORE_FEATURE_SET = "raw_core"
RAW_PRIORS_FEATURE_SET = "raw_priors"
SUPPORTED_FEATURE_SETS = {RAW_CORE_FEATURE_SET, RAW_PRIORS_FEATURE_SET}

USER_SOURCE_COLUMNS = [
    "review_count",
    "yelping_since",
    "useful",
    "funny",
    "cool",
    "elite",
    "friends",
    "fans",
    "average_stars",
    "compliment_hot",
    "compliment_more",
    "compliment_profile",
    "compliment_cute",
    "compliment_list",
    "compliment_note",
    "compliment_plain",
    "compliment_cool",
    "compliment_funny",
    "compliment_writer",
    "compliment_photos",
]

BUSINESS_SOURCE_COLUMNS = [
    "city",
    "state",
    "postal_code",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "is_open",
    "attributes",
    "categories",
    "hours",
]

CATEGORICAL_COLUMNS = [
    "business_city",
    "business_state",
    "business_postal_code",
    "business_train_support_bucket",
    "user_train_history_bucket",
]


@dataclass(slots=True)
class RawLGBMFeatureSpec:
    feature_set: str
    global_mean: float
    train_min_timestamp: pd.Timestamp
    train_max_timestamp: pd.Timestamp
    feature_columns: list[str]
    categorical_columns: list[str]
    categorical_levels: dict[str, list[str]]
    user_priors_table: pd.DataFrame
    business_priors_table: pd.DataFrame
    city_priors_table: pd.DataFrame
    state_priors_table: pd.DataFrame
    postal_priors_table: pd.DataFrame


def _normalize_text(value: Any, default: str = "__missing__") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    return text if text else default


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(np.float32)


def _support_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if not np.isfinite(numeric) or numeric <= 0.0:
        return "0"
    if numeric <= 1.0:
        return "1"
    if numeric <= 2.0:
        return "2"
    if numeric <= 3.0:
        return "3"
    if numeric <= 5.0:
        return "4-5"
    if numeric <= 10.0:
        return "6-10"
    if numeric <= 20.0:
        return "11-20"
    return ">20"


def _count_friends(value: Any) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() == "none":
        return 0.0
    return float(len([part for part in text.split(",") if part.strip()]))


def _count_elite_years(value: Any) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() == "none":
        return 0.0
    return float(len([part for part in re.split(r"[,;]", text) if part.strip()]))


def _required_subset(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in required:
        if column not in out.columns:
            out[column] = np.nan
    return out[required].copy()


def _prepare_users(users_df: pd.DataFrame, *, train_max_timestamp: pd.Timestamp) -> pd.DataFrame:
    users = _required_subset(users_df, ["user_id", *USER_SOURCE_COLUMNS])
    users = users.rename(columns={column: f"user_{column}" for column in users.columns if column != "user_id"})

    users["user_yelping_since"] = pd.to_datetime(users["user_yelping_since"], errors="coerce")
    users["user_tenure_days"] = (
        train_max_timestamp - users["user_yelping_since"].fillna(train_max_timestamp)
    ).dt.total_seconds().div(86400.0).fillna(0.0).clip(lower=0.0).astype(np.float32)
    users["user_tenure_years"] = (users["user_tenure_days"] / 365.25).astype(np.float32)
    users["user_yelping_since_missing"] = users["user_yelping_since"].isna().astype(np.float32)

    for column in ["user_review_count", "user_useful", "user_funny", "user_cool", "user_fans"]:
        users[column] = _safe_numeric(users[column])
    users["user_average_stars"] = _safe_numeric(users["user_average_stars"], default=np.nan)
    users["user_review_count_log1p"] = np.log1p(users["user_review_count"]).astype(np.float32)
    users["user_total_votes"] = (users["user_useful"] + users["user_funny"] + users["user_cool"]).astype(np.float32)
    users["user_total_votes_log1p"] = np.log1p(users["user_total_votes"]).astype(np.float32)
    users["user_engagement_log1p"] = np.log1p(users["user_total_votes"] + users["user_fans"]).astype(np.float32)

    users["user_elite_years_count"] = users["user_elite"].apply(_count_elite_years).astype(np.float32)
    users["user_elite_any"] = (users["user_elite_years_count"] > 0).astype(np.float32)
    users["user_friends_count"] = users["user_friends"].apply(_count_friends).astype(np.float32)
    users["user_friends_log1p"] = np.log1p(users["user_friends_count"]).astype(np.float32)

    compliment_columns = [column for column in users.columns if column.startswith("user_compliment_")]
    for column in compliment_columns:
        users[column] = _safe_numeric(users[column])
    users["user_compliment_total"] = users[compliment_columns].sum(axis=1).astype(np.float32)
    users["user_compliment_nonzero_count"] = (users[compliment_columns] > 0).sum(axis=1).astype(np.float32)
    users["user_compliment_log1p_total"] = np.log1p(users["user_compliment_total"]).astype(np.float32)

    return users


def _prepare_businesses(businesses_df: pd.DataFrame) -> pd.DataFrame:
    businesses = _required_subset(businesses_df, ["business_id", *BUSINESS_SOURCE_COLUMNS])
    businesses = businesses.rename(columns={column: f"business_{column}" for column in businesses.columns if column != "business_id"})

    businesses["business_city"] = businesses["business_city"].map(_normalize_text)
    businesses["business_state"] = businesses["business_state"].map(_normalize_text)
    businesses["business_postal_code"] = businesses["business_postal_code"].map(_normalize_text)
    businesses["business_latitude"] = _safe_numeric(businesses["business_latitude"], default=np.nan)
    businesses["business_longitude"] = _safe_numeric(businesses["business_longitude"], default=np.nan)
    businesses["business_stars"] = _safe_numeric(businesses["business_stars"], default=np.nan)
    businesses["business_review_count"] = _safe_numeric(businesses["business_review_count"])
    businesses["business_is_open"] = _safe_numeric(businesses["business_is_open"])
    businesses["business_review_count_log1p"] = np.log1p(businesses["business_review_count"]).astype(np.float32)
    businesses["business_has_latitude"] = businesses["business_latitude"].notna().astype(np.float32)
    businesses["business_has_longitude"] = businesses["business_longitude"].notna().astype(np.float32)

    categories = businesses["business_categories"].apply(parse_categories)
    businesses["business_categories_count"] = categories.apply(len).astype(np.float32)
    businesses["business_categories_missing"] = businesses["business_categories"].isna().astype(np.float32)
    businesses["business_has_categories"] = (~businesses["business_categories_missing"].astype(bool)).astype(np.float32)

    attributes = businesses["business_attributes"].apply(parse_attributes)
    businesses["business_attributes_count"] = attributes.apply(len).astype(np.float32)
    businesses["business_attributes_missing"] = businesses["business_attributes"].isna().astype(np.float32)
    businesses["business_has_attributes"] = (~businesses["business_attributes_missing"].astype(bool)).astype(np.float32)
    businesses["business_attribute_true_count"] = attributes.apply(lambda mapping: float(sum(1 for value in mapping.values() if value is True))).astype(np.float32)
    businesses["business_attribute_false_count"] = attributes.apply(lambda mapping: float(sum(1 for value in mapping.values() if value is False))).astype(np.float32)
    businesses["business_attribute_string_count"] = attributes.apply(lambda mapping: float(sum(1 for value in mapping.values() if isinstance(value, str)))).astype(np.float32)

    hours = businesses["business_hours"].apply(extract_hours_features)
    hours_df = pd.DataFrame(hours.tolist(), index=businesses.index)
    for column in hours_df.columns:
        businesses[f"business_{column}"] = _safe_numeric(hours_df[column])
    businesses["business_hours_missing"] = businesses["business_hours"].isna().astype(np.float32)
    businesses["business_has_hours"] = (~businesses["business_hours_missing"].astype(bool)).astype(np.float32)
    businesses["business_rating_per_review"] = (
        businesses["business_stars"] / np.maximum(businesses["business_review_count"], 1.0)
    ).astype(np.float32)
    businesses["business_geo_abs"] = (
        np.abs(businesses["business_latitude"].fillna(0.0)) + np.abs(businesses["business_longitude"].fillna(0.0))
    ).astype(np.float32)

    return businesses


def _prepare_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    reviews = canonicalize_reviews(reviews_df)
    if "timestamp" not in reviews.columns:
        raise ValueError("Review table is missing a timestamp/date column.")

    out = reviews.copy()
    if "review_id" not in out.columns:
        out["review_id"] = np.arange(len(out), dtype=np.int64)
    if "rating" not in out.columns:
        out["rating"] = np.nan
    out["review_date"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["review_useful"] = _safe_numeric(out.get("useful", pd.Series(index=out.index)))
    out["review_funny"] = _safe_numeric(out.get("funny", pd.Series(index=out.index)))
    out["review_cool"] = _safe_numeric(out.get("cool", pd.Series(index=out.index)))
    out["review_total_votes"] = (out["review_useful"] + out["review_funny"] + out["review_cool"]).astype(np.float32)
    # Yelp vote columns can contain -1 sentinels, so clamp before log1p to avoid -inf features.
    review_useful_nonnegative = out["review_useful"].clip(lower=0.0)
    review_funny_nonnegative = out["review_funny"].clip(lower=0.0)
    review_cool_nonnegative = out["review_cool"].clip(lower=0.0)
    out["review_useful_log1p"] = np.log1p(review_useful_nonnegative).astype(np.float32)
    out["review_funny_log1p"] = np.log1p(review_funny_nonnegative).astype(np.float32)
    out["review_cool_log1p"] = np.log1p(review_cool_nonnegative).astype(np.float32)
    out["review_year"] = out["review_date"].dt.year.fillna(0).astype(np.float32)
    out["review_month"] = out["review_date"].dt.month.fillna(0).astype(np.float32)
    out["review_weekday"] = out["review_date"].dt.weekday.fillna(0).astype(np.float32)
    out["review_hour"] = out["review_date"].dt.hour.fillna(0).astype(np.float32)
    out["review_weekend_flag"] = out["review_weekday"].isin([5.0, 6.0]).astype(np.float32)
    out["review_evening_flag"] = out["review_hour"].between(18.0, 23.0, inclusive="both").astype(np.float32)
    return out[[
        "review_id",
        "user",
        "item",
        "rating",
        "review_date",
        "review_useful",
        "review_funny",
        "review_cool",
        "review_total_votes",
        "review_useful_log1p",
        "review_funny_log1p",
        "review_cool_log1p",
        "review_year",
        "review_month",
        "review_weekday",
        "review_hour",
        "review_weekend_flag",
        "review_evening_flag",
    ]].copy()


def _build_priors(
    train_reviews: pd.DataFrame,
    businesses: pd.DataFrame,
    *,
    train_min_timestamp: pd.Timestamp,
    train_max_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    train = _prepare_reviews(train_reviews)
    train = train.merge(
        businesses[["business_id", "business_city", "business_state", "business_postal_code"]],
        left_on="item",
        right_on="business_id",
        how="left",
    )
    global_mean = float(train["rating"].mean())

    def _group_priors(id_col: str, prefix: str) -> pd.DataFrame:
        grouped = train.groupby(id_col, dropna=False)
        priors = grouped["rating"].agg(["count", "mean", "std"]).reset_index()
        priors = priors.rename(
            columns={
                id_col: f"{prefix}_id",
                "count": f"{prefix}_train_count",
                "mean": f"{prefix}_train_mean",
                "std": f"{prefix}_train_std",
            }
        )
        time_stats = grouped["review_date"].agg(["min", "max"]).reset_index()
        time_stats = time_stats.rename(
            columns={
                id_col: f"{prefix}_id",
                "min": f"{prefix}_first_seen",
                "max": f"{prefix}_last_seen",
            }
        )
        priors = priors.merge(time_stats, on=f"{prefix}_id", how="left")
        priors[f"{prefix}_train_std"] = priors[f"{prefix}_train_std"].fillna(0.0).astype(np.float32)
        priors[f"{prefix}_train_bias"] = (priors[f"{prefix}_train_mean"] - global_mean).astype(np.float32)
        priors[f"{prefix}_train_span_days"] = (
            pd.to_datetime(priors[f"{prefix}_last_seen"], errors="coerce")
            - pd.to_datetime(priors[f"{prefix}_first_seen"], errors="coerce")
        ).dt.total_seconds().div(86400.0).fillna(0.0).clip(lower=0.0).astype(np.float32)
        priors[f"{prefix}_train_days_since_last_review"] = (
            train_max_timestamp - pd.to_datetime(priors[f"{prefix}_last_seen"], errors="coerce")
        ).dt.total_seconds().div(86400.0).fillna(0.0).clip(lower=0.0).astype(np.float32)
        return priors.drop(columns=[f"{prefix}_first_seen", f"{prefix}_last_seen"])

    user_priors = _group_priors("user", "user")
    business_priors = _group_priors("item", "business")

    city_priors = train.groupby("business_city", dropna=False)["rating"].agg(["count", "mean"]).reset_index()
    city_priors = city_priors.rename(columns={"count": "city_train_count", "mean": "city_train_mean"})
    state_priors = train.groupby("business_state", dropna=False)["rating"].agg(["count", "mean"]).reset_index()
    state_priors = state_priors.rename(columns={"count": "state_train_count", "mean": "state_train_mean"})
    postal_priors = train.groupby("business_postal_code", dropna=False)["rating"].agg(["count", "mean"]).reset_index()
    postal_priors = postal_priors.rename(columns={"count": "postal_train_count", "mean": "postal_train_mean"})
    return user_priors, business_priors, city_priors, state_priors, postal_priors, global_mean


def fit_raw_feature_spec(
    train_reviews: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    *,
    feature_set: str = RAW_PRIORS_FEATURE_SET,
) -> RawLGBMFeatureSpec:
    if feature_set not in SUPPORTED_FEATURE_SETS:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    reviews = _prepare_reviews(train_reviews)
    train_min_timestamp = pd.to_datetime(reviews["review_date"], errors="coerce").min()
    train_max_timestamp = pd.to_datetime(reviews["review_date"], errors="coerce").max()
    prepared_users = _prepare_users(users_df, train_max_timestamp=train_max_timestamp)
    prepared_businesses = _prepare_businesses(businesses_df)
    user_priors, business_priors, city_priors, state_priors, postal_priors, global_mean = _build_priors(
        train_reviews,
        prepared_businesses,
        train_min_timestamp=train_min_timestamp,
        train_max_timestamp=train_max_timestamp,
    )

    temp_spec = RawLGBMFeatureSpec(
        feature_set=feature_set,
        global_mean=global_mean,
        train_min_timestamp=train_min_timestamp,
        train_max_timestamp=train_max_timestamp,
        feature_columns=[],
        categorical_columns=CATEGORICAL_COLUMNS.copy(),
        categorical_levels={},
        user_priors_table=user_priors,
        business_priors_table=business_priors,
        city_priors_table=city_priors,
        state_priors_table=state_priors,
        postal_priors_table=postal_priors,
    )
    train_frame = build_raw_feature_frame(train_reviews, users_df, businesses_df, temp_spec)

    metadata_columns = {"review_id", "user", "item", "rating", "review_date", "user_id", "business_id"}
    feature_columns = [
        column
        for column in train_frame.columns
        if column not in metadata_columns
        and (
            pd.api.types.is_numeric_dtype(train_frame[column])
            or isinstance(train_frame[column].dtype, pd.CategoricalDtype)
        )
    ]
    categorical_levels = {
        column: train_frame[column].astype("string").fillna("__missing__").drop_duplicates().tolist()
        for column in CATEGORICAL_COLUMNS
    }
    return RawLGBMFeatureSpec(
        feature_set=feature_set,
        global_mean=global_mean,
        train_min_timestamp=train_min_timestamp,
        train_max_timestamp=train_max_timestamp,
        feature_columns=feature_columns,
        categorical_columns=CATEGORICAL_COLUMNS.copy(),
        categorical_levels=categorical_levels,
        user_priors_table=user_priors,
        business_priors_table=business_priors,
        city_priors_table=city_priors,
        state_priors_table=state_priors,
        postal_priors_table=postal_priors,
    )


def build_raw_feature_frame(
    reviews_df: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    spec: RawLGBMFeatureSpec,
) -> pd.DataFrame:
    if spec.feature_set not in SUPPORTED_FEATURE_SETS:
        raise ValueError(f"Unsupported feature_set in spec: {spec.feature_set}")

    reviews = _prepare_reviews(reviews_df)
    users = _prepare_users(users_df, train_max_timestamp=spec.train_max_timestamp)
    businesses = _prepare_businesses(businesses_df)

    frame = reviews.merge(users, left_on="user", right_on="user_id", how="left", validate="many_to_one")
    frame = frame.merge(businesses, left_on="item", right_on="business_id", how="left", validate="many_to_one")

    global_mean = spec.global_mean
    user_train_ids = set(spec.user_priors_table["user_id"].astype(str))
    business_train_ids = set(spec.business_priors_table["business_id"].astype(str))
    frame["user_known_in_train"] = frame["user"].astype(str).isin(user_train_ids).astype(np.float32)
    frame["business_known_in_train"] = frame["item"].astype(str).isin(business_train_ids).astype(np.float32)
    frame["review_days_since_train_start"] = (
        frame["review_date"] - spec.train_min_timestamp
    ).dt.total_seconds().div(86400.0).fillna(0.0).astype(np.float32)
    # NOTE: clipped to 0 — negative values occur when spec.train_max_timestamp is
    # computed from 100% data (submission) vs 80% split (validation), causing
    # distribution shift.  Clipping makes the feature stable: 0 = "within training
    # window", positive = "days after train cutoff".
    frame["review_days_since_train_end"] = (
        frame["review_date"] - spec.train_max_timestamp
    ).dt.total_seconds().div(86400.0).fillna(0.0).clip(lower=0.0).astype(np.float32)

    frame["user_average_stars"] = frame["user_average_stars"].fillna(global_mean).astype(np.float32)
    frame["user_tenure_days"] = frame["user_tenure_days"].fillna(0.0).astype(np.float32)
    frame["user_tenure_years"] = frame["user_tenure_years"].fillna(0.0).astype(np.float32)
    frame["business_stars"] = frame["business_stars"].fillna(global_mean).astype(np.float32)
    frame["business_latitude"] = frame["business_latitude"].fillna(0.0).astype(np.float32)
    frame["business_longitude"] = frame["business_longitude"].fillna(0.0).astype(np.float32)
    frame["business_rating_per_review"] = frame["business_rating_per_review"].fillna(global_mean).astype(np.float32)
    frame["business_geo_abs"] = frame["business_geo_abs"].fillna(0.0).astype(np.float32)
    frame["business_train_support_bucket"] = "__missing__"
    frame["user_train_history_bucket"] = "__missing__"

    if spec.feature_set == RAW_PRIORS_FEATURE_SET:
        frame = frame.merge(spec.user_priors_table, on="user_id", how="left")
        frame = frame.merge(spec.business_priors_table, on="business_id", how="left")
        frame = frame.merge(spec.city_priors_table, on="business_city", how="left")
        frame = frame.merge(spec.state_priors_table, on="business_state", how="left")
        frame = frame.merge(spec.postal_priors_table, on="business_postal_code", how="left")
        frame["user_train_count"] = frame["user_train_count"].fillna(0.0).astype(np.float32)
        frame["user_train_mean"] = frame["user_train_mean"].fillna(global_mean).astype(np.float32)
        frame["user_train_std"] = frame["user_train_std"].fillna(0.0).astype(np.float32)
        frame["user_train_bias"] = frame["user_train_bias"].fillna(0.0).astype(np.float32)
        frame["user_train_span_days"] = frame["user_train_span_days"].fillna(0.0).astype(np.float32)
        frame["user_train_days_since_last_review"] = frame["user_train_days_since_last_review"].fillna(0.0).astype(np.float32)
        frame["business_train_count"] = frame["business_train_count"].fillna(0.0).astype(np.float32)
        frame["business_train_mean"] = frame["business_train_mean"].fillna(global_mean).astype(np.float32)
        frame["business_train_std"] = frame["business_train_std"].fillna(0.0).astype(np.float32)
        frame["business_train_bias"] = frame["business_train_bias"].fillna(0.0).astype(np.float32)
        frame["business_train_span_days"] = frame["business_train_span_days"].fillna(0.0).astype(np.float32)
        frame["business_train_days_since_last_review"] = frame["business_train_days_since_last_review"].fillna(0.0).astype(np.float32)
        frame["city_train_count"] = frame["city_train_count"].fillna(0.0).astype(np.float32)
        frame["city_train_mean"] = frame["city_train_mean"].fillna(global_mean).astype(np.float32)
        frame["state_train_count"] = frame["state_train_count"].fillna(0.0).astype(np.float32)
        frame["state_train_mean"] = frame["state_train_mean"].fillna(global_mean).astype(np.float32)
        frame["postal_train_count"] = frame["postal_train_count"].fillna(0.0).astype(np.float32)
        frame["postal_train_mean"] = frame["postal_train_mean"].fillna(global_mean).astype(np.float32)
        frame["user_business_train_gap"] = (frame["user_train_mean"] - frame["business_train_mean"]).astype(np.float32)
        frame["user_business_train_bias_sum"] = (frame["user_train_bias"] + frame["business_train_bias"]).astype(np.float32)
        frame["business_train_support_bucket"] = frame["business_train_count"].map(_support_bucket)
        frame["user_train_history_bucket"] = frame["user_train_count"].map(_support_bucket)
        frame["item_is_new"] = (frame["business_train_count"] <= 0.0).astype(np.float32)
        frame["item_support_log1p"] = np.log1p(frame["business_train_count"]).astype(np.float32)
        frame["user_history_count_log1p"] = np.log1p(frame["user_train_count"]).astype(np.float32)
        frame["user_history_is_2"] = (frame["user_train_count"] == 2.0).astype(np.float32)
        frame["user_history_is_3"] = (frame["user_train_count"] == 3.0).astype(np.float32)
        frame["user_history_is_4"] = (frame["user_train_count"] == 4.0).astype(np.float32)
        frame["user_history_is_5"] = (frame["user_train_count"] == 5.0).astype(np.float32)
        frame["user_history_is_short_2_3"] = frame["user_train_count"].between(2.0, 3.0, inclusive="both").astype(np.float32)
        frame["user_history_is_short_4_5"] = frame["user_train_count"].between(4.0, 5.0, inclusive="both").astype(np.float32)
        frame["user_item_support_interaction"] = (
            np.log1p(frame["user_train_count"]) * np.log1p(frame["business_train_count"])
        ).astype(np.float32)
        frame["item_support_per_user_history"] = (
            frame["business_train_count"] / np.maximum(frame["user_train_count"], 1.0)
        ).astype(np.float32)
        frame["item_support_x_known_user"] = (
            frame["business_train_count"] * frame["user_known_in_train"]
        ).astype(np.float32)

    frame["user_minus_global_mean"] = (frame["user_average_stars"] - global_mean).astype(np.float32)
    frame["business_minus_global_mean"] = (frame["business_stars"] - global_mean).astype(np.float32)
    frame["user_business_metadata_gap"] = (frame["user_average_stars"] - frame["business_stars"]).astype(np.float32)
    frame["user_review_count_x_business_review_count"] = (
        np.log1p(frame["user_review_count"]) * np.log1p(frame["business_review_count"])
    ).astype(np.float32)

    for column in CATEGORICAL_COLUMNS:
        frame[column] = pd.Categorical(
            frame[column].map(_normalize_text),
            categories=spec.categorical_levels.get(column, ["__missing__"]),
        )

    metadata_columns = {"review_id", "user", "item", "rating", "review_date", "user_id", "business_id"}
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
    out = frame[["review_id", "user", "item", "rating", "review_date", *feature_columns]].copy()
    return out


def history_band_from_count(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return ">20"


def build_train_user_stars(train_reviews_df: pd.DataFrame, global_mean: float) -> dict:
    """Compute per-user mean stars from train reviews only (no leakage from Yelp all-time aggregate)."""
    user_col = "user_id" if "user_id" in train_reviews_df.columns else "user"
    rating_col = "stars" if "stars" in train_reviews_df.columns else "rating"
    return (
        train_reviews_df.groupby(user_col)[rating_col]
        .mean()
        .astype(float)
        .to_dict()
    )


def build_temporal_loo_user_stars(
    reviews_df: pd.DataFrame,
    global_mean: float,
    date_col: str = "date",
) -> pd.Series:
    """Per-row temporal LOO user average stars.

    For each review, computes the user's mean stars from all their *previous* reviews
    (strictly before this review's date).  Cold rows (first review for a user) fall back
    to ``global_mean``.  This is causally correct: it mirrors exactly what the model
    would see at inference time and avoids within-sample target encoding.

    Returns a pd.Series aligned to ``reviews_df.index``.
    """
    user_col = "user_id" if "user_id" in reviews_df.columns else "user"
    rating_col = "stars" if "stars" in reviews_df.columns else "rating"

    df = reviews_df[[user_col, date_col, rating_col]].copy()
    df = df.sort_values([user_col, date_col], kind="stable")

    # Expanding cumulative sum and count per user — shift by 1 so current row is excluded.
    df["_cumsum"] = df.groupby(user_col)[rating_col].cumsum() - df[rating_col]
    df["_cumcount"] = df.groupby(user_col).cumcount()  # 0-based: count of rows BEFORE current

    loo = np.where(
        df["_cumcount"] > 0,
        df["_cumsum"] / df["_cumcount"],
        global_mean,
    ).astype(np.float32)

    return pd.Series(loo, index=df.index, name="user_loo_average_stars").reindex(reviews_df.index)
