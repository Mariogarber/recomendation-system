from __future__ import annotations

from typing import Any

import pandas as pd


def build_user_train_aggregates(
    train_reviews: pd.DataFrame,
    *,
    user_col: str = "user_id",
    rating_col: str = "stars",
    date_col: str = "date",
) -> pd.DataFrame:
    required = {user_col, rating_col}
    missing = required - set(train_reviews.columns)
    if missing:
        raise ValueError(f"Missing required columns in train_reviews: {missing}")

    grouped = train_reviews.groupby(user_col, dropna=False)[rating_col]
    result = grouped.agg(
        train_review_count="count",
        train_average_stars="mean",
        train_rating_std="std",
    ).reset_index()

    if date_col in train_reviews.columns:
        dates = pd.to_datetime(train_reviews[date_col], errors="coerce")
        date_df = (
            train_reviews.assign(_parsed_date=dates)
            .groupby(user_col, dropna=False)["_parsed_date"]
            .agg(train_first_review_date="min", train_last_review_date="max")
            .reset_index()
        )
        result = result.merge(date_df, on=user_col, how="left")

    return result


def build_business_train_aggregates(
    train_reviews: pd.DataFrame,
    *,
    item_col: str = "business_id",
    rating_col: str = "stars",
    date_col: str = "date",
) -> pd.DataFrame:
    required = {item_col, rating_col}
    missing = required - set(train_reviews.columns)
    if missing:
        raise ValueError(f"Missing required columns in train_reviews: {missing}")

    grouped = train_reviews.groupby(item_col, dropna=False)[rating_col]
    result = grouped.agg(
        train_review_count="count",
        train_average_stars="mean",
        train_rating_std="std",
    ).reset_index()

    if date_col in train_reviews.columns:
        dates = pd.to_datetime(train_reviews[date_col], errors="coerce")
        date_df = (
            train_reviews.assign(_parsed_date=dates)
            .groupby(item_col, dropna=False)["_parsed_date"]
            .agg(train_first_review_date="min", train_last_review_date="max")
            .reset_index()
        )
        result = result.merge(date_df, on=item_col, how="left")

    return result


def compare_user_metadata_with_train(
    users_df: pd.DataFrame,
    train_reviews: pd.DataFrame,
    *,
    user_col: str = "user_id",
    metadata_review_count_col: str = "review_count",
    metadata_average_stars_col: str = "average_stars",
) -> pd.DataFrame:
    aggregates = build_user_train_aggregates(train_reviews, user_col=user_col)

    cols = [user_col]
    if metadata_review_count_col in users_df.columns:
        cols.append(metadata_review_count_col)
    if metadata_average_stars_col in users_df.columns:
        cols.append(metadata_average_stars_col)

    out = users_df[cols].merge(aggregates, on=user_col, how="left")

    if metadata_review_count_col in out.columns:
        out["review_count_diff"] = out[metadata_review_count_col] - out["train_review_count"]
        out["review_count_exact_match"] = out["review_count_diff"] == 0

    if metadata_average_stars_col in out.columns:
        out["average_stars_abs_diff"] = (
            out[metadata_average_stars_col] - out["train_average_stars"]
        ).abs()

    out["seen_in_train"] = out["train_review_count"].notna()
    return out


def compare_business_metadata_with_train(
    businesses_df: pd.DataFrame,
    train_reviews: pd.DataFrame,
    *,
    item_col: str = "business_id",
    metadata_review_count_col: str = "review_count",
    metadata_average_stars_col: str = "stars",
) -> pd.DataFrame:
    aggregates = build_business_train_aggregates(train_reviews, item_col=item_col)

    cols = [item_col]
    if metadata_review_count_col in businesses_df.columns:
        cols.append(metadata_review_count_col)
    if metadata_average_stars_col in businesses_df.columns:
        cols.append(metadata_average_stars_col)

    out = businesses_df[cols].merge(aggregates, on=item_col, how="left")

    if metadata_review_count_col in out.columns:
        out["review_count_diff"] = out[metadata_review_count_col] - out["train_review_count"]
        out["review_count_exact_match"] = out["review_count_diff"] == 0

    if metadata_average_stars_col in out.columns:
        out["average_stars_abs_diff"] = (
            out[metadata_average_stars_col] - out["train_average_stars"]
        ).abs()

    out["seen_in_train"] = out["train_review_count"].notna()
    return out


def summarize_user_comparison(comparison_df: pd.DataFrame) -> dict[str, Any]:
    return _summarize_comparison(comparison_df)


def summarize_business_comparison(comparison_df: pd.DataFrame) -> dict[str, Any]:
    return _summarize_comparison(comparison_df)


def _summarize_comparison(comparison_df: pd.DataFrame) -> dict[str, Any]:
    seen_mask = comparison_df["seen_in_train"].fillna(False)
    seen_df = comparison_df.loc[seen_mask].copy()

    summary: dict[str, Any] = {
        "total_rows": int(len(comparison_df)),
        "rows_seen_in_train": int(seen_mask.sum()),
        "rows_unseen_in_train": int((~seen_mask).sum()),
    }

    if len(seen_df) == 0:
        return summary

    if "review_count_exact_match" in seen_df.columns:
        exact = seen_df["review_count_exact_match"].fillna(False)
        summary["review_count_exact_match_rate"] = float(exact.mean())
        if "review_count_diff" in seen_df.columns:
            diff = seen_df["review_count_diff"].dropna()
            summary["review_count_mean_abs_diff"] = float(diff.abs().mean()) if len(diff) else None

    if "average_stars_abs_diff" in seen_df.columns:
        diff = seen_df["average_stars_abs_diff"].dropna()
        summary["average_stars_mean_abs_diff"] = float(diff.mean()) if len(diff) else None
        summary["average_stars_median_abs_diff"] = float(diff.median()) if len(diff) else None

    return summary
