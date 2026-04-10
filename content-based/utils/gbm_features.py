from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
