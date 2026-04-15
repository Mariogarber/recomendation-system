from __future__ import annotations

import numpy as np
import pandas as pd

from utils.lgbm_tabular_moe import (
    TABULAR_BAND_TO_EXPERT,
    build_feature_columns_by_expert,
    collapse_history_band,
    compute_tabular_baseline_prediction,
    eval_prefix_frame,
    resolve_tabular_router_branches,
    train_prefix_frame,
)


def _train_like_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "user": ["u1", "u1", "u1", "u2"],
            "item": ["b1", "b2", "b3", "b1"],
            "rating": [5.0, 1.0, 4.0, 3.0],
            "review_date": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-02 10:00:00",
                    "2023-01-03 10:00:00",
                    "2023-01-04 10:00:00",
                ]
            ),
            "business_city_top": ["A", "A", "B", "A"],
            "business_state": ["CA", "CA", "NV", "CA"],
            "business_primary_category_family": ["Pizza", "Pizza", "Coffee", "Pizza"],
            "business_is_open": [1, 1, 0, 1],
            "business_stars": [4.5, 3.0, 4.0, 4.5],
            "user_average_stars": [4.2, 4.2, 4.2, 3.2],
            "user_train_mean": [4.0, 4.0, 4.0, 3.0],
            "user_train_count": [2.0, 2.0, 2.0, 1.0],
            "business_train_mean": [4.0, 3.5, 4.1, 4.0],
            "business_train_count": [10.0, 5.0, 4.0, 10.0],
            "business_review_count": [10.0, 5.0, 4.0, 10.0],
            "business_city": pd.Categorical(["A", "A", "B", "A"]),
            "business_postal_code": pd.Categorical(["111", "111", "222", "111"]),
        }
    )


def test_train_frame_uses_prefix_without_target_leakage():
    frame = train_prefix_frame(_train_like_frame(), global_mean=3.5)
    row_r2 = frame.set_index("review_id").loc["r2"]
    row_r3 = frame.set_index("review_id").loc["r3"]

    assert row_r2["prefix_user_count"] == 1.0
    assert row_r2["prefix_user_mean"] == 5.0
    assert row_r2["prefix_same_city_count"] == 1.0
    assert row_r2["prefix_same_category_count"] == 1.0
    assert row_r3["prefix_user_count"] == 2.0
    assert row_r3["prefix_user_mean"] == 3.0
    assert row_r3["prefix_same_city_count"] == 0.0
    assert row_r3["prefix_same_category_count"] == 0.0
    assert "user_train_mean" not in frame.columns


def test_eval_frame_uses_context_only():
    context = _train_like_frame().iloc[:2].copy()
    target = pd.DataFrame(
        {
            "review_id": ["v1"],
            "user": ["u1"],
            "item": ["b3"],
            "rating": [4.0],
            "review_date": pd.to_datetime(["2023-01-03 10:00:00"]),
            "business_city_top": ["B"],
            "business_state": ["NV"],
            "business_primary_category_family": ["Coffee"],
            "business_is_open": [0],
            "business_stars": [4.0],
            "user_average_stars": [4.2],
            "user_train_mean": [4.0],
            "user_train_count": [2.0],
            "business_train_mean": [4.1],
            "business_train_count": [4.0],
            "business_review_count": [4.0],
            "business_city": pd.Categorical(["B"]),
            "business_postal_code": pd.Categorical(["222"]),
        }
    )

    frame = eval_prefix_frame(target, context, global_mean=3.5)
    row = frame.iloc[0]

    assert row["prefix_user_count"] == 2.0
    assert row["prefix_user_mean"] == 3.0
    assert row["prefix_same_city_count"] == 0.0
    assert row["prefix_same_category_count"] == 0.0


def test_branch_resolution_maps_each_band_to_its_expert():
    branches = resolve_tabular_router_branches(np.array(["0", "1", "2-20", ">20"], dtype=object))
    assert branches.tolist() == [
        TABULAR_BAND_TO_EXPERT["0"],
        TABULAR_BAND_TO_EXPERT["1"],
        TABULAR_BAND_TO_EXPERT["2-20"],
        TABULAR_BAND_TO_EXPERT[">20"],
    ]
    assert collapse_history_band(2) == "2-20"
    assert collapse_history_band(20) == "2-20"


def test_feature_partition_and_baseline_prediction():
    frame = train_prefix_frame(_train_like_frame(), global_mean=3.5)
    all_feature_columns = [
        column
        for column in frame.columns
        if column not in {"review_id", "user", "item", "rating", "review_date"}
        and (pd.api.types.is_numeric_dtype(frame[column]) or isinstance(frame[column].dtype, pd.CategoricalDtype))
    ]
    feature_columns_by_expert, _, manifest = build_feature_columns_by_expert(
        base_feature_columns=all_feature_columns,
        categorical_columns=["business_city", "business_postal_code"],
    )
    baseline = compute_tabular_baseline_prediction(frame, global_mean=3.5)

    assert "prefix_same_category_mean" not in feature_columns_by_expert["very_short_history_tabular"]
    assert "prefix_same_category_mean" in feature_columns_by_expert["mid_history_tabular"]
    assert "user_train_mean" not in manifest["static_columns"]
    assert feature_columns_by_expert["mid_history_tabular"].count("prefix_user_bias_vs_global") == 1
    assert "user_train_mean" in manifest["forbidden_global_history_columns"]
    assert manifest["experts"]["cold_user_tabular"]["n_features"] < manifest["experts"]["mid_history_tabular"]["n_features"]
    assert np.all(np.isfinite(baseline))
    assert np.all((baseline >= 1.0) & (baseline <= 5.0))
