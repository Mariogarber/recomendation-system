from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.lgbm_raw_features import (
    RAW_CORE_FEATURE_SET,
    RAW_PRIORS_FEATURE_SET,
    build_raw_feature_frame,
    build_train_user_stars,
    fit_raw_feature_spec,
    history_band_from_count,
)


def _users_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u_new"],
            "review_count": [10, 4, 2],
            "yelping_since": ["2020-01-01 00:00:00", "2021-01-01 00:00:00", "2022-01-01 00:00:00"],
            "useful": [1, 2, 3],
            "funny": [2, 1, 0],
            "cool": [3, 0, 1],
            "elite": ["2020,2021", "2021", ""],
            "friends": ["a,b", "", "c"],
            "fans": [5, 0, 1],
            "average_stars": [4.2, 3.1, 4.8],
            "compliment_hot": [1, 0, 0],
            "compliment_more": [0, 0, 1],
            "compliment_profile": [0, 0, 0],
            "compliment_cute": [0, 1, 0],
            "compliment_list": [0, 0, 0],
            "compliment_note": [2, 0, 0],
            "compliment_plain": [0, 1, 0],
            "compliment_cool": [0, 0, 0],
            "compliment_funny": [0, 0, 0],
            "compliment_writer": [0, 0, 0],
            "compliment_photos": [0, 0, 0],
        }
    )


def _businesses_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_id": ["b1", "b2", "b_new"],
            "city": ["A City", "B City", "C City"],
            "state": ["CA", "NV", "TX"],
            "postal_code": ["11111", "22222", "33333"],
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [-1.0, -2.0, -3.0],
            "stars": [4.5, 3.0, 2.0],
            "review_count": [20, 5, 2],
            "is_open": [1, 0, 1],
            "attributes": [
                "{'BusinessAcceptsCreditCards': 'True'}",
                "{'BikeParking': 'False'}",
                np.nan,
            ],
            "categories": [
                "Restaurants, Pizza",
                "Shopping, Fashion",
                "Services",
            ],
            "hours": [
                "{'Monday': '9:00-17:00', 'Tuesday': '9:00-17:00'}",
                "{'Saturday': '10:00-14:00'}",
                np.nan,
            ],
        }
    )


def _train_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "user_id": ["u1", "u1", "u2"],
            "business_id": ["b1", "b2", "b1"],
            "stars": [5.0, 3.0, 2.0],
            "useful": [0, 1, 0],
            "funny": [0, 0, 1],
            "cool": [0, 0, 0],
            "date": ["2023-01-01 10:00:00", "2023-01-03 12:00:00", "2023-01-05 08:00:00"],
        }
    )


def _test_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["t1", "t2"],
            "user_id": ["u_new", "u1"],
            "business_id": ["b1", "b_new"],
            "useful": [2, 0],
            "funny": [0, 1],
            "cool": [1, 0],
            "date": ["2023-01-10 11:00:00", "2023-01-11 09:00:00"],
        }
    )


def test_history_band_helper():
    assert history_band_from_count(0) == "0"
    assert history_band_from_count(1) == "1"
    assert history_band_from_count(3) == "2-5"
    assert history_band_from_count(10) == "6-20"
    assert history_band_from_count(21) == ">20"


def test_raw_core_builder_keeps_rows_and_categories():
    spec = fit_raw_feature_spec(_train_reviews(), _users_df(), _businesses_df(), feature_set=RAW_CORE_FEATURE_SET)
    frame = build_raw_feature_frame(_test_reviews(), _users_df(), _businesses_df(), spec)

    assert len(frame) == 2
    assert "review_total_votes" in frame.columns
    assert "user_train_count" not in frame.columns
    assert str(frame["business_city"].dtype) == "category"


def test_raw_priors_builder_adds_train_priors_and_fallbacks():
    spec = fit_raw_feature_spec(_train_reviews(), _users_df(), _businesses_df(), feature_set=RAW_PRIORS_FEATURE_SET)
    frame = build_raw_feature_frame(_test_reviews(), _users_df(), _businesses_df(), spec)

    assert len(frame) == 2
    assert "user_train_count" in frame.columns
    assert "business_train_mean" in frame.columns
    assert frame.loc[0, "user_known_in_train"] == 0.0
    assert frame.loc[0, "user_train_count"] == 0.0
    assert np.isclose(frame.loc[1, "business_train_mean"], spec.global_mean)
    assert frame.loc[1, "business_known_in_train"] == 0.0
    assert frame.loc[1, "business_train_count"] == 0.0
    assert "business_train_support_bucket" in frame.columns
    assert "user_train_history_bucket" in frame.columns
    assert "user_history_is_2" in frame.columns
    assert "user_item_support_interaction" in frame.columns
    assert str(frame["business_train_support_bucket"].dtype) == "category"


def test_review_vote_logs_clip_negative_sentinels():
    train_reviews = _train_reviews()
    test_reviews = _test_reviews().copy()
    test_reviews.loc[0, "useful"] = -1

    spec = fit_raw_feature_spec(train_reviews, _users_df(), _businesses_df(), feature_set=RAW_CORE_FEATURE_SET)
    frame = build_raw_feature_frame(test_reviews, _users_df(), _businesses_df(), spec)

    assert np.isfinite(frame.loc[0, "review_useful_log1p"])
    assert frame.loc[0, "review_useful_log1p"] == 0.0


def test_raw_priors_builder_adds_short_history_flags_and_item_support_features():
    spec = fit_raw_feature_spec(_train_reviews(), _users_df(), _businesses_df(), feature_set=RAW_PRIORS_FEATURE_SET)
    frame = build_raw_feature_frame(_test_reviews(), _users_df(), _businesses_df(), spec)

    assert "item_is_new" in frame.columns
    assert "item_support_log1p" in frame.columns
    assert "user_history_is_short_2_3" in frame.columns
    assert "user_history_is_short_4_5" in frame.columns
    assert "item_support_per_user_history" in frame.columns
    assert frame.loc[0, "item_is_new"] == 0.0
    assert frame.loc[1, "item_is_new"] == 1.0


def test_build_train_user_stars_known_user():
    train = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "stars": [3.0, 5.0, 2.0]})
    result = build_train_user_stars(train, global_mean=3.5)
    assert abs(result["u1"] - 4.0) < 1e-5
    assert abs(result["u2"] - 2.0) < 1e-5


def test_build_train_user_stars_cold_user_gets_global_mean():
    train = pd.DataFrame({"user_id": ["u1"], "stars": [4.0]})
    result = build_train_user_stars(train, global_mean=3.5)
    assert "u_cold" not in result


def test_build_train_user_stars_returns_dict():
    train = pd.DataFrame({"user_id": ["u1"], "stars": [4.0]})
    result = build_train_user_stars(train, global_mean=3.5)
    assert isinstance(result, dict)
