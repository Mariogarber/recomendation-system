from pathlib import Path

import numpy as np
import pandas as pd

from utils.frozen_embedding_regression import FrozenEmbeddingBundle
from utils.lgbm_known_prefix_deep_features import (
    build_known_prefix_eval_frame,
    build_known_prefix_train_frame,
    resolve_router_branches,
)


def _make_bundle() -> FrozenEmbeddingBundle:
    return FrozenEmbeddingBundle(
        root=Path("."),
        user_ids=pd.Series(["u1", "u2"]),
        business_ids=pd.Series(["b1", "b2", "b3"]),
        user_embeddings=np.zeros((2, 2), dtype=np.float32),
        business_embeddings=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        user_table=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "history_count_train": [2.0, 0.0],
                "history_band": ["2-5", "0"],
            }
        ),
        summary={},
    )


def _make_train_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "user_id": ["u1", "u1", "u1", "u2"],
            "business_id": ["b1", "b2", "b3", "b1"],
            "stars": [5.0, 1.0, 4.0, 3.0],
            "date": [
                "2023-01-01 10:00:00",
                "2023-01-02 10:00:00",
                "2023-01-03 10:00:00",
                "2023-01-04 10:00:00",
            ],
        }
    )


def test_train_frame_uses_prefix_without_target_leakage():
    frame = build_known_prefix_train_frame(
        _make_train_reviews(),
        _make_bundle(),
        max_history_len=5,
        target_history_bands=("1", "2-5"),
    )

    row_r2 = frame.set_index("review_id").loc["r2"]
    row_r3 = frame.set_index("review_id").loc["r3"]

    assert row_r2["known_prefix_history_count"] == 1.0
    assert row_r2["known_prefix_history_mean_emb_000"] == 1.0
    assert row_r2["known_prefix_history_mean_emb_001"] == 0.0
    assert row_r3["known_prefix_history_count"] == 2.0
    assert row_r3["known_prefix_history_mean_emb_000"] == 0.5
    assert row_r3["known_prefix_history_mean_emb_001"] == 0.5


def test_eval_frame_uses_context_history_only():
    train_context = _make_train_reviews().iloc[:2].copy()
    val_target = pd.DataFrame(
        {
            "review_id": ["v1"],
            "user_id": ["u1"],
            "business_id": ["b3"],
            "stars": [4.0],
            "date": ["2023-01-03 10:00:00"],
        }
    )

    frame = build_known_prefix_eval_frame(
        val_target,
        train_context,
        _make_bundle(),
        max_history_len=5,
        target_history_bands=("2-5",),
    )
    row = frame.iloc[0]

    assert row["known_prefix_history_count"] == 2.0
    assert row["known_prefix_history_mean_emb_000"] == 0.5
    assert row["known_prefix_history_mean_emb_001"] == 0.5
    assert row["known_prefix_count_is_2"] == 1.0
    assert row["known_prefix_count_is_3"] == 0.0
    assert row["known_prefix_history_rating_range"] == 4.0
    assert row["known_prefix_history_similarity_mean_x_count"] != 0.0


def test_router_branch_resolution_prefers_enabled_known_prefix_bands():
    branches = resolve_router_branches(
        user_known_mask=np.array([False, True, True, True]),
        history_band=np.array(["0", "1", "2-5", ">20"], dtype=object),
        enabled_known_prefix_bands=("1", "2-5"),
    )

    assert branches.tolist() == [
        "cold_model",
        "known_prefix_deep_model",
        "known_prefix_deep_model",
        "known_model",
    ]
