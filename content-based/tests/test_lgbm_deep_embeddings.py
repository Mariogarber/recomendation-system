import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from utils.frozen_embedding_regression import FrozenEmbeddingBundle
from utils.lgbm_deep_embeddings import (
    build_lgbm_feature_matrix,
    compute_scalar_priors,
    fit_review_context_scaler,
    round_half_up,
    sample_cold_start_rows,
)


def _make_train_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "business_id": ["b1", "b2", "b1", "b3"],
            "stars": [4.0, 2.0, 5.0, 3.0],
        }
    )


def _make_bundle() -> FrozenEmbeddingBundle:
    user_ids = pd.Series(["u1", "u2", "u_new"])
    business_ids = pd.Series(["b1", "b2"])
    user_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    business_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    user_table = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u_new"],
            "history_count_train": [2.0, 1.0, 0.0],
            "history_band": ["2-5", "1", "0"],
        }
    )
    return FrozenEmbeddingBundle(
        root=Path("."),
        user_ids=user_ids,
        business_ids=business_ids,
        user_embeddings=user_embeddings,
        business_embeddings=business_embeddings,
        user_table=user_table,
        summary={},
    )


def _make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": ["u1", "u2", "u_new"],
            "item": ["b1", "b2", "b1"],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-09-01"]),
            "useful": [0.0, 1.0, 0.0],
            "funny": [0.0, 0.0, 0.0],
            "cool": [0.0, 0.0, 0.0],
        }
    )


def _build_matrix(*, forced_new_user_mask: np.ndarray | None = None):
    bundle = _make_bundle()
    priors = compute_scalar_priors(_make_train_reviews())
    frame = _make_frame()
    min_ts, means, stds = fit_review_context_scaler(frame)
    return build_lgbm_feature_matrix(
        frame,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=means,
        review_context_stds=stds,
        forced_new_user_mask=forced_new_user_mask,
    )


def test_matrix_keeps_all_rows():
    matrix, _ = _build_matrix()
    assert matrix.shape[0] == 3


def test_matrix_columns_match_feature_names():
    matrix, names = _build_matrix()
    assert matrix.shape[1] == len(names)


def test_actual_new_user_from_priors_is_zeroed_even_if_bundle_has_embedding():
    matrix, names = _build_matrix()
    user_emb_cols = [i for i, name in enumerate(names) if name.startswith("user_emb_")]
    is_new_col = names.index("is_new_user")

    assert matrix[2, is_new_col] == 1.0
    assert np.all(matrix[2, user_emb_cols] == 0.0)


def test_known_user_keeps_embedding():
    matrix, names = _build_matrix()
    user_emb_cols = [i for i, name in enumerate(names) if name.startswith("user_emb_")]

    assert matrix[0, user_emb_cols[0]] == pytest.approx(1.0)
    assert matrix[0, user_emb_cols[1]] == pytest.approx(0.0)


def test_forced_new_user_mask_zeroes_known_user_features():
    matrix, names = _build_matrix(forced_new_user_mask=np.array([True, False, False]))
    user_emb_cols = [i for i, name in enumerate(names) if name.startswith("user_emb_")]
    is_new_col = names.index("is_new_user")
    history_count_col = names.index("history_count")
    user_mean_col = names.index("user_mean_rating")

    assert matrix[0, is_new_col] == 1.0
    assert np.all(matrix[0, user_emb_cols] == 0.0)
    assert matrix[0, history_count_col] == 0.0
    assert matrix[0, user_mean_col] == pytest.approx(3.5)


def test_round_half_up_matches_submission_rule():
    values = np.array([1.2, 1.5, 2.49, 2.5, 4.99], dtype=np.float32)
    assert round_half_up(values).tolist() == [1, 2, 2, 3, 5]


def test_sample_cold_start_rows_is_deterministic():
    frame = pd.DataFrame({"x": np.arange(10)})
    sampled = sample_cold_start_rows(frame, 0.3, seed=7)
    assert len(sampled) == 3
    assert sampled["x"].tolist() == sample_cold_start_rows(frame, 0.3, seed=7)["x"].tolist()

