import numpy as np
import pandas as pd
import pytest

from utils.gbm_features import ScalarPriors, compute_scalar_priors


def _make_train_reviews() -> pd.DataFrame:
    """Minimal train reviews in raw format (user_id, business_id, stars)."""
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u3"],
        "business_id": ["b1", "b2", "b1", "b3"],
        "stars": [4.0, 2.0, 5.0, 3.0],
    })


def test_global_mean():
    priors = compute_scalar_priors(_make_train_reviews())
    assert abs(priors.global_mean - 3.5) < 1e-5


def test_user_mean():
    priors = compute_scalar_priors(_make_train_reviews())
    # u1: rated b1=4 and b2=2 → mean 3.0
    assert abs(priors.user_mean["u1"] - 3.0) < 1e-5
    # u2: rated b1=5 → mean 5.0
    assert abs(priors.user_mean["u2"] - 5.0) < 1e-5


def test_business_mean():
    priors = compute_scalar_priors(_make_train_reviews())
    # b1: received 4 and 5 → mean 4.5
    assert abs(priors.business_mean["b1"] - 4.5) < 1e-5


def test_user_count():
    priors = compute_scalar_priors(_make_train_reviews())
    assert priors.user_count["u1"] == 2
    assert priors.user_count["u2"] == 1


def test_single_review_std_is_zero():
    priors = compute_scalar_priors(_make_train_reviews())
    assert priors.user_std["u2"] == 0.0


def test_accepts_canonical_format():
    """compute_scalar_priors must also accept user/item/rating columns."""
    canonical = pd.DataFrame({
        "user": ["u1", "u2"],
        "item": ["b1", "b1"],
        "rating": [4.0, 2.0],
    })
    priors = compute_scalar_priors(canonical)
    assert abs(priors.global_mean - 3.0) < 1e-5
