from __future__ import annotations

import numpy as np
import pandas as pd
from surprise import SVD, SVDpp, Dataset, Reader
from surprise.prediction_algorithms.predictions import PredictionImpossible

from .base import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_surprise_trainset(df: pd.DataFrame, rating_scale=None):
    """Convert a pandas DataFrame to a Surprise full trainset."""
    if rating_scale is None:
        min_rating = float(df["rating"].min())
        max_rating = float(df["rating"].max())
        if min_rating >= max_rating:
            raise ValueError(
                f"Invalid rating scale inferred from data: min_rating ({min_rating}) "
                f"must be strictly less than max_rating ({max_rating})."
            )
        rating_scale = (min_rating, max_rating)
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    return data.build_full_trainset(), rating_scale


def _surprise_predict(algo, user, item, fallback: float) -> float:
    """Return algo.predict(...).est, or fallback on PredictionImpossible."""
    try:
        return float(algo.predict(str(user), str(item)).est)
    except PredictionImpossible:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# SurpriseSVDModel
# ─────────────────────────────────────────────────────────────────────────────

class SurpriseSVDModel(BaseModel):
    """
    Surprise SVD wrapper compatible with BaseModel.

    Default hyperparameters: n_factors=150, n_epochs=30, lr_all=0.005,
    reg_all=0.1, biased=True.

    Parameters
    ----------
    n_factors : int
        Number of latent factors.
    n_epochs : int
        Number of SGD iterations.
    lr_all : float
        Learning rate for all parameters.
    reg_all : float
        L2 regularisation for all parameters.
    biased : bool
        Whether to include user/item bias terms (strongly recommended
        for sparse data).
    random_state : int | None
        Random seed for reproducibility.
    rating_scale : tuple[float, float] | None
        (min_rating, max_rating).  Inferred from training data if None.
    clip_range : tuple[float, float] | None
        Clip predictions to this range after estimation.
    verbose : bool
        If True, Surprise prints training progress.
    name : str | None
        Human-readable model name.
    """

    def __init__(
        self,
        n_factors: int = 150,
        n_epochs: int = 30,
        lr_all: float = 0.005,
        reg_all: float = 0.1,
        biased: bool = True,
        random_state: int | None = 42,
        rating_scale: tuple | None = None,
        clip_range: tuple | None = None,
        verbose: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name, clip_range=clip_range)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.biased = biased
        self.random_state = random_state
        self.rating_scale = rating_scale
        self.verbose = verbose

        self.algo_ = None
        self.trainset_ = None
        self.global_mean_: float | None = None
        self.rating_scale_: tuple | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "SurpriseSVDModel":
        """
        Train on a DataFrame with columns ['user', 'item', 'rating'].
        """
        self._validate(df)
        train_df = df[["user", "item", "rating"]].copy()
        train_df["user"] = train_df["user"].astype(str)
        train_df["item"] = train_df["item"].astype(str)

        trainset, self.rating_scale_ = _build_surprise_trainset(
            train_df, self.rating_scale
        )
        self.trainset_ = trainset
        self.global_mean_ = float(trainset.global_mean)

        self.algo_ = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            biased=self.biased,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.algo_.fit(trainset)
        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        """Predict rating for a single (user, item) pair."""
        self._check_fitted()
        est = _surprise_predict(self.algo_, user, item, self.global_mean_)
        return float(self._clip(est))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame):
        missing = {"user", "item", "rating"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if len(df) == 0:
            raise ValueError("Training DataFrame is empty.")

    def __repr__(self):
        return (
            f"SurpriseSVDModel(n_factors={self.n_factors}, "
            f"n_epochs={self.n_epochs}, biased={self.biased})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SurpriseSVDppEnsemble
# ─────────────────────────────────────────────────────────────────────────────

class SurpriseSVDppEnsemble(BaseModel):
    """
    Ensemble of N SVD++ models trained with different random seeds.

    Each model uses identical hyperparameters; predictions are averaged
    across seeds to reduce variance from random initialisation.

    Default hyperparameters:
        seeds=[42, 7, 2026], n_factors=150, n_epochs=30,
        lr_all=0.005, reg_all=0.1, biased=True.

    Parameters
    ----------
    seeds : list[int]
        Random seeds, one model is trained per seed.
    n_factors, n_epochs, lr_all, reg_all, biased
        SVD++ hyperparameters (same for all seeds).
    rating_scale : tuple[float, float] | None
        Inferred from data if None.
    clip_range : tuple[float, float] | None
        Clip predictions after averaging.
    verbose : bool
    name : str | None
    """

    def __init__(
        self,
        seeds: list[int] | None = None,
        n_factors: int = 150,
        n_epochs: int = 30,
        lr_all: float = 0.005,
        reg_all: float = 0.1,
        biased: bool = True,
        rating_scale: tuple | None = None,
        clip_range: tuple | None = None,
        verbose: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name, clip_range=clip_range)
        self.seeds = seeds if seeds is not None else [42, 7, 2026]
        if not self.seeds:
            raise ValueError("'seeds' must be a non-empty list.")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.biased = biased
        self.rating_scale = rating_scale
        self.verbose = verbose

        self.algos_: list = []
        self.global_mean_: float | None = None
        self.rating_scale_: tuple | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "SurpriseSVDppEnsemble":
        """Train one SVD++ model per seed."""
        missing = {"user", "item", "rating"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if df.empty:
            raise ValueError("Training DataFrame is empty.")

        train_df = df[["user", "item", "rating"]].copy()
        train_df["user"] = train_df["user"].astype(str)
        train_df["item"] = train_df["item"].astype(str)

        trainset, self.rating_scale_ = _build_surprise_trainset(
            train_df, self.rating_scale
        )
        self.global_mean_ = float(trainset.global_mean)

        self.algos_ = []
        for seed in self.seeds:
            if self.verbose:
                print(f"[{self.name}] Training SVD++ seed={seed} …")
            algo = SVDpp(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                biased=self.biased,
                random_state=seed,
                verbose=self.verbose,
            )
            algo.fit(trainset)
            self.algos_.append(algo)

        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        """Average prediction across all trained SVD++ models."""
        self._check_fitted()
        preds = [
            _surprise_predict(algo, user, item, self.global_mean_)
            for algo in self.algos_
        ]
        est = float(np.mean(preds))
        return float(self._clip(est))

    def __repr__(self):
        return (
            f"SurpriseSVDppEnsemble(seeds={self.seeds}, "
            f"n_factors={self.n_factors}, n_epochs={self.n_epochs})"
        )
