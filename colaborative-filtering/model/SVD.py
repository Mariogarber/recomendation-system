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


class SurpriseSVDppModel(BaseModel):
    """
    Wrapper de Surprise SVDpp compatible con la API de BaseModel.

    Requisitos de entrada en fit:
        df con columnas ['user', 'item', 'rating']

    API heredada:
        - fit(df)
        - predict(user, item)

    Extras:
        - predict_batch(df)
        - recommend(user, top_k=10, exclude_seen=True)
        - get_user_embedding(user)
        - get_item_embedding(item)
        - get_implicit_item_embedding(item)
        - get_user_bias(user)
        - get_item_bias(item)
        - explain_prediction(user, item)

    Notas:
        - SVD++ modela:
              pred = mu + bu + bi + <qi, pu + |Iu|^(-1/2) * sum(yj)>
        - yj captura feedback implícito: que el usuario haya interactuado/rateado
          un ítem, independientemente del valor concreto del rating.
        - Para usuario/item desconocido, Surprise asume a 0 sus factores y sesgos,
          por lo que el fallback práctico queda en media global y/o sesgos conocidos.
    """

    def __init__(
        self,
        n_factors=20,
        n_epochs=20,
        init_mean=0.0,
        init_std_dev=0.1,
        lr_all=0.007,
        reg_all=0.02,
        lr_bu=None,
        lr_bi=None,
        lr_pu=None,
        lr_qi=None,
        lr_yj=None,
        reg_bu=None,
        reg_bi=None,
        reg_pu=None,
        reg_qi=None,
        reg_yj=None,
        random_state=42,
        verbose=False,
        cache_ratings=False,
        rating_scale=(1, 10),
        clip_range=None,
        name=None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.lr_yj = lr_yj
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.reg_yj = reg_yj
        self.random_state = random_state
        self.verbose = verbose
        self.cache_ratings = cache_ratings
        self.rating_scale = rating_scale

        # Internos
        self.algo_ = None
        self.trainset_ = None
        self.global_mean_ = None

        self.users_ = None
        self.items_ = None
        self.user_seen_items_ = None

    # =========================
    # Helpers internos
    # =========================
    def _validate_df(self, df: pd.DataFrame):
        required_cols = {"user", "item", "rating"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"El DataFrame debe contener las columnas {required_cols}. "
                f"Faltan: {missing}"
            )
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _clip(self, value: float) -> float:
        if self.clip_range is None:
            return float(value)
        lo, hi = self.clip_range
        return float(np.clip(value, lo, hi))

    def _build_surprise_dataset(self, df: pd.DataFrame):
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
        return data

    def _build_seen_dict(self, df: pd.DataFrame):
        return df.groupby("user")["item"].apply(set).to_dict()

    def _safe_inner_uid(self, user):
        try:
            return self.trainset_.to_inner_uid(user)
        except ValueError:
            return None

    def _safe_inner_iid(self, item):
        try:
            return self.trainset_.to_inner_iid(item)
        except ValueError:
            return None

    def _get_user_seen_inner_items(self, user):
        """
        Devuelve los inner ids de los ítems vistos por el usuario en train.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        seen_raw = self.user_seen_items_.get(user, set())
        inner_items = []
        for item in seen_raw:
            inner_iid = self._safe_inner_iid(item)
            if inner_iid is not None:
                inner_items.append(inner_iid)
        return inner_items

    # =========================
    # API BaseModel
    # =========================
    def fit(self, df: pd.DataFrame):
        self._validate_df(df)
        df = df.copy()

        data = self._build_surprise_dataset(df)
        trainset = data.build_full_trainset()

        algo = SVDpp(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            init_mean=self.init_mean,
            init_std_dev=self.init_std_dev,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            lr_bu=self.lr_bu,
            lr_bi=self.lr_bi,
            lr_pu=self.lr_pu,
            lr_qi=self.lr_qi,
            lr_yj=self.lr_yj,
            reg_bu=self.reg_bu,
            reg_bi=self.reg_bi,
            reg_pu=self.reg_pu,
            reg_qi=self.reg_qi,
            reg_yj=self.reg_yj,
            random_state=self.random_state,
            verbose=self.verbose,
            cache_ratings=self.cache_ratings,
        )
        algo.fit(trainset)

        self.algo_ = algo
        self.trainset_ = trainset
        self.global_mean_ = float(trainset.global_mean)

        self.users_ = set(df["user"].unique())
        self.items_ = set(df["item"].unique())
        self.user_seen_items_ = self._build_seen_dict(df)

        self.is_fitted_ = True
        return self

    def predict(self, user, item):
        self._check_fitted()
        est = float(self.algo_.predict(uid=user, iid=item).est)
        return self._clip(est)

    # =========================
    # Extras útiles
    # =========================
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()

        required_cols = {"user", "item"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"El DataFrame debe contener las columnas {required_cols}. "
                f"Faltan: {missing}"
            )

        preds = [
            self._clip(self.algo_.predict(uid=u, iid=i).est)
            for u, i in zip(df["user"].values, df["item"].values)
        ]
        return np.array(preds, dtype=float)

    def recommend(self, user, top_k=10, exclude_seen=True):
        self._check_fitted()

        candidate_items = self.items_
        if exclude_seen:
            seen = self.user_seen_items_.get(user, set())
            candidate_items = candidate_items - seen

        scored = [(item, self.predict(user, item)) for item in candidate_items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_user_embedding(self, user):
        """
        Devuelve el embedding p_u del usuario.
        Si el usuario no existe, devuelve None.
        """
        self._check_fitted()

        inner_uid = self._safe_inner_uid(user)
        if inner_uid is None:
            return None

        return np.array(self.algo_.pu[inner_uid], dtype=float)

    def get_item_embedding(self, item):
        """
        Devuelve el embedding q_i del ítem.
        Si el ítem no existe, devuelve None.
        """
        self._check_fitted()

        inner_iid = self._safe_inner_iid(item)
        if inner_iid is None:
            return None

        return np.array(self.algo_.qi[inner_iid], dtype=float)

    def get_implicit_item_embedding(self, item):
        """
        Devuelve el embedding implícito y_j del ítem.
        Si el ítem no existe, devuelve None.
        """
        self._check_fitted()

        inner_iid = self._safe_inner_iid(item)
        if inner_iid is None:
            return None

        return np.array(self.algo_.yj[inner_iid], dtype=float)

    def get_user_bias(self, user):
        self._check_fitted()

        inner_uid = self._safe_inner_uid(user)
        if inner_uid is None:
            return 0.0

        return float(self.algo_.bu[inner_uid])

    def get_item_bias(self, item):
        self._check_fitted()

        inner_iid = self._safe_inner_iid(item)
        if inner_iid is None:
            return 0.0

        return float(self.algo_.bi[inner_iid])

    def get_user_implicit_vector(self, user):
        """
        Devuelve |I_u|^{-1/2} * sum_{j in I_u} y_j.
        Si el usuario no existe o no tiene histórico, devuelve vector cero.
        """
        self._check_fitted()

        inner_items = self._get_user_seen_inner_items(user)
        if len(inner_items) == 0:
            return np.zeros(self.n_factors, dtype=float)

        y_sum = np.sum(self.algo_.yj[inner_items], axis=0)
        return np.array(y_sum / np.sqrt(len(inner_items)), dtype=float)

    def get_user_effective_embedding(self, user):
        """
        Devuelve p_u + componente implícita.
        Si el usuario es desconocido, usa solo el vector implícito si existe histórico
        en user_seen_items_; en uso normal para usuario desconocido será vector cero.
        """
        self._check_fitted()

        pu = self.get_user_embedding(user)
        implicit_vec = self.get_user_implicit_vector(user)

        if pu is None:
            return implicit_vec
        return pu + implicit_vec

    def explain_prediction(self, user, item):
        """
        Descompone la predicción de SVD++:
            pred = mu + bu + bi + <qi, pu + |I_u|^{-1/2} sum(y_j)>

        Ojo:
        - La descomposición exacta depende del histórico del usuario en train.
        - Para usuarios desconocidos, Surprise asume sesgo y factores a 0.
        - La predicción final se toma del propio Surprise para mantener consistencia.
        """
        self._check_fitted()

        qi = self.get_item_embedding(item)
        pu = self.get_user_embedding(user)
        implicit_vec = self.get_user_implicit_vector(user)
        effective_user_vec = self.get_user_effective_embedding(user)

        explicit_dot = None
        implicit_dot = None
        total_dot = None

        if qi is not None:
            if pu is not None:
                explicit_dot = float(np.dot(qi, pu))
            implicit_dot = float(np.dot(qi, implicit_vec))
            total_dot = float(np.dot(qi, effective_user_vec))

        mu = float(self.global_mean_)
        bu = self.get_user_bias(user)
        bi = self.get_item_bias(item)
        pred = self.predict(user, item)

        return {
            "global_mean": mu,
            "user_bias": bu,
            "item_bias": bi,
            "dot_qi_pu": explicit_dot,
            "dot_qi_implicit": implicit_dot,
            "dot_qi_effective_user": total_dot,
            "prediction": pred,
            "user_known": pu is not None,
            "item_known": qi is not None,
            "n_seen_items_user": len(self.user_seen_items_.get(user, set())),
        }
