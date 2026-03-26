import pandas as pd
import numpy as np
from surprise import Dataset, Reader, BaselineOnly, KNNBaseline
from surprise.prediction_algorithms.predictions import PredictionImpossible

from .base import BaseModel


class MeanBaseline(BaseModel):
    def fit(self, train_data):
        self.global_mean = train_data['rating'].mean()
        self.user_means = train_data.groupby('user')['rating'].mean()
        self.item_means = train_data.groupby('item')['rating'].mean()
        self.is_fitted_ = True

    def predict(self, user, item):
        user_mean = self.user_means.get(user, self.global_mean)
        item_mean = self.item_means.get(item, self.global_mean)
        return (user_mean + item_mean) / 2

    def predict_df(self, df):
        df['prediction'] = df.apply(lambda row: self.predict(row['user'], row['item']), axis=1)
        return df


class SurpriseBaselineOnlyModel(BaseModel):
    """
    Wrapper de Surprise BaselineOnly compatible con la API de BaseModel.

    Requisitos de entrada en fit:
        df con columnas ['user', 'item', 'rating']

    API heredada:
        - fit(df)
        - predict(user, item)

    Extras:
        - predict_batch(df)
        - recommend(user, top_k=10, exclude_seen=True)

    Notas:
        - Usa Surprise BaselineOnly: pred = global_mean + user_bias + item_bias
        - Para usuario/item desconocido, Surprise usa 0 para ese bias.
        - Si ambos son desconocidos, en la práctica cae a la media global.
    """

    def __init__(
        self,
        bsl_options=None,
        verbose=False,
        rating_scale=None,
        clip_range=None,
        name=None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.bsl_options = bsl_options if bsl_options is not None else {
            "method": "als",
            "n_epochs": 10,
            "reg_u": 12,
            "reg_i": 5,
        }
        self.verbose = verbose
        self.rating_scale = rating_scale  # por ejemplo (1, 10)

        # Objetos internos
        self.algo_ = None
        self.trainset_ = None
        self.global_mean_ = None

        # Para utilidades propias
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

    def _infer_rating_scale(self, df: pd.DataFrame):
        if self.rating_scale is not None:
            return self.rating_scale

        rmin = float(df["rating"].min())
        rmax = float(df["rating"].max())

        # Ojo: Surprise necesita un rating_scale fijo.
        # Inferirlo del train puede ser cómodo, pero en producción prefiero fijarlo.
        return (rmin, rmax)

    def _build_surprise_dataset(self, df: pd.DataFrame):
        rating_scale = self._infer_rating_scale(df)
        reader = Reader(rating_scale=rating_scale)
        data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
        return data

    def _build_seen_dict(self, df: pd.DataFrame):
        seen = df.groupby("user")["item"].apply(set).to_dict()
        return seen

    def _clip(self, value: float) -> float:
        if self.clip_range is None:
            return float(value)
        lo, hi = self.clip_range
        return float(np.clip(value, lo, hi))

    # =========================
    # Métodos obligatorios
    # =========================
    def fit(self, df: pd.DataFrame):
        """
        Entrena el wrapper con un DataFrame pandas.
        """
        self._validate_df(df)

        df = df.copy()
        data = self._build_surprise_dataset(df)
        trainset = data.build_full_trainset()

        algo = BaselineOnly(
            bsl_options=self.bsl_options,
            verbose=self.verbose,
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
        """
        Predice rating para un par (user, item).
        Compatible con la API BaseModel.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        pred = self.algo_.predict(uid=user, iid=item)

        # Surprise devuelve un objeto Prediction con .est
        est = float(pred.est)
        est = self._clip(est)
        return est

    # =========================
    # Métodos extra útiles
    # =========================
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice para un DataFrame con columnas ['user', 'item'].
        Devuelve un np.ndarray con las predicciones.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

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
        """
        Recomienda top_k ítems para un usuario.

        Limitación importante:
        BaselineOnly no usa factores latentes ni vecinos, así que esto es
        simplemente rankear ítems por baseline esperado para ese usuario.

        Si el usuario es desconocido:
            score(item) ~= global_mean + item_bias
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        candidate_items = self.items_

        if exclude_seen:
            seen = self.user_seen_items_.get(user, set())
            candidate_items = candidate_items - seen

        scored = []
        for item in candidate_items:
            score = self.predict(user, item)
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_user_bias(self, user):
        """
        Devuelve el bias del usuario si existe en train; si no, 0.0
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        try:
            inner_uid = self.trainset_.to_inner_uid(user)
            return float(self.algo_.bu[inner_uid])
        except ValueError:
            return 0.0

    def get_item_bias(self, item):
        """
        Devuelve el bias del ítem si existe en train; si no, 0.0
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        try:
            inner_iid = self.trainset_.to_inner_iid(item)
            return float(self.algo_.bi[inner_iid])
        except ValueError:
            return 0.0

    def explain_prediction(self, user, item):
        """
        Descompone la predicción en:
            pred = mu + bu + bi
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        mu = float(self.global_mean_)
        bu = self.get_user_bias(user)
        bi = self.get_item_bias(item)
        pred = self._clip(mu + bu + bi)

        return {
            "global_mean": mu,
            "user_bias": bu,
            "item_bias": bi,
            "prediction": pred,
        }
    

class SurpriseKNNBaselineWrapper(BaseModel):
    """
    Wrapper de Surprise KNNBaseline compatible con BaseModel.

    Espera en fit:
        DataFrame con columnas ['user', 'item', 'rating']

    Parámetros principales:
    -----------------------
    user_based : bool
        - True  -> vecinos entre usuarios
        - False -> vecinos entre ítems

    k : int
        Número máximo de vecinos.

    min_k : int
        Número mínimo de vecinos requeridos.

    sim_name : str
        Métrica de similitud de Surprise.
        Ejemplos: 'pearson_baseline', 'cosine', 'msd', 'pearson'

    shrinkage : int | float
        Regularización de similitud para pearson_baseline.

    bsl_options : dict | None
        Opciones del baseline interno de Surprise.
        Ejemplo:
            {
                "method": "als",
                "n_epochs": 10,
                "reg_u": 15,
                "reg_i": 10
            }

    min_rating, max_rating : float | None
        Rango de ratings para Reader. Si no se indican, se infieren del df de entrenamiento.

    unknown_strategy : str | float
        Qué hacer si el usuario/item no visto provoca predicción no fiable o imposible.
        Opciones:
            - "global_mean" -> media global de train
            - "nan"         -> np.nan
            - número        -> valor fijo
            - "surprise"    -> usar est tal cual devuelto por Surprise

    verbose : bool
        Si True, imprime info básica en fit.
    """

    def __init__(
        self,
        k=40,
        min_k=1,
        user_based=False,
        sim_name="pearson_baseline",
        shrinkage=100,
        bsl_options=None,
        min_rating=None,
        max_rating=None,
        unknown_strategy="global_mean",
        clip_range=None,
        verbose=False,
        name=None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.k = k
        self.min_k = min_k
        self.user_based = user_based
        self.sim_name = sim_name
        self.shrinkage = shrinkage
        self.bsl_options = bsl_options
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.unknown_strategy = unknown_strategy
        self.verbose = verbose

        self.model_ = None
        self.trainset_ = None
        self.global_mean_ = None
        self.rating_scale_ = None

    def fit(self, df):
        required_cols = {"user", "item", "rating"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas obligatorias en fit: {missing}")

        train_df = df[["user", "item", "rating"]].copy()

        # Surprise trabaja mejor con strings para ids arbitrarios
        train_df["user"] = train_df["user"].astype(str)
        train_df["item"] = train_df["item"].astype(str)
        train_df["rating"] = train_df["rating"].astype(float)

        min_rating = self.min_rating if self.min_rating is not None else float(train_df["rating"].min())
        max_rating = self.max_rating if self.max_rating is not None else float(train_df["rating"].max())

        if min_rating >= max_rating:
            raise ValueError(
                f"rating_scale inválido: min_rating={min_rating}, max_rating={max_rating}"
            )

        self.rating_scale_ = (min_rating, max_rating)
        self.global_mean_ = float(train_df["rating"].mean())

        reader = Reader(rating_scale=self.rating_scale_)
        data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)
        self.trainset_ = data.build_full_trainset()

        sim_options = {
            "name": self.sim_name,
            "user_based": self.user_based,
        }

        if self.shrinkage is not None:
            sim_options["shrinkage"] = self.shrinkage

        self.model_ = KNNBaseline(
            k=self.k,
            min_k=self.min_k,
            sim_options=sim_options,
            bsl_options=self.bsl_options,
            verbose=self.verbose,
        )

        self.model_.fit(self.trainset_)

        self.is_fitted_ = True

        if self.verbose:
            mode = "user-user" if self.user_based else "item-item"
            print(
                f"[{self.name}] entrenado | modo={mode} | "
                f"k={self.k} | min_k={self.min_k} | "
                f"sim={self.sim_name} | rating_scale={self.rating_scale_}"
            )

        return self

    def _fallback_prediction(self):
        """
        Fallback explícito cuando no queremos depender del comportamiento
        interno de Surprise para unknown users/items o predicciones imposibles.
        """
        if isinstance(self.unknown_strategy, (int, float)):
            return float(self.unknown_strategy)

        if self.unknown_strategy == "global_mean":
            return float(self.global_mean_)

        if self.unknown_strategy == "nan":
            return np.nan

        if self.unknown_strategy == "surprise":
            # señal de que no hay fallback manual; se usará el valor de Surprise
            return None

        raise ValueError(
            f"unknown_strategy no soportada: {self.unknown_strategy}"
        )

    def predict(self, user, item):
        self._check_fitted()

        raw_user = str(user)
        raw_item = str(item)

        manual_fallback = self._fallback_prediction()

        # Detectar si son desconocidos en train
        user_known = raw_user in self.trainset_._raw2inner_id_users
        item_known = raw_item in self.trainset_._raw2inner_id_items

        if (not user_known or not item_known) and self.unknown_strategy != "surprise":
            pred = manual_fallback
            return self._clip(pred) if not np.isnan(pred) else pred

        try:
            pred_obj = self.model_.predict(raw_user, raw_item)
            pred = float(pred_obj.est)
        except PredictionImpossible:
            pred = manual_fallback if manual_fallback is not None else self.global_mean_

        if np.isnan(pred):
            return pred

        return float(self._clip(pred))

    def __repr__(self):
        mode = "user-based" if self.user_based else "item-based"
        return (
            f"{self.name}("
            f"mode={mode}, k={self.k}, min_k={self.min_k}, "
            f"sim_name='{self.sim_name}')"
        )