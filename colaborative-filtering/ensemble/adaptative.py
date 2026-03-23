from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from model.base import BaseModel


class AdaptiveColdStartEnsemble(BaseModel):
    """
    Ensemble híbrido para recomendación con cold start.

    Idea:
    - `main_model` aprende la señal colaborativa principal.
    - una baseline robusta cubre casos de baja evidencia o cold start.
    - la predicción final mezcla ambas con un peso adaptativo según el
      número de interacciones observadas de usuario e ítem.

    Requisitos del modelo principal:
    - heredar de BaseModel
    - implementar fit(df) y predict(user, item)

    Requisitos de entrada en fit:
    - DataFrame con columnas ['user', 'item', 'rating']
    """

    def __init__(
        self,
        main_model: BaseModel,
        shrink_user: float = 10.0,
        shrink_item: float = 10.0,
        max_count_weight: int = 20,
        partial_cold_start_penalty: float = 0.35,
        full_cold_start_penalty: float = 0.0,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        use_item_popularity_in_weight: bool = True,
        fit_main_model: bool = True,
        clip_range=None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        if main_model is None:
            raise ValueError("`main_model` no puede ser None.")

        if shrink_user < 0 or shrink_item < 0:
            raise ValueError("`shrink_user` y `shrink_item` deben ser >= 0.")

        if max_count_weight <= 0:
            raise ValueError("`max_count_weight` debe ser > 0.")

        self.main_model = main_model
        self.shrink_user = float(shrink_user)
        self.shrink_item = float(shrink_item)
        self.max_count_weight = int(max_count_weight)
        self.partial_cold_start_penalty = float(partial_cold_start_penalty)
        self.full_cold_start_penalty = float(full_cold_start_penalty)
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.use_item_popularity_in_weight = bool(use_item_popularity_in_weight)
        self.fit_main_model = bool(fit_main_model)

        # modelo realmente entrenado (deepcopy para evitar efectos laterales)
        self.fitted_main_model_: Optional[BaseModel] = None

        # estadísticas de baseline
        self.global_mean_: Optional[float] = None
        self.user_mean_: Dict[Any, float] = {}
        self.item_mean_: Dict[Any, float] = {}
        self.user_count_: Dict[Any, int] = {}
        self.item_count_: Dict[Any, int] = {}
        self.user_sum_: Dict[Any, float] = {}
        self.item_sum_: Dict[Any, float] = {}

        # metadatos
        self.user_to_idx_: Dict[Any, int] = {}
        self.item_to_idx_: Dict[Any, int] = {}
        self.idx_to_user_ = []
        self.idx_to_item_ = []
        self.user_seen_items_: Dict[Any, set] = {}

    # =========================================================
    # API OBLIGATORIA
    # =========================================================
    def fit(self, df: pd.DataFrame):
        self._validate_fit_df(df)

        data = df[["user", "item", "rating"]].copy()

        self.global_mean_ = float(data["rating"].mean())

        if self.min_rating is None:
            self.min_rating_ = float(data["rating"].min())
        else:
            self.min_rating_ = float(self.min_rating)

        if self.max_rating is None:
            self.max_rating_ = float(data["rating"].max())
        else:
            self.max_rating_ = float(self.max_rating)

        if self.max_rating_ <= self.min_rating_:
            raise ValueError("`max_rating` debe ser mayor que `min_rating`.")

        users = data["user"].unique().tolist()
        items = data["item"].unique().tolist()

        self.user_to_idx_ = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user_ = users
        self.idx_to_item_ = items

        # historial de vistos
        self.user_seen_items_ = {}
        for user, g in data.groupby("user"):
            self.user_seen_items_[user] = set(g["item"].tolist())

        # estadísticas usuario/item para baseline shrinkage
        user_stats = data.groupby("user")["rating"].agg(["mean", "count", "sum"])
        item_stats = data.groupby("item")["rating"].agg(["mean", "count", "sum"])

        self.user_mean_ = user_stats["mean"].to_dict()
        self.user_count_ = user_stats["count"].astype(int).to_dict()
        self.user_sum_ = user_stats["sum"].to_dict()

        self.item_mean_ = item_stats["mean"].to_dict()
        self.item_count_ = item_stats["count"].astype(int).to_dict()
        self.item_sum_ = item_stats["sum"].to_dict()

        # entrenar modelo principal en una copia para no mutar la instancia externa
        self.fitted_main_model_ = copy.deepcopy(self.main_model)
        if self.fit_main_model:
            self.fitted_main_model_.fit(data)

        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        self._check_fitted()

        user_known = user in self.user_count_
        item_known = item in self.item_count_

        baseline = self._baseline_predict(user, item)

        # cold start absoluto -> baseline pura
        if not user_known and not item_known:
            return float(self._clip(baseline))

        model_pred = self._safe_model_predict(user, item, fallback=baseline)
        weight = self._confidence_weight(user, item)

        # penalizar explícitamente los casos parciales de cold start
        if not user_known or not item_known:
            weight *= self.partial_cold_start_penalty
        else:
            weight *= 1.0

        pred = weight * model_pred + (1.0 - weight) * baseline
        return float(self._clip(pred))

    # =========================================================
    # MÉTODOS EXTRA
    # =========================================================
    def predict_components(self, user, item) -> Dict[str, float]:
        """
        Devuelve los componentes de la predicción para depuración.
        """
        self._check_fitted()

        user_known = user in self.user_count_
        item_known = item in self.item_count_
        baseline = self._baseline_predict(user, item)
        model_pred = self._safe_model_predict(user, item, fallback=baseline)
        weight = self._confidence_weight(user, item)

        if not user_known or not item_known:
            weight *= self.partial_cold_start_penalty

        final_pred = weight * model_pred + (1.0 - weight) * baseline

        return {
            "baseline": float(self._clip(baseline)),
            "model": float(self._clip(model_pred)),
            "weight": float(np.clip(weight, 0.0, 1.0)),
            "prediction": float(self._clip(final_pred)),
            "user_known": float(user_known),
            "item_known": float(item_known),
            "user_count": float(self.user_count_.get(user, 0)),
            "item_count": float(self.item_count_.get(item, 0)),
        }

    def predict_df_with_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice un DataFrame completo y añade trazas útiles para análisis.
        """
        self._check_fitted()
        self._validate_predict_df(df)

        out = df[["user", "item"]].copy()
        details = [self.predict_components(row.user, row.item) for row in df.itertuples(index=False)]
        details_df = pd.DataFrame(details, index=out.index)
        out = pd.concat([out, details_df], axis=1)
        return out

    def recommend(self, user, top_k: int = 10, exclude_seen: bool = True):
        self._check_fitted()

        candidates = self.idx_to_item_
        if exclude_seen:
            seen = self.user_seen_items_.get(user, set())
            candidates = [item for item in candidates if item not in seen]

        scored = [(item, self.predict(user, item)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_main_model(self) -> BaseModel:
        self._check_fitted()
        return self.fitted_main_model_

    # =========================================================
    # BASELINE Y PESOS
    # =========================================================
    def _baseline_predict(self, user, item) -> float:
        g = self.global_mean_
        user_known = user in self.user_count_
        item_known = item in self.item_count_

        if user_known:
            user_mean_shrunk = self._shrunk_mean(
                mean_value=self.user_mean_[user],
                count=self.user_count_[user],
                global_mean=g,
                m=self.shrink_user,
            )
        else:
            user_mean_shrunk = g

        if item_known:
            item_mean_shrunk = self._shrunk_mean(
                mean_value=self.item_mean_[item],
                count=self.item_count_[item],
                global_mean=g,
                m=self.shrink_item,
            )
        else:
            item_mean_shrunk = g

        if user_known and item_known:
            # Mezcla simple y robusta. No uso suma de sesgos porque se dispara con pocas observaciones.
            pred = 0.5 * user_mean_shrunk + 0.5 * item_mean_shrunk
        elif user_known:
            pred = user_mean_shrunk
        elif item_known:
            pred = item_mean_shrunk
        else:
            pred = g

        return float(pred)

    def _confidence_weight(self, user, item) -> float:
        """
        Peso del modelo principal en [0, 1].

        Cuanta más evidencia tiene el usuario y el ítem, más dejamos mandar al
        modelo principal. Si no hay evidencia suficiente, domina la baseline.
        """
        user_count = self.user_count_.get(user, 0)
        item_count = self.item_count_.get(item, 0)

        conf_user = min(user_count / self.max_count_weight, 1.0)
        if self.use_item_popularity_in_weight:
            conf_item = min(item_count / self.max_count_weight, 1.0)
        else:
            conf_item = 1.0

        weight = float(np.sqrt(conf_user * conf_item))
        return float(np.clip(weight, 0.0, 1.0))

    @staticmethod
    def _shrunk_mean(mean_value: float, count: int, global_mean: float, m: float) -> float:
        if count < 0:
            raise ValueError("`count` no puede ser negativo.")
        if m < 0:
            raise ValueError("`m` no puede ser negativo.")
        if count == 0:
            return float(global_mean)
        return float((count * mean_value + m * global_mean) / (count + m))

    # =========================================================
    # UTILIDADES INTERNAS
    # =========================================================
    def _safe_model_predict(self, user, item, fallback: float) -> float:
        """
        Protege contra modelos que devuelvan NaN/inf o fallen en cold start.
        """
        if self.fitted_main_model_ is None:
            return float(fallback)

        try:
            pred = float(self.fitted_main_model_.predict(user, item))
        except Exception:
            return float(fallback)

        if not np.isfinite(pred):
            return float(fallback)

        return float(pred)

    def _validate_fit_df(self, df: pd.DataFrame):
        required = {"user", "item", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en fit: {missing}")
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _validate_predict_df(self, df: pd.DataFrame):
        required = {"user", "item"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en predict_df_with_details: {missing}")

    # =========================================================
    # AYUDA PARA DEPURACIÓN / ANÁLISIS
    # =========================================================
    def summarize_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resume cuántas filas hay por caso de cold start en un DataFrame.
        Requiere columnas: user, item.
        """
        self._check_fitted()
        self._validate_predict_df(df)

        tmp = df[["user", "item"]].copy()
        tmp["user_known"] = tmp["user"].isin(self.user_count_)
        tmp["item_known"] = tmp["item"].isin(self.item_count_)

        tmp["case"] = "both_known"
        tmp.loc[~tmp["user_known"] & tmp["item_known"], "case"] = "new_user"
        tmp.loc[tmp["user_known"] & ~tmp["item_known"], "case"] = "new_item"
        tmp.loc[~tmp["user_known"] & ~tmp["item_known"], "case"] = "new_both"

        summary = (
            tmp.groupby("case")
            .size()
            .rename("n_rows")
            .reset_index()
            .sort_values("n_rows", ascending=False)
            .reset_index(drop=True)
        )
        return summary

class AdaptivePosteriorColdStartEnsemble(BaseModel):
    """
    Ensemble adaptativo para recomendación con cold start y gating por posterior.

    Idea:
    - `main_model` aprende la señal colaborativa principal.
    - una baseline robusta cubre casos de baja evidencia o cold start.
    - la predicción final mezcla ambas con un peso adaptativo según:
        * número de interacciones del usuario e ítem
        * confianza del posterior del usuario (si su embedding es informativo)
        * confianza del posterior del ítem (si sus probabilidades están lejos del prior)

    Para sacar partido al gating posterior, el modelo principal debería exponer:
    - get_user_embedding(user)  -> vector 1D de tamaño K, idealmente prob. por factor
    - get_item_embedding(item)  -> vector 1D de tamaño K, idealmente prob. por factor o score por factor

    Esta clase sigue funcionando si esos métodos no existen, pero en ese caso el peso
    posterior se degrada automáticamente a un valor neutro.

    Requisitos de entrada en fit:
    - DataFrame con columnas ['user', 'item', 'rating']
    """

    def __init__(
        self,
        main_model: BaseModel,
        shrink_user: float = 10.0,
        shrink_item: float = 10.0,
        max_count_weight: int = 20,
        partial_cold_start_penalty: float = 0.35,
        full_cold_start_penalty: float = 0.0,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        use_item_popularity_in_weight: bool = True,
        fit_main_model: bool = True,
        # mezcla de señales de confianza
        count_weight_strength: float = 0.50,
        posterior_weight_strength: float = 0.50,
        # escalado de confianza posterior
        posterior_user_strength: float = 1.0,
        posterior_item_strength: float = 1.5,
        posterior_neutral_value: float = 0.25,
        # seguridad extra cuando el modelo principal colapsa al baseline
        collapse_penalty_threshold: float = 0.20,
        collapse_penalty_strength: float = 0.50,
        clip_range=None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        if main_model is None:
            raise ValueError("`main_model` no puede ser None.")
        if shrink_user < 0 or shrink_item < 0:
            raise ValueError("`shrink_user` y `shrink_item` deben ser >= 0.")
        if max_count_weight <= 0:
            raise ValueError("`max_count_weight` debe ser > 0.")
        if count_weight_strength < 0 or posterior_weight_strength < 0:
            raise ValueError("Las strengths deben ser >= 0.")
        if count_weight_strength == 0 and posterior_weight_strength == 0:
            raise ValueError("Al menos una de las strengths debe ser > 0.")
        if not (0.0 <= posterior_neutral_value <= 1.0):
            raise ValueError("`posterior_neutral_value` debe estar en [0, 1].")
        if collapse_penalty_threshold < 0:
            raise ValueError("`collapse_penalty_threshold` debe ser >= 0.")
        if not (0.0 <= collapse_penalty_strength <= 1.0):
            raise ValueError("`collapse_penalty_strength` debe estar en [0, 1].")

        self.main_model = main_model
        self.shrink_user = float(shrink_user)
        self.shrink_item = float(shrink_item)
        self.max_count_weight = int(max_count_weight)
        self.partial_cold_start_penalty = float(partial_cold_start_penalty)
        self.full_cold_start_penalty = float(full_cold_start_penalty)
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.use_item_popularity_in_weight = bool(use_item_popularity_in_weight)
        self.fit_main_model = bool(fit_main_model)

        self.count_weight_strength = float(count_weight_strength)
        self.posterior_weight_strength = float(posterior_weight_strength)
        self.posterior_user_strength = float(posterior_user_strength)
        self.posterior_item_strength = float(posterior_item_strength)
        self.posterior_neutral_value = float(posterior_neutral_value)

        self.collapse_penalty_threshold = float(collapse_penalty_threshold)
        self.collapse_penalty_strength = float(collapse_penalty_strength)

        self.fitted_main_model_: Optional[BaseModel] = None

        # estadísticas de baseline
        self.global_mean_: Optional[float] = None
        self.user_mean_: Dict[Any, float] = {}
        self.item_mean_: Dict[Any, float] = {}
        self.user_count_: Dict[Any, int] = {}
        self.item_count_: Dict[Any, int] = {}

        # metadatos
        self.user_to_idx_: Dict[Any, int] = {}
        self.item_to_idx_: Dict[Any, int] = {}
        self.idx_to_user_ = []
        self.idx_to_item_ = []
        self.user_seen_items_: Dict[Any, set] = {}

    # =========================================================
    # API OBLIGATORIA
    # =========================================================
    def fit(self, df: pd.DataFrame):
        self._validate_fit_df(df)

        data = df[["user", "item", "rating"]].copy()
        self.global_mean_ = float(data["rating"].mean())

        if self.min_rating is None:
            self.min_rating_ = float(data["rating"].min())
        else:
            self.min_rating_ = float(self.min_rating)

        if self.max_rating is None:
            self.max_rating_ = float(data["rating"].max())
        else:
            self.max_rating_ = float(self.max_rating)

        if self.max_rating_ <= self.min_rating_:
            raise ValueError("`max_rating` debe ser mayor que `min_rating`.")

        users = data["user"].unique().tolist()
        items = data["item"].unique().tolist()

        self.user_to_idx_ = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user_ = users
        self.idx_to_item_ = items

        self.user_seen_items_ = {}
        for user, g in data.groupby("user"):
            self.user_seen_items_[user] = set(g["item"].tolist())

        user_stats = data.groupby("user")["rating"].agg(["mean", "count"])
        item_stats = data.groupby("item")["rating"].agg(["mean", "count"])

        self.user_mean_ = user_stats["mean"].to_dict()
        self.user_count_ = user_stats["count"].astype(int).to_dict()
        self.item_mean_ = item_stats["mean"].to_dict()
        self.item_count_ = item_stats["count"].astype(int).to_dict()

        self.fitted_main_model_ = copy.deepcopy(self.main_model)
        if self.fit_main_model:
            self.fitted_main_model_.fit(data)

        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        self._check_fitted()

        user_known = user in self.user_count_
        item_known = item in self.item_count_

        baseline = self._baseline_predict(user, item)

        # cold start absoluto -> baseline pura
        if not user_known and not item_known:
            return float(self._clip(baseline))

        model_pred = self._safe_model_predict(user, item, fallback=baseline)
        weight = self._adaptive_weight(user, item, baseline=baseline, model_pred=model_pred)

        if not user_known or not item_known:
            weight *= self.partial_cold_start_penalty
        else:
            weight *= 1.0

        if not user_known and not item_known:
            weight *= self.full_cold_start_penalty

        pred = weight * model_pred + (1.0 - weight) * baseline
        return float(self._clip(pred))

    # =========================================================
    # MÉTODOS EXTRA
    # =========================================================
    def predict_components(self, user, item) -> Dict[str, float]:
        self._check_fitted()

        user_known = user in self.user_count_
        item_known = item in self.item_count_
        baseline = self._baseline_predict(user, item)
        model_pred = self._safe_model_predict(user, item, fallback=baseline)

        a_u = self._safe_get_user_embedding(user)
        b_i = self._safe_get_item_embedding(item)

        user_conf = self._posterior_user_confidence(a_u)
        item_conf = self._posterior_item_confidence(b_i)
        count_weight = self._count_weight(user, item)
        posterior_weight = self._posterior_weight(user_conf, item_conf)
        collapse_penalty = self._collapse_penalty(baseline, model_pred)
        raw_weight = self._combine_weights(count_weight, posterior_weight) * collapse_penalty

        weight = raw_weight
        if not user_known or not item_known:
            weight *= self.partial_cold_start_penalty

        final_pred = weight * model_pred + (1.0 - weight) * baseline

        return {
            "baseline": float(self._clip(baseline)),
            "model": float(self._clip(model_pred)),
            "count_weight": float(np.clip(count_weight, 0.0, 1.0)),
            "posterior_weight": float(np.clip(posterior_weight, 0.0, 1.0)),
            "collapse_penalty": float(np.clip(collapse_penalty, 0.0, 1.0)),
            "weight": float(np.clip(weight, 0.0, 1.0)),
            "prediction": float(self._clip(final_pred)),
            "user_conf": float(np.clip(user_conf, 0.0, 1.0)),
            "item_conf": float(np.clip(item_conf, 0.0, 1.0)),
            "user_known": float(user_known),
            "item_known": float(item_known),
            "user_count": float(self.user_count_.get(user, 0)),
            "item_count": float(self.item_count_.get(item, 0)),
        }

    def predict_df_with_details(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._validate_predict_df(df)

        out = df[["user", "item"]].copy()
        details = [self.predict_components(row.user, row.item) for row in df.itertuples(index=False)]
        details_df = pd.DataFrame(details, index=out.index)
        return pd.concat([out, details_df], axis=1)

    def recommend(self, user, top_k: int = 10, exclude_seen: bool = True):
        self._check_fitted()

        candidates = self.idx_to_item_
        if exclude_seen:
            seen = self.user_seen_items_.get(user, set())
            candidates = [item for item in candidates if item not in seen]

        scored = [(item, self.predict(user, item)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_main_model(self) -> BaseModel:
        self._check_fitted()
        return self.fitted_main_model_

    # =========================================================
    # BASELINE Y PESOS
    # =========================================================
    def _baseline_predict(self, user, item) -> float:
        g = self.global_mean_
        user_known = user in self.user_count_
        item_known = item in self.item_count_

        if user_known:
            user_mean_shrunk = self._shrunk_mean(
                mean_value=self.user_mean_[user],
                count=self.user_count_[user],
                global_mean=g,
                m=self.shrink_user,
            )
        else:
            user_mean_shrunk = g

        if item_known:
            item_mean_shrunk = self._shrunk_mean(
                mean_value=self.item_mean_[item],
                count=self.item_count_[item],
                global_mean=g,
                m=self.shrink_item,
            )
        else:
            item_mean_shrunk = g

        if user_known and item_known:
            pred = 0.5 * user_mean_shrunk + 0.5 * item_mean_shrunk
        elif user_known:
            pred = user_mean_shrunk
        elif item_known:
            pred = item_mean_shrunk
        else:
            pred = g

        return float(pred)

    def _adaptive_weight(self, user, item, baseline: float, model_pred: float) -> float:
        a_u = self._safe_get_user_embedding(user)
        b_i = self._safe_get_item_embedding(item)

        user_conf = self._posterior_user_confidence(a_u)
        item_conf = self._posterior_item_confidence(b_i)
        count_weight = self._count_weight(user, item)
        posterior_weight = self._posterior_weight(user_conf, item_conf)
        collapse_penalty = self._collapse_penalty(baseline, model_pred)

        weight = self._combine_weights(count_weight, posterior_weight)
        weight *= collapse_penalty

        return float(np.clip(weight, 0.0, 1.0))

    def _count_weight(self, user, item) -> float:
        user_count = self.user_count_.get(user, 0)
        item_count = self.item_count_.get(item, 0)

        conf_user = min(user_count / self.max_count_weight, 1.0)
        if self.use_item_popularity_in_weight:
            conf_item = min(item_count / self.max_count_weight, 1.0)
        else:
            conf_item = 1.0

        return float(np.clip(np.sqrt(conf_user * conf_item), 0.0, 1.0))

    def _posterior_weight(self, user_conf: float, item_conf: float) -> float:
        u = np.clip(user_conf * self.posterior_user_strength, 0.0, 1.0)
        i = np.clip(item_conf * self.posterior_item_strength, 0.0, 1.0)
        return float(np.clip(np.sqrt(u * i), 0.0, 1.0))

    def _combine_weights(self, count_weight: float, posterior_weight: float) -> float:
        total = self.count_weight_strength + self.posterior_weight_strength
        cw = self.count_weight_strength / total
        pw = self.posterior_weight_strength / total
        return float(np.clip(cw * count_weight + pw * posterior_weight, 0.0, 1.0))

    def _collapse_penalty(self, baseline: float, model_pred: float) -> float:
        # Si el modelo principal está casi calcando la baseline, no merece mucho peso.
        diff = abs(float(model_pred) - float(baseline))
        scale = max(self.max_rating_ - self.min_rating_, 1e-12)
        norm_diff = diff / scale

        if norm_diff >= self.collapse_penalty_threshold:
            return 1.0

        if self.collapse_penalty_threshold <= 0:
            return 1.0

        ratio = norm_diff / self.collapse_penalty_threshold
        # entre (1-strength) y 1.0
        penalty = (1.0 - self.collapse_penalty_strength) + self.collapse_penalty_strength * ratio
        return float(np.clip(penalty, 0.0, 1.0))

    @staticmethod
    def _shrunk_mean(mean_value: float, count: int, global_mean: float, m: float) -> float:
        if count < 0:
            raise ValueError("`count` no puede ser negativo.")
        if m < 0:
            raise ValueError("`m` no puede ser negativo.")
        if count == 0:
            return float(global_mean)
        return float((count * mean_value + m * global_mean) / (count + m))

    # =========================================================
    # CONFIANZA POSTERIOR
    # =========================================================
    def _safe_get_user_embedding(self, user):
        if self.fitted_main_model_ is None:
            return None
        if not hasattr(self.fitted_main_model_, "get_user_embedding"):
            return None
        try:
            emb = self.fitted_main_model_.get_user_embedding(user)
        except Exception:
            return None
        return self._sanitize_vector(emb)

    def _safe_get_item_embedding(self, item):
        if self.fitted_main_model_ is None:
            return None
        if not hasattr(self.fitted_main_model_, "get_item_embedding"):
            return None
        try:
            emb = self.fitted_main_model_.get_item_embedding(item)
        except Exception:
            return None
        return self._sanitize_vector(emb)

    def _sanitize_vector(self, x):
        if x is None:
            return None
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr

    def _posterior_user_confidence(self, a_u) -> float:
        if a_u is None:
            return self.posterior_neutral_value

        s = a_u.sum()
        if s <= 0:
            return self.posterior_neutral_value

        p = a_u / s
        k = len(p)
        if k <= 1:
            return 1.0

        uniform = np.full(k, 1.0 / k)
        dist = np.abs(p - uniform).sum()
        max_dist = 2.0 * (1.0 - 1.0 / k)
        if max_dist <= 0:
            return self.posterior_neutral_value

        conf = dist / max_dist
        return float(np.clip(conf, 0.0, 1.0))

    def _posterior_item_confidence(self, b_i) -> float:
        if b_i is None:
            return self.posterior_neutral_value

        # Para NNBPMF, b_i suele ser una probabilidad en [0,1] con prior neutro 0.5.
        # Cuanto más se aleja de 0.5, más informativo consideramos el ítem.
        conf = 2.0 * np.mean(np.abs(b_i - 0.5))
        return float(np.clip(conf, 0.0, 1.0))

    # =========================================================
    # UTILIDADES INTERNAS
    # =========================================================
    def _safe_model_predict(self, user, item, fallback: float) -> float:
        if self.fitted_main_model_ is None:
            return float(fallback)

        try:
            pred = float(self.fitted_main_model_.predict(user, item))
        except Exception:
            return float(fallback)

        if not np.isfinite(pred):
            return float(fallback)

        return float(pred)

    def _validate_fit_df(self, df: pd.DataFrame):
        required = {"user", "item", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en fit: {missing}")
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _validate_predict_df(self, df: pd.DataFrame):
        required = {"user", "item"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en predict_df_with_details: {missing}")

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a `fit`.")

    def _clip(self, x: float) -> float:
        if self.clip_range is not None:
            return float(np.clip(x, self.clip_range[0], self.clip_range[1]))
        return float(np.clip(x, self.min_rating_, self.max_rating_))

    # =========================================================
    # AYUDA PARA DEPURACIÓN / ANÁLISIS
    # =========================================================
    def summarize_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._validate_predict_df(df)

        tmp = df[["user", "item"]].copy()
        tmp["user_known"] = tmp["user"].isin(self.user_count_)
        tmp["item_known"] = tmp["item"].isin(self.item_count_)

        tmp["case"] = "both_known"
        tmp.loc[~tmp["user_known"] & tmp["item_known"], "case"] = "new_user"
        tmp.loc[tmp["user_known"] & ~tmp["item_known"], "case"] = "new_item"
        tmp.loc[~tmp["user_known"] & ~tmp["item_known"], "case"] = "new_both"

        return (
            tmp.groupby("case")
            .size()
            .rename("n_rows")
            .reset_index()
            .sort_values("n_rows", ascending=False)
            .reset_index(drop=True)
        )
