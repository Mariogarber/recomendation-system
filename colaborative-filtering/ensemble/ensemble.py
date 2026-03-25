import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from model.base import BaseModel

from typing import Optional, Dict, Any, Iterable


class RatingEnsemble:
    """
    Ensemble para predicción de ratings.

    Requisitos para cada modelo base:
        - método predict(user, item)

    Estrategias soportadas:
        - 'mean'
        - 'median'
        - 'weighted'
        - 'stacking'

    Parámetros
    ----------
    models : list
        Lista de modelos base.
    strategy : str
        Estrategia de combinación.
    weights : list[float] | None
        Pesos manuales para strategy='weighted'.
    clip_range : tuple | None
        Rango (min_rating, max_rating) para truncar predicciones.
    """

    def __init__(self, models, strategy="mean", weights=None, clip_range=None):
        self.models = models
        self.strategy = strategy
        self.clip_range = clip_range

        if len(models) == 0:
            raise ValueError("Debes proporcionar al menos un modelo.")

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if len(weights) != len(models):
                raise ValueError("El número de pesos debe coincidir con el número de modelos.")
            if np.all(weights == 0):
                raise ValueError("Los pesos no pueden ser todos cero.")
            self.weights = weights / weights.sum()
        else:
            self.weights = None

        self.meta_model_ = None
        self.is_fitted_ = False

    # =========================
    # Helpers internos
    # =========================
    def _clip(self, preds):
        if self.clip_range is None:
            return preds
        return np.clip(preds, self.clip_range[0], self.clip_range[1])

    def _base_predictions_one(self, user, item):
        preds = []
        for model in self.models:
            try:
                p = model.predict(user, item)
            except Exception:
                p = np.nan
            preds.append(p)
        return np.asarray(preds, dtype=float)

    def _base_predictions_df(self, df):
        """
        Devuelve matriz de shape (n_samples, n_models)
        """
        X = np.zeros((len(df), len(self.models)), dtype=float)

        for j, model in enumerate(self.models):
            col_preds = []
            for u, i in zip(df["user"], df["item"]):
                try:
                    p = model.predict(u, i)
                except Exception:
                    p = np.nan
                col_preds.append(p)
            X[:, j] = np.asarray(col_preds, dtype=float)

        return X

    def _combine_row(self, preds_row):
        mask = ~np.isnan(preds_row)
        valid_preds = preds_row[mask]

        if len(valid_preds) == 0:
            return np.nan

        if self.strategy == "mean":
            pred = np.mean(valid_preds)

        elif self.strategy == "median":
            pred = np.median(valid_preds)

        elif self.strategy == "weighted":
            if self.weights is None:
                raise ValueError("Para strategy='weighted' necesitas pesos.")
            valid_weights = self.weights[mask]
            valid_weights = valid_weights / valid_weights.sum()
            pred = np.sum(valid_weights * valid_preds)

        elif self.strategy == "stacking":
            if self.meta_model_ is None:
                raise RuntimeError("El meta-modelo no está ajustado. Usa fit_stacking().")
            # Relleno simple para NaNs
            row_filled = preds_row.copy()
            nan_mask = np.isnan(row_filled)
            if nan_mask.any():
                row_filled[nan_mask] = np.nanmean(row_filled) if not np.all(nan_mask) else 0.0
            pred = self.meta_model_.predict(row_filled.reshape(1, -1))[0]

        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")

        return float(self._clip(pred))

    # =========================
    # API pública
    # =========================
    def predict(self, user, item):
        preds = self._base_predictions_one(user, item)
        return self._combine_row(preds)

    def predict_df(self, df, round_predictions=False):
        preds = [self.predict(u, i) for u, i in zip(df["user"], df["item"])]
        out = df.copy()
        out["prediction"] = preds
        if round_predictions:
            out["prediction"] = out["prediction"].round()
        return out

    def rmse(self, df):
        if "rating" not in df.columns:
            raise ValueError("El DataFrame debe incluir la columna 'rating'.")

        pred_df = self.predict_df(df)
        y_true = pred_df["rating"].to_numpy(dtype=float)
        y_pred = pred_df["prediction"].to_numpy(dtype=float)

        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            return np.nan

        return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

    def mae(self, df, round_predictions=False):
        if "rating" not in df.columns:
            raise ValueError("El DataFrame debe incluir la columna 'rating'.")

        pred_df = self.predict_df(df, round_predictions=round_predictions)
        y_true = pred_df["rating"].to_numpy(dtype=float)
        y_pred = pred_df["prediction"].to_numpy(dtype=float)

        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            return np.nan

        return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    # =========================
    # Ajuste de pesos
    # =========================
    def fit_weights_from_errors(self, val_df, metric="rmse", temperature=1.0):
        """
        Ajusta pesos a partir del error individual de cada modelo.
        Usa softmax(-temperature * error).
        """
        errors = []

        for model in self.models:
            preds = []
            for u, i in zip(val_df["user"], val_df["item"]):
                try:
                    p = model.predict(u, i)
                except Exception:
                    p = np.nan
                preds.append(p)

            preds = np.asarray(preds, dtype=float)
            y_true = val_df["rating"].to_numpy(dtype=float)

            mask = ~np.isnan(preds)
            if mask.sum() == 0:
                err = np.inf
            else:
                if metric == "rmse":
                    err = np.sqrt(np.mean((y_true[mask] - preds[mask]) ** 2))
                elif metric == "mae":
                    err = np.mean(np.abs(y_true[mask] - preds[mask]))
                else:
                    raise ValueError("metric debe ser 'rmse' o 'mae'.")

            errors.append(err)

        errors = np.asarray(errors, dtype=float)

        # Softmax(-temp * error)
        scaled = -temperature * errors
        scaled -= np.max(scaled)  # estabilidad numérica
        weights = np.exp(scaled)
        weights /= weights.sum()

        self.weights = weights
        self.strategy = "weighted"
        self.is_fitted_ = True

        return weights

    def fit_weights_optimized(self, val_df):
        """
        Optimiza pesos para minimizar RMSE en validación.
        Restricciones:
            - pesos >= 0
            - sum(pesos) = 1
        """
        X = self._base_predictions_df(val_df)
        y = val_df["rating"].to_numpy(dtype=float)

        # Relleno simple de NaNs por media de fila
        row_means = np.nanmean(X, axis=1)
        row_means = np.where(np.isnan(row_means), 0.0, row_means)

        inds = np.where(np.isnan(X))
        X[inds] = row_means[inds[0]]

        n_models = X.shape[1]

        def objective(w):
            w = np.asarray(w, dtype=float)
            preds = X @ w
            preds = self._clip(preds)
            return np.sqrt(np.mean((y - preds) ** 2))

        x0 = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0)] * n_models
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints)

        if not result.success:
            raise RuntimeError(f"No se pudo optimizar pesos: {result.message}")

        self.weights = result.x
        self.strategy = "weighted"
        self.is_fitted_ = True

        return self.weights

    def fit_stacking(self, val_df, alpha=1.0):
        """
        Ajusta un meta-modelo Ridge sobre las predicciones base.
        """
        X = self._base_predictions_df(val_df)
        y = val_df["rating"].to_numpy(dtype=float)

        # Relleno simple de NaNs por media de columna
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)

        inds = np.where(np.isnan(X))
        X[inds] = col_means[inds[1]]

        self.meta_model_ = Ridge(alpha=alpha)
        self.meta_model_.fit(X, y)

        self.strategy = "stacking"
        self.is_fitted_ = True
        return self

    # =========================
    # Inspección
    # =========================
    def get_model_predictions(self, user, item):
        """
        Devuelve las predicciones individuales de cada modelo.
        """
        return self._base_predictions_one(user, item)

    def get_weights(self):
        return None if self.weights is None else self.weights.copy()

class ThresholdEnsembleModel:
    """
    Modelo híbrido basado en umbral de frecuencia del item.

    Idea:
        - si count(item) < threshold  -> usa rare_model
        - si count(item) >= threshold -> usa frequent_model

    Tanto rare_model como frequent_model pueden ser:
        - modelos individuales compatibles con BaseModel
        - ensembles que implementen predict(user, item)

    Esto hace que el objeto resultante también sea un BaseModel,
    así que puedes enchufarlo a cualquier Predictor existente
    sin crear un predictor especial.

    Parámetros
    ----------
    rare_model : BaseModel-like
        Modelo/ensemble para items raros.
    frequent_model : BaseModel-like
        Modelo/ensemble para items frecuentes.
    threshold : int
        Umbral de frecuencia del item en train.
    train_df : pd.DataFrame | None
        DataFrame con columnas ['user', 'item', 'rating'].
    item_counts : dict | pd.Series | None
        Conteos precomputados item -> frecuencia.
    clip_range : tuple | None
        Rango de clipping de predicción.
    default_prediction : float
        Predicción fallback si falla el modelo.
    unknown_item_policy : str
        {"rare", "frequent"} para items no vistos en train.
    name : str | None
        Nombre del modelo.
    verbose : bool
        Si True, imprime resumen en fit.
    """

    def __init__(
        self,
        rare_model,
        frequent_model,
        threshold: int,
        train_df: Optional[pd.DataFrame] = None,
        item_counts: Optional[Any] = None,
        clip_range: Optional[tuple] = None,
        default_prediction: float = 7.0,
        unknown_item_policy: str = "rare",
        name: Optional[str] = None,
        verbose: bool = False,
    ):

        if threshold < 1:
            raise ValueError("threshold debe ser >= 1")

        if unknown_item_policy not in {"rare", "frequent"}:
            raise ValueError("unknown_item_policy debe ser 'rare' o 'frequent'")

        self.rare_model = rare_model
        self.frequent_model = frequent_model
        self.threshold = threshold
        self.default_prediction = float(default_prediction)
        self.unknown_item_policy = unknown_item_policy
        self.verbose = verbose

        self.item_counts = self._build_item_counts(train_df=train_df, item_counts=item_counts)

        # Métricas internas de uso
        self.rare_predictions_count_ = 0
        self.frequent_predictions_count_ = 0
        self.unknown_item_count_ = 0
        self.errors_ = 0

        self.is_fitted_ = False

    # =========================
    # Helpers internos
    # =========================
    def _build_item_counts(
        self,
        train_df: Optional[pd.DataFrame],
        item_counts: Optional[Any],
    ) -> Dict[Any, int]:
        if item_counts is not None:
            if isinstance(item_counts, pd.Series):
                counts = item_counts.to_dict()
            elif isinstance(item_counts, dict):
                counts = dict(item_counts)
            else:
                raise TypeError("item_counts debe ser un dict o un pd.Series")

            return {k: int(v) for k, v in counts.items()}

        if train_df is None:
            raise ValueError("Debes proporcionar train_df o item_counts")

        required_cols = {"user", "item", "rating"}
        if not required_cols.issubset(train_df.columns):
            raise ValueError(
                f"train_df debe contener las columnas {required_cols}, "
                f"pero tiene {set(train_df.columns)}"
            )

        return train_df["item"].value_counts().astype(int).to_dict()

    def _safe_fit_submodel(self, model, df: pd.DataFrame):
        """
        Hace fit del submodelo solo si implementa fit y no parece ya ajustado.
        """
        if not hasattr(model, "fit"):
            return

        already_fitted = getattr(model, "is_fitted_", False)
        if already_fitted:
            return

        model.fit(df)

    def _select_model(self, item):
        count = self.item_counts.get(item, None)

        if count is None:
            self.unknown_item_count_ += 1
            return self.rare_model if self.unknown_item_policy == "rare" else self.frequent_model

        if count < self.threshold:
            return self.rare_model

        return self.frequent_model

    def _clip_prediction(self, pred: float) -> float:
        if self.clip_range is None:
            return float(pred)
        return float(np.clip(pred, self.clip_range[0], self.clip_range[1]))

    # =========================
    # API BaseModel
    # =========================
    def fit(self, df: pd.DataFrame):
        """
        Ajusta los submodelos si hace falta y recalcula item_counts desde df.
        """
        required_cols = {"user", "item", "rating"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"df debe contener las columnas {required_cols}, pero tiene {set(df.columns)}"
            )

        # Recalcular frecuencias desde el train real usado en fit
        self.item_counts = df["item"].value_counts().astype(int).to_dict()

        # Ajustar submodelos si no estaban ya ajustados
        self._safe_fit_submodel(self.rare_model, df)
        self._safe_fit_submodel(self.frequent_model, df)

        self.is_fitted_ = True

        if self.verbose:
            n_items = len(self.item_counts)
            n_rare = sum(v < self.threshold for v in self.item_counts.values())
            n_freq = sum(v >= self.threshold for v in self.item_counts.values())

            print(
                f"[ThresholdEnsembleModel] fitted | "
                f"n_items={n_items} | threshold={self.threshold} | "
                f"rare_items={n_rare} | frequent_items={n_freq}"
            )

        return self

    def predict(self, user, item):
        if not self.is_fitted_:
            # Si tus submodelos ya están entrenados y has pasado item_counts/train_df
            # puedes marcarlo a mano con self.is_fitted_ = True,
            # pero en general es mejor llamar a fit().
            raise RuntimeError("El modelo no está ajustado. Llama antes a fit().")

        model = self._select_model(item)

        if model is self.rare_model:
            self.rare_predictions_count_ += 1
        else:
            self.frequent_predictions_count_ += 1

        try:
            pred = model.predict(user, item)
        except Exception:
            pred = self.default_prediction
            self.errors_ += 1

        return self._clip_prediction(pred)

    def predict_df(self, df: pd.DataFrame, round_predictions: bool = False) -> pd.DataFrame:
        if not {"user", "item"}.issubset(df.columns):
            raise ValueError("df debe contener columnas ['user', 'item']")

        out = df.copy()
        out["prediction"] = [self.predict(u, i) for u, i in zip(df["user"], df["item"])]

        if round_predictions:
            out["prediction"] = out["prediction"].round()

        return out

    # =========================
    # Inspección
    # =========================
    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "rare_predictions": int(self.rare_predictions_count_),
            "frequent_predictions": int(self.frequent_predictions_count_),
            "unknown_items": int(self.unknown_item_count_),
            "errors": int(self.errors_),
        }

class ManualRoutingEnsemble(BaseModel):
    """
    Ensemble manual basado en reglas de decisión según presencia
    de user/item en el dataset de entrenamiento.

    Rutas:
        - warm:         user conocido, item conocido
        - cold_user:    user desconocido, item conocido
        - cold_item:    user conocido, item desconocido
        - cold_both:    user desconocido, item desconocido

    Además, opcionalmente puede considerar como "fríos" los users/items
    con muy pocas apariciones en train mediante umbrales.

    Parámetros
    ----------
    warm_model : BaseModel
        Modelo principal para pares warm (user,item conocidos).

    cold_user_model : BaseModel | None
        Modelo a usar cuando el user no está en train pero el item sí.
        Si es None, cae al fallback.

    cold_item_model : BaseModel | None
        Modelo a usar cuando el item no está en train pero el user sí.
        Si es None, cae al fallback.

    cold_both_model : BaseModel | None
        Modelo a usar cuando ni user ni item están en train.
        Si es None, cae al fallback.

    fallback_model : BaseModel | None
        Modelo de seguridad si falta alguno de los anteriores.
        Si también es None, usa la media global.

    fit_models : bool
        Si True, llama a fit(df) sobre todos los modelos no nulos.
        Si False, asume que ya están entrenados.

    user_cold_threshold : int
        Si > 0, un user con <= ese número de apariciones se trata como "cold".
        Ojo: esto cambia el routing incluso aunque el user exista en train.

    item_cold_threshold : int
        Igual que el anterior, pero para item.

    prefer_specialized_cold : bool
        Si True, los thresholds tienen prioridad sobre la mera presencia.
        Muy útil cuando quieres tratar distinto a users/items vistos 1 vez.
    """

    def __init__(
        self,
        warm_model,
        cold_user_model=None,
        cold_item_model=None,
        cold_both_model=None,
        fallback_model=None,
        fit_models=True,
        user_cold_threshold=0,
        item_cold_threshold=0,
        prefer_specialized_cold=True,
        clip_range=None,
        name=None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.warm_model = warm_model
        self.cold_user_model = cold_user_model
        self.cold_item_model = cold_item_model
        self.cold_both_model = cold_both_model
        self.fallback_model = fallback_model

        self.fit_models = fit_models
        self.user_cold_threshold = int(user_cold_threshold)
        self.item_cold_threshold = int(item_cold_threshold)
        self.prefer_specialized_cold = bool(prefer_specialized_cold)

        self.global_mean_ = None
        self.user_counts_ = None
        self.item_counts_ = None
        self.known_users_ = None
        self.known_items_ = None

    # =========================================================
    # FIT
    # =========================================================
    def fit(self, df: pd.DataFrame):
        self._validate_fit_df(df)

        df = df[["user", "item", "rating"]].copy()

        self.global_mean_ = float(df["rating"].mean())
        self.user_counts_ = df["user"].value_counts().to_dict()
        self.item_counts_ = df["item"].value_counts().to_dict()
        self.known_users_ = set(self.user_counts_.keys())
        self.known_items_ = set(self.item_counts_.keys())

        models = self._unique_models()

        if self.fit_models:
            for model in models:
                model.fit(df)

        self.is_fitted_ = True
        return self

    # =========================================================
    # PREDICT
    # =========================================================
    def predict(self, user, item):
        self._check_fitted()

        route, model = self._select_model(user, item)

        if model is None:
            pred = self.global_mean_
        else:
            pred = model.predict(user, item)

        return float(self._clip(pred))

    # =========================================================
    # EXPLICABILIDAD
    # =========================================================
    def explain_route(self, user, item):
        """
        Devuelve información de la ruta elegida para ese par.
        """
        self._check_fitted()

        user_known = user in self.known_users_
        item_known = item in self.known_items_

        user_count = self.user_counts_.get(user, 0)
        item_count = self.item_counts_.get(item, 0)

        user_is_cold = self._is_cold_user(user)
        item_is_cold = self._is_cold_item(item)

        route, model = self._select_model(user, item)

        return {
            "user_known": user_known,
            "item_known": item_known,
            "user_count": user_count,
            "item_count": item_count,
            "user_is_cold": user_is_cold,
            "item_is_cold": item_is_cold,
            "route": route,
            "model_name": None if model is None else model.name,
        }

    def predict_with_route(self, user, item):
        """
        Devuelve predicción y metadatos de la ruta usada.
        """
        pred = self.predict(user, item)
        info = self.explain_route(user, item)
        info["prediction"] = pred
        return info

    # =========================================================
    # BATCH
    # =========================================================
    def predict_df(self, df, round_predictions=False, return_route=False):
        self._check_fitted()

        preds = []
        routes = []
        model_names = []

        for u, i in zip(df["user"], df["item"]):
            route, model = self._select_model(u, i)

            if model is None:
                pred = self.global_mean_
                model_name = "global_mean"
            else:
                pred = model.predict(u, i)
                model_name = model.name

            preds.append(float(self._clip(pred)))
            routes.append(route)
            model_names.append(model_name)

        out = df.copy()
        out["prediction"] = preds

        if round_predictions:
            out["prediction"] = out["prediction"].round()

        if return_route:
            out["route"] = routes
            out["model_used"] = model_names

        return out

    # =========================================================
    # ROUTING
    # =========================================================
    def _select_model(self, user, item):
        user_known = user in self.known_users_
        item_known = item in self.known_items_

        user_is_cold = self._is_cold_user(user)
        item_is_cold = self._is_cold_item(item)

        # -----------------------------------------------------
        # Caso 1: routing estricto por "frialdad" especializada
        # -----------------------------------------------------
        if self.prefer_specialized_cold:
            if user_is_cold and item_is_cold:
                return "cold_both", self._resolve_model(self.cold_both_model)
            if user_is_cold and not item_is_cold:
                return "cold_user", self._resolve_model(self.cold_user_model)
            if not user_is_cold and item_is_cold:
                return "cold_item", self._resolve_model(self.cold_item_model)
            return "warm", self._resolve_model(self.warm_model)

        # -----------------------------------------------------
        # Caso 2: routing solo por presencia real en train
        # -----------------------------------------------------
        if user_known and item_known:
            return "warm", self._resolve_model(self.warm_model)
        if (not user_known) and item_known:
            return "cold_user", self._resolve_model(self.cold_user_model)
        if user_known and (not item_known):
            return "cold_item", self._resolve_model(self.cold_item_model)
        return "cold_both", self._resolve_model(self.cold_both_model)

    def _resolve_model(self, preferred_model):
        if preferred_model is not None:
            return preferred_model
        if self.fallback_model is not None:
            return self.fallback_model
        return None

    def _is_cold_user(self, user):
        count = self.user_counts_.get(user, 0)

        if user not in self.known_users_:
            return True

        if self.user_cold_threshold > 0 and count <= self.user_cold_threshold:
            return True

        return False

    def _is_cold_item(self, item):
        count = self.item_counts_.get(item, 0)

        if item not in self.known_items_:
            return True

        if self.item_cold_threshold > 0 and count <= self.item_cold_threshold:
            return True

        return False

    # =========================================================
    # HELPERS
    # =========================================================
    def _unique_models(self):
        models = [
            self.warm_model,
            self.cold_user_model,
            self.cold_item_model,
            self.cold_both_model,
            self.fallback_model,
        ]

        unique = []
        seen_ids = set()

        for m in models:
            if m is None:
                continue
            if id(m) in seen_ids:
                continue
            seen_ids.add(id(m))
            unique.append(m)

        return unique

    @staticmethod
    def _validate_fit_df(df):
        required = {"user", "item", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en fit: {missing}")
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")