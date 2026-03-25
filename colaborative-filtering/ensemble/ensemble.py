import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

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
        super().__init__(name=name, clip_range=clip_range)

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