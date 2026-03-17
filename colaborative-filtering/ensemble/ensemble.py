import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


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

    def predict_df(self, df):
        preds = [self.predict(u, i) for u, i in zip(df["user"], df["item"])]
        out = df.copy()
        out["prediction"] = preds
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

    def mae(self, df):
        if "rating" not in df.columns:
            raise ValueError("El DataFrame debe incluir la columna 'rating'.")

        pred_df = self.predict_df(df)
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