from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Interfaz base para todos los modelos de recomendación.

    Requisitos mínimos:
        - fit(df)
        - predict(user, item)
    """

    def __init__(self, name=None, clip_range=None):
        self.name = name if name is not None else self.__class__.__name__
        self.clip_range = clip_range
        self.is_fitted_ = False

    # =========================
    # MÉTODOS OBLIGATORIOS
    # =========================
    @abstractmethod
    def fit(self, df):
        """
        Entrena el modelo.
        df: DataFrame con columnas ['user', 'item', 'rating']
        """
        pass

    @abstractmethod
    def predict(self, user, item):
        """
        Predice rating para (user, item)
        """
        pass

    # =========================
    # MÉTODOS COMUNES (DEFAULT)
    # =========================
    def predict_df(self, df, round_predictions=False):
        """
        Predicciones en batch.
        """
        self._check_fitted()

        preds = [
            self.predict(u, i)
            for u, i in zip(df["user"], df["item"])
        ]

        out = df.copy()
        out["prediction"] = preds
        if round_predictions:
            out["prediction"] = out["prediction"].round()
        return out

    def rmse(self, df):
        self._check_fitted()

        pred_df = self.predict_df(df)

        y_true = pred_df["rating"].to_numpy(dtype=float)
        y_pred = pred_df["prediction"].to_numpy(dtype=float)

        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            return np.nan

        return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

    def mae(self, df, round_predictions=False):
        self._check_fitted()

        pred_df = self.predict_df(df, round_predictions=round_predictions)

        y_true = pred_df["rating"].to_numpy(dtype=float)
        y_pred = pred_df["prediction"].to_numpy(dtype=float)

        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            return np.nan

        return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    # =========================
    # UTILIDADES
    # =========================
    def _clip(self, pred):
        if self.clip_range is None:
            return pred
        return np.clip(pred, self.clip_range[0], self.clip_range[1])

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(f"El modelo {self.name} no está entrenado.")

    def __repr__(self):
        return f"{self.name}()"