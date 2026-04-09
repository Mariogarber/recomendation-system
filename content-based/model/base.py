from __future__ import annotations

from abc import ABC, abstractmethod

import joblib
import numpy as np


class BaseModel(ABC):
    """Minimal common interface for content-based models."""

    def __init__(self, name: str | None = None, clip_range: tuple[float, float] | None = None):
        self.name = name or self.__class__.__name__
        self.clip_range = clip_range
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, df):
        """Fit the model on a dataframe with user/item/rating columns."""

    @abstractmethod
    def predict(self, user, item) -> float:
        """Predict a rating for a single user-item pair."""

    def predict_df(self, df, round_predictions: bool = False):
        self._check_fitted()

        out = df.copy()
        out["prediction"] = [
            self.predict(user, item)
            for user, item in zip(out["user"], out["item"])
        ]

        if round_predictions:
            out["prediction"] = out["prediction"].round()
        return out

    def mae(self, df, round_predictions: bool = False) -> float:
        pred_df = self.predict_df(df, round_predictions=round_predictions)
        y_true = pred_df["rating"].to_numpy(dtype=float)
        y_pred = pred_df["prediction"].to_numpy(dtype=float)
        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    def rmse(self, df) -> float:
        pred_df = self.predict_df(df)
        y_true = pred_df["rating"].to_numpy(dtype=float)
        y_pred = pred_df["prediction"].to_numpy(dtype=float)
        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

    def _clip(self, prediction: float) -> float:
        if self.clip_range is None:
            return float(prediction)
        return float(np.clip(prediction, self.clip_range[0], self.clip_range[1]))

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"Model {self.name} is not fitted.")

    def save(self, filepath: str) -> None:
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        return joblib.load(filepath)

    def __repr__(self) -> str:
        return f"{self.name}()"
