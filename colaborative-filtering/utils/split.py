from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Any

import numpy as np
import pandas as pd


BandName = Literal["low", "mid", "high"]
SplitMode = Literal["quantile", "threshold"]
EntityType = Literal["user", "item"]


@dataclass
class BandSummary:
    band: BandName
    n_rows: int
    n_unique_entities: int
    min_count: Optional[int]
    max_count: Optional[int]
    mean_count: Optional[float]


class FrequencyBandSplitter:
    """
    Divide un dataframe de interacciones en 3 subconjuntos (low/mid/high)
    según la frecuencia de aparición de usuarios o ítems.

    Espera un DataFrame con columnas:
        - user
        - item
        - rating

    Ejemplo de uso:
        splitter = FrequencyBandSplitter(entity="item", mode="quantile")
        splitter.fit(df_train)
        splits = splitter.transform(df_test)

        df_low = splits["low"]
        df_mid = splits["mid"]
        df_high = splits["high"]

    Parámetros
    ----------
    entity : {"user", "item"}
        Sobre qué columna quieres medir frecuencia.
    mode : {"quantile", "threshold"}
        - "quantile": calcula automáticamente cortes en 3 grupos.
        - "threshold": usa umbrales manuales.
    low_threshold : int | None
        Solo para mode="threshold". Frecuencia <= low_threshold -> "low"
    high_threshold : int | None
        Solo para mode="threshold". Frecuencia >= high_threshold -> "high"
        Lo que quede entre ambos será "mid".
    include_unknown_as : {"low", "mid", "high"} | None
        Qué hacer con usuarios/ítems no vistos en fit.
        - Si None, se excluyen del resultado.
        - Si "low"/"mid"/"high", se asignan a esa banda.
    user_col, item_col, rating_col : str
        Nombres de columnas.
    """

    def __init__(
        self,
        entity: EntityType = "item",
        mode: SplitMode = "quantile",
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        include_unknown_as: Optional[BandName] = "low",
        user_col: str = "user",
        item_col: str = "item",
        rating_col: str = "rating",
    ) -> None:
        self.entity = entity
        self.mode = mode
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.include_unknown_as = include_unknown_as

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        self.entity_col = self.user_col if self.entity == "user" else self.item_col

        self.counts_: Optional[pd.Series] = None
        self.q1_: Optional[float] = None
        self.q2_: Optional[float] = None
        self.is_fitted_: bool = False

        self._validate_init()

    def _validate_init(self) -> None:
        if self.mode not in {"quantile", "threshold"}:
            raise ValueError("mode debe ser 'quantile' o 'threshold'.")

        if self.include_unknown_as not in {None, "low", "mid", "high"}:
            raise ValueError("include_unknown_as debe ser None, 'low', 'mid' o 'high'.")

        if self.mode == "threshold":
            if self.low_threshold is None or self.high_threshold is None:
                raise ValueError(
                    "En mode='threshold' debes indicar low_threshold y high_threshold."
                )
            if self.low_threshold >= self.high_threshold:
                raise ValueError(
                    "Debe cumplirse low_threshold < high_threshold."
                )

    def _validate_df(self, df: pd.DataFrame) -> None:
        required = {self.user_col, self.item_col, self.rating_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")

    def fit(self, df: pd.DataFrame) -> "FrequencyBandSplitter":
        """
        Aprende las frecuencias por usuario o por ítem usando df.
        """
        self._validate_df(df)

        counts = df[self.entity_col].value_counts(dropna=False).sort_index()
        self.counts_ = counts

        if self.mode == "quantile":
            # Cuantiles sobre la distribución de frecuencias de entidades
            freq_values = counts.values.astype(float)

            if len(freq_values) == 0:
                raise ValueError("El DataFrame está vacío.")

            self.q1_ = float(np.quantile(freq_values, 1 / 3))
            self.q2_ = float(np.quantile(freq_values, 2 / 3))

        self.is_fitted_ = True
        return self

    def _assign_band_from_count(self, count: Optional[int]) -> Optional[BandName]:
        """
        Asigna banda a una frecuencia.
        count=None significa entidad desconocida en transform.
        """
        if count is None:
            return self.include_unknown_as

        if self.mode == "threshold":
            if count <= self.low_threshold:   # type: ignore[operator]
                return "low"
            if count >= self.high_threshold:  # type: ignore[operator]
                return "high"
            return "mid"

        # mode == "quantile"
        assert self.q1_ is not None and self.q2_ is not None
        if count <= self.q1_:
            return "low"
        if count <= self.q2_:
            return "mid"
        return "high"

    def transform(
        self,
        df: pd.DataFrame,
        return_with_metadata: bool = False,
    ) -> Dict[BandName, pd.DataFrame]:
        """
        Divide df en 3 subconjuntos según las frecuencias aprendidas en fit().

        Importante:
        - Si transformas sobre test, la frecuencia usada es la aprendida en fit(train).
        - Así evitas leakage.
        """
        if not self.is_fitted_ or self.counts_ is None:
            raise RuntimeError("Debes ejecutar fit() antes de transform().")

        self._validate_df(df)

        df_out = df.copy()

        # Mapear frecuencia aprendida
        df_out["_entity_count"] = df_out[self.entity_col].map(self.counts_)

        # Convertir NaN (desconocidos) a None para tratarlo explícitamente
        def count_or_none(x: Any) -> Optional[int]:
            if pd.isna(x):
                return None
            return int(x)

        df_out["_band"] = df_out["_entity_count"].apply(
            lambda x: self._assign_band_from_count(count_or_none(x))
        )

        if self.include_unknown_as is None:
            df_out = df_out[df_out["_band"].notna()].copy()

        splits: Dict[BandName, pd.DataFrame] = {}
        for band in ("low", "mid", "high"):
            part = df_out[df_out["_band"] == band].copy()
            if not return_with_metadata:
                part = part.drop(columns=["_entity_count", "_band"], errors="ignore")
            splits[band] = part

        return splits

    def fit_transform(
        self,
        df: pd.DataFrame,
        return_with_metadata: bool = False,
    ) -> Dict[BandName, pd.DataFrame]:
        self.fit(df)
        return self.transform(df, return_with_metadata=return_with_metadata)

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve resumen del split aplicado a df.
        """
        splits = self.transform(df, return_with_metadata=True)

        rows: List[Dict[str, Any]] = []
        for band, part in splits.items():
            if len(part) == 0:
                rows.append(
                    {
                        "band": band,
                        "n_rows": 0,
                        "n_unique_entities": 0,
                        "min_count": None,
                        "max_count": None,
                        "mean_count": None,
                    }
                )
                continue

            rows.append(
                {
                    "band": band,
                    "n_rows": int(len(part)),
                    "n_unique_entities": int(part[self.entity_col].nunique()),
                    "min_count": int(part["_entity_count"].min()),
                    "max_count": int(part["_entity_count"].max()),
                    "mean_count": float(part["_entity_count"].mean()),
                }
            )

        return pd.DataFrame(rows).sort_values("band").reset_index(drop=True)

    def get_entity_frequencies(self) -> pd.Series:
        if not self.is_fitted_ or self.counts_ is None:
            raise RuntimeError("Debes ejecutar fit() antes de consultar frecuencias.")
        return self.counts_.copy()


class FrequencyBandEvaluator:
    """
    Evalúa un modelo por bandas de frecuencia.

    Requisitos del modelo:
        - método predict(user, item)

    Métricas soportadas:
        - rmse
        - mae

    También puedes pasar métricas custom.
    """

    def __init__(
        self,
        splitter: FrequencyBandSplitter,
        user_col: str = "user",
        item_col: str = "item",
        rating_col: str = "rating",
    ) -> None:
        self.splitter = splitter
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def _predict_dataframe(self, model: Any, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for row in df.itertuples(index=False):
            user = getattr(row, self.user_col)
            item = getattr(row, self.item_col)
            pred = model.predict(user, item)
            preds.append(pred)
        return np.asarray(preds, dtype=float)

    def evaluate(
        self,
        model: Any,
        df: pd.DataFrame,
        metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    ) -> pd.DataFrame:
        """
        Evalúa el modelo en low/mid/high sobre el df dado.
        """
        if metrics is None:
            metrics = {
                "rmse": self.rmse,
                "mae": self.mae,
            }

        splits = self.splitter.transform(df, return_with_metadata=False)
        rows = []

        for band, part in splits.items():
            if len(part) == 0:
                row = {"band": band, "n_rows": 0}
                for metric_name in metrics:
                    row[metric_name] = np.nan
                rows.append(row)
                continue

            y_true = part[self.rating_col].to_numpy(dtype=float)
            y_pred = self._predict_dataframe(model, part)

            row = {
                "band": band,
                "n_rows": int(len(part)),
            }
            for metric_name, metric_fn in metrics.items():
                row[metric_name] = metric_fn(y_true, y_pred)
            rows.append(row)

        return pd.DataFrame(rows).sort_values("band").reset_index(drop=True)