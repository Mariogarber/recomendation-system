import numpy as np
import pandas as pd

from typing import Optional, Dict, Any
import os

from model.base import BaseModel

class Predictor:
    def __init__(self, test_path: str, save_path: str, round_predictions: bool = False):
        self.path = test_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_path = save_path
        self.round_predictions = round_predictions

        self.errors = 0

    def load_test_from_csv(self):
        """
        Carga un CSV con columnas 'user', 'item' y 'prediction' y devuelve un DataFrame.
        """
        import pandas as pd
        df = pd.read_csv(self.path)
        if not all(col in df.columns for col in ["ID", "user", "item"]):
            raise ValueError("El CSV debe contener las columnas 'ID', 'user' y 'item'.")
        
        ids = df["ID"].to_numpy()
        return df[["user", "item"]], ids

    def compute_predictions(self, model: BaseModel, test_df: pd.DataFrame):
        """
        Dado un modelo y un DataFrame con columnas 'user' e 'item', devuelve un array con las predicciones.
        """
        preds = []
        for u, i in zip(test_df["user"], test_df["item"]):
            try:
                p = model.predict(u, i)
            except Exception:
                p = 7.0  # Valor por defecto en caso de error
                self.errors += 1

            if self.round_predictions:
                p = round(p)
            preds.append(p)
        return np.asarray(preds, dtype=float)
    
    def predict(self, model: BaseModel):
        test_df, ids = self.load_test_from_csv()
        predictions = self.compute_predictions(model, test_df)
        solution_df = pd.DataFrame({"ID": ids, "rating": predictions})
        solution_df.to_csv(self.save_path, index=False)
        print(f"Predictions saved to {self.save_path}. Number of errors: {self.errors}")
        return solution_df


class ThresholdItemPredictor(Predictor):
    """
    Predictor híbrido que hereda de Predictor y selecciona entre dos modelos
    según la frecuencia del item en train.

    Regla:
        - si count(item) < threshold  -> usa rare_model
        - si count(item) >= threshold -> usa frequent_model

    Requisitos:
        - test CSV con columnas: ['ID', 'user', 'item']
        - train_df con columnas: ['user', 'item', 'rating']
          o item_counts precomputado como dict / pd.Series
    """

    def __init__(
        self,
        test_path: str,
        save_path: str,
        rare_model: BaseModel,
        frequent_model: BaseModel,
        threshold: int,
        train_df: Optional[pd.DataFrame] = None,
        item_counts: Optional[Any] = None,
        round_predictions: bool = False,
        default_prediction: float = 7.0,
        unknown_item_policy: str = "rare",  # {"rare", "frequent"}
        verbose: bool = True,
    ):
        super().__init__(
            test_path=test_path,
            save_path=save_path,
            round_predictions=round_predictions,
        )

        if threshold < 1:
            raise ValueError("threshold debe ser >= 1")

        if unknown_item_policy not in {"rare", "frequent"}:
            raise ValueError("unknown_item_policy debe ser 'rare' o 'frequent'")

        self.rare_model = rare_model
        self.frequent_model = frequent_model
        self.threshold = threshold
        self.default_prediction = default_prediction
        self.unknown_item_policy = unknown_item_policy
        self.verbose = verbose

        self.item_counts = self._build_item_counts(train_df=train_df, item_counts=item_counts)

        self.rare_predictions_count = 0
        self.frequent_predictions_count = 0
        self.unknown_item_count = 0

    def _build_item_counts(
        self,
        train_df: Optional[pd.DataFrame],
        item_counts: Optional[Any],
    ) -> Dict[Any, int]:
        """
        Construye el diccionario item -> número de apariciones.
        """
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

    def _select_model(self, item):
        """
        Selecciona qué modelo usar para un item concreto.
        """
        count = self.item_counts.get(item, None)

        if count is None:
            self.unknown_item_count += 1
            if self.unknown_item_policy == "rare":
                return self.rare_model
            return self.frequent_model

        if count < self.threshold:
            return self.rare_model

        return self.frequent_model

    def compute_predictions(self, test_df: pd.DataFrame):
        """
        Sobrescribe el método base para decidir por fila qué modelo usar.
        """
        preds = []

        for u, i in zip(test_df["user"], test_df["item"]):
            model = self._select_model(i)

            if model is self.rare_model:
                self.rare_predictions_count += 1
            else:
                self.frequent_predictions_count += 1

            try:
                p = model.predict(u, i)
            except Exception:
                p = self.default_prediction
                self.errors += 1

            if self.round_predictions:
                p = round(p)

            preds.append(p)

        return np.asarray(preds, dtype=float)

    def predict(self):
        """
        Misma idea que Predictor.predict(), pero usando rare_model/frequent_model.
        """
        test_df, ids = self.load_test_from_csv()
        predictions = self.compute_predictions(test_df)

        solution_df = pd.DataFrame({
            "ID": ids,
            "rating": predictions
        })
        solution_df.to_csv(self.save_path, index=False)

        if self.verbose:
            print(f"Predictions saved to {self.save_path}")
            print(f"Errors: {self.errors}")
            print(f"Predictions with rare_model: {self.rare_predictions_count}")
            print(f"Predictions with frequent_model: {self.frequent_predictions_count}")
            print(f"Unknown items in test: {self.unknown_item_count}")

        return solution_df