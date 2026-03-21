import numpy as np
import pandas as pd

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