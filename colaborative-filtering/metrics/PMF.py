import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class MatrixFactorization:
    """
    Factorización matricial básica para ratings explícitos usando SGD.

    Modelo:
        r_hat(u, i) = global_mean + user_bias[u] + item_bias[i] + P[u] @ Q[i]

    Parámetros:
        n_factors: dimensión latente
        lr: learning rate
        reg: regularización L2
        n_epochs: número de épocas
        use_bias: si usar sesgos global/user/item
        init_std: desviación típica para inicializar embeddings
        random_state: semilla
        shuffle: si barajar interacciones en cada época
        clip_range: tuple (min_rating, max_rating) o None
        verbose: mostrar progreso
    """

    def __init__(
        self,
        n_factors=20,
        lr=0.01,
        reg=0.02,
        n_epochs=20,
        use_bias=True,
        init_std=0.1,
        random_state=42,
        shuffle=True,
        clip_range=None,
        verbose=True,
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.use_bias = use_bias
        self.init_std = init_std
        self.random_state = random_state
        self.shuffle = shuffle
        self.clip_range = clip_range
        self.verbose = verbose

        self.is_fitted_ = False

    def fit(self, df):
        """
        df: DataFrame con columnas ['user', 'item', 'rating']
        """
        required_cols = {"user", "item", "rating"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        df = df.copy()

        # Mapear ids externos a índices internos
        self.user_ids_ = df["user"].unique()
        self.item_ids_ = df["item"].unique()

        self.user_to_idx_ = {u: idx for idx, u in enumerate(self.user_ids_)}
        self.item_to_idx_ = {i: idx for idx, i in enumerate(self.item_ids_)}

        self.idx_to_user_ = {idx: u for u, idx in self.user_to_idx_.items()}
        self.idx_to_item_ = {idx: i for i, idx in self.item_to_idx_.items()}

        df["u_idx"] = df["user"].map(self.user_to_idx_)
        df["i_idx"] = df["item"].map(self.item_to_idx_)

        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)

        ratings = df[["u_idx", "i_idx", "rating"]].to_numpy()
        rng = np.random.default_rng(self.random_state)

        # Media global
        self.global_mean_ = df["rating"].mean() if self.use_bias else 0.0

        # Factores latentes
        self.P_ = rng.normal(0, self.init_std, size=(self.n_users_, self.n_factors))
        self.Q_ = rng.normal(0, self.init_std, size=(self.n_items_, self.n_factors))

        # Sesgos
        self.user_bias_ = np.zeros(self.n_users_, dtype=float)
        self.item_bias_ = np.zeros(self.n_items_, dtype=float)

        self.train_history_ = []

        for epoch in range(self.n_epochs):
            if self.shuffle:
                rng.shuffle(ratings)

            se = 0.0  # squared error acumulado

            for u, i, r in ratings:
                u = int(u)
                i = int(i)
                r = float(r)

                pred = self._predict_idx(u, i)
                err = r - pred

                se += err ** 2

                # Copias para update consistente
                pu = self.P_[u].copy()
                qi = self.Q_[i].copy()

                # Sesgos
                if self.use_bias:
                    self.user_bias_[u] += self.lr * (err - self.reg * self.user_bias_[u])
                    self.item_bias_[i] += self.lr * (err - self.reg * self.item_bias_[i])

                # Factores latentes
                self.P_[u] += self.lr * (err * qi - self.reg * pu)
                self.Q_[i] += self.lr * (err * pu - self.reg * qi)

            rmse = np.sqrt(se / len(ratings))
            self.train_history_.append(rmse)

            if self.verbose:
                print(f"Epoch {epoch+1:03d}/{self.n_epochs} - RMSE train: {rmse:.4f}")

        self.is_fitted_ = True
        return self

    def _predict_idx(self, u_idx, i_idx):
        pred = np.dot(self.P_[u_idx], self.Q_[i_idx])

        if self.use_bias:
            pred += self.global_mean_
            pred += self.user_bias_[u_idx]
            pred += self.item_bias_[i_idx]

        if self.clip_range is not None:
            pred = np.clip(pred, self.clip_range[0], self.clip_range[1])

        return pred

    def predict(self, user, item):
        """
        Predicción para ids externos.
        Manejo básico de cold start.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit().")

        user_known = user in self.user_to_idx_
        item_known = item in self.item_to_idx_

        # Casos cold-start
        if not user_known and not item_known:
            pred = self.global_mean_
        elif not user_known:
            i = self.item_to_idx_[item]
            pred = self.global_mean_ + self.item_bias_[i] if self.use_bias else self.global_mean_
        elif not item_known:
            u = self.user_to_idx_[user]
            pred = self.global_mean_ + self.user_bias_[u] if self.use_bias else self.global_mean_
        else:
            u = self.user_to_idx_[user]
            i = self.item_to_idx_[item]
            pred = self._predict_idx(u, i)

        if self.clip_range is not None:
            pred = np.clip(pred, self.clip_range[0], self.clip_range[1])

        return float(pred)

    def predict_df(self, df):
        """
        Devuelve predicciones para un DataFrame con columnas ['user', 'item'].
        """
        if not {"user", "item"}.issubset(df.columns):
            raise ValueError("El DataFrame debe tener columnas ['user', 'item'].")

        out = df.copy()
        out["prediction"] = [self.predict(u, i) for u, i in zip(out["user"], out["item"])]
        return out

    def rmse(self, df):
        """
        Calcula RMSE sobre un DataFrame con ['user', 'item', 'rating'].
        """
        if not {"user", "item", "rating"}.issubset(df.columns):
            raise ValueError("El DataFrame debe tener columnas ['user', 'item', 'rating'].")

        preds = np.array([self.predict(u, i) for u, i in zip(df["user"], df["item"])], dtype=float)
        y_true = df["rating"].to_numpy(dtype=float)
        return float(np.sqrt(np.mean((y_true - preds) ** 2)))

    def recommend(self, user, known_df, top_k=10):
        """
        Recomienda ítems no vistos para un usuario.

        known_df: DataFrame original de entrenamiento con ['user', 'item', 'rating']
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado.")

        if user not in self.user_to_idx_:
            raise ValueError("Usuario desconocido; este método no maneja bien cold start de recomendación.")

        seen_items = set(known_df.loc[known_df["user"] == user, "item"].unique())
        candidate_items = [item for item in self.item_ids_ if item not in seen_items]

        scores = [(item, self.predict(user, item)) for item in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(scores[:top_k], columns=["item", "score"])

    def plot_training(self):
        """
        Plotea la evolución del RMSE durante el entrenamiento.
        """
        if not hasattr(self, "train_history_") or len(self.train_history_) == 0:
            raise ValueError("No hay historial de entrenamiento. Ejecuta fit() primero.")


        plt.figure()
        plt.plot(self.train_history_, marker='o')
        plt.title("Training RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.show()