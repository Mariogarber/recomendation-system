import numpy as np
import matplotlib.pyplot as plt
from .base import BaseModel


class MatrixFactorization(BaseModel):
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
        name=None
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.use_bias = use_bias
        self.init_std = init_std
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose

    # =========================
    # FIT
    # =========================
    def fit(self, df):
        required_cols = {"user", "item", "rating"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        df = df.copy()

        # Mapear ids
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
        self.user_bias_ = np.zeros(self.n_users_)
        self.item_bias_ = np.zeros(self.n_items_)

        self.train_history_ = []

        # =========================
        # SGD
        # =========================
        for epoch in range(self.n_epochs):

            if self.shuffle:
                rng.shuffle(ratings)

            se = 0.0

            for u, i, r in ratings:
                u = int(u)
                i = int(i)
                r = float(r)

                pred = self._predict_idx(u, i)
                err = r - pred

                se += err ** 2

                pu = self.P_[u].copy()
                qi = self.Q_[i].copy()

                # biases
                if self.use_bias:
                    self.user_bias_[u] += self.lr * (err - self.reg * self.user_bias_[u])
                    self.item_bias_[i] += self.lr * (err - self.reg * self.item_bias_[i])

                # factors
                self.P_[u] += self.lr * (err * qi - self.reg * pu)
                self.Q_[i] += self.lr * (err * pu - self.reg * qi)

            rmse = np.sqrt(se / len(ratings))
            self.train_history_.append(rmse)

            if self.verbose:
                print(f"[{self.name}] Epoch {epoch+1}/{self.n_epochs} - RMSE: {rmse:.4f}")

        self.is_fitted_ = True
        return self

    # =========================
    # PREDICCIÓN INTERNA
    # =========================
    def _predict_idx(self, u_idx, i_idx):
        pred = np.dot(self.P_[u_idx], self.Q_[i_idx])

        if self.use_bias:
            pred += self.global_mean_
            pred += self.user_bias_[u_idx]
            pred += self.item_bias_[i_idx]

        return self._clip(pred)

    # =========================
    # API PÚBLICA
    # =========================
    def predict(self, user, item):
        self._check_fitted()

        user_known = user in self.user_to_idx_
        item_known = item in self.item_to_idx_

        if not user_known and not item_known:
            pred = self.global_mean_

        elif not user_known:
            i = self.item_to_idx_[item]
            pred = self.global_mean_ + self.item_bias_[i]

        elif not item_known:
            u = self.user_to_idx_[user]
            pred = self.global_mean_ + self.user_bias_[u]

        else:
            u = self.user_to_idx_[user]
            i = self.item_to_idx_[item]
            pred = self._predict_idx(u, i)

        return float(self._clip(pred))

    # =========================
    # VISUALIZACIÓN
    # =========================
    def plot_training(self):
        self._check_fitted()

        if len(self.train_history_) == 0:
            raise ValueError("No hay historial de entrenamiento.")

        plt.figure()
        plt.plot(self.train_history_, marker='o')
        plt.title(f"{self.name} - Training RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.show()