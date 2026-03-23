import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, NMF

from .base import BaseModel

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.utils.validation import check_is_fitted


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
                err = np.clip(r - pred, -100, 100)  # Evitar errores extremos

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

                self.P_[u] = np.clip(self.P_[u], -10, 10)
                self.Q_[i] = np.clip(self.Q_[i], -10, 10)

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
        

class PMFRegressor(BaseModel):
    """"
    Este modelo aprovecha las características latentes aprendidas por un modelo de 
    factorización matricial (MatrixFactorization) para entrenar un regresor 
    (por ejemplo, RandomForestRegressor) que predice ratings.
    """

    def __init__(self, pmf: MatrixFactorization, model: str | RandomForestRegressor | SVR | Ridge | Lasso | ElasticNet, name=None, clip_range=None, n_jobs=16):
        super().__init__(name, clip_range)
        self.pmf = pmf
        self.regressor = self._parse_model(model, n_jobs)
        try:
            check_is_fitted(model)
            self.is_fitted_ = True
        except Exception:
            self.is_fitted_ = False

    def _parse_model(self, model, n_jobs=16):
        if isinstance(model, str):
            model = model.lower()
            if model == "randomforest":
                return RandomForestRegressor(random_state=42, n_jobs=n_jobs)
            elif model == "svr":
                return SVR()
            elif model == "ridge":
                return Ridge(random_state=42,)
            elif model == "lasso":
                return Lasso(random_state=42)
            elif model == "elasticnet":
                return ElasticNet(random_state=42)
            else:
                raise ValueError(f"Modelo desconocido: {model}")
        elif isinstance(model, (RandomForestRegressor, SVR, Ridge, Lasso, ElasticNet)):
            return model
        else:
            raise ValueError("El modelo debe ser un string o una instancia de un regresor compatible.")

    def _build_feature_matrix(self, df: pd.DataFrame):
        required_cols = {"user", "item", "rating"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        
        df = df.copy()
        df["u_idx"] = df["user"].map(self.pmf.user_to_idx_)
        df["i_idx"] = df["item"].map(self.pmf.item_to_idx_)
        df = df.dropna(subset=["u_idx", "i_idx"])

        df["u_idx"] = df["u_idx"].astype(int)
        df["i_idx"] = df["i_idx"].astype(int)
        X = np.hstack([self.pmf.P_[df["u_idx"]], self.pmf.Q_[df["i_idx"]]])
        y = df["rating"].to_numpy()
        return X, y

    def fit(self, df: pd.DataFrame):
        X, y = self._build_feature_matrix(df)
        self.regressor.fit(X, y)
        self.is_fitted_ = True
        return self
        
    def predict(self, user, item):
        user_known = user in self.pmf.user_to_idx_
        item_known = item in self.pmf.item_to_idx_

        if not user_known or not item_known:
            pred = self.pmf.predict(user, item)
        else:
            u_idx = self.pmf.user_to_idx_[user]
            i_idx = self.pmf.item_to_idx_[item]
            features = np.hstack([self.pmf.P_[u_idx], self.pmf.Q_[i_idx]]).reshape(1, -1)
            pred = self.regressor.predict(features)[0]

        return float(self._clip(pred))


class SurpriseNMFModel(BaseModel):
    """
    Wrapper de Surprise NMF compatible con la API de BaseModel.

    Requisitos de entrada en fit:
        df con columnas ['user', 'item', 'rating']

    API heredada:
        - fit(df)
        - predict(user, item)

    Extras:
        - predict_batch(df)
        - recommend(user, top_k=10, exclude_seen=True)
        - get_user_embedding(user)
        - get_item_embedding(item)
        - get_user_bias(user)
        - get_item_bias(item)
        - explain_prediction(user, item)

    Notas:
        - NMF de Surprise usa factores no negativos.
        - Si biased=False:
              pred = <p_u, q_i>
        - Si biased=True:
              pred = mu + bu + bi + <p_u, q_i>
        - Para user/item desconocido, Surprise resuelve de forma robusta vía su API de predict.
    """

    def __init__(
        self,
        n_factors=15,
        n_epochs=50,
        biased=True,
        reg_pu=0.06,
        reg_qi=0.06,
        reg_bu=0.02,
        reg_bi=0.02,
        lr_bu=0.005,
        lr_bi=0.005,
        init_low=0.0,
        init_high=1.0,
        random_state=42,
        verbose=True,
        rating_scale=(1, 10),
        clip_range=None,
        name=None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.init_low = init_low
        self.init_high = init_high
        self.random_state = random_state
        self.verbose = verbose
        self.rating_scale = rating_scale

        # Internos
        self.algo_ = None
        self.trainset_ = None
        self.global_mean_ = None

        self.users_ = None
        self.items_ = None
        self.user_seen_items_ = None

    # =========================
    # Helpers internos
    # =========================
    def _validate_df(self, df: pd.DataFrame):
        required_cols = {"user", "item", "rating"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"El DataFrame debe contener las columnas {required_cols}. "
                f"Faltan: {missing}"
            )
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _clip(self, value: float) -> float:
        if self.clip_range is None:
            return float(value)
        lo, hi = self.clip_range
        return float(np.clip(value, lo, hi))

    def _build_surprise_dataset(self, df: pd.DataFrame):
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
        return data

    def _build_seen_dict(self, df: pd.DataFrame):
        return df.groupby("user")["item"].apply(set).to_dict()

    def _safe_inner_uid(self, user):
        try:
            return self.trainset_.to_inner_uid(user)
        except ValueError:
            return None

    def _safe_inner_iid(self, item):
        try:
            return self.trainset_.to_inner_iid(item)
        except ValueError:
            return None

    # =========================
    # API BaseModel
    # =========================
    def fit(self, df: pd.DataFrame):
        self._validate_df(df)
        df = df.copy()

        data = self._build_surprise_dataset(df)
        trainset = data.build_full_trainset()

        algo = NMF(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            biased=self.biased,
            reg_pu=self.reg_pu,
            reg_qi=self.reg_qi,
            reg_bu=self.reg_bu,
            reg_bi=self.reg_bi,
            lr_bu=self.lr_bu,
            lr_bi=self.lr_bi,
            init_low=self.init_low,
            init_high=self.init_high,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        algo.fit(trainset)

        self.algo_ = algo
        self.trainset_ = trainset
        self.global_mean_ = float(trainset.global_mean)

        self.users_ = set(df["user"].unique())
        self.items_ = set(df["item"].unique())
        self.user_seen_items_ = self._build_seen_dict(df)

        self.is_fitted_ = True
        return self

    def predict(self, user, item):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        est = float(self.algo_.predict(uid=user, iid=item).est)
        return self._clip(est)

    # =========================
    # Extras útiles
    # =========================
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        required_cols = {"user", "item"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"El DataFrame debe contener las columnas {required_cols}. "
                f"Faltan: {missing}"
            )

        preds = [
            self._clip(self.algo_.predict(uid=u, iid=i).est)
            for u, i in zip(df["user"].values, df["item"].values)
        ]
        return np.array(preds, dtype=float)

    def recommend(self, user, top_k=10, exclude_seen=True):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        candidate_items = self.items_

        if exclude_seen:
            seen = self.user_seen_items_.get(user, set())
            candidate_items = candidate_items - seen

        scored = [(item, self.predict(user, item)) for item in candidate_items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_user_embedding(self, user):
        """
        Devuelve el embedding latente no negativo del usuario.
        Si el usuario no existe, devuelve None.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        inner_uid = self._safe_inner_uid(user)
        if inner_uid is None:
            return None

        # En Surprise NMF, pu contiene los factores de usuario
        return np.array(self.algo_.pu[inner_uid], dtype=float)

    def get_item_embedding(self, item):
        """
        Devuelve el embedding latente no negativo del ítem.
        Si el ítem no existe, devuelve None.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        inner_iid = self._safe_inner_iid(item)
        if inner_iid is None:
            return None

        # En Surprise NMF, qi contiene los factores de ítem
        return np.array(self.algo_.qi[inner_iid], dtype=float)

    def get_user_bias(self, user):
        """
        Solo relevante si biased=True.
        Si el usuario no existe o biased=False, devuelve 0.0.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        if not self.biased:
            return 0.0

        inner_uid = self._safe_inner_uid(user)
        if inner_uid is None:
            return 0.0

        return float(self.algo_.bu[inner_uid])

    def get_item_bias(self, item):
        """
        Solo relevante si biased=True.
        Si el ítem no existe o biased=False, devuelve 0.0.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        if not self.biased:
            return 0.0

        inner_iid = self._safe_inner_iid(item)
        if inner_iid is None:
            return 0.0

        return float(self.algo_.bi[inner_iid])

    def explain_prediction(self, user, item):
        """
        Descompone la predicción.

        biased=False:
            pred = dot(pu, qi)

        biased=True:
            pred = mu + bu + bi + dot(pu, qi)
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado. Llama antes a fit(df).")

        pu = self.get_user_embedding(user)
        qi = self.get_item_embedding(item)

        dot = None
        if pu is not None and qi is not None:
            dot = float(np.dot(pu, qi))

        mu = float(self.global_mean_)
        bu = self.get_user_bias(user)
        bi = self.get_item_bias(item)

        # La predicción final la tomo desde Surprise para no inventar comportamiento
        pred = self.predict(user, item)

        return {
            "global_mean": mu if self.biased else 0.0,
            "user_bias": bu,
            "item_bias": bi,
            "dot_pu_qi": dot,
            "prediction": pred,
            "user_known": pu is not None,
            "item_known": qi is not None,
            "biased": self.biased,
        }