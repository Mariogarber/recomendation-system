import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma

from .base import BaseModel


@dataclass
class BPMFSample:
    U: np.ndarray
    V: np.ndarray
    mu_u: np.ndarray
    mu_v: np.ndarray
    Lambda_u: np.ndarray
    Lambda_v: np.ndarray


class BayesianPMF(BaseModel):
    """
    Bayesian Probabilistic Matrix Factorization compatible con BaseModel.

    Requisitos de entrada en fit:
        df con columnas ['user', 'item', 'rating']

    API heredada:
        - fit(df)
        - predict(user, item)

    Métodos extra:
        - recommend(user, top_k=10, exclude_seen=True)
        - get_user_embedding(user)
        - get_item_embedding(item)
    """

    def __init__(
        self,
        n_factors=20,
        n_iters=100,
        burn_in=50,
        thin=2,
        rating_std=1.0,
        clip_range=None,
        random_state=42,
        verbose=True,
        name=None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.n_factors = n_factors
        self.n_iters = n_iters
        self.burn_in = burn_in
        self.thin = thin
        self.rating_std = rating_std
        self.random_state = random_state
        self.verbose = verbose

        self.rng_ = np.random.default_rng(random_state)

        # Mapeos
        self.user_to_idx_ = {}
        self.item_to_idx_ = {}
        self.idx_to_user_ = []
        self.idx_to_item_ = []

        # Parámetros aprendidos
        self.global_mean_ = None
        self.U_ = None
        self.V_ = None

        # Hiperparámetros posteriores medios
        self.mu_u_ = None
        self.mu_v_ = None
        self.Lambda_u_ = None
        self.Lambda_v_ = None

        # Muestras MCMC
        self.samples_ = []

        # Historial
        self.user_seen_items_ = {}

    # =========================================================
    # API OBLIGATORIA
    # =========================================================
    def fit(self, df):
        self._validate_fit_df(df)

        df = df[["user", "item", "rating"]].copy()
        self.global_mean_ = float(df["rating"].mean())

        users = df["user"].unique().tolist()
        items = df["item"].unique().tolist()

        self.user_to_idx_ = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user_ = users
        self.idx_to_item_ = items

        n_users = len(users)
        n_items = len(items)
        K = self.n_factors

        ratings_by_user = [[] for _ in range(n_users)]
        ratings_by_item = [[] for _ in range(n_items)]
        self.user_seen_items_ = {u: set() for u in range(n_users)}

        for row in df.itertuples(index=False):
            user = row.user
            item = row.item
            rating = float(row.rating)

            u_idx = self.user_to_idx_[user]
            i_idx = self.item_to_idx_[item]
            r_centered = rating - self.global_mean_

            ratings_by_user[u_idx].append((i_idx, r_centered))
            ratings_by_item[i_idx].append((u_idx, r_centered))
            self.user_seen_items_[u_idx].add(i_idx)

        # Inicialización
        U = 0.1 * self.rng_.standard_normal((n_users, K))
        V = 0.1 * self.rng_.standard_normal((n_items, K))

        # Priors Normal-Wishart
        mu0_u = np.zeros(K)
        mu0_v = np.zeros(K)
        beta0_u = 2.0
        beta0_v = 2.0
        nu0_u = K
        nu0_v = K
        W0_u = np.eye(K)
        W0_v = np.eye(K)

        alpha = 1.0 / (self.rating_std ** 2)

        self.samples_ = []

        for it in range(self.n_iters):
            # 1) Sampleo hiperparámetros
            mu_u, Lambda_u = self._sample_hyperparams(
                X=U,
                mu0=mu0_u,
                beta0=beta0_u,
                W0=W0_u,
                nu0=nu0_u,
            )
            mu_v, Lambda_v = self._sample_hyperparams(
                X=V,
                mu0=mu0_v,
                beta0=beta0_v,
                W0=W0_v,
                nu0=nu0_v,
            )

            # 2) Sampleo factores usuario
            for u in range(n_users):
                obs = ratings_by_user[u]

                if len(obs) == 0:
                    cov = np.linalg.inv(Lambda_u)
                    U[u] = self.rng_.multivariate_normal(mu_u, self._symmetrize(cov))
                    continue

                item_idx = np.array([i for i, _ in obs], dtype=int)
                r_u = np.array([r for _, r in obs], dtype=float)
                V_u = V[item_idx]

                A = Lambda_u + alpha * (V_u.T @ V_u)
                b = Lambda_u @ mu_u + alpha * (V_u.T @ r_u)

                cov = np.linalg.inv(A)
                mean = cov @ b
                U[u] = self.rng_.multivariate_normal(mean, self._symmetrize(cov))

            # 3) Sampleo factores item
            for i in range(n_items):
                obs = ratings_by_item[i]

                if len(obs) == 0:
                    cov = np.linalg.inv(Lambda_v)
                    V[i] = self.rng_.multivariate_normal(mu_v, self._symmetrize(cov))
                    continue

                user_idx = np.array([u for u, _ in obs], dtype=int)
                r_i = np.array([r for _, r in obs], dtype=float)
                U_i = U[user_idx]

                A = Lambda_v + alpha * (U_i.T @ U_i)
                b = Lambda_v @ mu_v + alpha * (U_i.T @ r_i)

                cov = np.linalg.inv(A)
                mean = cov @ b
                V[i] = self.rng_.multivariate_normal(mean, self._symmetrize(cov))

            # Guardar muestras posteriores
            if it >= self.burn_in and ((it - self.burn_in) % self.thin == 0):
                self.samples_.append(
                    BPMFSample(
                        U=U.copy(),
                        V=V.copy(),
                        mu_u=mu_u.copy(),
                        mu_v=mu_v.copy(),
                        Lambda_u=Lambda_u.copy(),
                        Lambda_v=Lambda_v.copy(),
                    )
                )

            if self.verbose and ((it + 1) % max(1, self.n_iters // 10) == 0 or it == 0):
                rmse_train = self._train_rmse(U, V, ratings_by_user)
                print(f"[{self.name}] iter {it+1}/{self.n_iters} - train_rmse={rmse_train:.5f}")

        # Media posterior
        if len(self.samples_) > 0:
            self.U_ = np.mean([s.U for s in self.samples_], axis=0)
            self.V_ = np.mean([s.V for s in self.samples_], axis=0)
            self.mu_u_ = np.mean([s.mu_u for s in self.samples_], axis=0)
            self.mu_v_ = np.mean([s.mu_v for s in self.samples_], axis=0)
            self.Lambda_u_ = np.mean([s.Lambda_u for s in self.samples_], axis=0)
            self.Lambda_v_ = np.mean([s.Lambda_v for s in self.samples_], axis=0)
        else:
            # fallback si no quedaron muestras
            self.U_ = U
            self.V_ = V
            self.mu_u_ = mu_u
            self.mu_v_ = mu_v
            self.Lambda_u_ = Lambda_u
            self.Lambda_v_ = Lambda_v

        self.is_fitted_ = True
        return self

    def predict(self, user, item):
        """
        Predice rating para (user, item), compatible con BaseModel.
        """
        self._check_fitted()

        user_known = user in self.user_to_idx_
        item_known = item in self.item_to_idx_

        # Cold start
        if not user_known and not item_known:
            pred = self.global_mean_

        elif user_known and item_known:
            u_vec = self.U_[self.user_to_idx_[user]]
            i_vec = self.V_[self.item_to_idx_[item]]
            pred = self.global_mean_ + float(u_vec @ i_vec)

        elif user_known and not item_known:
            u_vec = self.U_[self.user_to_idx_[user]]
            pred = self.global_mean_ + float(u_vec @ self.mu_v_)

        else:  # not user_known and item_known
            i_vec = self.V_[self.item_to_idx_[item]]
            pred = self.global_mean_ + float(self.mu_u_ @ i_vec)

        return float(self._clip(pred))

    # =========================================================
    # MÉTODOS EXTRA
    # =========================================================
    def predict_pair_with_uncertainty(self, user, item):
        """
        Devuelve media y std posterior para un par (user, item).
        """
        self._check_fitted()

        if len(self.samples_) == 0:
            pred = self.predict(user, item)
            return pred, 0.0

        preds = []
        user_known = user in self.user_to_idx_
        item_known = item in self.item_to_idx_

        for s in self.samples_:
            if not user_known and not item_known:
                pred = self.global_mean_

            elif user_known and item_known:
                u_vec = s.U[self.user_to_idx_[user]]
                i_vec = s.V[self.item_to_idx_[item]]
                pred = self.global_mean_ + float(u_vec @ i_vec)

            elif user_known and not item_known:
                u_vec = s.U[self.user_to_idx_[user]]
                pred = self.global_mean_ + float(u_vec @ s.mu_v)

            else:
                i_vec = s.V[self.item_to_idx_[item]]
                pred = self.global_mean_ + float(s.mu_u @ i_vec)

            preds.append(float(self._clip(pred)))

        return float(np.mean(preds)), float(np.std(preds))

    def recommend(self, user, top_k=10, exclude_seen=True):
        self._check_fitted()

        candidates = self.idx_to_item_

        if exclude_seen and user in self.user_to_idx_:
            u_idx = self.user_to_idx_[user]
            seen = self.user_seen_items_.get(u_idx, set())
            candidates = [
                item for item in self.idx_to_item_
                if self.item_to_idx_[item] not in seen
            ]

        scored = [(item, self.predict(user, item)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_user_embedding(self, user):
        self._check_fitted()
        if user in self.user_to_idx_:
            return self.U_[self.user_to_idx_[user]].copy()
        return self.mu_u_.copy()

    def get_item_embedding(self, item):
        self._check_fitted()
        if item in self.item_to_idx_:
            return self.V_[self.item_to_idx_[item]].copy()
        return self.mu_v_.copy()

    # =========================================================
    # UTILIDADES INTERNAS
    # =========================================================
    def _validate_fit_df(self, df):
        required = {"user", "item", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en fit: {missing}")
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _sample_hyperparams(self, X, mu0, beta0, W0, nu0):
        """
        Posterior NIW:
            Lambda ~ Wishart(W_n, nu_n)
            mu | Lambda ~ N(mu_n, (beta_n * Lambda)^-1)
        """
        n, d = X.shape

        x_bar = X.mean(axis=0)
        X_centered = X - x_bar
        S = X_centered.T @ X_centered

        beta_n = beta0 + n
        nu_n = nu0 + n

        diff = (x_bar - mu0).reshape(-1, 1)
        Wn_inv = np.linalg.inv(W0) + S + (beta0 * n / beta_n) * (diff @ diff.T)
        Wn = np.linalg.inv(self._symmetrize(Wn_inv))
        mu_n = (beta0 * mu0 + n * x_bar) / beta_n

        Lambda = self._sample_wishart(Wn, nu_n)
        cov_mu = np.linalg.inv(beta_n * Lambda)
        mu = self.rng_.multivariate_normal(mu_n, self._symmetrize(cov_mu))

        return mu, self._symmetrize(Lambda)

    def _sample_wishart(self, scale, df):
        """
        Bartlett decomposition para Wishart(scale, df).
        """
        p = scale.shape[0]
        if df < p:
            raise ValueError(f"Wishart requiere df >= dim. Recibido df={df}, dim={p}")

        A = np.zeros((p, p))
        for i in range(p):
            A[i, i] = math.sqrt(self.rng_.chisquare(df - i))
            for j in range(i):
                A[i, j] = self.rng_.normal()

        L = np.linalg.cholesky(self._symmetrize(scale))
        LA = L @ A
        W = LA @ LA.T
        return self._symmetrize(W)

    def _train_rmse(self, U, V, ratings_by_user):
        errors = []
        for u, obs in enumerate(ratings_by_user):
            for i, r in obs:
                pred = U[u] @ V[i]
                errors.append((r - pred) ** 2)

        if len(errors) == 0:
            return 0.0
        return float(np.sqrt(np.mean(errors)))

    @staticmethod
    def _symmetrize(M):
        return 0.5 * (M + M.T)
    

@dataclass
class BPMFBiasSample:
    U: np.ndarray
    V: np.ndarray
    b_u: np.ndarray
    b_i: np.ndarray
    mu_u: np.ndarray
    mu_v: np.ndarray
    Lambda_u: np.ndarray
    Lambda_v: np.ndarray


class BayesianPMFWithBiases(BaseModel):
    """
    Bayesian PMF con bias de usuario e ítem.

    Modelo:
        r_ui = global_mean + b_u[u] + b_i[i] + U[u]·V[i] + eps

    con:
        eps ~ N(0, rating_std^2)
        b_u ~ N(0, user_bias_std^2)
        b_i ~ N(0, item_bias_std^2)

    Los factores U y V siguen un esquema BPMF con priors Gaussian-Wishart.
    """

    def __init__(
        self,
        n_factors=20,
        n_iters=100,
        burn_in=50,
        thin=2,
        rating_std=1.0,
        user_bias_std=0.5,
        item_bias_std=0.5,
        clip_range=None,
        random_state=42,
        verbose=True,
        name=None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.n_factors = n_factors
        self.n_iters = n_iters
        self.burn_in = burn_in
        self.thin = thin
        self.rating_std = rating_std
        self.user_bias_std = user_bias_std
        self.item_bias_std = item_bias_std
        self.random_state = random_state
        self.verbose = verbose

        self.rng_ = np.random.default_rng(random_state)

        self.user_to_idx_ = {}
        self.item_to_idx_ = {}
        self.idx_to_user_ = []
        self.idx_to_item_ = []

        self.global_mean_ = None

        self.U_ = None
        self.V_ = None
        self.b_u_ = None
        self.b_i_ = None

        self.mu_u_ = None
        self.mu_v_ = None
        self.Lambda_u_ = None
        self.Lambda_v_ = None

        self.samples_ = []
        self.user_seen_items_ = {}

    # =========================================================
    # API obligatoria
    # =========================================================
    def fit(self, df):
        self._validate_fit_df(df)

        df = df[["user", "item", "rating"]].copy()
        self.global_mean_ = float(df["rating"].mean())

        users = df["user"].unique().tolist()
        items = df["item"].unique().tolist()

        self.user_to_idx_ = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user_ = users
        self.idx_to_item_ = items

        n_users = len(users)
        n_items = len(items)
        K = self.n_factors

        ratings_by_user = [[] for _ in range(n_users)]
        ratings_by_item = [[] for _ in range(n_items)]
        self.user_seen_items_ = {u: set() for u in range(n_users)}

        for row in df.itertuples(index=False):
            user = row.user
            item = row.item
            rating = float(row.rating)

            u_idx = self.user_to_idx_[user]
            i_idx = self.item_to_idx_[item]

            # rating centrado
            r_centered = rating - self.global_mean_

            ratings_by_user[u_idx].append((i_idx, r_centered))
            ratings_by_item[i_idx].append((u_idx, r_centered))
            self.user_seen_items_[u_idx].add(i_idx)

        # Inicialización
        U = 0.1 * self.rng_.standard_normal((n_users, K))
        V = 0.1 * self.rng_.standard_normal((n_items, K))
        b_u = np.zeros(n_users, dtype=float)
        b_i = np.zeros(n_items, dtype=float)

        # Priors BPMF para factores
        mu0_u = np.zeros(K)
        mu0_v = np.zeros(K)
        beta0_u = 2.0
        beta0_v = 2.0
        nu0_u = K
        nu0_v = K
        W0_u = np.eye(K)
        W0_v = np.eye(K)

        sigma2 = self.rating_std ** 2
        alpha = 1.0 / sigma2

        sigma2_bu = self.user_bias_std ** 2
        sigma2_bi = self.item_bias_std ** 2

        self.samples_ = []

        for it in range(self.n_iters):
            # 1) Hiperparámetros de U y V
            mu_u, Lambda_u = self._sample_hyperparams(
                X=U,
                mu0=mu0_u,
                beta0=beta0_u,
                W0=W0_u,
                nu0=nu0_u,
            )
            mu_v, Lambda_v = self._sample_hyperparams(
                X=V,
                mu0=mu0_v,
                beta0=beta0_v,
                W0=W0_v,
                nu0=nu0_v,
            )

            # 2) Sampleo biases de usuario
            for u in range(n_users):
                obs = ratings_by_user[u]

                if len(obs) == 0:
                    post_var = sigma2_bu
                    post_mean = 0.0
                else:
                    residual_sum = 0.0
                    for i, r in obs:
                        residual_sum += (r - b_i[i] - np.dot(U[u], V[i]))

                    post_var = 1.0 / (len(obs) / sigma2 + 1.0 / sigma2_bu)
                    post_mean = post_var * (residual_sum / sigma2)

                b_u[u] = self.rng_.normal(post_mean, np.sqrt(post_var))

            # 3) Sampleo biases de ítem
            for i in range(n_items):
                obs = ratings_by_item[i]

                if len(obs) == 0:
                    post_var = sigma2_bi
                    post_mean = 0.0
                else:
                    residual_sum = 0.0
                    for u, r in obs:
                        residual_sum += (r - b_u[u] - np.dot(U[u], V[i]))

                    post_var = 1.0 / (len(obs) / sigma2 + 1.0 / sigma2_bi)
                    post_mean = post_var * (residual_sum / sigma2)

                b_i[i] = self.rng_.normal(post_mean, np.sqrt(post_var))

            # 4) Sampleo factores usuario
            for u in range(n_users):
                obs = ratings_by_user[u]

                if len(obs) == 0:
                    cov = np.linalg.inv(Lambda_u)
                    U[u] = self.rng_.multivariate_normal(mu_u, self._symmetrize(cov))
                    continue

                item_idx = np.array([i for i, _ in obs], dtype=int)
                y_u = np.array(
                    [r - b_u[u] - b_i[i] for i, r in obs],
                    dtype=float,
                )
                V_u = V[item_idx]

                A = Lambda_u + alpha * (V_u.T @ V_u)
                b = Lambda_u @ mu_u + alpha * (V_u.T @ y_u)

                cov = np.linalg.inv(A)
                mean = cov @ b
                U[u] = self.rng_.multivariate_normal(mean, self._symmetrize(cov))

            # 5) Sampleo factores ítem
            for i in range(n_items):
                obs = ratings_by_item[i]

                if len(obs) == 0:
                    cov = np.linalg.inv(Lambda_v)
                    V[i] = self.rng_.multivariate_normal(mu_v, self._symmetrize(cov))
                    continue

                user_idx = np.array([u for u, _ in obs], dtype=int)
                y_i = np.array(
                    [r - b_u[u] - b_i[i] for u, r in obs],
                    dtype=float,
                )
                U_i = U[user_idx]

                A = Lambda_v + alpha * (U_i.T @ U_i)
                b = Lambda_v @ mu_v + alpha * (U_i.T @ y_i)

                cov = np.linalg.inv(A)
                mean = cov @ b
                V[i] = self.rng_.multivariate_normal(mean, self._symmetrize(cov))

            # Guardar muestras posteriores
            if it >= self.burn_in and ((it - self.burn_in) % self.thin == 0):
                self.samples_.append(
                    BPMFBiasSample(
                        U=U.copy(),
                        V=V.copy(),
                        b_u=b_u.copy(),
                        b_i=b_i.copy(),
                        mu_u=mu_u.copy(),
                        mu_v=mu_v.copy(),
                        Lambda_u=Lambda_u.copy(),
                        Lambda_v=Lambda_v.copy(),
                    )
                )

            if self.verbose and ((it + 1) % max(1, self.n_iters // 10) == 0 or it == 0):
                rmse_train = self._train_rmse(U, V, b_u, b_i, ratings_by_user)
                print(f"[{self.name}] iter {it+1}/{self.n_iters} - train_rmse={rmse_train:.5f}")

        # Media posterior
        if len(self.samples_) > 0:
            self.U_ = np.mean([s.U for s in self.samples_], axis=0)
            self.V_ = np.mean([s.V for s in self.samples_], axis=0)
            self.b_u_ = np.mean([s.b_u for s in self.samples_], axis=0)
            self.b_i_ = np.mean([s.b_i for s in self.samples_], axis=0)
            self.mu_u_ = np.mean([s.mu_u for s in self.samples_], axis=0)
            self.mu_v_ = np.mean([s.mu_v for s in self.samples_], axis=0)
            self.Lambda_u_ = np.mean([s.Lambda_u for s in self.samples_], axis=0)
            self.Lambda_v_ = np.mean([s.Lambda_v for s in self.samples_], axis=0)
        else:
            self.U_ = U
            self.V_ = V
            self.b_u_ = b_u
            self.b_i_ = b_i
            self.mu_u_ = mu_u
            self.mu_v_ = mu_v
            self.Lambda_u_ = Lambda_u
            self.Lambda_v_ = Lambda_v

        self.is_fitted_ = True
        return self

    def predict(self, user, item):
        self._check_fitted()

        user_known = user in self.user_to_idx_
        item_known = item in self.item_to_idx_

        mu = self.global_mean_

        # versión simple y estable
        if user_known and item_known:
            u_idx = self.user_to_idx_[user]
            i_idx = self.item_to_idx_[item]
            pred = mu + self.b_u_[u_idx] + self.b_i_[i_idx] + float(self.U_[u_idx] @ self.V_[i_idx])

        elif user_known and not item_known:
            u_idx = self.user_to_idx_[user]
            pred = mu + self.b_u_[u_idx]

        elif not user_known and item_known:
            i_idx = self.item_to_idx_[item]
            pred = mu + self.b_i_[i_idx]

        else:
            pred = mu

        return float(self._clip(pred))

    # =========================================================
    # Métodos extra
    # =========================================================
    def recommend(self, user, top_k=10, exclude_seen=True):
        self._check_fitted()

        candidates = self.idx_to_item_

        if exclude_seen and user in self.user_to_idx_:
            u_idx = self.user_to_idx_[user]
            seen = self.user_seen_items_.get(u_idx, set())
            candidates = [
                item for item in self.idx_to_item_
                if self.item_to_idx_[item] not in seen
            ]

        scored = [(item, self.predict(user, item)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_user_embedding(self, user):
        self._check_fitted()
        if user in self.user_to_idx_:
            return self.U_[self.user_to_idx_[user]].copy()
        return self.mu_u_.copy()

    def get_item_embedding(self, item):
        self._check_fitted()
        if item in self.item_to_idx_:
            return self.V_[self.item_to_idx_[item]].copy()
        return self.mu_v_.copy()

    # =========================================================
    # Utilidades internas
    # =========================================================
    def _validate_fit_df(self, df):
        required = {"user", "item", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en fit: {missing}")
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _sample_hyperparams(self, X, mu0, beta0, W0, nu0):
        n, d = X.shape

        x_bar = X.mean(axis=0)
        X_centered = X - x_bar
        S = X_centered.T @ X_centered

        beta_n = beta0 + n
        nu_n = nu0 + n

        diff = (x_bar - mu0).reshape(-1, 1)
        Wn_inv = np.linalg.inv(W0) + S + (beta0 * n / beta_n) * (diff @ diff.T)
        Wn = np.linalg.inv(self._symmetrize(Wn_inv))
        mu_n = (beta0 * mu0 + n * x_bar) / beta_n

        Lambda = self._sample_wishart(Wn, nu_n)
        cov_mu = np.linalg.inv(beta_n * Lambda)
        mu = self.rng_.multivariate_normal(mu_n, self._symmetrize(cov_mu))

        return mu, self._symmetrize(Lambda)

    def _sample_wishart(self, scale, df):
        p = scale.shape[0]
        if df < p:
            raise ValueError(f"Wishart requiere df >= dim. Recibido df={df}, dim={p}")

        A = np.zeros((p, p))
        for i in range(p):
            A[i, i] = math.sqrt(self.rng_.chisquare(df - i))
            for j in range(i):
                A[i, j] = self.rng_.normal()

        L = np.linalg.cholesky(self._symmetrize(scale))
        LA = L @ A
        W = LA @ LA.T
        return self._symmetrize(W)

    def _train_rmse(self, U, V, b_u, b_i, ratings_by_user):
        errors = []
        for u, obs in enumerate(ratings_by_user):
            for i, r in obs:
                pred = b_u[u] + b_i[i] + float(U[u] @ V[i])
                errors.append((r - pred) ** 2)

        if len(errors) == 0:
            return 0.0
        return float(np.sqrt(np.mean(errors)))

    @staticmethod
    def _symmetrize(M):
        return 0.5 * (M + M.T)



@dataclass
class BNMFState:
    gamma: np.ndarray      # (n_users, K)
    ep: np.ndarray         # (n_items, K)
    em: np.ndarray         # (n_items, K)
    a: np.ndarray          # (n_users, K)
    b: np.ndarray          # (n_items, K)


class BayesianNonNegativeMF(BaseModel):
    """
    Bayesian Non-negative Matrix Factorization (BNMF) con inferencia variacional.

    Interpretación:
        - a[u, k]: probabilidad / peso de pertenencia del usuario u al cluster k
        - b[i, k]: afinidad/probabilidad del item i respecto al cluster k

    Predicción normalizada:
        r_hat_norm(u, i) = sum_k a[u, k] * b[i, k]

    Predicción en escala original:
        r_hat(u, i) = min_rating + r_hat_norm(u, i) * (max_rating - min_rating)

    Requisitos de entrada en fit:
        df con columnas ['user', 'item', 'rating']
    """

    def __init__(
        self,
        n_factors: int = 20,
        n_iters: int = 50,
        alpha: float = 0.3,
        beta: float = 0.3,
        R: float = 10.0,
        init_scale: float = 0.1,
        eps: float = 1e-10,
        clip_range=None,
        random_state: int = 42,
        verbose: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        self.n_factors = n_factors
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.R = R
        self.init_scale = init_scale
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose

        self.rng_ = np.random.default_rng(random_state)

        # mapeos
        self.user_to_idx_ = {}
        self.item_to_idx_ = {}
        self.idx_to_user_ = []
        self.idx_to_item_ = []

        # límites de rating
        self.min_rating_ = None
        self.max_rating_ = None
        self.global_mean_ = None
        self.global_mean_norm_ = None

        # parámetros aprendidos
        self.gamma_ = None
        self.ep_ = None
        self.em_ = None
        self.a_ = None
        self.b_ = None

        # fallback para cold start
        self.a_prior_ = None
        self.b_prior_ = None

        # historial de vistos
        self.user_seen_items_ = {}

    # =========================================================
    # API obligatoria
    # =========================================================
    def fit(self, df: pd.DataFrame):
        self._validate_fit_df(df)

        df = df[["user", "item", "rating"]].copy()

        self.min_rating_ = float(df["rating"].min())
        self.max_rating_ = float(df["rating"].max())
        self.global_mean_ = float(df["rating"].mean())

        if self.max_rating_ <= self.min_rating_:
            raise ValueError("max_rating debe ser mayor que min_rating.")

        self.global_mean_norm_ = self._normalize_rating(self.global_mean_)

        users = df["user"].unique().tolist()
        items = df["item"].unique().tolist()

        self.user_to_idx_ = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user_ = users
        self.idx_to_item_ = items

        n_users = len(users)
        n_items = len(items)
        K = self.n_factors

        # observaciones indexadas
        ratings_by_user = [[] for _ in range(n_users)]
        ratings_by_item = [[] for _ in range(n_items)]
        observations = []

        self.user_seen_items_ = {u: set() for u in range(n_users)}

        for row in df.itertuples(index=False):
            user = row.user
            item = row.item
            rating = float(row.rating)

            u_idx = self.user_to_idx_[user]
            i_idx = self.item_to_idx_[item]

            r_norm = self._normalize_rating(rating)
            rp = self.R * r_norm
            rm = self.R * (1.0 - r_norm)

            observations.append((u_idx, i_idx, r_norm, rp, rm))
            ratings_by_user[u_idx].append((i_idx, r_norm, rp, rm))
            ratings_by_item[i_idx].append((u_idx, r_norm, rp, rm))
            self.user_seen_items_[u_idx].add(i_idx)

        # Inicialización variacional
        gamma = self.alpha + self.init_scale * self.rng_.random((n_users, K))
        ep = self.beta + self.init_scale * self.rng_.random((n_items, K))
        em = self.beta + self.init_scale * self.rng_.random((n_items, K))

        # lambda para cada observación
        # lo almacenamos como lista paralela a observations
        lambdas = np.full((len(observations), K), 1.0 / K, dtype=float)

        for it in range(self.n_iters):
            # =====================================================
            # 1) E-step: actualizar lambda
            # =====================================================
            for idx, (u, i, _, rp, rm) in enumerate(observations):
                log_lambda = (
                    digamma(gamma[u] + self.eps)
                    + rp * digamma(ep[i] + self.eps)
                    + rm * digamma(em[i] + self.eps)
                    - self.R * digamma(ep[i] + em[i] + self.eps)
                )

                # softmax estable
                log_lambda = log_lambda - np.max(log_lambda)
                lam = np.exp(log_lambda)
                lam_sum = lam.sum()

                if lam_sum <= 0:
                    lam = np.full(K, 1.0 / K, dtype=float)
                else:
                    lam /= lam_sum

                lambdas[idx] = lam

            # =====================================================
            # 2) M-step: reiniciar y acumular gamma, ep, em
            # =====================================================
            gamma = np.full((n_users, K), self.alpha, dtype=float)
            ep = np.full((n_items, K), self.beta, dtype=float)
            em = np.full((n_items, K), self.beta, dtype=float)

            for idx, (u, i, _, rp, rm) in enumerate(observations):
                lam = lambdas[idx]
                gamma[u] += lam
                ep[i] += lam * rp
                em[i] += lam * rm

            # =====================================================
            # 3) Parámetros interpretables a y b
            # =====================================================
            a = gamma / np.clip(gamma.sum(axis=1, keepdims=True), self.eps, None)
            b = ep / np.clip(ep + em, self.eps, None)

            if self.verbose and ((it + 1) % max(1, self.n_iters // 10) == 0 or it == 0):
                mae = self._train_mae(observations, a, b)
                print(f"[{self.name}] iter {it+1}/{self.n_iters} - train_mae={mae:.5f}")

        self.gamma_ = gamma
        self.ep_ = ep
        self.em_ = em
        self.a_ = a
        self.b_ = b

        # fallbacks para cold start
        # usuario nuevo -> prior uniforme en componentes
        self.a_prior_ = np.full(self.n_factors, 1.0 / self.n_factors, dtype=float)

        # item nuevo -> usar media por componente aprendida
        self.b_prior_ = np.mean(self.b_, axis=0)

        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        self._check_fitted()

        a_u = self.get_user_embedding(user)
        b_i = self.get_item_embedding(item)

        pred_norm = float(np.dot(a_u, b_i))
        pred = self._denormalize_rating(pred_norm)

        return float(self._clip(pred))

    # =========================================================
    # Métodos extra
    # =========================================================
    def recommend(self, user, top_k: int = 10, exclude_seen: bool = True):
        self._check_fitted()

        candidates = self.idx_to_item_

        if exclude_seen and user in self.user_to_idx_:
            u_idx = self.user_to_idx_[user]
            seen = self.user_seen_items_.get(u_idx, set())
            candidates = [
                item for item in self.idx_to_item_
                if self.item_to_idx_[item] not in seen
            ]

        scored = [(item, self.predict(user, item)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_user_embedding(self, user) -> np.ndarray:
        """
        Devuelve a_u: distribución del usuario sobre factores.
        """
        self._check_fitted()

        if user in self.user_to_idx_:
            return self.a_[self.user_to_idx_[user]].copy()
        return self.a_prior_.copy()

    def get_item_embedding(self, item) -> np.ndarray:
        """
        Devuelve b_i: afinidad/probabilidad del item por factor.
        """
        self._check_fitted()

        if item in self.item_to_idx_:
            return self.b_[self.item_to_idx_[item]].copy()
        return self.b_prior_.copy()

    def predict_normalized(self, user, item) -> float:
        """
        Predicción en [0, 1].
        """
        self._check_fitted()
        a_u = self.get_user_embedding(user)
        b_i = self.get_item_embedding(item)
        return float(np.clip(np.dot(a_u, b_i), 0.0, 1.0))

    def explain_prediction(self, user, item, top_k_components: int = 5):
        """
        Descompone la predicción por componentes:
            contrib_k = a_u[k] * b_i[k]
        """
        self._check_fitted()

        a_u = self.get_user_embedding(user)
        b_i = self.get_item_embedding(item)
        contrib = a_u * b_i

        order = np.argsort(contrib)[::-1][:top_k_components]
        return [
            {
                "factor": int(k),
                "user_weight": float(a_u[k]),
                "item_affinity": float(b_i[k]),
                "contribution": float(contrib[k]),
            }
            for k in order
        ]

    # =========================================================
    # Utilidades internas
    # =========================================================
    def _validate_fit_df(self, df: pd.DataFrame):
        required = {"user", "item", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en fit: {missing}")
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _normalize_rating(self, rating: float) -> float:
        x = (rating - self.min_rating_) / (self.max_rating_ - self.min_rating_)
        return float(np.clip(x, 0.0, 1.0))

    def _denormalize_rating(self, rating_norm: float) -> float:
        x = np.clip(rating_norm, 0.0, 1.0)
        return float(self.min_rating_ + x * (self.max_rating_ - self.min_rating_))

    def _train_mae(self, observations, a, b) -> float:
        errors = []

        for u, i, r_norm, _, _ in observations:
            pred = float(np.dot(a[u], b[i]))
            errors.append(abs(r_norm - pred))

        if not errors:
            return 0.0
        return float(np.mean(errors))