from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma

try:
    from .base import BaseModel
except Exception:
    class BaseModel:
        def __init__(self, name=None, clip_range=None):
            self.name = name if name is not None else self.__class__.__name__
            self.clip_range = clip_range
            self.is_fitted_ = False

        def _clip(self, x):
            if self.clip_range is None:
                return x
            lo, hi = self.clip_range
            return float(np.clip(x, lo, hi))

        def _check_fitted(self):
            if not getattr(self, "is_fitted_", False):
                raise RuntimeError("El modelo no está entrenado.")


@dataclass
class BNMFState:
    gamma: np.ndarray
    ep: np.ndarray
    em: np.ndarray
    a: np.ndarray
    b: np.ndarray


class BayesianNonNegativeMF(BaseModel):
    """
    Implementación del modelo del paper de Hernando et al. adaptado a ratings discretos 0..10.

    Idea del paper:
      - a[u, k] = P(z_ui = k | u) ~ Dirichlet posterior media
      - b[i, k] = P(usuario del grupo k gusta del item i) ~ Beta posterior media
      - p_ui = sum_k a[u, k] * b[i, k]
      - la predicción final es DISCRETA y se obtiene proyectando p_ui a la rejilla
        de ratings normalizados {0/R, 1/R, ..., R/R}.

    Adaptación al caso 0..10:
      - Los ratings deben ser discretos enteros en {0, 1, ..., 10}.
      - Usamos R=10 por defecto.
      - r_norm = rating / 10 pertenece a {0.0, 0.1, ..., 1.0}.
      - La predicción devuelta por predict() también es entera en 0..10.

    Importante:
      - Esta versión es mucho más fiel al paper que una interpolación lineal continua.
      - Si tus ratings NO son enteros en 0..10, este modelo no encaja bien tal cual.
    """

    def __init__(
        self,
        n_factors: int = 6,
        n_iters: int = 100,
        alpha: float = 0.3,
        beta: float = 5.0,
        R: int = 10,
        init_noise: float = 1e-3,
        eps: float = 1e-12,
        clip_range=(0.0, 10.0),
        random_state: int = 42,
        verbose: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        if R <= 0 or int(R) != R:
            raise ValueError("R debe ser un entero positivo.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("Según el paper, alpha debe estar en (0, 1).")
        if beta <= 0.0:
            raise ValueError("beta debe ser > 0.")
        if n_factors <= 0:
            raise ValueError("n_factors debe ser > 0.")
        if n_iters <= 0:
            raise ValueError("n_iters debe ser > 0.")

        self.n_factors = int(n_factors)
        self.n_iters = int(n_iters)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.R = int(R)
        self.init_noise = float(init_noise)
        self.eps = float(eps)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.rng_ = np.random.default_rng(random_state)

        self.user_to_idx_ = {}
        self.item_to_idx_ = {}
        self.idx_to_user_ = []
        self.idx_to_item_ = []

        self.gamma_ = None
        self.ep_ = None
        self.em_ = None
        self.a_ = None
        self.b_ = None
        self.state_history_ = []

        self.user_seen_items_ = {}
        self.observations_ = None

        self.rating_values_ = np.arange(self.R + 1, dtype=int)
        self.rating_grid_norm_ = self.rating_values_ / self.R

        # Cold start fiel al prior / posterior medio, no a heurísticas de interpolación continua.
        self.a_prior_ = np.full(self.n_factors, 1.0 / self.n_factors, dtype=float)
        self.b_prior_ = None
        self.global_prob_like_ = None

    # =========================================================
    # API obligatoria
    # =========================================================
    def fit(self, df: pd.DataFrame):
        self._validate_fit_df(df)
        df = df[["user", "item", "rating"]].copy()

        # En esta adaptación al paper exigimos ratings discretos 0..R.
        bad_mask = (~np.isfinite(df["rating"].to_numpy(dtype=float)))
        if bad_mask.any():
            raise ValueError("Hay ratings no finitos.")

        bad_range = ~df["rating"].isin(self.rating_values_)
        if bad_range.any():
            bad_vals = sorted(df.loc[bad_range, "rating"].unique().tolist())[:10]
            raise ValueError(
                f"Este modelo fiel al paper requiere ratings enteros en 0..{self.R}. "
                f"Valores problemáticos: {bad_vals}"
            )

        users = df["user"].unique().tolist()
        items = df["item"].unique().tolist()
        self.user_to_idx_ = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx_ = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user_ = users
        self.idx_to_item_ = items

        n_users = len(users)
        n_items = len(items)
        K = self.n_factors

        self.user_seen_items_ = {u: set() for u in range(n_users)}
        observations: List[Tuple[int, int, int]] = []

        for row in df.itertuples(index=False):
            u = self.user_to_idx_[row.user]
            i = self.item_to_idx_[row.item]
            r = int(row.rating)
            observations.append((u, i, r))
            self.user_seen_items_[u].add(i)

        self.observations_ = observations

        # Inicialización variacional.
        gamma = self.alpha + self.init_noise * self.rng_.random((n_users, K))
        ep = self.beta + self.init_noise * self.rng_.random((n_items, K))
        em = self.beta + self.init_noise * self.rng_.random((n_items, K))
        lambdas = np.full((len(observations), K), 1.0 / K, dtype=float)

        self.state_history_ = []

        for it in range(self.n_iters):
            # E-step: q(z_ui)
            for obs_idx, (u, i, r) in enumerate(observations):
                r_minus = self.R - r
                log_lambda = (
                    digamma(gamma[u] + self.eps)
                    + r * digamma(ep[i] + self.eps)
                    + r_minus * digamma(em[i] + self.eps)
                    - self.R * digamma(ep[i] + em[i] + self.eps)
                )
                log_lambda -= np.max(log_lambda)
                lam = np.exp(log_lambda)
                lam_sum = lam.sum()
                if lam_sum <= 0.0 or not np.isfinite(lam_sum):
                    lam[:] = 1.0 / K
                else:
                    lam /= lam_sum
                lambdas[obs_idx] = lam

            # M-step: actualizar parámetros variacionales
            gamma.fill(self.alpha)
            ep.fill(self.beta)
            em.fill(self.beta)

            for obs_idx, (u, i, r) in enumerate(observations):
                lam = lambdas[obs_idx]
                gamma[u] += lam
                ep[i] += lam * r
                em[i] += lam * (self.R - r)

            a = gamma / np.clip(gamma.sum(axis=1, keepdims=True), self.eps, None)
            b = ep / np.clip(ep + em, self.eps, None)

            if self.verbose and ((it + 1) % max(1, self.n_iters // 10) == 0 or it == 0):
                mae = self._train_mae_discrete(observations, a, b)
                mae_soft = self._train_mae_soft(observations, a, b)
                print(
                    f"[{self.name}] iter {it+1}/{self.n_iters} "
                    f"- train_mae_discrete={mae:.5f} "
                    f"- train_mae_soft={mae_soft:.5f}"
                )

        self.gamma_ = gamma.copy()
        self.ep_ = ep.copy()
        self.em_ = em.copy()
        self.a_ = a.copy()
        self.b_ = b.copy()

        # Posterior medio global de “gustar” por factor para item nuevo.
        self.b_prior_ = np.mean(self.b_, axis=0)
        self.global_prob_like_ = float(np.mean(self.b_))

        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        """
        Devuelve un rating discreto entero en 0..R, siguiendo el espíritu del paper.
        """
        self._check_fitted()
        p_ui = self.predict_normalized(user, item)
        r_hat = self._project_probability_to_discrete_rating(p_ui)
        return float(self._clip(r_hat))

    # =========================================================
    # Métodos extra
    # =========================================================
    def predict_normalized(self, user, item) -> float:
        """
        Devuelve p_ui = sum_k a_uk b_ik en [0,1].
        Esta es la cantidad probabilística central del paper.
        """
        self._check_fitted()
        a_u = self.get_user_embedding(user)
        b_i = self.get_item_embedding(item)
        return float(np.clip(np.dot(a_u, b_i), 0.0, 1.0))

    def predict_expected_rating(self, user, item) -> float:
        """
        Valor esperado continuo R * p_ui.
        No es la predicción discreta final del paper, pero sirve para diagnóstico.
        """
        p_ui = self.predict_normalized(user, item)
        return float(self.R * p_ui)

    def predict_proba_each_rating(self, user, item) -> np.ndarray:
        """
        Distribución Binomial(r | R, p_ui) sobre ratings 0..R.
        Esto es coherente con la historia generativa del paper.
        """
        self._check_fitted()
        p = self.predict_normalized(user, item)
        vals = np.array([
            self._binom_pmf(r, self.R, p) for r in self.rating_values_
        ], dtype=float)
        vals /= np.clip(vals.sum(), self.eps, None)
        return vals

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
        self._check_fitted()
        if user in self.user_to_idx_:
            return self.a_[self.user_to_idx_[user]].copy()
        return self.a_prior_.copy()

    def get_item_embedding(self, item) -> np.ndarray:
        self._check_fitted()
        if item in self.item_to_idx_:
            return self.b_[self.item_to_idx_[item]].copy()
        return self.b_prior_.copy()

    def explain_prediction(self, user, item, top_k_components: int = 5):
        self._check_fitted()
        a_u = self.get_user_embedding(user)
        b_i = self.get_item_embedding(item)
        contrib = a_u * b_i
        order = np.argsort(contrib)[::-1][:top_k_components]
        return [
            {
                "factor": int(k),
                "user_prob_group": float(a_u[k]),
                "item_like_prob_given_group": float(b_i[k]),
                "contribution_to_p_ui": float(contrib[k]),
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

    def _project_probability_to_discrete_rating(self, p_ui: float) -> int:
        """
        Proyección discreta natural a la rejilla {0,1,...,R}.
        Equivale a elegir el rating cuyo valor normalizado r/R está más cerca de p_ui.
        """
        idx = int(np.argmin(np.abs(self.rating_grid_norm_ - p_ui)))
        return int(self.rating_values_[idx])

    def _binom_pmf(self, r: int, n: int, p: float) -> float:
        from math import comb
        return float(comb(n, r) * (p ** r) * ((1.0 - p) ** (n - r)))

    def _train_mae_discrete(self, observations, a, b) -> float:
        errs = []
        for u, i, r in observations:
            p = float(np.clip(np.dot(a[u], b[i]), 0.0, 1.0))
            pred = self._project_probability_to_discrete_rating(p)
            errs.append(abs(r - pred))
        return float(np.mean(errs)) if errs else 0.0

    def _train_mae_soft(self, observations, a, b) -> float:
        errs = []
        for u, i, r in observations:
            pred = self.R * float(np.clip(np.dot(a[u], b[i]), 0.0, 1.0))
            errs.append(abs(r - pred))
        return float(np.mean(errs)) if errs else 0.0
