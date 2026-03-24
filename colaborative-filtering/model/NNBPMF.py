from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln

from .base import BaseModel


@dataclass
class BNMFState:
    gamma: np.ndarray   # (n_users, K)
    ep: np.ndarray      # (n_items, K)
    em: np.ndarray      # (n_items, K)
    a: np.ndarray       # E[phi_u] aprox
    b: np.ndarray       # E[kappa_ik] aprox


class BayesianNonNegativeMF(BaseModel):
    """
    Implementación al modelo variacional bayesiano del paper de Hernando:

        phi_u      ~ Dir(alpha, ..., alpha)
        kappa_i,k  ~ Beta(beta, beta)
        z_ui       ~ Categorical(phi_u)
        rho_ui     ~ Binomial(R, kappa_i,z_ui)
        r_ui*      = rho_ui / R
    """

    def __init__(
        self,
        n_factors: int = 6,
        n_iters: int = 100,
        alpha: float = 0.3,
        beta: float = 1.0,
        R: int = 10,
        init_noise: float = 0.05,
        eps: float = 1e-12,
        clip_range=(0.0, 10.0),
        random_state: int = 42,
        verbose: bool = True,
        store_history: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, clip_range=clip_range)

        if n_factors <= 0:
            raise ValueError("n_factors debe ser > 0.")
        if n_iters <= 0:
            raise ValueError("n_iters debe ser > 0.")
        if R <= 0 or int(R) != R:
            raise ValueError("R debe ser un entero positivo.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("Según el paper, alpha debe estar en (0, 1).")
        if beta <= 0.0:
            raise ValueError("beta debe ser > 0.")
        if init_noise < 0.0:
            raise ValueError("init_noise debe ser >= 0.")
        if eps <= 0.0:
            raise ValueError("eps debe ser > 0.")

        self.n_factors = int(n_factors)
        self.n_iters = int(n_iters)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.R = int(R)
        self.init_noise = float(init_noise)
        self.eps = float(eps)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.store_history = bool(store_history)

        self.rng_ = np.random.default_rng(self.random_state)

        self.user_to_idx_: Dict[Any, int] = {}
        self.item_to_idx_: Dict[Any, int] = {}
        self.idx_to_user_: List[Any] = []
        self.idx_to_item_: List[Any] = []

        self.gamma_: Optional[np.ndarray] = None
        self.ep_: Optional[np.ndarray] = None
        self.em_: Optional[np.ndarray] = None
        self.a_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None

        self.state_history_: List[BNMFState] = []
        self.training_trace_: List[Dict[str, float]] = []

        self.user_seen_items_: Dict[int, set] = {}
        self.observations_: Optional[List[Tuple[int, int, int]]] = None

        self.rating_values_ = np.arange(self.R + 1, dtype=int)
        self.rating_grid_norm_ = self.rating_values_ / self.R

        # Priors / fallback cold start
        self.a_prior_ = np.full(self.n_factors, 1.0 / self.n_factors, dtype=float)
        self.b_prior_ = None
        self.global_prob_like_ = None

    # =========================================================
    # API principal
    # =========================================================
    def fit(self, df: pd.DataFrame):
        self._validate_fit_df(df)

        df = df[["user", "item", "rating"]].copy()
        df["rating"] = self._validate_and_cast_ratings(df["rating"])

        # Agregar duplicados user-item si los hay.
        df = (
            df.groupby(["user", "item"], as_index=False)["rating"]
            .mean()
        )
        df["rating"] = np.rint(df["rating"]).astype(int)

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

        # Inicialización menos simétrica que tu versión.
        gamma = self.alpha + self.init_noise * self.rng_.gamma(shape=1.0, scale=1.0, size=(n_users, K))
        ep = self.beta + self.init_noise * self.rng_.gamma(shape=1.0, scale=1.0, size=(n_items, K))
        em = self.beta + self.init_noise * self.rng_.gamma(shape=1.0, scale=1.0, size=(n_items, K))

        # q(z_ui)
        lambdas = np.full((len(observations), K), 1.0 / K, dtype=float)

        self.state_history_ = []
        self.training_trace_ = []

        for it in range(self.n_iters):
            # -------------------------
            # E-step: actualizar q(z_ui)
            # -------------------------
            dig_gamma = digamma(np.clip(gamma, self.eps, None))
            dig_ep = digamma(np.clip(ep, self.eps, None))
            dig_em = digamma(np.clip(em, self.eps, None))
            dig_epem = digamma(np.clip(ep + em, self.eps, None))

            for obs_idx, (u, i, r) in enumerate(observations):
                r_minus = self.R - r

                # E[log phi_uk] + r E[log kappa_ik] + (R-r) E[log(1-kappa_ik)]
                log_lambda = (
                    dig_gamma[u]
                    + r * (dig_ep[i] - dig_epem[i])
                    + r_minus * (dig_em[i] - dig_epem[i])
                )

                log_lambda -= np.max(log_lambda)
                lam = np.exp(log_lambda)
                s = lam.sum()

                if not np.isfinite(s) or s <= 0.0:
                    lam[:] = 1.0 / K
                else:
                    lam /= s

                lambdas[obs_idx] = lam

            # -------------------------
            # M-step variacional:
            # actualizar gamma, ep, em
            # -------------------------
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

            trace = self._compute_diagnostics(observations, a, b, lambdas)
            self.training_trace_.append(trace)

            if self.store_history:
                self.state_history_.append(
                    BNMFState(
                        gamma=gamma.copy(),
                        ep=ep.copy(),
                        em=em.copy(),
                        a=a.copy(),
                        b=b.copy(),
                    )
                )

            if self.verbose and ((it == 0) or ((it + 1) % max(1, self.n_iters // 10) == 0)):
                print(
                    f"[{self.name}] iter {it+1}/{self.n_iters} "
                    f"- train_mae_map={trace['train_mae_map']:.5f} "
                    f"- train_mae_exp={trace['train_mae_exp']:.5f} "
                    f"- user_entropy={trace['user_entropy']:.5f} "
                    f"- peak_resp={trace['peak_resp']:.5f} "
                    f"- item_factor_std={trace['item_factor_std']:.5f}"
                )

        self.gamma_ = gamma.copy()
        self.ep_ = ep.copy()
        self.em_ = em.copy()
        self.a_ = a.copy()
        self.b_ = b.copy()

        # Cold start:
        # - usuario nuevo: prior uniforme sobre grupos
        # - item nuevo: media posterior por factor
        self.b_prior_ = np.mean(self.b_, axis=0)
        self.global_prob_like_ = float(np.mean(self.b_))

        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        """
        Predicción discreta coherente con el modelo Binomial:
        elegimos el rating MAP sobre {0, ..., R}.
        """
        self._check_fitted()
        probs = self.predict_proba_each_rating(user, item)
        r_hat = int(self.rating_values_[np.argmax(probs)])
        return float(self._clip(r_hat))

    # =========================================================
    # Métodos auxiliares / extendidos
    # =========================================================
    def predict_expected_rating(self, user, item) -> float:
        """
        Esperanza del rating bajo el modelo: E[r_ui] = R * p_ui.
        """
        self._check_fitted()
        return float(self._clip(self.R * self.predict_normalized(user, item)))

    def predict_normalized(self, user, item) -> float:
        """
        p_ui = sum_k a_uk * b_ik, en [0,1].
        """
        self._check_fitted()
        a_u = self.get_user_embedding(user)
        b_i = self.get_item_embedding(item)
        return float(np.clip(np.dot(a_u, b_i), 0.0, 1.0))

    def predict_proba_each_rating(self, user, item) -> np.ndarray:
        """
        Distribución Binomial(r | R, p_ui), r=0..R.
        """
        self._check_fitted()
        p = self.predict_normalized(user, item)
        vals = np.array([self._binom_pmf(r, self.R, p) for r in self.rating_values_], dtype=float)
        vals /= np.clip(vals.sum(), self.eps, None)
        return vals

    def recommend(self, user, top_k: int = 10, exclude_seen: bool = True):
        self._check_fitted()

        if top_k <= 0:
            return []

        candidates = self.idx_to_item_

        if exclude_seen and user in self.user_to_idx_:
            u_idx = self.user_to_idx_[user]
            seen = self.user_seen_items_.get(u_idx, set())
            candidates = [
                item for item in self.idx_to_item_
                if self.item_to_idx_[item] not in seen
            ]

        scored = [(item, self.predict_expected_rating(user, item)) for item in candidates]
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

    def get_training_trace(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(self.training_trace_)

    def get_user_idx(self, user):
        self._check_fitted()
        if user not in self.user_to_idx_:
            return None
        return self.user_to_idx_[user]

    def get_item_idx(self, item):
        self._check_fitted()
        if item not in self.item_to_idx_:
            return None
        return self.item_to_idx_[item]

    def get_user_posterior_summary(self, user):
        self._check_fitted()
        u = self.get_user_idx(user)
        if u is None:
            return {
                "seen_in_train": False,
                "a": self.a_prior_.copy(),
                "gamma": None,
                "entropy": float(-np.sum(self.a_prior_ * np.log(np.clip(self.a_prior_, 1e-12, None)))),
                "dominant_factor": int(np.argmax(self.a_prior_)),
            }

        a = self.a_[u].copy()
        gamma = self.gamma_[u].copy()
        entropy = float(-np.sum(a * np.log(np.clip(a, 1e-12, None))))

        return {
            "seen_in_train": True,
            "a": a,
            "gamma": gamma,
            "entropy": entropy,
            "dominant_factor": int(np.argmax(a)),
            "gamma_mass": float(np.sum(gamma)),
        }

    def get_item_posterior_summary(self, item):
        self._check_fitted()
        i = self.get_item_idx(item)
        if i is None:
            b = self.b_prior_.copy()
            return {
                "seen_in_train": False,
                "b": b,
                "ep": None,
                "em": None,
                "beta_var": None,
                "dominant_factor": int(np.argmax(b)),
            }

        b = self.b_[i].copy()
        ep = self.ep_[i].copy()
        em = self.em_[i].copy()

        beta_var = (ep * em) / (((ep + em) ** 2) * (ep + em + 1))

        return {
            "seen_in_train": True,
            "b": b,
            "ep": ep,
            "em": em,
            "beta_var": beta_var,
            "dominant_factor": int(np.argmax(b)),
            "posterior_mass": float(np.sum(ep + em)),
        }

    def get_interaction_lambda(self, user, item):
        """
        Devuelve q(z_ui) si la interacción concreta (user,item)
        existe en observations_ del entrenamiento.
        """
        self._check_fitted()

        if not hasattr(self, "lambdas_"):
            return None

        u = self.get_user_idx(user)
        i = self.get_item_idx(item)

        if u is None or i is None:
            return None

        for obs_idx, (uu, ii, r) in enumerate(self.observations_):
            if uu == u and ii == i:
                return self.lambdas_[obs_idx].copy()

        return None

    def analyze_case(self, user, item, true_rating=None):
        """
        Resumen interpretable completo para un caso (user, item).
        """
        self._check_fitted()

        user_info = self.get_user_posterior_summary(user)
        item_info = self.get_item_posterior_summary(item)

        p_ui = self.predict_normalized(user, item)
        pred_exp = self.predict_expected_rating(user, item)
        pred_map = self.predict(user, item)
        rating_probs = self.predict_proba_each_rating(user, item)

        a = self.get_user_embedding(user)
        b = self.get_item_embedding(item)
        contributions = a * b
        top_factors = np.argsort(contributions)[::-1]

        lambda_ui = self.get_interaction_lambda(user, item)

        result = {
            "user": user,
            "item": item,
            "true_rating": true_rating,
            "pred_expected": float(pred_exp),
            "pred_map": float(pred_map),
            "p_ui": float(p_ui),
            "top_factor_by_contribution": int(top_factors[0]),
            "top_factor_contribution": float(contributions[top_factors[0]]),

            # Usuario (phi_u)
            "user_seen_in_train": user_info["seen_in_train"],
            "user_entropy": user_info["entropy"],
            "user_dominant_factor": user_info["dominant_factor"],
            "user_a": user_info["a"],
            "user_gamma": user_info["gamma"],

            # Ítem (kappa_i,k)
            "item_seen_in_train": item_info["seen_in_train"],
            "item_dominant_factor": item_info["dominant_factor"],
            "item_b": item_info["b"],
            "item_ep": item_info["ep"],
            "item_em": item_info["em"],
            "item_beta_var": item_info["beta_var"],

            # Interacción (z_ui)
            "lambda_ui": lambda_ui,
            "lambda_dominant_factor": None if lambda_ui is None else int(np.argmax(lambda_ui)),

            # Distribución de rating
            "rating_probs": rating_probs,
            "contributions": contributions,
        }

        return result

    # =========================================================
    # Internals
    # =========================================================
    def _validate_fit_df(self, df: pd.DataFrame):
        required = {"user", "item", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en fit: {missing}")
        if len(df) == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")

    def _validate_and_cast_ratings(self, s: pd.Series) -> np.ndarray:
        vals = s.to_numpy(dtype=float)

        if not np.all(np.isfinite(vals)):
            raise ValueError("Hay ratings no finitos.")

        # Fiel al paper: ratings discretos en 0..R
        rounded = np.rint(vals)
        if not np.allclose(vals, rounded):
            bad = np.unique(vals[np.abs(vals - rounded) > 1e-8])[:10]
            raise ValueError(
                f"El modelo requiere ratings discretos enteros en 0..{self.R}. "
                f"Valores no enteros detectados: {bad.tolist()}"
            )

        rounded = rounded.astype(int)

        bad_range = (rounded < 0) | (rounded > self.R)
        if np.any(bad_range):
            bad = np.unique(rounded[bad_range])[:10]
            raise ValueError(
                f"El modelo requiere ratings enteros en 0..{self.R}. "
                f"Valores problemáticos: {bad.tolist()}"
            )

        return rounded

    def _binom_logpmf(self, r: int, n: int, p: float) -> float:
        p = float(np.clip(p, self.eps, 1.0 - self.eps))
        return (
            gammaln(n + 1)
            - gammaln(r + 1)
            - gammaln(n - r + 1)
            + r * np.log(p)
            + (n - r) * np.log(1.0 - p)
        )

    def _binom_pmf(self, r: int, n: int, p: float) -> float:
        return float(np.exp(self._binom_logpmf(r, n, p)))

    def _predict_map_from_p(self, p_ui: float) -> int:
        logps = np.array(
            [self._binom_logpmf(r, self.R, p_ui) for r in self.rating_values_],
            dtype=float,
        )
        return int(self.rating_values_[np.argmax(logps)])

    def _compute_diagnostics(self, observations, a, b, lambdas) -> Dict[str, float]:
        errs_map = []
        errs_exp = []

        for u, i, r in observations:
            p = float(np.clip(np.dot(a[u], b[i]), 0.0, 1.0))
            pred_map = self._predict_map_from_p(p)
            pred_exp = self.R * p
            errs_map.append(abs(r - pred_map))
            errs_exp.append(abs(r - pred_exp))

        user_entropy = -np.mean(np.sum(a * np.log(np.clip(a, self.eps, None)), axis=1))
        item_factor_std = float(np.mean(np.std(b, axis=0)))
        peak_resp = float(np.mean(np.max(lambdas, axis=1)))

        return {
            "train_mae_map": float(np.mean(errs_map)) if errs_map else 0.0,
            "train_mae_exp": float(np.mean(errs_exp)) if errs_exp else 0.0,
            "user_entropy": float(user_entropy),
            "item_factor_std": float(item_factor_std),
            "peak_resp": float(peak_resp),
        }