import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DatasetProfiler:
    """
    Analizador de datasets de recomendación.

    Parámetros
    ----------
    user_col : str
        Nombre de la columna de usuario.
    item_col : str
        Nombre de la columna de item.
    rating_col : str or None
        Nombre de la columna de rating. Si es None, no se analizan ratings.
    timestamp_col : str or None
        Nombre de la columna temporal, si existe.
    max_threshold : int
        Máximo umbral para curvas de soporte.
    """

    def __init__(
        self,
        user_col="user",
        item_col="item",
        rating_col="rating",
        timestamp_col=None,
        max_threshold=100
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.max_threshold = max_threshold

        self.df_ = None
        self.report_ = None
        self.user_counts_ = None
        self.item_counts_ = None
        self.is_fitted_ = False

    # =========================
    # API pública
    # =========================
    def fit(self, df: pd.DataFrame):
        self._validate_input(df)

        self.df_ = df.copy()

        self.user_counts_ = self.df_[self.user_col].value_counts().sort_values(ascending=False)
        self.item_counts_ = self.df_[self.item_col].value_counts().sort_values(ascending=False)

        self.report_ = {
            "dataset_summary": self._build_dataset_summary(),
            "user_profile": self._build_entity_profile(self.user_counts_, entity_name="users"),
            "item_profile": self._build_entity_profile(self.item_counts_, entity_name="items"),
            "user_support_curve": self._build_support_curve(self.user_counts_),
            "item_support_curve": self._build_support_curve(self.item_counts_),
            "diagnostics": self._build_diagnostics()
        }

        self.is_fitted_ = True
        return self

    def get_report(self):
        self._check_is_fitted()
        return self.report_

    def to_dict(self):
        self._check_is_fitted()
        return self.report_

    def save_report(self, path: str):
        self._check_is_fitted()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report_, f, indent=4, ensure_ascii=False)

    def print_report(self, decimals=4):
        self._check_is_fitted()

        r = self.report_
        ds = r["dataset_summary"]
        up = r["user_profile"]
        ip = r["item_profile"]
        dg = r["diagnostics"]

        def fmt(x):
            if isinstance(x, float):
                return round(x, decimals)
            return x

        print("=" * 70)
        print("RECOMMENDER DATASET REPORT")
        print("=" * 70)

        print("\n[1] DATASET SUMMARY")
        for k, v in ds.items():
            print(f"  - {k}: {fmt(v)}")

        print("\n[2] USER PROFILE")
        self._print_profile_block(up, fmt)

        print("\n[3] ITEM PROFILE")
        self._print_profile_block(ip, fmt)

        print("\n[4] DIAGNOSTICS")
        for k, v in dg.items():
            print(f"  - {k}: {fmt(v)}")

        print("=" * 70)

    def plot_support_curves(self, figsize=(10, 5), log_y=False):
        self._check_is_fitted()

        user_curve = self.report_["user_support_curve"]
        item_curve = self.report_["item_support_curve"]

        thresholds_u = user_curve["thresholds"]
        coverage_u = user_curve["coverage"]

        thresholds_i = item_curve["thresholds"]
        coverage_i = item_curve["coverage"]

        plt.figure(figsize=figsize)
        plt.plot(thresholds_u, coverage_u, marker="o", label="Usuarios")
        plt.plot(thresholds_i, coverage_i, marker="o", label="Ítems")
        plt.xlabel("Mínimo número de interacciones")
        plt.ylabel("Proporción con al menos n interacciones")
        plt.title("Support / survival curve")
        if log_y:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def plot_count_distributions(self, bins=50, figsize=(10, 5), log_y=True):
        self._check_is_fitted()

        plt.figure(figsize=figsize)
        plt.hist(self.user_counts_.values, bins=bins, alpha=0.6, label="Usuarios")
        plt.hist(self.item_counts_.values, bins=bins, alpha=0.6, label="Ítems")
        plt.xlabel("Número de interacciones")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de interacciones por entidad")
        if log_y:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    # =========================
    # Construcción del reporte
    # =========================
    def _build_dataset_summary(self):
        n_interactions = len(self.df_)
        n_users = self.df_[self.user_col].nunique()
        n_items = self.df_[self.item_col].nunique()

        density = n_interactions / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0
        sparsity = 1.0 - density

        summary = {
            "n_interactions": int(n_interactions),
            "n_users": int(n_users),
            "n_items": int(n_items),
            "density": float(density),
            "sparsity": float(sparsity),
            "avg_interactions_per_user": float(n_interactions / n_users) if n_users > 0 else 0.0,
            "avg_interactions_per_item": float(n_interactions / n_items) if n_items > 0 else 0.0,
        }

        if self.rating_col is not None and self.rating_col in self.df_.columns:
            ratings = self.df_[self.rating_col].astype(float)
            summary.update({
                "rating_mean": float(ratings.mean()),
                "rating_std": float(ratings.std(ddof=1)) if len(ratings) > 1 else 0.0,
                "rating_min": float(ratings.min()),
                "rating_max": float(ratings.max()),
                "rating_median": float(ratings.median())
            })

        return summary

    def _build_entity_profile(self, counts: pd.Series, entity_name="entities"):
        counts_arr = counts.values.astype(int)
        total_entities = len(counts_arr)
        total_interactions = counts_arr.sum()

        profile = {
            "n_" + entity_name: int(total_entities),
            "min_interactions": int(np.min(counts_arr)) if total_entities > 0 else 0,
            "max_interactions": int(np.max(counts_arr)) if total_entities > 0 else 0,
            "mean_interactions": float(np.mean(counts_arr)) if total_entities > 0 else 0.0,
            "median_interactions": float(np.median(counts_arr)) if total_entities > 0 else 0.0,
            "std_interactions": float(np.std(counts_arr, ddof=1)) if total_entities > 1 else 0.0,
            "p25_interactions": float(np.percentile(counts_arr, 25)) if total_entities > 0 else 0.0,
            "p75_interactions": float(np.percentile(counts_arr, 75)) if total_entities > 0 else 0.0,
            "p90_interactions": float(np.percentile(counts_arr, 90)) if total_entities > 0 else 0.0,
            "p95_interactions": float(np.percentile(counts_arr, 95)) if total_entities > 0 else 0.0,
            "p99_interactions": float(np.percentile(counts_arr, 99)) if total_entities > 0 else 0.0,
            "share_with_1_interaction": float(np.mean(counts_arr == 1)) if total_entities > 0 else 0.0,
            "share_with_le_2_interactions": float(np.mean(counts_arr <= 2)) if total_entities > 0 else 0.0,
            "share_with_le_5_interactions": float(np.mean(counts_arr <= 5)) if total_entities > 0 else 0.0,
            "share_with_le_10_interactions": float(np.mean(counts_arr <= 10)) if total_entities > 0 else 0.0,
            "share_with_ge_10_interactions": float(np.mean(counts_arr >= 10)) if total_entities > 0 else 0.0,
            "share_with_ge_20_interactions": float(np.mean(counts_arr >= 20)) if total_entities > 0 else 0.0,
            "head_1pct_interaction_share": self._top_k_share(counts_arr, top_fraction=0.01),
            "head_5pct_interaction_share": self._top_k_share(counts_arr, top_fraction=0.05),
            "head_10pct_interaction_share": self._top_k_share(counts_arr, top_fraction=0.10),
            "gini_interaction_concentration": self._gini(counts_arr),
            "total_interactions_explained": int(total_interactions),
        }

        return profile

    def _build_support_curve(self, counts: pd.Series):
        thresholds = list(range(1, self.max_threshold + 1))
        total = len(counts)

        if total == 0:
            coverage = [0.0 for _ in thresholds]
        else:
            coverage = [float((counts.values >= t).mean()) for t in thresholds]

        return {
            "thresholds": thresholds,
            "coverage": coverage
        }

    def _build_diagnostics(self):
        ds = self._build_dataset_summary()
        up = self._build_entity_profile(self.user_counts_, entity_name="users")
        ip = self._build_entity_profile(self.item_counts_, entity_name="items")

        diagnostics = {}

        # Diagnóstico cualitativo simple
        if ds["sparsity"] > 0.99:
            diagnostics["sparsity_level"] = "very_high"
        elif ds["sparsity"] > 0.95:
            diagnostics["sparsity_level"] = "high"
        elif ds["sparsity"] > 0.90:
            diagnostics["sparsity_level"] = "moderate"
        else:
            diagnostics["sparsity_level"] = "low"

        if ip["share_with_1_interaction"] > 0.5:
            diagnostics["item_tail_severity"] = "extreme"
        elif ip["share_with_1_interaction"] > 0.3:
            diagnostics["item_tail_severity"] = "high"
        else:
            diagnostics["item_tail_severity"] = "moderate_or_low"

        if up["share_with_1_interaction"] > 0.5:
            diagnostics["user_tail_severity"] = "extreme"
        elif up["share_with_1_interaction"] > 0.3:
            diagnostics["user_tail_severity"] = "high"
        else:
            diagnostics["user_tail_severity"] = "moderate_or_low"

        diagnostics["matrix_factorization_risk"] = self._mf_risk_label(
            sparsity=ds["sparsity"],
            user_one_shot=up["share_with_1_interaction"],
            item_one_shot=ip["share_with_1_interaction"]
        )

        diagnostics["cold_start_risk_items"] = (
            "high" if ip["share_with_le_5_interactions"] > 0.7 else
            "moderate" if ip["share_with_le_5_interactions"] > 0.4 else
            "low"
        )

        diagnostics["cold_start_risk_users"] = (
            "high" if up["share_with_le_5_interactions"] > 0.7 else
            "moderate" if up["share_with_le_5_interactions"] > 0.4 else
            "low"
        )

        return diagnostics

    # =========================
    # Helpers internos
    # =========================
    def _validate_input(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame")

        required_cols = [self.user_col, self.item_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Falta la columna obligatoria: '{col}'")

        if len(df) == 0:
            raise ValueError("El DataFrame está vacío")

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Debes llamar a fit(df) antes de usar esta clase")

    @staticmethod
    def _top_k_share(counts_arr, top_fraction=0.1):
        if len(counts_arr) == 0:
            return 0.0
        counts_sorted = np.sort(counts_arr)[::-1]
        k = max(1, int(np.ceil(len(counts_sorted) * top_fraction)))
        return float(counts_sorted[:k].sum() / counts_sorted.sum())

    @staticmethod
    def _gini(x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            return 0.0
        if np.any(x < 0):
            raise ValueError("Gini no está definido para valores negativos")
        if np.all(x == 0):
            return 0.0

        x_sorted = np.sort(x)
        n = len(x_sorted)
        cumx = np.cumsum(x_sorted)
        gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        return float(gini)

    @staticmethod
    def _mf_risk_label(sparsity, user_one_shot, item_one_shot):
        score = 0

        if sparsity > 0.99:
            score += 2
        elif sparsity > 0.95:
            score += 1

        if user_one_shot > 0.4:
            score += 1

        if item_one_shot > 0.4:
            score += 2
        elif item_one_shot > 0.2:
            score += 1

        if score >= 4:
            return "high"
        if score >= 2:
            return "moderate"
        return "low"

    @staticmethod
    def _print_profile_block(profile, fmt):
        for k, v in profile.items():
            print(f"  - {k}: {fmt(v)}")