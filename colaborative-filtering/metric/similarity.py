import numpy as np
import pandas as pd


class WeightedUserSimilarity:
    def __init__(
        self,
        alpha_jaccard=0.3,
        beta_significance=50,
        shrinkage=10,
        min_common=2,
        use_abs=True,
        jaccard_mode="mix",  # "mix" | "multiply" | "none"
    ):
        self.alpha_jaccard = alpha_jaccard
        self.beta_significance = beta_significance
        self.shrinkage = shrinkage
        self.min_common = min_common
        self.use_abs = use_abs
        self.jaccard_mode = jaccard_mode

    # -----------------------------
    # FIT
    # -----------------------------
    def fit(self, df):
        """
        df: DataFrame con columnas ['user', 'item', 'rating']
        """
        self.df = df.copy()

        # matriz usuario-item
        self.user_item = df.pivot(index='user', columns='item', values='rating')

        # medias por usuario
        self.user_means = df.groupby('user')['rating'].mean()

        # cache de sets de items por usuario
        self.user_items = {
            u: set(self.user_item.loc[u].dropna().index)
            for u in self.user_item.index
        }

        return self

    # -----------------------------
    # COMPONENTES
    # -----------------------------
    def _pearson(self, u, v, common_items):
        u_r = self.user_item.loc[u, common_items]
        v_r = self.user_item.loc[v, common_items]

        u_c = u_r - self.user_means[u]
        v_c = v_r - self.user_means[v]

        denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
        if denom == 0:
            return 0.0

        return np.dot(u_c, v_c) / denom

    def _jaccard(self, u, v):
        u_items = self.user_items[u]
        v_items = self.user_items[v]

        inter = len(u_items & v_items)
        union = len(u_items | v_items)

        return inter / union if union != 0 else 0.0

    def _significance(self, n_common):
        return min(n_common / self.beta_significance, 1.0)

    def _shrinkage(self, sim, n_common):
        # penalización bayesiana
        return (n_common / (n_common + self.shrinkage)) * sim

    def _apply_jaccard(self, sim, jacc):
        if self.jaccard_mode == "none":
            return sim

        elif self.jaccard_mode == "multiply":
            return sim * jacc

        elif self.jaccard_mode == "mix":
            return sim * (self.alpha_jaccard * jacc + (1 - self.alpha_jaccard))

        else:
            raise ValueError("Invalid jaccard_mode")

    # -----------------------------
    # SIMILARIDAD FINAL
    # -----------------------------
    def similarity(self, u, v):
        u_items = self.user_items[u]
        v_items = self.user_items[v]

        common_items = list(u_items & v_items)
        n_common = len(common_items)

        if n_common < self.min_common:
            return 0.0

        # 1. Pearson
        sim = self._pearson(u, v, common_items)

        # 2. Shrinkage (muy importante)
        sim = self._shrinkage(sim, n_common)

        # 3. Significance weighting
        sim *= self._significance(n_common)

        # 4. Jaccard
        jacc = self._jaccard(u, v)
        sim = self._apply_jaccard(sim, jacc)

        return sim

    # -----------------------------
    # MATRIZ DE SIMILITUD (costoso)
    # -----------------------------
    def compute_similarity_matrix(self):
        users = self.user_item.index
        n = len(users)

        sim_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=users,
            columns=users
        )

        for i, u in enumerate(users):
            for j, v in enumerate(users):
                if i >= j:
                    continue

                sim = self.similarity(u, v)

                sim_matrix.loc[u, v] = sim
                sim_matrix.loc[v, u] = sim

        self.sim_matrix = sim_matrix
        return sim_matrix

    # -----------------------------
    # PREDICCIÓN
    # -----------------------------
    def predict(self, user_id, item_id, k=20):
        if item_id not in self.user_item.columns:
            return self.user_means.get(user_id, np.nan)

        # usuarios que han valorado el item
        item_ratings = self.user_item[item_id].dropna()

        if user_id not in self.user_item.index:
            return np.nan

        sims = {
            v: self.similarity(user_id, v)
            for v in item_ratings.index if v != user_id
        }

        sims = pd.Series(sims)

        # top-k vecinos
        sims = sims.sort_values(ascending=False).head(k)

        if sims.sum() == 0:
            return self.user_means[user_id]

        ratings = item_ratings[sims.index]
        means = self.user_means[sims.index]

        num = ((ratings - means) * sims).sum()
        den = np.abs(sims).sum() if self.use_abs else sims.sum()

        return self.user_means[user_id] + num / den