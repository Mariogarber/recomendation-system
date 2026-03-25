from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel


class ItemKNNModel(BaseModel):
    """
    Item-based KNN collaborative filter (sparse).

    For each prediction the model:
      1. Retrieves the k most similar items (by the chosen distance metric on
         their user-rating vectors).
      2. Collects the target user's ratings for those neighbour items.
      3. Returns a similarity-weighted average of those ratings.
      4. Falls back to the global mean when there are no usable neighbours
         or when the user/item was unseen in training.

    Parameters
    ----------
    k : int
        Maximum number of neighbour items to use.
    metric : str
        Distance metric for NearestNeighbors. Must be one of
        ``ItemKNNModel.SUPPORTED_METRICS`` (default: ``'cosine'``).

        * ``'cosine'`` / ``'correlation'`` – similarity is computed as
          ``1 - distance`` (distance lies in [0, 2]; [0, 1] for
          non-negative rating vectors).
        * ``'euclidean'`` / ``'manhattan'`` / ``'minkowski'`` –
          similarity is computed as ``1 / (1 + distance)``, mapping
          [0, ∞) to (0, 1].
    clip_range : tuple[float, float] | None
        Clip predictions to this range after estimation.
    n_jobs : int
        Parallel jobs for NearestNeighbors (-1 = all cores).
    name : str | None
    """

    SUPPORTED_METRICS: frozenset[str] = frozenset(
        {"cosine", "euclidean", "manhattan", "minkowski", "correlation"}
    )

    def __init__(
        self,
        k: int = 10,
        metric: str = "cosine",
        clip_range: tuple | None = None,
        n_jobs: int = -1,
        name: str | None = None,
    ):
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Supported metrics: {sorted(self.SUPPORTED_METRICS)}"
            )
        super().__init__(name=name, clip_range=clip_range)
        self.k = k
        self.metric = metric
        self.n_jobs = n_jobs

        # Fitted attributes
        self._knn: NearestNeighbors | None = None
        self._item_user_sparse: csr_matrix | None = None
        self._user_encoder: LabelEncoder | None = None
        self._item_encoder: LabelEncoder | None = None
        self._global_mean: float | None = None
        self._n_users: int = 0
        self._n_items: int = 0

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ItemKNNModel":
        """
        Train on a DataFrame with columns ['user', 'item', 'rating'].

        Duplicated (user, item) pairs are averaged before building the
        sparse matrix.
        """
        self._validate(df)

        train = df.groupby(["user", "item"], as_index=False)["rating"].mean()

        self._global_mean = float(train["rating"].mean())

        self._user_encoder = LabelEncoder()
        self._item_encoder = LabelEncoder()
        train["user_enc"] = self._user_encoder.fit_transform(train["user"])
        train["item_enc"] = self._item_encoder.fit_transform(train["item"])

        self._n_users = int(train["user_enc"].nunique())
        self._n_items = int(train["item_enc"].nunique())

        # Build item × user sparse matrix (items are rows)
        self._item_user_sparse = csr_matrix(
            (
                train["rating"].values,
                (train["item_enc"].values, train["user_enc"].values),
            ),
            shape=(self._n_items, self._n_users),
        )

        self._knn = NearestNeighbors(
            metric=self.metric,
            algorithm="brute",
            n_jobs=self.n_jobs,
        )
        self._knn.fit(self._item_user_sparse)

        self.is_fitted_ = True
        return self

    def predict(self, user, item) -> float:
        """Predict rating for a single (user, item) pair."""
        self._check_fitted()

        user_enc = self._safe_encode_user(user)
        item_enc = self._safe_encode_item(item)

        if user_enc == -1 or item_enc == -1 or item_enc >= self._n_items:
            return float(self._clip(self._global_mean))

        item_vector = self._item_user_sparse[item_enc]

        # Cold item: no ratings at all
        if item_vector.nnz == 0:
            return float(self._clip(self._global_mean))

        n_neighbors = min(self.k + 1, self._n_items)
        distances, indices = self._knn.kneighbors(
            item_vector, n_neighbors=n_neighbors
        )

        # Exclude the query item itself
        neighbor_idx = [i for i in indices[0] if i != item_enc][: self.k]
        neighbor_dist = [
            d for i, d in zip(indices[0], distances[0]) if i != item_enc
        ][: self.k]

        # Ratings the target user gave to each neighbour item
        neighbor_ratings = np.asarray(
            self._item_user_sparse[neighbor_idx, user_enc].todense()
        ).flatten()

        rated_mask = neighbor_ratings != 0
        rated = neighbor_ratings[rated_mask]

        if len(rated) == 0:
            return float(self._clip(self._global_mean))

        # Convert distances to similarity weights using the appropriate
        # formula for the chosen metric.
        sims = self._dist_to_sim(np.array(neighbor_dist))[rated_mask]

        if sims.sum() == 0:
            est = float(rated.mean())
        else:
            est = float(np.dot(sims, rated) / sims.sum())

        return float(self._clip(est))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dist_to_sim(self, distances: np.ndarray) -> np.ndarray:
        """Convert neighbour distances to non-negative similarity weights.

        * ``cosine`` / ``correlation``: ``sim = 1 - distance``
          (distance in [0, 2]; [0, 1] for non-negative vectors).
        * All other supported metrics: ``sim = 1 / (1 + distance)``
          which maps [0, ∞) → (0, 1].
        """
        if self.metric in ("cosine", "correlation"):
            return 1.0 - distances
        return 1.0 / (1.0 + distances)

    def _validate(self, df: pd.DataFrame):
        missing = {"user", "item", "rating"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if len(df) == 0:
            raise ValueError("Training DataFrame is empty.")

    def _safe_encode_user(self, user) -> int:
        known = set(self._user_encoder.classes_)
        if user not in known:
            return -1
        return int(self._user_encoder.transform([user])[0])

    def _safe_encode_item(self, item) -> int:
        known = set(self._item_encoder.classes_)
        if item not in known:
            return -1
        return int(self._item_encoder.transform([item])[0])

    def __repr__(self):
        return f"ItemKNNModel(k={self.k}, metric='{self.metric}')"
