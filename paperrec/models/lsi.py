"""Latent Semantic Indexing (LSI) recommender using Truncated SVD on TF-IDF."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import Recommender


class LSIRecommender(Recommender):
    """LSI Recommender: Truncated SVD on TF-IDF representation of papers.

    Strictly adheres to inductive splitting: fits vocabulary, IDF, and SVD
    transformation matrix ONLY on `train_items`, then projects all catalog papers
    into the k-dimensional latent semantic space.
    """

    method = "lsi"

    def __init__(
        self,
        *,
        n_components: int = 200,
        sublinear_tf: bool = False,
        min_df: int = 1,
        max_df: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_components = int(n_components)
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "method": self.method,
            "n_components": self.n_components,
            "sublinear_tf": self.sublinear_tf,
            "min_df": self.min_df,
            "max_df": self.max_df,
        }

    def _fit(self):
        train_texts = [self.papers[i].text for i in self.train_items]

        # 1. Fit TF-IDF on training items only
        self.vectorizer_ = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            smooth_idf=True,
            norm="l2",
            dtype=np.float32,
        )
        x_train = self.vectorizer_.fit_transform(train_texts)

        # 2. Fit TruncatedSVD on training TF-IDF matrix
        # Limit components if vocabulary is smaller than n_components
        actual_components = min(self.n_components, x_train.shape[1] - 1)
        if actual_components <= 0:
            actual_components = 1

        self.svd_ = TruncatedSVD(
            n_components=actual_components,
            random_state=self.seed,
            algorithm="randomized",
        )
        self.svd_.fit(x_train)

        # 3. Transform all catalog papers
        all_texts = [p.text for p in self.papers]
        x_all = self.vectorizer_.transform(all_texts)
        raw_vectors = self.svd_.transform(x_all).astype(np.float32)

        # 4. L2-normalize vectors so cosine similarity becomes dot product
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.paper_vectors_ = raw_vectors / norms

    def score(self, profile, user_index: int | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        ids = self._item_ids(profile)
        if not len(ids):
            return np.zeros(len(self.papers), dtype=np.float32)

        # User profile = L2-normalized sum of observed paper vectors
        centroid = self.paper_vectors_[ids].sum(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        scores = self.paper_vectors_ @ centroid
        return np.asarray(scores, dtype=np.float32).ravel()

    def get_embeddings(self) -> np.ndarray:
        """Return the L2-normalized paper embeddings matrix of shape (n_papers, n_components)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_embeddings()")
        return self.paper_vectors_
