"""Latent Dirichlet Allocation (LDA) recommender and topic distribution generator."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .base import Recommender


class LDARecommender(Recommender):
    """LDA Recommender: Topic modeling on bag-of-words document representations.

    Also generates the paper-topic mixture matrix Theta (shape: n_papers, n_topics),
    which is directly exported for Collaborative Topic Regression (CTR).
    """

    method = "lda"

    def __init__(
        self,
        *,
        n_topics: int = 100,
        max_iter: int = 20,
        min_df: int = 5,
        max_df: float = 0.5,
        learning_method: str = "online",
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        if n_topics <= 0:
            raise ValueError("n_topics must be a positive integer")
        self.n_topics = int(n_topics)
        self.max_iter = int(max_iter)
        self.min_df = min_df
        self.max_df = max_df
        self.learning_method = learning_method

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "method": self.method,
            "n_topics": self.n_topics,
            "max_iter": self.max_iter,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "learning_method": self.learning_method,
        }

    def _fit(self):
        train_texts = [self.papers[i].text for i in self.train_items]

        # 1. Fit CountVectorizer on training items only
        self.vectorizer_ = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            min_df=self.min_df,
            max_df=self.max_df,
        )
        counts_train = self.vectorizer_.fit_transform(train_texts)

        # 2. Fit Latent Dirichlet Allocation
        actual_topics = min(self.n_topics, counts_train.shape[1] - 1)
        if actual_topics <= 0:
            actual_topics = 1

        self.lda_ = LatentDirichletAllocation(
            n_components=actual_topics,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            random_state=self.seed,
            batch_size=128,
            n_jobs=1,
        )
        self.lda_.fit(counts_train)

        # 3. Transform all catalog papers to topic distributions (Theta matrix)
        all_texts = [p.text for p in self.papers]
        counts_all = self.vectorizer_.transform(all_texts)
        theta = self.lda_.transform(counts_all).astype(np.float32)

        # theta rows sum to 1.0 (probabilities). In case of zero-count document:
        row_sums = theta.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.theta_ = theta / row_sums

        # Precompute L2 normalized theta for fast cosine similarity scoring
        norms = np.linalg.norm(self.theta_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.theta_normalized_ = self.theta_ / norms

    def score(self, profile, user_index: int | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        ids = self._item_ids(profile)
        if not len(ids):
            return np.zeros(len(self.papers), dtype=np.float32)

        # User profile topic distribution: mean of normalized topic vectors
        user_vector = self.theta_normalized_[ids].sum(axis=0)
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector /= norm

        scores = self.theta_normalized_ @ user_vector
        return np.asarray(scores, dtype=np.float32).ravel()

    def get_topic_distribution(self) -> np.ndarray:
        """Return the unnormalized topic mixture matrix Theta of shape (n_papers, n_topics).

        Useful for CTR (Collaborative Topic Regression).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before get_topic_distribution()")
        return self.theta_

    def get_embeddings(self) -> np.ndarray:
        """Return normalized topic vectors for vector-based ranking."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_embeddings()")
        return self.theta_normalized_
