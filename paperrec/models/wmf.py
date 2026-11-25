"""Weighted matrix factorization (Hu, Koren & Volinsky, 2008) with masked pairs.

Every training pair has weight 1, saved pairs have weight 1 + alpha, and pairs that a
split holds out have weight 0. Libraries treat held-out pairs as observed zeros, which
would break the protocol in splits.py, so ALS is written here around the pair mask.

This module imports torch, so it is not exported from paperrec.models:
    from paperrec.models.wmf import WMFRecommender
"""

import numpy as np
import torch

from .base import Recommender


def _solve(other, weights, positives, alpha, regularization, batch_size):
    """Closed-form ALS update; weights and positives have one row per solved factor."""
    size = other.shape[1]
    outer = (other.unsqueeze(2) * other.unsqueeze(1)).reshape(len(other), size * size)
    ridge = regularization * torch.eye(size, device=other.device)
    result = torch.empty(len(weights), size, device=other.device)
    for start in range(0, len(weights), batch_size):
        rows = slice(start, start + batch_size)
        gram = (weights[rows] @ outer).reshape(-1, size, size) + ridge
        target = (1 + alpha) * (positives[rows] @ other)
        result[rows] = torch.linalg.solve(gram, target)
    return result


class WMFRecommender(Recommender):
    """Rank by the dot product of user and paper factors; cold papers score zero."""

    def __init__(
        self,
        *,
        factors=64,
        alpha=40.0,
        regularization=1.0,
        iterations=20,
        batch_size=256,
        device=None,
        seed=42,
    ):
        super().__init__(seed=seed)
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_config(self):
        return {
            **super().get_config(),
            "method": "wmf_masked_als",
            "factors": self.factors,
            "alpha": self.alpha,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }

    def _fit(self):
        n_users, n_items = self.train.shape
        allowed = self.training_mask(np.arange(n_users))
        positives = torch.from_numpy(self.train.toarray().astype(np.float32)).to(self.device)
        weights = torch.from_numpy(allowed.astype(np.float32)).to(self.device)
        weights = weights + self.alpha * positives
        generator = torch.Generator().manual_seed(self.seed)
        items = 0.01 * torch.randn(n_items, self.factors, generator=generator)
        items = items.to(self.device)
        options = (self.alpha, self.regularization, self.batch_size)
        for _ in range(self.iterations):
            users = _solve(items, weights, positives, *options)
            items = _solve(users, weights.T, positives.T, *options)
        self.user_factors_ = users.cpu().numpy()
        self.item_factors_ = items.cpu().numpy()

    def score(self, profile, user_index=None):
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        if user_index is None:
            raise ValueError("WMF needs user_index")
        return self.item_factors_ @ self.user_factors_[user_index]
