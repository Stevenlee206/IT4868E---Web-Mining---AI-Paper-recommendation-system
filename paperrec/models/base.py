"""Shared training interface and deterministic ranking for future models."""

from abc import ABC, abstractmethod

import numpy as np

from ..data import Paper
from ..splits import Split


class Recommender(ABC):
    """Implement _fit() and score(); call fit(papers, split) to train.

    _fit uses self.papers, self.train, self.train_items and training_mask(users).
    Warm negative sampling/losses must respect training_mask. Cold collaborative
    parameters and learned text preprocessing must use train_items only; all
    catalog text remains available for transforming unseen papers.
    """

    def __init__(self, seed=42):
        self.seed = seed
        self._fitted = False

    def fit(self, papers: list[Paper], split: Split):
        self._fitted = False
        if len(papers) != split.train.shape[1]:
            raise ValueError("Paper count must match the interaction columns")
        if any(paper.paper_id != i for i, paper in enumerate(papers)):
            raise ValueError("Papers must be ordered by zero-based paper ID")
        self.papers = list(papers)
        self.train = split.train.copy()
        self.train_items = split.train_items.copy()
        # Provide pair membership without passing held-out positive labels to _fit.
        self._training_mask = lambda users: split.pair_mask("train", users)
        self.tie_order = np.random.default_rng(self.seed).permutation(len(papers))
        self._fit()
        self._fitted = True
        return self

    def training_mask(self, users):
        """Return allowed training pairs for a batch of user indices."""
        return self._training_mask(users)

    @abstractmethod
    def _fit(self):
        """Learn from the training inputs set by fit()."""

    @abstractmethod
    def score(self, profile, user_index=None):
        """Return a one-dimensional array of finite scores for every paper."""

    def _item_ids(self, values):
        ids = np.asarray(values)
        if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("Paper indices must be a one-dimensional integer sequence")
        if np.any((ids < 0) | (ids >= len(self.papers))):
            raise ValueError("Unknown paper index")
        return np.unique(ids.astype(np.int64))

    def recommend(self, profile, k=10, user_index=None, candidates=None, exclude=()):
        """Rank candidates, excluding observed papers; break score ties by seed."""
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend()")
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        profile = self._item_ids(profile)
        if not len(profile):
            raise ValueError("At least one observed paper is required")
        if user_index is not None and (
            isinstance(user_index, (bool, np.bool_))
            or not isinstance(user_index, (int, np.integer))
            or not 0 <= user_index < self.train.shape[0]
        ):
            raise ValueError("Unknown user index")
        ids = np.arange(len(self.papers)) if candidates is None else self._item_ids(candidates)
        ids = np.setdiff1d(ids, np.union1d(profile, self._item_ids(exclude)))
        if not len(ids):
            return []
        scores = np.asarray(self.score(profile, user_index), dtype=float)
        if scores.shape != (len(self.papers),) or not np.isfinite(scores).all():
            raise ValueError("Model must return one finite score per paper")
        order = np.lexsort((self.tie_order[ids], -scores[ids]))
        return [(int(item), float(scores[item])) for item in ids[order[:k]]]
