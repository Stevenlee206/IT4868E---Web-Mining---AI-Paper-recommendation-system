"""Item-based k-nearest neighbours with cosine similarity on training saves."""

import numpy as np
from scipy import sparse

from .base import Recommender


class ItemKNNRecommender(Recommender):
    """Score a paper by its summed similarity to the user's saved papers.

    Each paper keeps its own k most similar papers. A paper without training saves
    has no neighbours and scores zero, so cold papers are ranked by seeded ties only.
    """

    def __init__(self, *, neighbors=100, seed=42):
        super().__init__(seed=seed)
        self.neighbors = neighbors

    def get_config(self):
        return {**super().get_config(), "method": "item_knn", "neighbors": self.neighbors}

    def _fit(self):
        train = self.train.astype(np.float64)
        norms = np.sqrt(np.asarray(train.sum(axis=0)).ravel())  # Saves are binary.
        scale = np.divide(1, norms, out=np.zeros_like(norms), where=norms > 0)
        normalized = (train @ sparse.diags(scale)).tocsc()
        rows, cols, values = [], [], []
        for start in range(0, train.shape[1], 1024):
            block = (normalized[:, start : start + 1024].T @ normalized).tocsr()
            for offset in range(block.shape[0]):
                item = start + offset
                low, high = block.indptr[offset : offset + 2]
                ids, scores = block.indices[low:high], block.data[low:high]
                keep = (ids != item) & (scores > 0)
                ids, scores = ids[keep], scores[keep]
                top = np.lexsort((ids, -scores))[: self.neighbors]
                rows.append(np.full(len(top), item))
                cols.append(ids[top])
                values.append(scores[top])
        self.similarity_ = sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(train.shape[1], train.shape[1]),
        )

    def score(self, profile, user_index=None):
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        saved = np.zeros(len(self.papers))
        saved[self._item_ids(profile)] = 1
        return self.similarity_ @ saved
