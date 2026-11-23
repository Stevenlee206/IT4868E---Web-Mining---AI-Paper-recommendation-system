"""SPECTER2 recommender and document representation encoder.

Uses allenai/specter2_base to produce 768-dimensional document-level representations
from scientific title and abstract. Supports offline caching for fast reuse and
handover to WP3 (SPECTER2+GCN).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Recommender


def encode_arxiv_paper(
    title: str,
    abstract: str,
    model=None,
    tokenizer=None,
    device=None,
) -> np.ndarray:
    """Encode a single paper (title + abstract) into a 768-d SPECTER2 vector.

    Provided for WP4 (arXiv Demo) to support content-based recommendations
    for newly published papers.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    if model is None:
        model = AutoModel.from_pretrained("allenai/specter2_base").to(device)
        model.eval()

    text = f"{title.strip()} [SEP] {abstract.strip()}"
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**inputs)
        # SPECTER2 uses CLS token embedding
        embedding = output.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding.astype(np.float32)


class SPECTER2Recommender(Recommender):
    """SPECTER2 Recommender: Dense scientific transformer embeddings with offline caching.

    Can load precomputed embeddings directly from `cache_path` to avoid re-encoding.
    """

    method = "specter2"

    def __init__(
        self,
        *,
        model_name: str = "allenai/specter2_base",
        cache_path: str | Path | None = "artifacts/embeddings/specter2_embeddings.npy",
        batch_size: int = 64,
        max_length: int = 512,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "method": self.method,
            "model_name": self.model_name,
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }

    def _fit(self):
        # 1. Try loading from precomputed cache
        if self.cache_path is not None and self.cache_path.exists():
            embeddings = np.load(self.cache_path)
            if embeddings.shape[0] != len(self.papers):
                raise ValueError(
                    f"Cache at {self.cache_path} has {embeddings.shape[0]} items, "
                    f"expected {len(self.papers)}"
                )
            # Ensure L2 normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings_ = (embeddings / norms).astype(np.float32)
            self.cache_hit_ = True
            return

        # 2. Encode on-the-fly using PyTorch and Transformers
        self.cache_hit_ = False
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(device)
        model.eval()

        texts = [f"{p.title.strip()} [SEP] {p.abstract.strip()}" for p in self.papers]
        all_embeddings = []

        for start_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[start_idx : start_idx + self.batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output = model(**inputs)
                batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(batch_embeddings)

        raw = np.vstack(all_embeddings).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings_ = raw / norms

        # Save to cache if path specified
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, self.embeddings_)

    def score(self, profile, user_index: int | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        ids = self._item_ids(profile)
        if not len(ids):
            return np.zeros(len(self.papers), dtype=np.float32)

        # User profile = L2-normalized centroid of observed paper embeddings
        user_vector = self.embeddings_[ids].sum(axis=0)
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector /= norm

        scores = self.embeddings_ @ user_vector
        return np.asarray(scores, dtype=np.float32).ravel()

    def get_embeddings(self) -> np.ndarray:
        """Return the (n_papers, 768) normalized embeddings matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_embeddings()")
        return self.embeddings_
