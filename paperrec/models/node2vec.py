"""Node2Vec recommender (Grover & Leskovec, 2016) on leakage-safe graphs.

Walks use NumPy; embeddings use gensim skip-gram with negative sampling. Install
gensim with: python -m pip install -e ".[graph]"

Only the training graph reaches the walks. A paper left without an embedding
receives the normalized mean of its embedded neighbours in the inference graph.
When cold, those neighbours are train papers only, so held-out papers never
shape the embeddings. Papers with no embedded neighbour rank below every paper
that has an embedding, and their mutual order is the seeded tie order.

A user profile is the normalized mean of the user's saved paper embeddings.
Scores are cosine similarities.

Warm citation embeddings ignore saves entirely, so all warm folds share one graph
and one cached embedding. Their fold spread reflects the splits and not training
randomness. workers=1 is reproducible for a seed; gensim is not with more workers.

Gensim 4.4 may print "Exception ignored in: ...our_dot_float" without a traceback.
It declares BLAS sdot with "except -1", so a dot product of exactly -1.0 looks like
an error. No exception exists; only that single rare update is affected.
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy import sparse

from ..graph import (
    bipartite_graph,
    citation_inference_graph,
    citation_training_graph,
    graph_fingerprint,
    validate_bipartite_graph,
    validate_citation_graphs,
)
from .base import Recommender
from .lexical import _json_hash

CACHE_VERSION = 1
NO_EMBEDDING_SCORE = -2.0  # Below every possible cosine similarity.


def node2vec_walks(adjacency, *, num_walks, walk_length, p, q, seed) -> np.ndarray:
    """Return int32 walks of shape (num_walks * connected_nodes, walk_length).

    Every node with an edge starts num_walks walks. Isolated nodes never appear.
    Second-order steps use rejection sampling: draw a uniform neighbour x of the
    current node, then accept it with probability alpha(previous, x) / max_alpha,
    where alpha is 1/p for a return, 1 for a neighbour of the previous node and
    1/q otherwise. This matches the exact node2vec transition distribution
    without storing a table per edge, which matters on hub-heavy graphs.
    """
    if num_walks < 1 or walk_length < 2:
        raise ValueError("num_walks must be >= 1 and walk_length must be >= 2")
    if not (np.isfinite(p) and np.isfinite(q) and p > 0 and q > 0):
        raise ValueError("p and q must be finite and positive")
    if adjacency.shape[0] != adjacency.shape[1] or (adjacency != adjacency.T).nnz:
        raise ValueError("Walks need a symmetric adjacency, so that no walk gets stuck")

    graph = sparse.csr_matrix(adjacency)
    graph.sort_indices()
    n_nodes = graph.shape[0]
    indptr, indices = graph.indptr, graph.indices
    degree = np.diff(indptr)
    starts = np.flatnonzero(degree > 0)
    if not len(starts):
        raise ValueError("Graph has no edges")

    # Canonical CSR order makes these keys sorted, so edge lookups can bisect.
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), degree)
    edge_keys = rows * n_nodes + indices

    def is_edge(sources, targets):
        keys = sources.astype(np.int64) * n_nodes + targets
        positions = np.minimum(np.searchsorted(edge_keys, keys), len(edge_keys) - 1)
        return edge_keys[positions] == keys

    rng = np.random.default_rng(seed)

    def uniform_neighbours(nodes):
        return indices[indptr[nodes] + rng.integers(0, degree[nodes])]

    return_weight, outward_weight = 1 / p, 1 / q
    max_weight = max(return_weight, 1.0, outward_weight)
    walks = np.empty((num_walks * len(starts), walk_length), dtype=np.int32)
    for round_index in range(num_walks):
        batch = walks[round_index * len(starts) : (round_index + 1) * len(starts)]
        batch[:, 0] = rng.permutation(starts)
        batch[:, 1] = uniform_neighbours(batch[:, 0])
        for step in range(2, walk_length):
            previous, current = batch[:, step - 2], batch[:, step - 1]
            pending = np.arange(len(batch))
            while len(pending):
                candidates = uniform_neighbours(current[pending])
                weights = np.full(len(pending), outward_weight)
                weights[is_edge(previous[pending], candidates)] = 1.0
                weights[candidates == previous[pending]] = return_weight
                accepted = rng.random(len(pending)) * max_weight < weights
                batch[pending[accepted], step] = candidates[accepted]
                pending = pending[~accepted]
    return walks


class _WalkCorpus:
    """Re-iterable corpus; gensim scans it once per epoch plus once for vocabulary."""

    def __init__(self, walks):
        self.walks = walks

    def __iter__(self):
        for walk in self.walks:
            yield walk.tolist()


def train_embeddings(
    walks, n_nodes, *, dimensions, window, negative, epochs, workers, seed
) -> tuple[np.ndarray, np.ndarray]:
    """Train skip-gram; return (embeddings by node ID, mask of nodes that have one)."""
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=_WalkCorpus(walks),
        vector_size=dimensions,
        window=window,
        min_count=0,
        sg=1,
        hs=0,
        negative=negative,
        sample=0,  # Hubs are structure here, not stop words.
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    # Tokens are integers, so read key_to_index: wv[int] would mean a vocabulary row.
    nodes = np.fromiter(model.wv.key_to_index.keys(), dtype=np.int64)
    positions = np.fromiter(model.wv.key_to_index.values(), dtype=np.int64)
    embeddings = np.zeros((n_nodes, dimensions), dtype=np.float32)
    embeddings[nodes] = model.wv.vectors[positions]
    has_embedding = np.zeros(n_nodes, dtype=bool)
    has_embedding[nodes] = True
    return embeddings, has_embedding


def _normalized(matrix: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(lengths > 0, lengths, 1)


class Node2VecRecommender(Recommender):
    """Rank papers by cosine similarity to the mean of the user's saved papers.

    graph="citation" works for warm and cold. graph="bipartite" walks on training
    saves and is warm only, because held-out cold papers have no edges in it.
    """

    def __init__(
        self,
        citations=None,
        *,
        graph="citation",
        dimensions=128,
        num_walks=10,
        walk_length=80,
        window=10,
        p=1.0,
        q=1.0,
        negative=5,
        epochs=1,
        workers=1,
        seed=42,
        cache_root="artifacts/features/node2vec",
    ):
        super().__init__(seed=seed)
        if graph not in ("citation", "bipartite"):
            raise ValueError('graph must be "citation" or "bipartite"')
        if graph == "citation" and citations is None:
            raise ValueError("The citation graph needs dataset.citations")
        self.citations = citations
        self.graph = graph
        self.dimensions = dimensions
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window = window
        self.p, self.q = float(p), float(q)
        self.negative = negative
        self.epochs = epochs
        self.workers = workers
        self.cache_root = Path(cache_root) if cache_root is not None else None

    def get_config(self):
        return {
            **super().get_config(),
            "method": "node2vec",
            "graph": self.graph,
            "dimensions": self.dimensions,
            "num_walks": self.num_walks,
            "walk_length": self.walk_length,
            "window": self.window,
            "p": self.p,
            "q": self.q,
            "negative": self.negative,
            "epochs": self.epochs,
            "workers": self.workers,
            "profile": "normalized_mean_of_l2_paper_vectors",
        }

    def fit(self, papers, split):
        # Build graphs here because _fit must not receive held-out labels via the split.
        if self.graph == "bipartite":
            if split.scenario == "cold":
                raise ValueError("Bipartite Node2Vec cannot embed held-out cold papers")
            self._training_graph = bipartite_graph(split)
            validate_bipartite_graph(self._training_graph, split)
            self._inference_graph = None
        else:
            self._training_graph = citation_training_graph(self.citations, split)
            self._inference_graph = citation_inference_graph(self.citations, split)
            validate_citation_graphs(
                self._training_graph, self._inference_graph, self.citations, split
            )
        return super().fit(papers, split)

    def _train_or_load(self):
        self.cache_hit_ = False
        self.cache_path_ = None
        identity = None
        if self.cache_root is not None:
            import gensim

            config = self.get_config()
            config.pop("profile")
            identity = {
                "format_version": CACHE_VERSION,
                "graph_fingerprint": graph_fingerprint(self._training_graph),
                "config": config,
                "versions": {"numpy": np.__version__, "gensim": gensim.__version__},
            }
            self.cache_path_ = self.cache_root / f"{_json_hash(identity)}.npz"
            if self.cache_path_.exists():
                self.cache_hit_ = True
                return self._load_cache(identity)
        walks = node2vec_walks(
            self._training_graph,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            p=self.p,
            q=self.q,
            seed=self.seed,
        )
        result = train_embeddings(
            walks,
            self._training_graph.shape[0],
            dimensions=self.dimensions,
            window=self.window,
            negative=self.negative,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )
        if identity is not None:
            self._save_cache(identity, *result)
        return result

    def _save_cache(self, identity, embeddings, has_embedding):
        self.cache_path_.parent.mkdir(parents=True, exist_ok=True)
        # Publish atomically so an interrupted run cannot leave a reusable partial cache.
        with NamedTemporaryFile(
            dir=self.cache_path_.parent, prefix=".building-", suffix=".npz", delete=False
        ) as stream:
            try:
                np.savez(
                    stream,
                    embeddings=embeddings,
                    has_embedding=has_embedding,
                    identity=np.array(_json_hash(identity)),
                )
                stream.close()
                os.replace(stream.name, self.cache_path_)
            except BaseException:
                stream.close()
                os.unlink(stream.name)
                raise

    def _load_cache(self, identity):
        with np.load(self.cache_path_, allow_pickle=False) as data:
            embeddings, has_embedding = data["embeddings"], data["has_embedding"]
            stored_identity = str(data["identity"])
        n_nodes = self._training_graph.shape[0]
        if (
            stored_identity != _json_hash(identity)
            or embeddings.shape != (n_nodes, self.dimensions)
            or has_embedding.shape != (n_nodes,)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(f"Invalid Node2Vec cache at {self.cache_path_}")
        return embeddings, has_embedding

    def _fit(self):
        node_embeddings, node_has_embedding = self._train_or_load()
        n_users, n_items = self.train.shape
        offset = n_users if self.graph == "bipartite" else 0
        embeddings = _normalized(node_embeddings[offset : offset + n_items])
        has_embedding = node_has_embedding[offset : offset + n_items].copy()

        folded_in = np.zeros(n_items, dtype=bool)
        if self._inference_graph is not None:
            senders = self._inference_graph @ sparse.diags(has_embedding.astype(np.float32))
            can_fold_in = (~has_embedding) & (senders.getnnz(axis=1) > 0)
            folded_in[can_fold_in] = True
            embeddings[can_fold_in] = _normalized(senders[can_fold_in] @ embeddings)
            has_embedding |= can_fold_in

        self.embeddings_ = embeddings
        self.has_embedding_ = has_embedding
        held_out = np.setdiff1d(np.arange(n_items), self.train_items)
        saved_with_embedding = self.train @ has_embedding.astype(np.float32)
        has_saves = np.diff(self.train.indptr) > 0
        self.fit_statistics_ = {
            "papers_with_trained_embedding": int(np.count_nonzero(has_embedding & ~folded_in)),
            "papers_folded_in": int(np.count_nonzero(folded_in)),
            "papers_without_embedding": int(np.count_nonzero(~has_embedding)),
            "held_out_papers_without_embedding": int(np.count_nonzero(~has_embedding[held_out])),
            "held_out_papers": len(held_out),
            "users_without_profile_embedding": int(
                np.count_nonzero(has_saves & (saved_with_embedding == 0))
            ),
        }
        # Keep only what scoring needs.
        self._training_graph = self._inference_graph = None

    def score(self, profile, user_index=None):
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        ids = self._item_ids(profile)
        centroid = self.embeddings_[ids].sum(axis=0)
        length = np.linalg.norm(centroid)
        if length:
            centroid /= length
        scores = self.embeddings_ @ centroid
        scores[~self.has_embedding_] = NO_EMBEDDING_SCORE
        return scores
