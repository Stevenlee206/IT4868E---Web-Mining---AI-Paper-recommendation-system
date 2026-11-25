"""Simplified PR-SGCN: SPECTER2 features propagated by a GCN over citations.

This is an adaptation to CiteULike-a, NOT a full reproduction of PR-SGCN (Guo et
al., 2025). The dataset lacks the paper-author network and timestamps used by that
method's heterogeneous and temporal components. Only SPECTER2 features, a GCN on
the citation graph and a margin-based loss are kept.

Why plain PyTorch instead of PyTorch Geometric: the cold inference graph is one-way
and needs the custom normalization in graph.normalized_message_matrix, which keeps
train-to-train weights identical between training and inference. Writing the layer
around that explicit matrix makes the rule easy to inspect, test and report.

Ablation options: residual keeps each paper's own features beside the propagated
ones, so "concat" can express the no-graph model; edge_dropout removes citation edges;
graph_source="knn" swaps citations for SPECTER2 nearest neighbours.

Training never reads held-out saves. It optimizes leave-one-out triplets: a saved
paper must score above sampled negatives against the mean of the user's OTHER saved
papers. Negatives come from training_mask(), so warm held-out pairs and cold held-out
papers are never treated as negatives. There is no early stopping; select settings
on validation, then run test once with those fixed settings.

This module imports torch, so it is not exported from paperrec.models:
    from paperrec.models.gcn import SimplifiedPRSGCNRecommender
"""

from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import nn

from ..graph import (
    citation_inference_graph,
    citation_training_graph,
    drop_edges,
    knn_graph,
    normalized_message_matrix,
    validate_citation_graphs,
)
from .base import Recommender
from .node2vec import _normalized


def _to_torch_sparse(matrix) -> torch.Tensor:
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
        torch.from_numpy(coo.data.astype(np.float32)),
        coo.shape,
        check_invariants=True,
    ).coalesce()


class GCN(nn.Module):
    """Stack of H' = A H W layers, with L2-normalized output; A may be asymmetric.

    residual="add" feeds H + A H to each layer; residual="concat" feeds [H, A H].
    normalize_propagated=True L2-normalizes each row of A H first.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        residual=None,
        normalize_propagated=False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        width = 2 if residual == "concat" else 1
        self.layers = nn.ModuleList(nn.Linear(width * a, b) for a, b in pairwise(sizes))
        self.dropout = dropout
        self.residual = residual
        self.normalize_propagated = normalize_propagated

    def forward(self, features, propagation):
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = F.dropout(hidden, self.dropout, self.training)
            propagated = torch.sparse.mm(propagation, hidden)
            if self.normalize_propagated:
                propagated = F.normalize(propagated, dim=1)
            if self.residual == "add":
                propagated = hidden + propagated
            elif self.residual == "concat":
                propagated = torch.cat((hidden, propagated), dim=1)
            hidden = layer(propagated)
            if index < len(self.layers) - 1:
                hidden = F.relu(hidden)
        return F.normalize(hidden, dim=1)


def ranking_loss(positive, negative, *, loss, margin, scale):
    """Loss on cosine scores with shapes (batch,) and (batch, negatives).

    margin: mean of max(0, margin - positive + negative). bce: cross-entropy on
    scale * cosine, with positives labelled 1 and negatives labelled 0.
    """
    if loss == "margin":
        return F.relu(margin - positive.unsqueeze(1) + negative).mean()
    return F.binary_cross_entropy_with_logits(
        scale * positive, torch.ones_like(positive)
    ) + F.binary_cross_entropy_with_logits(scale * negative, torch.zeros_like(negative))


class _FeatureRecommender(Recommender):
    """Rank by cosine similarity to the normalized mean of saved paper embeddings."""

    def score(self, profile, user_index=None):
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        centroid = self.embeddings_[self._item_ids(profile)].sum(axis=0)
        length = np.linalg.norm(centroid)
        if length:
            centroid /= length
        return self.embeddings_ @ centroid


class Specter2Recommender(_FeatureRecommender):
    """Raw SPECTER2 reference: shows what the GCN adds beyond the features."""

    def __init__(self, features, *, seed=42):
        super().__init__(seed=seed)
        self.features = features

    def get_config(self):
        return {
            **super().get_config(),
            "method": "specter2",
            "feature_shape": list(self.features.shape),
        }

    def _fit(self):
        self.embeddings_ = _normalized(np.asarray(self.features, dtype=np.float32))


class SimplifiedPRSGCNRecommender(_FeatureRecommender):
    """GCN over SPECTER2 features. use_graph=False keeps self-loops only, with no citations."""

    def __init__(
        self,
        features,
        citations=None,
        *,
        use_graph=True,
        include_heldout_edges=False,
        residual=None,
        input_norm=None,
        graph_source="citation",
        knn_k=10,
        edge_dropout=0.0,
        hidden_size=256,
        output_size=128,
        num_layers=1,
        dropout=0.1,
        loss="margin",
        margin=0.2,
        scale=10.0,
        learning_rate=1e-3,
        weight_decay=1e-5,
        epochs=50,
        batch_users=512,
        num_negatives=5,
        max_profile_size=50,
        seed=42,
    ):
        super().__init__(seed=seed)
        if loss not in ("margin", "bce"):
            raise ValueError('loss must be "margin" or "bce"')
        if residual not in (None, "add", "concat"):
            raise ValueError('residual must be None, "add" or "concat"')
        if input_norm not in (None, "l2", "row"):
            raise ValueError('input_norm must be None, "l2" or "row"')
        if graph_source not in ("citation", "knn"):
            raise ValueError('graph_source must be "citation" or "knn"')
        if use_graph and graph_source == "citation" and citations is None:
            raise ValueError("The citation GCN needs dataset.citations")
        if not use_graph and (include_heldout_edges or residual or input_norm or edge_dropout):
            raise ValueError("Graph options require use_graph=True")
        if include_heldout_edges and graph_source == "knn":
            raise ValueError("The kNN graph has no edges between held-out papers")
        self.features = features
        self.citations = citations
        self.use_graph = use_graph
        self.include_heldout_edges = include_heldout_edges
        self.residual = residual
        self.input_norm = input_norm
        self.graph_source = graph_source
        self.knn_k = knn_k
        self.edge_dropout = edge_dropout
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.loss = loss
        self.margin = margin
        self.scale = scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_users = batch_users
        self.num_negatives = num_negatives
        self.max_profile_size = max_profile_size

    def get_config(self):
        return {
            **super().get_config(),
            "method": "simplified_pr_sgcn",
            "use_graph": self.use_graph,
            "include_heldout_edges": self.include_heldout_edges,
            "residual": self.residual,
            "input_norm": self.input_norm,
            "graph_source": self.graph_source,
            "knn_k": self.knn_k,
            "edge_dropout": self.edge_dropout,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "loss": self.loss,
            "margin": self.margin,
            "scale": self.scale,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_users": self.batch_users,
            "num_negatives": self.num_negatives,
            "max_profile_size": self.max_profile_size,
        }

    def fit(self, papers, split):
        # Build graphs here because _fit must not receive held-out labels via the split.
        n_items = split.train.shape[1]
        if self.features.shape[0] != n_items:
            raise ValueError("Features need one row per paper")
        if self.use_graph:
            edges = self.citations
            if self.graph_source == "knn":
                edges = knn_graph(self.features, self.knn_k, split.train_items)
            if self.edge_dropout:
                edges = drop_edges(edges, self.edge_dropout, self.seed)
            self._training_graph = citation_training_graph(edges, split)
            self._inference_graph = citation_inference_graph(
                edges, split, include_heldout_edges=self.include_heldout_edges
            )
            validate_citation_graphs(
                self._training_graph,
                self._inference_graph,
                edges,
                split,
                include_heldout_edges=self.include_heldout_edges,
            )
        return super().fit(papers, split)

    def _propagation_matrices(self):
        if not self.use_graph:
            identity = _to_torch_sparse(
                sparse.eye(len(self.papers), format="csr", dtype=np.float32)
            )
            return identity, identity
        mode = "row" if self.input_norm == "row" else "symmetric"
        return (
            _to_torch_sparse(normalized_message_matrix(self._training_graph, mode)),
            _to_torch_sparse(normalized_message_matrix(self._inference_graph, mode)),
        )

    def _sample_triplets(self, users, rng):
        """Return (profile indices, profile offsets, positives, negatives) for users."""
        allowed = self.training_mask(users) & (self.train[users].toarray() == 0)
        profile_items, offsets, positives = [], [], []
        for user in users:
            saved = self.train[user].indices
            target = rng.integers(len(saved))
            others = np.delete(saved, target)
            if len(others) > self.max_profile_size:
                others = rng.choice(others, self.max_profile_size, replace=False)
            offsets.append(sum(len(items) for items in profile_items))
            profile_items.append(others)
            positives.append(saved[target])

        pools = [np.flatnonzero(row) for row in allowed]
        negatives = np.stack(
            [
                rng.choice(pool, self.num_negatives, replace=len(pool) < self.num_negatives)
                for pool in pools
            ]
        )
        return (
            torch.from_numpy(np.concatenate(profile_items).astype(np.int64)),
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(np.array(positives), dtype=torch.int64),
            torch.from_numpy(negatives.astype(np.int64)),
        )

    def _fit(self):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        features = torch.from_numpy(_normalized(np.asarray(self.features, dtype=np.float32)))
        training_matrix, inference_matrix = self._propagation_matrices()
        model = GCN(
            features.shape[1],
            self.hidden_size,
            self.output_size,
            self.num_layers,
            self.dropout,
            self.residual,
            self.input_norm == "l2",
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # A user needs one saved paper as target and at least one as profile.
        trainable_users = np.flatnonzero(np.diff(self.train.indptr) >= 2)
        if not len(trainable_users):
            raise ValueError("Training needs a user with at least two saved papers")

        epoch_losses = []
        model.train()
        for _ in range(self.epochs):
            total, steps = 0.0, 0
            permuted = rng.permutation(trainable_users)
            for start in range(0, len(permuted), self.batch_users):
                users = permuted[start : start + self.batch_users]
                indices, offsets, positives, negatives = self._sample_triplets(users, rng)
                embeddings = model(features, training_matrix)
                profiles = F.normalize(
                    F.embedding_bag(indices, embeddings, offsets, mode="mean"), dim=1
                )
                positive_scores = (profiles * embeddings[positives]).sum(dim=1)
                negative_scores = torch.einsum("bd,bkd->bk", profiles, embeddings[negatives])
                loss = ranking_loss(
                    positive_scores,
                    negative_scores,
                    loss=self.loss,
                    margin=self.margin,
                    scale=self.scale,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
            epoch_losses.append(total / steps)

        self.network_ = model
        model.eval()
        with torch.no_grad():
            self.embeddings_ = model(features, inference_matrix).numpy().astype(np.float32)
        self.fit_statistics_ = {
            "trainable_users": len(trainable_users),
            "first_epoch_loss": epoch_losses[0] if epoch_losses else None,
            "last_epoch_loss": epoch_losses[-1] if epoch_losses else None,
        }
        # Keep only what scoring needs.
        self._training_graph = self._inference_graph = None
