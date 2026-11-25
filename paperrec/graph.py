"""Leakage-safe graphs for the WP3 models (Node2Vec, Simplified PR-SGCN).

Requires only NumPy and SciPy. Every adjacency is a binary float32 CSR matrix.

Paper-paper citation graph (primary; the only graph signal usable when cold):
* Warm: all citation edges in training and inference. Citations are metadata
  and are not derived from saves, so no interaction label enters the graph.
* Cold training: only edges with BOTH endpoints in train_items.
* Cold inference: rows RECEIVE messages from columns, so the matrix is
  deliberately asymmetric. Held-out papers receive from train papers; they send
  to nobody. Train embeddings therefore do not depend on held-out papers, and
  validation and test papers cannot influence each other. This is why a single
  inference graph serves both partitions, although score() is unaware of them.
* Edges between two held-out papers are excluded. include_heldout_edges=True
  adds within-partition edges for the ablation study; validation-test edges
  are never added.

User-paper bipartite graph: built from split.train only. Warm comparison only,
because held-out cold papers have no edges in it.

item-tag.dat is intentionally unused: tags are created by users saving papers,
so a paper nobody has saved tends to have none.

Reporting caveat: citations.dat is symmetric and undated, so a held-out paper's
links may include papers that cited it later. Truly new papers lack those links,
which makes citation-based cold scores optimistic. State this assumption when
reporting, and do not reuse these edges for the arXiv demo.

SPECTER2+GCN in this project is a "Simplified PR-SGCN" adapted to CiteULike-a.
It is NOT a full reproduction of PR-SGCN (Guo et al., 2025): this dataset has no
paper-author network or timestamps for that method's heterogeneous and temporal
components.

CLI (from the project root):
    python -m paperrec.graph --scenario both
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse

from .splits import Split, _hash_matrix


def _require_citations(citations, split: Split) -> sparse.csr_matrix:
    if citations is None:
        raise ValueError("Dataset has no citations; citations.dat was not found")
    n_items = split.train.shape[1]
    if not sparse.isspmatrix_csr(citations) or citations.shape != (n_items, n_items):
        raise ValueError("Citations must be CSR with one row and column per paper")
    if citations.diagonal().any():
        raise ValueError("Citations must not contain self-loops")
    if (citations != citations.T).nnz:
        raise ValueError("Citations must be symmetric")
    return citations


def _cold_groups(split: Split) -> np.ndarray:
    """Label each paper 0=train, 1=validation, 2=test."""
    groups = np.full(split.train.shape[1], -1, dtype=np.int8)
    groups[split.train_items] = 0
    groups[split.validation_items] = 1
    groups[split.test_items] = 2
    if np.any(groups < 0):
        raise ValueError("Cold item pools must cover every paper")
    return groups


def _keep_edges(matrix: sparse.csr_matrix, keep_rule) -> sparse.csr_matrix:
    coo = matrix.tocoo()
    keep = keep_rule(coo.row, coo.col)
    result = sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=matrix.shape, dtype=np.float32
    )
    result.sort_indices()
    return result


def _block_nnz(graph: sparse.csr_matrix, receivers: np.ndarray, senders: np.ndarray) -> int:
    return graph[receivers][:, senders].nnz


def validate_citation_graphs(
    training, inference, citations, split: Split, include_heldout_edges=False
):
    """Raise if a citation graph breaks the visibility rules for this split."""
    for graph in (training, inference):
        if (graph - graph.multiply(citations)).nnz:
            raise ValueError("Graph contains an edge that is not a citation")
    if split.scenario == "warm":
        if (training != citations).nnz or (inference != citations).nnz:
            raise ValueError("Warm graphs must contain every citation edge")
        return
    train, validation, test = split.train_items, split.validation_items, split.test_items
    heldout = np.concatenate((validation, test))
    if _block_nnz(training, heldout, np.arange(training.shape[1])):
        raise ValueError("Cold training graph touches a held-out paper")
    if _block_nnz(training, np.arange(training.shape[1]), heldout):
        raise ValueError("Cold training graph touches a held-out paper")
    if _block_nnz(inference, train, heldout):
        raise ValueError("Train papers must not receive messages from held-out papers")
    if _block_nnz(inference, validation, test) or _block_nnz(inference, test, validation):
        raise ValueError("Validation and test papers must not exchange messages")
    if not include_heldout_edges and (
        _block_nnz(inference, validation, validation) or _block_nnz(inference, test, test)
    ):
        raise ValueError("Strict inference graph contains a held-out to held-out edge")


def citation_training_graph(citations, split: Split) -> sparse.csr_matrix:
    """Symmetric citation graph that a model may train on."""
    citations = _require_citations(citations, split)
    if split.scenario == "warm":
        return citations.copy()
    groups = _cold_groups(split)
    return _keep_edges(citations, lambda r, c: (groups[r] == 0) & (groups[c] == 0))


def citation_inference_graph(
    citations, split: Split, *, include_heldout_edges=False
) -> sparse.csr_matrix:
    """Message-passing graph for scoring: row i receives from column j.

    Asymmetric when cold. Algorithms that assume symmetry, including symmetric
    GCN normalization, must account for this.
    """
    citations = _require_citations(citations, split)
    if split.scenario == "warm":
        return citations.copy()
    groups = _cold_groups(split)
    if include_heldout_edges:
        return _keep_edges(citations, lambda r, c: (groups[c] == 0) | (groups[r] == groups[c]))
    return _keep_edges(citations, lambda r, c: groups[c] == 0)


def bipartite_graph(split: Split) -> sparse.csr_matrix:
    """Symmetric user-paper adjacency from split.train.

    Nodes are users 0..n_users-1, then papers offset by n_users. Validation and
    test saves are never read.
    """
    n_users, n_items = split.train.shape
    held_out_items = np.setdiff1d(np.arange(n_items), split.train_items)
    if np.isin(split.train.indices, held_out_items).any():
        raise ValueError("split.train contains a paper outside train_items")
    return sparse.bmat(
        [
            [sparse.csr_matrix((n_users, n_users), dtype=np.float32), split.train],
            [split.train.T, sparse.csr_matrix((n_items, n_items), dtype=np.float32)],
        ],
        format="csr",
        dtype=np.float32,
    )


def validate_bipartite_graph(graph, split: Split):
    """Raise unless the graph holds exactly the training saves, in both directions."""
    n_users = split.train.shape[0]
    if (graph[:n_users][:, n_users:] != split.train).nnz:
        raise ValueError("Bipartite graph must contain exactly the training saves")
    if (graph != graph.T).nnz:
        raise ValueError("Bipartite graph must be symmetric")
    if graph[:n_users][:, :n_users].nnz or graph[n_users:][:, n_users:].nnz:
        raise ValueError("Bipartite graph must only link users to papers")


def drop_edges(adjacency, rate, seed) -> sparse.csr_matrix:
    """Remove each undirected edge with probability rate (edge dropout ablation).

    Ignores saves, so it adds no label information. Pass the result to the citation
    graph functions above, which still apply the warm/cold visibility rules.
    """
    if not 0 <= rate <= 1:
        raise ValueError("Edge dropout rate must be between 0 and 1")
    upper = sparse.triu(adjacency, k=1, format="coo")
    keep = np.random.default_rng(seed).random(upper.nnz) >= rate
    kept = sparse.csr_matrix(
        (upper.data[keep], (upper.row[keep], upper.col[keep])),
        shape=adjacency.shape,
        dtype=np.float32,
    )
    result = (kept + kept.T).tocsr()
    result.sort_indices()
    return result


def knn_graph(features, k, train_items, batch_size=1024) -> sparse.csr_matrix:
    """Symmetric union of each paper's k most cosine-similar TRAIN papers.

    Uses paper text features only, so no save enters the graph. Neighbours are always
    train papers, so train-to-train edges never depend on held-out text. Pass the
    result to the citation graph functions above, which make it one-way when cold.
    """
    n_items = features.shape[0]
    train_items = np.asarray(train_items)
    if not 1 <= k < len(train_items):
        raise ValueError("k must be at least 1 and below the number of train papers")
    vectors = np.asarray(features, dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    senders = vectors[train_items].T
    train_position = np.full(n_items, -1, dtype=np.int64)
    train_position[train_items] = np.arange(len(train_items))

    rows, columns = [], []
    for start in range(0, n_items, batch_size):
        batch = np.arange(start, min(start + batch_size, n_items))
        similarity = vectors[batch] @ senders
        is_train = train_position[batch] >= 0
        similarity[is_train, train_position[batch][is_train]] = -np.inf  # Never itself.
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        rows.append(np.repeat(batch, k))
        columns.append(train_items[nearest].ravel())
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    directed = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(n_items, n_items)
    )
    result = ((directed + directed.T) > 0).astype(np.float32).tocsr()
    result.sort_indices()
    return result


def normalized_message_matrix(graph, mode="symmetric") -> sparse.csr_matrix:
    """GCN propagation matrix: weight(i <- j) = 1 / sqrt(d_i * d_j) with self-loops.

    mode="row" uses weight(i <- j) = 1 / d_i instead, so each row is a plain mean of the
    paper and its senders. Both modes leave train rows unchanged between graphs.

    d_i = 1 + the number of messages node i RECEIVES (its row count). On a symmetric
    graph this is exactly the Kipf & Welling normalization.

    On the one-way cold inference graph, counting a train paper's links to held-out
    papers would change train-to-train weights, so they would differ from training
    and depend on the held-out papers. A train paper receives only from train papers,
    so counting received messages keeps its row identical to the training graph.
    Library GCN layers normalize from the edge list and assume one degree per node,
    which is why this project writes the GCN in plain PyTorch around this matrix.
    """
    if mode not in ("symmetric", "row"):
        raise ValueError('mode must be "symmetric" or "row"')
    n_nodes = graph.shape[0]
    if mode == "row":
        inverse = 1 / (1 + graph.getnnz(axis=1)).astype(np.float32)
        scale = sparse.diags(inverse, format="csr", dtype=np.float32)
        result = (scale @ (graph + sparse.eye(n_nodes, format="csr", dtype=np.float32))).tocsr()
        result = result.astype(np.float32)
        result.sort_indices()
        return result
    inverse_root = 1 / np.sqrt(1 + graph.getnnz(axis=1)).astype(np.float32)
    scale = sparse.diags(inverse_root, format="csr", dtype=np.float32)
    with_loops = graph + sparse.eye(n_nodes, format="csr", dtype=np.float32)
    result = (scale @ with_loops @ scale).tocsr().astype(np.float32)
    result.sort_indices()
    return result


def graph_fingerprint(graph) -> str:
    """Identify a graph in embedding cache keys.

    dataset_fingerprint does not cover citations, so caches need this as well.
    """
    digest = hashlib.sha256(b"paperrec-graph-v1")
    _hash_matrix(digest, graph)
    return digest.hexdigest()


def graph_statistics(citations, split: Split) -> dict:
    """Check all graphs for one split and summarize them."""
    training = citation_training_graph(citations, split)
    inference = citation_inference_graph(citations, split)
    validate_citation_graphs(training, inference, citations, split)
    bipartite = bipartite_graph(split)
    validate_bipartite_graph(bipartite, split)
    has_sender = np.diff(inference.indptr) > 0
    result = {
        "scenario": split.scenario,
        "fold": split.fold,
        "training_edges": training.nnz // 2,
        "inference_messages": inference.nnz,
        "papers_without_messages": int(np.count_nonzero(~has_sender)),
        "bipartite_edges": bipartite.nnz // 2,
    }
    if split.scenario == "cold":
        heldout = np.concatenate((split.validation_items, split.test_items))
        result["heldout_papers_without_messages"] = float(np.mean(~has_sender[heldout]))
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from .data import load_citeulike_a
    from .splits import load_folds

    parser = argparse.ArgumentParser(description="Validate and summarize graphs on saved folds")
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--scenario", choices=("warm", "cold", "both"), default="both")
    args = parser.parse_args()
    dataset = load_citeulike_a(args.data)
    for scenario in ("warm", "cold") if args.scenario == "both" else (args.scenario,):
        for fold in load_folds(args.splits / scenario, dataset):
            print(json.dumps(graph_statistics(dataset.citations, fold)), flush=True)
