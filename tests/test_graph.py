import numpy as np
import pytest
from scipy import sparse

from paperrec.data import load_citations, load_citeulike_a
from paperrec.graph import (
    bipartite_graph,
    citation_inference_graph,
    citation_training_graph,
    drop_edges,
    graph_fingerprint,
    graph_statistics,
    knn_graph,
    validate_bipartite_graph,
    validate_citation_graphs,
)
from paperrec.splits import Split

# Papers 0-2 train, 3-4 validation, 5-6 test.
TRAIN_ITEMS, VAL_ITEMS, TEST_ITEMS = [0, 1, 2], [3, 4], [5, 6]
EDGES = {
    "train_train": (0, 1),
    "val_train": (3, 0),
    "test_train": (5, 1),
    "val_val": (3, 4),
    "test_test": (5, 6),
    "val_test": (4, 6),
}


def citation_matrix():
    rows, cols = zip(*EDGES.values())
    upper = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(7, 7))
    return (upper + upper.T).tocsr()


def binary(pairs, shape=(3, 7)):
    rows, cols = zip(*pairs)
    matrix = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=shape)
    matrix.sort_indices()
    return matrix


def make_split(scenario="cold"):
    if scenario == "cold":
        train = binary([(0, 0), (0, 1), (1, 2)])
        validation, test = binary([(0, 3)]), binary([(1, 5)])
        items = TRAIN_ITEMS, VAL_ITEMS, TEST_ITEMS
    else:
        train = binary([(0, 0), (0, 3), (1, 2)])
        validation, test = binary([(1, 3)]), binary([(2, 3)])
        items = list(range(7)), [3], [3]
    return Split(
        train,
        validation,
        test,
        *(np.array(values) for values in items),
        scenario,
        42,
        0,
        5,
        0.1,
    )


def has(graph, receiver, sender):
    return graph[receiver, sender] != 0


def test_load_citations_removes_self_loops_and_keeps_symmetry(tmp_path):
    path = tmp_path / "citations.dat"
    path.write_text("2 1 0\n1 0\n0\n")
    matrix, loops = load_citations(path, 3)
    assert loops == 1
    assert matrix.toarray().tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    "content",
    [
        "1 1\n0\n",  # Asymmetric: 0 lists 1, but 1 does not list 0.
        "2 1\n1 0\n",  # Wrong count.
        "1 5\n0\n",  # ID out of range.
        "\n0\n",  # Blank row.
        "0\n",  # Too few rows.
        "x\n0\n",  # Not an integer.
    ],
)
def test_load_citations_rejects_malformed_files(tmp_path, content):
    path = tmp_path / "citations.dat"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_citations(path, 2)


def test_dataset_citations_are_optional_and_reported(tmp_path):
    (tmp_path / "raw-data.csv").write_text(
        '"doc.id","title","citeulike.id","raw.title","raw.abstract"\n'
        '1,"a",1,"A","x"\n2,"b",2,"B","y"\n3,"c",3,"C","z"\n'
    )
    (tmp_path / "users.dat").write_text("2 0 1\n")
    without = load_citeulike_a(tmp_path)
    assert without.citations is None
    assert "citation_edges" not in without.statistics()

    (tmp_path / "citations.dat").write_text("1 1\n1 0\n1 2\n")
    statistics = load_citeulike_a(tmp_path).statistics()
    assert statistics["citation_edges"] == 1
    assert statistics["citation_isolated_papers"] == 1
    assert statistics["citation_self_loops_removed"] == 1


def test_missing_citations_give_a_clear_error():
    with pytest.raises(ValueError, match="no citations"):
        citation_training_graph(None, make_split())


def test_cold_training_graph_only_links_train_papers():
    graph = citation_training_graph(citation_matrix(), make_split())
    assert sorted(zip(*graph.nonzero())) == [(0, 1), (1, 0)]


def test_cold_inference_is_one_way_from_train_to_heldout():
    graph = citation_inference_graph(citation_matrix(), make_split())
    assert has(graph, 0, 1) and has(graph, 1, 0)
    assert has(graph, 3, 0) and has(graph, 5, 1)
    assert not has(graph, 0, 3) and not has(graph, 1, 5)
    for a, b in (EDGES["val_val"], EDGES["test_test"], EDGES["val_test"]):
        assert not has(graph, a, b) and not has(graph, b, a)


def test_ablation_adds_within_partition_edges_but_never_validation_test():
    graph = citation_inference_graph(citation_matrix(), make_split(), include_heldout_edges=True)
    for a, b in (EDGES["val_val"], EDGES["test_test"]):
        assert has(graph, a, b) and has(graph, b, a)
    a, b = EDGES["val_test"]
    assert not has(graph, a, b) and not has(graph, b, a)
    assert not has(graph, 0, 3)


def test_warm_uses_every_citation():
    citations, split = citation_matrix(), make_split("warm")
    assert (citation_training_graph(citations, split) != citations).nnz == 0
    assert (citation_inference_graph(citations, split) != citations).nnz == 0


def test_generated_graphs_pass_validation_including_ablation():
    citations, split = citation_matrix(), make_split()
    training = citation_training_graph(citations, split)
    ablation = citation_inference_graph(citations, split, include_heldout_edges=True)
    validate_citation_graphs(training, ablation, citations, split, include_heldout_edges=True)
    with pytest.raises(ValueError, match="held-out to held-out"):
        validate_citation_graphs(training, ablation, citations, split)


@pytest.mark.parametrize(
    ("graph_name", "edge", "message"),
    [
        ("training", (3, 0), "training graph touches"),
        ("training", (0, 3), "training graph touches"),
        ("inference", (0, 3), "must not receive"),
        ("inference", (4, 6), "must not exchange"),
        ("inference", (3, 4), "held-out to held-out"),
    ],
)
def test_validation_detects_injected_leaks(graph_name, edge, message):
    citations, split = citation_matrix(), make_split()
    graphs = {
        "training": citation_training_graph(citations, split).tolil(),
        "inference": citation_inference_graph(citations, split).tolil(),
    }
    graphs[graph_name][edge] = 1
    with pytest.raises(ValueError, match=message):
        validate_citation_graphs(
            graphs["training"].tocsr(), graphs["inference"].tocsr(), citations, split
        )


def test_validation_rejects_edges_that_are_not_citations():
    citations, split = citation_matrix(), make_split()
    training = citation_training_graph(citations, split).tolil()
    training[0, 2] = 1
    inference = citation_inference_graph(citations, split)
    with pytest.raises(ValueError, match="not a citation"):
        validate_citation_graphs(training.tocsr(), inference, citations, split)


@pytest.mark.parametrize("scenario", ["warm", "cold"])
def test_bipartite_uses_only_training_saves(scenario):
    split = make_split(scenario)
    graph = bipartite_graph(split)
    validate_bipartite_graph(graph, split)
    n_users = split.train.shape[0]
    assert graph.shape == (10, 10)
    assert graph.nnz == 2 * split.train.nnz
    for matrix in (split.validation, split.test):
        for user, item in zip(*matrix.nonzero()):
            assert graph[user, n_users + item] == 0


def test_bipartite_validation_detects_a_held_out_save():
    split = make_split()
    graph = bipartite_graph(split).tolil()
    graph[0, 3 + 3] = graph[3 + 3, 0] = 1  # User 0 saved paper 3 in validation.
    with pytest.raises(ValueError, match="exactly the training saves"):
        validate_bipartite_graph(graph.tocsr(), split)


def test_bipartite_rejects_saves_on_heldout_papers():
    split = make_split()
    split.train = binary([(0, 0), (0, 3)])
    with pytest.raises(ValueError, match="outside train_items"):
        bipartite_graph(split)


def test_statistics_on_small_cold_split():
    statistics = graph_statistics(citation_matrix(), make_split())
    assert statistics["training_edges"] == 1
    assert statistics["inference_messages"] == 4
    assert statistics["bipartite_edges"] == 3
    assert statistics["heldout_papers_without_messages"] == 0.5


def test_fingerprint_changes_with_the_graph():
    citations, split = citation_matrix(), make_split()
    training = citation_training_graph(citations, split)
    assert graph_fingerprint(citations) == graph_fingerprint(citations.copy())
    assert graph_fingerprint(citations) != graph_fingerprint(training)


# ---------- ablation graphs ----------


def _ablation_edges():
    rows, cols = zip(*[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6), (1, 5)])
    upper = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(7, 7))
    return (upper + upper.T).tocsr()


def test_drop_edges_keeps_a_symmetric_subset_and_is_reproducible():
    edges = _ablation_edges()
    dropped = drop_edges(edges, 0.5, seed=1)
    assert (dropped != dropped.T).nnz == 0
    assert (dropped - dropped.multiply(edges)).nnz == 0
    assert (dropped != drop_edges(edges, 0.5, seed=1)).nnz == 0


def test_drop_edges_limits():
    edges = _ablation_edges()
    assert (drop_edges(edges, 0.0, seed=1) != edges).nnz == 0
    assert drop_edges(edges, 1.0, seed=1).nnz == 0
    with pytest.raises(ValueError):
        drop_edges(edges, 1.5, seed=1)


def _knn_features():
    return np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32)


def test_knn_graph_links_each_paper_to_its_nearest_train_papers():
    features = _knn_features()
    train = np.array([0, 1, 2, 3])
    graph = knn_graph(features, 2, train, batch_size=3)
    assert (graph != graph.T).nnz == 0
    assert graph.diagonal().sum() == 0
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity = unit @ unit[train].T
    for paper in range(7):
        if paper in train:
            similarity[paper, paper] = -np.inf
        for neighbour in train[np.argsort(-similarity[paper])[:2]]:
            assert graph[paper, neighbour] == 1


def test_knn_graph_never_links_two_held_out_papers():
    graph = knn_graph(_knn_features(), 2, np.array([0, 1, 2, 3]))
    assert graph[4:][:, 4:].nnz == 0


def test_knn_train_edges_ignore_held_out_text():
    features = _knn_features()
    train = np.array([0, 1, 2, 3])
    changed = features.copy()
    changed[4:] = np.random.default_rng(5).normal(size=(3, 5))
    base = knn_graph(features, 2, train)[train][:, train]
    other = knn_graph(changed, 2, train)[train][:, train]
    assert (base != other).nnz == 0


def test_knn_graph_rejects_invalid_k():
    with pytest.raises(ValueError):
        knn_graph(_knn_features(), 4, np.array([0, 1, 2, 3]))
