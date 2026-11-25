import numpy as np
import pytest
from scipy import sparse

from paperrec.data import Paper
from paperrec.models.node2vec import (
    NO_EMBEDDING_SCORE,
    Node2VecRecommender,
    node2vec_walks,
)
from paperrec.splits import Split


def symmetric(edges, n_nodes):
    rows, cols = zip(*edges)
    upper = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(n_nodes, n_nodes)
    )
    return ((upper + upper.T) > 0).astype(np.float32).tocsr()


def walk_args(**overrides):
    return {"num_walks": 2, "walk_length": 6, "p": 1.0, "q": 1.0, "seed": 0, **overrides}


# ---------- walks ----------


def test_walks_follow_edges_and_skip_isolated_nodes():
    graph = symmetric([(0, 1), (1, 2), (2, 3), (3, 0)], 6)  # Nodes 4 and 5 are isolated.
    walks = node2vec_walks(graph, **walk_args(p=0.5, q=2.0))
    assert walks.shape == (2 * 4, 6)
    assert set(np.unique(walks)) == {0, 1, 2, 3}
    assert np.all(np.asarray(graph[walks[:, :-1].ravel(), walks[:, 1:].ravel()]).ravel() == 1)
    # Each round starts once from every connected node.
    assert sorted(walks[:4, 0]) == [0, 1, 2, 3] and sorted(walks[4:, 0]) == [0, 1, 2, 3]


def test_walks_are_reproducible_for_a_seed():
    graph = symmetric([(0, 1), (1, 2), (2, 0), (2, 3)], 4)
    first = node2vec_walks(graph, **walk_args(q=0.5))
    assert np.array_equal(first, node2vec_walks(graph, **walk_args(q=0.5)))
    assert not np.array_equal(first, node2vec_walks(graph, **walk_args(q=0.5, seed=1)))


@pytest.mark.parametrize(("p", "q"), [(1.0, 1.0), (0.25, 4.0), (4.0, 0.25)])
def test_second_order_transitions_match_node2vec_distribution(p, q):
    # After 0 -> 1: node 0 is a return, node 2 is adjacent to 0, node 3 is not.
    graph = symmetric([(0, 1), (1, 2), (0, 2), (1, 3)], 4)
    walks = node2vec_walks(graph, num_walks=3000, walk_length=12, p=p, q=q, seed=3)
    previous, current, following = (
        walks[:, :-2].ravel(),
        walks[:, 1:-1].ravel(),
        walks[:, 2:].ravel(),
    )
    observed = following[(previous == 0) & (current == 1)]
    frequencies = np.bincount(observed, minlength=4) / len(observed)
    weights = np.array([1 / p, 0.0, 1.0, 1 / q])
    assert len(observed) > 2000
    assert frequencies == pytest.approx(weights / weights.sum(), abs=0.03)


@pytest.mark.parametrize(
    "overrides", [{"p": 0.0}, {"q": -1.0}, {"p": np.inf}, {"num_walks": 0}, {"walk_length": 1}]
)
def test_walks_reject_invalid_settings(overrides):
    with pytest.raises(ValueError):
        node2vec_walks(symmetric([(0, 1)], 2), **walk_args(**overrides))


def test_walks_reject_asymmetric_or_empty_graphs():
    with pytest.raises(ValueError, match="symmetric"):
        node2vec_walks(sparse.csr_matrix(np.array([[0, 1.0], [0, 0]])), **walk_args())
    with pytest.raises(ValueError, match="no edges"):
        node2vec_walks(sparse.csr_matrix((2, 2)), **walk_args())


# ---------- recommender ----------

# Train: 0-3 form a square with a chord, 4-7 likewise, joined by nothing; 8 is an
# isolated train paper. Validation: 9 (links to 0), 10 (links only to isolated 8).
# Test: 11 (links to 4), 12 (no links).
N_PAPERS = 13
TRAIN_ITEMS = np.arange(9)
VAL_ITEMS = np.array([9, 10])
TEST_ITEMS = np.array([11, 12])
TRAIN_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (4, 5), (5, 6), (6, 7), (7, 4), (4, 6)]
HELD_OUT_EDGES = [(9, 0), (10, 8), (11, 4)]
PAPERS = [Paper(i, f"title {i}", "abstract") for i in range(N_PAPERS)]
SMALL = {"dimensions": 8, "num_walks": 20, "walk_length": 10, "window": 3, "epochs": 3}


def binary(pairs, shape):
    if not pairs:
        return sparse.csr_matrix(shape, dtype=np.float32)
    rows, cols = zip(*pairs)
    matrix = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=shape)
    matrix.sort_indices()
    return matrix


def cold_split(validation=((1, 9),), test=((1, 11),)):
    shape = (3, N_PAPERS)
    train = binary([(0, 0), (0, 1), (1, 4), (1, 5), (2, 8)], shape)
    return Split(
        train,
        binary(list(validation), shape),
        binary(list(test), shape),
        TRAIN_ITEMS,
        VAL_ITEMS,
        TEST_ITEMS,
        "cold",
        42,
        0,
        5,
        0.1,
    )


def warm_split():
    shape = (3, N_PAPERS)
    train = binary([(0, 0), (0, 1), (1, 4), (1, 5), (2, 8), (2, 9)], shape)
    every = np.arange(N_PAPERS)
    return Split(
        train,
        binary([(0, 2)], shape),
        binary([(1, 6)], shape),
        every,
        np.array([2]),
        np.array([6]),
        "warm",
        42,
        0,
        5,
        0.1,
    )


def fit(split=None, citations=None, **overrides):
    pytest.importorskip("gensim")
    citations = (
        symmetric(TRAIN_EDGES + HELD_OUT_EDGES, N_PAPERS) if citations is None else citations
    )
    config = {"cache_root": None, **SMALL, **overrides}
    return Node2VecRecommender(citations, **config).fit(PAPERS, split or cold_split())


def test_communities_are_separated_for_warm_users():
    model = fit(warm_split(), dimensions=16, num_walks=40, epochs=5)
    scores = model.score([0, 1])
    assert scores[[2, 3]].min() > scores[[5, 6, 7]].max()


def test_cold_paper_embedding_comes_from_train_neighbours_only():
    model = fit()
    assert model.has_embedding_[9] and model.has_embedding_[11]
    np.testing.assert_allclose(model.embeddings_[9], model.embeddings_[0], atol=1e-6)
    np.testing.assert_allclose(model.embeddings_[11], model.embeddings_[4], atol=1e-6)


def test_fold_in_averages_several_neighbours():
    citations = symmetric(TRAIN_EDGES + [(9, 0), (9, 4)], N_PAPERS)
    model = fit(citations=citations)
    expected = model.embeddings_[0] + model.embeddings_[4]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(model.embeddings_[9], expected, atol=1e-6)


def test_papers_without_embedded_neighbours_rank_last_and_are_counted():
    model = fit()
    # Paper 8 trained nothing (isolated), so 10 has no embedded neighbour; 12 has no links.
    assert not model.has_embedding_[[8, 10, 12]].any()
    scores = model.score([0])
    assert (scores[[8, 10, 12]] == NO_EMBEDDING_SCORE).all()
    assert scores[model.has_embedding_].min() >= -1.000001
    ranking = [item for item, _ in model.recommend([0], k=4, candidates=[9, 10, 11, 12])]
    assert set(ranking[-2:]) == {10, 12}
    assert model.fit_statistics_ == {
        "papers_with_trained_embedding": 8,
        "papers_folded_in": 2,
        "papers_without_embedding": 3,
        "held_out_papers_without_embedding": 2,
        "held_out_papers": 4,
        "users_without_profile_embedding": 1,  # User 2 saved only isolated paper 8.
    }


def test_user_without_profile_embedding_still_gets_finite_scores():
    scores = fit().score([8])
    assert np.isfinite(scores).all()


def test_held_out_papers_do_not_change_train_embeddings():
    base = fit()
    extra = symmetric(TRAIN_EDGES + HELD_OUT_EDGES + [(9, 11), (9, 12), (10, 12), (9, 1)], N_PAPERS)
    changed = fit(citations=extra)
    np.testing.assert_array_equal(base.embeddings_[TRAIN_ITEMS], changed.embeddings_[TRAIN_ITEMS])
    # A held-out to held-out edge adds nothing; (9, 1) is a real train link and does.
    assert not changed.has_embedding_[12]
    assert not np.array_equal(base.embeddings_[9], changed.embeddings_[9])


def test_held_out_labels_cannot_change_embeddings():
    base = fit()
    other = fit(cold_split(validation=((0, 10), (2, 9)), test=((0, 12),)))
    np.testing.assert_array_equal(base.embeddings_, other.embeddings_)


def test_bipartite_warm_embeds_only_training_saves():
    model = fit(warm_split(), graph="bipartite")
    saved = {0, 1, 4, 5, 8, 9}
    assert set(np.flatnonzero(model.has_embedding_)) == saved
    assert model.fit_statistics_["papers_folded_in"] == 0
    # Paper 2 is user 0's validation save, so it must not gain an embedding.
    assert model.score([0])[2] == NO_EMBEDDING_SCORE


def test_bipartite_warm_ignores_validation_and_test_saves():
    base = fit(warm_split(), graph="bipartite")
    split = warm_split()
    split.validation, split.test = split.test, split.validation
    other = fit(split, graph="bipartite")
    np.testing.assert_array_equal(base.embeddings_, other.embeddings_)


def test_bipartite_is_rejected_for_cold():
    with pytest.raises(ValueError, match="cold"):
        fit(graph="bipartite")


def test_citation_graph_requires_citations():
    with pytest.raises(ValueError, match="citations"):
        Node2VecRecommender(None)


def test_recommend_ranks_every_candidate_once():
    candidates = [9, 10, 11, 12]
    ranking = fit().recommend([0, 1], k=4, candidates=candidates)
    assert sorted(item for item, _ in ranking) == candidates


def test_training_is_reproducible_with_one_worker():
    np.testing.assert_array_equal(fit().embeddings_, fit().embeddings_)


def test_cache_is_reused_and_separated_by_graph_and_settings(tmp_path):
    first = fit(cache_root=tmp_path)
    second = fit(cache_root=tmp_path)
    assert (first.cache_hit_, second.cache_hit_) == (False, True)
    np.testing.assert_array_equal(first.embeddings_, second.embeddings_)
    assert not fit(cache_root=tmp_path, q=2.0).cache_hit_
    other_graph = symmetric(TRAIN_EDGES[:-1] + HELD_OUT_EDGES, N_PAPERS)
    assert not fit(cache_root=tmp_path, citations=other_graph).cache_hit_
    assert not list(tmp_path.glob(".building-*"))


def test_invalid_cache_raises(tmp_path):
    first = fit(cache_root=tmp_path)
    np.savez(
        first.cache_path_,
        embeddings=np.zeros((1, 1)),
        has_embedding=np.zeros(1, bool),
        identity=np.array("wrong"),
    )
    with pytest.raises(ValueError, match="Invalid Node2Vec cache"):
        fit(cache_root=tmp_path)


def test_config_is_json_ready_and_records_settings():
    import json

    config = Node2VecRecommender(symmetric([(0, 1)], 2), p=0.5, q=2).get_config()
    assert json.loads(json.dumps(config))["p"] == 0.5
    assert config["graph"] == "citation"
