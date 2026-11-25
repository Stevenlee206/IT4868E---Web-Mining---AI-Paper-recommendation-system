import numpy as np
import pytest
from scipy import sparse

from paperrec.analyze_degree import degree_breakdown, degree_buckets
from paperrec.data import Paper
from paperrec.graph import (
    citation_inference_graph,
    citation_training_graph,
    normalized_message_matrix,
)
from paperrec.specter import DIMENSIONS, specter2_features
from paperrec.splits import Split

torch = pytest.importorskip("torch")

from paperrec.models.gcn import (
    GCN,
    SimplifiedPRSGCNRecommender,
    Specter2Recommender,
    _to_torch_sparse,
    ranking_loss,
)

N_PAPERS = 12
PAPERS = [Paper(i, f"title {i}", "abstract " * (i + 1)) for i in range(N_PAPERS)]
TRAIN_ITEMS, VAL_ITEMS, TEST_ITEMS = np.arange(8), np.array([8, 9]), np.array([10, 11])
EDGES = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 0), (10, 4), (8, 9), (10, 11), (9, 11)]
# Two layers, so that leakage tests also cover messages passed over two hops.
SMALL = {
    "hidden_size": 8,
    "output_size": 4,
    "num_layers": 2,
    "epochs": 2,
    "batch_users": 2,
    "num_negatives": 2,
}


def symmetric(edges=EDGES, n=N_PAPERS):
    rows, cols = zip(*edges)
    upper = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    result = ((upper + upper.T) > 0).astype(np.float32).tocsr()
    result.sort_indices()
    return result


def binary(pairs):
    rows, cols = zip(*pairs)
    matrix = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(3, N_PAPERS))
    matrix.sort_indices()
    return matrix


def cold_split(validation=((0, 8),), test=((1, 10),)):
    return Split(
        binary([(0, 0), (0, 1), (0, 2), (1, 4), (1, 5), (2, 7)]),
        binary(list(validation)),
        binary(list(test)),
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
    validation_pairs = np.zeros((3, N_PAPERS), np.uint8)
    test_pairs = np.zeros((3, N_PAPERS), np.uint8)
    validation_pairs[0, 3] = validation_pairs[0, 8] = 1  # Saved 3; unobserved pair 8.
    test_pairs[0, 9] = test_pairs[1, 6] = 1  # Unobserved pair 9; saved 6.
    return Split(
        binary([(0, 0), (0, 1), (0, 2), (1, 4), (1, 5), (2, 7)]),
        binary([(0, 3)]),
        binary([(1, 6)]),
        np.arange(N_PAPERS),
        np.array([3, 8]),
        np.array([6, 9]),
        "warm",
        42,
        0,
        5,
        0.1,
        np.packbits(validation_pairs, axis=1, bitorder="little"),
        np.packbits(test_pairs, axis=1, bitorder="little"),
    )


def features():
    return np.random.default_rng(0).normal(size=(N_PAPERS, 16)).astype(np.float32)


def fit(split=None, citations=None, **overrides):
    citations = symmetric() if citations is None else citations
    model = SimplifiedPRSGCNRecommender(features(), citations, **{**SMALL, **overrides})
    return model.fit(PAPERS, split or cold_split())


# ---------- normalization ----------


def test_normalization_matches_standard_gcn_on_symmetric_graphs():
    graph = symmetric([(0, 1), (1, 2)], 4)
    expected_degree = np.array([2, 3, 2, 1], float)
    dense = (graph.toarray() + np.eye(4)) / np.sqrt(np.outer(expected_degree, expected_degree))
    np.testing.assert_allclose(normalized_message_matrix(graph).toarray(), dense, rtol=1e-6)


def test_cold_train_rows_match_training_graph_exactly():
    citations, split = symmetric(), cold_split()
    training = normalized_message_matrix(citation_training_graph(citations, split)).toarray()
    inference = normalized_message_matrix(citation_inference_graph(citations, split)).toarray()
    np.testing.assert_array_equal(inference[TRAIN_ITEMS], training[TRAIN_ITEMS])


def test_cold_train_weights_ignore_links_to_held_out_papers():
    split = cold_split()
    base = normalized_message_matrix(citation_inference_graph(symmetric(), split)).toarray()
    extra = symmetric(EDGES + [(9, 0), (11, 1), (8, 1)])
    changed = normalized_message_matrix(citation_inference_graph(extra, split)).toarray()
    np.testing.assert_array_equal(base[TRAIN_ITEMS], changed[TRAIN_ITEMS])


def test_held_out_row_uses_own_received_count_and_sender_training_degree():
    matrix = normalized_message_matrix(citation_inference_graph(symmetric(), cold_split()))
    # Paper 8 receives from train paper 0 only: d=2. Paper 0 has train neighbour 1: d=2.
    assert matrix[8, 0] == pytest.approx(1 / 2)
    assert matrix[8, 8] == pytest.approx(1 / 2)
    assert matrix[0, 8] == 0


def test_isolated_paper_keeps_its_own_features():
    matrix = normalized_message_matrix(citation_inference_graph(symmetric(), cold_split()))
    assert matrix[9, 9] == pytest.approx(1.0) and matrix[9].nnz == 1


# ---------- network and loss ----------


def test_gcn_output_is_normalized_and_trainable():
    network = GCN(16, 8, 4, 2, 0.0)
    matrix = _to_torch_sparse(normalized_message_matrix(symmetric()))
    output = network(torch.from_numpy(features()), matrix)
    assert output.shape == (N_PAPERS, 4)
    lengths = torch.linalg.norm(output, dim=1).detach().numpy()
    assert lengths == pytest.approx(np.ones(N_PAPERS), abs=1e-5)
    output.sum().backward()
    assert all(p.grad is not None for p in network.parameters())


def test_message_passing_follows_row_receives_from_column():
    network = GCN(2, 2, 2, 1, 0.0)
    with torch.no_grad():
        network.layers[0].weight.copy_(torch.eye(2))
        network.layers[0].bias.zero_()
    one_way = _to_torch_sparse(sparse.csr_matrix(np.array([[1, 0], [1, 1]], np.float32)))
    output = network(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), one_way)
    # Node 0 receives only from itself; node 1 mixes in node 0.
    assert output[0].tolist() == pytest.approx([1.0, 0.0])
    assert output[1, 0] > 0 and output[1, 1] > 0


def test_margin_loss_values():
    positive, negative = torch.tensor([0.9]), torch.tensor([[0.1, 0.8]])
    # max(0, 0.2-0.9+0.1)=0 and max(0, 0.2-0.9+0.8)=0.1; the mean is 0.05.
    value = ranking_loss(positive, negative, loss="margin", margin=0.2, scale=10.0)
    assert value.item() == pytest.approx(0.05)


def test_bce_rewards_separated_scores():
    good = ranking_loss(torch.tensor([0.9]), torch.tensor([[-0.9]]), loss="bce", margin=0, scale=10)
    bad = ranking_loss(torch.tensor([-0.9]), torch.tensor([[0.9]]), loss="bce", margin=0, scale=10)
    assert good < bad


# ---------- sampling ----------


@pytest.mark.parametrize("split_factory", [cold_split, warm_split])
def test_negatives_respect_masks_and_targets_stay_out_of_profiles(split_factory):
    split = split_factory()
    model = SimplifiedPRSGCNRecommender(features(), symmetric(), **SMALL)
    model.fit(PAPERS, split)
    users = np.array([0, 1])
    allowed = split.pair_mask("train", users)
    rng = np.random.default_rng(0)
    for _ in range(50):
        indices, offsets, positives, negatives = model._sample_triplets(users, rng)
        bounds = offsets.tolist() + [len(indices)]
        for row, user in enumerate(users):
            saved = set(split.train[user].indices)
            profile = set(indices[bounds[row] : bounds[row + 1]].tolist())
            assert positives[row].item() in saved
            assert positives[row].item() not in profile and profile <= saved
            for negative in negatives[row].tolist():
                assert negative not in saved and allowed[row, negative]


def test_warm_negatives_avoid_held_out_positive_and_unobserved_pairs():
    split = warm_split()
    model = SimplifiedPRSGCNRecommender(features(), symmetric(), **SMALL).fit(PAPERS, split)
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        seen |= set(model._sample_triplets(np.array([0]), rng)[3].ravel().tolist())
    assert not seen & {3, 8, 9}  # User 0's validation and test pairs.
    assert seen == {4, 5, 6, 7, 10, 11}


def test_cold_negatives_are_train_papers_only():
    model = fit()
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(100):
        seen |= set(model._sample_triplets(np.array([0, 1]), rng)[3].ravel().tolist())
    assert seen <= set(TRAIN_ITEMS.tolist())


# ---------- leakage ----------


def test_held_out_labels_cannot_change_embeddings():
    base = fit()
    other = fit(cold_split(validation=((2, 9), (1, 8)), test=((0, 11),)))
    np.testing.assert_array_equal(base.embeddings_, other.embeddings_)


def test_warm_held_out_labels_cannot_change_embeddings():
    base = fit(warm_split())
    split = warm_split()
    split.validation, split.test = binary([(0, 8)]), binary([(0, 9)])
    np.testing.assert_array_equal(base.embeddings_, fit(split).embeddings_)


def test_held_out_edges_do_not_change_train_embeddings():
    base = fit()
    changed = fit(citations=symmetric(EDGES + [(9, 0), (11, 1), (8, 2)]))
    np.testing.assert_array_equal(base.embeddings_[TRAIN_ITEMS], changed.embeddings_[TRAIN_ITEMS])
    assert not np.array_equal(base.embeddings_[9], changed.embeddings_[9])


def test_validation_and_test_papers_do_not_influence_each_other():
    base = fit()
    changed = fit(citations=symmetric(EDGES + [(11, 0)]))  # Only test paper 11 gains a link.
    np.testing.assert_array_equal(base.embeddings_[VAL_ITEMS], changed.embeddings_[VAL_ITEMS])
    assert not np.array_equal(base.embeddings_[11], changed.embeddings_[11])


def test_ablation_flag_lets_held_out_edges_change_only_held_out_embeddings():
    strict = fit()
    ablation = fit(include_heldout_edges=True)
    np.testing.assert_array_equal(
        strict.embeddings_[TRAIN_ITEMS], ablation.embeddings_[TRAIN_ITEMS]
    )
    assert not np.array_equal(strict.embeddings_[8], ablation.embeddings_[8])


# ---------- recommender ----------


def test_every_paper_gets_finite_scores_and_a_full_ranking():
    model = fit()
    scores = model.score([0, 1])
    assert scores.shape == (N_PAPERS,) and np.isfinite(scores).all()
    ranking = model.recommend([0, 1], k=4, candidates=VAL_ITEMS.tolist() + TEST_ITEMS.tolist())
    assert sorted(item for item, _ in ranking) == [8, 9, 10, 11]


def test_training_reduces_loss_and_is_reproducible():
    first = fit(epochs=30, batch_users=3)
    second = fit(epochs=30, batch_users=3)
    np.testing.assert_array_equal(first.embeddings_, second.embeddings_)
    statistics = first.fit_statistics_
    assert statistics["last_epoch_loss"] < statistics["first_epoch_loss"]
    assert statistics["trainable_users"] == 2  # User 2 has one save, so no profile remains.


def test_no_graph_mode_ignores_citations():
    base = fit(use_graph=False)
    other = fit(use_graph=False, citations=symmetric(EDGES + [(0, 7)]))
    np.testing.assert_array_equal(base.embeddings_, other.embeddings_)


# ---------- ablations ----------


@pytest.mark.parametrize("residual", ["add", "concat"])
def test_residual_keeps_cold_rules(residual):
    base = fit(residual=residual)
    changed = fit(residual=residual, citations=symmetric(EDGES + [(9, 0), (11, 1), (8, 2)]))
    np.testing.assert_array_equal(base.embeddings_[TRAIN_ITEMS], changed.embeddings_[TRAIN_ITEMS])
    assert not np.array_equal(base.embeddings_[9], changed.embeddings_[9])
    assert np.isfinite(base.embeddings_).all()


def test_residual_inputs_are_add_and_concat():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    propagation = torch.tensor([[0.0, 1.0], [0.0, 0.0]]).to_sparse().coalesce()
    add = GCN(2, 2, 2, 1, 0.0, residual="add").eval()
    concat = GCN(2, 2, 2, 1, 0.0, residual="concat").eval()
    assert concat.layers[0].in_features == 4
    seen = []
    for model in (add, concat):
        model.layers[0].register_forward_pre_hook(lambda _, args: seen.append(args[0]))
        model(hidden, propagation)
    np.testing.assert_array_equal(seen[0].numpy(), [[4.0, 6.0], [3.0, 4.0]])
    np.testing.assert_array_equal(seen[1].numpy(), [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 0.0, 0.0]])


@pytest.mark.parametrize("input_norm", ["l2", "row"])
def test_input_norm_keeps_cold_rules(input_norm):
    options = {"residual": "add", "input_norm": input_norm}
    base = fit(**options)
    changed = fit(citations=symmetric(EDGES + [(9, 0), (11, 1), (8, 2)]), **options)
    np.testing.assert_array_equal(base.embeddings_[TRAIN_ITEMS], changed.embeddings_[TRAIN_ITEMS])
    assert not np.array_equal(base.embeddings_[9], changed.embeddings_[9])


def test_row_mode_averages_a_paper_and_its_senders():
    graph = symmetric([(0, 1), (1, 2)], 4)
    matrix = normalized_message_matrix(graph, "row").toarray()
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    np.testing.assert_allclose(matrix[1], [1 / 3, 1 / 3, 1 / 3, 0])
    np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])


def test_row_mode_keeps_cold_train_rows_equal_to_training():
    training = normalized_message_matrix(citation_training_graph(symmetric(), cold_split()), "row")
    inference = normalized_message_matrix(
        citation_inference_graph(symmetric(), cold_split()), "row"
    )
    np.testing.assert_array_equal(training[TRAIN_ITEMS].toarray(), inference[TRAIN_ITEMS].toarray())


def test_l2_mode_gives_every_propagated_row_unit_length():
    hidden = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    propagation = torch.tensor([[1.0, 0.0], [0.0, 0.5]]).to_sparse().coalesce()
    model = GCN(2, 2, 2, 1, 0.0, normalize_propagated=True).eval()
    seen = []
    model.layers[0].register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    model(hidden, propagation)
    np.testing.assert_allclose(seen[0].numpy(), [[0.6, 0.8], [1.0, 0.0]], rtol=1e-6)


def test_full_edge_dropout_equals_the_no_graph_model():
    dropped = fit(edge_dropout=1.0)
    no_graph = fit(use_graph=False)
    np.testing.assert_allclose(dropped.embeddings_, no_graph.embeddings_, rtol=1e-5, atol=1e-6)


def test_full_edge_dropout_removes_added_links_too():
    model = fit(edge_dropout=0.5)
    with_extra_links = fit(edge_dropout=1.0, citations=symmetric(EDGES + [(9, 0), (11, 1)]))
    assert np.isfinite(model.embeddings_).all()
    np.testing.assert_allclose(
        with_extra_links.embeddings_, fit(edge_dropout=1.0).embeddings_, rtol=1e-5, atol=1e-6
    )


def test_knn_source_needs_no_citations_and_keeps_cold_rules():
    model = SimplifiedPRSGCNRecommender(features(), graph_source="knn", knn_k=2, **SMALL)
    base = model.fit(PAPERS, cold_split())
    changed_features = features()
    changed_features[VAL_ITEMS] = 0.5
    other = SimplifiedPRSGCNRecommender(changed_features, graph_source="knn", knn_k=2, **SMALL)
    other.fit(PAPERS, cold_split())
    np.testing.assert_array_equal(base.embeddings_[TRAIN_ITEMS], other.embeddings_[TRAIN_ITEMS])
    assert not np.array_equal(base.embeddings_[8], other.embeddings_[8])


def test_degree_breakdown_counts_pairs_hits_and_slots():
    buckets = degree_buckets(np.array([0, 1, 2, 3, 4, 7, 8, 100]))
    np.testing.assert_array_equal(buckets, [0, 1, 2, 2, 3, 3, 4, 4])
    targets = sparse.csr_matrix(np.array([[1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0]]))
    per_user = [
        {"user_id": 0, "top_items": [0, 2]},  # Hits degree 0; misses degree 8.
        {"user_id": 1, "top_items": [3, 4]},  # Misses degree 1.
    ]
    result = degree_breakdown(per_user, targets, buckets)
    assert result["positives"] == [1, 1, 0, 0, 1]
    assert result["hits"] == [1, 0, 0, 0, 0]
    assert result["hit_rate"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert result["top_share"] == [0.25, 0.0, 0.5, 0.25, 0.0]


def test_bce_mode_trains():
    assert np.isfinite(fit(loss="bce").embeddings_).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"loss": "hinge"},
        {"use_graph": False, "include_heldout_edges": True},
        {"residual": "multiply"},
        {"input_norm": "max"},
        {"use_graph": False, "input_norm": "l2"},
        {"use_graph": False, "residual": "add"},
        {"use_graph": False, "edge_dropout": 0.5},
        {"graph_source": "pagerank"},
        {"graph_source": "knn", "include_heldout_edges": True},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        SimplifiedPRSGCNRecommender(features(), symmetric(), **overrides)


def test_missing_citations_are_rejected():
    with pytest.raises(ValueError, match="citations"):
        SimplifiedPRSGCNRecommender(features())


def test_feature_rows_must_match_papers():
    model = SimplifiedPRSGCNRecommender(features()[:3], symmetric(), **SMALL)
    with pytest.raises(ValueError, match="one row per paper"):
        model.fit(PAPERS, cold_split())


def test_specter2_reference_uses_cosine_of_features():
    model = Specter2Recommender(features()).fit(PAPERS, cold_split())
    assert model.score([0])[0] == pytest.approx(1.0)


# ---------- SPECTER2 cache ----------


class FakeEncoder:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.seen = []
        self.fail_after = fail_after

    def __call__(self, papers):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("interrupted")
        self.calls += 1
        self.seen += [p.paper_id for p in papers]
        output = np.zeros((len(papers), DIMENSIONS), np.float32)
        output[:, 0] = [p.paper_id for p in papers]
        return output


def test_features_are_in_paper_order_and_cached(tmp_path):
    encoder = FakeEncoder()
    matrix = specter2_features(PAPERS, cache_root=tmp_path, batch_size=5, encoder=encoder)
    assert matrix[:, 0].tolist() == list(range(N_PAPERS))
    assert encoder.seen[0] == N_PAPERS - 1  # Longest text goes first.
    repeat = FakeEncoder()
    np.testing.assert_array_equal(
        matrix, specter2_features(PAPERS, cache_root=tmp_path, encoder=repeat)
    )
    assert repeat.calls == 0
    assert not list(tmp_path.rglob("*.partial.npy")) and not list(tmp_path.rglob("progress.json"))


def test_interrupted_encoding_resumes_without_repeating_work(tmp_path):
    with pytest.raises(RuntimeError, match="interrupted"):
        specter2_features(
            PAPERS, cache_root=tmp_path, batch_size=5, encoder=FakeEncoder(fail_after=1)
        )
    resumed = FakeEncoder()
    matrix = specter2_features(PAPERS, cache_root=tmp_path, batch_size=5, encoder=resumed)
    assert len(resumed.seen) == N_PAPERS - 5
    assert matrix[:, 0].tolist() == list(range(N_PAPERS))


def test_changed_text_uses_a_new_cache(tmp_path):
    specter2_features(PAPERS, cache_root=tmp_path, encoder=FakeEncoder())
    changed = [Paper(p.paper_id, p.title + "!", p.abstract) for p in PAPERS]
    encoder = FakeEncoder()
    specter2_features(changed, cache_root=tmp_path, encoder=encoder)
    assert encoder.calls > 0


def test_invalid_encoder_output_is_rejected(tmp_path):
    def broken(papers):
        return np.full((len(papers), DIMENSIONS), np.nan, np.float32)

    with pytest.raises(ValueError, match="invalid vectors"):
        specter2_features(PAPERS, cache_root=tmp_path, encoder=broken)
