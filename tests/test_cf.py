import numpy as np
import pytest
from scipy import sparse

from paperrec.data import Paper
from paperrec.models.item_knn import ItemKNNRecommender
from paperrec.models.wmf import WMFRecommender
from paperrec.splits import make_folds


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    matrix = sparse.csr_matrix((rng.random((40, 30)) < 0.4).astype(np.float32))
    return [Paper(i, f"title {i}", f"abstract {i}") for i in range(30)], matrix


@pytest.mark.parametrize("scenario", ["warm", "cold"])
@pytest.mark.parametrize(
    "model",
    [
        lambda: ItemKNNRecommender(neighbors=5),
        lambda: WMFRecommender(factors=4, iterations=3, device="cpu"),
    ],
)
def test_held_out_saves_never_change_scores(data, scenario, model):
    papers, matrix = data
    split = make_folds(matrix, scenario=scenario, seed=1)[0]
    before = model().fit(papers, split).score(split.train[0].indices, 0)
    split.test = sparse.csr_matrix(split.test.shape, dtype=np.float32)
    split.validation = sparse.csr_matrix(split.validation.shape, dtype=np.float32)
    after = model().fit(papers, split).score(split.train[0].indices, 0)
    np.testing.assert_array_equal(before, after)
    if scenario == "cold":
        assert not after[split.test_items].any()


def test_item_knn_matches_dense_cosine(data):
    papers, matrix = data
    split = make_folds(matrix, scenario="warm", seed=1)[0]
    model = ItemKNNRecommender(neighbors=100).fit(papers, split)
    dense = split.train.toarray().astype(float)
    unit = dense / np.linalg.norm(dense, axis=0)
    cosine = unit.T @ unit
    np.fill_diagonal(cosine, 0)
    np.testing.assert_allclose(model.similarity_.toarray(), cosine, atol=1e-12)


def test_wmf_update_matches_normal_equations(data):
    papers, matrix = data
    split = make_folds(matrix, scenario="warm", seed=1)[0]
    model = WMFRecommender(
        factors=3, alpha=2.0, regularization=0.5, iterations=4, device="cpu"
    ).fit(papers, split)
    users, item = model.user_factors_.astype(float), 7
    saved = split.train.toarray()[:, item]
    weight = split.pair_mask("train")[:, item] + 2.0 * saved
    gram = users.T @ (weight[:, None] * users) + 0.5 * np.eye(3)
    expected = np.linalg.solve(gram, users.T @ (3.0 * saved))
    np.testing.assert_allclose(model.item_factors_[item], expected, rtol=1e-3, atol=1e-4)
