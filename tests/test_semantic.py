"""Unit tests for semantic recommenders (LSI, LDA, SPECTER2)."""

import numpy as np
import pytest
from scipy import sparse

from paperrec.data import Paper
from paperrec.models.lda import LDARecommender
from paperrec.models.lsi import LSIRecommender
from paperrec.models.specter2 import SPECTER2Recommender, encode_arxiv_paper
from paperrec.splits import Split


@pytest.fixture
def dummy_catalog():
    """Create a minimal synthetic catalog of 6 papers with distinct scientific vocabulary."""
    return [
        Paper(0, "Deep Learning for Computer Vision", "Convolutional neural networks image classification object detection"),
        Paper(1, "Transformers in Natural Language Processing", "Attention mechanisms large language models text generation BERT"),
        Paper(2, "Graph Neural Networks for Molecules", "Graph convolutional networks molecular property prediction biochemistry"),
        Paper(3, "Vision Transformers and Attention", "Self-attention mechanisms applied to image patches vision models"),
        Paper(4, "Drug Discovery with Graph Models", "Molecules chemistry graph representation learning pharmacology"),
        Paper(5, "Pretrained Language Representations", "BERT RoBERTa transformer embeddings transfer learning text"),
    ]


@pytest.fixture
def dummy_split():
    """Create a mock 2-user, 6-item split with cold-start items."""
    # User 0 likes paper 0 (Vision)
    # User 1 likes paper 1 (NLP)
    rows = np.array([0, 1])
    cols = np.array([0, 1])
    data = np.ones(2, dtype=np.float32)
    train_mat = sparse.csr_matrix((data, (rows, cols)), shape=(2, 6))

    val_mat = sparse.csr_matrix((np.ones(1), (np.array([0]), np.array([3]))), shape=(2, 6))
    test_mat = sparse.csr_matrix((np.ones(1), (np.array([1]), np.array([5]))), shape=(2, 6))

    train_items = np.array([0, 1, 2, 4])  # 3 and 5 are cold items
    val_items = np.array([3])
    test_items = np.array([5])

    return Split(
        train=train_mat,
        validation=val_mat,
        test=test_mat,
        train_items=train_items,
        validation_items=val_items,
        test_items=test_items,
        scenario="cold",
        seed=42,
        fold=0,
        n_folds=5,
        validation_fraction=0.1,
    )


def test_lsi_recommender(dummy_catalog, dummy_split):
    model = LSIRecommender(n_components=3, min_df=1, seed=42)
    model.fit(dummy_catalog, dummy_split)

    # 1. Check paper vectors
    embeddings = model.get_embeddings()
    assert embeddings.shape == (6, 3)
    assert np.all(np.isfinite(embeddings))

    # 2. Check scoring
    profile = np.array([0])  # User likes Vision paper
    scores = model.score(profile)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))

    # Paper 3 (Vision Transformers) should score higher for User 0 than Paper 1 (NLP)
    assert scores[3] > scores[1]

    # 3. Check recommend() excludes profile
    recs = model.recommend(profile, k=3, candidates=np.array([1, 2, 3, 4, 5]))
    rec_items = [item for item, _ in recs]
    assert 0 not in rec_items
    assert len(recs) == 3


def test_lda_recommender(dummy_catalog, dummy_split):
    model = LDARecommender(n_topics=3, min_df=1, max_iter=10, seed=42)
    model.fit(dummy_catalog, dummy_split)

    # 1. Check topic distributions (Theta)
    theta = model.get_topic_distribution()
    assert theta.shape == (6, 3)
    assert np.all(np.isfinite(theta))
    # Probabilities must sum to ~1.0
    np.testing.assert_allclose(theta.sum(axis=1), np.ones(6), atol=1e-5)

    # 2. Check scoring
    profile = np.array([1])  # User likes NLP
    scores = model.score(profile)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))

    # 3. Recommend properly ranks candidates
    recs = model.recommend(profile, k=2, candidates=np.array([0, 2, 3, 4, 5]))
    assert len(recs) == 2
    rec_items = [item for item, _ in recs]
    assert 1 not in rec_items


def test_specter2_recommender_cached(dummy_catalog, dummy_split, tmp_path):
    # Test using pre-generated dummy cache
    cache_file = tmp_path / "mock_specter2.npy"
    rng = np.random.default_rng(42)
    mock_vecs = rng.normal(size=(6, 768)).astype(np.float32)
    norms = np.linalg.norm(mock_vecs, axis=1, keepdims=True)
    np.save(cache_file, mock_vecs / norms)

    model = SPECTER2Recommender(cache_path=cache_file, seed=42)
    model.fit(dummy_catalog, dummy_split)

    assert model.cache_hit_ is True
    embeddings = model.get_embeddings()
    assert embeddings.shape == (6, 768)

    profile = np.array([2, 4])  # Graph / Drug
    scores = model.score(profile)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))

    recs = model.recommend(profile, k=4)
    rec_items = [item for item, _ in recs]
    assert 2 not in rec_items and 4 not in rec_items
