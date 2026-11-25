import gzip
import json
import re
from functools import cache
from pathlib import Path

import hf
import numpy as np
from scipy import sparse
from scipy.special import psi

DATA = Path(__file__).with_name("demo_data")
TEXT_MODELS = {
    "TF-IDF": "tfidf",
    "BM25": "bm25",
    "LSI": "lsi",
    "LDA": "lda",
    "SPECTER2 Base": "specter2-base",
    "SPECTER2 + Adapter": "specter2-adapter",
    "Plain GCN": "gcn-bce",
    "nograph-bce": "nograph-bce",
    "gcn-bce-add": "gcn-bce-add",
}
ID_MODELS = {
    "ItemKNN": "item-knn",
    "WMF": "wmf",
    "Node2Vec Citation": "node2vec-citation",
    "Node2Vec Bipartite": "node2vec-bipartite",
}
MODELS = {**TEXT_MODELS, **ID_MODELS}
HOSTED = ("specter2-base", "specter2-adapter", "gcn-bce", "nograph-bce", "gcn-bce-add")


def unit(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms > 0, norms, 1)


def read_json(name):
    with gzip.open(DATA / name, "rt", encoding="utf-8") as f:
        return json.load(f)


@cache
def papers():
    return read_json("papers.json.gz")


@cache
def vectors(name):
    x = np.load(DATA / f"{name}.npy").astype(np.float32)
    scale = DATA / f"{name}_scale.npy"
    return unit(x * np.load(scale) if scale.exists() else x)


@cache
def counts():
    return sparse.load_npz(DATA / "counts.npz").astype(np.float32)


@cache
def vocabulary():
    return {term: i for i, term in enumerate(read_json("vocabulary.json.gz"))}


@cache
def document_frequency():
    return np.bincount(counts().indices, minlength=counts().shape[1])


def tfidf_weights(x):
    idf = np.log((1 + counts().shape[0]) / (1 + document_frequency())) + 1
    x = sparse.csr_matrix(x.multiply(idf), dtype=np.float32)
    norms = sparse.linalg.norm(x, axis=1)
    return sparse.diags(1 / np.where(norms > 0, norms, 1)) @ x


@cache
def tfidf():
    return tfidf_weights(counts()).tocsr()


@cache
def bm25(k1=1.2, b=0.75):
    x = counts().tocsr()
    df = document_frequency()
    idf = np.log1p((x.shape[0] - df + 0.5) / (df + 0.5))
    length = np.asarray(x.sum(axis=1)).ravel()
    norm = k1 * (1 - b + b * length / length.mean())
    rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
    data = idf[x.indices] * x.data * (k1 + 1) / (x.data + norm[rows])
    return sparse.csr_matrix((data, x.indices, x.indptr), shape=x.shape)


@cache
def item_knn(neighbors=400):
    saves = sparse.load_npz(DATA / "saves.npz").astype(np.float64)
    norms = np.sqrt(np.asarray(saves.sum(axis=0)).ravel())
    x = (saves @ sparse.diags(1 / np.where(norms > 0, norms, 1))).tocsc()
    similarity = (x.T @ x).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    for i in range(similarity.shape[0]):
        row = slice(similarity.indptr[i], similarity.indptr[i + 1])
        if row.stop - row.start > neighbors:
            order = np.lexsort((similarity.indices[row], -similarity.data[row]))
            similarity.data[row.start + order[neighbors:]] = 0
    similarity.eliminate_zeros()
    return similarity


def bag_of_words(text):
    ids = [vocabulary()[t] for t in re.findall(r"(?u)\b\w\w+\b", text.lower()) if t in vocabulary()]
    row = np.bincount(ids, minlength=counts().shape[1]).astype(np.float32)
    return sparse.csr_matrix(row)


def lda_topics(text, steps=100):
    terms = np.load(DATA / "lda_terms.npy")
    words = bag_of_words(text).toarray()[0, terms]
    seen = np.flatnonzero(words)
    topics = np.exp(np.load(DATA / "lda_topics.npy")[:, seen].astype(np.float32))
    gamma = np.ones(len(topics))
    for _ in range(steps):
        theta = np.exp(psi(gamma) - psi(gamma.sum()))
        last, gamma = gamma, theta * (topics @ (words[seen] / (theta @ topics + 1e-10)))
        gamma += 1 / len(topics)
        if np.abs(gamma - last).mean() < 1e-3:
            break
    return gamma


def encode(name, title, abstract):
    text = f"{title} {abstract}"
    if name == "lsi":
        raw = np.load(DATA / "lsi.npy").astype(np.float32)
        similar = (tfidf() @ tfidf_weights(bag_of_words(text)).T).T @ raw
        return unit(similar / np.load(DATA / "lsi_sigma.npy") ** 2)[0]
    if name == "lda":
        return unit(lda_topics(text))
    vector = unit(hf.embed(title, abstract, name != "specter2-base"))
    if name.startswith("specter2"):
        return vector
    # A typed paper has no citations, so it only receives its own message.
    layer = np.load(DATA / f"{name}_layer.npy")
    return unit((2 if name == "gcn-bce-add" else 1) * vector @ layer[:-1] + layer[-1])


def scores(name, picked, title="", abstract=""):
    typed = bool(title or abstract)
    if name == "item-knn":
        return np.asarray(item_knn()[:, picked].sum(axis=1)).ravel()
    if name == "wmf":
        y = np.load(DATA / "wmf.npy").astype(np.float64)
        gram = y.T @ y + 40 * y[picked].T @ y[picked] + 10 * np.eye(y.shape[1])
        return y @ np.linalg.solve(gram, 41 * y[picked].sum(axis=0))
    if name in ("tfidf", "bm25"):
        x = tfidf() if name == "tfidf" else bm25()
        rows = x[picked]
        if typed:
            words = bag_of_words(f"{title} {abstract}")
            rows = sparse.vstack((rows, tfidf_weights(words) if name == "tfidf" else words)).tocsr()
        profile = np.asarray(rows.sum(axis=0)).ravel()
        return x @ (unit(profile) if name == "tfidf" else (profile > 0).astype(np.float32))
    x = vectors(name)
    profile = x[picked].sum(axis=0)
    if typed:
        profile = profile + encode(name, title, abstract)
    result = x @ unit(profile)
    result[~x.any(axis=1)] = -np.inf
    return result


def recommend(name, k, picked, title="", abstract=""):
    picked = np.array(picked, dtype=int)
    result = scores(name, picked, title, abstract)
    if not np.any(result[np.isfinite(result)]):
        return []
    result[picked] = -np.inf
    top = np.argsort(-result, kind="stable")[:k]
    return [(int(i), float(result[i])) for i in top if np.isfinite(result[i])]
