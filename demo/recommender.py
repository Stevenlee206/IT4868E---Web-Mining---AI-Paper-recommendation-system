import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paperrec.data import Paper, load_citeulike_a


def load_papers():
    return load_citeulike_a(ROOT / "data/raw/citeulike-a").papers


class Tfidf:
    def __init__(self, papers):
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
        self.vectors = self.vectorizer.fit_transform([p.text for p in papers])

    def encode(self, title, abstract):
        return self.vectorizer.transform([f"{title} {abstract}"])

    def scores(self, selected, title="", abstract=""):
        profile = self.vectors[selected].sum(axis=0) if selected else 0
        if title or abstract:
            profile = profile + self.encode(title, abstract).toarray()
        return np.asarray(self.vectors @ normalize(np.asarray(profile)).T).ravel()


class Specter:
    """SPECTER2 vectors passed through the trained nograph-bce projection."""

    def __init__(self):
        model = np.load(Path(__file__).with_name("model.npz"))
        self.vectors = normalize(model["papers"].astype(np.float32))
        self.weight, self.bias = model["weight"], model["bias"]
        self.encoder = None

    def encode(self, title, abstract):
        if self.encoder is None:
            from paperrec.specter import default_device, make_batch_encoder

            self.encoder = make_batch_encoder(device=default_device(), max_length=512)
        vector = normalize(self.encoder([Paper(0, title, abstract)]))
        return normalize(vector @ self.weight.T + self.bias)

    def scores(self, selected, title="", abstract=""):
        profile = self.vectors[selected].sum(axis=0, keepdims=True)
        if title or abstract:
            profile = profile + self.encode(title, abstract)
        return (self.vectors @ normalize(profile).T).ravel()


def recommend(model, k, selected, title="", abstract=""):
    scores = model.scores(selected, title, abstract)
    scores[selected] = -np.inf
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]
