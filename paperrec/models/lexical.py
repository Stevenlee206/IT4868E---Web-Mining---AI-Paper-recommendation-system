"""Inductive lexical recommenders and content-addressed sparse feature caches.

Only train_items fit vocabulary/IDF/average length. All catalog rows are then
transformed with frozen statistics. Caches never store interactions or profiles.
"""

import hashlib
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .base import Recommender

CACHE_VERSION = 1
TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _LexicalRecommender(Recommender):
    def __init__(
        self,
        *,
        seed=42,
        cache_root="artifacts/features",
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        min_df=1,
        max_df=1.0,
    ):
        super().__init__(seed=seed)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.min_df = min_df
        self.max_df = max_df

    def _text_config(self):
        return {
            "lowercase": self.lowercase,
            "token_pattern": self.token_pattern,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": [1, 1],
            "stop_words": None,
        }

    def get_config(self):
        return {
            **super().get_config(),
            "method": self.method,
            "text": self._text_config(),
            **self._weight_config(),
        }

    def _vectorizer_args(self):
        config = self._text_config()
        config["ngram_range"] = tuple(config["ngram_range"])
        return config

    def fit(self, papers, split):
        self._cache_scope = (split.scenario, split.fold if split.scenario == "cold" else None)
        return super().fit(papers, split)

    def _fit(self):
        self.cache_hit_ = False
        self.cache_path_ = None
        if not len(self.train_items):
            raise ValueError("Lexical models need at least one permitted training paper")
        self.cache_metadata_ = None
        if self.cache_root is not None:
            scenario, fold = self._cache_scope
            config = self.get_config().copy()
            config.pop("seed")  # Features are deterministic; the seed only breaks ranking ties.
            identity = {
                "format_version": CACHE_VERSION,
                "text_fingerprint": _json_hash(
                    [(p.paper_id, p.title, p.abstract) for p in self.papers]
                ),
                "train_items_fingerprint": _json_hash(self.train_items.tolist()),
                "train_item_count": len(self.train_items),
                "paper_count": len(self.papers),
                "scenario": scenario,
                "fold": fold,
                "config": config,
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__,
                },
            }
            directory = self.cache_root / scenario
            if fold is not None:
                directory /= f"fold_{fold}"
            self.cache_path_ = directory / _json_hash(identity) / self.method
            if self.cache_path_.exists():
                self._load_cache(identity)
                self.cache_hit_ = True
                return
        self._build_features()
        if self.cache_path_ is not None:
            self._save_cache(identity)

    def _save_cache(self, identity):
        self.cache_path_.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete caches; interruption cannot leave a reusable partial cache.
        with TemporaryDirectory(prefix=".building-", dir=self.cache_path_.parent) as temporary:
            directory = Path(temporary) / self.method
            directory.mkdir()
            sparse.save_npz(directory / "features.npz", self.features_)
            np.savez_compressed(directory / "state.npz", **self._state())
            vocabulary = {term: int(index) for term, index in self.vectorizer_.vocabulary_.items()}
            (directory / "vocabulary.json").write_text(
                json.dumps(vocabulary, sort_keys=True, ensure_ascii=True), encoding="utf-8"
            )
            metadata = {
                "identity": identity,
                "shape": list(self.features_.shape),
                "dtype": str(self.features_.dtype),
                "checksums": {
                    name: _file_hash(directory / name)
                    for name in ("features.npz", "state.npz", "vocabulary.json")
                },
            }
            (directory / "metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
            )
            try:
                directory.rename(self.cache_path_)
            except FileExistsError:
                # Another process may have finished the same deterministic cache first.
                self._load_cache(identity)
            else:
                self.cache_metadata_ = metadata

    def _load_cache(self, identity):
        try:
            directory = self.cache_path_
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata["identity"] != identity:
                raise ValueError("identity mismatch")
            for name in ("features.npz", "state.npz", "vocabulary.json"):
                if _file_hash(directory / name) != metadata["checksums"][name]:
                    raise ValueError(f"checksum mismatch: {name}")
            vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
            features = sparse.load_npz(directory / "features.npz").tocsr()
            if (
                features.shape != (len(self.papers), len(vocabulary))
                or list(features.shape) != metadata["shape"]
                or str(features.dtype) != metadata["dtype"]
                or not np.isfinite(features.data).all()
                or sorted(vocabulary.values()) != list(range(len(vocabulary)))
            ):
                raise ValueError("invalid feature matrix or vocabulary")
            with np.load(directory / "state.npz", allow_pickle=False) as state:
                self._restore_state(vocabulary, state)
            if self.idf_.shape != (len(vocabulary),) or not np.isfinite(self.idf_).all():
                raise ValueError("invalid IDF state")
            self.features_ = features
            self.cache_metadata_ = metadata
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid feature cache at {self.cache_path_}: {exc}") from exc

    def _profile_ids(self, profile):
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        return self._item_ids(profile)


class TfidfRecommender(_LexicalRecommender):
    """Cosine similarity to an equally weighted centroid of saved paper vectors."""

    method = "tfidf"

    def __init__(self, *, sublinear_tf=False, **kwargs):
        super().__init__(**kwargs)
        self.sublinear_tf = sublinear_tf

    def _weight_config(self):
        return {
            "sublinear_tf": self.sublinear_tf,
            "smooth_idf": True,
            "norm": "l2",
            "profile": "normalized_mean_of_l2_paper_vectors",
        }

    def _make_vectorizer(self, **kwargs):
        return TfidfVectorizer(
            **self._vectorizer_args(),
            sublinear_tf=self.sublinear_tf,
            smooth_idf=True,
            norm="l2",
            dtype=np.float32,
            **kwargs,
        )

    def _build_features(self):
        self.vectorizer_ = self._make_vectorizer()
        self.vectorizer_.fit([self.papers[i].text for i in self.train_items])
        self.features_ = self.vectorizer_.transform([p.text for p in self.papers]).tocsr()
        self.idf_ = self.vectorizer_.idf_.copy()

    def _state(self):
        return {"idf": self.idf_}

    def _restore_state(self, vocabulary, state):
        self.idf_ = state["idf"].copy()
        self.vectorizer_ = self._make_vectorizer(vocabulary=vocabulary)
        self.vectorizer_.idf_ = self.idf_

    def score(self, profile, user_index=None):
        ids = self._profile_ids(profile)
        if not len(ids):
            return np.zeros(len(self.papers), dtype=np.float32)
        centroid = np.asarray(self.features_[ids].sum(axis=0)).ravel()
        # Normalizing the sum is identical to normalizing the mean.
        length = np.linalg.norm(centroid)
        if length:
            centroid /= length
        return np.asarray(self.features_ @ centroid).ravel()


class BM25Recommender(_LexicalRecommender):
    """Okapi BM25 with positive IDF and a binary union of saved-paper query terms.

    Document length counts all analyzed tokens, including tokens outside the
    fitted vocabulary. Average length is frozen from training papers only.
    """

    method = "bm25"

    def __init__(self, *, k1=1.2, b=0.75, **kwargs):
        super().__init__(**kwargs)
        if not np.isfinite(k1) or k1 < 0:
            raise ValueError("k1 must be finite and nonnegative")
        if not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between zero and one")
        self.k1, self.b = float(k1), float(b)

    def _weight_config(self):
        return {
            "k1": self.k1,
            "b": self.b,
            "idf": "log1p((N-df+0.5)/(df+0.5))",
            "profile": "unique_terms",
            "document_length": "all_analyzed_tokens",
        }

    def _build_features(self):
        self.vectorizer_ = CountVectorizer(**self._vectorizer_args())
        train_counts = self.vectorizer_.fit_transform(
            [self.papers[i].text for i in self.train_items]
        ).tocsr()
        df = np.asarray((train_counts > 0).sum(axis=0)).ravel()
        self.idf_ = np.log1p((len(self.train_items) - df + 0.5) / (df + 0.5))
        analyze = self.vectorizer_.build_analyzer()
        self.document_lengths_ = np.array([len(analyze(p.text)) for p in self.papers])
        self.avgdl_ = float(self.document_lengths_[self.train_items].mean())
        counts = self.vectorizer_.transform([p.text for p in self.papers]).tocsr()
        rows = np.repeat(np.arange(len(self.papers)), np.diff(counts.indptr))
        frequencies = counts.data.astype(np.float64)
        length_norm = 1 - self.b + self.b * self.document_lengths_[rows] / self.avgdl_
        weights = (
            self.idf_[counts.indices]
            * frequencies
            * (self.k1 + 1)
            / (frequencies + self.k1 * length_norm)
        )
        self.features_ = sparse.csr_matrix(
            (weights.astype(np.float32), counts.indices, counts.indptr), shape=counts.shape
        )

    def _state(self):
        return {
            "idf": self.idf_,
            "avgdl": np.array(self.avgdl_),
            "document_lengths": self.document_lengths_,
        }

    def _restore_state(self, vocabulary, state):
        self.idf_ = state["idf"].copy()
        self.avgdl_ = float(state["avgdl"])
        self.document_lengths_ = state["document_lengths"].copy()
        if (
            not np.isfinite(self.avgdl_)
            or self.avgdl_ <= 0
            or self.document_lengths_.shape != (len(self.papers),)
            or not np.isfinite(self.document_lengths_).all()
            or np.any(self.document_lengths_ < 0)
        ):
            raise ValueError("invalid document length state")
        self.vectorizer_ = CountVectorizer(**self._vectorizer_args(), vocabulary=vocabulary)

    def score(self, profile, user_index=None):
        ids = self._profile_ids(profile)
        query = np.zeros(self.features_.shape[1], dtype=np.float32)
        # BM25 weights are positive, so their sparsity also encodes term presence.
        query[self.features_[ids].indices] = 1
        return np.asarray(self.features_ @ query).ravel()
