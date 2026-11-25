"""Train all 13 models on warm fold 0 and save small files for the demo app.

Runs on Kaggle through scripts/kaggle/launch.py demo. Besides the demo data it writes
reference.json: the original models' top-20 lists, used to check the demo's scoring.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .benchmark import MODELS, make_factory
from .data import load_citeulike_a
from .specter import specter2_features
from .splits import load_split

SETTINGS = {
    "item-knn": {"neighbors": 400},
    "wmf": {"factors": 128, "alpha": 40.0, "regularization": 10.0},
}


def unit(matrix):
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(lengths > 0, lengths, 1)


def save_quantized(path, matrix):
    """Save int8 values and one scale per dimension, since a few dimensions are much larger."""
    scale = np.abs(matrix).max(axis=0) / 127
    np.save(path, np.round(matrix / scale).astype(np.int8))
    np.save(path.with_name(f"{path.stem}_scale.npy"), scale.astype(np.float32))


def fit_map(source, target, ridge=0.03):
    """Ridge regression with intercept; rows of the result are [weights; intercept]."""
    x = np.hstack((source, np.ones((len(source), 1))))
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ target)


def adapter_map(features_dir, papers, output):
    """Predict adapter vectors from base vectors, which is all the hosted API provides."""
    base = unit(np.load(features_dir / "specter2_base.npy").astype(np.float64))
    adapter = unit(specter2_features(papers, cache_root=features_dir / "specter2").astype(float))
    order = np.random.default_rng(0).permutation(len(base))
    test, train = order[:2000], order[2000:]
    weights = fit_map(base[train], adapter[train])
    predicted = unit(base[test] @ weights[:-1] + weights[-1])
    overlap, baseline = [], []
    for i, row in enumerate(test[:500]):
        true = np.argsort(-(adapter @ adapter[row]))[1:21]
        ours = np.argsort(-(adapter @ predicted[i]))
        raw = np.argsort(-(adapter @ base[row]))
        overlap.append(len(set(true) & set(ours[ours != row][:20])) / 20)
        baseline.append(len(set(true) & set(raw[raw != row][:20])) / 20)
    report = {
        "held_out_cosine_mapped": float((predicted * adapter[test]).sum(1).mean()),
        "held_out_cosine_unmapped": float((base[test] * adapter[test]).sum(1).mean()),
        "top20_overlap_mapped": float(np.mean(overlap)),
        "top20_overlap_unmapped": float(np.mean(baseline)),
    }
    print(json.dumps(report), flush=True)
    np.save(output / "adapter_map.npy", fit_map(base, adapter).astype(np.float32))
    save_quantized(output / "specter2-base.npy", base)
    save_quantized(output / "specter2-adapter.npy", adapter)
    return report


def export_model(name, model, output, vocabulary):
    if name == "lsi":
        x = model.vectorizer_.transform([p.text for p in model.papers])
        np.save(output / "lsi.npy", model.svd_.transform(x).astype(np.float16))
        np.save(output / "lsi_sigma.npy", model.svd_.singular_values_.astype(np.float32))
    elif name == "lda":
        np.save(output / "lda.npy", model.theta_normalized_.astype(np.float16))
        terms = model.vectorizer_.get_feature_names_out()
        np.save(output / "lda_terms.npy", np.array([vocabulary[t] for t in terms], dtype=np.int32))
        topics = np.log(model.lda_.exp_dirichlet_component_)
        np.save(output / "lda_topics.npy", topics.astype(np.float16))
    elif name == "item-knn":
        sparse.save_npz(output / "saves.npz", model.train)
    elif name == "wmf":
        np.save(output / "wmf.npy", model.item_factors_.astype(np.float16))
    elif name.startswith("node2vec"):
        vectors = np.where(model.has_embedding_[:, None], model.embeddings_, 0)
        np.save(output / f"{name}.npy", vectors.astype(np.float16))
    elif name in ("gcn-bce", "nograph-bce", "gcn-bce-add"):
        np.save(output / f"{name}.npy", model.embeddings_.astype(np.float16))
        layer = model.network_.layers[0]
        weights = np.vstack((layer.weight.detach().numpy().T, layer.bias.detach().numpy()))
        np.save(output / f"{name}_layer.npy", weights.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output / "demo_data"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_citeulike_a("data/raw/citeulike-a")
    split = load_split("artifacts/splits/warm/fold_0", dataset)
    papers = dataset.papers

    with gzip.open(output / "papers.json.gz", "wt", encoding="utf-8") as f:
        json.dump([[p.title, p.abstract] for p in papers], f, ensure_ascii=False)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b")
    counts = vectorizer.fit_transform([p.text for p in papers])
    sparse.save_npz(output / "counts.npz", counts.astype(np.int32))
    vocabulary = {term: int(i) for term, i in vectorizer.vocabulary_.items()}
    with gzip.open(output / "vocabulary.json.gz", "wt", encoding="utf-8") as f:
        json.dump(sorted(vocabulary, key=vocabulary.get), f, ensure_ascii=False)
    reference = {"adapter_map": adapter_map(args.features, papers, output), "top": {}}

    rng = np.random.default_rng(1)
    users = np.sort(rng.choice(split.evaluation_users("test"), 40, replace=False))
    reference["profiles"] = [split.train[u].indices.tolist() for u in users]
    for name in MODELS:
        if name == "wmf-unmasked":
            continue
        model = make_factory(name, "warm", dataset, args.features, SETTINGS.get(name, {}))()
        model.fit(papers, split)
        export_model(name, model, output, vocabulary)
        reference["top"][name] = [
            [i for i, _ in model.recommend(split.train[u].indices, k=20, user_index=int(u))]
            for u in users
        ]
        print(f"{name} exported", flush=True)
    (args.output / "reference.json").write_text(json.dumps(reference), encoding="utf-8")


if __name__ == "__main__":
    main()
