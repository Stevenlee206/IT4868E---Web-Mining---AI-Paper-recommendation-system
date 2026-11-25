"""Evaluate every method on the same saved folds and partitions.

Each fit is scored on validation and test. Settings were chosen on validation, or are
chosen here by `tune` on the validation partition of warm fold 0, and never on test.

    python -m paperrec.benchmark tune --models item-knn wmf --output results/benchmark
    python -m paperrec.benchmark run --model tfidf --scenario warm --output results/benchmark
"""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import load_citeulike_a
from .evaluation import evaluate
from .splits import load_folds, load_split

KS = (5, 10, 20, 50)
TOP = 20
WARM_ONLY = ("wmf-unmasked", "node2vec-bipartite")
MODELS = (
    "tfidf",
    "bm25",
    "lsi",
    "lda",
    "item-knn",
    "wmf",
    "wmf-unmasked",
    "specter2-base",
    "specter2-adapter",
    "node2vec-citation",
    "node2vec-bipartite",
    "gcn-bce",
    "nograph-bce",
    "gcn-bce-add",
)
WMF_GRID = [
    {"factors": f, "alpha": a, "regularization": r}
    for f, a, r in product((64, 128), (5.0, 40.0), (1.0, 10.0, 100.0))
]
GRIDS = {
    "item-knn": [{"neighbors": k} for k in (50, 100, 200, 400)],
    "wmf": WMF_GRID,
    "wmf-unmasked": WMF_GRID,
}


def make_factory(name, scenario, dataset, features_dir, settings):
    """Return a zero-argument factory for one method."""
    features_dir = Path(features_dir)
    if name == "tfidf":
        from .models import TfidfRecommender

        return lambda: TfidfRecommender(cache_root=None)
    if name == "bm25":
        from .models import BM25Recommender

        return lambda: BM25Recommender(cache_root=None)
    if name == "lsi":
        from .models import LSIRecommender

        return lambda: LSIRecommender(n_components=300)
    if name == "lda":
        from .models import LDARecommender

        return lambda: LDARecommender(n_topics=150)
    if name == "item-knn":
        from .models.item_knn import ItemKNNRecommender

        return lambda: ItemKNNRecommender(**settings)
    if name in ("wmf", "wmf-unmasked"):
        from .models.wmf import WMFRecommender

        return lambda: WMFRecommender(masked=name == "wmf", **settings)
    if name == "specter2-base":
        from .models import SPECTER2Recommender

        return lambda: SPECTER2Recommender(cache_path=features_dir / "specter2_base.npy")
    if name.startswith("node2vec"):
        from .models import Node2VecRecommender

        graph = name.split("-")[1]
        p, q = {"citation": (4, 1), "bipartite": (0.25, 1)}[graph]
        if scenario == "cold":
            p, q = 1, 1
        return lambda: Node2VecRecommender(
            dataset.citations, graph=graph, p=p, q=q, cache_root=features_dir / "node2vec"
        )

    from .models.gcn import SimplifiedPRSGCNRecommender, Specter2Recommender
    from .specter import specter2_features

    features = specter2_features(dataset.papers, cache_root=features_dir / "specter2")
    if name == "specter2-adapter":
        return lambda: Specter2Recommender(features)
    options = {
        "gcn-bce": {},
        "nograph-bce": {"use_graph": False},
        "gcn-bce-add": {"residual": "add"},
    }[name]
    return lambda: SimplifiedPRSGCNRecommender(features, dataset.citations, loss="bce", **options)


def aggregate(folds, partition):
    """Equal-weight fold mean and sample standard deviation of each metric."""
    rows = [fold[partition] for fold in folds]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=1)),
        }
        for key in rows[0]
        if key != "under_3_seconds_p95"
    }


def run(args):
    dataset = load_citeulike_a(args.data)
    settings = {}
    if args.model in GRIDS:
        tuning = json.loads((args.output / "tuning" / f"{args.model}.json").read_text())
        settings = tuning["best"]
    factory = make_factory(args.model, args.scenario, dataset, args.features, settings)
    folds, arrays = [], {}
    for split in load_folds(args.splits / args.scenario, dataset):
        start = perf_counter()
        model = factory().fit(dataset.papers, split)
        row = {
            "fold": split.fold,
            "split_fingerprint": split.fingerprint(),
            "config": model.get_config(),
            "fit_seconds": perf_counter() - start,
            "fit_statistics": getattr(model, "fit_statistics_", None),
        }
        for partition in ("validation", "test"):
            row[partition], per_user = evaluate(model, dataset, split, partition, KS)
            prefix = f"{partition}_{split.fold}"
            arrays[f"{prefix}_users"] = np.array([r["user_id"] for r in per_user], dtype=np.int16)
            arrays[f"{prefix}_top"] = np.array(
                [r["top_items"][:TOP] for r in per_user], dtype=np.int16
            )
            for metric in ("recall@20", "ndcg@20", "mrr"):
                arrays[f"{prefix}_{metric}"] = np.array(
                    [r["metrics"][metric] for r in per_user], dtype=np.float32
                )
        folds.append(row)
        print(
            f"{args.scenario} {args.model} fold {split.fold}: "
            f"val R@20={row['validation']['recall@20']:.4f} "
            f"test R@20={row['test']['recall@20']:.4f}",
            flush=True,
        )
    destination = args.output / args.scenario
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "scenario": args.scenario,
        "folds": folds,
        "validation": aggregate(folds, "validation"),
        "test": aggregate(folds, "test"),
    }
    text = json.dumps(report, indent=1, allow_nan=False)
    (destination / f"{args.model}.json").write_text(text, encoding="utf-8")
    np.savez_compressed(destination / f"{args.model}.npz", **arrays)


def tune(args):
    dataset = load_citeulike_a(args.data)
    split = load_split(args.splits / "warm" / "fold_0", dataset)
    for name in args.models:
        rows = []
        for settings in GRIDS[name]:
            model = make_factory(name, "warm", dataset, args.features, settings)()
            model.fit(dataset.papers, split)
            metrics, _ = evaluate(model, dataset, split, "validation", (20,))
            rows.append(
                {
                    "settings": settings,
                    "ndcg@20": metrics["ndcg@20"],
                    "recall@20": metrics["recall@20"],
                }
            )
            print(f"tune {name} {settings}: {rows[-1]['ndcg@20']:.4f}", flush=True)
        best = max(rows, key=lambda row: row["ndcg@20"])["settings"]
        path = args.output / "tuning" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"best": best, "grid": rows}, indent=1), encoding="utf-8")


def export_demo(args):
    """Save the no-graph model's paper vectors and projection for the demo app."""
    dataset = load_citeulike_a(args.data)
    split = load_split(args.splits / "warm" / "fold_0", dataset)
    factory = make_factory("nograph-bce", "warm", dataset, args.features, {})
    model = factory().fit(dataset.papers, split)
    layer = model.network_.layers[0]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "demo_model.npz",
        papers=model.embeddings_.astype(np.float16),
        weight=layer.weight.detach().numpy(),
        bias=layer.bias.detach().numpy(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("run", run), ("tune", tune), ("export-demo", export_demo)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
        command.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
        command.add_argument("--features", type=Path, default=Path("artifacts/features"))
        command.add_argument("--output", type=Path, default=Path("results/benchmark"))
        if name == "run":
            command.add_argument("--model", choices=MODELS, required=True)
            command.add_argument("--scenario", choices=("warm", "cold"), required=True)
        elif name == "tune":
            command.add_argument("--models", nargs="+", choices=tuple(GRIDS), required=True)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
