"""Run Node2Vec on the saved warm/cold folds.

Example: python -m paperrec.run_node2vec --output results/node2vec-validation

The bipartite graph runs only for warm: held-out cold papers have no edges in it.
Select settings on validation, then run test once with those fixed settings.
"""

import argparse
from functools import partial
from pathlib import Path

from .data import load_citeulike_a
from .experiment import run_experiment
from .models import Node2VecRecommender
from .splits import load_folds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--scenario", choices=("warm", "cold", "both"), default="both")
    parser.add_argument("--graph", choices=("citation", "bipartite", "both"), default="both")
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/features/node2vec"))
    parser.add_argument(
        "--no-cache", action="store_true", help="Train without reading/writing caches"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--walk-length", type=int, default=80)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, default=1, help="More than 1 is faster but not reproducible"
    )
    args = parser.parse_args(argv)

    scenarios = ("warm", "cold") if args.scenario == "both" else (args.scenario,)
    graphs = ("citation", "bipartite") if args.graph == "both" else (args.graph,)
    plans = {
        scenario: [g for g in graphs if not (scenario == "cold" and g == "bipartite")]
        for scenario in scenarios
    }
    if not any(plans.values()):
        parser.error("The bipartite graph cannot rank held-out cold papers")
    for scenario in scenarios:
        destination = args.output / scenario
        if plans[scenario] and destination.exists() and any(destination.iterdir()):
            parser.error(f"Choose a new or empty output directory: {destination}")

    dataset = load_citeulike_a(args.data)
    common = {
        "dimensions": args.dimensions,
        "num_walks": args.num_walks,
        "walk_length": args.walk_length,
        "window": args.window,
        "p": args.p,
        "q": args.q,
        "negative": args.negative,
        "epochs": args.epochs,
        "workers": args.workers,
        "seed": args.seed,
        "cache_root": None if args.no_cache else args.cache_root,
    }
    for scenario, scenario_graphs in plans.items():
        if not scenario_graphs:
            continue
        models = {
            f"node2vec-{name}": partial(
                Node2VecRecommender, dataset.citations, graph=name, **common
            )
            for name in scenario_graphs
        }
        print(f"Evaluating {scenario} {args.partition}: {', '.join(models)}", flush=True)
        folds = load_folds(args.splits / scenario, dataset)
        report = run_experiment(
            dataset, folds, models, args.output / scenario, partition=args.partition
        )
        for name, result in report["models"].items():
            scores = result["aggregate"]
            print(
                f"{scenario} {name}: "
                f"Recall@20={scores['recall@20']['mean']:.4f}"
                f"±{scores['recall@20']['std']:.4f}, "
                f"NDCG@20={scores['ndcg@20']['mean']:.4f}"
                f"±{scores['ndcg@20']['std']:.4f}",
                flush=True,
            )
        del folds


if __name__ == "__main__":
    main()
