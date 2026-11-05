"""Run fixed lexical baselines on the saved warm/cold folds.

Example: python -m paperrec.baselines --scenario both --output results/lexical-validation
"""

import argparse
from functools import partial
from pathlib import Path

from .data import load_citeulike_a
from .experiment import run_experiment
from .models import BM25Recommender, TfidfRecommender
from .splits import load_folds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--scenario", choices=("warm", "cold", "both"), default="both")
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/features"))
    parser.add_argument(
        "--no-cache", action="store_true", help="Recompute without reading/writing caches"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--sublinear-tf", action="store_true")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args(argv)
    scenarios = ("warm", "cold") if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        destination = args.output / scenario
        if destination.exists() and any(destination.iterdir()):
            parser.error(f"Choose a new or empty output directory: {destination}")
    common = {
        "seed": args.seed,
        "cache_root": None if args.no_cache else args.cache_root,
        "min_df": args.min_df,
        "max_df": args.max_df,
    }
    models = {
        "tfidf": partial(TfidfRecommender, sublinear_tf=args.sublinear_tf, **common),
        "bm25": partial(BM25Recommender, k1=args.k1, b=args.b, **common),
    }
    # Check constructor settings before loading the full dataset or writing results.
    for factory in models.values():
        factory()
    dataset = load_citeulike_a(args.data)
    for scenario in scenarios:
        print(f"Evaluating {scenario} {args.partition}: TF-IDF and BM25", flush=True)
        folds = load_folds(args.splits / scenario, dataset)
        report = run_experiment(
            dataset, folds, models, args.output / scenario, partition=args.partition
        )
        for name, result in report["models"].items():
            scores = result["aggregate"]
            print(
                f"{scenario} {name}: Recall@20={scores['recall@20']['mean']:.4f}, "
                f"NDCG@20={scores['ndcg@20']['mean']:.4f}",
                flush=True,
            )
        del folds


if __name__ == "__main__":
    main()
