"""Hyperparameter tuning and grid search for semantic recommenders (LSI & LDA).

Usage:
    python -m paperrec.tune --scenario both --partition validation --output results/tuning
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from .data import load_citeulike_a
from .experiment import run_experiment
from .models import LDARecommender, LSIRecommender, SPECTER2Recommender
from .splits import load_folds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--scenario", choices=("warm", "cold", "both"), default="both")
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("results/tuning"))
    parser.add_argument("--include-specter2", action="store_true", help="Include SPECTER2 in the evaluation run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    scenarios = ("warm", "cold") if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        dest = args.output / scenario
        if dest.exists() and any(dest.iterdir()):
            parser.error(f"Target directory exists and is nonempty: {dest}")

    # Build model grid for LSI (K=100, 200, 300) and LDA (K=50, 100, 150)
    models = {}
    for k in (100, 200, 300):
        models[f"lsi_k{k}"] = partial(LSIRecommender, n_components=k, seed=args.seed)

    for k in (50, 100, 150):
        models[f"lda_k{k}"] = partial(LDARecommender, n_topics=k, seed=args.seed)

    if args.include_specter2:
        models["specter2"] = partial(SPECTER2Recommender, seed=args.seed)

    print(f"Registered {len(models)} model configurations for tuning: {list(models.keys())}", flush=True)

    dataset = load_citeulike_a(args.data)
    for scenario in scenarios:
        print(f"\n=======================================================", flush=True)
        print(f"Tuning on {scenario.upper()} ({args.partition.upper()})", flush=True)
        print(f"=======================================================", flush=True)
        folds = load_folds(args.splits / scenario, dataset)
        report = run_experiment(
            dataset,
            folds,
            models,
            args.output / scenario,
            partition=args.partition,
        )

        print("\n--- Summary Ranking Metrics ---", flush=True)
        for name, res in report["models"].items():
            agg = res["aggregate"]
            print(
                f"[{scenario}] {name:12s} | "
                f"Recall@20 = {agg['recall@20']['mean']:.4f} (±{agg['recall@20']['std']:.4f}) | "
                f"NDCG@20 = {agg['ndcg@20']['mean']:.4f} (±{agg['ndcg@20']['std']:.4f}) | "
                f"MRR = {agg['mrr']['mean']:.4f}",
                flush=True,
            )
        del folds


if __name__ == "__main__":
    main()
