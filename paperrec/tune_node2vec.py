"""Grid-search Node2Vec p and q on the validation partition of one fold.

Example: python -m paperrec.tune_node2vec --output results/node2vec-tuning.json

Only fold 0 selects settings by default, so folds 1-4 can then compare the chosen
setting with the default without selection bias. Run the choice with run_node2vec.
"""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

from .data import load_citeulike_a
from .evaluation import evaluate
from .models import Node2VecRecommender
from .splits import load_split

PLANS = (("warm", "citation"), ("warm", "bipartite"), ("cold", "citation"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--p", type=float, nargs="+", default=[0.25, 1.0, 4.0])
    parser.add_argument("--q", type=float, nargs="+", default=[0.25, 1.0, 4.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Choose a new output file: {args.output}")

    dataset = load_citeulike_a(args.data)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario, graph in PLANS:
        split = load_split(args.splits / scenario / f"fold_{args.fold}", dataset)
        for p, q in product(args.p, args.q):
            start = perf_counter()
            model = Node2VecRecommender(
                dataset.citations, graph=graph, p=p, q=q, seed=args.seed, cache_root=None
            ).fit(dataset.papers, split)
            seconds = perf_counter() - start
            metrics, _ = evaluate(model, dataset, split, "validation", (20,))
            rows.append(
                {
                    "scenario": scenario,
                    "graph": graph,
                    "fold": args.fold,
                    "p": p,
                    "q": q,
                    "recall@20": metrics["recall@20"],
                    "ndcg@20": metrics["ndcg@20"],
                    "training_seconds": seconds,
                }
            )
            print(
                f"{scenario} {graph} p={p} q={q}: Recall@20={metrics['recall@20']:.4f}, "
                f"NDCG@20={metrics['ndcg@20']:.4f}, {seconds:.0f}s",
                flush=True,
            )
            args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for scenario, graph in PLANS:
        options = [row for row in rows if (row["scenario"], row["graph"]) == (scenario, graph)]
        best = max(options, key=lambda row: row["recall@20"])
        print(f"Best {scenario} {graph}: p={best['p']} q={best['q']}", flush=True)


if __name__ == "__main__":
    main()
