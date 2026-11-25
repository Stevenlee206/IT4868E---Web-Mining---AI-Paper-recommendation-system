"""Run Simplified PR-SGCN and its references on the saved warm/cold folds.

Example: python -m paperrec.run_gcn --output results/gcn-validation

Models: specter2 (raw features), gcn-margin, gcn-bce, and nograph-margin/nograph-bce
(same network with self-loops only, showing what the citation graph adds). Simplified
PR-SGCN is an adaptation to CiteULike-a, not a full reproduction of Guo et al. (2025).
Select settings on validation, then run test once with those fixed settings.

Ablations run only when named in --models; each changes one thing in gcn-bce:
    gcn-bce-add, gcn-bce-concat: keep each paper's own features beside the propagated ones
    gcn-bce-add-l2, gcn-bce-add-row: also L2-normalize or row-normalize the propagated input
    gcn-bce-heldout, gcn-bce-add-heldout: cold only, also use edges within a held-out partition
    gcn-bce-drop25, -drop50, -drop75: remove that share of the citation edges
    gcn-bce-knn5, -knn20: replace citations with SPECTER2 nearest train papers
"""

import argparse
from functools import partial
from pathlib import Path

from .data import load_citeulike_a
from .experiment import run_experiment
from .models.gcn import SimplifiedPRSGCNRecommender, Specter2Recommender
from .specter import specter2_features
from .splits import load_folds

MODEL_NAMES = ("specter2", "gcn-margin", "gcn-bce", "nograph-margin", "nograph-bce")
ABLATIONS = {
    "gcn-bce-add": {"residual": "add"},
    "gcn-bce-concat": {"residual": "concat"},
    "gcn-bce-add-l2": {"residual": "add", "input_norm": "l2"},
    "gcn-bce-add-row": {"residual": "add", "input_norm": "row"},
    "gcn-bce-heldout": {"include_heldout_edges": True},
    "gcn-bce-add-heldout": {"residual": "add", "include_heldout_edges": True},
    "gcn-bce-drop25": {"edge_dropout": 0.25},
    "gcn-bce-drop50": {"edge_dropout": 0.5},
    "gcn-bce-drop75": {"edge_dropout": 0.75},
    "gcn-bce-knn5": {"graph_source": "knn", "knn_k": 5},
    "gcn-bce-knn20": {"graph_source": "knn", "knn_k": 20},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--scenario", choices=("warm", "cold", "both"), default="both")
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=(*MODEL_NAMES, *ABLATIONS), default=list(MODEL_NAMES)
    )
    parser.add_argument("--feature-cache", type=Path, default=Path("artifacts/features/specter2"))
    parser.add_argument("--include-heldout-edges", action="store_true", help="Ablation only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-users", type=int, default=512)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--max-profile-size", type=int, default=50)
    args = parser.parse_args(argv)

    scenarios = ("warm", "cold") if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        destination = args.output / scenario
        if destination.exists() and any(destination.iterdir()):
            parser.error(f"Choose a new or empty output directory: {destination}")

    dataset = load_citeulike_a(args.data)
    features = specter2_features(dataset.papers, cache_root=args.feature_cache)
    common = {
        "hidden_size": args.hidden_size,
        "output_size": args.output_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "margin": args.margin,
        "scale": args.scale,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_users": args.batch_users,
        "num_negatives": args.num_negatives,
        "max_profile_size": args.max_profile_size,
        "seed": args.seed,
    }
    network = partial(SimplifiedPRSGCNRecommender, features, dataset.citations, **common)
    factories = {
        "specter2": partial(Specter2Recommender, features, seed=args.seed),
        "gcn-margin": partial(
            network, loss="margin", include_heldout_edges=args.include_heldout_edges
        ),
        "gcn-bce": partial(network, loss="bce", include_heldout_edges=args.include_heldout_edges),
        "nograph-margin": partial(network, loss="margin", use_graph=False),
        "nograph-bce": partial(network, loss="bce", use_graph=False),
        **{name: partial(network, loss="bce", **change) for name, change in ABLATIONS.items()},
    }
    models = {name: factories[name] for name in args.models}
    for scenario in scenarios:
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
