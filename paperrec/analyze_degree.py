"""Break saved results down by how many citations each held-out paper receives.

Example:
    python -m paperrec.analyze_degree --results results/gcn-validation \
        --models gcn-bce nograph-bce --output results/degree-validation.json

No model is trained: this reads the top-20 lists already stored in results.json.
Degree is the number of messages a paper receives in the strict citation inference
graph, whichever graph the model used. Hit rate is pooled over held-out pairs, so it
weights active users more than the macro Recall@20. Degree is observational: it may
go with popularity or topic, so a gap in one bucket does not prove the graph caused it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_citeulike_a
from .graph import citation_inference_graph
from .splits import load_folds

BUCKETS = ("0", "1", "2-3", "4-7", "8+")
_BUCKET_STARTS = np.array([1, 2, 4, 8])


def degree_buckets(degrees) -> np.ndarray:
    """Return the BUCKETS index of each degree."""
    return np.searchsorted(_BUCKET_STARTS, degrees, side="right")


def degree_breakdown(per_user, targets, buckets) -> dict:
    """Count held-out pairs, top-k hits and recommended slots per bucket for one fold."""
    size = len(BUCKETS)
    positives, hits, slots = np.zeros((3, size), dtype=np.int64)
    for row in per_user:
        relevant = targets[row["user_id"]].indices
        top_items = np.asarray(row["top_items"], dtype=np.int64)
        positives += np.bincount(buckets[relevant], minlength=size)
        hits += np.bincount(buckets[top_items[np.isin(top_items, relevant)]], minlength=size)
        slots += np.bincount(buckets[top_items], minlength=size)
    return {
        "positives": positives.tolist(),
        "hits": hits.tolist(),
        "hit_rate": (hits / np.maximum(positives, 1)).tolist(),
        "top_share": (slots / max(slots.sum(), 1)).tolist(),
    }


def analyze(dataset, folds, report, model_names) -> dict:
    """Return {model: [one degree_breakdown per fold]} for a saved report."""
    fold_by_number = {split.fold: split for split in folds}
    result = {}
    for name in model_names:
        result[name] = []
        for row in report["models"][name]["folds"]:
            split = fold_by_number[row["fold"]]
            if split.fingerprint() != row["split_fingerprint"]:
                raise ValueError("Saved results do not match the current split")
            graph = citation_inference_graph(dataset.citations, split)
            buckets = degree_buckets(graph.getnnz(axis=1))
            targets = getattr(split, report["partition"])
            result[name].append(degree_breakdown(row["per_user"], targets, buckets))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--baseline", default="nograph-bce")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Choose a new output file: {args.output}")

    dataset = load_citeulike_a(args.data)
    output = {}
    for scenario in ("warm", "cold"):
        reports = [
            json.loads((path / scenario / "results.json").read_text(encoding="utf-8"))
            for path in args.results
        ]
        partitions = {report["partition"] for report in reports}
        if len(partitions) != 1:
            raise ValueError("All result folders must use the same partition")
        folds = load_folds(args.splits / scenario, dataset)
        output[scenario] = {}
        for report in reports:
            names = [name for name in args.models if name in report["models"]]
            output[scenario].update(analyze(dataset, folds, report, names))
        missing = set(args.models) - set(output[scenario])
        if missing:
            raise ValueError(f"Models not found for {scenario}: {sorted(missing)}")

        print(f"{scenario}: hit rate@20 by citation degree (mean of folds)")
        print("  " + f"{'model':20s}" + "".join(f"{bucket:>9s}" for bucket in BUCKETS))
        rates = {
            name: np.array([fold["hit_rate"] for fold in rows])
            for name, rows in output[scenario].items()
        }
        for name, values in rates.items():
            line = "".join(f"{value:9.4f}" for value in values.mean(axis=0))
            print(f"  {name:20s}{line}")
            if name != args.baseline and args.baseline in rates:
                gaps = values - rates[args.baseline]
                line = "".join(f"{value:+9.4f}" for value in gaps.mean(axis=0))
                print(f"  {'  minus baseline':20s}{line}")
                wins = (gaps > 0).sum(axis=0)
                print(f"  {'  folds above':20s}" + "".join(f"{win:7d}/{len(gaps)}" for win in wins))
        first = next(iter(output[scenario].values()))
        share = np.array([fold["positives"] for fold in first]).sum(axis=0)
        print(f"  {'held-out pairs':20s}" + "".join(f"{value:9d}" for value in share))
        for name, rows in output[scenario].items():
            line = np.mean([fold["top_share"] for fold in rows], axis=0)
            print(f"  {name + ' top share':20s}" + "".join(f"{value:9.4f}" for value in line))
        del folds

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
