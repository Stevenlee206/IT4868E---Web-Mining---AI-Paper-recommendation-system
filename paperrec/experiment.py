"""Run future models on the same folds and record metrics and efficiency."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from .evaluation import _validate_ks, evaluate
from .splits import dataset_fingerprint, validate_folds


class MemorySampler:
    """Sample total process RSS every 50 ms, including native arrays (not GPU)."""

    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def run_experiment(dataset, folds, models, output, partition="validation", ks=(5, 10, 20)):
    """Run all folds for a mapping {name: zero-argument model factory}.

    Each factory must return a fresh Recommender. Configure seeds and parameters
    in the factory. Use validation to select settings, then run test once with
    fixed settings. The runner never refits on validation or selects on test.
    """
    ks = _validate_ks(ks)
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test")
    if not models or not all(isinstance(name, str) and callable(fn) for name, fn in models.items()):
        raise ValueError("models must map names to zero-argument model factories")
    validate_folds(folds, dataset)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset.statistics(),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "scenario": folds[0].scenario,
        "split_seed": folds[0].seed,
        "partition": partition,
        "ks": list(ks),
        "folds": [fold.statistics() for fold in folds],
        "models": {},
    }
    for name, factory in models.items():
        results = []
        for split in folds:
            with MemorySampler() as memory:
                start = perf_counter()
                model = factory()
                model.fit(dataset.papers, split)
                training_seconds = perf_counter() - start
                metrics, per_user = evaluate(model, dataset, split, partition, ks)
            results.append(
                {
                    "fold": split.fold,
                    "split_fingerprint": split.fingerprint(),
                    "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                    "model_seed": int(model.seed),
                    "model_config": model.get_config(),
                    "feature_cache_hit": getattr(model, "cache_hit_", None),
                    "feature_cache_path": str(model.cache_path_)
                    if getattr(model, "cache_path_", None) is not None
                    else None,
                    "training_seconds": training_seconds,
                    "process_rss_peak_mb": memory.peak / 1024**2,
                    "process_rss_increase_mb": (memory.peak - memory.baseline) / 1024**2,
                    "metrics": metrics,
                    "per_user": per_user,
                }
            )
            del model
        # Equal-weight fold means and sample standard deviations, not pooled users.
        values = [
            {
                **row["metrics"],
                **{
                    key: row[key]
                    for key in (
                        "training_seconds",
                        "process_rss_peak_mb",
                        "process_rss_increase_mb",
                    )
                },
            }
            for row in results
        ]
        aggregate = {
            key: {
                "mean": float(np.mean([row[key] for row in values])),
                "std": float(np.std([row[key] for row in values], ddof=1)),
            }
            for key in values[0]
            if key != "under_3_seconds_p95"
        }
        report["models"][name] = {"folds": results, "aggregate": aggregate}
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
