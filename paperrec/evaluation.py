"""Macro user metrics over the candidate pools defined by each saved split."""

from time import perf_counter

import numpy as np


def _validate_ks(ks):
    ks = tuple(ks)
    if not ks or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1 for k in ks
    ):
        raise ValueError("Specify at least one positive integer metric cutoff")
    return tuple(sorted({int(k) for k in ks}))


def ranking_metrics(ranking, relevant, ks=(5, 10, 20)):
    """Compute MRR and metrics at K for one user.

    Precision uses K even when fewer candidates exist. AP@K divides by
    min(K, number of relevant papers). MRR uses the entire supplied ranking.
    """
    ks = _validate_ks(ks)
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Metrics need nonempty relevance")
    ranking = list(ranking)
    if len(set(ranking)) != len(ranking):
        raise ValueError("Ranking contains duplicates")
    hits = np.array([item in relevant for item in ranking], dtype=float)
    positions = np.flatnonzero(hits)
    result = {"mrr": float(1 / (positions[0] + 1)) if len(positions) else 0.0}
    for k in ks:
        top = hits[:k]
        count = top.sum()
        precision, recall = count / k, count / len(relevant)
        discounts = 1 / np.log2(np.arange(2, len(top) + 2))
        ideal = (1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum()
        average_precision = (top * np.cumsum(top) / np.arange(1, len(top) + 1)).sum()
        result.update(
            {
                f"precision@{k}": float(precision),
                f"recall@{k}": float(recall),
                f"ndcg@{k}": float((top * discounts).sum() / ideal),
                f"map@{k}": float(average_precision / min(k, len(relevant))),
                f"f1@{k}": float(2 * precision * recall / (precision + recall))
                if precision + recall
                else 0.0,
            }
        )
    return result


def evaluate(model, dataset, split, partition="test", ks=(5, 10, 20)):
    """Evaluate training profiles against split-specific validation/test pools.

    Users need both training history and positive targets. Topic diversity is
    deferred to the proposal's session extension, which needs topic metadata.
    """
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test")
    ks = _validate_ks(ks)
    if dataset.interactions.shape != split.train.shape:
        raise ValueError("Dataset and split shapes differ")
    if len(dataset.papers) != split.train.shape[1]:
        raise ValueError("Paper count must match the split columns")
    targets = getattr(split, partition)
    users = split.evaluation_users(partition)
    if not len(users):
        raise ValueError("No evaluable users: each needs a training profile and held-out saves")

    per_user, durations, candidate_counts = [], [], []
    covered = {k: set() for k in ks}
    for user in users:
        profile = split.train[user].indices
        relevant = targets[user].indices
        candidates = split.candidate_items(int(user), partition)
        if not np.isin(relevant, candidates).all():
            raise ValueError("Held-out positives must belong to the ranking candidate pool")
        start = perf_counter()
        ranking = [
            item
            for item, _ in model.recommend(
                profile, k=len(candidates), user_index=int(user), candidates=candidates
            )
        ]
        durations.append(perf_counter() - start)
        # Full permutations are required for untruncated MRR and fair comparison.
        if len(ranking) != len(candidates) or set(ranking) != set(candidates):
            raise ValueError("Model must rank every split candidate exactly once")
        candidate_counts.append(len(candidates))
        metrics = ranking_metrics(ranking, relevant, ks)
        per_user.append(
            {
                "user_id": int(user),
                "metrics": metrics,
                "top_items": [int(item) for item in ranking[: max(ks)]],
            }
        )
        for k in ks:
            covered[k].update(ranking[:k])

    summary = {
        key: float(np.mean([row["metrics"][key] for row in per_user]))
        for key in per_user[0]["metrics"]
    }
    for k in ks:
        summary[f"catalog_coverage@{k}"] = len(covered[k]) / len(dataset.papers)
    has_history = np.diff(split.train.indptr) > 0
    has_target = np.diff(targets.indptr) > 0
    summary.update(
        {
            "evaluated_users": len(per_user),
            "skipped_no_profile": int(np.count_nonzero(~has_history)),
            "skipped_no_targets": int(np.count_nonzero(has_history & ~has_target)),
            "candidate_count_min": min(candidate_counts),
            "candidate_count_mean": float(np.mean(candidate_counts)),
            "candidate_count_max": max(candidate_counts),
            "response_seconds_mean": float(np.mean(durations)),
            "response_seconds_p95": float(np.percentile(durations, 95)),
            "response_seconds_max": float(max(durations)),
            "under_3_seconds_p95": bool(np.percentile(durations, 95) < 3),
        }
    )
    return summary, per_user
