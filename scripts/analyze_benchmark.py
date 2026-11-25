"""Summarize benchmark results and analyze gcn-bce-add against nograph-bce when cold.

    python scripts/analyze_benchmark.py

Reads results/benchmark, trains nothing, and writes summary.csv, folds.csv and
error_analysis.json there, plus LaTeX fragments in report/generated.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paperrec.analyze_degree import BUCKETS, degree_buckets
from paperrec.benchmark import KS, MODELS, WARM_ONLY
from paperrec.data import load_citeulike_a
from paperrec.graph import citation_inference_graph
from paperrec.splits import load_folds

RESULTS, GENERATED = ROOT / "results" / "benchmark", ROOT / "report" / "generated"
GRAPH, PLAIN, NONE = "gcn-bce-add", "gcn-bce", "nograph-bce"
CF = ("item-knn", "wmf")
LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "lsi": "LSI ($K{=}300$)",
    "lda": "LDA ($K{=}150$)",
    "item-knn": "ItemKNN",
    "wmf": "WMF (masked)",
    "wmf-unmasked": "WMF (unmasked control)",
    "specter2-base": "SPECTER2 base",
    "specter2-adapter": "SPECTER2 + adapter",
    "node2vec-citation": "Node2Vec citation",
    "node2vec-bipartite": "Node2Vec bipartite",
    "gcn-bce": "Plain GCN (\\texttt{gcn-bce})",
    "nograph-bce": "No graph (\\texttt{nograph-bce})",
    "gcn-bce-add": "Residual GCN (\\texttt{gcn-bce-add})",
}
METRICS = [f"{name}@{k}" for name in ("recall", "ndcg") for k in KS] + ["mrr"]


def load_reports():
    reports = {}
    for scenario in ("warm", "cold"):
        for model in MODELS:
            if scenario == "cold" and model in WARM_ONLY:
                continue
            path = RESULTS / scenario / f"{model}.json"
            reports[scenario, model] = json.loads(path.read_text(encoding="utf-8"))
    for scenario in ("warm", "cold"):
        prints = {
            tuple(fold["split_fingerprint"] for fold in report["folds"])
            for (s, _), report in reports.items()
            if s == scenario
        }
        if len(prints) != 1:
            raise ValueError(f"{scenario} models were evaluated on different splits")
    return reports


def write_tables(reports):
    with (RESULTS / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario", "partition", "model"]
            + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
        )
        for (scenario, model), report in reports.items():
            for partition in ("validation", "test"):
                cells = [report[partition][m][s] for m in METRICS for s in ("mean", "std")]
                writer.writerow([scenario, partition, model] + [f"{c:.6f}" for c in cells])
    with (RESULTS / "folds.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "partition", "model", "fold"] + METRICS)
        for (scenario, model), report in reports.items():
            for partition in ("validation", "test"):
                for fold in report["folds"]:
                    cells = [f"{fold[partition][m]:.6f}" for m in METRICS]
                    writer.writerow([scenario, partition, model, fold["fold"]] + cells)

    GENERATED.mkdir(parents=True, exist_ok=True)
    columns = ("recall@10", "recall@20", "recall@50", "ndcg@10", "ndcg@20", "mrr")
    for scenario in ("warm", "cold"):
        for partition in ("validation", "test"):
            models = [m for m in MODELS if (scenario, m) in reports]
            table = {m: reports[scenario, m][partition] for m in models}
            best = {c: max(table[m][c]["mean"] for m in models) for c in columns}
            lines = []
            for model in models:
                cells = []
                for c in columns:
                    value = table[model][c]
                    text = f"{value['mean']:.4f}"
                    if value["mean"] == best[c]:
                        text = f"\\textbf{{{text}}}"
                    if c == "recall@20":
                        text += f" {{\\scriptsize$\\pm${value['std']:.4f}}}"
                    cells.append(text)
                mark = "$^\\dagger$" if scenario == "cold" and model in CF else ""
                lines.append(f"{LABELS[model]}{mark} & " + " & ".join(cells) + r" \\")
                if model in ("bm25", "lda", "specter2-adapter", "node2vec-citation") and (
                    model != "node2vec-citation" or scenario == "cold"
                ):
                    lines.append(r"\addlinespace")
                if model == ("wmf-unmasked" if scenario == "warm" else "wmf"):
                    lines.append(r"\addlinespace")
                if model == "node2vec-bipartite":
                    lines.append(r"\addlinespace")
            (GENERATED / f"{scenario}_{partition}.tex").write_text("\n".join(lines) + "\n")


def write_overview(reports):
    """Warm and cold test results side by side, for the slides and the paper's main table."""
    columns = [(s, m) for s in ("warm", "cold") for m in ("recall@20", "ndcg@20", "mrr")]
    chosen = [m for m in MODELS if m != "wmf-unmasked"]
    best = {
        (s, m): max(reports[s, x]["test"][m]["mean"] for x in chosen if (s, x) in reports)
        for s, m in columns
    }
    lines = []
    for model in chosen:
        cells = []
        for scenario, metric in columns:
            if (scenario, model) not in reports:
                cells.append("--")
                continue
            value = reports[scenario, model]["test"][metric]["mean"]
            text = f"{value:.4f}"
            cells.append(f"\\textbf{{{text}}}" if value == best[scenario, metric] else text)
        mark = "$^\\dagger$" if model in CF else ""
        lines.append(f"{LABELS[model]}{mark} & " + " & ".join(cells) + r" \\")
        if model in ("bm25", "lda", "wmf", "specter2-adapter", "node2vec-bipartite"):
            lines.append(r"\addlinespace")
    (GENERATED / "overview.tex").write_text("\n".join(lines) + "\n")


def paired(reports, scenario, partition, a, b):
    """Fold-level and user-level paired differences a - b."""
    result = {}
    arrays = {m: np.load(RESULTS / scenario / f"{m}.npz") for m in (a, b)}
    for metric in ("recall@20", "ndcg@20", "mrr"):
        fold_gaps = [
            x[partition][metric] - y[partition][metric]
            for x, y in zip(
                reports[scenario, a]["folds"], reports[scenario, b]["folds"], strict=True
            )
        ]
        totals, counts = np.zeros(6000), np.zeros(6000)
        better = worse = pairs = 0
        for fold in range(len(fold_gaps)):
            users = arrays[a][f"{partition}_{fold}_users"]
            if not np.array_equal(users, arrays[b][f"{partition}_{fold}_users"]):
                raise ValueError("Models were evaluated on different users")
            gap = (
                arrays[a][f"{partition}_{fold}_{metric}"].astype(float)
                - arrays[b][f"{partition}_{fold}_{metric}"]
            )
            np.add.at(totals, users, gap)
            np.add.at(counts, users, 1)
            better, worse, pairs = (
                better + (gap > 0).sum(),
                worse + (gap < 0).sum(),
                pairs + len(gap),
            )
        user_gaps = totals[counts > 0] / counts[counts > 0]  # One value per user over folds.
        rng = np.random.default_rng(0)
        draws = [
            user_gaps[rng.integers(0, len(user_gaps), len(user_gaps))].mean() for _ in range(2000)
        ]
        result[metric] = {
            "mean_gap": float(np.mean(fold_gaps)),
            "folds_above": int(np.sum(np.array(fold_gaps) > 0)),
            "user_bootstrap_95ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            "user_folds_better": float(better / pairs),
            "user_folds_worse": float(worse / pairs),
        }
    return result


def held_out_pairs(dataset, folds, scenario):
    """One row per held-out test save: user, paper, citation bucket, link to profile."""
    rows = {}
    for split in folds:
        graph = citation_inference_graph(dataset.citations, split)
        degrees = graph.getnnz(axis=1)
        linked = (split.train @ graph.T).tocsr()  # Profile papers that the paper is linked to.
        users = split.evaluation_users("test")
        test = split.test[users]
        pair_users = np.repeat(users, np.diff(test.indptr))
        items = test.indices
        links = np.asarray(linked[pair_users, items]).ravel()
        rows[split.fold] = {
            "users": pair_users,
            "items": items,
            "degree": degrees[items],
            "links": links,
            "item_degrees": degrees,
            "pool": split.test_items,
        }
    return rows


def hits(scenario, model, rows):
    arrays = np.load(RESULTS / scenario / f"{model}.npz")
    result = {}
    for fold, row in rows.items():
        users, top = arrays[f"test_{fold}_users"], arrays[f"test_{fold}_top"]
        position = np.full(6000, -1)
        position[users] = np.arange(len(users))
        result[fold] = (top[position[row["users"]]] == row["items"][:, None]).any(axis=1)
    return result


def grouped(rows, hit, group_of, size):
    """Hit rate per group and fold, shape (folds, groups), plus pair counts."""
    rates, counts = [], []
    for fold, row in rows.items():
        group = group_of(row)
        n = np.bincount(group, minlength=size)
        rates.append(np.bincount(group, weights=hit[fold], minlength=size) / np.maximum(n, 1))
        counts.append(n)
    return np.array(rates), np.array(counts)


def link_group(row):
    """0: no citations, 1: citations but none to the profile, 2: one link, 3: two or more."""
    links = np.minimum(row["links"], 2).astype(int)
    return np.where(row["degree"] == 0, 0, 1 + links)


def analyze(dataset, reports):
    output = {}
    for scenario in ("warm", "cold"):
        folds = load_folds(ROOT / "artifacts" / "splits" / scenario, dataset)
        rows = held_out_pairs(dataset, folds, scenario)
        section = {"paired_test": {}, "paired_validation": {}}
        for partition in ("validation", "test"):
            for a, b in (
                (GRAPH, NONE),
                (PLAIN, NONE),
                (NONE, "specter2-adapter"),
                ("specter2-adapter", "specter2-base"),
            ):
                section[f"paired_{partition}"][f"{a} - {b}"] = paired(
                    reports, scenario, partition, a, b
                )
        tables = {}
        all_hits = {}
        for model in ("specter2-adapter", NONE, PLAIN, GRAPH):
            all_hits[model] = hits(scenario, model, rows)
            by_degree, n_degree = grouped(
                rows, all_hits[model], lambda r: degree_buckets(r["degree"]), len(BUCKETS)
            )
            by_link, n_link = grouped(rows, all_hits[model], link_group, 4)
            arrays = np.load(RESULTS / scenario / f"{model}.npz")
            slots = np.array(
                [
                    np.bincount(
                        degree_buckets(row["item_degrees"][arrays[f"test_{fold}_top"]]).ravel(),
                        minlength=len(BUCKETS),
                    )
                    for fold, row in rows.items()
                ]
            )
            tables[model] = {
                "hit_by_degree": by_degree.mean(0).tolist(),
                "hit_by_degree_folds": by_degree.tolist(),
                "hit_by_link": by_link.mean(0).tolist(),
                "hit_by_link_folds": by_link.tolist(),
                "top20_share_by_degree": (slots.sum(0) / slots.sum()).tolist(),
            }
        section["models"] = tables
        section["degree_buckets"] = BUCKETS
        section["pairs_by_degree"] = n_degree.sum(0).tolist()
        section["pairs_by_link"] = n_link.sum(0).tolist()
        pool = np.concatenate([degree_buckets(r["item_degrees"][r["pool"]]) for r in rows.values()])
        section["pool_share_by_degree"] = (
            np.bincount(pool, minlength=len(BUCKETS)) / len(pool)
        ).tolist()
        gained = lost = both = neither = 0
        for fold in rows:
            a, b = all_hits[GRAPH][fold], all_hits[NONE][fold]
            gained, lost = gained + (a & ~b).sum(), lost + (~a & b).sum()
            both, neither = both + (a & b).sum(), neither + (~a & ~b).sum()
        section["hit_changes"] = {
            "only_graph": int(gained),
            "only_nograph": int(lost),
            "both": int(both),
            "neither": int(neither),
        }
        if scenario == "cold":
            section["cases"] = cases(dataset, folds[0], rows[0], all_hits, scenario)
        output[scenario] = section
        del folds
    return output


def cases(dataset, split, row, all_hits, scenario):
    """Fold 0 users with the largest Recall@20 loss and gain; ties go to the lower user ID.

    Users need at least five test saves, so that one hit does not decide the choice.
    """
    arrays = {m: np.load(RESULTS / scenario / f"{m}.npz") for m in (GRAPH, NONE)}
    users = arrays[GRAPH]["test_0_users"]
    gap = arrays[GRAPH]["test_0_recall@20"] - arrays[NONE]["test_0_recall@20"]
    enough = np.diff(split.test.indptr)[users] >= 5
    result = {}
    for name, order in (
        ("loss", np.argsort(gap, kind="stable")),
        ("gain", np.argsort(-gap, kind="stable")),
    ):
        index = next(i for i in order if enough[i])
        user = int(users[index])
        mine = row["users"] == user
        result[name] = {
            "user": user,
            "recall_gap": float(gap[index]),
            "profile_size": int(split.train[user].nnz),
            "profile": [dataset.papers[i].title for i in split.train[user].indices[:5]],
            "papers": [
                {
                    "paper": int(item),
                    "title": dataset.papers[item].title,
                    "citations": int(degree),
                    "links_to_profile": int(links),
                    "hit_graph": bool(graph),
                    "hit_nograph": bool(none),
                }
                for item, degree, links, graph, none in zip(
                    row["items"][mine],
                    row["degree"][mine],
                    row["links"][mine],
                    all_hits[GRAPH][0][mine],
                    all_hits[NONE][0][mine],
                    strict=True,
                )
            ],
        }
    return result


def write_analysis_tables(output):
    labels = {
        "specter2-adapter": "SPECTER2 + adapter",
        NONE: "No graph",
        PLAIN: "Plain GCN",
        GRAPH: "Residual GCN",
    }
    cold = output["cold"]
    for key, source in (("degree", "hit_by_degree"), ("link", "hit_by_link")):
        lines = []
        values = {m: cold["models"][m][source] for m in labels}
        for model, label in labels.items():
            cells = [
                f"\\textbf{{{v:.4f}}}" if v == max(values[m][i] for m in labels) else f"{v:.4f}"
                for i, v in enumerate(values[model])
            ]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        gaps = np.array(cold["models"][GRAPH][f"{source}_folds"]) - np.array(
            cold["models"][NONE][f"{source}_folds"]
        )
        lines.append(r"\midrule")
        cells = [f"${g:+.4f}$" for g in gaps.mean(0)]
        lines.append("Residual $-$ no graph & " + " & ".join(cells) + r" \\")
        lines.append("Folds above & " + " & ".join(f"{w}/5" for w in (gaps > 0).sum(0)) + r" \\")
        counts = np.array(cold[f"pairs_by_{key}"])
        lines.append(
            "Share of pairs & "
            + " & ".join(f"{100 * c:.1f}\\%" for c in counts / counts.sum())
            + r" \\"
        )
        (GENERATED / f"cold_{key}.tex").write_text("\n".join(lines) + "\n")
    lines = ["bucket nograph plain residual"]
    for i, bucket in enumerate(BUCKETS):
        row = [cold["models"][m]["hit_by_degree"][i] for m in (NONE, PLAIN, GRAPH)]
        lines.append(f"{{{bucket.replace('-', '--')}}} " + " ".join(f"{v:.4f}" for v in row))
    (GENERATED / "cold_degree.dat").write_text("\n".join(lines) + "\n")
    lines = []
    for scenario in ("warm", "cold"):
        for pair, label in (
            (f"{GRAPH} - {NONE}", "Residual GCN $-$ no graph"),
            (f"{PLAIN} - {NONE}", "Plain GCN $-$ no graph"),
            (f"{NONE} - specter2-adapter", "No graph $-$ SPECTER2 + adapter"),
            ("specter2-adapter - specter2-base", "SPECTER2 adapter $-$ base"),
        ):
            row = output[scenario]["paired_test"][pair]
            cells = []
            for metric in ("recall@20", "ndcg@20"):
                m = row[metric]
                low, high = m["user_bootstrap_95ci"]
                cells += [
                    f"${m['mean_gap']:+.4f}$",
                    f"[${low:+.4f}$, ${high:+.4f}$]",
                    f"{m['folds_above']}/5",
                ]
            lines.append(f"{scenario.title()} & {label} & " + " & ".join(cells) + r" \\")
        if scenario == "warm":
            lines.append(r"\addlinespace")
    (GENERATED / "paired_test.tex").write_text("\n".join(lines) + "\n")


def main():
    reports = load_reports()
    write_tables(reports)
    write_overview(reports)
    output = analyze(load_citeulike_a(ROOT / "data" / "raw" / "citeulike-a"), reports)
    (RESULTS / "error_analysis.json").write_text(json.dumps(output, indent=1), encoding="utf-8")
    write_analysis_tables(output)
    print(f"Wrote {RESULTS} and {GENERATED}")


if __name__ == "__main__":
    main()
