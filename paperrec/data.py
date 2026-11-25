"""CiteULike-a loader for https://github.com/js05212/citeulike-a.

Requires Python 3.10+, NumPy, and SciPy. Interactions are binary float32 CSR.
Valid UTF-8 text is recovered per CSV field; undecodable fields are preserved
using Latin-1 and reported. This cannot reconstruct corrupted source text.
"""

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Paper:
    paper_id: int
    title: str
    abstract: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


@dataclass
class CiteULikeDataset:
    papers: list[Paper]  # papers[i].paper_id == interaction column i
    interactions: sparse.csr_matrix
    text_decode_fallback_ids: tuple[int, ...] = ()  # Zero-based paper IDs
    duplicate_interactions_removed: int = 0
    citations: sparse.csr_matrix | None = None  # Undirected paper-paper; see load_citations
    citation_self_loops_removed: int = 0

    def statistics(self) -> dict:
        n_users, n_items = self.interactions.shape
        total = n_users * n_items
        citation_statistics = {}
        if self.citations is not None:
            citation_statistics = {
                "citation_edges": self.citations.nnz // 2,
                "citation_isolated_papers": int(
                    np.count_nonzero(np.diff(self.citations.indptr) == 0)
                ),
                "citation_self_loops_removed": self.citation_self_loops_removed,
            }
        return {
            "users": n_users,
            "papers": n_items,
            "interactions": self.interactions.nnz,
            "sparsity": 1.0 - self.interactions.nnz / total if total else None,
            "missing_title": sum(not p.title for p in self.papers),
            "missing_abstract": sum(not p.abstract for p in self.papers),
            "missing_text": sum(not p.text for p in self.papers),
            "text_decode_fallback_papers": len(self.text_decode_fallback_ids),
            "duplicate_interactions_removed": self.duplicate_interactions_removed,
            **citation_statistics,
        }


def load_citations(path: str | Path, n_items: int) -> tuple[sparse.csr_matrix, int]:
    """Read citations.dat as (binary symmetric CSR without self-loops, loops removed).

    Line i holds: count neighbour_1 neighbour_2 ... (zero-based IDs). The source
    is fully symmetric and has no dates, so citing and cited papers cannot be
    told apart. Asymmetric input is rejected rather than silently repaired.
    """
    rows: list[int] = []
    cols: list[int] = []
    n_lines = 0
    with Path(path).expanduser().open(encoding="utf-8") as f:
        for item_id, line in enumerate(f):
            n_lines += 1
            try:
                values = [int(token) for token in line.split()]
            except ValueError as exc:
                raise ValueError(f"Non-integer value on citations.dat line {n_lines}") from exc
            if not values:
                raise ValueError(f"Blank row on citations.dat line {n_lines}; use '0'")
            count, *neighbours = values
            if count < 0 or count != len(neighbours):
                raise ValueError(f"Invalid neighbour count on citations.dat line {n_lines}")
            if any(item < 0 or item >= n_items for item in neighbours):
                raise ValueError(f"Paper ID out of range on citations.dat line {n_lines}")
            rows.extend([item_id] * count)
            cols.extend(neighbours)
    if n_lines != n_items:
        raise ValueError(f"citations.dat has {n_lines} rows for {n_items} papers")

    row_ids = np.asarray(rows, dtype=np.int64)
    col_ids = np.asarray(cols, dtype=np.int64)
    is_loop = row_ids == col_ids
    self_loops = int(np.count_nonzero(is_loop))
    mat = sparse.csr_matrix(
        (
            np.ones(len(row_ids) - self_loops, dtype=np.float32),
            (row_ids[~is_loop], col_ids[~is_loop]),
        ),
        shape=(n_items, n_items),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    mat.data.fill(1.0)
    mat.sort_indices()
    if (mat != mat.T).nnz:
        raise ValueError("citations.dat must be symmetric")
    return mat, self_loops


def _decode_text(value: str | None) -> tuple[str, bool]:
    """Recover UTF-8 from a lossless Latin-1 CSV read, if valid."""
    raw = (value or "").encode("latin-1")
    try:
        return raw.decode("utf-8").strip(), False
    except UnicodeDecodeError:
        return raw.decode("latin-1").strip(), True


def load_citeulike_a(directory: str | Path) -> CiteULikeDataset:
    directory = Path(directory).expanduser()
    papers_by_id: dict[int, Paper] = {}
    fallback_ids: set[int] = set()

    # Latin-1 preserves each byte while the CSV parser separates fields.
    with (directory / "raw-data.csv").open(encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        if (
            "doc.id" not in headers
            or not headers.intersection({"raw.title", "title"})
            or not headers.intersection({"raw.abstract", "abstract"})
        ):
            raise ValueError("raw-data.csv is missing required ID or text columns")

        for record_no, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record {record_no}")
            try:
                item_id = int(row["doc.id"]) - 1
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid doc.id in CSV record {record_no}") from exc

            if item_id < 0:
                raise ValueError(f"doc.id must be positive in CSV record {record_no}")
            if item_id in papers_by_id:
                raise ValueError(f"Duplicate doc.id: {item_id + 1}")

            title, title_fallback = _decode_text(row.get("raw.title"))
            if not title:
                title, title_fallback = _decode_text(row.get("title"))
            abstract, abstract_fallback = _decode_text(row.get("raw.abstract"))
            if not abstract:
                abstract, abstract_fallback = _decode_text(row.get("abstract"))

            if title_fallback or abstract_fallback:
                fallback_ids.add(item_id)
            papers_by_id[item_id] = Paper(item_id, title, abstract)

    if not papers_by_id:
        raise ValueError("raw-data.csv contains no papers")
    n_items = len(papers_by_id)
    if set(papers_by_id) != set(range(n_items)):
        raise ValueError("doc.id values must be contiguous from 1 to the paper count")
    papers = [papers_by_id[i] for i in range(n_items)]

    # This repository uses: count item_id_1 item_id_2 ... (zero-based IDs).
    # Every physical line is a user. A zero-interaction user is written as '0'.
    rows: list[int] = []
    cols: list[int] = []
    n_users = 0
    with (directory / "users.dat").open(encoding="utf-8") as f:
        for user_id, line in enumerate(f):
            n_users += 1
            try:
                values = [int(token) for token in line.split()]
            except ValueError as exc:
                raise ValueError(f"Non-integer value on users.dat line {n_users}") from exc
            if not values:
                raise ValueError(f"Blank row on users.dat line {n_users}; use '0'")

            count, *items = values
            if count < 0 or count != len(items):
                raise ValueError(f"Invalid item count on users.dat line {n_users}")
            if any(item < 0 or item >= n_items for item in items):
                raise ValueError(f"Item ID out of range on users.dat line {n_users}")
            rows.extend([user_id] * count)
            cols.extend(items)

    if n_users == 0:
        raise ValueError("users.dat contains no users")

    mat = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    mat.data.fill(1.0)

    if fallback_ids:
        warnings.warn(
            f"{len(fallback_ids)} papers have text fields that could not be decoded "
            "as UTF-8. These fields were preserved using Latin-1; inspect "
            "dataset.text_decode_fallback_ids before using the text.",
            UnicodeWarning,
            stacklevel=2,
        )

    # Optional so that interaction-only datasets keep loading; graph builders
    # raise a clear error when citations are required but absent.
    citations, self_loops = None, 0
    if (directory / "citations.dat").exists():
        citations, self_loops = load_citations(directory / "citations.dat", n_items)

    return CiteULikeDataset(
        papers, mat, tuple(sorted(fallback_ids)), len(rows) - mat.nnz, citations, self_loops
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Path to the cloned dataset folder")
    args = parser.parse_args()
    dataset = load_citeulike_a(args.directory)
    print(json.dumps(dataset.statistics(), indent=2))
