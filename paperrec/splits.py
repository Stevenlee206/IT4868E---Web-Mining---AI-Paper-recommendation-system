"""Reproducible CiteULike-a cross-validation (Python 3.10+, NumPy, SciPy).

Protocol: Wang & Blei (KDD 2011), Section 4, with an INNER validation holdout.
https://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf

* Warm: stratify each article's positive AND zero user-item pairs over five
  outer folds. Articles with <5 saves stay in training in every fold.
* Cold: split ALL article IDs into five outer folds; hold out whole columns.
* In outer fold f, test=f. Draw validation only from the other four folds.
  validation_fraction=0.1 means 10% of the original eligible population,
  i.e. 12.5% of the outer-training population. This aims for 70/10/20,
  NOT 60/20/20. Rounding, rare items and item popularity affect actual counts.
* The paper describes 80/20 train/test CV. Our added validation stage is an
  extension, not a claim of exact reproduction of its published scores.

IMPORTANT: train/validation/test contain ONLY positive saves. In warm mode,
use split.pair_mask('train', users) to mask a pairwise/dense training loss or
restrict negative sampling. Merely treating every zero in split.train as a
training negative does NOT implement this masked-pair protocol. In cold mode,
fit collaborative parameters only on split.train[:, split.train_items].
Validation labels are for selection/early stopping; test labels only for final
evaluation. No automatic refit on train+validation is performed.

Rank only split.candidate_items(user, 'validation'/'test'). These pools include
zero/unobserved pairs, NOT just the held-out positives. All users and item IDs
are retained. Evaluate users returned by split.evaluation_users(phase); report
skips via split.statistics(). Empty-history users are not repaired using labels.

For strict inductive cold-start text features, fit learned preprocessing on
train_items and transform validation/test content. Precomputed corpus-wide
features imply a different, transductive content assumption; report that choice.

Example:
    from paperrec.data import load_citeulike_a
    from paperrec.splits import make_folds, save_folds, load_folds
    dataset = load_citeulike_a('citeulike-a')
    for scenario in ('warm', 'cold'):
        folds = make_folds(dataset.interactions, scenario=scenario, seed=42)
        save_folds(folds, f'saved_splits/{scenario}', dataset)
    folds = load_folds('saved_splits/cold', dataset)

CLI (from the project root):
    python -m paperrec.splits data/raw/citeulike-a --output artifacts/splits

Warm masks are bit-packed: about 24 MB per split at 5,551 x 16,980.
Construction also uses a temporary one-byte-per-pair array (~94 MB).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

FORMAT_VERSION = 1
PHASES = ("train", "validation", "test")


def _binary_csr(value) -> sparse.csr_matrix:
    """Canonicalize storage, but reject non-binary or duplicate positive input."""
    if not sparse.issparse(value) or value.ndim != 2:
        raise TypeError("Expected a two-dimensional SciPy sparse matrix")
    values = value.tocoo(copy=False).data
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("Interactions must be finite real binary values")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("Interactions must be binary (0 or 1)")
    result = sparse.csr_matrix(value, dtype=np.float32, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if np.any(result.data != 1):
        raise ValueError("Duplicate positive coordinates: deduplicate in the loader")
    return result


def _source_matrix(dataset):
    return _binary_csr(getattr(dataset, "interactions", dataset))


def _hash_array(digest, value, dtype):
    array = np.asarray(value, dtype=dtype, order="C")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _hash_matrix(digest, matrix):
    matrix = _binary_csr(matrix)
    _hash_array(digest, matrix.shape, "<i8")
    _hash_array(digest, matrix.indptr, "<i8")
    _hash_array(digest, matrix.indices, "<i8")
    _hash_array(digest, matrix.data, "<f4")


def dataset_fingerprint(dataset) -> str:
    """Hash dataset content and ID order for saved-split validation.

    When given the dataset object, hashes ordered paper IDs and text as well as
    the matrix. When given just a matrix, checks only that matrix and its order.
    Use the same input kind when saving and loading.
    """
    digest = hashlib.sha256(b"citeulike-dataset-v1")
    _hash_matrix(digest, _source_matrix(dataset))
    papers = getattr(dataset, "papers", None)
    digest.update(b"with-papers" if papers is not None else b"matrix-only")
    if papers is not None:
        for paper in papers:
            payload = json.dumps(
                [int(paper.paper_id), paper.title, paper.abstract],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def _packed_contains(packed, rows, cols):
    return ((packed[rows, cols // 8] >> (cols % 8)) & 1).astype(bool)


def _select_positive_pairs(original, rows, keep):
    return sparse.csr_matrix(
        (np.ones(np.count_nonzero(keep), dtype=np.float32), (rows[keep], original.indices[keep])),
        shape=original.shape,
    )


@dataclass
class Split:
    train: sparse.csr_matrix
    validation: sparse.csr_matrix
    test: sparse.csr_matrix
    train_items: np.ndarray
    validation_items: np.ndarray  # cold: exact pool; warm: eligible-item universe
    test_items: np.ndarray  # warm: use candidate_items(user), not this alone
    scenario: str
    seed: int
    fold: int  # zero-based outer TEST fold
    n_folds: int
    validation_fraction: float  # relative to ORIGINAL eligible population
    validation_candidates_packed: np.ndarray | None = None
    test_candidates_packed: np.ndarray | None = None

    def pair_mask(self, phase: str, users=None) -> np.ndarray:
        """Boolean [selected_users, all_items] mask; request user batches.

        True means this PAIR belongs to the selected partition, irrespective
        of whether its observed label is 0 or 1. Cold masks select columns.
        """
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        n_users, n_items = self.train.shape
        if users is None:
            users = np.arange(n_users)
        else:
            users = np.atleast_1d(users)
            if users.ndim != 1 or not np.issubdtype(users.dtype, np.integer):
                raise ValueError("users must be a one-dimensional integer index array")
            if np.any((users < 0) | (users >= n_users)):
                raise ValueError("User ID out of range")
        if self.scenario == "cold":
            mask = np.zeros(n_items, dtype=bool)
            mask[getattr(self, f"{phase}_items")] = True
            return np.broadcast_to(mask, (len(users), n_items)).copy()
        if phase == "train":
            packed = self.validation_candidates_packed[users] | self.test_candidates_packed[users]
            return ~np.unpackbits(packed, axis=1, count=n_items, bitorder="little").astype(bool)
        packed = getattr(self, f"{phase}_candidates_packed")[users]
        return np.unpackbits(packed, axis=1, count=n_items, bitorder="little").astype(bool)

    def candidate_items(self, user: int, phase: str = "test") -> np.ndarray:
        """Full ranking candidates (positive and zero pairs), in global ID order."""
        if phase not in ("validation", "test"):
            raise ValueError("Ranking phase must be validation or test")
        return np.flatnonzero(self.pair_mask(phase, [user])[0])

    def evaluation_users(self, phase: str = "test") -> np.ndarray:
        """Users with >=1 training save and >=1 positive target for this phase."""
        if phase not in ("validation", "test"):
            raise ValueError("Evaluation phase must be validation or test")
        has_history = np.diff(self.train.indptr) > 0
        has_target = np.diff(getattr(self, phase).indptr) > 0
        return np.flatnonzero(has_history & has_target)

    def statistics(self) -> dict:
        result = {"scenario": self.scenario, "fold": self.fold, "shape": list(self.train.shape)}
        has_history = np.diff(self.train.indptr) > 0
        for phase in PHASES:
            matrix = getattr(self, phase)
            result[f"{phase}_interactions"] = int(matrix.nnz)
            if phase != "train":
                has_target = np.diff(matrix.indptr) > 0
                result[f"{phase}_evaluated_users"] = int(np.count_nonzero(has_history & has_target))
                result[f"{phase}_users_without_targets"] = int(np.count_nonzero(~has_target))
                result[f"{phase}_target_users_without_history"] = int(
                    np.count_nonzero(has_target & ~has_history)
                )
        return result

    def validate(self, original):
        original = _source_matrix(original)
        if self.scenario not in ("warm", "cold"):
            raise ValueError("scenario must be warm or cold")
        _check_parameters(self.seed, self.n_folds, self.validation_fraction)
        if not isinstance(self.fold, (int, np.integer)) or not 0 <= self.fold < self.n_folds:
            raise ValueError("Invalid fold index")
        matrices = [getattr(self, phase) for phase in PHASES]
        for matrix in matrices:
            if not sparse.isspmatrix_csr(matrix) or matrix.shape != original.shape:
                raise ValueError("Split matrices must be CSR with original shape")
            if not matrix.has_canonical_format or np.any(matrix.data != 1):
                raise ValueError("Split matrices must be canonical binary CSR without stored zeros")
        for a, b in ((0, 1), (0, 2), (1, 2)):
            if matrices[a].multiply(matrices[b]).nnz:
                raise ValueError("Positive interaction leakage between partitions")
        if (self.train + self.validation + self.test != original).nnz:
            raise ValueError("Partitions must reconstruct every original save")
        n_users, n_items = original.shape
        for phase in PHASES:
            items = getattr(self, f"{phase}_items")
            if (
                not isinstance(items, np.ndarray)
                or items.ndim != 1
                or not np.issubdtype(items.dtype, np.integer)
                or np.any((items < 0) | (items >= n_items))
                or not np.array_equal(items, np.unique(items))
            ):
                raise ValueError(f"{phase}_items must be sorted unique valid IDs")

        if self.scenario == "cold":
            if (
                self.validation_candidates_packed is not None
                or self.test_candidates_packed is not None
            ):
                raise ValueError("Cold candidate pools must use item IDs, not pair masks")
            combined = np.concatenate([getattr(self, f"{p}_items") for p in PHASES])
            if not np.array_equal(np.sort(combined), np.arange(n_items)):
                raise ValueError("Cold item pools must partition ALL item IDs")
            for phase, matrix in zip(PHASES, matrices):
                if not np.all(np.isin(matrix.indices, getattr(self, f"{phase}_items"))):
                    raise ValueError(f"Cold {phase} contains an item from another pool")
        else:
            eligible = np.flatnonzero(
                np.bincount(original.indices, minlength=n_items) >= self.n_folds
            )
            if (
                not np.array_equal(self.train_items, np.arange(n_items))
                or not np.array_equal(self.validation_items, eligible)
                or not np.array_equal(self.test_items, eligible)
            ):
                raise ValueError("Invalid warm item universe")
            expected_shape = (n_users, (n_items + 7) // 8)
            for packed in (self.validation_candidates_packed, self.test_candidates_packed):
                if (
                    not isinstance(packed, np.ndarray)
                    or packed.shape != expected_shape
                    or packed.dtype != np.uint8
                ):
                    raise ValueError("Invalid packed warm candidate mask")
                if n_items % 8 and np.any(packed[:, -1] >> (n_items % 8)):
                    raise ValueError("Nonzero padding bits in candidate mask")
            vp, tp = self.validation_candidates_packed, self.test_candidates_packed
            if np.any(vp & tp):
                raise ValueError("Validation and test candidate pairs overlap")
            eligible_mask = np.zeros(n_items, dtype=np.uint8)
            eligible_mask[eligible] = 1
            allowed = np.packbits(eligible_mask, bitorder="little")
            if np.any((vp | tp) & ~allowed):
                raise ValueError("Rare-item candidate pairs must stay in training")
            for phase, matrix in zip(PHASES, matrices):
                rows = np.repeat(np.arange(n_users), np.diff(matrix.indptr))
                in_val = _packed_contains(vp, rows, matrix.indices)
                in_test = _packed_contains(tp, rows, matrix.indices)
                correct = (
                    ~(in_val | in_test)
                    if phase == "train"
                    else in_val
                    if phase == "validation"
                    else in_test
                )
                if not np.all(correct):
                    raise ValueError(f"{phase} positives disagree with pair assignment")
            counts = np.bincount(self.train.indices, minlength=n_items)
            if np.any(counts[eligible] == 0):
                raise ValueError("Warm held-out items must have positive training history")

    def _metadata(self):
        return {
            "format_version": FORMAT_VERSION,
            "scenario": self.scenario,
            "seed": int(self.seed),
            "fold": int(self.fold),
            "n_folds": int(self.n_folds),
            "validation_fraction": float(self.validation_fraction),
            **{f"{p}_items": getattr(self, f"{p}_items").tolist() for p in PHASES},
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256(json.dumps(self._metadata(), sort_keys=True).encode())
        for phase in PHASES:
            _hash_matrix(digest, getattr(self, phase))
        for packed in (self.validation_candidates_packed, self.test_candidates_packed):
            digest.update(b"none" if packed is None else b"packed")
            if packed is not None:
                _hash_array(digest, packed, "u1")
        return digest.hexdigest()

    def save(self, directory, dataset_fingerprint: str):
        """Save to a new/empty directory; never silently overwrite old splits."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        for phase in PHASES:
            sparse.save_npz(directory / f"{phase}.npz", getattr(self, phase))
        if self.scenario == "warm":
            np.savez_compressed(
                directory / "candidates.npz",
                validation=self.validation_candidates_packed,
                test=self.test_candidates_packed,
            )
        metadata = self._metadata()
        metadata.update(
            dataset_fingerprint=dataset_fingerprint, split_fingerprint=self.fingerprint()
        )
        (directory / "split.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _check_parameters(seed, n_folds, validation_fraction):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(n_folds, (bool, np.bool_))
        or not isinstance(n_folds, (int, np.integer))
        or not 3 <= n_folds <= 127
    ):
        raise ValueError("n_folds must be an integer from 3 to 127")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1 - 1 / n_folds:
        raise ValueError("validation_fraction must be >0 and leave room for training")


def _validation_sample(users, fraction, rng, keep_one=False):
    """Stochastic rounding avoids forcing one validation save per rare item."""
    expected = len(users) * fraction
    count = int(np.floor(expected))
    count += int(rng.random() < expected - count)
    count = min(count, max(0, len(users) - int(keep_one)))
    return rng.permutation(users)[:count]


def make_folds(
    interactions, scenario="warm", seed=42, n_folds=5, validation_fraction=0.1
) -> list[Split]:
    """Build genuine outer CV: each eligible save is tested exactly once.

    Validation is freshly sampled WITHIN each outer training partition, not
    taken from that fold's test set. Validation sets may overlap across runs;
    this is normal in outer CV. Outer test sets never overlap. Fractions refer
    to eligible pairs for warm and item counts for cold, not cold save counts.
    Warm articles with fewer than n_folds saves never enter validation/test.
    """
    if scenario not in ("warm", "cold"):
        raise ValueError("scenario must be warm or cold")
    _check_parameters(seed, n_folds, validation_fraction)
    seed, n_folds = int(seed), int(n_folds)
    original = _source_matrix(interactions)
    n_users, n_items = original.shape
    if not n_users or not n_items or not original.nnz:
        raise ValueError("Cannot split an empty interaction dataset")
    rows = np.repeat(np.arange(n_users), np.diff(original.indptr))
    cols = original.indices
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    all_items = np.arange(n_items, dtype=np.int64)
    folds = []

    if scenario == "warm":
        csc = original.tocsc()
        eligible = np.flatnonzero(np.diff(csc.indptr) >= n_folds)
        if not eligible.size:
            raise ValueError(f"Warm CV needs at least one article with >= {n_folds} saves")
        # Temporary [item, user] labels. -1 means training-only in every fold.
        pair_fold = np.full((n_items, n_users), -1, dtype=np.int8)
        for item in eligible:
            positives = csc.indices[csc.indptr[item] : csc.indptr[item + 1]]
            is_zero = np.ones(n_users, dtype=bool)
            is_zero[positives] = False
            # Stratify the 1s and 0s separately so every fold has positive
            # coverage and similar candidate populations. Random offsets
            # prevent remainder users from always going into the first fold.
            for users in (positives, np.flatnonzero(is_zero)):
                shuffled = rng.permutation(users)
                offset = int(rng.integers(n_folds))
                pair_fold[item, shuffled] = (np.arange(len(users)) + offset) % n_folds
        positive_outer_fold = pair_fold[cols, rows]
        inner_fraction = validation_fraction / (1 - 1 / n_folds)
    else:
        if n_items < n_folds:
            raise ValueError("Cold CV needs at least n_folds articles")
        item_fold = np.empty(n_items, dtype=np.int16)
        for fold, items in enumerate(np.array_split(rng.permutation(all_items), n_folds)):
            item_fold[items] = fold

    for fold in range(n_folds):
        inner_rng = np.random.default_rng(np.random.SeedSequence([seed, 1, fold]))
        vp = tp = None
        if scenario == "warm":
            tp = np.packbits(pair_fold == fold, axis=0, bitorder="little").T.copy()
            vp = np.zeros_like(tp)
            for item in eligible:
                positives = csc.indices[csc.indptr[item] : csc.indptr[item + 1]]
                remaining_positive = positives[pair_fold[item, positives] != fold]
                zero_mask = pair_fold[item] != fold
                zero_mask[positives] = False
                for users, keep_one in (
                    (remaining_positive, True),
                    (np.flatnonzero(zero_mask), False),
                ):
                    chosen = _validation_sample(users, inner_fraction, inner_rng, keep_one)
                    vp[chosen, item // 8] |= np.uint8(1 << (item % 8))
            is_test = positive_outer_fold == fold
            is_val = _packed_contains(vp, rows, cols)
            train_items, val_items, test_items = all_items, eligible, eligible
        else:
            test_items = np.flatnonzero(item_fold == fold)
            remaining = np.flatnonzero(item_fold != fold)
            n_val = min(max(1, round(n_items * validation_fraction)), len(remaining) - 1)
            val_items = np.sort(inner_rng.permutation(remaining)[:n_val])
            train_items = np.setdiff1d(remaining, val_items)
            is_test = item_fold[cols] == fold
            is_val = np.isin(cols, val_items)
        result = Split(
            _select_positive_pairs(original, rows, ~(is_test | is_val)),
            _select_positive_pairs(original, rows, is_val),
            _select_positive_pairs(original, rows, is_test),
            train_items.copy(),
            val_items.copy(),
            test_items.copy(),
            scenario,
            seed,
            fold,
            n_folds,
            float(validation_fraction),
            vp,
            tp,
        )
        result.validate(original)
        folds.append(result)
    validate_folds(folds, original, validate_individual=False)
    return folds


def validate_folds(folds, original, *, validate_individual=True):
    """Also check cross-fold disjointness and complete eligible test coverage."""
    original = _source_matrix(original)
    if not folds:
        raise ValueError("No folds supplied")
    first = folds[0]
    if len(folds) != first.n_folds or [f.fold for f in folds] != list(range(first.n_folds)):
        raise ValueError("Supply all outer folds in order, without duplicates")
    for fold in folds:
        if (fold.scenario, fold.seed, fold.n_folds, fold.validation_fraction) != (
            first.scenario,
            first.seed,
            first.n_folds,
            first.validation_fraction,
        ):
            raise ValueError("Folds have inconsistent settings")
        if validate_individual:
            fold.validate(original)
    total = sparse.csr_matrix(original.shape, dtype=np.float32)
    for fold in folds:
        if total.multiply(fold.test).nnz:
            raise ValueError("A positive save is tested in more than one outer fold")
        total += fold.test
    n_items = original.shape[1]
    if first.scenario == "warm":
        eligible = np.bincount(original.indices, minlength=n_items) >= first.n_folds
        expected = original.multiply(eligible).tocsr()
        expected.eliminate_zeros()
        seen = np.zeros_like(first.test_candidates_packed)
        for fold in folds:
            if np.any(seen & fold.test_candidates_packed):
                raise ValueError("A candidate pair is tested in multiple outer folds")
            seen |= fold.test_candidates_packed
        if not np.all(seen == np.packbits(eligible, bitorder="little")):
            raise ValueError("Warm outer test pools must cover all eligible 1 and 0 pairs")
    else:
        expected = original
        test_items = np.concatenate([fold.test_items for fold in folds])
        if not np.array_equal(np.sort(test_items), np.arange(n_items)):
            raise ValueError("Every article must be a cold test article exactly once")
    if (total != expected).nnz:
        raise ValueError("Outer test folds do not cover every eligible positive save")


def _load_split(directory, original, expected_dataset_fingerprint):
    directory = Path(directory)
    metadata = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported split format; regenerate using this module")
    if metadata["dataset_fingerprint"] != expected_dataset_fingerprint:
        raise ValueError("Saved split belongs to a different dataset, text or ID order")
    vp = tp = None
    if metadata["scenario"] == "warm":
        with np.load(directory / "candidates.npz", allow_pickle=False) as data:
            vp, tp = data["validation"], data["test"]
    result = Split(
        *(sparse.load_npz(directory / f"{phase}.npz") for phase in PHASES),
        *(np.asarray(metadata[f"{phase}_items"], dtype=np.int64) for phase in PHASES),
        *(metadata[key] for key in ("scenario", "seed", "fold", "n_folds", "validation_fraction")),
        vp,
        tp,
    )
    result.validate(original)
    if result.fingerprint() != metadata["split_fingerprint"]:
        raise ValueError("Saved split checksum mismatch")
    return result


def load_split(directory, dataset) -> Split:
    return _load_split(directory, _source_matrix(dataset), dataset_fingerprint(dataset))


def save_folds(folds, directory, dataset):
    """Save one scenario's folds; dataset may be a dataset object or a matrix."""
    validate_folds(folds, dataset)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(dataset)
    for fold in folds:
        fold.save(directory / f"fold_{fold.fold}", fingerprint)
    manifest = {
        "format_version": FORMAT_VERSION,
        "n_folds": len(folds),
        "dataset_fingerprint": fingerprint,
        "split_fingerprints": [fold.fingerprint() for fold in folds],
    }
    (directory / "cv.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_folds(directory, dataset) -> list[Split]:
    directory = Path(directory)
    manifest = json.loads((directory / "cv.json").read_text(encoding="utf-8"))
    fingerprint = dataset_fingerprint(dataset)
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported CV format")
    if manifest["dataset_fingerprint"] != fingerprint:
        raise ValueError("Saved CV belongs to a different dataset, text or ID order")
    original = _source_matrix(dataset)
    folds = [
        _load_split(directory / f"fold_{i}", original, fingerprint)
        for i in range(manifest["n_folds"])
    ]
    validate_folds(folds, original, validate_individual=False)
    if [fold.fingerprint() for fold in folds] != manifest["split_fingerprints"]:
        raise ValueError("Cross-validation manifest checksum mismatch")
    return folds


if __name__ == "__main__":
    import argparse

    from .data import load_citeulike_a

    parser = argparse.ArgumentParser(
        description="Create CiteULike-a 5-fold train/validation/test splits"
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("saved_splits"))
    parser.add_argument("--scenario", choices=("warm", "cold", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    dataset = load_citeulike_a(args.dataset_directory)
    scenarios = ("warm", "cold") if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        folds = make_folds(
            dataset.interactions,
            scenario=scenario,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )
        save_folds(folds, args.output / scenario, dataset)
        print(json.dumps([fold.statistics() for fold in folds], indent=2))
        del folds
