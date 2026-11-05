# Data and evaluation foundation

Date: 5 November 2026
Tóm tắt: đã chia dataset theo warm/cold-start là lưu trong artifacts/. Dùng luôn.

Viết model mới trong models/, kế thừa base.py

## Dataset verification

Source: the existing local `data/raw/citeulike-a` directory. Raw files were not changed.

| Statistic | Observed value |
| --- | ---: |
| Users | 5,551 |
| Papers | 16,980 |
| Unique saved user-paper pairs | 204,986 |
| Sparsity | 99.7825% |
| Duplicate interaction entries removed | 0 |
| Missing titles | 0 |
| Missing abstracts | 0 |
| Missing combined text | 0 |
| Papers needing Latin-1 fallback in a selected text field | 13 |

CSV document IDs are converted from one-based to zero-based indices. Each
physical `users.dat` line identifies a user, and its paper IDs are already
zero-based. Paper order always matches interaction columns. Duplicate saves
collapse to one binary positive; duplicate document IDs are rejected because
they make metadata alignment ambiguous. Explicit zero-history rows remain.

Interactions use a binary float32 CSR matrix. Text is recovered as UTF-8 where
possible; undecodable fields retain their original bytes through Latin-1 and
produce a warning. This preserves data but does not repair source corruption.
Affected indices are available through `dataset.text_decode_fallback_ids`. Lỗi 
format này do lẫn lộn tiếng Anh-Đức nhưng có thể bỏ qua, không quan trọng lắm.

## Splitting protocol

Defaults remain five outer folds, seed 42, and validation fraction 0.1.
For each outer fold, its test population is held out first. Validation is
sampled from the remaining four folds. The target proportions are approximately
70/10/20 for train/validation/test, with the original rounding rules retained.

- **Warm start:** positive and zero user-paper pairs are stratified separately
  for each eligible paper. Papers with fewer than five saves stay entirely in
  training. Validation sampling preserves at least one training positive per
  eligible paper. Each eligible pair appears in the outer test pools exactly
  once. Validation and test candidates vary by user.
- **Cold start:** all paper IDs are partitioned into
  five outer test groups. Validation papers are sampled from the remaining
  papers. Entire columns are withheld from training. Text remains available.
  Every paper is a test paper exactly once.
- **Both:** positive matrices are disjoint and reconstruct the original
  interactions. Candidate pools include unobserved pairs as well as positives.
  Users and paper indices are retained. Empty histories are never repaired
  using validation/test labels.

The README describes warm splitting at a high level as splitting libraries.
The existing implementation is more specific: it assigns pairs per paper and
ranks only assigned candidate pools. That existing behavior was preserved as
requested. It must be disclosed when comparing with full-catalog benchmarks.

Before editing, all ten fold fingerprints were captured on a fixed synthetic
dataset containing eligible, rare, and unsaved papers. The resulting fingerprints
are exactly unchanged after the edits, including positive matrices, candidate
masks, item pools, and split metadata.

Full local dataset splits were generated under `artifacts/splits/warm` and
`artifacts/splits/cold`. Fold numbers below are zero-based; counts are positive
interactions, not candidate-pair counts.

| Scenario | Fold | Train | Validation | Test | Validation users | Test users |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm | 0 | 145,011 | 19,974 | 40,001 | 4,769 | 5,389 |
| Warm | 1 | 144,831 | 20,062 | 40,093 | 4,749 | 5,363 |
| Warm | 2 | 144,869 | 20,064 | 40,053 | 4,823 | 5,341 |
| Warm | 3 | 144,926 | 20,015 | 40,045 | 4,790 | 5,381 |
| Warm | 4 | 144,871 | 20,056 | 40,059 | 4,791 | 5,379 |
| Cold | 0 | 143,301 | 20,965 | 40,720 | 4,870 | 5,407 |
| Cold | 1 | 142,674 | 20,686 | 41,626 | 4,838 | 5,414 |
| Cold | 2 | 143,495 | 19,912 | 41,579 | 4,845 | 5,408 |
| Cold | 3 | 144,533 | 20,485 | 39,968 | 4,814 | 5,390 |
| Cold | 4 | 142,858 | 21,035 | 41,093 | 4,906 | 5,404 |

Cold folds contain 11,886 training, 1,698 validation, and 3,396 test papers each.
Their interaction counts differ because paper popularity differs.

## Model interface 

Subclass `paperrec.models.Recommender` and implement two methods:

1. `_fit(self)` learns from `self.train`, a copy of the training interactions.
   Paper text is available through `self.papers` and permitted training paper
   IDs through `self.train_items`. Use `self.seed` for model randomness.
2. `score(self, profile, user_index=None)` returns a one-dimensional finite
   score array with one value per catalog paper. Higher scores rank first.

Call `model.fit(dataset.papers, split)`; it prepares the inputs, invokes `_fit`,
and returns the model. `recommend` handles candidates, observed-paper exclusions,
optional additional exclusions, top-K truncation, and seeded score ties.

Warm collaborative losses and negative sampling **must use**
`self.training_mask(user_batch)`. Zeros in `self.train` alone cannot distinguish
training negatives from withheld pairs. Cold collaborative parameters must be
trained only for `self.train_items`. Under strict inductive text evaluation,
fit vocabulary, IDF, dimensionality reduction, and other learned preprocessing
on those training papers, then transform the rest of the catalog. A future
model must follow these rules; the abstract interface cannot enforce its
internal learning algorithm.

The former embedding and vectorizer helpers were removed: no current model
used them, and fitting a vectorizer on every paper would silently introduce
transductive preprocessing in cold-start experiments.

## Evaluation definitions

Default cutoffs are 5, 10, and 20. Only users with training history and at least
one target in the evaluated partition contribute. Each user has equal weight.
Skipped users are reported as disjoint groups: no training profile, or a
profile but no targets. An entirely unevaluable fold raises an error.

For each user, rank every item returned by
`split.candidate_items(user, partition)`. No negative sampling or alternative
catalog scope is used. Validation labels never become profile input for test
evaluation. Full rankings are needed for untruncated MRR.

| Metric | Definition |
| --- | --- |
| Recall@K | Hits in the first K positions divided by the number of relevant papers. |
| Precision@K | Hits divided by K, even if the candidate list is shorter than K. |
| NDCG@K | Binary gains discounted by `1/log2(rank + 1)`, divided by ideal DCG for `min(K, relevant_count)` positives. |
| MAP@K | Mean user AP@K; each AP sums precision at relevant ranks within K and divides by `min(K, relevant_count)`. |
| F1@K | Harmonic mean of each user's Precision@K and Recall@K, then averaged over users. |
| MRR | Mean reciprocal rank of the first relevant result over the complete candidate ranking. |
| Catalog coverage@K | Distinct papers recommended in the first K positions across evaluated users divided by all catalog papers. |

Recall and NDCG remain the main proposal metrics. Precision is included to
make F1 interpretable. Topic diversity is deferred to the proposal's session
extension: the current data model has no topic annotations, so reporting a
topic score now would be unsupported.

The experiment runner creates a fresh model per fold. It reports equally
weighted means and sample standard deviations across folds, retaining the
individual fold results and user counts. Validation selects settings; only
after fixing those settings should test evaluation be run.

Efficiency measurements include model construction plus fitting time, full
recommendation time per evaluated user (mean, p95, maximum), a p95-under-three-
seconds indicator, and sampled process RSS peak/increase in MiB. Candidate
selection and metric calculation are outside the recommendation timer. RSS
includes the dataset, splits, and other process allocations; 50 ms sampling can
miss short peaks and does not measure GPU memory. These are diagnostics, not
isolated model-memory measurements or demonstrated latency for future models.

## Running and extending the pipeline

From the repository root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m paperrec.data data/raw/citeulike-a
.\.venv\Scripts\python.exe -m paperrec.splits data/raw/citeulike-a --output artifacts/splits --scenario both
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check paperrec tests
```

The split output already exists from verification. Load it instead of rerunning
the creation command, or choose a new directory. Saved data/text fingerprints
and split checksums are verified on load; nonempty output directories are never
silently overwritten.

```python
from paperrec.data import load_citeulike_a
from paperrec.splits import load_folds

dataset = load_citeulike_a("data/raw/citeulike-a")
folds = load_folds("artifacts/splits/warm", dataset)
print(folds[0].statistics())
```

After implementing `YourRecommender`, pass factories that construct fresh models:

```python
from paperrec.experiment import run_experiment

report = run_experiment(
    dataset,
    folds,
    models={"your-model-fixed-settings": lambda: YourRecommender(seed=42)},
    output="results/your-model-warm-validation",
    partition="validation",
)
```

Use a separate output directory and `partition="test"` for the final evaluation.
Repeat with the saved cold folds. Results are written to `results.json` with
dataset/split fingerprints, model class/seed, fold statistics, metrics, timing,
memory, and per-user recommendations. Keep the factory code and its settings
with each experiment; arbitrary future model hyperparameters are not
automatically serialized. arXiv remains outside this offline pipeline.

