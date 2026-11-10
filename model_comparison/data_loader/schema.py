"""Data types keep held-out labels outside the training boundary."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.sparse import csr_matrix

class DataError(ValueError):
    pass

@dataclass(frozen=True)
class Paper:
    paper_id: int
    doc_id: int
    citeulike_id: str | None
    title: str | None
    raw_title: str | None
    abstract: str | None

@dataclass
class InteractionParseResult:
    matrix: csr_matrix
    raw_pairs: int
    duplicates: int

@dataclass
class MetadataParseResult:
    papers: list[Paper]
    encoding: str
    warnings: list[str]

@dataclass(frozen=True)
class DatasetManifest:
    values: dict[str, Any]

@dataclass(frozen=True)
class TrainingData:
    train: csr_matrix
    bow: csr_matrix
    num_users: int
    num_items: int
    dataset_id: str

@dataclass(frozen=True)
class EvaluationData:
    train: csr_matrix
    test: csr_matrix
    eligible_user_ids: np.ndarray
    num_items: int
    split_id: str

@dataclass
class PreparedDataset:
    catalog: list[Paper]
    vocabulary: tuple[str, ...]
    train: csr_matrix
    test: csr_matrix
    bow: csr_matrix
    manifest: dict
    exclusions: list[dict]

    def training_data(self):
        return TrainingData(self.train, self.bow, *self.train.shape, self.manifest['dataset_id'])

    def evaluation_data(self):
        eligible = np.flatnonzero((self.train.getnnz(axis=1)>0) & (self.test.getnnz(axis=1)>0))
        return EvaluationData(self.train, self.test, eligible, self.train.shape[1], self.manifest['split_id'])
