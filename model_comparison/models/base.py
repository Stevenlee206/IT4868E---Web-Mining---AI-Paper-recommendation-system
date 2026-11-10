"""Common model interface and validation."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import json
import numpy as np
from data_loader.schema import TrainingData

class ModelError(ValueError):
    pass

@dataclass
class FitSummary:
    iterations: int
    elapsed_time: float
    diagnostics: dict = field(default_factory=dict)

@dataclass
class UserRanking:
    user_id: int
    item_ids: np.ndarray
    scores: np.ndarray

class Recommender(Protocol):
    def fit(self, data: TrainingData) -> FitSummary: ...
    def score(self, user_ids, item_ids) -> np.ndarray: ...

class BaseModel:
    def __init__(self, config=None, seed=42, **kwargs):
        self.config = config or {}
        self.seed = seed
        self.fitted = False

    def bind(self, data):
        if not isinstance(data, TrainingData):
            raise ModelError('fit requires TrainingData without test labels')
        self.data = data
        self.fitted = False

    def validate_score(self, user_ids, item_ids):
        if not self.fitted:
            raise ModelError('scoring before fit')
        raw_u, raw_i = np.asarray(user_ids), np.asarray(item_ids)
        if raw_u.ndim != 1 or raw_i.ndim != 1 or raw_u.dtype.kind not in 'iu' or raw_i.dtype.kind not in 'iu':
            raise ModelError('user and item IDs must be one-dimensional integer arrays')
        u, i = raw_u.astype(np.int64), raw_i.astype(np.int64)
        if np.any((u<0)|(u>=self.data.num_users)) or np.any((i<0)|(i>=self.data.num_items)):
            raise ModelError('unknown user or item ID')
        if np.any(self.data.train.getnnz(axis=1)[u] == 0):
            raise ModelError('unsupported training-empty user')
        return u, i

    def save_checkpoint(self, directory):
        p = Path(directory); p.mkdir(parents=True, exist_ok=True)
        payload = {key: getattr(self, key) for key in ('user_factors','item_factors','theta','beta') if hasattr(self,key)}
        np.savez_compressed(p/'factors.npz', **payload)
        self.checkpoint_metadata(p)

    def checkpoint_metadata(self, path):
        from pipeline.reproducibility import environment
        (path/'metadata.json').write_text(json.dumps({'schema_version':1,'model':type(self).__name__,'config':self.config,'seed':self.seed,'dataset_id':self.data.dataset_id,'shape':self.data.train.shape,'environment':environment()}, indent=2), encoding='utf-8')
