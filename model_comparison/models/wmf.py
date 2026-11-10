"""Implicit-feedback weighted ALS adapter with explicit confidence semantics."""
import time
import numpy as np
from .base import BaseModel, FitSummary, ModelError

class WMF(BaseModel):
    def fit(self,data):
        from implicit.cpu.als import AlternatingLeastSquares
        started=time.perf_counter(); self.bind(data)
        self.confidence=data.train.copy().astype(np.float32)
        self.confidence.data=1+self.config.get('confidence_scale',40)*self.confidence.data
        self.backend=AlternatingLeastSquares(factors=self.config.get('factors',64),regularization=self.config.get('regularization',.1),iterations=self.config.get('iterations',20),alpha=1.0,use_cg=self.config.get('use_cg',False),random_state=self.seed,num_threads=1)
        self.backend.fit(self.confidence,show_progress=False)
        self.user_factors=self.backend.user_factors; self.item_factors=self.backend.item_factors
        if self.user_factors.shape[0]!=data.num_users or self.item_factors.shape[0]!=data.num_items: raise ModelError('ALS factor orientation mismatch')
        if not np.isfinite(self.user_factors).all() or not np.isfinite(self.item_factors).all(): raise ModelError('non-finite ALS factors')
        self.fitted=True
        return FitSummary(self.config.get('iterations',20),time.perf_counter()-started,{'observed_confidence':float(1+self.config.get('confidence_scale',40)),'backend_alpha':1})

    def score(self,user_ids,item_ids):
        u,i=self.validate_score(user_ids,item_ids)
        return self.user_factors[u]@self.item_factors[i].T
