"""Sparse, candidate-oriented cosine neighbors."""
import time
import numpy as np
from scipy.sparse import csr_matrix, diags, save_npz
from .base import BaseModel, FitSummary

class ItemKNN(BaseModel):
    def fit(self,data):
        started=time.perf_counter(); self.bind(data)
        r=data.train.astype(np.float64)
        norms=np.sqrt(np.asarray(r.power(2).sum(axis=0)).ravel())
        inv=np.divide(1,norms,out=np.zeros_like(norms),where=norms>0)
        x=(r@diags(inv)).tocsr(); k=self.config.get('neighbors',100); block=self.config.get('similarity_block_size',256)
        indices=[]; values=[]; indptr=[0]
        for start in range(0,data.num_items,block):
            sims=(x[:,start:start+block].T@x).tocsr()
            for row in range(sims.shape[0]):
                lo,hi=sims.indptr[row:row+2]; ids=sims.indices[lo:hi]; scores=sims.data[lo:hi]
                good=(ids!=start+row)&(scores>0); ids=ids[good]; scores=scores[good]
                order=np.lexsort((ids,-scores))[:k]
                chosen=np.argsort(ids[order]); picked=order[chosen]
                indices.extend(ids[picked]); values.extend(scores[picked]); indptr.append(len(indices))
        self.similarity=csr_matrix((np.asarray(values,dtype=np.float32),indices,indptr),shape=(data.num_items,data.num_items))
        self.fitted=True
        return FitSummary(1,time.perf_counter()-started,{'similarity_nonzeros':self.similarity.nnz,'block_size':block})

    def score(self,user_ids,item_ids):
        u,i=self.validate_score(user_ids,item_ids)
        return (self.data.train[u]@self.similarity[i].T).toarray()

    def save_checkpoint(self,directory):
        from pathlib import Path
        p=Path(directory); p.mkdir(parents=True,exist_ok=True)
        save_npz(p/'similarity.npz',self.similarity); self.checkpoint_metadata(p)
