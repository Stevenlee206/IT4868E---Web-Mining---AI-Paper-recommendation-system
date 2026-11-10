"""One candidate protocol and tie policy for every model."""
import numpy as np
from models.base import UserRanking, ModelError

def rank_users(model,train,user_ids,top_n=50,user_batch_size=32,pair_batch_size=8192):
    if top_n<=0 or user_batch_size<=0: raise ValueError('rank sizes must be positive')
    items=np.arange(train.shape[1],dtype=np.int64)
    if hasattr(model,'pair_batch_size'): model.pair_batch_size=pair_batch_size
    for start in range(0,len(user_ids),user_batch_size):
        users=np.asarray(user_ids[start:start+user_batch_size],dtype=np.int64)
        scores=np.asarray(model.score(users,items))
        if scores.shape!=(len(users),len(items)): raise ModelError('incomplete rectangular candidate scores')
        for row,user in enumerate(users):
            eligible=np.ones(len(items),dtype=bool)
            eligible[train.indices[train.indptr[user]:train.indptr[user+1]]]=False
            candidates=items[eligible]; values=scores[row,eligible]
            if not np.isfinite(values).all(): raise ModelError('non-finite candidate scores')
            order=np.lexsort((candidates,-values))[:top_n]
            yield UserRanking(int(user),candidates[order],values[order])
