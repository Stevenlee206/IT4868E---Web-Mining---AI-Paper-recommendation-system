"""Stream candidate scores and retain only top-N rankings and per-user metrics."""
from dataclasses import dataclass
import numpy as np
from .metrics import recall_at_k, ndcg_at_k
from .ranking import rank_users

@dataclass
class UserMetrics:
    user_id: int
    recall_20: float
    recall_50: float
    ndcg_20: float
    ndcg_50: float

@dataclass
class EvaluationResult:
    metrics: dict
    per_user: list[dict]
    rankings: list
    evaluated_users: int


def evaluate(model,evaluation_data,config):
    d=evaluation_data; users=d.eligible_user_ids
    if len(users)==0: raise ValueError('evaluation has no eligible users')
    expected=np.flatnonzero((d.train.getnnz(axis=1)>0)&(d.test.getnnz(axis=1)>0))
    if not np.array_equal(users,expected): raise ValueError('evaluation population mismatch')
    if config.get('top_n',50)<50: raise ValueError('top_n must cover both requested cutoffs')
    per_user=[]; rankings=[]
    for ranking in rank_users(model,d.train,users,config.get('top_n',50),config.get('user_batch_size',32),config.get('pair_batch_size',8192)):
        user=ranking.user_id; relevant=d.test.indices[d.test.indptr[user]:d.test.indptr[user+1]]
        row={'user_id':user}
        for k in (20,50):
            row[f'recall_{k}']=recall_at_k(ranking.item_ids,relevant,k)
            row[f'ndcg_{k}']=ndcg_at_k(ranking.item_ids,relevant,k)
        per_user.append(row); rankings.append(ranking)
    if [x['user_id'] for x in per_user]!=users.tolist(): raise ValueError('missing/duplicate evaluated user predictions')
    metrics={key:float(np.mean([row[key] for row in per_user])) for key in ('recall_20','recall_50','ndcg_20','ndcg_50')}
    if not all(np.isfinite(v) and 0<=v<=1+1e-12 for v in metrics.values()): raise ValueError('invalid aggregate metric')
    return EvaluationResult(metrics,per_user,rankings,len(users))
