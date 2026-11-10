"""Binary relevance ranking metrics with explicit denominators."""
import numpy as np

def _validate(ranked_ids,relevant_ids,k):
    ranked=list(ranked_ids); relevant=set(relevant_ids)
    if type(k) is not int or k<=0: raise ValueError('k must be a positive integer')
    if len(ranked)!=len(set(ranked)): raise ValueError('duplicate ranked IDs')
    if not relevant: raise ValueError('empty relevance set must be excluded by evaluator')
    return ranked,relevant

def recall_at_k(ranked_ids,relevant_ids,k):
    ranked,relevant=_validate(ranked_ids,relevant_ids,k)
    return len(set(ranked[:k])&relevant)/len(relevant)

def ndcg_at_k(ranked_ids,relevant_ids,k):
    ranked,relevant=_validate(ranked_ids,relevant_ids,k)
    dcg=sum(1/np.log2(rank+2) for rank,item in enumerate(ranked[:k]) if item in relevant)
    ideal=sum(1/np.log2(rank+2) for rank in range(min(k,len(relevant))))
    return float(dcg/ideal)
