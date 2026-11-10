import numpy as np
import pytest
from scipy.sparse import csr_matrix
from data_loader.schema import EvaluationData
from evaluation.evaluator import evaluate

class Fixed:
    def score(self,users,items): return np.tile(np.array([0,3,2,1])[items],(len(users),1))

def test_population_and_macro():
    tr=csr_matrix([[1,0,0,0],[1,0,0,0],[0,0,0,0]])
    te=csr_matrix([[0,1,0,0],[0,0,1,1],[0,0,0,0]])
    data=EvaluationData(tr,te,np.array([0,1]),4,'fixed')
    out=evaluate(Fixed(),data,{'cutoffs':[20,50],'top_n':50,'user_batch_size':1,'pair_batch_size':10})
    assert out.evaluated_users==2 and out.metrics['recall_20']==1
    assert out.metrics['ndcg_20']<1
    assert [x.item_ids.tolist() for x in out.rankings]==[[1,2,3],[1,2,3]]

def test_empty_population():
    data=EvaluationData(csr_matrix((1,2)),csr_matrix((1,2)),np.array([],dtype=int),2,'empty')
    with pytest.raises(ValueError): evaluate(Fixed(),data,{})

def test_candidate_shape_mismatch():
    from scipy.sparse import csr_matrix
    from data_loader.schema import EvaluationData
    class Incomplete:
        def score(self,users,items):return np.zeros((len(users),len(items)-1))
    d=EvaluationData(csr_matrix([[1,0]]),csr_matrix([[0,1]]),np.array([0]),2,'fixed')
    with pytest.raises(ValueError,match='candidate scores'):evaluate(Incomplete(),d,{})
