import numpy as np
import pytest
from evaluation.ranking import rank_users
from scipy.sparse import csr_matrix

class Scores:
    def __init__(self,scores): self.scores=np.array(scores)
    def score(self,users,items): return np.tile(self.scores[items],(len(users),1))

def test_ties_and_short_lists():
    r=csr_matrix([[1,0,0,0]])
    out=list(rank_users(Scores([100,0,0,-1]),r,np.array([0]),50,1,100))[0]
    assert out.item_ids.tolist()==[1,2,3]
    assert len(set(out.item_ids))==3

def test_bad_scores():
    with pytest.raises(ValueError): list(rank_users(Scores([0,np.nan]),csr_matrix([[1,0]]),np.array([0]),1,1,100))
