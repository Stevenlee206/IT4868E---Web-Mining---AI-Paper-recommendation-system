import math
import pytest
from evaluation.metrics import recall_at_k, ndcg_at_k

@pytest.mark.parametrize('k',[20,50])
def test_oracles(k):
    assert recall_at_k([1,2],[1,2],k)==1
    assert ndcg_at_k([1,2],[1,2],k)==1
    assert recall_at_k([3,4],[1,2],k)==0
    assert ndcg_at_k([3,4],[1,2],k)==0
    assert recall_at_k([2],[1,2],k)==.5
    assert ndcg_at_k([2],[1,2],k)==pytest.approx(1/(1+1/math.log2(3)),abs=1e-6)
    assert ndcg_at_k([4,1],[1],k)==pytest.approx(1/math.log2(3),abs=1e-6)
    assert recall_at_k(list(range(k+1)),[k],k)==0

@pytest.mark.parametrize('fn',[recall_at_k,ndcg_at_k])
@pytest.mark.parametrize('ranking,relevant,k',[([1,1],[1],20),([1],[],20),([1],[1],0)])
def test_invalid(fn,ranking,relevant,k):
    with pytest.raises(ValueError):fn(ranking,relevant,k)
