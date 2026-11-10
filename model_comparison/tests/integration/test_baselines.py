import numpy as np
import pytest
from models import create_model
from evaluation.ranking import rank_users

@pytest.mark.parametrize('name',['item_knn','wmf'])
def test_baseline(prepared,name):
    m=create_model(name,{'factors':3,'iterations':2,'neighbors':2},seed=42)
    m.fit(prepared.training_data())
    for out in rank_users(m,prepared.train,np.array([0,1,2]),50):
        seen=prepared.train[out.user_id].indices
        assert not set(out.item_ids)&set(seen)
        assert len(set(out.item_ids))==len(out.item_ids)
        assert 5 in out.item_ids
