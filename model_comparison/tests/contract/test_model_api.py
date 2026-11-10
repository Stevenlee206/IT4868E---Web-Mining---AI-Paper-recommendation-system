import numpy as np
import pytest
from models import create_model

@pytest.mark.parametrize('name',['item_knn','wmf','neumf','ctr'])
def test_model_contract(prepared,name):
    config={'factors':3,'iterations':2,'neighbors':2,'topics':2,'lda_iterations':2,'outer_iterations':2,'topic_iterations':50,'epochs':1,'gmf_dim':3,'mlp_embedding_dim':3,'hidden_sizes':[6,3]}
    m=create_model(name,config,seed=42)
    with pytest.raises(ValueError): m.score(np.array([0]),np.array([0]))
    m.fit(prepared.training_data())
    a=m.score(np.array([2,0]),np.array([5,1]))
    assert a.shape==(2,2) and np.isfinite(a).all()
    with pytest.raises(ValueError): m.score(np.array([99]),np.array([0]))
    with pytest.raises(ValueError): m.score(np.array([4]),np.array([0]))
    with pytest.raises(ValueError): m.score(np.array([0]),np.array([99]))
    with pytest.raises(ValueError): m.fit(prepared)
