import numpy as np
import pytest
from pipeline.config import load_config
from pipeline.runner import fit_model
from evaluation.ranking import rank_users

@pytest.mark.parametrize('name',['neumf','ctr'])
def test_advanced_model(prepared,name):
    config=load_config(overrides={'neumf':{'gmf_dim':3,'mlp_embedding_dim':3,'hidden_sizes':[6,3],'epochs':2,'batch_size':8},'ctr':{'topics':2,'lda_iterations':2,'outer_iterations':2,'topic_iterations':50}})
    model,summary=fit_model(name,prepared.training_data(),config)
    for ranking in rank_users(model,prepared.train,np.array([0,1,2]),50):
        assert len(set(ranking.item_ids))==len(ranking.item_ids)
        assert not set(ranking.item_ids)&set(prepared.train[ranking.user_id].indices)
        assert np.isfinite(ranking.scores).all() and 5 in ranking.item_ids
