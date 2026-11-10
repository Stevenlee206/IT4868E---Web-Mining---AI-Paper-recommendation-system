import numpy as np
import pytest
import torch
from models.neumf import NeuMF, NeuMFNetwork, sample_negatives
from scipy.sparse import csr_matrix

def test_branches_receive_gradients():
    net=NeuMFNetwork(3,6,3,3,[6,3])
    loss=net(torch.tensor([0,1,2]),torch.tensor([1,2,3])).sum(); loss.backward()
    for name in ('gmf_user','gmf_item','mlp_user','mlp_item'):
        assert getattr(net,name).weight.grad.abs().sum()>0

def test_training_only_sampling():
    r=csr_matrix([[1,1,1,0],[0,1,0,0]],dtype=np.float32)
    users=np.array([0]*40+[1]*40)
    items=sample_negatives(r,users,np.random.default_rng(42))
    assert np.all(items[:40]==3)
    assert np.all(items[40:]!=1)
    with pytest.raises(ValueError): sample_negatives(csr_matrix([[1,1]]),np.array([0]),np.random.default_rng(1))

def test_neural_fit(prepared):
    m=NeuMF({'gmf_dim':3,'mlp_embedding_dim':3,'hidden_sizes':[6,3],'epochs':2,'batch_size':8,'negative_ratio':2},seed=42)
    result=m.fit(prepared.training_data())
    assert np.isfinite(result.diagnostics['losses']).all()
    m.pair_batch_size=3
    assert m.score(np.array([0,2]),np.array([5,1,0])).shape==(2,3)

def test_regularized_fit_restores_reference_numeric_mode(prepared,monkeypatch):
    calls=[]
    original=torch.set_flush_denormal
    def record(enabled):
        calls.append(enabled)
        return original(enabled)
    monkeypatch.setattr(torch,'set_flush_denormal',record)
    m=NeuMF({'gmf_dim':3,'mlp_embedding_dim':3,'hidden_sizes':[6,3],'epochs':1,'batch_size':8,'weight_decay':1e-5},seed=42)
    result=m.fit(prepared.training_data())
    assert calls==[True,False]
    assert result.diagnostics['flush_denormal_during_fit'] is True
