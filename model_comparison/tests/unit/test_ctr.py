import numpy as np
from scipy.sparse import csr_matrix
from data_loader.schema import TrainingData
from models.ctr import CTR, update_theta, factor_update

def test_theta_objective():
    old=np.array([.5,.5]); v=np.array([.1,.9]); counts=np.array([2.,8.])
    theta=update_theta(old,v,counts,1.,1e-12,50)
    objective=lambda t: counts@np.log(t)-.5*np.sum((v-t)**2)
    assert np.isclose(theta.sum(),1) and np.all(theta>0)
    assert objective(theta)>=objective(old)

def test_factor_prior_solve():
    others=np.array([[1.,0.],[0.,1.]])
    result=factor_update(others,np.array([0]),1.,.1,.5,np.array([.2,.8]))
    expected=np.linalg.solve(np.diag([1.,.1])+.5*np.eye(2),np.array([1.,0.])+.5*np.array([.2,.8]))
    assert np.allclose(result,expected)

def test_coupled_topics(prepared):
    cfg={'topics':2,'lda_iterations':2,'outer_iterations':2,'topic_iterations':50,'lambda_v':20}
    a=CTR(cfg,seed=7); a.fit(prepared.training_data())
    altered=prepared.train.copy().tolil(); altered[0,:]=0; altered[0,4]=1
    d=TrainingData(altered.tocsr(),prepared.bow,5,6,'altered')
    b=CTR(cfg,seed=7); b.fit(d)
    assert np.allclose(a.theta.sum(axis=1),1) and np.allclose(a.beta.sum(axis=1),1)
    assert np.allclose(a.theta[5],[.5,.5])
    assert np.allclose(a.initial_theta,b.initial_theta)
    assert not np.allclose(a.theta[:5],b.theta[:5])
    assert np.isfinite(a.score(np.array([0]),np.arange(6))).all()
