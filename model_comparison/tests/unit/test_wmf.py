import numpy as np
from scipy.sparse import csr_matrix
from implicit.cpu.als import least_squares
from models.wmf import WMF

def test_confidence_and_analytical_solve(prepared):
    m=WMF({'factors':3,'confidence_scale':4,'regularization':.1,'iterations':2})
    m.fit(prepared.training_data())
    assert np.all(m.confidence.data==5) and m.backend.alpha==1
    assert m.user_factors.shape==(5,3) and m.item_factors.shape==(6,3)
    y=np.array([[1.,0.],[0.,1.],[1.,1.]],dtype=np.float32)
    c=csr_matrix([[5.,0,5.]],dtype=np.float32); x=np.zeros((1,2),dtype=np.float32)
    least_squares(c,x,y,.1,num_threads=1)
    diag=np.array([5.,1.,5.]); expected=np.linalg.solve(y.T@(diag[:,None]*y)+.1*np.eye(2),y.T@np.array([5.,0.,5.]))
    assert np.allclose(x[0],expected,atol=1e-6)
    assert np.isfinite(m.score(np.array([0]),np.arange(6))).all()
