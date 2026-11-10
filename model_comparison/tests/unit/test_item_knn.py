import numpy as np
from models.item_knn import ItemKNN
from scipy.sparse import csr_matrix
from data_loader.schema import TrainingData

def test_cosine_and_pruning():
    r=csr_matrix([[1,1,0,0],[1,0,1,0],[0,1,0,0]],dtype=np.float32)
    d=TrainingData(r,csr_matrix((4,2)),3,4,'fixture')
    m=ItemKNN({'neighbors':1,'similarity_block_size':2}); m.fit(d)
    s=m.similarity.toarray()
    assert np.allclose(s[0],[0,0,1/np.sqrt(2),0])
    assert s[1,0]==0.5 and s[1,1]==0
    assert np.allclose(m.score(np.array([0]),np.arange(4)),r[[0]]@s.T)
    assert np.all(s[3]==0)
