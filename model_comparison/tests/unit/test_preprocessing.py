import numpy as np
from data_loader.preprocessing import prepare_dataset

def test_split_identity_and_no_leak(raw_dir):
    a=prepare_dataset(raw_dir,42); b=prepare_dataset(raw_dir,42)
    assert (a.train!=b.train).nnz==0 and (a.test!=b.test).nnz==0
    assert a.train.multiply(a.test).nnz==0
    assert a.train.shape==(5,6) and a.bow.shape==(6,4)
    assert np.array_equal(a.test.getnnz(axis=1),[1,1,1,0,0])
    assert a.manifest['counts']['duplicates']==1
    assert not hasattr(a.training_data(),'test')
    assert not hasattr(a.training_data(),'catalog')
    assert list(a.evaluation_data().eligible_user_ids)==[0,1,2]
    assert a.manifest['split_id']==b.manifest['split_id']
