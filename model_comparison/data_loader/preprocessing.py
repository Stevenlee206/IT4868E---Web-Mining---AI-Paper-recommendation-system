"""Validated per-user preparation, with no catalog row filtering."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.sparse import csr_matrix
from .schema import PreparedDataset, DataError
from .content import load_bow, load_vocabulary
from .interactions import load_interactions
from .metadata import load_metadata

FILES=('users.dat','raw-data.csv','mult.dat','vocabulary.dat')
RULE='min(n-1,max(1,ceil(0.2*n))) for n>=2; singleton training-only'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def split_identity(train,test,seed):
    h=hashlib.sha256(json.dumps({'shape':train.shape,'seed':seed,'rule':RULE},sort_keys=True).encode())
    for matrix in (train,test):
        h.update(np.asarray(matrix.indptr,dtype='<i8').tobytes()); h.update(np.asarray(matrix.indices,dtype='<i8').tobytes())
    return h.hexdigest()

def prepare_dataset(data_dir, seed=42, metadata_encoding='auto'):
    if type(seed) is not int or seed<0: raise DataError('split seed must be nonnegative integer')
    root=Path(data_dir).resolve()
    for name in FILES:
        if not (root/name).is_file(): raise DataError(f'missing required file: {root/name}')
    vocab=load_vocabulary(root/'vocabulary.dat'); bow=load_bow(root/'mult.dat',len(vocab))
    metadata=load_metadata(root/'raw-data.csv',bow.shape[0],metadata_encoding)
    parsed=load_interactions(root/'users.dat',bow.shape[0]); raw=parsed.matrix
    rng=np.random.default_rng(seed); tr=[]; te=[]; trp=[0]; tep=[0]; exclusions=[]
    for user in range(raw.shape[0]):
        items=raw.indices[raw.indptr[user]:raw.indptr[user+1]]
        n=len(items)
        if n>=2:
            held=np.sort(rng.choice(items,size=min(n-1,max(1,math.ceil(.2*n))),replace=False))
            kept=np.setdiff1d(items,held,assume_unique=True)
        else:
            held=np.array([],dtype=int); kept=items
            exclusions.append({'user_id':user,'reason':'no_interactions' if n==0 else 'singleton'})
        tr.extend(kept); te.extend(held); trp.append(len(tr)); tep.append(len(te))
    train=csr_matrix((np.ones(len(tr),dtype=np.float32),tr,trp),shape=raw.shape)
    test=csr_matrix((np.ones(len(te),dtype=np.float32),te,tep),shape=raw.shape)
    sources={name:{'sha256':sha256(root/name),'bytes':(root/name).stat().st_size} for name in FILES}
    manifest={'schema_version':1,'parser_version':'1','source_directory':str(root),'source_files':sources,'metadata_encoding_requested':metadata_encoding,'metadata_encoding_used':metadata.encoding,'decoding_warnings':metadata.warnings,'split':{'seed':seed,'test_fraction':.2,'rounding_rule':RULE},'shapes':{'train':list(train.shape),'test':list(test.shape),'bow':list(bow.shape)},'counts':{'users':raw.shape[0],'items':raw.shape[1],'terms':len(vocab),'raw_pairs':parsed.raw_pairs,'unique_pairs':raw.nnz,'duplicates':parsed.duplicates,'train_pairs':train.nnz,'test_pairs':test.nnz,'bow_nonzeros':bow.nnz}}
    manifest['dataset_id']=identity({k:manifest[k] for k in ('schema_version','parser_version','source_files','metadata_encoding_requested','metadata_encoding_used','split')})
    manifest['split_id']=split_identity(train,test,seed)
    return PreparedDataset(metadata.papers,vocab,train,test,bow,manifest,exclusions)
