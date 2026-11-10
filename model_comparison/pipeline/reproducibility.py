"""Seed controls; this module deliberately imports no ML backend at import time."""
import hashlib
import importlib.metadata
import os
import platform
import random
import sys

def model_seed(base, name):
    return int.from_bytes(hashlib.sha256(f'{base}:{name}'.encode()).digest()[:4], 'little')

def setup(seed=42, threads=1, deterministic=True, torch_backend=False):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=threads)
    if torch_backend:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
        if torch.cuda.is_available():
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=False

def environment():
    versions={}
    for name in ('numpy','scipy','implicit','torch','scikit-learn','pytest','threadpoolctl'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {'python':sys.version,'platform':platform.platform(),'versions':versions,'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},'reproducibility':'Same recorded environment only; cross-release/platform/device equality is not guaranteed.'}
