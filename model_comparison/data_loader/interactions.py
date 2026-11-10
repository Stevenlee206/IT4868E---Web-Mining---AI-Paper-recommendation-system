"""Strict users.dat parser."""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .schema import DataError, InteractionParseResult

def load_interactions(path, num_items):
    path=Path(path); indices=[]; indptr=[0]; raw=0; duplicates=0
    for record,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        try:
            values=list(map(int,line.split()))
            if not values or values[0]<0 or values[0]!=len(values)-1: raise ValueError('declared interaction count mismatch')
            items=values[1:]
            if any(i<0 or i>=num_items for i in items): raise ValueError('paper ID out of range')
            unique=sorted(set(items)); raw+=len(items); duplicates+=len(items)-len(unique)
            indices.extend(unique); indptr.append(len(indices))
        except ValueError as exc: raise DataError(f'{path}: record {record}: {exc}') from exc
    if len(indptr)==1: raise DataError(f'{path}: no users')
    matrix=csr_matrix((np.ones(len(indices),dtype=np.float32),indices,indptr),shape=(len(indptr)-1,num_items))
    return InteractionParseResult(matrix,raw,duplicates)
