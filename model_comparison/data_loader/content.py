"""Read the supplied count-based content without retokenizing."""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .schema import DataError

def load_vocabulary(path):
    terms=tuple(Path(path).read_text(encoding='utf-8-sig').splitlines())
    if not terms or any(not term for term in terms): raise DataError(f'{path}: empty vocabulary term')
    return terms

def load_bow(path, vocab_size):
    path=Path(path); indices=[]; values=[]; indptr=[0]
    for record,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        try:
            parts=line.split()
            if not parts or int(parts[0])<0 or int(parts[0])!=len(parts)-1: raise ValueError('declared unique-term count mismatch')
            terms={}
            for pair in parts[1:]:
                term,count=map(int,pair.split(':'))
                if not 0<=term<vocab_size or count<=0 or term in terms: raise ValueError('invalid/duplicate term or nonpositive count')
                terms[term]=count
            for term,count in sorted(terms.items()): indices.append(term); values.append(count)
            indptr.append(len(indices))
        except ValueError as exc: raise DataError(f'{path}: record {record}: {exc}') from exc
    if len(indptr)==1: raise DataError(f'{path}: no paper content rows')
    return csr_matrix((np.asarray(values,dtype=np.int64),indices,indptr),shape=(len(indptr)-1,vocab_size))
