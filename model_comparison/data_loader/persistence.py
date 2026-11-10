"""Validated sparse persistence; manifest is published last."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import warnings
import numpy as np
from scipy.sparse import save_npz, load_npz, csr_matrix
from .schema import PreparedDataset, Paper, DataError
from .preprocessing import sha256, split_identity, identity, RULE

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)

def save_prepared(dataset, output_dir):
    p=Path(output_dir)
    if p.exists() and any(p.iterdir()): raise DataError(f'{p}: refusing to overwrite prepared artifacts')
    p.mkdir(parents=True,exist_ok=True)
    for name in ('train','test','bow'):
        temporary=p/f'{name}.tmp.npz'; save_npz(temporary,getattr(dataset,name)); os.replace(temporary,p/f'{name}.npz')
    (p/'papers.jsonl').write_text(''.join(json.dumps(asdict(paper),ensure_ascii=True)+'\n' for paper in dataset.catalog),encoding='utf-8')
    write_json(p/'vocabulary.json',dataset.vocabulary); write_json(p/'exclusions.json',dataset.exclusions)
    manifest=dict(dataset.manifest)
    manifest['artifact_sha256']={f.name:sha256(f) for f in p.iterdir() if f.is_file()}
    write_json(p/'manifest.json',manifest)
    return p

def load_prepared(path, expected_config=None, expected_seed=None):
    p=Path(path)
    try:
        m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
        if m['schema_version']!=1: raise ValueError('unsupported schema')
        if m['parser_version']!='1' or m['split']['rounding_rule']!=RULE or m['split']['test_fraction']!=0.2: raise ValueError('unsupported parser/split policy')
        if type(m['split']['seed']) is not int or m['split']['seed']<0: raise ValueError('invalid split seed')
        expected_id=identity({k:m[k] for k in ('schema_version','parser_version','source_files','metadata_encoding_requested','metadata_encoding_used','split')})
        if expected_id!=m['dataset_id']: raise ValueError('dataset fingerprint mismatch')
        if expected_config is not None: expected_seed=expected_config['split']['seed']
        if expected_seed is not None and m['split']['seed']!=expected_seed: raise ValueError('prepared split seed mismatch')
        required={'train.npz','test.npz','bow.npz','papers.jsonl','vocabulary.json','exclusions.json'}
        if set(m['artifact_sha256'])!=required: raise ValueError('incomplete artifact manifest')
        for name,digest in m['artifact_sha256'].items():
            if sha256(p/name)!=digest: raise ValueError(f'artifact hash mismatch: {name}')
        matrices={name:csr_matrix(load_npz(p/f'{name}.npz')) for name in ('train','test','bow')}
        for name,a in matrices.items():
            a.check_format(full_check=True)
            if list(a.shape)!=m['shapes'][name] or not a.has_canonical_format: raise ValueError(f'{name}: invalid shape or indices')
            if not np.isfinite(a.data).all() or np.any(a.data<=0): raise ValueError('invalid matrix values')
            if name!='bow' and not np.all(a.data==1): raise ValueError('interaction matrix is not binary')
            if name=='bow' and not np.all(a.data==np.floor(a.data)): raise ValueError('fractional counts')
        train,test,bow=(matrices[k] for k in ('train','test','bow'))
        if train.shape!=test.shape or train.shape[1]!=bow.shape[0] or train.multiply(test).nnz: raise ValueError('alignment or split overlap')
        catalog=[Paper(**json.loads(line)) for line in (p/'papers.jsonl').read_text(encoding='utf-8').splitlines()]
        vocab=tuple(json.loads((p/'vocabulary.json').read_text(encoding='utf-8')))
        if len(catalog)!=train.shape[1] or any(x.paper_id!=i or x.doc_id!=i+1 for i,x in enumerate(catalog)) or len(vocab)!=bow.shape[1]: raise ValueError('metadata/vocabulary alignment')
        if split_identity(train,test,m['split']['seed'])!=m['split_id']: raise ValueError('split fingerprint mismatch')
        total=train.getnnz(axis=1)+test.getnnz(axis=1)
        expected=np.where(total>=2,np.minimum(total-1,np.maximum(1,np.ceil(.2*total))),0)
        if not np.array_equal(expected,test.getnnz(axis=1)): raise ValueError('invalid split sizes')
        exclusions=json.loads((p/'exclusions.json').read_text(encoding='utf-8'))
        wanted=[{'user_id':int(i),'reason':'no_interactions' if n==0 else 'singleton'} for i,n in enumerate(total) if n<2]
        if exclusions!=wanted: raise ValueError('invalid exclusions')
        if (m['counts']['users'],m['counts']['items'],m['counts']['terms'])!=(train.shape[0],train.shape[1],bow.shape[1]) or m['counts']['unique_pairs']!=train.nnz+test.nnz: raise ValueError('manifest count mismatch')
        counts=m['counts']
        if counts['train_pairs']!=train.nnz or counts['test_pairs']!=test.nnz or counts['bow_nonzeros']!=bow.nnz or counts['raw_pairs']!=counts['unique_pairs']+counts['duplicates']: raise ValueError('manifest count mismatch')
        source=Path(m['source_directory'])
        if all((source/name).is_file() for name in m['source_files']):
            for name,details in m['source_files'].items():
                if sha256(source/name)!=details['sha256']: raise ValueError(f'source fingerprint mismatch: {name}')
        else: warnings.warn('Raw sources unavailable: artifact-only validation',RuntimeWarning)
        return PreparedDataset(catalog,vocab,train,test,bow,m,exclusions)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise DataError(f'{p}: prepared-artifact validation: {exc}') from exc
