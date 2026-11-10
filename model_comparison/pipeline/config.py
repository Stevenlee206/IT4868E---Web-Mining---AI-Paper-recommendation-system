"""Configuration validation without importing numerical backends."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path

NAMES = ('item_knn','wmf','neumf','ctr')
DEFAULTS = {
 'data': {'data_dir':None,'prepared_dir':None,'metadata_encoding':'auto'},
 'split': {'seed':42,'test_fraction':0.2},
 'experiment': {'models':'all','model_seed':42,'output_dir':'artifacts/runs','deterministic':True,'threads':1,'device':'cpu','checkpoint':False},
 'item_knn': {'neighbors':100,'similarity_block_size':256},
 'wmf': {'factors':64,'confidence_scale':40.0,'regularization':0.1,'iterations':20,'use_cg':False},
 'neumf': {'gmf_dim':32,'mlp_embedding_dim':32,'hidden_sizes':[64,32,16],'learning_rate':0.001,'epochs':10,'batch_size':1024,'negative_ratio':4,'weight_decay':0.0},
 'ctr': {'topics':32,'lda_iterations':10,'outer_iterations':10,'topic_iterations':20,'lambda_u':0.1,'lambda_v':1.0,'confidence_observed':1.0,'confidence_unobserved':0.01,'tolerance':1e-4,'numerical_floor':1e-12},
 'evaluation': {'cutoffs':[20,50],'top_n':50,'user_batch_size':32,'pair_batch_size':8192},
}

@dataclass(frozen=True)
class Settings:
    values: dict


def load_config(path=None, overrides=None):
    config = deepcopy(DEFAULTS)
    if path:
        path = Path(path).resolve()
        incoming = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(incoming, dict): raise ValueError('configuration must be an object')
        for section, values in incoming.items():
            if section not in config or not isinstance(values,dict): raise ValueError(f'unknown/invalid configuration section: {section}')
            for key, value in values.items():
                if key not in config[section]: raise ValueError(f'unknown configuration key: {section}.{key}')
                if key in ('data_dir','prepared_dir','output_dir') and value is not None:
                    value = str((path.parent/str(value)).resolve())
                config[section][key] = value
        if 'seed' in incoming.get('split',{}) and 'model_seed' not in incoming.get('experiment',{}):
            config['experiment']['model_seed'] = config['split']['seed']
    for section, values in (overrides or {}).items():
        for key,value in values.items():
            if key not in config.get(section,{}): raise ValueError(f'unknown configuration key: {section}.{key}')
            config[section][key] = value
    validate(config)
    return config


def validate(c):
    integer = {'seed','model_seed','threads','neighbors','similarity_block_size','factors','iterations','gmf_dim','mlp_embedding_dim','epochs','batch_size','negative_ratio','topics','lda_iterations','outer_iterations','topic_iterations','top_n','user_batch_size','pair_batch_size'}
    for section, values in c.items():
        for key, value in values.items():
            if key in integer:
                lower = 0 if key in ('seed','model_seed','negative_ratio') else 1
                if type(value) is not int or value < lower: raise ValueError(f'{section}.{key} must be integer >= {lower}')
    for section, keys in {'wmf':['confidence_scale','regularization'], 'neumf':['learning_rate'], 'ctr':['lambda_u','lambda_v','confidence_observed','confidence_unobserved','tolerance','numerical_floor']}.items():
        for key in keys:
            v=c[section][key]
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0: raise ValueError(f'{section}.{key} must be finite and positive')
    wd=c['neumf']['weight_decay']
    if type(wd) not in (int,float) or not math.isfinite(wd) or wd<0: raise ValueError('weight_decay must be finite and nonnegative')
    if c['split']['test_fraction'] != 0.2: raise ValueError('test_fraction must be 0.2')
    if c['experiment']['models'] not in (*NAMES,'all'): raise ValueError('unknown model')
    if c['experiment']['device'] not in ('cpu','cuda'): raise ValueError('device must be cpu or cuda')
    for section,key in [('experiment','deterministic'),('experiment','checkpoint'),('wmf','use_cg')]:
        if type(c[section][key]) is not bool: raise ValueError(f'{key} must be boolean')
    hidden=c['neumf']['hidden_sizes']
    if not isinstance(hidden,list) or not hidden or any(type(v) is not int or v<=0 for v in hidden): raise ValueError('hidden_sizes must contain positive integers')
    if c['evaluation']['cutoffs'] != [20,50] or c['evaluation']['top_n']<50: raise ValueError('cutoffs must be [20,50] and top_n >=50')
    ctr=c['ctr']
    if ctr['confidence_observed']<=ctr['confidence_unobserved']: raise ValueError('CTR requires observed confidence > unobserved confidence')
    if ctr['numerical_floor']*max(ctr['topics'],1)>=1 or ctr['tolerance']>=1: raise ValueError('CTR numerical floor/tolerance too large')
    for key in ('data_dir','prepared_dir','metadata_encoding'):
        if c['data'][key] is not None and not isinstance(c['data'][key],str): raise ValueError(f'{key} must be a path/encoding string')
    if not isinstance(c['experiment']['output_dir'],str): raise ValueError('output_dir must be a path string')
