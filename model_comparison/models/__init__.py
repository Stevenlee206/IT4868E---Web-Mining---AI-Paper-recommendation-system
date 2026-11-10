"""Lazy factory: unrelated backends are never imported."""
from importlib import import_module

MODELS={'item_knn':('item_knn','ItemKNN'),'wmf':('wmf','WMF'),'neumf':('neumf','NeuMF'),'ctr':('ctr','CTR')}

def create_model(name,config=None,seed=42,**kwargs):
    if name not in MODELS: raise ValueError(f'unknown model: {name}')
    module,cls=MODELS[name]
    return getattr(import_module(f'models.{module}'),cls)(config,seed=seed,**kwargs)
