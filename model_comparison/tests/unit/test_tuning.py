"""Validation split/selection must remain independent of held-out test labels."""
from copy import deepcopy
import json
import numpy as np
import pytest
from pipeline.config import load_config
from pipeline.tuning import validation_split, candidate_configs, select_configs


def test_inner_split_only_uses_training(prepared):
    train, val = validation_split(prepared.training_data(), seed=71)
    assert train.train.multiply(val.test).nnz == 0
    assert train.train.multiply(prepared.test).nnz == 0
    assert val.test.multiply(prepared.test).nnz == 0
    assert ((train.train + val.test) != prepared.train).nnz == 0
    other, again = validation_split(prepared.training_data(), seed=71)
    assert val.split_id == again.split_id
    assert not hasattr(train, 'test')
    with pytest.raises(TypeError): validation_split(prepared)


def test_validate_search_before_fitting():
    c = load_config('configs/smoke.json')
    assert len(candidate_configs(c, 'wmf', [{}, {'factors':3}])) == 1
    with pytest.raises(ValueError): candidate_configs(c, 'wmf', [{'typo':1}])
    with pytest.raises(ValueError): candidate_configs(c, 'neumf', [{'weight_decay':-1}])


def test_selection_uses_validation_not_test(prepared, tmp_path, monkeypatch):
    from pipeline import tuning
    class Summary:
        elapsed_time = 0
    calls = []
    def fake_fit(name, data, config):
        assert not hasattr(data, 'test')
        calls.append(config[name]['neighbors'])
        return config[name]['neighbors'], Summary()
    class Result:
        def __init__(self, value):self.metrics={'ndcg_20':value, 'recall_50':value}
    monkeypatch.setattr(tuning, 'fit_model', fake_fit)
    monkeypatch.setattr(tuning, 'evaluate', lambda model, data, config: Result({2:.1,3:.3,4:.2}[model]))
    c=load_config('configs/smoke.json', {'experiment':{'models':'item_knn'}})
    selected, report=select_configs(prepared.training_data(), c, {'item_knn':[{'neighbors':3},{'neighbors':4}]}, tmp_path/'tuning')
    assert calls == [2,3,4] and selected['item_knn']['neighbors']==3
    assert report['outer_test_used_for_selection'] is False
    assert json.loads((tmp_path/'tuning/tuning.json').read_text())['status']=='completed'
    assert c['item_knn']['neighbors']==2

def test_resume_reuses_completed_candidates(prepared,tmp_path,monkeypatch):
    from pipeline import tuning
    c=load_config('configs/smoke.json',{'experiment':{'models':'item_knn'}})
    destination=tmp_path/'resume'
    first,report=select_configs(prepared.training_data(),c,{},destination)
    def must_not_fit(*args):raise AssertionError('completed trial rerun')
    monkeypatch.setattr(tuning,'fit_model',must_not_fit)
    second,resumed=select_configs(prepared.training_data(),c,{},destination,resume=True)
    assert first==second and resumed['resumed_completed_trials']==1
    with pytest.raises(ValueError,match='identical'):
        select_configs(prepared.training_data(),c,{'item_knn':[{'neighbors':3}]},destination,resume=True)
