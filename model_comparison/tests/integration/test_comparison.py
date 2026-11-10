from pathlib import Path
import json
from pipeline.config import load_config
from pipeline.runner import run_experiment

def config(tmp_path,model='all'):
    return load_config('configs/smoke.json',{'experiment':{'output_dir':str(tmp_path),'models':model}})

def signatures(report):
    signatures={}
    for result in report['results']:
        path=Path(report['output_dir'])/result['artifact_path']
        signatures[result['model']]={'metrics':result['metrics'],'seed':result['seed'],'rankings':(path/'recommendations.csv').read_text(),'per_user':(path/'per_user_metrics.csv').read_text()}
    return signatures

def test_comparison_repeatability_and_single_model(tmp_path):
    a=run_experiment(config(tmp_path));b=run_experiment(config(tmp_path))
    assert a['status']==b['status']=='completed'
    assert a['experiment_id']!=b['experiment_id']
    assert a['split_id']==b['split_id']
    assert len(a['results'])==4 and all(r['evaluated_users']==3 for r in a['results'])
    assert signatures(a)==signatures(b)
    single=run_experiment(config(tmp_path,'neumf'))
    assert signatures(single)['neumf']==signatures(a)['neumf']

def test_failure_isolation(tmp_path,monkeypatch):
    from pipeline import runner
    original=runner.fit_model
    def failing(name,data,c):
        if name=='wmf': raise RuntimeError('simulated training failure')
        return original(name,data,c)
    monkeypatch.setattr(runner,'fit_model',failing)
    report=run_experiment(config(tmp_path))
    assert report['status']=='partial_failed' and len(report['results'])==4
    assert [r['status'] for r in report['results']]==['completed','failed','completed','completed']
    assert report['results'][1]['metrics'] is None

def test_optional_checkpoints(tmp_path):
    c=config(tmp_path);c['experiment']['checkpoint']=True
    report=run_experiment(c)
    assert report['status']=='completed'
    for row in report['results']:
        p=Path(report['output_dir'])/row['artifact_path']/'checkpoint'
        assert (p/'metadata.json').exists() and (p/'provenance.json').exists()
        assert 'checkpoint_warning' not in row
