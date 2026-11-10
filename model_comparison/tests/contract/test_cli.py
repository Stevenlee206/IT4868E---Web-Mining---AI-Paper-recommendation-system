import subprocess
import sys
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]

def cli(*args):
    return subprocess.run([sys.executable,str(ROOT/'main.py'),*map(str,args)],cwd=ROOT,capture_output=True,text=True)

def test_help_and_invalid_flags(tmp_path):
    assert cli('--help').returncode==0
    assert cli('run','--model','unknown').returncode==2
    assert cli('run','--data-dir','x','--prepared-dir','y').returncode==2
    bad=tmp_path/'bad.json';bad.write_text('{"unknown":{}}')
    assert cli('run','--config',bad).returncode==2
    assert cli('prepare','--data-dir','missing','--prepared-dir',tmp_path/'out').returncode==3

def test_prepare_and_run_cli(raw_dir,tmp_path):
    p=tmp_path/'prepared'
    assert cli('prepare','--data-dir',raw_dir,'--prepared-dir',p).returncode==0
    r=cli('run','--config','configs/smoke.json','--prepared-dir',p,'--model','item_knn','--output-dir',tmp_path/'runs')
    assert r.returncode==0,r.stderr
    report=json.loads(next((tmp_path/'runs').glob('*/comparison.json')).read_text())
    assert report['results'][0]['status']=='completed'
    assert cli('run','--prepared-dir',p,'--model','item_knn','--seed',7,'--output-dir',tmp_path/'conflict').returncode==3

def test_itemknn_does_not_import_torch(tmp_path):
    code="import sys; from main import main; code=main(['run','--config','configs/smoke.json','--model','item_knn','--output-dir',sys.argv[1]]); assert 'torch' not in sys.modules; raise SystemExit(code)"
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'runs')],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_paths_and_precedence(tmp_path):
    from pipeline.config import load_config
    config=tmp_path/'config.json'; config.write_text('{"data":{"data_dir":"relative"},"split":{"seed":7}}')
    c=load_config(config,{'experiment':{'models':'wmf'}})
    assert c['data']['data_dir']==str((tmp_path/'relative').resolve())
    assert c['experiment']['model_seed']==7 and c['experiment']['models']=='wmf'

@pytest.mark.parametrize('section,key,value',[('split','seed',-1),('neumf','negative_ratio',-1),('wmf','regularization',0),('ctr','numerical_floor',1),('evaluation','top_n',20),('experiment','threads',True)])
def test_invalid_config(section,key,value):
    from pipeline.config import load_config
    with pytest.raises(ValueError): load_config(overrides={section:{key:value}})

def test_unavailable_cuda(tmp_path):
    import torch
    if torch.cuda.is_available(): pytest.skip('CUDA is available on this environment')
    assert cli('run','--config','configs/smoke.json','--model','neumf','--device','cuda','--output-dir',tmp_path/'cuda').returncode==2
