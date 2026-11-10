import pytest
from data_loader.persistence import save_prepared, load_prepared

def test_roundtrip_and_tamper(prepared,tmp_path):
    out=tmp_path/'prepared'; save_prepared(prepared,out)
    loaded=load_prepared(out,expected_seed=42)
    assert loaded.manifest['split_id']==prepared.manifest['split_id']
    assert (loaded.train!=prepared.train).nnz==0
    with pytest.raises(ValueError): load_prepared(out,expected_seed=1)
    with pytest.raises(ValueError): save_prepared(prepared,out)
    (out/'vocabulary.json').write_text('[]')
    with pytest.raises(ValueError,match='hash'): load_prepared(out)

def test_incomplete(tmp_path):
    with pytest.raises(ValueError): load_prepared(tmp_path)

def test_manifest_identity_and_counts(prepared,tmp_path):
    import json
    out=tmp_path/'prepared';save_prepared(prepared,out)
    path=out/'manifest.json'; original=json.loads(path.read_text())
    corrupt=dict(original);corrupt['dataset_id']='invalid'
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError,match='fingerprint'):load_prepared(out)
    corrupt=dict(original);corrupt['counts']=dict(original['counts']);corrupt['counts']['train_pairs']+=1
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError,match='count'):load_prepared(out)

def test_moved_artifacts_and_source_change(raw_dir,tmp_path):
    import shutil
    from data_loader.preprocessing import prepare_dataset
    data=prepare_dataset(raw_dir);out=tmp_path/'out';save_prepared(data,out)
    source=raw_dir/'users.dat';original=source.read_bytes();source.write_bytes(original+b'0\n')
    with pytest.raises(ValueError,match='source fingerprint'):load_prepared(out)
    source.write_bytes(original)
    shutil.move(str(raw_dir),str(tmp_path/'moved-raw'))
    with pytest.warns(RuntimeWarning,match='artifact-only'):loaded=load_prepared(out)
    assert loaded.manifest['split_id']==data.manifest['split_id']
