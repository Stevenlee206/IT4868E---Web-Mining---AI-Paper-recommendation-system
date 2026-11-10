from data_loader.preprocessing import prepare_dataset, sha256, FILES
from data_loader.persistence import save_prepared, load_prepared

def test_prepare_roundtrip(raw_dir,tmp_path):
    before={f:sha256(raw_dir/f) for f in FILES}
    a=prepare_dataset(raw_dir,42); b=prepare_dataset(raw_dir,42)
    for name,data in [('a',a),('b',b)]: save_prepared(data,tmp_path/name)
    assert load_prepared(tmp_path/'a').manifest['dataset_id']==load_prepared(tmp_path/'b').manifest['dataset_id']
    assert before=={f:sha256(raw_dir/f) for f in FILES}
