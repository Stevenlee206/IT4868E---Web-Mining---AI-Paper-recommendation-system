import pytest
from data_loader.interactions import load_interactions
from data_loader.metadata import load_metadata
from data_loader.content import load_bow, load_vocabulary

def test_fixture(raw_dir):
    parsed=load_interactions(raw_dir/'users.dat',6)
    assert parsed.matrix.shape==(5,6) and parsed.duplicates==1
    assert parsed.matrix.nnz==10 and parsed.raw_pairs==11
    assert load_metadata(raw_dir/'raw-data.csv',6).papers[0].doc_id==1
    assert len(load_vocabulary(raw_dir/'vocabulary.dat'))==4
    assert load_bow(raw_dir/'mult.dat',4).shape==(6,4)

@pytest.mark.parametrize('text',['2 0','1 6','-1','1 -1','1 x'])
def test_interaction_errors(raw_dir,text):
    p=raw_dir/'users.dat'; p.write_text(text)
    with pytest.raises(ValueError,match='users.dat.*record 1'): load_interactions(p,6)

@pytest.mark.parametrize('text',['2 0:1','2 0:1 0:2','1 4:1','1 0:0','1 0:1.5'])
def test_bow_errors(raw_dir,text):
    p=raw_dir/'mult.dat'; p.write_text(text)
    with pytest.raises(ValueError,match='mult.dat.*record 1'): load_bow(p,4)

def test_encoding_and_mapping(raw_dir):
    p=raw_dir/'raw-data.csv'
    p.write_bytes(p.read_bytes().replace(b'Paper one',b'Pap\xe9r one'))
    result=load_metadata(p,6)
    assert result.encoding=='latin-1' and result.warnings
    assert result.papers[0].paper_id==0
    assert result.papers[-1].abstract is None
    p.write_bytes(p.read_bytes().replace(b'6,"Paper six"',b'5,"Paper six"'))
    with pytest.raises(ValueError): load_metadata(p,6)

def test_extra_csv_field(raw_dir):
    p=raw_dir/'raw-data.csv'; p.write_text(p.read_text()+ '7,x,1,x,x,extra\n')
    with pytest.raises(ValueError): load_metadata(p,6)
