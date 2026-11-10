import json
import csv
from evaluation.reporting import write_model_results, write_comparison
from evaluation.evaluator import evaluate
from models import create_model

def test_report_contract(prepared,tmp_path):
    m=create_model('item_knn',{'neighbors':2}); m.fit(prepared.training_data())
    result=evaluate(m,prepared.evaluation_data(),{'cutoffs':[20,50],'top_n':50,'user_batch_size':2,'pair_batch_size':20})
    write_model_results(tmp_path/'model',result,prepared,'item_knn')
    metrics=json.loads((tmp_path/'model/metrics.json').read_text())
    assert metrics['schema_version']==1 and metrics['evaluated_users']==3
    rows=list(csv.DictReader((tmp_path/'model/recommendations.csv').open()))
    assert {'user_id','paper_id','rank','score','title'}==set(rows[0])
    report={'schema_version':1,'results':[{'model':'ctr','status':'failed','metrics':None,'error':'example'}]}
    write_comparison(tmp_path,report)
    assert json.loads((tmp_path/'comparison.json').read_text())['results'][0]['metrics'] is None
    row=list(csv.DictReader((tmp_path/'comparison.csv').open()))[0]
    assert row['recall_20']=='' and row['error']=='example'
