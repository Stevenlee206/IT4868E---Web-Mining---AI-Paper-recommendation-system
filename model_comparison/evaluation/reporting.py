"""Readable and machine-readable experiment artifacts."""
import csv
from pathlib import Path
from collections import Counter
from data_loader.persistence import write_json

METRICS=['recall_20','recall_50','ndcg_20','ndcg_50']

def write_csv(path,fields,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)

def write_model_results(path,result,dataset,name):
    p=Path(path); p.mkdir(parents=True,exist_ok=True)
    write_json(p/'metrics.json',{'schema_version':1,'model':name,'dataset_id':dataset.manifest['dataset_id'],'split_id':dataset.manifest['split_id'],'eligible_users':len(dataset.evaluation_data().eligible_user_ids),'evaluated_users':result.evaluated_users,'excluded_users':len(dataset.exclusions),'exclusions_by_reason':dict(Counter(x['reason'] for x in dataset.exclusions)),'candidate_protocol':'full_catalog_minus_training','tie_policy':'descending_score_then_ascending_paper_id','cutoffs':[20,50],'metrics':result.metrics})
    write_csv(p/'per_user_metrics.csv',['user_id',*METRICS],result.per_user)
    def recommendation_rows():
        for ranking in result.rankings:
            for rank,(item,score) in enumerate(zip(ranking.item_ids,ranking.scores),1):
                yield {'user_id':ranking.user_id,'paper_id':int(item),'rank':rank,'score':float(score),'title':dataset.catalog[item].title}
    write_csv(p/'recommendations.csv',['user_id','paper_id','rank','score','title'],recommendation_rows())

def write_comparison(path,comparison):
    p=Path(path); p.mkdir(parents=True,exist_ok=True)
    write_json(p/'comparison.json',comparison)
    rows=[]
    for model in comparison['results']:
        rows.append({**model,**(model.get('metrics') or {})})
    write_csv(p/'comparison.csv',['model','status',*METRICS,'evaluated_users','train_seconds','evaluate_seconds','error'],rows)

def print_comparison(comparison):
    print('model       status       Recall@20 Recall@50 NDCG@20  NDCG@50')
    for result in comparison['results']:
        metrics=result.get('metrics')
        values=' '.join(f'{metrics[k]:.6f}' for k in METRICS) if metrics else '--       --       --       --'
        print(f"{result['model']:<11} {result['status']:<12} {values}")
