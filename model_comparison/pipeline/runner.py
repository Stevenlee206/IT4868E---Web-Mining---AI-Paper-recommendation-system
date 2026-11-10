"""Stage orchestration. Numerical imports remain inside command paths."""
def prepare(config, output_dir):
    from data_loader.preprocessing import prepare_dataset
    from data_loader.persistence import save_prepared
    data=prepare_dataset(config['data']['data_dir'],config['split']['seed'],config['data']['metadata_encoding'])
    save_prepared(data,output_dir)
    return data

def fit_model(name,training_data,config):
    from models import create_model
    from pipeline.reproducibility import model_seed,setup
    options=config['experiment']; seed=model_seed(options['model_seed'],name)
    setup(seed,options['threads'],options['deterministic'],torch_backend=name=='neumf')
    model=create_model(name,config[name],seed,device=options['device'])
    summary=model.fit(training_data)
    return model,summary

def run_experiment(config, explicit_seed=False):
    import gc
    import sys
    import time
    import uuid
    from pathlib import Path
    from datetime import datetime, timezone
    from dataclasses import asdict
    from data_loader.persistence import load_prepared, write_json
    from evaluation.evaluator import evaluate
    from evaluation.reporting import write_model_results,write_comparison,print_comparison
    from pipeline.config import NAMES
    from pipeline.reproducibility import environment,model_seed
    from pipeline.resources import memory_usage
    options=config['experiment']
    if options['device']=='cuda':
        import torch
        if not torch.cuda.is_available(): raise ValueError('requested CUDA unavailable')
    parent=Path(options['output_dir']).resolve()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    output=parent/run_id; output.mkdir(parents=True,exist_ok=False)
    if config['data']['prepared_dir']:
        data=load_prepared(config['data']['prepared_dir'],expected_seed=config['split']['seed'])
    else:
        data=prepare(config,output/'prepared')
    for warning in data.manifest['decoding_warnings']: print('data: '+warning,file=sys.stderr)
    write_json(output/'config.json',config)
    env=environment(); env.update({'device':options['device'],'deterministic':options['deterministic']})
    write_json(output/'environment.json',env)
    selected=NAMES if options['models']=='all' else [options['models']]
    comparison={'schema_version':1,'experiment_id':run_id,'output_dir':str(output),'dataset_id':data.manifest['dataset_id'],'split_id':data.manifest['split_id'],'eligible_users':len(data.evaluation_data().eligible_user_ids),'excluded_users':len(data.exclusions),'candidate_protocol':'full_catalog_minus_training','status':'running','results':[]}
    for name in selected:
        destination=output/'models'/name; destination.mkdir(parents=True)
        row={'model':name,'status':'training','stage':'training','metrics':None,'evaluated_users':0,'settings':config[name],'seed':model_seed(options['model_seed'],name),'started_at':datetime.now(timezone.utc).isoformat(),'error':None,'train_seconds':None,'evaluate_seconds':None,'artifact_path':str(destination.relative_to(output))}
        write_json(destination/'status.json',row)
        model=None; result=None; started=time.perf_counter()
        print(f'{name}: training',file=sys.stderr,flush=True)
        try:
            model,summary=fit_model(name,data.training_data(),config)
            row['train_seconds']=time.perf_counter()-started; row['fit_summary']=asdict(summary)
            row.update(status='evaluating',stage='evaluation'); write_json(destination/'status.json',row)
            print(f'{name}: evaluating',file=sys.stderr,flush=True)
            before=time.perf_counter(); result=evaluate(model,data.evaluation_data(),config['evaluation'])
            row['evaluate_seconds']=time.perf_counter()-before
            write_model_results(destination,result,data,name)
            if options['checkpoint']:
                try:
                    model.save_checkpoint(destination/'checkpoint')
                    write_json(destination/'checkpoint'/'provenance.json',{'dataset_id':data.manifest['dataset_id'],'split_id':data.manifest['split_id']})
                except Exception as exc:
                    row['checkpoint_warning']=str(exc); print(f'{name}: checkpoint warning: {exc}',file=sys.stderr)
            row.update(status='completed',stage='completed',metrics=result.metrics,evaluated_users=result.evaluated_users)
        except KeyboardInterrupt:
            row.update(status='interrupted',error='KeyboardInterrupt')
            write_json(destination/'status.json',row)
            comparison['results'].append(row); comparison['status']='interrupted'; write_comparison(output,comparison)
            raise
        except Exception as exc:
            row.update(status='failed',error=f'{type(exc).__name__}: {exc}',metrics=None)
            print(f"{name}: {row['stage']} failed: {row['error']}",file=sys.stderr,flush=True)
        finally:
            row['ended_at']=datetime.now(timezone.utc).isoformat(); row['duration_seconds']=time.perf_counter()-started
            row['memory']=memory_usage()
            write_json(destination/'status.json',row)
            del model,result; gc.collect()
        comparison['results'].append(row); write_comparison(output,comparison)
    comparison['status']='completed' if all(r['status']=='completed' for r in comparison['results']) else 'partial_failed'
    write_comparison(output,comparison); print_comparison(comparison)
    print(f'Results: {output}',flush=True)
    return comparison
