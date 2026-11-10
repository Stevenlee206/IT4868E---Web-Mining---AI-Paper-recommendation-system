"""Local paper recommendation experiment entry point."""
import argparse
import json
import sys
from pathlib import Path
from pipeline.config import load_config
from pipeline.reproducibility import setup

def main(argv=None):
    parser=argparse.ArgumentParser(description='CiteULike-A recommendations: full catalog minus training positives; deterministic paper-ID ties.')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run'):
        p=sub.add_parser(name)
        p.add_argument('--config'); p.add_argument('--data-dir'); p.add_argument('--prepared-dir')
        p.add_argument('--seed',type=int); p.add_argument('--metadata-encoding')
        if name=='run':
            p.add_argument('--model',choices=['item_knn','wmf','neumf','ctr','all'])
            p.add_argument('--model-seed',type=int); p.add_argument('--output-dir'); p.add_argument('--device',choices=['cpu','cuda'])
    args=parser.parse_args(argv)
    try:
        overrides={}
        def put(section,key,value):
            if value is not None: overrides.setdefault(section,{})[key]=value
        if args.command=='run' and args.data_dir and args.prepared_dir: raise ValueError('run requires exactly one input source')
        for key in ('data_dir','prepared_dir'):
            value=getattr(args,key,None)
            if value is not None: put('data',key,str(Path(value).resolve()))
        if args.command=='run':
            if args.data_dir: overrides.setdefault('data',{})['prepared_dir']=None
            if args.prepared_dir: overrides.setdefault('data',{})['data_dir']=None
            for flag,key in [('model','models'),('model_seed','model_seed'),('output_dir','output_dir'),('device','device')]: put('experiment',key,getattr(args,flag,None))
        if args.seed is not None:
            put('split','seed',args.seed)
            if getattr(args,'model_seed',None) is None: put('experiment','model_seed',args.seed)
        put('data','metadata_encoding',args.metadata_encoding)
        config=load_config(args.config,overrides)
        if args.command=='prepare':
            if not config['data']['data_dir'] or not config['data']['prepared_dir']: raise ValueError('prepare requires --data-dir and --prepared-dir')
        else:
            if bool(config['data']['data_dir'])==bool(config['data']['prepared_dir']): raise ValueError('run requires exactly one input source')
        setup(config['split']['seed'],config['experiment']['threads'],config['experiment']['deterministic'])
        from pipeline import runner
        if args.command=='prepare':
            data=runner.prepare(config,config['data']['prepared_dir'])
            print(json.dumps(data.manifest['counts'],indent=2))
            for warning in data.manifest['decoding_warnings']: print(warning,file=sys.stderr)
            return 0
        result=runner.run_experiment(config, explicit_seed=args.seed is not None)
        return 0 if result['status']=='completed' else 4
    except KeyboardInterrupt:
        print('Interrupted',file=sys.stderr); return 130
    except Exception as exc:
        from data_loader.schema import DataError
        code=3 if isinstance(exc,(DataError,OSError)) else 2
        print(f'{"data" if code==3 else "configuration"}: {exc}',file=sys.stderr)
        return code

if __name__=='__main__':
    raise SystemExit(main())
