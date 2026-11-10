"""Select configurations on train-only validation; optionally refit and test once."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from pipeline.config import load_config
from pipeline.reproducibility import setup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', required=True)
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--search', default='configs/search.json')
    parser.add_argument('--output-dir', default='artifacts/tuning')
    parser.add_argument('--model', choices=['all','item_knn','wmf','neumf','ctr'], default='all')
    parser.add_argument('--validation-seed', type=int, default=71)
    parser.add_argument('--resume', help='Resume an existing tuning directory with the same data, configuration, and search')
    parser.add_argument('--evaluate', action='store_true', help='After selection, refit on all outer training positives and evaluate test once')
    args = parser.parse_args(argv)
    config = load_config(args.config, {'data': {'data_dir': None, 'prepared_dir': str(Path(args.prepared_dir).resolve())}, 'experiment': {'models': args.model}})
    setup(config['split']['seed'], config['experiment']['threads'], config['experiment']['deterministic'])
    from data_loader.persistence import load_prepared, write_json
    from pipeline.tuning import select_configs
    data = load_prepared(args.prepared_dir, expected_seed=config['split']['seed'])
    search = json.loads(Path(args.search).read_text(encoding='utf-8-sig'))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    output = Path(args.resume).resolve() if args.resume else Path(args.output_dir).resolve() / run_id
    chosen, report = select_configs(data.training_data(), config, search, output, args.validation_seed, resume=bool(args.resume))
    chosen['experiment']['output_dir'] = str(output / 'test')
    write_json(output / 'selected_config.json', chosen)
    write_json(output / 'selection_provenance.json', {'dataset_id':data.manifest['dataset_id'], 'outer_split_id':data.manifest['split_id'], 'validation_split_id':report['validation_split_id'], 'selection_metric':'ndcg_20','test_consulted_during_selection':False})
    print(f'Selected configuration: {output / "selected_config.json"}', flush=True)
    if args.evaluate:
        from pipeline.runner import run_experiment
        result = run_experiment(chosen)
        write_json(output / 'test_result.json', {'comparison_path':str(Path(result['output_dir'])/'comparison.json'),'status':result['status']})
        return 0 if result['status'] == 'completed' else 4
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
