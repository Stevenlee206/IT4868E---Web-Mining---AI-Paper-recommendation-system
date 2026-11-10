"""Train-only validation and reproducible model-configuration selection."""
from copy import deepcopy
from pathlib import Path
import gc
import json
import time

import numpy as np
from scipy.sparse import csr_matrix

from data_loader.schema import TrainingData, EvaluationData
from data_loader.preprocessing import split_identity
from data_loader.persistence import write_json
from evaluation.evaluator import evaluate
from pipeline.config import NAMES, validate
from pipeline.runner import fit_model


def validation_split(training, seed=71):
    """Hold out 20% of outer-training positives; never accept outer-test data."""
    if not isinstance(training, TrainingData):
        raise TypeError('validation_split requires TrainingData, without test labels')
    source = training.train
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    train_ptr, val_ptr = [0], [0]
    for user in range(source.shape[0]):
        items = source.indices[source.indptr[user]:source.indptr[user + 1]]
        if len(items) >= 2:
            size = min(len(items) - 1, max(1, int(np.ceil(.2 * len(items)))))
            held = np.sort(rng.choice(items, size=size, replace=False))
            kept = np.setdiff1d(items, held, assume_unique=True)
        else:
            kept, held = items, np.empty(0, dtype=np.int64)
        train_indices.extend(kept)
        val_indices.extend(held)
        train_ptr.append(len(train_indices))
        val_ptr.append(len(val_indices))
    def matrix(indices, ptr):
        return csr_matrix((np.ones(len(indices), dtype=np.float32), indices, ptr), shape=source.shape)
    train, val = matrix(train_indices, train_ptr), matrix(val_indices, val_ptr)
    eligible = np.flatnonzero((train.getnnz(axis=1) > 0) & (val.getnnz(axis=1) > 0))
    if len(eligible) == 0:
        raise ValueError('No eligible inner-validation users')
    fit_data = TrainingData(train, training.bow, training.num_users, training.num_items, training.dataset_id)
    eval_data = EvaluationData(train, val, eligible, training.num_items, split_identity(train, val, seed))
    return fit_data, eval_data


def candidate_configs(base, name, overrides):
    """Baseline is always first; validate the complete search before fitting."""
    if not isinstance(overrides, list) or any(not isinstance(value, dict) for value in overrides):
        raise ValueError(f'{name}: search candidates must be a list of objects')
    configs = [deepcopy(base)]
    seen = [base[name]]
    for changes in overrides:
        unknown = set(changes) - set(base[name])
        if unknown:
            raise ValueError(f'{name}: unknown search settings: {sorted(unknown)}')
        candidate = deepcopy(base)
        candidate[name].update(changes)
        validate(candidate)
        if candidate[name] not in seen:
            configs.append(candidate)
            seen.append(deepcopy(candidate[name]))
    return configs


def select_configs(training, base, search, output_dir, validation_seed=71, resume=False):
    """Selection uses only inner validation. Test evaluation is a separate step."""
    if not isinstance(search, dict) or set(search) - set(NAMES):
        raise ValueError('Search must map supported model names to candidate lists')
    names = list(NAMES) if base['experiment']['models'] == 'all' else [base['experiment']['models']]
    candidates = {name: candidate_configs(base, name, search.get(name, [])) for name in names}
    fit_data, eval_data = validation_split(training, validation_seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=resume)
    report = {
        'schema_version': 1,
        'status': 'running',
        'selection_metric': 'ndcg_20',
        'tie_policy': 'first candidate wins; baseline is first',
        'validation_seed': validation_seed,
        'validation_split_id': eval_data.split_id,
        'dataset_id': training.dataset_id,
        'inner_train_pairs': fit_data.train.nnz,
        'validation_pairs': eval_data.test.nnz,
        'validation_users': len(eval_data.eligible_user_ids),
        'candidate_protocol': 'full_catalog_minus_inner_training',
        'outer_test_used_for_selection': False,
        'trials': [],
        'selected': {},
    }
    completed = {}
    if resume:
        previous = json.loads((output / 'tuning.json').read_text(encoding='utf-8'))
        previous_base = json.loads((output / 'base_config.json').read_text(encoding='utf-8'))
        previous_search = json.loads((output / 'search.json').read_text(encoding='utf-8'))
        if previous_base != base or previous_search != search or previous['validation_split_id'] != eval_data.split_id or previous['dataset_id'] != training.dataset_id:
            raise ValueError('Resume requires identical base configuration, search, data, and validation split')
        for trial in previous['trials']:
            if trial['status'] == 'completed':
                completed[(trial['model'], trial['candidate'])] = trial
        report['resumed_completed_trials'] = len(completed)
    write_json(output / 'search.json', search)
    write_json(output / 'base_config.json', base)
    selected = deepcopy(base)
    for name in names:
        best = None
        for index, config in enumerate(candidates[name]):
            cached = completed.get((name, index))
            if cached is not None:
                if cached['settings'] != config[name]:
                    raise ValueError('Cached candidate settings do not match')
                report['trials'].append(cached)
                if best is None or cached['metrics']['ndcg_20'] > best['metrics']['ndcg_20']:
                    best = deepcopy(cached)
                print(f'{name}: reusing completed validation trial {index + 1}', flush=True)
                continue
            print(f'{name}: validation trial {index + 1}/{len(candidates[name])}', flush=True)
            row = {'model': name, 'candidate': index, 'settings': config[name], 'status': 'training'}
            report['trials'].append(row)
            write_json(output / 'tuning.json', report)
            started = time.perf_counter()
            model = result = None
            try:
                model, summary = fit_model(name, fit_data, config)
                result = evaluate(model, eval_data, config['evaluation'])
                row.update(status='completed', metrics=result.metrics, train_seconds=summary.elapsed_time)
                if best is None or row['metrics']['ndcg_20'] > best['metrics']['ndcg_20']:
                    best = deepcopy(row)
                print(f"{name}: validation NDCG@20={result.metrics['ndcg_20']:.6f}, Recall@50={result.metrics['recall_50']:.6f}", flush=True)
            except KeyboardInterrupt:
                row['status'] = 'interrupted'
                report['status'] = 'interrupted'
                write_json(output / 'tuning.json', report)
                raise
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                print(f"{name}: trial failed: {row['error']}", flush=True)
            finally:
                row['duration_seconds'] = time.perf_counter() - started
                del model, result
                gc.collect()
            write_json(output / 'tuning.json', report)
        if best is None:
            report['status'] = 'failed'
            write_json(output / 'tuning.json', report)
            raise RuntimeError(f'All validation candidates failed for {name}')
        selected[name] = deepcopy(best['settings'])
        report['selected'][name] = {'candidate': best['candidate'], 'settings': best['settings'], 'validation_metrics': best['metrics']}
        write_json(output / 'tuning.json', report)
    report['status'] = 'completed'
    write_json(output / 'tuning.json', report)
    return selected, report
