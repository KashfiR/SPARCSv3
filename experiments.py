"""Run the benchmark one dataset at a time, replicated across seeds.

Every (dataset, method, seed) result is written to disk the moment it finishes,
so an interrupted session never costs more than the single replicate that was
running, and adding seeds later only computes the new ones.

    python experiments.py --smoke                       # 2 minute sanity check
    python experiments.py --dataset s1
    python experiments.py --dataset s2
    python experiments.py --dataset liver    --liver    Liver_GSE14520_U133A.csv
    python experiments.py --dataset leukemia --leukemia Leukemia_GSE28497.csv
    python experiments.py --status                      # what is done so far

A seed controls the train/test split and every selector, on both the synthetic
and the real benchmarks, so all four datasets follow the same resampling
protocol. Holding the underlying data fixed is what makes selection stability
measurable: the same feature universe is on offer in every replicate, so the
overlap between one method's own selections across seeds is meaningful. Pass
--redraw-data to also redraw the synthetic datasets each replicate, which widens
the error bars to full Monte Carlo variability but leaves the synthetic feature
sets incomparable, so stability is then reported only for the real datasets.

Layout under --outdir:
    checkpoints/<dataset>__<method>__seed<seed>.json   one finished replicate
    datasets/<dataset>.json                            one aggregated dataset
    results.json                                       merged, read by figures.py
    selected_features.json                             every selected set
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from asgd import ASGDC, ASGDR, select_by_importance
from datasets import S1, S2, generate_synthetic_dataset, prepare_data
from evaluation import (aggregate_replicates, aggregate_similarity,
                        evaluate_classification, evaluate_regression,
                        feature_recovery, pairwise_similarity, selection_stability)
from isis_cpss import CPSS, ISIS, cpss_error_bound
from preprocessing import load_cumida, preprocess_data, split_and_normalize
from sparcs import SPARCS

MAX_FEATURES = 100
SEEDS = [42, 43, 44, 45, 46]

SPARCS_PARAMS = dict(M_prefilter=500, M_rdc=200, n_perm=30, alpha=0.05,
                     min_candidates=10, B_stability=30, stability_tau=0.6,
                     k_add=3, shrink_rate=0.9, min_pool=50, max_features=MAX_FEATURES)
ISIS_PARAMS = dict(max_features=MAX_FEATURES)
CPSS_PARAMS = dict(B=30, tau=0.6, max_features=MAX_FEATURES)

TASKS = {'s1': 'regression', 's2': 'regression',
         'liver': 'binary', 'leukemia': 'multiclass'}
ALL_DATASETS = list(TASKS)

_START = time.time()


def log(message):
    stamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{stamp} | {(time.time()-_START)/60:7.1f} min] {message}", flush=True)


def config_fingerprint(key, method, args):
    """Identify the settings a checkpoint was produced under.

    Restoring a replicate computed at a different problem size or with
    different method parameters would silently mix incompatible runs, so the
    fingerprint is stored alongside each result and checked on restore.
    """
    payload = {
        'method': method,
        'max_features': MAX_FEATURES,
        'search': args.search,
        'n_iter': args.n_iter,
        'params': {'SPARCS': SPARCS_PARAMS, 'Spearman-ISIS': ISIS_PARAMS,
                   'CPSS-Spearman': CPSS_PARAMS}.get(method, {}),
    }
    if key in ('s1', 's2'):
        payload['shape'] = [args.n_features, args.n_samples, args.redraw_data]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def method_names(task):
    return ['SPARCS', 'Spearman-ISIS', 'CPSS-Spearman',
            'ASGDR' if task == 'regression' else 'ASGDC']


# --------------------------------------------------------------------------
# data, redrawn per seed
# --------------------------------------------------------------------------

def build_synthetic(key, seed, n_features=10000, n_samples=500, redraw=False):
    """Replicate the split; optionally redraw the dataset itself."""
    spec = dict(S1 if key == 's1' else S2)
    if redraw:
        spec['random_state'] = seed
    X, y, meta = generate_synthetic_dataset(n_samples=n_samples, n_features=n_features, **spec)
    X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test, meta


def build_cumida(path, seed):
    """The data is fixed, so the replicate redraws the stratified split."""
    X, _, y_encoded, _, _ = preprocess_data(load_cumida(path), verbose=False)
    X_train, X_test, y_train, y_test, _ = split_and_normalize(
        X, y_encoded, test_size=0.3, random_state=seed, verbose=False)
    return X_train, X_test, y_train, y_test


def load_dataset(key, seed, args):
    if key in ('s1', 's2'):
        X_train, X_test, y_train, y_test, meta = build_synthetic(
            key, seed, args.n_features, args.n_samples, args.redraw_data)
        return X_train, X_test, y_train, y_test, meta['true_feature_names'], meta

    path = getattr(args, key)
    if not path:
        raise SystemExit(f"--{key} is required to run the {key} dataset "
                         f"(path to the CuMiDa csv)")
    X_train, X_test, y_train, y_test = build_cumida(path, seed)
    return X_train, X_test, y_train, y_test, None, None


# --------------------------------------------------------------------------
# one replicate
# --------------------------------------------------------------------------

def run_method(method, X_train, y_train, task, seed, n_jobs, verbose):
    start = time.perf_counter()
    extra = {}

    if method == 'SPARCS':
        selected = SPARCS(X_train, y_train, task=task, random_state=seed,
                          n_jobs=n_jobs, verbose=verbose, **SPARCS_PARAMS)
        extra['history'] = getattr(SPARCS, 'last_history_', [])

    elif method == 'Spearman-ISIS':
        selected = ISIS(X_train, y_train, task=task, random_state=seed,
                        verbose=verbose, **ISIS_PARAMS)

    elif method == 'CPSS-Spearman':
        selected = CPSS(X_train, y_train, task=task, random_state=seed,
                        n_jobs=n_jobs, verbose=verbose, **CPSS_PARAMS)
        q_hat = getattr(CPSS, 'last_qhat_', 0.0)
        extra['q_hat'] = q_hat
        extra['expected_false_positives_bound'] = cpss_error_bound(
            q_hat, CPSS_PARAMS['tau'], X_train.shape[1])

    else:
        model = (ASGDR(random_state=seed) if task == 'regression'
                 else ASGDC(random_state=seed))
        model.fit(X_train, y_train, feature_names=list(X_train.columns))
        selected = select_by_importance(model, MAX_FEATURES)

    return selected, (time.perf_counter() - start) / 60.0, extra


def evaluate_method(selected, task, X_train, X_test, y_train, y_test,
                    true_features, seed, search, n_iter):
    entry = {'n_selected': len(selected),
             'sparsity_ratio': len(selected) / float(X_train.shape[1])}
    kwargs = {'search': search, 'random_state': seed}
    if n_iter:
        kwargs['n_iter'] = n_iter

    if not selected:
        entry['note'] = 'no features selected, downstream model skipped'
    elif task == 'regression':
        entry['prediction'] = evaluate_regression(X_train, X_test, y_train, y_test,
                                                  selected, **kwargs)
    else:
        entry['prediction'] = evaluate_classification(X_train, X_test, y_train, y_test,
                                                      selected, task=task, **kwargs)

    if true_features is not None:
        entry['recovery'] = feature_recovery(selected, true_features, X_train.shape[1])
    return entry


def headline(flat, task):
    if task == 'regression':
        return flat.get('test_r2')
    if task == 'binary':
        return flat.get('test_auc')
    return flat.get('test_f1_macro')


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_dataset(key, args, dirs, seeds):
    task = TASKS[key]
    methods = method_names(task)

    # replicate -> checkpoint, keyed by (method, seed)
    store, selections = {}, {}

    for seed in seeds:
        def outstanding(m):
            path = dirs['checkpoints'] / f"{key}__{m}__seed{seed}.json"
            if args.fresh or not path.exists():
                return True
            return json.loads(path.read_text()).get('fingerprint') != \
                config_fingerprint(key, m, args)

        missing = [m for m in methods if outstanding(m)]

        data = None
        if missing:
            log(f"{key} seed {seed}: loading data "
                f"({len(missing)} of {len(methods)} methods to run)")
            data = load_dataset(key, seed, args)
            X_train, X_test, y_train, y_test, true_features, meta = data
            log(f"{key} seed {seed}: task={task} n_train={X_train.shape[0]} "
                f"n_test={X_test.shape[0]} p={X_train.shape[1]}")

        for method in methods:
            checkpoint = dirs['checkpoints'] / f"{key}__{method}__seed{seed}.json"

            fingerprint = config_fingerprint(key, method, args)

            if checkpoint.exists() and not args.fresh:
                saved = json.loads(checkpoint.read_text())
                if saved.get('fingerprint') == fingerprint:
                    store.setdefault(method, []).append(saved['entry'])
                    selections.setdefault(method, []).append(saved['selected'])
                    log(f"{key}/{method}/seed{seed}: restored "
                        f"(|S|={len(saved['selected'])}, "
                        f"{saved['entry']['runtime_min']:.2f} min)")
                    continue
                log(f"{key}/{method}/seed{seed}: checkpoint was made under different "
                    f"settings, recomputing")
                if data is None:
                    data = load_dataset(key, seed, args)

            X_train, X_test, y_train, y_test, true_features, meta = data
            log(f"{key}/{method}/seed{seed}: selecting")
            selected, minutes, extra = run_method(method, X_train, y_train, task,
                                                  seed, args.n_jobs, args.verbose)
            log(f"{key}/{method}/seed{seed}: {len(selected)} features in "
                f"{minutes:.2f} min, evaluating")

            entry = evaluate_method(selected, task, X_train, X_test, y_train, y_test,
                                    true_features, seed, args.search, args.n_iter)
            entry['runtime_min'] = minutes
            entry['seed'] = seed
            entry.update(extra)

            checkpoint.write_text(json.dumps(
                {'seed': seed, 'fingerprint': fingerprint,
                 'selected': selected, 'entry': entry}, indent=2, default=str))
            store.setdefault(method, []).append(entry)
            selections.setdefault(method, []).append(selected)

            score = headline({**(entry.get('prediction') or {})}, task)
            line = f"{key}/{method}/seed{seed}: done, |S|={len(selected)}"
            if score is not None:
                line += f", score={score:.4f}"
            if 'recovery' in entry:
                line += (f", TPR={entry['recovery']['tpr']:.3f}"
                         f", FDR={entry['recovery']['fdr']:.3f}")
            log(line)

    return assemble_dataset(key, task, methods, store, selections, seeds, dirs, args)


def assemble_dataset(key, task, methods, store, selections, seeds, dirs, args):
    """Aggregate replicates into means, spreads, CIs and cross-seed stability."""
    # A method's selections are only comparable across replicates when the same
    # feature universe was on offer each time.
    comparable = not (args.redraw_data and key in ('s1', 's2'))

    entries = {}
    for method in methods:
        replicates = store.get(method, [])
        if not replicates:
            continue
        entries[method] = {
            'n_replicates': len(replicates),
            'seeds': [r.get('seed') for r in replicates],
            'aggregate': aggregate_replicates(replicates, args.confidence),
            'selection_stability': (selection_stability(selections.get(method, []))
                                    if comparable else None),
            'replicates': replicates,
        }

    # method-pair overlap, computed within each seed then aggregated
    per_seed = []
    for index, _ in enumerate(seeds):
        snapshot = {m: selections[m][index] for m in methods
                    if m in selections and index < len(selections[m])}
        if len(snapshot) > 1:
            per_seed.append(pairwise_similarity(snapshot))

    shape_path = dirs['datasets'] / f"{key}.json"
    previous = json.loads(shape_path.read_text()) if shape_path.exists() else {}

    record = {
        'task': task,
        'data_redrawn_per_seed': not comparable,
        'n_features': previous.get('n_features'),
        'n_train': previous.get('n_train'),
        'n_test': previous.get('n_test'),
        'methods': entries,
        'similarity': aggregate_similarity(per_seed, args.confidence) if per_seed else {},
    }
    first = next(iter(entries.values()), None)
    if first and first['aggregate'].get('sparsity_ratio'):
        ratio = first['aggregate']['sparsity_ratio']['mean']
        size = first['aggregate']['n_selected']['mean']
        if ratio:
            record['n_features'] = int(round(size / ratio))

    shape_path.write_text(json.dumps(record, indent=2, default=str))
    return record


def merge(dirs, args, seeds):
    datasets, selections = {}, {}

    for key in ALL_DATASETS:
        path = dirs['datasets'] / f"{key}.json"
        if path.exists():
            datasets[key] = json.loads(path.read_text())

        picked = {}
        for method in method_names(TASKS[key]):
            by_seed = {}
            for seed in seeds:
                checkpoint = dirs['checkpoints'] / f"{key}__{method}__seed{seed}.json"
                if checkpoint.exists():
                    by_seed[str(seed)] = json.loads(checkpoint.read_text())['selected']
            if by_seed:
                picked[method] = by_seed
        if picked:
            selections[key] = picked

    payload = {
        'config': {
            'seeds': seeds,
            'confidence': args.confidence,
            'redraw_data': args.redraw_data,
            'max_features': MAX_FEATURES,
            'downstream_search': args.search,
            'n_iter': args.n_iter,
            'sparcs': SPARCS_PARAMS,
            'isis': ISIS_PARAMS,
            'cpss': CPSS_PARAMS,
            'synthetic_spec': {'s1': S1, 's2': S2},
            'generated': datetime.now().isoformat(timespec='seconds'),
        },
        'datasets': datasets,
    }
    (dirs['root'] / 'results.json').write_text(json.dumps(payload, indent=2, default=str))
    (dirs['root'] / 'selected_features.json').write_text(json.dumps(selections, indent=2))
    return datasets


def status(dirs, seeds):
    rows = []
    for key in ALL_DATASETS:
        for method in method_names(TASKS[key]):
            done = []
            for seed in seeds:
                if (dirs['checkpoints'] / f"{key}__{method}__seed{seed}.json").exists():
                    done.append(seed)
            rows.append((key, method, f"{len(done)}/{len(seeds)}",
                         ','.join(str(s) for s in done) or '-'))

    table = pd.DataFrame(rows, columns=['dataset', 'method', 'replicates', 'seeds done'])
    complete = sum(int(r[2].split('/')[0]) for r in rows)
    try:
        print(table.to_string(index=False))
        print(f"\n{complete} of {len(rows) * len(seeds)} replicates complete")
    except BrokenPipeError:  # piping into head closes the stream early
        pass


def smoke(args):
    global MAX_FEATURES
    MAX_FEATURES = 12
    SPARCS_PARAMS.update(M_prefilter=120, M_rdc=40, B_stability=4,
                         min_pool=20, max_features=12)
    ISIS_PARAMS.update(max_features=12)
    CPSS_PARAMS.update(B=3, max_features=12)
    seeds = [1, 2]

    log("smoke: synthetic regression, 2 seeds")
    store, sels = {}, {}
    for seed in seeds:
        X_train, X_test, y_train, y_test, meta = build_synthetic(
            's1', seed, n_features=400, n_samples=150, redraw=False)
        for method in method_names('regression'):
            selected, minutes, _ = run_method(method, X_train, y_train, 'regression',
                                              seed, args.n_jobs, False)
            entry = evaluate_method(selected, 'regression', X_train, X_test, y_train,
                                    y_test, meta['true_feature_names'], seed, 'random', 5)
            entry['runtime_min'] = minutes
            store.setdefault(method, []).append(entry)
            sels.setdefault(method, []).append(selected)

    for method, replicates in store.items():
        agg = aggregate_replicates(replicates)
        r2 = agg.get('test_r2')
        log(f"  {method}: R2 {r2['mean']:+.3f} +/- {r2['std']:.3f}, "
            f"cross-seed stability {selection_stability(sels[method]):.3f}")

    log("smoke: classification paths")
    X, y, _ = generate_synthetic_dataset(n_samples=150, n_features=400, **S1)
    for task, labels in (('binary', (y > np.median(y)).astype(int)),
                         ('multiclass', pd.qcut(y, 3, labels=False))):
        X_train, X_test, y_train, y_test = prepare_data(X, np.asarray(labels), test_size=0.3)
        for method in method_names(task):
            selected, _, _ = run_method(method, X_train, y_train, task, 1, args.n_jobs, False)
            entry = evaluate_method(selected, task, X_train, X_test, y_train, y_test,
                                    None, 1, 'random', 5)
            log(f"  {task}/{method}: |S|={len(selected)} "
                f"acc={entry['prediction']['test_accuracy']:.3f}")

    log("smoke passed, the install and every code path work")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=ALL_DATASETS + ['all'],
                    help='which benchmark to run in this session')
    ap.add_argument('--liver', help='path to Liver_GSE14520_U133A.csv')
    ap.add_argument('--leukemia', help='path to Leukemia_GSE28497.csv')
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--seeds', type=int, default=5,
                    help='number of replicates, taken from the front of the seed list')
    ap.add_argument('--seed-list', type=int, nargs='+',
                    help='explicit seeds, overrides --seeds')
    ap.add_argument('--confidence', type=float, default=0.95,
                    help='confidence level for the reported interval')
    ap.add_argument('--search', choices=['none', 'random', 'grid'], default='random')
    ap.add_argument('--n-iter', type=int, default=60,
                    help='configurations drawn when --search random')
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--n-features', type=int, default=10000)
    ap.add_argument('--n-samples', type=int, default=500)
    ap.add_argument('--redraw-data', action='store_true',
                    help='redraw the synthetic datasets each seed instead of only the split')
    ap.add_argument('--fresh', action='store_true', help='ignore checkpoints and recompute')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    seeds = args.seed_list if args.seed_list else SEEDS[:args.seeds]
    if not args.seed_list and args.seeds > len(SEEDS):
        seeds = SEEDS + list(range(SEEDS[-1] + 1, SEEDS[-1] + 1 + args.seeds - len(SEEDS)))

    root = Path(args.outdir)
    dirs = {'root': root, 'checkpoints': root / 'checkpoints', 'datasets': root / 'datasets'}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    if args.status:
        status(dirs, seeds)
        return

    if args.smoke:
        smoke(args)
        return

    if not args.dataset:
        ap.error("pass --dataset, or --status, or --smoke")

    log(f"seeds: {seeds}")
    for key in (ALL_DATASETS if args.dataset == 'all' else [args.dataset]):
        run_dataset(key, args, dirs, seeds)

    datasets = merge(dirs, args, seeds)
    log(f"merged {len(datasets)} dataset(s) into {root/'results.json'}")
    log("next: python figures.py")


if __name__ == '__main__':
    main()
