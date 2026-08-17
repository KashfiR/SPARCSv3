"""Two-stage evaluation: feature selection, then a tuned LightGBM predictor.

Every method is judged by the same downstream model, so differences reflect the
quality of the selected feature set rather than the predictor.
"""

import numpy as np
from itertools import combinations
from scipy import stats

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

# Search space quoted in the paper. Exhaustive enumeration is about 1.5 million
# configurations, so the default protocol is a randomized draw of N_ITER points
# under 5-fold cross validation.
PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10, -1],
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 1],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
    'reg_lambda': [0, 0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
}
N_ITER = 60
N_SPLITS = 5


def _tuned(estimator, X, y, scoring, stratified, search, param_grid, n_iter, random_state):
    if search is None:
        estimator.fit(X, y)
        return estimator, None

    grid = PARAM_GRID if param_grid is None else param_grid
    cv = (StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
          if stratified else KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state))

    if search == 'grid':
        fitted = GridSearchCV(estimator, grid, scoring=scoring, cv=cv, n_jobs=-1)
    else:
        fitted = RandomizedSearchCV(estimator, grid, n_iter=n_iter, scoring=scoring,
                                    cv=cv, n_jobs=-1, random_state=random_state)
    fitted.fit(X, y)
    return fitted.best_estimator_, fitted.best_params_


def evaluate_regression(X_train, X_test, y_train, y_test, features,
                        search='random', param_grid=None, n_iter=N_ITER, random_state=42):
    features = list(features)
    model, best = _tuned(LGBMRegressor(random_state=random_state, verbose=-1),
                         X_train[features], y_train, 'r2', False,
                         search, param_grid, n_iter, random_state)

    pred = model.predict(X_test[features])
    train_pred = model.predict(X_train[features])
    return {
        'test_r2': float(r2_score(y_test, pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
        'test_mae': float(mean_absolute_error(y_test, pred)),
        'train_r2': float(r2_score(y_train, train_pred)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
        'best_params': best,
    }


def evaluate_classification(X_train, X_test, y_train, y_test, features, task='binary',
                            search='random', param_grid=None, n_iter=N_ITER, random_state=42):
    features = list(features)
    scoring = 'roc_auc' if task == 'binary' else 'f1_macro'
    model, best = _tuned(LGBMClassifier(random_state=random_state, verbose=-1),
                         X_train[features], y_train, scoring, True,
                         search, param_grid, n_iter, random_state)

    pred = model.predict(X_test[features])
    out = {
        'test_accuracy': float(accuracy_score(y_test, pred)),
        'train_accuracy': float(accuracy_score(y_train, model.predict(X_train[features]))),
        'best_params': best,
    }

    if task == 'binary':
        proba = model.predict_proba(X_test[features])[:, 1]
        out.update({
            'test_f1': float(f1_score(y_test, pred)),
            'test_precision': float(precision_score(y_test, pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, pred, zero_division=0)),
            'test_auc': float(roc_auc_score(y_test, proba)),
        })
    else:
        out.update({
            'test_f1_macro': float(f1_score(y_test, pred, average='macro')),
            'test_f1_weighted': float(f1_score(y_test, pred, average='weighted')),
        })
    return out


def feature_recovery(selected, true_features, n_features=None):
    """TPR, FDR, precision and F1 against a known ground truth set."""
    selected, true = set(selected), set(true_features)
    tp = len(selected & true)

    precision = tp / len(selected) if selected else 0.0
    tpr = tp / len(true) if true else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0

    out = {
        'n_selected': len(selected),
        'n_true': len(true),
        'tp': tp,
        'fp': len(selected - true),
        'fn': len(true - selected),
        'precision': precision,
        'tpr': tpr,
        'fdr': 1.0 - precision,
        'f1': f1,
        'jaccard': tp / len(selected | true) if (selected | true) else 0.0,
    }
    if n_features:
        out['sparsity_ratio'] = len(selected) / float(n_features)
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def selection_stability(selections_by_seed):
    """Mean pairwise Jaccard between one method's selections across seeds.

    This is the direct measurement of selection stability: how much a method's
    chosen feature set moves when only the seed changes. 1.0 means the same set
    every time, 0.0 means no feature is ever chosen twice.
    """
    sets = list(selections_by_seed)
    if len(sets) < 2:
        return None
    scores = [jaccard(a, b) for a, b in combinations(sets, 2)]
    return float(np.mean(scores))


def summarize(values, confidence=0.95):
    """Mean, spread, and a t-based confidence interval over replicate values."""
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    n = len(clean)
    if n == 0:
        return None

    mean = float(np.mean(clean))
    out = {'n': n, 'mean': mean, 'values': clean,
           'std': 0.0, 'sem': 0.0, 'ci_low': mean, 'ci_high': mean}

    if n > 1:
        std = float(np.std(clean, ddof=1))
        sem = std / np.sqrt(n)
        margin = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sem
        out.update({'std': std, 'sem': sem,
                    'ci_low': mean - margin, 'ci_high': mean + margin})
    return out


# Metrics lifted out of each replicate's nested prediction/recovery blocks so
# that aggregation and plotting see one flat namespace.
FLAT_KEYS = ['n_selected', 'sparsity_ratio', 'runtime_min',
             'test_r2', 'test_rmse', 'test_mae',
             'test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_auc',
             'test_f1_macro', 'test_f1_weighted',
             'tpr', 'fdr', 'precision', 'f1', 'jaccard', 'tp', 'fp', 'fn']


def flatten_replicate(replicate):
    flat = {}
    for key in ('prediction', 'recovery'):
        flat.update({k: v for k, v in (replicate.get(key) or {}).items()
                     if isinstance(v, (int, float))})
    flat.update({k: v for k, v in replicate.items() if isinstance(v, (int, float))})
    return flat


def aggregate_replicates(replicates, confidence=0.95):
    """Collapse a list of per-seed replicates into mean/std/CI per metric."""
    flats = [flatten_replicate(r) for r in replicates]
    keys = [k for k in FLAT_KEYS if any(k in f for f in flats)]
    return {k: summarize([f.get(k) for f in flats], confidence) for k in keys}


def aggregate_similarity(per_seed, confidence=0.95):
    """Aggregate the pairwise-overlap dictionaries produced for each seed."""
    out = {}
    for kind in ('jaccard', 'overlap', 'intersection'):
        pairs = {}
        for seed_result in per_seed:
            for pair, value in seed_result.get(kind, {}).items():
                pairs.setdefault(pair, []).append(value)
        out[kind] = {pair: summarize(values, confidence)
                     for pair, values in pairs.items()}
    return out


def overlap_coefficient(a, b):
    a, b = set(a), set(b)
    smaller = min(len(a), len(b))
    return len(a & b) / smaller if smaller else 0.0


def pairwise_similarity(selections):
    out = {'jaccard': {}, 'overlap': {}, 'intersection': {}}
    for m1, m2 in combinations(selections, 2):
        key = f"{m1}|{m2}"
        out['jaccard'][key] = jaccard(selections[m1], selections[m2])
        out['overlap'][key] = overlap_coefficient(selections[m1], selections[m2])
        out['intersection'][key] = len(set(selections[m1]) & set(selections[m2]))
    return out
