"""Baseline selectors: Spearman-ISIS and Complementary Pairs Stability Selection."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from utils import (coefficient_magnitude, compute_residuals, make_base_model,
                   sis_size, spearman_corr)


def ISIS(X, y, task='regression', max_features=100, random_state=42, verbose=False):
    """Iterative Sure Independence Screening on Spearman rank correlations.

    Each round ranks the remaining features by their rank correlation with the
    current residual, fits an Adaptive Elastic Net on the selected set plus the
    top d candidates, and greedily commits the single candidate with the largest
    coefficient magnitude.
    """
    if task not in ('regression', 'binary', 'multiclass'):
        raise ValueError("task must be 'regression', 'binary' or 'multiclass'")

    X = X.copy()
    X.columns = X.columns.astype(str)
    names = list(X.columns)
    y = np.asarray(y).ravel()
    n, p = X.shape

    d = sis_size(n)
    max_features = min(max_features, p)
    base = make_base_model(task, random_state=random_state)

    if verbose:
        print(f"ISIS | task={task} n={n} p={p} d={d} max_features={max_features}")

    selected = []
    resid = y - np.mean(y) if task == 'regression' else y - np.mean(y)

    for t in range(1, max_features + 1):
        if len(selected) >= max_features:
            break

        remaining = [c for c in names if c not in set(selected)]
        if not remaining:
            if verbose:
                print("stop: all features considered")
            break

        corrs = np.abs(spearman_corr(X[remaining], resid))
        k = min(d, len(remaining))
        candidates = [remaining[i] for i in np.argsort(corrs)[-k:][::-1]]

        cols = selected + candidates
        model = clone(base)
        model.fit(X[cols], y, feature_names=cols)
        magnitude = coefficient_magnitude(model)[len(selected):]

        if magnitude.size == 0 or np.max(magnitude) < 1e-6:
            if verbose:
                print("stop: no candidate shows a conditional association")
            break

        best = candidates[int(np.argmax(magnitude))]
        selected.append(best)

        refit = clone(base)
        refit.fit(X[selected], y, feature_names=selected)
        resid = compute_residuals(refit, X[selected], y, task)

        if verbose:
            print(f"iter {t}: +{best} |beta|={np.max(magnitude):.4f} "
                  f"resid_sd={np.std(resid):.4f}")

    if verbose:
        print(f"ISIS done | {len(selected)} features selected")
    return selected


def CPSS(X, y, task='regression', base_selector=None, B=30, tau=0.6,
         max_features=100, random_state=42, n_jobs=-1, verbose=False):
    """Complementary Pairs Stability Selection wrapped around Spearman-ISIS.

    Each of the B pairs splits the sample into two disjoint halves, runs the base
    selector on each, and credits a feature if either half selects it. Features
    whose selection frequency reaches tau are returned.
    """
    X = X.copy()
    X.columns = X.columns.astype(str)
    names = list(X.columns)
    y = np.asarray(y).ravel()
    n = X.shape[0]
    half = n // 2

    if base_selector is None:
        def base_selector(X_sub, y_sub, seed):
            return ISIS(X_sub, y_sub, task=task, max_features=max_features,
                        random_state=seed, verbose=False)

    if verbose:
        print(f"CPSS | B={B} pairs tau={tau} n={n} p={X.shape[1]}")

    def process_pair(b):
        rng = np.random.RandomState(random_state + b)
        order = rng.permutation(n)
        hit = set()
        for idx in (order[:half], order[half:2 * half]):
            hit.update(base_selector(X.iloc[idx], y[idx], random_state + b))
        return hit

    pairs = Parallel(n_jobs=n_jobs)(delayed(process_pair)(b) for b in range(B))

    counts = pd.Series(0.0, index=names)
    for hit in pairs:
        counts[list(hit)] += 1.0
    freq = counts / float(B)

    selected = freq.index[freq >= tau].tolist()
    if verbose:
        print(f"CPSS done | frequency max={freq.max():.2f}, {len(selected)} features "
              f"at or above tau={tau}")

    CPSS.last_frequencies_ = freq
    CPSS.last_qhat_ = float(np.mean([len(h) for h in pairs])) if pairs else 0.0
    return selected


def cpss_error_bound(q_hat, tau, p):
    """Shah and Samworth expected false positive bound, valid for tau > 0.5."""
    if tau <= 0.5:
        return float('inf')
    return (q_hat ** 2) / ((2.0 * tau - 1.0) * p)
