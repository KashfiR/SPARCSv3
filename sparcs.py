"""SPARCS: Stable Permutative Adaptive Rank-Based Correlation Screening.

Nine-step procedure:

  1. rank-based SIS warm start of size d = ceil(n / log n)
  2. iterative residual refinement against an Adaptive Elastic Net working model
  3. staged RDC screening of the remaining features
  4. permutation calibration of the surviving candidates
  5. batch complementary pairs stability selection
  6. adaptive stability threshold by elbow detection
  7. multi-add of the top k_add stable candidates
  8. geometric shrinkage of the candidate pool cap
  9. stopping on exhausted candidates, feature budget, or empty pool
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from utils import (coefficient_magnitude, compute_residuals, elbow_threshold,
                   make_base_model, rdc, sis_size, spearman_corr)

RDC_THRESHOLD_DEFAULT = {'regression': 0.2, 'binary': 0.1, 'multiclass': 0.1}
RDC_K, RDC_S = 20, 1.0  # projection count and bandwidth for the random feature maps


def hybrid_filter(pvals, rdc_vals, alpha=0.05, min_keep=10):
    """Keep every calibrated candidate, plus the strongest RDC scores as a floor.

    The floor guarantees the iteration has something to score even when the
    permutation null is not separable at the chosen alpha.
    """
    pvals = np.asarray(pvals)
    keep = pvals <= alpha
    if min_keep > 0:
        top = np.argsort(rdc_vals)[-min(min_keep, len(rdc_vals)):]
        keep[top] = True
    return np.nonzero(keep)[0].tolist()


def staged_rdc_screening(X_rem, resid, y, task, M_pre=500, M_rdc=200, n_perm=30,
                         alpha=0.05, min_candidates=10, iter_num=1,
                         n_jobs=-1, random_state=42, verbose=False):
    """Spearman prefilter, then RDC ranking, then permutation calibration."""
    names = list(X_rem.columns)
    seed = random_state + 1009 * iter_num

    M_pre = min(M_pre, len(names))
    pre_idx = np.argsort(np.abs(spearman_corr(X_rem, resid)))[-M_pre:][::-1]
    pre_names = [names[i] for i in pre_idx]
    X_pre = X_rem[pre_names].to_numpy(dtype=float)
    if verbose:
        print(f"    stage 1 (Spearman): {len(pre_names)} kept")

    rdcs = np.array(Parallel(n_jobs=n_jobs)(
        delayed(rdc)(X_pre[:, i], resid, RDC_K, RDC_S, np.random.RandomState(seed + i))
        for i in range(X_pre.shape[1])))

    M_rdc = min(M_rdc, len(pre_names))
    top_idx = np.argsort(rdcs)[-M_rdc:][::-1]
    cand_names = [pre_names[i] for i in top_idx]
    cand_rdc = rdcs[top_idx]
    X_cand = X_pre[:, top_idx]
    if verbose:
        print(f"    stage 2 (RDC): {len(cand_names)} kept, max={cand_rdc.max():.3f}")

    pvals = np.ones(len(cand_names))
    if n_perm > 0:
        strata = None if task == 'regression' else np.asarray(y).ravel()

        def permuted(pi):
            r = np.random.RandomState(seed + 7919 * (pi + 1))
            if strata is None:
                shuffled = r.permutation(resid)
            else:
                shuffled = np.array(resid, dtype=float, copy=True)
                for c in np.unique(strata):
                    m = strata == c
                    shuffled[m] = r.permutation(shuffled[m])
            return np.array([rdc(X_cand[:, j], shuffled, RDC_K, RDC_S,
                                 np.random.RandomState(seed + j))
                             for j in range(X_cand.shape[1])])

        null = np.vstack(Parallel(n_jobs=n_jobs)(
            delayed(permuted)(pi) for pi in range(n_perm)))
        pvals = (1.0 + np.sum(null >= cand_rdc, axis=0)) / (1.0 + n_perm)
        if verbose:
            print(f"    stage 3 (permutation): p in [{pvals.min():.3f}, {pvals.max():.3f}], "
                  f"{int(np.sum(pvals <= alpha))} significant")

    keep = hybrid_filter(pvals, cand_rdc, alpha=alpha, min_keep=min_candidates)
    return ([cand_names[i] for i in keep], cand_rdc[keep], pvals[keep])


def batch_cpss(X, y, selected, candidates, base_model, B=30, random_state=42, verbose=False):
    """Complementary pairs stability selection, one fit per half over all candidates.

    B pairs cost 2B model fits regardless of how many candidates are in play,
    which is what makes stability affordable inside the screening loop.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y).ravel()
    n = X.shape[0]
    half = n // 2
    cols = list(selected) + list(candidates)
    offset = len(selected)

    rng = np.random.RandomState(random_state)
    counts = np.zeros(len(candidates))
    fits = 0

    for b in range(B):
        order = rng.permutation(n)
        for idx in (order[:half], order[half:2 * half]):
            model = clone(base_model)
            model.fit(X.iloc[idx][cols], y[idx], feature_names=cols)
            magnitude = coefficient_magnitude(model)
            fits += 1
            if magnitude.size >= len(cols):
                counts += (magnitude[offset:offset + len(candidates)] > 1e-8).astype(float)

    stability = counts / max(fits, 1)
    if verbose:
        print(f"    stage 4 (batch CPSS): {fits} fits, max stability={stability.max():.2f}")
    return stability


def SPARCS(X, y, task='regression',
           initial_screening_size=None,
           M_prefilter=500, M_rdc=200, n_perm=30,
           alpha=0.05, min_candidates=10,
           B_stability=30, stability_tau=0.6,
           k_add=3, rdc_threshold=None,
           shrink_rate=0.9, min_pool=50,
           max_features=100, random_state=42, n_jobs=-1, verbose=False):
    """Return the list of selected feature names."""
    if task not in ('regression', 'binary', 'multiclass'):
        raise ValueError(f"task must be 'regression', 'binary' or 'multiclass', got {task}")

    X = X.copy()
    X.columns = X.columns.astype(str)
    y = np.asarray(y).ravel()
    n, p = X.shape

    if rdc_threshold is None:
        rdc_threshold = RDC_THRESHOLD_DEFAULT[task]
    if initial_screening_size is None:
        initial_screening_size = sis_size(n)

    base = make_base_model(task, random_state=random_state)

    # Step 1
    d = min(initial_screening_size, p, max_features)
    marginal = pd.Series(np.abs(spearman_corr(X, y)), index=X.columns)
    selected = list(marginal.nlargest(d).index)
    if verbose:
        print(f"SPARCS | task={task} n={n} p={p} | step 1 SIS warm start: {len(selected)} features")

    history = []
    pool_cap = M_rdc

    for t in range(1, max_features + 1):
        if len(selected) >= max_features:
            if verbose:
                print(f"stop: reached max_features={max_features}")
            break

        # Step 2
        model = clone(base)
        model.fit(X[selected], y, feature_names=selected)
        resid = compute_residuals(model, X[selected], y, task)

        remaining = X.drop(columns=selected)
        if remaining.shape[1] == 0:
            if verbose:
                print("stop: candidate pool exhausted")
            break

        # Step 8, applied before screening so it caps this round's pool
        pool_cap = max(min_pool, int(np.floor(M_rdc * shrink_rate ** (t - 1))))
        if verbose:
            print(f"iter {t} | |S|={len(selected)} resid_sd={np.std(resid):.4f} pool_cap={pool_cap}")

        # Steps 3 and 4
        candidates, cand_rdc, cand_p = staged_rdc_screening(
            remaining, resid, y, task, M_pre=M_prefilter, M_rdc=pool_cap,
            n_perm=n_perm, alpha=alpha, min_candidates=min_candidates,
            iter_num=t, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

        if not candidates:
            if verbose:
                print("stop: no candidates survived screening")
            break

        # Step 5
        stability = batch_cpss(X, y, selected, candidates, base, B=B_stability,
                               random_state=random_state + t, verbose=verbose)

        # Step 6
        threshold = elbow_threshold(stability, floor=stability_tau)

        # Step 7
        eligible = np.nonzero((stability >= threshold) & (cand_rdc >= rdc_threshold))[0]
        if eligible.size == 0:
            if verbose:
                print(f"stop: no candidate cleared stability>={threshold:.2f} "
                      f"and RDC>={rdc_threshold}")
            break

        order = eligible[np.argsort(-stability[eligible])][:k_add]
        room = max_features - len(selected)
        chosen = [candidates[i] for i in order[:room]]
        selected.extend(chosen)

        history.append({
            'iteration': t,
            'threshold': threshold,
            'pool_cap': pool_cap,
            'n_candidates': len(candidates),
            'added': chosen,
            'stability': [float(stability[i]) for i in order[:room]],
            'rdc': [float(cand_rdc[i]) for i in order[:room]],
            'p_value': [float(cand_p[i]) for i in order[:room]],
        })

        if verbose:
            print(f"    step 7: added {chosen} (threshold={threshold:.2f}) -> |S|={len(selected)}")

    if verbose:
        print(f"SPARCS done | {len(selected)} features selected")

    SPARCS.last_history_ = history
    return selected
