"""Shared primitives used by SPARCS and the baseline selectors."""

import numpy as np
from scipy.stats import rankdata


def spearman_corr(X, y):
    """Spearman rank correlation between every column of X and y."""
    Xv = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float).ravel()
    n = Xv.shape[0]
    if n < 2:
        return np.zeros(Xv.shape[1])

    Xr = rankdata(Xv, axis=0)
    yr = rankdata(yv)

    Xc = Xr - Xr.mean(axis=0)
    yc = yr - yr.mean()
    xstd = Xr.std(axis=0, ddof=0) + 1e-12
    ystd = yr.std(ddof=0) + 1e-12

    return np.nan_to_num((Xc.T @ yc) / (n * xstd * ystd))


def sis_size(n):
    """Sure Independence Screening size d = ceil(n / log n)."""
    return max(1, int(np.ceil(n / np.log(max(n, 2)))))


def _inv_sqrt(C, reg):
    """Regularized inverse square root of a symmetric positive semidefinite matrix."""
    C = C + reg * np.eye(C.shape[0])
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, reg)
    return (vecs / np.sqrt(vals)) @ vecs.T


def rdc(x, y, k=20, s=1.0, rng=None, reg=1e-6):
    """Randomized Dependence Coefficient (Lopez-Paz et al., 2013).

    Marginals are mapped to the empirical copula, pushed through k random
    sine/cosine projections, and the largest canonical correlation between the
    two projected spaces is returned. The canonical correlation is obtained by
    whitening each projected space before the cross-covariance SVD, which is
    what makes the statistic a correlation on [0, 1] rather than a
    scale-dependent covariance norm.
    """
    rng = np.random.RandomState(0) if rng is None else rng

    x = np.asarray(x, dtype=float)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    if n < 3:
        return 0.0

    x_cop = np.column_stack([rankdata(x, axis=0) / float(n), np.ones(n)])
    y_cop = np.column_stack([rankdata(y, axis=0) / float(n), np.ones(n)])

    Wx = rng.normal(0, s, (x_cop.shape[1], k))
    Wy = rng.normal(0, s, (y_cop.shape[1], k))

    Px = np.column_stack([np.cos(x_cop @ Wx), np.sin(x_cop @ Wx)])
    Py = np.column_stack([np.cos(y_cop @ Wy), np.sin(y_cop @ Wy)])

    Px = Px - Px.mean(axis=0)
    Py = Py - Py.mean(axis=0)

    Cxx = Px.T @ Px / n
    Cyy = Py.T @ Py / n
    Cxy = Px.T @ Py / n

    M = _inv_sqrt(Cxx, reg) @ Cxy @ _inv_sqrt(Cyy, reg)
    return float(np.clip(np.linalg.svd(M, compute_uv=False)[0], 0.0, 1.0))


def make_base_model(task, random_state=42, **kwargs):
    """Adaptive Elastic Net used as the internal working model everywhere.

    Imported lazily so that asgd.py can import utils without a circular import.
    """
    from asgd import ASGDR, ASGDC
    if task == 'regression':
        return ASGDR(random_state=random_state, **kwargs)
    return ASGDC(random_state=random_state, **kwargs)


def coefficient_magnitude(model):
    """Per-feature |beta|, collapsing the class axis for multiclass fits."""
    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        return np.abs(coef)
    if coef.shape[0] == 1:
        return np.abs(coef.ravel())
    return np.linalg.norm(coef, axis=0)


def compute_residuals(model, X, y, task):
    """Working residual driving the next screening round."""
    if task == 'regression':
        return np.asarray(y, dtype=float) - model.predict(X)

    probs = model.predict_proba(X)
    y_int = np.asarray(y).astype(int)
    if task == 'binary':
        return y_int - probs[:, 1]
    return np.array([1.0 - probs[i, y_int[i]] for i in range(len(y_int))])


def elbow_threshold(scores, floor=0.6):
    """Maximum-curvature point of the sorted stability profile, floored.

    Sorts scores in decreasing order and takes the second difference; the index
    of maximum curvature is the elbow. Values below `floor` are never accepted.
    """
    scores = np.sort(np.asarray(scores, dtype=float))[::-1]
    if scores.size < 3:
        return floor

    # light smoothing first: on a short, spiky profile the raw second difference
    # latches onto single-point noise instead of the actual bend
    if scores.size >= 5:
        kernel = np.ones(3) / 3.0
        smooth = np.convolve(scores, kernel, mode='same')
        smooth[0], smooth[-1] = scores[0], scores[-1]
    else:
        smooth = scores

    second_diff = np.diff(smooth, n=2)
    elbow = int(np.argmax(np.abs(second_diff))) + 1
    return float(max(floor, scores[elbow]))
