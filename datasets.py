"""Synthetic high-dimensional regression benchmarks (datasets s1 and s2).

Both datasets carry 10,000 gene-like features over 500 patient-like samples with
80 truly informative features, block correlation of roughly 0.3 within a block to
imitate gene modules, and mixed marginals including heavy-tailed components. They
differ in the fraction of informative features acting through a nonlinear link,
the number of pairwise interaction terms, and the noise level.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

S1 = dict(nonlinear_frac=0.2, n_interactions=5, noise_level=0.35, random_state=42)
S2 = dict(nonlinear_frac=0.8, n_interactions=10, noise_level=0.50, random_state=7)


def _block_correlated(n_samples, n_features, block_size, rho, rng):
    """Equicorrelated blocks: within-block correlation is exactly rho."""
    X = np.empty((n_samples, n_features))
    a, b = np.sqrt(rho), np.sqrt(1.0 - rho)
    for start in range(0, n_features, block_size):
        end = min(start + block_size, n_features)
        shared = rng.standard_normal((n_samples, 1))
        X[:, start:end] = a * shared + b * rng.standard_normal((n_samples, end - start))
    return X


def _mix_marginals(X, heavy_frac, rng):
    """Push a fraction of columns to heavy-tailed or skewed marginals.

    The copula is preserved by mapping through the normal CDF, so the block
    correlation structure survives the transformation.
    """
    n_features = X.shape[1]
    n_heavy = int(round(heavy_frac * n_features))
    picked = rng.choice(n_features, size=n_heavy, replace=False)
    half = n_heavy // 2

    u = np.clip(norm.cdf(X[:, picked]), 1e-6, 1 - 1e-6)
    heavy = student_t.ppf(u[:, :half], df=3) / np.sqrt(3.0)
    skewed = np.exp(0.5 * norm.ppf(u[:, half:]))
    skewed = (skewed - skewed.mean(axis=0)) / (skewed.std(axis=0) + 1e-12)

    X[:, picked[:half]] = heavy
    X[:, picked[half:]] = skewed
    return X


def _nonlinear(x, kind):
    if kind == 0:
        return x ** 2 - 1.0
    if kind == 1:
        return np.log1p(np.abs(x))
    if kind == 2:
        return np.tanh(x)
    return (x > np.quantile(x, 0.7)).astype(float)


def generate_synthetic_dataset(n_samples=500, n_features=10000, n_informative=80,
                               nonlinear_frac=0.2, n_interactions=5, noise_level=0.35,
                               block_size=50, rho=0.3, heavy_frac=0.3, random_state=42):
    """Return (X, y, meta) with meta carrying the ground truth feature names."""
    rng = np.random.default_rng(random_state)

    X = _block_correlated(n_samples, n_features, block_size, rho, rng)
    X = _mix_marginals(X, heavy_frac, rng)

    true_idx = np.sort(rng.choice(n_features, size=n_informative, replace=False))
    n_nonlinear = int(round(nonlinear_frac * n_informative))
    shuffled = rng.permutation(true_idx)
    nonlinear_idx, linear_idx = shuffled[:n_nonlinear], shuffled[n_nonlinear:]

    beta = rng.uniform(1.0, 2.0, size=n_informative) * rng.choice([-1.0, 1.0], size=n_informative)
    beta = pd.Series(beta, index=true_idx)

    signal = np.zeros(n_samples)
    for j in linear_idx:
        signal += beta[j] * X[:, j]
    for rank, j in enumerate(nonlinear_idx):
        signal += beta[j] * _nonlinear(X[:, j], rank % 4)

    pairs = rng.choice(true_idx, size=(n_interactions, 2), replace=True)
    for j, k in pairs:
        signal += 1.5 * X[:, j] * X[:, k]

    noise = rng.normal(0.0, noise_level * np.std(signal), n_samples)
    y = signal + noise

    names = [f"Gene_{i}" for i in range(n_features)]
    meta = {
        'true_feature_indices': true_idx,
        'true_feature_names': [names[i] for i in true_idx],
        'n_nonlinear': int(n_nonlinear),
        'n_interactions': int(n_interactions),
        'noise_level': float(noise_level),
        'snr': float(np.var(signal) / np.var(noise)),
        'within_block_rho': float(rho),
    }
    return pd.DataFrame(X, columns=names), y, meta


def generate_s1(**kwargs):
    return generate_synthetic_dataset(**{**S1, **kwargs})


def generate_s2(**kwargs):
    return generate_synthetic_dataset(**{**S2, **kwargs})


def prepare_data(X, y, test_size=0.2, random_state=42, n_strata=5):
    """Split and standardize. Regression targets are stratified by response quantile."""
    labels = np.asarray(y)
    if len(np.unique(labels)) >= 20:
        labels = pd.qcut(labels, n_strata, labels=False, duplicates='drop')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=labels)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test
