"""Adaptive Elastic Net fitted through column rescaling and an SGD solver.

The adaptive penalty

    lambda [ (1 - alpha) ||beta||_2^2 + alpha sum_j w_j |beta_j| ],
    w_j = 1 / (|beta_j^init| + eps)^gamma

is reduced to a standard elastic net by rescaling column j to X_j / w_j and
recovering beta_j = beta_tilde_j / w_j, which leaves X beta unchanged and turns
the weighted L1 term into a plain L1 term.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, SGDRegressor, SGDClassifier

EPS = 1e-8


def _feature_names(X, feature_names):
    if feature_names is not None:
        return list(feature_names)
    if hasattr(X, 'columns'):
        return list(X.columns)
    return [f"X{i}" for i in range(np.asarray(X).shape[1])]


class ASGDR(BaseEstimator, RegressorMixin):

    def __init__(self, l1_ratio=0.5, alpha=0.1, gamma=1.0, ridge_alpha=0.1,
                 max_iter=1000, tol=1e-3, random_state=None):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y, feature_names=None):
        self.feature_names_ = _feature_names(X, feature_names)
        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float).ravel()

        init = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        init.fit(Xv, yv)
        init_coef = np.abs(np.atleast_1d(init.coef_).ravel())

        self.adaptive_weights_ = 1.0 / (init_coef + EPS) ** self.gamma
        scale = 1.0 / self.adaptive_weights_

        self.model_ = SGDRegressor(
            penalty='elasticnet', l1_ratio=self.l1_ratio, alpha=self.alpha,
            max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
        self.model_.fit(Xv * scale, yv)

        self.coef_ = self.model_.coef_ * scale
        self.intercept_ = self.model_.intercept_

        self.coefficient_importance_ = np.abs(self.coef_)
        self.adaptive_importance_ = np.abs(self.coef_) / (self.adaptive_weights_ + EPS)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def get_feature_importance(self, importance_type='adaptive'):
        if not hasattr(self, 'coef_'):
            raise ValueError("model must be fitted before requesting feature importance")

        if importance_type == 'adaptive':
            df = pd.DataFrame({'feature': self.feature_names_,
                               'adaptive_importance': self.adaptive_importance_,
                               'coefficient': self.coef_})
            df = df.sort_values('adaptive_importance', ascending=False)
        else:
            df = pd.DataFrame({'feature': self.feature_names_,
                               'coefficient_importance': self.coefficient_importance_,
                               'coefficient': self.coef_})
            df = df.sort_values('coefficient_importance', ascending=False)

        return df.reset_index(drop=True)


class ASGDC(BaseEstimator, ClassifierMixin):

    def __init__(self, l1_ratio=0.5, alpha=1e-4, gamma=1.0, ridge_alpha=0.1,
                 max_iter=1000, tol=1e-3, random_state=None):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y, feature_names=None):
        self.feature_names_ = _feature_names(X, feature_names)
        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y).ravel()
        self.classes_ = np.unique(yv)

        init = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=self.ridge_alpha,
            max_iter=self.max_iter, tol=self.tol, class_weight='balanced',
            random_state=self.random_state)
        init.fit(Xv, yv)

        init_coef = (np.abs(init.coef_.ravel()) if len(self.classes_) == 2
                     else np.linalg.norm(init.coef_, axis=0))

        self.adaptive_weights_ = 1.0 / (init_coef + EPS) ** self.gamma
        scale = 1.0 / self.adaptive_weights_

        self.model_ = SGDClassifier(
            loss='log_loss', penalty='elasticnet', l1_ratio=self.l1_ratio,
            alpha=self.alpha, max_iter=self.max_iter, tol=self.tol,
            class_weight='balanced', random_state=self.random_state)
        self.model_.fit(Xv * scale, yv)

        self.coef_ = self.model_.coef_ * scale
        self.intercept_ = self.model_.intercept_

        magnitude = (np.abs(self.coef_.ravel()) if len(self.classes_) == 2
                     else np.linalg.norm(self.coef_, axis=0))
        self.coefficient_importance_ = magnitude
        self.adaptive_importance_ = magnitude / (self.adaptive_weights_ + EPS)
        return self

    def _scaled(self, X):
        return np.asarray(X, dtype=float) * (1.0 / self.adaptive_weights_)

    def predict(self, X):
        return self.model_.predict(self._scaled(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._scaled(X))

    def decision_function(self, X):
        return self.model_.decision_function(self._scaled(X))

    def get_feature_importance(self, importance_type='adaptive'):
        if not hasattr(self, 'coef_'):
            raise ValueError("model must be fitted before requesting feature importance")

        column = ('adaptive_importance' if importance_type == 'adaptive'
                  else 'coefficient_importance')
        values = (self.adaptive_importance_ if importance_type == 'adaptive'
                  else self.coefficient_importance_)

        df = pd.DataFrame({'feature': self.feature_names_,
                           column: values,
                           'coefficient_magnitude': self.coefficient_importance_})
        return df.sort_values(column, ascending=False).reset_index(drop=True)


def select_by_importance(model, k):
    """Top-k features by adaptive importance, the selection rule used in the paper."""
    ranked = model.get_feature_importance(importance_type='adaptive')
    return ranked['feature'].head(int(k)).tolist()
