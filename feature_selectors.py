"""Leakage-safe feature-selection transformers shared across notebooks.

Defined once here and imported by the modelling notebooks. Each transformer learns only
from the data passed to fit(), so within a LOSO pipeline none of them can access the
held-out subject.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Drop columns whose missing fraction exceeds `threshold`.

    The threshold is recomputed on each training fold, so the held-out subject never
    influences which columns are considered usable.
    """
    def __init__(self, threshold=0.30):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        na_fraction = np.isnan(X).mean(axis=0)
        self.keep_ = np.where(na_fraction <= self.threshold)[0].tolist()
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop features whose absolute correlation with an already-kept feature exceeds `threshold`.
    Candidates are visited in decreasing order of relevance with the target, so of two
    correlated features the more informative one is always kept rather than whichever happens to
    come first in column order.

    `score_func` selects the relevance estimator: mutual_info_classif for classification
    (default) and mutual_info_regression for regression. Passing the classification estimator
    on a continuous target raises no error in scikit-learn, it silently treats every distinct
    value as a class, so the scorer must be set explicitly by the caller.
    """
    def __init__(self, threshold=0.9, score_func=None, random_state=0):
        self.threshold = threshold
        self.score_func = score_func          
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        c = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        n = c.shape[0]
        score_func = self.score_func if self.score_func is not None else mutual_info_classif
        relevance = score_func(X, y, random_state=self.random_state)
        order = np.argsort(relevance)[::-1]   # column indices, most informative first
        drop = set()
        for pos, i in enumerate(order):
            if i in drop:
                continue
            for j in order[pos + 1:]:
                if j not in drop and c[i, j] > self.threshold:
                    drop.add(int(j))
        self.keep_ = [i for i in range(n) if i not in drop]
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Greedy minimum-redundancy maximum-relevance selection.

    At each step adds the feature maximising mutual information with the class minus the
    mean absolute correlation with the already-selected features. `selected_` is ordered,
    so its first k entries form the size-k selection.
    """
    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, float)
        n = X.shape[1]
        k = min(self.k, n)
        relevance = mutual_info_classif(X, y, random_state=0)
        redundancy = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        selected = [int(np.argmax(relevance))]
        candidates = [i for i in range(n) if i != selected[0]]
        while len(selected) < k:
            scores = [relevance[f] - redundancy[f, selected].mean() for f in candidates]
            best = candidates[int(np.argmax(scores))]
            selected.append(best)
            candidates.remove(best)
        self.selected_ = selected
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.selected_]
