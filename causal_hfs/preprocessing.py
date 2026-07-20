"""Stage 1 - Data Preprocessing and Feature Transposition (Section 3.2).

* Median imputation of missing values (robust to outliers).
* Z-score standardisation:  z = (x - mu) / sigma.
* Conceptual "feature-as-point" transposition, which turns the feature-selection
  problem into a clustering task in an n-sample-dimensional space.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Median-impute then z-score standardise a feature matrix.

    Parameters
    ----------
    with_std:
        If ``False`` only centring is applied (used rarely; kept for flexibility).
    """

    def __init__(self, with_std: bool = True) -> None:
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler(with_std=with_std)
        self.fitted_ = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = self._imputer.fit_transform(X)
        X = self._scaler.fit_transform(X)
        # Guard against constant columns that produce nan after scaling.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.fitted_ = True
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        X = np.asarray(X, dtype=float)
        X = self._imputer.transform(X)
        X = self._scaler.transform(X)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def transpose_features(X: np.ndarray) -> np.ndarray:
    """Return the "feature-as-point" view.

    Given ``X`` of shape ``(n_samples, n_features)`` this returns an array of
    shape ``(n_features, n_samples)`` where each row is a feature described by its
    vector of observations across samples (Section 3.2).
    """
    return np.asarray(X, dtype=float).T
