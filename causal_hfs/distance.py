"""Stage 4 - Hybrid Distance Computation (Section 3.4).

* Statistical distance (Eq. 3):  d_stat(f_i, f_j) = 1 - |corr(f_i, f_j)|.
* Causal distance: the paper introduces a causal-alignment term but leaves its
  exact form to the implementation. We define it from the causal-relevance
  vector ``R`` (Eq. 1) as the normalised absolute difference in causal relevance,

      d_causal(f_i, f_j) = |R_i - R_j| / max_kl |R_k - R_l|

  so that features playing a similar causal role toward the target are "close"
  in causal space and features with conflicting causal roles are far apart. This
  is exactly the signal surfaced to the expert in the HITL interface
  ("difference in causal-relevance scores", "conflicting causal roles").
* Hybrid distance (Eq. 4):
      d_hybrid = alpha * d_stat + (1 - alpha) * d_causal.
"""

from __future__ import annotations

import numpy as np

from .graph import correlation_matrix


def statistical_distance(X: np.ndarray) -> np.ndarray:
    """Statistical distance matrix, Eq. (3)."""
    return 1.0 - correlation_matrix(X)


def causal_distance(relevance: np.ndarray) -> np.ndarray:
    """Causal distance matrix from the causal-relevance vector ``R``."""
    R = np.asarray(relevance, dtype=float).reshape(-1, 1)
    D = np.abs(R - R.T)
    m = D.max()
    return D / m if m > 0 else D


def hybrid_distance(
    X: np.ndarray,
    relevance: np.ndarray,
    alpha: float = 0.5,
    enable_causal: bool = True,
) -> np.ndarray:
    """Hybrid causal-statistical distance matrix, Eq. (4).

    If ``enable_causal`` is False the causal term is dropped (ablation:
    "w/o Causality"), reducing the matrix to the purely statistical distance.
    """
    d_stat = statistical_distance(X)
    if not enable_causal:
        D = d_stat
    else:
        d_causal = causal_distance(relevance)
        D = alpha * d_stat + (1.0 - alpha) * d_causal
    D = 0.5 * (D + D.T)          # enforce symmetry
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, None)
