"""Stage 2 - Causal Relevance and Markov Blanket Discovery (Section 3.3).

Implements:

* A lightweight **Markov Blanket** discovery routine using the IAMB algorithm
  with a Fisher-Z partial-correlation conditional-independence test. This matches
  the paper's description of a "lightweight PC-skeleton approximation for causal
  discovery" (Section 6.4).
* Binary causal-priority scores ``C_i`` (1 if feature i is in MB(Y), else 0).
* Mutual information ``MI(f_i, Y)`` between each feature and the target.
* The causal-relevance score of Eq. (1):

      R_i = lam * C_i + (1 - lam) * MI(f_i, Y)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# --------------------------------------------------------------------------- #
# Conditional independence test
# --------------------------------------------------------------------------- #
def _partial_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Partial correlation of x and y given the columns of Z (may be empty)."""
    if Z.size == 0:
        r = np.corrcoef(x, y)[0, 1]
        return float(np.clip(r, -0.999999, 0.999999))
    # Regress x and y on Z, correlate residuals.
    Z1 = np.column_stack([np.ones(len(x)), Z])
    beta_x, *_ = np.linalg.lstsq(Z1, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    rx = x - Z1 @ beta_x
    ry = y - Z1 @ beta_y
    denom = np.std(rx) * np.std(ry)
    if denom < 1e-12:
        return 0.0
    r = np.mean((rx - rx.mean()) * (ry - ry.mean())) / denom
    return float(np.clip(r, -0.999999, 0.999999))


def fisher_z_test(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Return the p-value of the Fisher-Z test for CI(x, y | Z)."""
    n = len(x)
    k = 0 if Z.size == 0 else Z.shape[1]
    dof = n - k - 3
    if dof <= 0:
        return 1.0  # Not enough data -> assume independence (conservative).
    r = _partial_corr(x, y, Z)
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(dof) * abs(z)
    p = 2.0 * (1.0 - stats.norm.cdf(stat))
    return float(p)


# --------------------------------------------------------------------------- #
# Markov Blanket discovery (IAMB)
# --------------------------------------------------------------------------- #
def iamb_markov_blanket(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    max_cond_set: int = 3,
) -> List[int]:
    """Discover the Markov Blanket of the target ``y`` among the columns of ``X``.

    IAMB (Incremental Association Markov Blanket): a grow phase adds the feature
    most associated with the target conditioned on the current blanket, followed
    by a shrink phase that removes now-redundant members.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,)
    alpha : CI-test significance level.
    max_cond_set : cap on conditioning-set size (keeps the search lightweight).

    Returns
    -------
    Sorted list of column indices forming the estimated Markov Blanket.
    """
    n, p = X.shape
    y = np.asarray(y, dtype=float)
    mb: List[int] = []
    candidates = set(range(p))

    def assoc_strength(j: int, cond: Sequence[int]) -> float:
        Z = X[:, list(cond)] if cond else np.empty((n, 0))
        # 1 - p_value used as an association proxy (higher -> more dependent).
        return 1.0 - fisher_z_test(X[:, j], y, Z)

    # ----- Grow phase -----
    changed = True
    while changed:
        changed = False
        best_j, best_score = None, -np.inf
        cond = mb[-max_cond_set:]
        for j in candidates - set(mb):
            s = assoc_strength(j, cond)
            if s > best_score:
                best_score, best_j = s, j
        if best_j is not None:
            Z = X[:, cond] if cond else np.empty((n, 0))
            p_val = fisher_z_test(X[:, best_j], y, Z)
            if p_val < alpha:  # dependent given current MB -> add
                mb.append(best_j)
                changed = True

    # ----- Shrink phase -----
    changed = True
    while changed and mb:
        changed = False
        for j in list(mb):
            others = [m for m in mb if m != j][:max_cond_set]
            Z = X[:, others] if others else np.empty((n, 0))
            p_val = fisher_z_test(X[:, j], y, Z)
            if p_val >= alpha:  # conditionally independent -> remove
                mb.remove(j)
                changed = True
                break
    return sorted(mb)


# --------------------------------------------------------------------------- #
# Mutual information & causal relevance
# --------------------------------------------------------------------------- #
def mutual_information(
    X: np.ndarray, y: np.ndarray, discrete_target: bool = True, random_state: int = 0
) -> np.ndarray:
    """MI(f_i, Y) for every feature, normalised to [0, 1]."""
    if discrete_target:
        mi = mutual_info_classif(X, y, random_state=random_state)
    else:
        mi = mutual_info_regression(X, y, random_state=random_state)
    mi = np.asarray(mi, dtype=float)
    m = mi.max()
    return mi / m if m > 0 else mi


def causal_relevance(
    mb_mask: np.ndarray, mi: np.ndarray, lam: float
) -> np.ndarray:
    """Causal-relevance score of Eq. (1):  R_i = lam*C_i + (1-lam)*MI(f_i, Y).

    The second term is the "predictive association" component. By default it is the
    marginal mutual information, but the framework may pass a *conditional*
    relevance vector instead (see :func:`conditional_relevance_scores`).
    """
    C = np.asarray(mb_mask, dtype=float)  # binary causal-priority scores
    mi = np.asarray(mi, dtype=float)
    return lam * C + (1.0 - lam) * mi


def conditional_relevance_scores(
    X: np.ndarray, y: np.ndarray, max_cond: int = 12
) -> np.ndarray:
    """Continuous direct-effect score (Improvement #1).

    For each feature ``i`` this returns ``|partial_corr(f_i, y | Z)|`` where ``Z``
    is the set of the ``max_cond`` other features most marginally associated with
    the target. A genuine causal driver keeps a non-zero partial correlation after
    conditioning, whereas a spurious aggregate (a function of several drivers)
    drops toward zero once its parents are in ``Z``. Normalised to ``[0, 1]``.

    Unlike marginal mutual information, this discriminates true drivers from
    strongly-correlated-but-non-causal aggregates, which is what lets the framework
    prefer drivers over spurious hubs.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    marg = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 1e-9
                     else 0.0 for j in range(p)])
    marg = np.nan_to_num(marg)
    order = np.argsort(marg)[::-1]
    sc = np.zeros(p)
    for i in range(p):
        cond = [j for j in order if j != i][:max_cond]
        sc[i] = abs(_partial_corr(X[:, i], y, X[:, cond]))
    m = sc.max()
    return sc / m if m > 0 else sc


def rf_relevance_scores(X: np.ndarray, y: np.ndarray, discrete_target: bool = True,
                        random_state: int = 0, n_estimators: int = 100) -> np.ndarray:
    """Random-Forest feature importance, normalised to ``[0, 1]`` (supervised, non-linear)."""
    if discrete_target:
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    imp = np.asarray(est.fit(X, y).feature_importances_, dtype=float)
    m = imp.max()
    return imp / m if m > 0 else imp


class CausalAnalyzer:
    """Convenience wrapper computing MB, MI and the relevance vector ``R``."""

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 3,
                 discrete_target: bool = True, random_state: int = 0,
                 use_conditional: bool = False, cond_max_set: int = 12,
                 rf_relevance: bool = False) -> None:
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.discrete_target = discrete_target
        self.random_state = random_state
        self.use_conditional = use_conditional
        self.cond_max_set = cond_max_set
        self.rf_relevance = rf_relevance
        self.mb_: List[int] = []
        self.mb_mask_: np.ndarray | None = None
        self.mi_: np.ndarray | None = None
        self.cond_rel_: np.ndarray | None = None
        self.predictive_: np.ndarray | None = None   # the score used in R (mi or cond_rel)
        self.relevance_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, lam: float) -> "CausalAnalyzer":
        p = X.shape[1]
        self.mb_ = iamb_markov_blanket(X, y, self.alpha, self.max_cond_set)
        mask = np.zeros(p, dtype=float)
        mask[self.mb_] = 1.0
        self.mb_mask_ = mask
        self.mi_ = mutual_information(X, y, self.discrete_target, self.random_state)
        if self.rf_relevance:
            self.cond_rel_ = rf_relevance_scores(X, y, self.discrete_target, self.random_state)
            self.predictive_ = self.cond_rel_
        elif self.use_conditional:
            self.cond_rel_ = conditional_relevance_scores(X, y, self.cond_max_set)
            self.predictive_ = self.cond_rel_
        else:
            self.predictive_ = self.mi_
        self.relevance_ = causal_relevance(mask, self.predictive_, lam)
        return self
