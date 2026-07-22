"""Evaluation protocol and metrics (Sections 4.2 & 5).

Metrics
-------
* **Accuracy**   - 5-fold KNN classification accuracy on the method's output
                   representation (Table 4).
* **Stability**  - mean pairwise Jaccard across bootstrap selections (Section 3.6).
* **Trustworthiness** - sklearn neighbourhood-preservation trustworthiness between
                   the original standardised space and the reduced space (Table 2).
* **Runtime**    - wall-clock seconds to fit + transform.

Statistical tests
-----------------
* Friedman test across methods for accuracy and stability (Section 5.2).
* Pairwise Wilcoxon signed-rank tests, proposed vs. each baseline (Table 3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.manifold import trustworthiness as sk_trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from .consensus import stability_index
from .preprocessing import Preprocessor


def suggest_k(X, y, k_min=2, k_max=20, discrete_target=True, random_state=0,
              tol=0.005, beta=1.0):
    """Suggest the number of features ``k`` at the downstream-accuracy elbow.

    Ranks features once with the proposed greedy max-relevance/min-redundancy
    order (Random-Forest importance vs. correlation redundancy), then scores every
    prefix length by 5-fold KNN accuracy. Returns the *smallest* ``k`` whose
    accuracy is within ``tol`` of the best (a parsimony / elbow rule that favours a
    compact, interpretable subset), along with the per-k accuracy curve.
    """
    from .graph import correlation_matrix
    from .clustering import greedy_select
    Xp = Preprocessor().fit_transform(X)
    y = np.asarray(y)
    p = Xp.shape[1]
    k_max = int(min(k_max, p))
    k_min = int(max(2, min(k_min, k_max)))
    if discrete_target:
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rel = np.asarray(est.fit(Xp, y).feature_importances_, dtype=float)
    rel = rel / (rel.max() or 1.0)
    corr = correlation_matrix(Xp)
    order = greedy_select(rel, corr, k_max, beta=beta, return_order=True)

    curve = {}
    for kk in range(k_min, k_max + 1):
        feats = sorted(order[:kk])
        curve[kk] = knn_accuracy(Xp[:, feats], y, random_state=random_state)
    best_acc = max(curve.values())
    best_k = min(kk for kk in curve if curve[kk] >= best_acc - tol)
    return int(best_k), curve


def knn_accuracy(rep: np.ndarray, y: np.ndarray, n_splits: int = 5,
                 n_neighbors: int = 5, random_state: int = 42) -> float:
    """5-fold stratified KNN accuracy on a representation ``rep``."""
    y = np.asarray(y)
    n_splits = min(n_splits, np.min(np.bincount(y)) if y.dtype.kind in "iu" else n_splits)
    n_splits = max(2, int(n_splits))
    clf = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(y) - 1))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, rep, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


def trustworthiness_score(X_full: np.ndarray, rep: np.ndarray,
                          n_neighbors: int = 5) -> float:
    """Neighbourhood-preservation trustworthiness in [0, 1] (higher is better)."""
    n = X_full.shape[0]
    k = min(n_neighbors, max(1, (n - 1) // 2))
    try:
        return float(sk_trustworthiness(X_full, rep, n_neighbors=k))
    except Exception:
        return float("nan")


def causal_plausibility(selected, true_relevant) -> float:
    """Fraction of selected features that are genuine causal drivers.

    Standard Jaccard stability only measures whether a method selects the *same*
    features across resamples — not whether those features are *correct*. On a
    dataset with a known ground-truth Markov Blanket ``true_relevant``, this
    metric reports the precision of the selection:

        CP = |selected ∩ true_relevant| / |selected|

    A method can be perfectly stable (Jaccard = 1) yet have low causal
    plausibility if it consistently selects spurious correlates rather than the
    true drivers. Returns NaN when no ground truth is available.
    """
    if true_relevant is None:
        return float("nan")
    sel = set(int(s) for s in selected)
    tr = set(int(t) for t in true_relevant)
    if not sel:
        return 0.0
    return len(sel & tr) / len(sel)


def causal_recall(selected, true_relevant) -> float:
    """Fraction of the true Markov Blanket that was recovered (|sel∩true|/|true|)."""
    if true_relevant is None:
        return float("nan")
    sel = set(int(s) for s in selected)
    tr = set(int(t) for t in true_relevant)
    if not tr:
        return float("nan")
    return len(sel & tr) / len(tr)


@dataclass
class MethodResult:
    method: str
    accuracy: float
    stability: float
    trustworthiness: float
    runtime: float
    selected_features: List[int] = field(default_factory=list)
    causal_plausibility: float = float("nan")
    causal_recall: float = float("nan")


def evaluate_method(
    method_name: str,
    build_fn,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    n_bootstrap: int = 15,
    random_state: int = 42,
    true_relevant=None,
    progress_cb=None,
) -> MethodResult:
    """Fit a method, then compute accuracy, stability, trustworthiness, runtime.

    ``build_fn(seed)`` must return a fresh, unfitted method object exposing
    ``fit``, ``transform`` and ``selected_features_`` (both baselines and the
    proposed framework satisfy this via light adapters in ``run_experiments``).

    When ``true_relevant`` (the ground-truth Markov Blanket indices) is supplied,
    the Causal-Plausibility and Causal-Recall metrics are also computed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = X.shape[0]

    def _tick(frac):
        if progress_cb is not None:
            try:
                progress_cb(min(1.0, max(0.0, frac)))
            except Exception:
                pass

    # --- fit on full data + timing ---
    _tick(0.02)
    t0 = time.perf_counter()
    model = build_fn(random_state)
    model.fit(X, y)
    rep = model.transform(X)
    runtime = time.perf_counter() - t0
    _tick(0.30)                       # full-data fit is the bulk of one method

    # --- accuracy ---
    acc = knn_accuracy(rep, y, random_state=random_state)

    # --- trustworthiness (vs. full standardised space) ---
    Xstd = Preprocessor().fit_transform(X)
    trust = trustworthiness_score(Xstd, np.asarray(rep, dtype=float))
    _tick(0.38)

    # --- stability across bootstraps ---
    # Each resample gets its own seed so that data-agnostic selectors (e.g.
    # Random) exhibit their true instability rather than a fixed-seed artefact.
    rng = np.random.default_rng(random_state)
    selections: List[List[int]] = []
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = build_fn(random_state + i + 1)
        m.fit(X[idx], y[idx])
        selections.append(list(getattr(m, "selected_features_", [])))
        _tick(0.38 + 0.60 * (i + 1) / max(1, n_bootstrap))
    stab = stability_index(selections)
    _tick(1.0)

    # --- causal plausibility (needs a known ground-truth Markov Blanket) ---
    sel = list(getattr(model, "selected_features_", []))
    cp = causal_plausibility(sel, true_relevant)
    cr = causal_recall(sel, true_relevant)

    return MethodResult(
        method=method_name, accuracy=acc, stability=stab,
        trustworthiness=trust, runtime=runtime,
        selected_features=sel, causal_plausibility=cp, causal_recall=cr,
    )


# --------------------------------------------------------------------------- #
# Statistical significance
# --------------------------------------------------------------------------- #
def friedman_test(score_matrix: np.ndarray) -> tuple[float, float]:
    """Friedman test across methods. ``score_matrix`` is (datasets, methods)."""
    columns = [score_matrix[:, j] for j in range(score_matrix.shape[1])]
    stat, p = stats.friedmanchisquare(*columns)
    return float(stat), float(p)


def wilcoxon_vs_baselines(
    proposed: np.ndarray, baselines: Dict[str, np.ndarray]
) -> Dict[str, dict]:
    """Pairwise Wilcoxon signed-rank tests, proposed vs. each baseline.

    Returns per-baseline dict with mean delta (proposed - baseline) and p-value.
    """
    out: Dict[str, dict] = {}
    for name, vals in baselines.items():
        delta = float(np.mean(proposed - vals))
        try:
            _, p = stats.wilcoxon(proposed, vals)
        except ValueError:  # all-zero differences
            p = 1.0
        out[name] = {"delta": delta, "p_value": float(p)}
    return out
