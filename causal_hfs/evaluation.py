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

    # Bound the elbow search for speed on wide data (the accuracy elbow for an
    # interpretable subset is virtually always small) and sweep a <=40-point grid.
    search_max = int(min(k_max, 100))
    order = greedy_select(rel, corr, search_max, beta=beta, return_order=True)
    if search_max - k_min > 40:
        step = max(1, (search_max - k_min) // 40)
        ks = list(range(k_min, search_max + 1, step))
        if ks[-1] != search_max:
            ks.append(search_max)
    else:
        ks = list(range(k_min, search_max + 1))

    curve = {}
    for kk in ks:
        feats = sorted(order[:kk])
        curve[kk] = knn_accuracy(Xp[:, feats], y, random_state=random_state)
    best_acc = max(curve.values())
    best_k = min(kk for kk in curve if curve[kk] >= best_acc - tol)
    return int(best_k), curve


def kuncheva_index(a: Sequence[int], b: Sequence[int], p: int) -> float:
    """Kuncheva consistency index for two size-k subsets of ``p`` features.

    K = (r - k^2/p) / (k - k^2/p), where r = |a ∩ b|. Unlike Jaccard it corrects
    for the overlap expected by chance, so it is the right stability measure for
    *fixed-size* feature sets (a size-k Jaccard is inflated for large k/p).
    """
    a, b = set(a), set(b)
    k = len(a)
    if k == 0 or k >= p:
        return 1.0
    r = len(a & b)
    exp = k * k / p
    denom = k - exp
    if abs(denom) < 1e-12:
        return 1.0
    return float((r - exp) / denom)


def _core_ranking(X_train, y_train, k_max, random_state=0, beta=1.0,
                  improved_lam=0.15, discrete_target=True, prefilter_top=150):
    """Leakage-safe ranked feature path for the *proposed* method on a train fold.

    Fits preprocessing on the training partition ONLY, computes the causal-aware
    relevance R = λ·rank(MB) + (1-λ)·rank(RF-importance), then returns the greedy
    max-relevance / min-redundancy ORDER (so prefixes give nested sets S_k). Returns
    indices into the original columns of ``X_train``.
    """
    from .graph import correlation_matrix
    from .clustering import greedy_select
    from .causal import CausalAnalyzer
    Xp = Preprocessor().fit_transform(X_train)
    y = np.asarray(y_train)
    p = Xp.shape[1]
    # High-dim guard: ANOVA-F prefilter inside the fold (still leakage-safe).
    pf = np.arange(p)
    if prefilter_top and p > prefilter_top:
        from sklearn.feature_selection import f_classif
        F, _ = f_classif(Xp, y)
        F = np.nan_to_num(F)
        pf = np.argsort(F)[::-1][:prefilter_top]
        Xp = Xp[:, pf]
    analyzer = CausalAnalyzer(discrete_target=discrete_target,
                              random_state=random_state, rf_relevance=True,
                              rank_norm=True).fit(Xp, y, lam=improved_lam)
    R = analyzer.relevance_
    corr = correlation_matrix(Xp)
    kk = int(min(k_max, Xp.shape[1]))
    order_local = greedy_select(R, corr, kk, beta=beta, return_order=True)
    return [int(pf[j]) for j in order_local]


def select_k_nested(X, y, k_grid=None, k_max=None, n_splits=5, n_repeats=3,
                    tau_stability=0.6, metric="balanced_accuracy",
                    complexity_weight=0.02, discrete_target=True, random_state=0,
                    models=("logreg", "knn")):
    """Choose ``k`` by repeated nested CV with a composite, stability-aware rule.

    Implements the recommended hierarchy:
      1. Build a leakage-safe ranked feature path *inside every training fold*.
      2. Score each prefix S_k with a model-agnostic downstream metric.
      3. Keep k within one standard error of the best mean performance.
      4. Drop k below the minimum Kuncheva-stability threshold.
      5. Among survivors pick lowest redundancy + complexity, tie-break smaller k.
      6. If none pass the stability floor, fall back to a normalised composite score.

    Returns ``(k_star, diagnostics)`` where diagnostics carries per-k mean/SE
    performance, Kuncheva stability, and redundancy for plotting.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import get_scorer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from .graph import correlation_matrix

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, p = X.shape
    # Default k_max: min(p, 100, n/10) — guards small-n, large-p overfitting.
    if k_max is None:
        k_max = int(max(2, min(p, 100, n // 10)))
    # Never let auto-k select ALL features: at k = p every method keeps everything,
    # so the comparison degenerates (all methods tie). Cap strictly below p.
    k_max = int(min(k_max, max(2, p - 1)))
    if k_grid is None:
        base = list(range(1, min(20, k_max) + 1))
        base += [k for k in (25, 30, 40, 50, 75, 100) if k <= k_max]
        k_grid = sorted(set(k for k in base if 1 <= k <= k_max))
    scorer = get_scorer(metric)

    def _build_models():
        ms = []
        for m in models:
            if m == "knn":
                ms.append(KNeighborsClassifier(n_neighbors=min(5, max(1, n - 1))))
            elif m == "logreg":
                ms.append(LogisticRegression(max_iter=1000))
        return ms or [KNeighborsClassifier(n_neighbors=5)]

    minc = np.min(np.bincount(y)) if y.dtype.kind in "iu" else n
    nsp = max(2, min(n_splits, int(minc)))
    rskf = RepeatedStratifiedKFold(n_splits=nsp, n_repeats=n_repeats,
                                   random_state=random_state)
    perf = {k: [] for k in k_grid}
    red = {k: [] for k in k_grid}
    sets = {k: [] for k in k_grid}
    for tr, va in rskf.split(X, y):
        Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
        order = _core_ranking(Xtr, ytr, max(k_grid), random_state=random_state,
                              discrete_target=discrete_target)
        # Preprocessing (z-score) fit on train only, applied to both partitions.
        sc = StandardScaler().fit(Xtr)
        Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
        corr_tr = correlation_matrix(Ztr)
        for k in k_grid:
            S = order[:k]
            sets[k].append(tuple(sorted(S)))
            best = -np.inf
            for mdl in _build_models():
                try:
                    mdl.fit(Ztr[:, S], ytr)
                    sc_val = scorer(mdl, Zva[:, S], yva)
                except Exception:
                    sc_val = np.nan
                if np.isfinite(sc_val):
                    best = max(best, sc_val)
            perf[k].append(best if np.isfinite(best) else np.nan)
            if k >= 2:
                sub = corr_tr[np.ix_(S, S)]
                iu = np.triu_indices(k, 1)
                red[k].append(float(np.nanmean(np.abs(sub[iu]))))
            else:
                red[k].append(0.0)

    mean_perf, se_perf, stab, redun = {}, {}, {}, {}
    for k in k_grid:
        v = np.array([x for x in perf[k] if np.isfinite(x)])
        mean_perf[k] = float(np.mean(v)) if v.size else float("nan")
        se_perf[k] = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        redun[k] = float(np.nanmean(red[k])) if red[k] else 0.0
        ss = sets[k]
        pair = [kuncheva_index(ss[i], ss[j], p)
                for i in range(len(ss)) for j in range(i + 1, len(ss))]
        stab[k] = float(np.mean(pair)) if pair else 1.0

    k_best = max(mean_perf, key=lambda k: (mean_perf[k] if np.isfinite(mean_perf[k]) else -np.inf))
    thr = mean_perf[k_best] - se_perf[k_best]
    eligible = [k for k in k_grid if np.isfinite(mean_perf[k]) and mean_perf[k] >= thr]
    stable = [k for k in eligible if stab[k] >= tau_stability]
    if stable:
        k_star = min(stable, key=lambda k: (redun[k] + complexity_weight * k / max(k_grid), k))
        rule = "one-SE ∩ stability, min redundancy"
    else:
        # Fallback: normalised composite among all k (perf dominant).
        def _nrm(d):
            vals = np.array([d[k] for k in k_grid], float)
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            return {k: (0.0 if hi - lo < 1e-12 else (d[k] - lo) / (hi - lo)) for k in k_grid}
        nP, nS, nR = _nrm(mean_perf), _nrm(stab), _nrm(redun)
        comp = {k: nP[k] + 0.20 * nS[k] - 0.10 * nR[k] - 0.05 * k / max(k_grid)
                for k in k_grid}
        k_star = max(comp, key=lambda k: comp[k])
        rule = "composite score (no k met stability floor)"

    diag = {"k_grid": k_grid, "mean_perf": mean_perf, "se_perf": se_perf,
            "stability": stab, "redundancy": redun, "k_best": k_best,
            "one_se_threshold": thr, "eligible": eligible, "stable": stable,
            "metric": metric, "rule": rule}
    return int(k_star), diag


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
