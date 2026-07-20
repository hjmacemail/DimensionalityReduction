"""Smoke tests for the Causal-HFS package (no network, offline sklearn data).

Run locally with ``pytest -q``; also run in CI on every push (see
``.github/workflows/ci.yml``).
"""

import numpy as np
from sklearn.datasets import load_wine

from causal_hfs import CausalHFS, FrameworkConfig
from causal_hfs.baselines import BASELINES, build_baseline
from causal_hfs.evaluation import evaluate_method


def _wine():
    d = load_wine()
    return d.data, d.target


def test_framework_fits_and_selects():
    X, y = _wine()
    m = CausalHFS(FrameworkConfig(n_representatives=5, n_bootstrap=4)).fit(X, y)
    assert len(m.selected_features_) == 5
    assert m.transform(X).shape == (X.shape[0], 5)
    assert 0.0 <= m.stability_ <= 1.0
    assert all(0 <= i < X.shape[1] for i in m.selected_features_)


def test_strict_causal_mode_and_relevance_length():
    X, y = _wine()
    cfg = FrameworkConfig(n_representatives=5, n_bootstrap=4, restrict_to_mb=True,
                          mb_max_cond_set=4, conditional_relevance=True,
                          prototype_by="relevance")
    m = CausalHFS(cfg).fit(X, y)
    assert 1 <= len(m.selected_features_) <= 5
    assert len(m.relevance_) == X.shape[1]      # full-length even with MB isolation


def test_all_baselines_produce_k_features():
    X, y = _wine()
    k = 5
    for name in BASELINES:
        b = build_baseline(name, k=k)
        b.fit(X, y)
        assert len(b.selected_features_) == k
        assert b.transform(X).shape[0] == X.shape[0]


def test_evaluate_method_returns_valid_metrics():
    X, y = _wine()
    r = evaluate_method(
        "Proposed",
        lambda s: CausalHFS(FrameworkConfig(n_representatives=5, n_bootstrap=4, random_state=s)),
        X, y, 5, n_bootstrap=4,
    )
    assert 0.0 <= r.accuracy <= 1.0
    assert 0.0 <= r.stability <= 1.0


def test_highdim_prefilter_path():
    # Widen real data by tiling to trigger the high-dimensional prefilter.
    X, y = _wine()
    Xwide = np.hstack([X + 0.01 * i for i in range(20)])   # 260 features
    m = CausalHFS(FrameworkConfig(n_representatives=6, n_bootstrap=3,
                                  prefilter_top=50)).fit(Xwide, y)
    assert len(m.selected_features_) >= 1
    assert all(0 <= i < Xwide.shape[1] for i in m.selected_features_)


def test_wrapper_refinement_runs():
    X, y = _wine()
    m = CausalHFS(FrameworkConfig(n_representatives=5, n_bootstrap=3,
                                  restrict_to_mb=True, mb_max_cond_set=4,
                                  wrapper_refine=True)).fit(X, y)
    assert 1 <= len(m.selected_features_) <= 5
