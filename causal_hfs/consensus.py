"""Stage 7 - Consensus Stability and Evaluation (Section 3.6).

* Bootstrap resampling of the full pipeline.
* Frequency-based consensus voting: a feature enters the final subset if it is
  selected in at least a fraction ``tau`` of the bootstrap runs.
* Pairwise Jaccard stability index across bootstrap selections.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Callable, List, Sequence

import numpy as np


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Jaccard similarity between two feature subsets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def stability_index(selections: Sequence[Sequence[int]]) -> float:
    """Mean pairwise Jaccard stability across bootstrap selections (Section 3.6)."""
    sels = [s for s in selections if s is not None]
    if len(sels) < 2:
        return 1.0 if sels else 0.0
    scores = [jaccard(a, b) for a, b in combinations(sels, 2)]
    return float(np.mean(scores))


def consensus_selection(
    selections: Sequence[Sequence[int]],
    tau: float,
    k: int | None = None,
) -> List[int]:
    """Frequency-based consensus voting.

    Retains features selected in at least a fraction ``tau`` of runs. If ``k`` is
    given, the result is trimmed/padded to exactly ``k`` features by selection
    frequency so every method reports the same subset size (Section 3.5).
    """
    B = len(selections)
    if B == 0:
        return []
    counts = Counter()
    for sel in selections:
        counts.update(set(sel))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    retained = [f for f, c in ranked if c / B >= tau]
    if k is not None:
        if len(retained) >= k:
            retained = retained[:k]
        else:  # top up with next most-frequent features
            extra = [f for f, _ in ranked if f not in retained]
            retained = retained + extra[: k - len(retained)]
    return sorted(retained)


def cluster_consensus_selection(selections, full_labels, cluster_reps, tau, k=None):
    """Frequency voting at the redundancy-GROUP level (optional variant).

    Each selected feature is mapped to its full-data cluster id (features with no
    cluster get a unique singleton id). Clusters chosen in at least ``tau`` of the
    bootstraps are kept and replaced by their canonical full-data representative,
    so swapping two members of the same group leaves the final set unchanged.
    """
    B = len(selections)
    if B == 0:
        return []
    counts = Counter()
    for sel in selections:
        counts.update(set(full_labels.get(f, -(f + 1)) for f in sel))
    ranked = [cid for cid, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    kept = [cid for cid in ranked if counts[cid] / B >= tau]
    if k is not None and len(kept) < k:
        kept = kept + [cid for cid in ranked if cid not in kept][: k - len(kept)]

    def rep_of(cid):
        if cid in cluster_reps:
            return cluster_reps[cid]
        return (-cid - 1) if cid < 0 else None

    reps = []
    for cid in kept:
        r = rep_of(cid)
        if r is not None and r not in reps:
            reps.append(r)
    if k is not None:
        reps = reps[:k]
    return sorted(reps)


def bootstrap_runs(
    n_samples: int,
    n_bootstrap: int,
    select_fn: Callable[[np.ndarray], List[int]],
    random_state: int = 42,
) -> List[List[int]]:
    """Execute ``select_fn`` on ``n_bootstrap`` resamples of the row indices.

    ``select_fn`` receives an array of resampled row indices and returns the
    feature subset selected on that resample.
    """
    rng = np.random.default_rng(random_state)
    out: List[List[int]] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        out.append(list(select_fn(idx)))
    return out
