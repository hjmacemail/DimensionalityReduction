"""Stages 5 & 6 - Hierarchical Agglomeration and Prototype Selection (Section 3.5).

* Average-linkage agglomerative clustering on the precomputed hybrid distance
  matrix (Eq. 5), stopping when ``k`` clusters remain.
* Each cluster is represented by one prototype feature chosen by the composite
  score of Eq. (6):

      S_i = B1 * Centrality_i + B2 * R_i + B3 * MI(f_i, Y),   sum(B) = 1.

The merge history is retained so the HITL module can inspect the sequence of
merges and flag causally ambiguous ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


@dataclass
class MergeEvent:
    """One agglomeration step recorded for auditing / HITL review."""

    step: int
    left_cluster: Tuple[int, ...]
    right_cluster: Tuple[int, ...]
    distance: float
    forced: bool = False          # True if a HITL veto overrode the natural merge
    approved: Optional[bool] = None  # HITL decision, if consulted


@dataclass
class ClusteringResult:
    labels: np.ndarray                     # cluster id per feature
    clusters: Dict[int, List[int]]         # cluster id -> feature indices
    merges: List[MergeEvent] = field(default_factory=list)


def _condensed(D: np.ndarray) -> np.ndarray:
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return squareform(D, checks=False)


def average_linkage_labels(D: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage agglomerative clustering, cut into ``k`` clusters."""
    from scipy.cluster.hierarchy import fcluster

    p = D.shape[0]
    k = int(np.clip(k, 1, p))
    if p == 1:
        return np.zeros(1, dtype=int)
    Z = linkage(_condensed(D), method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels.astype(int) - 1


def agglomerate(D: np.ndarray, k: int, record_merges: bool = True) -> ClusteringResult:
    """Cluster features and optionally record the full merge trajectory.

    The merge trajectory is rebuilt from SciPy's linkage matrix so that each
    :class:`MergeEvent` carries the two clusters being joined and the linkage
    distance at which they merged.
    """
    p = D.shape[0]
    k = int(np.clip(k, 1, p))
    labels = average_linkage_labels(D, k)
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)

    merges: List[MergeEvent] = []
    if record_merges and p > 1:
        Z = linkage(_condensed(D), method="average")
        members: Dict[int, Tuple[int, ...]] = {i: (i,) for i in range(p)}
        for step, row in enumerate(Z):
            a, b, dist = int(row[0]), int(row[1]), float(row[2])
            left, right = members[a], members[b]
            merges.append(
                MergeEvent(step=step, left_cluster=left, right_cluster=right,
                           distance=dist)
            )
            members[p + step] = tuple(sorted(left + right))
    return ClusteringResult(labels=labels, clusters=clusters, merges=merges)


def composite_score(
    centrality: np.ndarray,
    relevance: np.ndarray,
    mi: np.ndarray,
    beta: Tuple[float, float, float],
) -> np.ndarray:
    """Composite prototype score ``S_i`` of Eq. (6)."""
    b1, b2, b3 = beta
    return (b1 * np.asarray(centrality, float)
            + b2 * np.asarray(relevance, float)
            + b3 * np.asarray(mi, float))


def select_representatives(
    clusters: Dict[int, List[int]],
    scores: np.ndarray,
) -> List[int]:
    """Pick the highest-scoring feature (prototype) from each cluster."""
    reps: List[int] = []
    for _, members in sorted(clusters.items()):
        best = max(members, key=lambda idx: scores[idx])
        reps.append(int(best))
    return sorted(reps)
