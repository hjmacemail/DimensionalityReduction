"""Human-in-the-Loop extension (Sections 3.7 & 4.3).

Targeted HITL checkpoints are raised only during *causally ambiguous* merge
decisions - cases where competing cluster pairs exhibit statistically
indistinguishable hybrid distances or conflicting causal roles. A domain expert
(or, for automated evaluation, a rule-based oracle) approves or vetoes each
flagged merge, and the decisions feed back into the final selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clustering import MergeEvent


@dataclass
class AmbiguousMerge:
    """A merge flagged for expert review."""

    step: int
    left_cluster: Tuple[int, ...]
    right_cluster: Tuple[int, ...]
    distance: float
    next_distance: float                 # distance of the competing merge
    distance_gap: float                  # |distance - next_distance|
    causal_gap: float                    # |mean R(left) - mean R(right)|
    conflicting_causal: bool             # opposite MB membership across clusters
    decision: Optional[bool] = None      # True = approve, False = veto


def find_ambiguous_merges(
    merges: Sequence[MergeEvent],
    relevance: np.ndarray,
    mb_mask: np.ndarray,
    tol: float = 0.05,
) -> List[AmbiguousMerge]:
    """Flag merges whose linkage distance is within ``tol`` of the next merge.

    The "ambiguity" test compares each merge distance with the following merge
    distance; a small gap means two competing merges are statistically
    indistinguishable (Section 3.7). Causal context (relevance gap and conflicting
    Markov-Blanket membership) is attached for the expert.
    """
    flagged: List[AmbiguousMerge] = []
    dists = [m.distance for m in merges]
    span = (max(dists) - min(dists)) if len(dists) > 1 else 1.0
    span = span or 1.0
    for i, m in enumerate(merges):
        nxt = merges[i + 1].distance if i + 1 < len(merges) else m.distance
        gap = abs(m.distance - nxt) / span
        r_left = float(np.mean([relevance[x] for x in m.left_cluster]))
        r_right = float(np.mean([relevance[x] for x in m.right_cluster]))
        causal_gap = abs(r_left - r_right)
        mb_left = any(mb_mask[x] > 0 for x in m.left_cluster)
        mb_right = any(mb_mask[x] > 0 for x in m.right_cluster)
        conflicting = mb_left != mb_right
        if gap <= tol or conflicting:
            flagged.append(
                AmbiguousMerge(
                    step=m.step,
                    left_cluster=m.left_cluster,
                    right_cluster=m.right_cluster,
                    distance=m.distance,
                    next_distance=nxt,
                    distance_gap=gap,
                    causal_gap=causal_gap,
                    conflicting_causal=conflicting,
                )
            )
    return flagged


def rule_based_oracle(causal_conflict_veto: bool = True,
                      causal_gap_threshold: float = 0.4) -> Callable[[AmbiguousMerge], bool]:
    """Return an oracle that emulates a domain expert (Section 4.3, 5.6).

    Heuristic: **veto** (reject) a merge when the two clusters play conflicting
    causal roles, i.e. one is inside the Markov Blanket and the other is not, or
    when their causal-relevance gap is large. Otherwise **approve**.

    Returns a callable mapping an :class:`AmbiguousMerge` to ``True`` (approve) or
    ``False`` (veto).
    """
    def oracle(m: AmbiguousMerge) -> bool:
        if causal_conflict_veto and m.conflicting_causal:
            return False
        if m.causal_gap >= causal_gap_threshold:
            return False
        return True
    return oracle


@dataclass
class HITLSession:
    """Collects and applies expert decisions over flagged merges."""

    decisions: Dict[int, bool] = field(default_factory=dict)  # step -> approve?
    log: List[AmbiguousMerge] = field(default_factory=list)

    def resolve(
        self,
        flagged: Sequence[AmbiguousMerge],
        decide: Callable[[AmbiguousMerge], bool],
    ) -> "HITLSession":
        """Apply a decision function (expert UI callback or oracle) to each flag."""
        for m in flagged:
            d = bool(decide(m))
            m.decision = d
            self.decisions[m.step] = d
            self.log.append(m)
        return self

    @property
    def veto_rate(self) -> float:
        if not self.log:
            return 0.0
        vetoes = sum(1 for m in self.log if m.decision is False)
        return vetoes / len(self.log)

    def vetoed_pairs(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Cluster pairs the expert refused to merge."""
        return [(m.left_cluster, m.right_cluster)
                for m in self.log if m.decision is False]
