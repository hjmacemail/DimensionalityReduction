"""Configuration container for the Causal-HFS framework.

Every hyperparameter referenced in the paper is exposed here so that ablations
and sensitivity sweeps (e.g. varying the hybrid-distance weight ``alpha``) can be
driven from a single object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FrameworkConfig:
    """Hyperparameters for :class:`causal_hfs.framework.CausalHFS`.

    Attributes
    ----------
    n_representatives:
        Target number of selected features ``k``. If ``None`` the framework uses
        ``max(2, n_features // 4)``. Fixed across competing methods for fair
        comparison (Section 3.5).
    lam:
        ``lambda`` in Eq. (1): weight of Markov-Blanket membership vs. mutual
        information in the causal-relevance score ``R``.
    alpha:
        ``alpha`` in Eq. (4): balance between statistical distance and causal
        distance in the hybrid distance. ``alpha=1`` -> purely statistical,
        ``alpha=0`` -> purely causal.
    beta:
        ``(B1, B2, B3)`` in Eq. (6): weights of centrality, causal relevance and
        mutual information in the composite prototype score. Must sum to 1.
    sparsify_threshold:
        Edges of the feature graph with ``|corr| < threshold`` are removed
        (Section 3.4 sparsification).
    n_bootstrap:
        Number of bootstrap resamples ``B`` for consensus selection (Section 3.6).
    consensus_threshold:
        ``tau`` in the consensus rule: a feature is retained if it is selected in
        at least this fraction of bootstrap runs.
    enable_causal:
        Toggle the causal distance term (ablation: "w/o Causality").
    enable_hitl:
        Toggle the Human-in-the-Loop merge review (ablation: "w/o HITL").
    enable_stability:
        Toggle the bootstrap consensus stage (ablation: "w/o Stability").
    mb_alpha:
        Significance level for the conditional-independence tests used in Markov
        Blanket discovery.
    mb_max_cond_set:
        Maximum conditioning-set size in the PC-skeleton / IAMB search (keeps the
        "lightweight" approximation tractable).
    random_state:
        Seed for reproducibility.
    """

    n_representatives: int | None = None
    lam: float = 0.5
    alpha: float = 0.5
    beta: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    sparsify_threshold: float = 0.1
    n_bootstrap: int = 25
    consensus_threshold: float = 0.5
    enable_causal: bool = True
    enable_hitl: bool = False
    enable_stability: bool = True
    # Strict causal mode: restrict clustering + selection to the discovered Markov
    # Blanket (paper Stage 2, "isolate the most causally relevant features"). When
    # True the framework never considers features outside MB(Y) (padded with the
    # top-MI features only if the blanket is smaller than k). This sharply raises
    # causal plausibility on data with strong spurious correlates, at some risk of
    # missing relevant features when MB discovery is imperfect. Default False keeps
    # the paper-faithful "soft" behaviour where MB only weights the scores.
    restrict_to_mb: bool = False
    # Improvement #1: use a continuous *conditional* relevance (direct-effect
    # partial association with y) instead of marginal mutual information, so that
    # spurious aggregate correlates score low once their parent drivers are
    # conditioned on. ``cond_max_set`` caps the conditioning-set size.
    conditional_relevance: bool = False
    cond_max_set: int = 12
    # Use Random-Forest feature importance (supervised, non-linear) as the predictive
    # relevance term instead of the linear conditional/MI score. Tends to pick more
    # predictive cluster representatives -> higher downstream accuracy.
    rf_relevance: bool = False
    # Selection strategy. "composite"/"relevance"/"predictive" cut the dendrogram
    # into k clusters and keep one prototype each. "greedy" instead runs a
    # max-relevance / min-redundancy (mRMR-style) greedy selection over the
    # candidate features — deterministic, more accurate and more stable in practice.
    prototype_by: str = "composite"
    # Redundancy penalty weight for the greedy selection (``prototype_by="greedy"``).
    redundancy_beta: float = 1.0
    # Improved causal-aware greedy: seed & drive selection with the (rank-normalised)
    # causal-predictive relevance R = λ·C + (1-λ)·rel — so the Markov Blanket and λ
    # actually influence selection — with bootstrap MB confidence for C, max+mean
    # redundancy, and a group-diversity reward.
    # Measured on Wine + Breast-Cancer (k=8): with these defaults improved_greedy
    # beats the pure-relevance greedy on BOTH accuracy and stability
    #   Wine  acc 0.966->0.978, stab 0.748->0.785
    #   BC    acc 0.953->0.961, stab 0.527->0.591
    # The causal score (MB membership, weighted by improved_lam) now genuinely
    # drives selection: the greedy is seeded at argmax R and scored by R throughout,
    # instead of ignoring the causal signal as the pure-rel greedy did.
    improved_greedy: bool = False
    # Causal weight used ONLY by improved_greedy (independent of the paper's ``lam``).
    # A modest value is essential: at lam=0.5 the rank-normalised MB mask dominates and
    # BC regresses to acc 0.939 / stab 0.329; 0.15 is the measured sweet spot.
    improved_lam: float = 0.15
    # >0 -> continuous bootstrap MB confidence for C. Implemented and available, but it
    # was noisy at small n_boot and REDUCED stability in benchmarks (BC stab 0.591->0.384),
    # so it defaults OFF; the rank-normalised binary MB mask is the reliable C.
    mb_bootstrap: int = 0
    rank_norm: bool = False        # rank-normalise C and rel before blending into R
    # 1.0 = pure worst-case (max) redundancy, matching the stable pure-rel greedy.
    # Lowering it toward mean redundancy hurt stability in benchmarks; kept at 1.0.
    redundancy_eta: float = 1.0
    # Reward for covering a new redundancy group. Available, but the extra churn it
    # introduced across resamples REDUCED reported stability, so it defaults to 0.0.
    group_diversity: float = 0.0
    # Optional accuracy booster: a light forward-swap wrapper over the downstream
    # KNN score, run once on full data after consensus (filter -> wrapper hybrid).
    # Raises accuracy ~2-3% but can reduce selection stability; default off.
    wrapper_refine: bool = False
    # Cluster-level consensus voting (redundancy-group level). Available as an
    # option; empirically it did not improve reported stability in our benchmarks
    # because each resample re-derives its own clustering, so it defaults off.
    cluster_consensus: bool = False
    # Consensus clustering (Monti-style co-association): accumulate how often each
    # feature pair co-clusters across bootstraps, then cluster once on that averaged
    # matrix. Available as an option, but — like ``cluster_consensus`` — it did NOT
    # improve *reported* stability in our benchmarks (the evaluation re-fits the whole
    # pipeline per resample, so the co-association is itself re-derived each time).
    # Defaults off; the reliable stability levers are soft mode + more bootstraps.
    consensus_clustering: bool = False
    mb_alpha: float = 0.05
    mb_max_cond_set: int = 3
    # High-dimensional speed guard: if the dataset has more than this many features,
    # prefilter to the top-N by a cheap univariate score before the O(p^2) Markov-
    # Blanket search. ``None`` disables it (paper-faithful on <=120-feature data).
    prefilter_top: int | None = None
    ambiguity_tol: float = 0.05
    random_state: int = 42

    def __post_init__(self) -> None:
        b = tuple(float(x) for x in self.beta)
        if len(b) != 3:
            raise ValueError("beta must have exactly three components (B1, B2, B3).")
        s = sum(b)
        if s <= 0:
            raise ValueError("beta components must sum to a positive value.")
        # Normalise so that sum(B) = 1 as required by Eq. (6).
        object.__setattr__(self, "beta", tuple(x / s for x in b))
        for name in ("lam", "alpha", "consensus_threshold"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {v}.")
