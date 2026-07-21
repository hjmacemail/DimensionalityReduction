"""The Causal-HFS orchestrator: wires the seven pipeline stages together.

Usage
-----
>>> from causal_hfs import CausalHFS, FrameworkConfig
>>> model = CausalHFS(FrameworkConfig(n_representatives=10, enable_hitl=True))
>>> model.fit(X, y)
>>> X_reduced = model.transform(X)
>>> model.selected_features_          # indices of chosen representatives
>>> model.stability_                  # Jaccard stability across bootstraps
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .config import FrameworkConfig
from .preprocessing import Preprocessor
from .causal import CausalAnalyzer
from .graph import build_feature_graph, graph_centrality
from .distance import hybrid_distance
from .clustering import (
    agglomerate,
    average_linkage_labels,
    composite_score,
    greedy_select,
    select_representatives,
)
from .graph import correlation_matrix
from .consensus import (
    cluster_consensus_selection,
    consensus_selection,
    stability_index,
)
from .hitl import (
    HITLSession,
    find_ambiguous_merges,
    rule_based_oracle,
)


class CausalHFS:
    """Causality-Aware Stable Hierarchical Feature Selector.

    Parameters
    ----------
    config:
        A :class:`FrameworkConfig`. If ``None`` the defaults are used.
    hitl_callback:
        Optional callable used to resolve flagged ambiguous merges. It receives an
        :class:`~causal_hfs.hitl.AmbiguousMerge` and returns ``True`` (approve) or
        ``False`` (veto). When ``None`` and HITL is enabled, the built-in
        rule-based oracle is used (Section 4.3 / 5.6).
    discrete_target:
        Whether the target is categorical (classification) - controls the MI
        estimator.
    """

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        hitl_callback: Optional[Callable] = None,
        discrete_target: bool = True,
        step_callback: Optional[Callable] = None,
    ) -> None:
        self.config = config or FrameworkConfig()
        self.hitl_callback = hitl_callback
        self.discrete_target = discrete_target
        # ``step_callback(entry)`` (optional) is invoked live as each pipeline
        # stage completes on the full-data pass; ``trace_`` stores the same records.
        self.step_callback = step_callback
        self.trace_: List[dict] = []
        # Fitted attributes
        self.selected_features_: List[int] = []
        self.stability_: float = 0.0
        self.markov_blanket_: List[int] = []
        self.relevance_: Optional[np.ndarray] = None
        self.mutual_info_: Optional[np.ndarray] = None
        self.distance_matrix_: Optional[np.ndarray] = None
        self.clustering_ = None
        self.hitl_session_: Optional[HITLSession] = None
        self.candidate_features_: List[int] = []
        self._full_labels: dict = {}
        self._cluster_reps: dict = {}
        self._preprocessor: Optional[Preprocessor] = None
        self._n_features: int = 0

    # ------------------------------------------------------------------ #
    # Algorithm description & live tracing
    # ------------------------------------------------------------------ #
    @staticmethod
    def describe_pipeline():
        """Static description of the algorithm's stages (for 'show the steps')."""
        return [
            {"stage": "1. Preprocessing",
             "latex": r"z = \dfrac{x - \mu}{\sigma}",
             "detail": "Median-impute missing values, z-score standardise, and view "
                       "each feature as a point in sample space."},
            {"stage": "2. Causal discovery & relevance",
             "latex": r"R_i = \lambda\, C_i + (1-\lambda)\,\mathrm{rel}_i",
             "detail": "Discover the Markov Blanket MB(Y) (IAMB + partial-correlation "
                       "CI tests) → binary priority C_i. Combine with a predictive "
                       "score rel_i (marginal MI, or conditional/direct relevance)."},
            {"stage": "3. Structural mapping",
             "latex": r"w_{ij} = \lvert \mathrm{corr}(f_i, f_j) \rvert",
             "detail": "Build a sparsified weighted feature graph capturing "
                       "redundancy among variables."},
            {"stage": "4. Hybrid distance",
             "latex": r"d = \alpha\, d_{\mathrm{stat}} + (1-\alpha)\, d_{\mathrm{causal}}",
             "detail": "Combine statistical distance (1−|corr|) with a causal "
                       "distance derived from R."},
            {"stage": "5. Hierarchical agglomeration",
             "latex": r"\text{average linkage} \;\rightarrow\; k\ \text{clusters}",
             "detail": "Group redundant features while preserving their identity."},
            {"stage": "6. Representative extraction",
             "latex": r"\arg\max_{i}\; S_i \quad (\text{composite or relevance})",
             "detail": "Pick one prototype per cluster — the causal driver of its "
                       "redundancy group."},
            {"stage": "7. Consensus stability",
             "latex": r"\text{keep } f \text{ if } \mathrm{freq} \ge \tau \;;\quad \text{Jaccard stability}",
             "detail": "Repeat over B bootstraps and keep features selected "
                       "consistently; report Jaccard stability."},
        ]

    def _log(self, stage: str, detail: str, **data) -> None:
        entry = {"stage": stage, "detail": detail, "data": data}
        self.trace_.append(entry)
        if self.step_callback is not None:
            try:
                self.step_callback(entry)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Core single-pass selection (used directly and inside bootstrapping)
    # ------------------------------------------------------------------ #
    def _select_once(self, X: np.ndarray, y: np.ndarray, record: bool = False):
        """Run stages 1-6 on a single (already-sized) dataset.

        Returns the selected representative indices. When ``record`` is True the
        intermediate artefacts (distance matrix, clustering, HITL session) are
        stashed on ``self`` for inspection/plotting.
        """
        cfg = self.config
        k = cfg.n_representatives or max(2, X.shape[1] // 4)

        if record:
            self.trace_ = []

        # Stage 1 - preprocessing
        Xp = Preprocessor().fit_transform(X)
        if record:
            self._log("1. Preprocessing",
                      f"Imputed + z-scored {X.shape[1]} features over {X.shape[0]} samples.",
                      n_features=int(X.shape[1]), n_samples=int(X.shape[0]))

        # High-dimensional prefilter: on very wide data, keep the top-N features by a
        # cheap univariate score before the O(p^2) Markov-Blanket search. ``pf`` maps
        # prefiltered positions back to original feature indices (identity if off).
        pf = np.arange(X.shape[1])
        if cfg.prefilter_top and X.shape[1] > cfg.prefilter_top:
            try:
                from sklearn.feature_selection import f_classif
                fscore = np.nan_to_num(f_classif(Xp, y)[0])
            except Exception:
                fscore = Xp.var(axis=0)
            pf = np.sort(np.argsort(fscore)[::-1][: cfg.prefilter_top]).astype(int)
            Xp = Xp[:, pf]
            if record:
                self._log("1b. High-dim prefilter",
                          f"Reduced {X.shape[1]} → {len(pf)} features by univariate "
                          f"relevance before causal discovery.", n_kept=int(len(pf)))

        # Stage 2 - causal discovery & relevance
        analyzer = CausalAnalyzer(
            alpha=cfg.mb_alpha,
            max_cond_set=cfg.mb_max_cond_set,
            discrete_target=self.discrete_target,
            random_state=cfg.random_state,
            use_conditional=cfg.conditional_relevance,
            cond_max_set=cfg.cond_max_set,
            rf_relevance=cfg.rf_relevance,
        ).fit(Xp, y, lam=cfg.lam)
        R = analyzer.relevance_
        mi = analyzer.mi_
        pred = analyzer.predictive_          # mi, or conditional relevance if enabled
        mb_mask = analyzer.mb_mask_
        if record:
            rel_kind = "conditional/direct" if cfg.conditional_relevance else "marginal MI"
            self._log("2. Causal discovery & relevance",
                      f"Markov Blanket of size {len(analyzer.mb_)}: "
                      f"{analyzer.mb_[:12]}. Relevance uses {rel_kind}.",
                      mb=list(analyzer.mb_), mb_size=len(analyzer.mb_), relevance_kind=rel_kind)

        # Strict causal mode - isolate the Markov Blanket before clustering.
        # ``cand`` indexes into the (possibly prefiltered) working space; ``pf``
        # then maps those positions back to original feature indices.
        cand = np.arange(Xp.shape[1])
        if cfg.restrict_to_mb and len(analyzer.mb_) >= 2:
            pool = list(analyzer.mb_)
            if len(pool) < k:  # pad with the highest-MI non-MB features
                extra = [int(i) for i in np.argsort(mi)[::-1] if int(i) not in pool]
                pool = pool + extra[: k - len(pool)]
            cand = np.array(sorted(pool), dtype=int)
        Xp_c = Xp[:, cand]
        R_c, mi_c, mb_mask_c, pred_c = R[cand], mi[cand], mb_mask[cand], pred[cand]
        if record and cfg.restrict_to_mb:
            self._log("2b. Markov-Blanket isolation",
                      f"Restricted candidate pool to {len(cand)} features "
                      f"(strict causal mode).", n_candidates=int(len(cand)))

        # Stage 3 - structural mapping (on the candidate subset)
        G = build_feature_graph(Xp_c, sparsify_threshold=cfg.sparsify_threshold)
        centrality_c = graph_centrality(G, Xp_c.shape[1])
        if record:
            self._log("3. Structural mapping",
                      f"Built feature graph: {G.number_of_edges()} edges after "
                      f"sparsification (threshold {cfg.sparsify_threshold}).",
                      n_edges=int(G.number_of_edges()))

        # Stage 4 - hybrid distance
        D = hybrid_distance(Xp_c, R_c, alpha=cfg.alpha, enable_causal=cfg.enable_causal)
        if record:
            self._log("4. Hybrid distance",
                      f"Computed hybrid distance (α={cfg.alpha}: "
                      f"{cfg.alpha:.0%} statistical / {1-cfg.alpha:.0%} causal).",
                      alpha=cfg.alpha)

        # Stage 5 - agglomeration (record merges for HITL / plotting)
        k_eff = int(min(k, Xp_c.shape[1]))
        result = agglomerate(D, k_eff, record_merges=(record or cfg.enable_hitl))
        if record:
            sizes = sorted((len(v) for v in result.clusters.values()), reverse=True)
            self._log("5. Hierarchical agglomeration",
                      f"Average-linkage clustering into {k_eff} groups; "
                      f"cluster sizes {sizes[:8]}.", n_clusters=int(k_eff))

        # HITL - review causally ambiguous merges and veto some
        session: Optional[HITLSession] = None
        if cfg.enable_hitl and result.merges:
            flagged = find_ambiguous_merges(
                result.merges, R_c, mb_mask_c, tol=cfg.ambiguity_tol
            )
            decide = self.hitl_callback or rule_based_oracle()
            session = HITLSession().resolve(flagged, decide)
            if record:
                self._log("5b. Human-in-the-Loop review",
                          f"Reviewed {len(flagged)} ambiguous merges; "
                          f"{len(session.vetoed_pairs())} vetoed.",
                          flagged=len(flagged), vetoes=len(session.vetoed_pairs()))
            if session.vetoed_pairs():
                D = self._apply_vetoes(D, session)
                result = agglomerate(D, k_eff, record_merges=record)

        # Stage 6 - prototype selection, then map local indices back to originals.
        # ``prototype_by="relevance"`` picks the most causally-relevant cluster
        # member (Improvement #2); the default composite also weights centrality/MI.
        if cfg.prototype_by == "greedy":
            # Greedy max-relevance / min-redundancy (mRMR-style) over candidates.
            C_c = correlation_matrix(Xp_c)
            reps_local = greedy_select(pred_c, C_c, k_eff, cfg.redundancy_beta)
        elif cfg.prototype_by == "relevance":
            reps_local = select_representatives(result.clusters, R_c)
        elif cfg.prototype_by == "predictive":
            reps_local = select_representatives(result.clusters, pred_c)
        else:
            score_vec = composite_score(centrality_c, R_c, pred_c, cfg.beta)
            reps_local = select_representatives(result.clusters, score_vec)
        # Map: local cluster index -> working (prefiltered) index -> original index.
        reps = sorted(int(pf[cand[i]]) for i in reps_local)
        if record:
            by = "causal relevance" if cfg.prototype_by == "relevance" else "composite score"
            self._log("6. Representative extraction",
                      f"Selected {len(reps)} prototypes by {by}: {reps}.",
                      selected=list(reps))

        if record:
            # Expand relevance / MI back to the full feature space (0 outside prefilter).
            R_full = np.zeros(X.shape[1]); R_full[pf] = R
            mi_full = np.zeros(X.shape[1]); mi_full[pf] = mi
            self.markov_blanket_ = [int(pf[i]) for i in analyzer.mb_]
            self.relevance_ = R_full
            self.mutual_info_ = mi_full
            self.distance_matrix_ = D          # rows aligned with candidate_features_
            self.candidate_features_ = [int(pf[c]) for c in cand]
            self.clustering_ = result
            self.hitl_session_ = session
            # Redundancy-group labels + canonical representatives (original indices)
            # for the optional cluster-level consensus.
            self._full_labels = {int(pf[cand[li]]): int(cid)
                                 for li, cid in enumerate(result.labels)}
            self._cluster_reps = {int(result.labels[rl]): int(pf[cand[rl]])
                                  for rl in reps_local}
        return reps

    @staticmethod
    def _apply_vetoes(D: np.ndarray, session: HITLSession) -> np.ndarray:
        """Push apart clusters an expert refused to merge.

        The pairwise distances between the two vetoed groups are set to the matrix
        maximum so average-linkage will not join them early.
        """
        D = D.copy()
        big = D.max() * 10.0 + 1.0
        for left, right in session.vetoed_pairs():
            for i in left:
                for j in right:
                    D[i, j] = big
                    D[j, i] = big
        np.fill_diagonal(D, 0.0)
        return D

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CausalHFS":
        """Fit the selector, including bootstrap consensus (Stage 7) if enabled."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._n_features = X.shape[1]
        cfg = self.config
        k = cfg.n_representatives or max(2, X.shape[1] // 4)

        # Fit the standardiser on the full data so transform() returns z-scored
        # representative columns (keeps scale-sensitive downstream models, e.g.
        # KNN, on the same footing as the projection baselines).
        self._preprocessor = Preprocessor()
        self._preprocessor.fit_transform(X)

        # Full-data pass (records artefacts for plotting / HITL / case study).
        full_selection = self._select_once(X, y, record=True)

        if not cfg.enable_stability:
            self.selected_features_ = sorted(full_selection)
            self.stability_ = 1.0
            return self

        # Consensus clustering (co-association) — a more stable alternative to
        # re-clustering + voting on exact indices each resample.
        if cfg.consensus_clustering and self.candidate_features_:
            self._consensus_clustering_fit(X, y, k)
            if cfg.wrapper_refine:
                self.selected_features_ = self._wrapper_refine(X, y, self.selected_features_)
            return self

        # Stage 7 - bootstrap consensus
        rng = np.random.default_rng(cfg.random_state)
        n = X.shape[0]
        selections: List[List[int]] = []
        for _ in range(cfg.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            sel = self._select_once(X[idx], y[idx], record=False)
            selections.append(sel)

        if cfg.cluster_consensus and self._full_labels:
            self.selected_features_ = cluster_consensus_selection(
                selections, self._full_labels, self._cluster_reps,
                tau=cfg.consensus_threshold, k=k)
            cl = [sorted(set(self._full_labels.get(f, -(f + 1)) for f in sel))
                  for sel in selections]
            self.stability_ = stability_index(cl)
        else:
            self.stability_ = stability_index(selections)
            self.selected_features_ = consensus_selection(
                selections, tau=cfg.consensus_threshold, k=k)
        if not self.selected_features_:
            self.selected_features_ = sorted(full_selection)
        self._log("7. Consensus stability",
                  f"Voted over {cfg.n_bootstrap} bootstraps (τ={cfg.consensus_threshold}); "
                  f"Jaccard stability {self.stability_:.3f}. "
                  f"Final subset: {self.selected_features_}.",
                  stability=float(self.stability_), selected=list(self.selected_features_))

        # Optional filter -> wrapper refinement to boost downstream accuracy.
        if cfg.wrapper_refine:
            self.selected_features_ = self._wrapper_refine(X, y, self.selected_features_)
            self._log("8. Wrapper refinement",
                      f"Forward-swap refinement over the KNN score; "
                      f"final subset: {self.selected_features_}.",
                      selected=list(self.selected_features_))
        return self

    def _wrapper_refine(self, X, y, selected, max_iter: int = 2):
        """Greedy forward-swap over the downstream KNN CV score (accuracy booster).

        Swaps each selected feature for a non-selected candidate whenever it
        improves 3-fold KNN accuracy. Runs once on full data; the candidate pool is
        the discovered Markov Blanket (or all features), keeping it cheap.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        Xp = self._preprocessor.transform(X)
        selected = list(selected)
        pool = list(self.candidate_features_) or list(range(X.shape[1]))

        def score(feats):
            feats = list(feats)
            if len(feats) < 1:
                return -1.0
            try:
                clf = KNeighborsClassifier(n_neighbors=min(5, len(y) - 1))
                return float(cross_val_score(clf, Xp[:, feats], y, cv=3).mean())
            except Exception:
                return -1.0

        best = score(selected)
        improved, it = True, 0
        while improved and it < max_iter:
            improved, it = False, it + 1
            for i in range(len(selected)):
                for g in pool:
                    if g in selected:
                        continue
                    trial = list(selected)
                    trial[i] = g
                    s = score(trial)
                    if s > best + 1e-9:
                        selected, best, improved = trial, s, True
        return sorted(set(selected))

    def _consensus_clustering_fit(self, X, y, k):
        """Co-association consensus clustering (Monti-style) for a stable partition.

        Over the FIXED candidate feature set, accumulate how often each feature pair
        lands in the same cluster across bootstrap resamples, then cluster once on the
        averaged co-association matrix. Averaging removes per-resample clustering
        noise, so the final partition (and its representatives) is far more stable.
        """
        cand = np.array(self.candidate_features_, dtype=int)
        R = self.relevance_
        R_c = R[cand]
        p_c = len(cand)
        k_eff = int(min(k, p_c))
        rng = np.random.default_rng(self.config.random_state)
        n = X.shape[0]

        coassoc = np.zeros((p_c, p_c))
        for _ in range(self.config.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            Xb = Preprocessor().fit_transform(X[idx])[:, cand]
            D = hybrid_distance(Xb, R_c, alpha=self.config.alpha,
                                enable_causal=self.config.enable_causal)
            labels = average_linkage_labels(D, k_eff)
            coassoc += (labels[:, None] == labels[None, :]).astype(float)
        coassoc /= max(1, self.config.n_bootstrap)

        Dco = 1.0 - coassoc
        np.fill_diagonal(Dco, 0.0)
        cons_labels = average_linkage_labels(Dco, k_eff)
        clusters = {}
        for i, lab in enumerate(cons_labels):
            clusters.setdefault(int(lab), []).append(i)

        reps_local = select_representatives(clusters, R_c)
        self.selected_features_ = sorted(int(cand[i]) for i in reps_local)

        # Report the mean within-cluster co-association as a partition-stability score.
        within = []
        for members in clusters.values():
            if len(members) > 1:
                sub = coassoc[np.ix_(members, members)]
                within.append(sub[np.triu_indices(len(members), 1)].mean())
        self.stability_ = float(np.mean(within)) if within else 1.0
        self._log("7. Consensus clustering",
                  f"Averaged co-association over {self.config.n_bootstrap} bootstraps; "
                  f"partition stability {self.stability_:.3f}. "
                  f"Final subset: {self.selected_features_}.",
                  stability=float(self.stability_), selected=list(self.selected_features_))
        return self

    def transform(self, X: np.ndarray, standardize: bool = True) -> np.ndarray:
        """Project ``X`` onto the selected representative features.

        By default the same z-score standardisation fitted during ``fit`` is
        applied before column selection. Set ``standardize=False`` to recover the
        original-scale values of the selected features (their identity and units
        are preserved either way).
        """
        if not self.selected_features_:
            raise RuntimeError("CausalHFS must be fitted before transform().")
        X = np.asarray(X, dtype=float)
        if standardize and getattr(self, "_preprocessor", None) is not None:
            X = self._preprocessor.transform(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # Convenience -------------------------------------------------------- #
    @property
    def veto_rate_(self) -> float:
        return self.hitl_session_.veto_rate if self.hitl_session_ else 0.0

    def get_support(self) -> np.ndarray:
        """Boolean mask of selected features (sklearn-style)."""
        mask = np.zeros(self._n_features, dtype=bool)
        mask[self.selected_features_] = True
        return mask
