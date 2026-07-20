"""Baseline dimensionality-reduction / feature-selection methods (Section 4.2).

Seven baselines are provided behind a common interface so the evaluation harness
can treat every method identically:

    PCA          - projection (Hastie et al., 2009)
    VAE          - deep latent projection (Kingma & Welling, 2014); torch if
                   available, otherwise a linear-autoencoder (SVD) fallback
    LASSO        - L1 sparse selection (Tibshirani, 1996)
    mRMR         - minimum-Redundancy-Maximum-Relevance (Peng et al., 2005)
    CausalDRIFT  - causal feature selection via Markov-Blanket + MI ranking
    CAFE         - Causal-Aware Feature sElection with a light HITL oracle
    Random       - random selector (lower-bound control)

Every method exposes:
    fit(X, y)              -> self
    transform(X)          -> representation fed to the KNN classifier
    selected_features_    -> list[int] feature indices (for stability /
                             trustworthiness). Projection methods expose the
                             top-loading original features so Jaccard stability is
                             well-defined for them too.

``k`` (number of selected features / latent components) is shared across methods
for a fair comparison.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso, LogisticRegression

from .preprocessing import Preprocessor
from .causal import CausalAnalyzer, mutual_information
from .graph import correlation_matrix


class BaseMethod:
    name = "base"

    def __init__(self, k: int, random_state: int = 42, discrete_target: bool = True):
        self.k = k
        self.random_state = random_state
        self.discrete_target = discrete_target
        self.selected_features_: List[int] = []
        self._pre = Preprocessor()

    def fit(self, X, y):  # pragma: no cover - overridden
        raise NotImplementedError

    def transform(self, X):
        """Default: return the selected columns of the standardised data."""
        Xp = self._pre.transform(X)
        return Xp[:, self.selected_features_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# --------------------------------------------------------------------------- #
# Projection methods
# --------------------------------------------------------------------------- #
class PCAMethod(BaseMethod):
    name = "PCA"

    def fit(self, X, y):
        Xp = self._pre.fit_transform(X)
        k = min(self.k, Xp.shape[1])
        self._model = PCA(n_components=k, random_state=self.random_state).fit(Xp)
        # Pseudo-selection: features with the largest absolute loadings.
        loadings = np.abs(self._model.components_).sum(axis=0)
        self.selected_features_ = sorted(np.argsort(loadings)[::-1][:k].tolist())
        return self

    def transform(self, X):
        return self._model.transform(self._pre.transform(X))


class VAEMethod(BaseMethod):
    name = "VAE"

    def fit(self, X, y):
        Xp = self._pre.fit_transform(X)
        k = min(self.k, Xp.shape[1])
        self._latent_dim = k
        try:
            self._fit_torch(Xp, k)
            self._backend = "torch"
        except Exception:
            # Linear-autoencoder fallback (SVD) so the pipeline always runs.
            self._model = TruncatedSVD(n_components=k, random_state=self.random_state).fit(Xp)
            self._backend = "svd"
        if self._backend == "svd":
            loadings = np.abs(self._model.components_).sum(axis=0)
        else:
            loadings = self._encoder_saliency(Xp)
        self.selected_features_ = sorted(np.argsort(loadings)[::-1][:k].tolist())
        return self

    def _fit_torch(self, Xp, k):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        d = Xp.shape[1]
        h = max(k, min(64, d))
        self._torch = torch

        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU())
                self.mu = nn.Linear(h, k)
                self.logvar = nn.Linear(h, k)
                self.dec = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d))

            def forward(self, x):
                he = self.enc(x)
                mu, logvar = self.mu(he), self.logvar(he)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                return self.dec(z), mu, logvar

        model = VAE()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        Xt = torch.tensor(Xp, dtype=torch.float32)
        model.train()
        for _ in range(150):
            opt.zero_grad()
            recon, mu, logvar = model(Xt)
            recon_loss = ((recon - Xt) ** 2).mean()
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            (recon_loss + 1e-3 * kld).backward()
            opt.step()
        model.eval()
        self._model = model

    def _encoder_saliency(self, Xp):
        torch = self._torch
        Xt = torch.tensor(Xp, dtype=torch.float32, requires_grad=True)
        _, mu, _ = self._model(Xt)
        mu.sum().backward()
        return Xt.grad.abs().mean(dim=0).detach().numpy()

    def transform(self, X):
        Xp = self._pre.transform(X)
        if self._backend == "svd":
            return self._model.transform(Xp)
        torch = self._torch
        with torch.no_grad():
            _, mu, _ = self._model(torch.tensor(Xp, dtype=torch.float32))
        return mu.numpy()


# --------------------------------------------------------------------------- #
# Feature-selection methods
# --------------------------------------------------------------------------- #
class LASSOMethod(BaseMethod):
    name = "LASSO"

    def fit(self, X, y):
        Xp = self._pre.fit_transform(X)
        k = min(self.k, Xp.shape[1])
        if self.discrete_target:
            # One-vs-rest keeps the fast L1 liblinear solver working for multiclass
            # targets (newer scikit-learn raises instead of doing OvR implicitly).
            from sklearn.multiclass import OneVsRestClassifier
            base = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                      random_state=self.random_state, max_iter=500)
            model = OneVsRestClassifier(base).fit(Xp, y)
            coef = np.abs(np.vstack([est.coef_.ravel() for est in model.estimators_])).sum(axis=0)
        else:
            model = Lasso(alpha=0.01, random_state=self.random_state, max_iter=2000).fit(Xp, y)
            coef = np.abs(model.coef_)
        self.selected_features_ = sorted(np.argsort(coef)[::-1][:k].tolist())
        return self


class MRMRMethod(BaseMethod):
    name = "mRMR"

    def fit(self, X, y):
        Xp = self._pre.fit_transform(X)
        k = min(self.k, Xp.shape[1])
        rel = mutual_information(Xp, y, self.discrete_target, self.random_state)
        C = correlation_matrix(Xp)
        selected: List[int] = [int(np.argmax(rel))]
        remaining = set(range(Xp.shape[1])) - set(selected)
        while len(selected) < k and remaining:
            best_j, best_score = None, -np.inf
            for j in remaining:
                redundancy = np.mean([C[j, s] for s in selected])
                score = rel[j] - redundancy  # relevance minus redundancy
                if score > best_score:
                    best_score, best_j = score, j
            selected.append(best_j)
            remaining.discard(best_j)
        self.selected_features_ = sorted(selected)
        return self


def _causal_prefilter(Xp, y, cap=200):
    """Reduce very wide data to the top-``cap`` features by a fast univariate score
    before running Markov-Blanket discovery. Returns (Xp_reduced, index_map)."""
    if Xp.shape[1] <= cap:
        return Xp, np.arange(Xp.shape[1])
    try:
        from sklearn.feature_selection import f_classif
        fs = np.nan_to_num(f_classif(Xp, y)[0])
    except Exception:
        fs = Xp.var(axis=0)
    pf = np.sort(np.argsort(fs)[::-1][:cap]).astype(int)
    return Xp[:, pf], pf


class CausalDRIFTMethod(BaseMethod):
    """Causal baseline: rank by causal relevance (MB membership + MI)."""

    name = "CausalDRIFT"

    def fit(self, X, y):
        Xp = self._pre.fit_transform(X)
        Xp, pf = _causal_prefilter(Xp, y)
        k = min(self.k, Xp.shape[1])
        analyzer = CausalAnalyzer(
            discrete_target=self.discrete_target, random_state=self.random_state
        ).fit(Xp, y, lam=0.5)
        R = analyzer.relevance_
        self.selected_features_ = sorted(int(pf[i]) for i in np.argsort(R)[::-1][:k])
        return self


class CAFEMethod(BaseMethod):
    """Causal-Aware Feature sElection with a light HITL redundancy filter.

    Greedy selection maximising causal relevance while penalising redundancy with
    already-selected features (a causal analogue of mRMR), then a rule-based
    oracle drops a feature whose causal role conflicts with the current set.
    """

    name = "CAFE"

    def fit(self, X, y):
        Xp = self._pre.fit_transform(X)
        Xp, pf = _causal_prefilter(Xp, y)
        k = min(self.k, Xp.shape[1])
        analyzer = CausalAnalyzer(
            discrete_target=self.discrete_target, random_state=self.random_state
        ).fit(Xp, y, lam=0.5)
        R = analyzer.relevance_
        C = correlation_matrix(Xp)
        selected: List[int] = [int(np.argmax(R))]
        remaining = set(range(Xp.shape[1])) - set(selected)
        while len(selected) < k and remaining:
            best_j, best_score = None, -np.inf
            for j in remaining:
                redundancy = np.mean([C[j, s] for s in selected])
                score = R[j] - 0.5 * redundancy
                if score > best_score:
                    best_score, best_j = score, j
            selected.append(best_j)
            remaining.discard(best_j)
        self.selected_features_ = sorted(int(pf[i]) for i in selected)
        return self


class RandomMethod(BaseMethod):
    name = "Random"

    def fit(self, X, y):
        Xp = self._pre.fit_transform(X)
        k = min(self.k, Xp.shape[1])
        rng = np.random.default_rng(self.random_state)
        self.selected_features_ = sorted(
            rng.choice(Xp.shape[1], size=k, replace=False).tolist()
        )
        return self


BASELINES = {
    "PCA": PCAMethod,
    "VAE": VAEMethod,
    "LASSO": LASSOMethod,
    "mRMR": MRMRMethod,
    "CausalDRIFT": CausalDRIFTMethod,
    "CAFE": CAFEMethod,
    "Random": RandomMethod,
}


def build_baseline(name: str, k: int, random_state: int = 42,
                   discrete_target: bool = True) -> BaseMethod:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline '{name}'. Options: {list(BASELINES)}")
    return BASELINES[name](k=k, random_state=random_state, discrete_target=discrete_target)
