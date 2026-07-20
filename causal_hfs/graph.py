"""Stage 3 - Structural Mapping: weighted feature graph (Section 3.4).

Nodes are features; edge weights are the absolute Pearson correlation between
feature pairs (Eq. 2). A sparsification threshold removes weak/noisy edges so the
graph captures only the dominant dependency structure.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation matrix ``|corr(f_i, f_j)|`` (Eq. 2)."""
    Xc = np.asarray(X, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xc, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)  # constant features -> zero correlation
    np.fill_diagonal(C, 1.0)
    return np.abs(C)


def build_feature_graph(X: np.ndarray, sparsify_threshold: float = 0.1) -> nx.Graph:
    """Construct the sparsified weighted feature graph.

    Edges with ``|corr| < sparsify_threshold`` are dropped (Section 3.4).
    """
    W = correlation_matrix(X)
    p = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(p))
    iu = np.triu_indices(p, k=1)
    for i, j in zip(*iu):
        w = W[i, j]
        if w >= sparsify_threshold:
            G.add_edge(int(i), int(j), weight=float(w))
    return G


def graph_centrality(G: nx.Graph, n_features: int) -> np.ndarray:
    """Weighted-degree centrality of each feature node, normalised to [0, 1].

    Used as the "cluster centrality" term in the composite prototype score
    (Eq. 6): structurally central features aggregate strong correlations with
    their neighbours.
    """
    cent = np.zeros(n_features, dtype=float)
    for node in G.nodes():
        cent[node] = sum(d.get("weight", 0.0) for _, _, d in G.edges(node, data=True))
    m = cent.max()
    return cent / m if m > 0 else cent
