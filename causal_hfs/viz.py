"""Visualisation helpers - the causal dendrogram of Section 5.7 (Figure 6).

Leaves are coloured by each feature's causal strength toward the target; selected
representatives can be emphasised. Returns a Matplotlib figure so it renders both
in scripts and inside the Streamlit app.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def causal_dendrogram(
    distance_matrix: np.ndarray,
    relevance: np.ndarray,
    feature_names: Sequence[str],
    selected: Optional[Sequence[int]] = None,
    title: str = "Causal dendrogram",
):
    """Build a dendrogram coloured by causal strength.

    Parameters
    ----------
    distance_matrix : hybrid distance matrix (from a fitted framework).
    relevance : causal-relevance vector R (leaf colour).
    feature_names : names for the leaves.
    selected : indices of representative features to highlight in bold.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    D = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")

    fig, ax = plt.subplots(
        figsize=(5.0, max(2.4, len(feature_names) * 0.16)), dpi=110)
    dn = dendrogram(Z, labels=list(feature_names), orientation="right",
                    ax=ax, color_threshold=0, above_threshold_color="#888888")
    ax.tick_params(labelsize=7)

    # Colour leaf labels by causal strength.
    rmax = relevance.max() or 1.0
    norm = relevance / rmax
    selected = set(selected or [])
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for lbl in ax.get_ymajorticklabels():
        idx = name_to_idx.get(lbl.get_text())
        if idx is None:
            continue
        lbl.set_color(cm.viridis(norm[idx]))
        if idx in selected:
            lbl.set_fontweight("bold")
            lbl.set_fontsize(lbl.get_fontsize() + 1)

    sm = cm.ScalarMappable(cmap="viridis")
    sm.set_array(relevance)
    cb = fig.colorbar(sm, ax=ax, shrink=0.7)
    cb.set_label("causal strength to target", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("hybrid distance", fontsize=8)
    fig.tight_layout()
    return fig
