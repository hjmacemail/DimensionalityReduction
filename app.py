"""Causal-HFS platform — interactive Streamlit app.

Two views:

1. **Experiments** (default): choose one or more datasets in the left sidebar,
   click *Run experiment*, and compare the proposed framework against every
   baseline (PCA, VAE, LASSO, mRMR, CausalDRIFT, CAFE, Random). The best result
   in each metric is highlighted.
2. **Human-in-the-Loop**: review causally ambiguous merges for a single dataset
   and fold expert approve/veto decisions into the final selection (Section 4.3).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from causal_hfs import CausalHFS, FrameworkConfig
from causal_hfs.baselines import BASELINES, build_baseline
from causal_hfs.causal import CausalAnalyzer
from causal_hfs.clustering import agglomerate
from causal_hfs.distance import hybrid_distance
from causal_hfs.graph import build_feature_graph, graph_centrality, correlation_matrix
from causal_hfs.hitl import find_ambiguous_merges, rule_based_oracle
from causal_hfs.preprocessing import Preprocessor
from causal_hfs.evaluation import (
    evaluate_method,
    friedman_test,
    knn_accuracy,
    wilcoxon_vs_baselines,
)
from causal_hfs.datasets import (
    DATASET_REGISTRY,
    Dataset,
    apply_common_schema,
    load_csv,
    load_dataset,
)
from causal_hfs.viz import causal_dendrogram

st.set_page_config(page_title="Causal-HFS Platform", layout="wide")

METHOD_ORDER = ["Proposed"] + list(BASELINES)
HIGHER_BETTER = ("accuracy", "stability", "trustworthiness")


# --------------------------------------------------------------------------- #
# Dataset catalogue (offline sklearn built-ins + synthetic + UCI/OpenML)
# --------------------------------------------------------------------------- #
def _sklearn_builtin(loader_name, name, n_classes, mf=120, ms=600):
    from sklearn import datasets as skd

    loader = getattr(skd, loader_name)
    d = loader()
    X, y, names = apply_common_schema(
        d.data, d.target, list(getattr(d, "feature_names", [])) or None,
        max_features=mf, max_samples=ms,
    )
    return Dataset(name, X, y, names, n_classes)


def _olivetti_faces():
    """Olivetti faces — 400 photos, 4096 pixels (kept to the 1200 highest-variance)."""
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces()
    X, y, names = apply_common_schema(d.data, d.target, None, max_features=1200, max_samples=600)
    return Dataset("Olivetti Faces", X, y, names, 40)


def _mnist_784():
    """MNIST handwritten digits — 784 pixels, sub-sampled to 700 rows."""
    import numpy as np
    from sklearn.datasets import fetch_openml
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y, names = apply_common_schema(np.asarray(d.data, dtype=float),
                                      np.asarray(d.target, dtype=int), None,
                                      max_features=784, max_samples=700)
    return Dataset("MNIST", X, y, names, 10)


# name -> (loader callable, needs_network). Real datasets only.
CATALOGUE = {
    "Iris (sklearn)": (lambda: _sklearn_builtin("load_iris", "Iris", 3), False),
    "Wine (sklearn)": (lambda: _sklearn_builtin("load_wine", "Wine", 3), False),
    "Breast Cancer (sklearn)": (lambda: _sklearn_builtin("load_breast_cancer", "Breast Cancer", 2), False),
    "Digits (sklearn)": (lambda: _sklearn_builtin("load_digits", "Digits", 10), False),
}
# The 12 UCI/OpenML datasets from the paper (need network; capped to ≤120 features).
for _nm in DATASET_REGISTRY:
    CATALOGUE[f"{_nm} (UCI)"] = ((lambda n=_nm: load_dataset(n)), not DATASET_REGISTRY[_nm].offline)
# Well-known high-dimensional real datasets (> 500 features; need download).
CATALOGUE["Isolet (617 feat)"] = ((lambda: load_dataset("Isolet", max_features=617, max_samples=600)), True)
CATALOGUE["MNIST digits (784 feat)"] = (_mnist_784, True)
CATALOGUE["Olivetti Faces (1200 feat)"] = (_olivetti_faces, True)

OFFLINE_DEFAULTS = ["Wine (sklearn)", "Breast Cancer (sklearn)"]

# Known (post common-schema) sizes: rows × dimensions, shown in every picker.
_UCI_SHAPES = {
    "Zoo": (101, 16), "Wine": (178, 13), "Glass": (214, 9), "Ionosphere": (351, 34),
    "Breast Cancer": (569, 30), "Crowdsourced": (600, 28), "Sonar": (208, 60),
    "Spambase": (600, 57), "Arrhythmia": (452, 120), "Isolet": (600, 120),
    "Musk": (600, 120), "Madelon": (600, 120),
}
DATASET_META = {
    "Iris (sklearn)": (150, 4), "Wine (sklearn)": (178, 13),
    "Breast Cancer (sklearn)": (569, 30), "Digits (sklearn)": (600, 64),
    "Isolet (617 feat)": (600, 617), "MNIST digits (784 feat)": (700, 784),
    "Olivetti Faces (1200 feat)": (400, 1200),
}
for _nm in DATASET_REGISTRY:
    if _nm in _UCI_SHAPES:
        DATASET_META[f"{_nm} (UCI)"] = _UCI_SHAPES[_nm]


def _ds_label(name):
    """Selector label with size, e.g. 'Wine (sklearn)  ·  178×13'."""
    m = DATASET_META.get(name)
    return f"{name}  ·  {m[0]}×{m[1]}" if m else name


@st.cache_data(show_spinner=False)
def load_catalogue_dataset(display_name: str):
    loader, _ = CATALOGUE[display_name]
    ds = loader()
    return ds.X, ds.y, ds.feature_names, ds.n_classes, ds.name, ds.true_relevant


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #
# NB: rendered as plain HTML (via st.markdown) rather than a pandas Styler, so the
# app does not depend on jinja2 >= 3 being installed in the environment.
_GREEN = "background-color:#c6f0d0;font-weight:700;"
_BLUE_BORDER = "border-left:2px solid #2b6cb0;border-right:2px solid #2b6cb0;"


def _fmt(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def best_per_column_mask(df: pd.DataFrame):
    """Boolean mask: True where a cell is the best in its metric column."""
    m = np.zeros(df.shape, dtype=bool)
    for j, c in enumerate(df.columns):
        col = df[c].to_numpy(dtype=float)
        lower_better = "runtime" in str(c).lower()
        best = np.nanmin(col) if lower_better else np.nanmax(col)
        m[:, j] = col == best
    return m


def best_per_row_mask(df: pd.DataFrame):
    """Boolean mask: True where a cell is the best (max) in its row."""
    arr = df.to_numpy(dtype=float)
    m = np.zeros(arr.shape, dtype=bool)
    for i in range(arr.shape[0]):
        m[i, :] = arr[i, :] == np.nanmax(arr[i, :])
    return m


def render_table(df: pd.DataFrame, highlight, proposed_col=None, proposed_row=None,
                 display: pd.DataFrame = None):
    """Build an HTML table with best cells highlighted green (no jinja2 needed).

    ``display`` (optional) provides the cell *text* (e.g. "0.89 ± 0.01") while the
    green highlight is still computed from the numeric ``df``/``highlight``.
    """
    disp = display if display is not None else df
    cols = list(df.columns)
    h = ['<table style="border-collapse:collapse;width:100%;font-size:14px;">',
         '<thead><tr>',
         f'<th style="text-align:left;padding:6px 10px;border-bottom:2px solid #cbd5e0;">'
         f'{df.index.name or ""}</th>']
    for c in cols:
        extra = _BLUE_BORDER if c == proposed_col else ""
        h.append(f'<th style="text-align:right;padding:6px 10px;'
                 f'border-bottom:2px solid #cbd5e0;{extra}">{c}</th>')
    h.append('</tr></thead><tbody>')
    for i, (idx, row) in enumerate(df.iterrows()):
        h.append('<tr>')
        lbl = "background-color:#e8f0fe;" if idx == proposed_row else ""
        h.append(f'<td style="text-align:left;padding:6px 10px;font-weight:600;'
                 f'border-bottom:1px solid #edf2f7;{lbl}">{idx}</td>')
        for j, c in enumerate(cols):
            base = "padding:6px 10px;text-align:right;border-bottom:1px solid #edf2f7;"
            if c == proposed_col:
                base += _BLUE_BORDER
            if bool(highlight[i][j]):
                base += _GREEN
            h.append(f'<td style="{base}">{_fmt(disp.iloc[i][c])}</td>')
        h.append('</tr>')
    h.append('</tbody></table>')
    return "".join(h)


# ---- researcher-facing helpers -------------------------------------------- #
METRIC_LABELS = {
    "accuracy": "Accuracy", "stability": "Stability (Jaccard)",
    "trustworthiness": "Trustworthiness", "runtime_s": "Runtime (s)",
    "causal_plausibility": "Causal plausibility", "causal_recall": "Causal recall",
}
METRIC_GLOSSARY = {
    "Accuracy": "5-fold KNN classification accuracy on the selected representation. Higher is better.",
    "Stability (Jaccard)": "Mean pairwise Jaccard overlap of the selected feature subsets across bootstrap resamples. Higher = more reproducible selection.",
    "Trustworthiness": "scikit-learn neighbourhood-preservation score between the original standardised space and the reduced space. Higher = local structure preserved.",
    "Runtime (s)": "Wall-clock fit + transform time. Lower is better.",
    "Causal plausibility": "Fraction of selected features that are true causal drivers (ground-truth datasets only). Higher = fewer spurious correlates.",
    "Causal recall": "Fraction of the true Markov Blanket recovered (ground-truth datasets only).",
}
LOWER_BETTER = {"runtime_s"}


def average_ranks(per_dataset_means: pd.DataFrame, metrics):
    """Average rank (1 = best) of each method across datasets, per metric."""
    ranks = {}
    for metric in metrics:
        mat = per_dataset_means.pivot(index="dataset", columns="method", values=metric)
        asc = metric in LOWER_BETTER
        r = mat.rank(axis=1, ascending=asc)  # rank 1 = best within each dataset
        ranks[METRIC_LABELS.get(metric, metric)] = r.mean(axis=0)
    out = pd.DataFrame(ranks)
    out.index.name = "method"
    return out


def mean_std_tables(df: pd.DataFrame, metrics, methods_used):
    """Return (numeric mean df, 'mean ± std' text df) grouped by method."""
    g = df.groupby("method")
    mean = g[metrics].mean().reindex(methods_used)
    std = g[metrics].std().reindex(methods_used).fillna(0.0)
    mean.columns = [METRIC_LABELS.get(c, c) for c in mean.columns]
    std.columns = mean.columns
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
    n_ds = df["dataset"].nunique()
    text = mean.copy().astype(object)
    for r in mean.index:
        for c in mean.columns:
            if (n_seeds * n_ds) > 1 and std.loc[r, c] > 0:
                text.loc[r, c] = f"{mean.loc[r, c]:.3f} ± {std.loc[r, c]:.3f}"
            else:
                text.loc[r, c] = f"{mean.loc[r, c]:.3f}"
    mean.index.name = "method"
    return mean, text


def _is_lower_better(label):
    return "runtime" in str(label).lower()


def pareto_scatter(mean_df, std_df, x_label, y_label):
    """Trade-off scatter for any two metrics, with error bars and Pareto frontier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_lower, y_lower = _is_lower_better(x_label), _is_lower_better(y_label)
    xs, ys = mean_df[x_label], mean_df[y_label]
    xe, ye = std_df[x_label], std_df[y_label]

    def beq(a, b, lower):
        return (a <= b) if lower else (a >= b)

    def strictly(a, b, lower):
        return (a < b) if lower else (a > b)

    frontier = []
    for m in mean_df.index:
        dominated = any(
            o != m and beq(xs[o], xs[m], x_lower) and beq(ys[o], ys[m], y_lower)
            and (strictly(xs[o], xs[m], x_lower) or strictly(ys[o], ys[m], y_lower))
            for o in mean_df.index)
        if not dominated:
            frontier.append(m)

    fig, ax = plt.subplots(figsize=(4.3, 3.2), dpi=110)
    if len(frontier) > 1:
        fpts = sorted(((xs[m], ys[m]) for m in frontier))
        fx, fy = zip(*fpts)
        ax.plot(fx, fy, "--", color="#94a3b8", lw=1, zorder=1, label="Pareto frontier")
    for m in mean_df.index:
        is_prop = (m == "Proposed")
        on_front = m in frontier
        ax.errorbar(xs[m], ys[m], xerr=xe[m], yerr=ye[m], fmt="o",
                    ms=9 if is_prop else 6,
                    color="#dc2626" if is_prop else ("#16a34a" if on_front else "#2b6cb0"),
                    ecolor="#cbd5e0", elinewidth=1, capsize=2, zorder=3)
        ax.annotate(m, (xs[m], ys[m]), xytext=(4, 3), textcoords="offset points",
                    fontsize=7, fontweight="bold" if is_prop else "normal")
    ax.set_xlabel(f"{x_label} {'←' if x_lower else '→'}", fontsize=8)
    ax.set_ylabel(f"{y_label} {'←' if y_lower else '→'}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return fig


def metric_bars(mean_df, std_df, labels):
    """Grouped bar chart (one subplot per metric) with std error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(labels)
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(max(2.4, 1.9 * n), 2.7), dpi=110)
    if n == 1:
        axes = [axes]
    methods = list(mean_df.index)
    colors = ["#dc2626" if m == "Proposed" else "#2b6cb0" for m in methods]
    for ax, lbl in zip(axes, labels):
        vals = mean_df[lbl].to_numpy(dtype=float)
        errs = std_df[lbl].to_numpy(dtype=float)
        ax.bar(range(len(methods)), vals, yerr=errs, color=colors, capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": "#64748b"})
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        arrow = "↓ lower" if _is_lower_better(lbl) else "↑ higher"
        ax.set_title(f"{lbl}\n({arrow} better)", fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def metric_heatmap(per_dataset_means, metric):
    """Datasets × methods heatmap for one metric (annotated)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat = per_dataset_means.pivot(index="dataset", columns="method", values=metric)
    cmap = "RdYlGn_r" if metric in LOWER_BETTER else "RdYlGn"
    fig, ax = plt.subplots(
        figsize=(0.72 * len(mat.columns) + 1.4, 0.42 * len(mat.index) + 1.1), dpi=110)
    im = ax.imshow(mat.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.to_numpy()[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="#111")
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} per dataset", fontsize=8.5)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig


def to_latex_table(mean_df, std_df, caption="Mean metrics across datasets."):
    """Paper-ready booktabs LaTeX table (mean ± std), best per column in bold."""
    cols = list(mean_df.columns)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Method & " + " & ".join(str(c) for c in cols) + r" \\",
             r"\midrule"]
    best = {c: (mean_df[c].min() if "Runtime" in c else mean_df[c].max()) for c in cols}
    for m in mean_df.index:
        cells = []
        for c in cols:
            val = mean_df.loc[m, c]
            txt = f"{val:.3f}"
            if std_df is not None and std_df.loc[m, c] > 0:
                txt += f" $\\pm$ {std_df.loc[m, c]:.3f}"
            if val == best[c]:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        name = m.replace("_", r"\_")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Step-by-step simulation (visual)
# --------------------------------------------------------------------------- #
def run_simulation(X, y, names, k, strict, true_relevant=None):
    """Fit the framework (recording artefacts) and gather everything the visual
    simulator needs to render each of the 7 stages."""
    from collections import Counter
    if strict:
        cfg = FrameworkConfig(n_representatives=k, n_bootstrap=8, restrict_to_mb=True,
                              mb_max_cond_set=5, conditional_relevance=True,
                              prototype_by="relevance", prefilter_top=150, lam=0.7, alpha=0.4)
    else:
        cfg = FrameworkConfig(n_representatives=k, n_bootstrap=8, mb_max_cond_set=3,
                              prefilter_top=150)
    model = CausalHFS(cfg).fit(X, y)
    Xp = Preprocessor().fit_transform(X)
    cand = model.candidate_features_ or list(range(X.shape[1]))
    corr = correlation_matrix(Xp[:, cand])
    # consensus frequency over a few resamples
    rng = np.random.default_rng(0)
    n = X.shape[0]
    cnt = Counter()
    B = 8
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        cnt.update(model._select_once(X[idx], y[idx], record=False))
    freq = np.array([cnt.get(i, 0) / B for i in range(X.shape[1])])
    return dict(model=model, X_raw=np.asarray(X, dtype=float), Xp=Xp, cand=list(cand),
                corr=corr, freq=freq, names=list(names), k=int(k),
                tau=cfg.consensus_threshold, sparsify=cfg.sparsify_threshold,
                true_relevant=true_relevant)


def sim_data_preview(step, sim, nrows=5):
    """Sample rows of the data as it enters and leaves a stage.

    Returns (left_df, left_label, right_df, right_label). For stages that only
    build structure (graph/distance/clusters) the right side shows the produced
    artefact rather than the unchanged feature columns.
    """
    Xr, Xp = sim["X_raw"], sim["Xp"]
    names, cand, m = sim["names"], sim["cand"], sim["model"]
    sel = list(m.selected_features_)
    nr = min(nrows, Xr.shape[0])

    def tbl(X, cols):
        cols = list(cols)[:8]
        return pd.DataFrame(np.round(X[:nr][:, cols], 2),
                            columns=[str(names[c]) for c in cols])

    if step == 1:
        return (tbl(Xr, range(Xr.shape[1])), "Raw dataset (sample rows)",
                tbl(Xp, range(Xp.shape[1])), "After z-scoring (μ=0, σ=1)")
    if step == 2:
        return (tbl(Xp, range(Xp.shape[1])), f"All {Xp.shape[1]} standardised features",
                tbl(Xp, cand), f"After: {len(cand)} causal candidates kept (Markov Blanket)")
    if step == 3:
        cn = [str(names[c]) for c in cand][:8]
        cdf = pd.DataFrame(np.round(sim["corr"][:8, :8], 2), index=cn, columns=cn)
        return (tbl(Xp, cand), "Working feature columns",
                cdf, "Produced: |correlation| between features")
    if step == 4:
        cn = [str(names[c]) for c in cand][:8]
        ddf = pd.DataFrame(np.round(m.distance_matrix_[:8, :8], 2), index=cn, columns=cn)
        return (tbl(Xp, cand), "Working feature columns",
                ddf, "Produced: hybrid distance matrix")
    if step == 5:
        cdf = pd.DataFrame({"feature": [str(names[c]) for c in cand],
                            "cluster": [int(l) for l in m.clustering_.labels]})
        return (tbl(Xp, cand), "Working feature columns",
                cdf, "Produced: cluster assignment per feature")
    if step == 6:
        return (tbl(Xp, cand), f"Before: {len(cand)} candidate features",
                tbl(Xp, sel), f"After: reduced to {len(sel)} representatives")
    if step == 7:
        fdf = pd.DataFrame({"feature": [str(names[i]) for i in sel],
                            "selection_freq": [round(float(sim["freq"][i]), 2) for i in sel]})
        return (tbl(Xp, sel), f"Selected {len(sel)} features",
                fdf, "Consensus: kept if frequency ≥ τ across bootstraps")
    return None, None, None, None


def _short_names(names, idxs, maxn=40):
    return [names[i] if len(str(names[i])) <= 14 else f"f{i}" for i in idxs][:maxn]


def sim_figure(step, sim):
    """Return a small matplotlib figure visualising a given pipeline stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = sim["model"]
    cand = sim["cand"]
    names = sim["names"]
    cand_names = [names[i] for i in cand]
    R = m.relevance_
    mb = set(m.markov_blanket_)

    if step == 1:  # standardized data heatmap
        Xp = sim["Xp"]
        cols = cand[:40]
        sub = Xp[:min(60, Xp.shape[0]), :][:, cols]
        fig, ax = plt.subplots(figsize=(min(4.2, 0.12 * len(cols) + 1.4), 2.4), dpi=110)
        im = ax.imshow(sub, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_xlabel("features →", fontsize=7); ax.set_ylabel("samples →", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.ax.tick_params(labelsize=6)
        fig.tight_layout(); return fig

    if step == 2:  # causal relevance bars, MB highlighted
        idxs = cand[:40]
        vals = [R[i] for i in idxs]
        colors = ["#f59e0b" if i in mb else "#94a3b8" for i in idxs]
        fig, ax = plt.subplots(figsize=(min(4.6, 0.16 * len(idxs) + 1.4), 2.4), dpi=110)
        ax.bar(range(len(idxs)), vals, color=colors)
        ax.set_xticks(range(len(idxs)))
        ax.set_xticklabels(_short_names(names, idxs), rotation=90, fontsize=6)
        ax.set_ylabel("relevance R", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#f59e0b")],
                  labels=["Markov Blanket"], fontsize=6, loc="upper right")
        fig.tight_layout(); return fig

    if step == 3:  # feature graph
        import networkx as nx
        corr = sim["corr"]
        thr = sim["sparsify"]
        p = len(cand)
        G = nx.Graph()
        G.add_nodes_from(range(p))
        for a in range(p):
            for b in range(a + 1, p):
                if corr[a, b] >= thr:
                    G.add_edge(a, b, weight=float(corr[a, b]))
        fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=110)
        pos = nx.spring_layout(G, seed=1, k=0.6)
        node_col = ["#f59e0b" if cand[i] in mb else "#60a5fa" for i in range(p)]
        for u, v, d in G.edges(data=True):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="#cbd5e0", lw=0.4 + 1.6 * d["weight"], alpha=0.5, zorder=1)
        for i in range(p):
            ax.scatter(*pos[i], s=80, c=node_col[i], edgecolors="#334155", lw=0.5, zorder=2)
            ax.annotate(_short_names(names, [cand[i]])[0], pos[i], fontsize=5,
                        ha="center", va="center")
        ax.axis("off")
        ax.set_title(f"{G.number_of_edges()} edges (|corr| ≥ {thr})", fontsize=7.5)
        fig.tight_layout(); return fig

    if step == 4:  # distance matrix heatmap
        D = m.distance_matrix_
        fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=110)
        im = ax.imshow(D, cmap="viridis")
        ax.set_xticks(range(len(cand))); ax.set_yticks(range(len(cand)))
        ax.set_xticklabels(_short_names(names, cand), rotation=90, fontsize=5)
        ax.set_yticklabels(_short_names(names, cand), fontsize=5)
        cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.ax.tick_params(labelsize=6)
        fig.tight_layout(); return fig

    if step in (5, 6):  # dendrogram; step 6 highlights selected representatives
        sel_local = [cand.index(f) for f in m.selected_features_ if f in cand] if step == 6 else []
        fig = causal_dendrogram(m.distance_matrix_, R[cand], cand_names,
                                selected=sel_local,
                                title=("Clusters" if step == 5 else "Selected representatives (bold)"))
        return fig

    if step == 7:  # consensus frequency
        freq = sim["freq"]
        idxs = cand[:40]
        vals = [freq[i] for i in idxs]
        sel = set(m.selected_features_)
        colors = ["#16a34a" if i in sel else "#94a3b8" for i in idxs]
        fig, ax = plt.subplots(figsize=(min(4.6, 0.16 * len(idxs) + 1.4), 2.4), dpi=110)
        ax.bar(range(len(idxs)), vals, color=colors)
        ax.axhline(sim["tau"], color="#dc2626", ls="--", lw=1, label=f"τ = {sim['tau']}")
        ax.set_xticks(range(len(idxs)))
        ax.set_xticklabels(_short_names(names, idxs), rotation=90, fontsize=6)
        ax.set_ylabel("selection freq", fontsize=7); ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(fontsize=6)
        fig.tight_layout(); return fig
    return None


# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #
def run_experiments(selected, k, n_bootstrap, methods, progress=None, strict_causal=False,
                    n_seeds=1, base_seed=0, accuracy_refine=False):
    """Run each method on each dataset across ``n_seeds`` repetitions.

    Returns a *long* DataFrame with one row per (dataset, method, seed) so that
    mean ± std, ranks and significance tests can be computed downstream.
    """
    rows = []
    loaded, skipped = [], []
    total = max(1, len(selected) * len(methods) * n_seeds)
    done = 0
    for disp in selected:
        try:
            X, y, names, ncls, real_name, true_relevant = load_catalogue_dataset(disp)
        except Exception as exc:
            skipped.append((disp, str(exc)[:120]))
            done += len(methods) * n_seeds
            if progress:
                progress.progress(min(1.0, done / total))
            continue
        loaded.append(disp)
        kk = min(k, X.shape[1])

        def prop_build_factory(seed):
            if strict_causal:
                # Improved causal mode: Markov-Blanket isolation + conditional/direct
                # relevance + relevance-based prototype. Uses the full bootstrap
                # budget for consensus (accuracy + stability) and an optional
                # wrapper-refinement pass for extra accuracy.
                return CausalHFS(FrameworkConfig(
                    n_representatives=kk, n_bootstrap=max(6, n_bootstrap),
                    random_state=seed, restrict_to_mb=True, mb_max_cond_set=5,
                    conditional_relevance=True, prototype_by="relevance",
                    wrapper_refine=accuracy_refine, prefilter_top=150,
                    lam=0.7, alpha=0.4))
            return CausalHFS(FrameworkConfig(
                n_representatives=kk, n_bootstrap=max(4, n_bootstrap // 2),
                random_state=seed, mb_max_cond_set=2, wrapper_refine=accuracy_refine,
                prefilter_top=150))

        for si in range(n_seeds):
            seed = base_seed + si
            for m in methods:
                if progress:
                    progress.progress(min(1.0, done / total),
                                      text=f"{real_name} · {m} · seed {seed}")
                if m == "Proposed":
                    bf = (lambda sd: prop_build_factory(sd))
                else:
                    bf = (lambda nm: (lambda sd: build_baseline(nm, k=kk, random_state=sd)))(m)
                r = evaluate_method(m, bf, X, y, kk, n_bootstrap=n_bootstrap,
                                    random_state=seed, true_relevant=true_relevant)
                rows.append({
                    "dataset": real_name, "method": m, "seed": seed,
                    "accuracy": r.accuracy, "stability": r.stability,
                    "trustworthiness": r.trustworthiness, "runtime_s": r.runtime,
                    "causal_plausibility": r.causal_plausibility,
                    "causal_recall": r.causal_recall,
                })
                done += 1
    if progress:
        progress.progress(1.0, text="Done")
    return pd.DataFrame(rows), loaded, skipped


# =========================================================================== #
# UI
# =========================================================================== #
st.title("Causality-Aware Stable Hierarchical Feature Selection")

tab_exp, tab_algo, tab_sim, tab_hitl = st.tabs(
    ["🔬 Experiments", "🧠 Algorithm", "🎬 Simulate", "🧑‍🔬 Human-in-the-Loop"])

# --------------------------------------------------------------------------- #
# TAB 1 — Experiments
# --------------------------------------------------------------------------- #
CORE_METRICS = ["accuracy", "stability", "trustworthiness", "runtime_s"]

with st.sidebar:
    st.header("① Datasets")
    st.caption("Select one or many, then run.")
    select_all = st.checkbox("Select ALL datasets", value=False)
    all_names = list(CATALOGUE)
    if select_all:
        selected = all_names
        st.multiselect("Datasets", all_names, default=all_names, disabled=True,
                       key="ds_disabled", format_func=_ds_label)
    else:
        selected = st.multiselect("Datasets", all_names, default=OFFLINE_DEFAULTS,
                                  format_func=_ds_label)
    st.caption("Labels show rows×dims · (sklearn) offline · (UCI) & high-dim sets need "
               "internet · >200-dim sets use a top-150 univariate prefilter to stay fast "
               "(still slower — keep methods/seeds modest).")
    with st.expander("📋 Dataset reference (rows × dims)"):
        _ref = pd.DataFrame([
            {"dataset": nm.replace(" (sklearn)", "").replace(" (UCI)", ""),
             "rows": DATASET_META.get(nm, ("?", "?"))[0],
             "dims": DATASET_META.get(nm, ("?", "?"))[1],
             "loads": "online" if CATALOGUE[nm][1] else "offline"}
            for nm in CATALOGUE])
        st.dataframe(_ref, hide_index=True, use_container_width=True, height=280)

    st.header("② Protocol")
    k = st.slider("k — features / components", 2, 40, 10, 1)
    n_bootstrap = st.slider("Bootstrap resamples (stability)", 3, 30, 8, 1)
    n_seeds = st.slider("Repetitions (random seeds)", 1, 10, 1, 1,
                        help="Repeat every run with a different seed to report "
                             "mean ± std and error bars. More seeds = more robust "
                             "but slower.")
    base_seed = st.number_input("Base seed", 0, 9999, 0, 1,
                                help="First seed; repetitions use base_seed, +1, +2, …")
    strict_causal = st.checkbox(
        "Strict causal mode (improved)", value=True,
        help="Markov-Blanket isolation + conditional relevance + relevance-based "
             "prototype selection. Recovers true drivers; slower.")
    accuracy_refine = st.checkbox(
        "Accuracy refinement (wrapper)", value=False,
        help="Runs a light forward-swap over the KNN score after selection. "
             "Raises accuracy ~2-3% but can lower selection stability. Slower.")

    st.header("③ Methods & metrics")
    methods = st.multiselect("Methods", METHOD_ORDER, default=METHOD_ORDER)
    if "Proposed" not in methods:
        methods = ["Proposed"] + methods
    metric_choice = st.multiselect(
        "Metrics to display", [METRIC_LABELS[m] for m in CORE_METRICS],
        default=[METRIC_LABELS[m] for m in CORE_METRICS])
    selected_metrics = [m for m in CORE_METRICS if METRIC_LABELS[m] in metric_choice] or CORE_METRICS

    run = st.button("▶ Run experiment", type="primary", use_container_width=True)

with tab_exp:
    st.subheader("Proposed framework vs. baselines")
    if run:
        if not selected:
            st.warning("Pick at least one dataset in the sidebar.")
        elif len(methods) < 2:
            st.warning("Select the Proposed method and at least one baseline.")
        else:
            bar = st.progress(0.0, text="Running experiments…")
            t0 = time.perf_counter()
            df, loaded, skipped = run_experiments(
                selected, k, n_bootstrap, methods, bar, strict_causal=strict_causal,
                n_seeds=n_seeds, base_seed=base_seed, accuracy_refine=accuracy_refine)
            bar.empty()
            if df.empty:
                st.error("No datasets could be loaded. UCI sets need internet access.")
            else:
                st.session_state["exp_df"] = df
                st.session_state["exp_meta"] = dict(
                    loaded=loaded, skipped=skipped, k=k, nb=n_bootstrap,
                    n_seeds=n_seeds, base_seed=base_seed, strict_causal=strict_causal,
                    secs=time.perf_counter() - t0, methods=methods)

    df = st.session_state.get("exp_df")
    meta = st.session_state.get("exp_meta")
    if df is None:
        st.info("Choose datasets on the left and click **Run experiment**.")
    else:
        methods_used = [m for m in METHOD_ORDER if m in df["method"].unique()]
        ds_names = list(df["dataset"].unique())
        n_ds = len(ds_names)
        nseeds = df["seed"].nunique() if "seed" in df.columns else 1
        st.success(
            f"Ran **{n_ds} dataset(s) × {len(methods_used)} methods × {nseeds} seed(s)** "
            f"in {meta['secs']:.1f}s  ·  datasets: **{', '.join(ds_names)}**.")
        for disp, err in meta["skipped"]:
            st.warning(f"Skipped **{disp}** — {err}")

        # ---- Aggregates (shared across sub-tabs) ----
        per_ds = (df.groupby(["dataset", "method"], as_index=False)
                  [CORE_METRICS + ["causal_plausibility", "causal_recall"]].mean())
        mean_full, text_full = mean_std_tables(df, CORE_METRICS, methods_used)
        std_full = (df.groupby("method")[CORE_METRICS].std().reindex(methods_used).fillna(0.0))
        std_full.columns = [METRIC_LABELS[c] for c in CORE_METRICS]
        sel_labels = [METRIC_LABELS[m] for m in selected_metrics]
        gt = df[df["causal_plausibility"].notna()]
        axis_mean, axis_std = mean_full.copy(), std_full.copy()
        if not gt.empty:
            for raw in ("causal_plausibility", "causal_recall"):
                lbl = METRIC_LABELS[raw]
                axis_mean[lbl] = gt.groupby("method")[raw].mean().reindex(methods_used)
                axis_std[lbl] = gt.groupby("method")[raw].std().reindex(methods_used).fillna(0.0)
        acc_col, stab_col = METRIC_LABELS["accuracy"], METRIC_LABELS["stability"]
        bullets = []
        for lbl in sel_labels:
            lower = "runtime" in lbl.lower()
            best_m = mean_full[lbl].idxmin() if lower else mean_full[lbl].idxmax()
            bullets.append(f"**{lbl}**: {best_m} ({mean_full.loc[best_m, lbl]:.3f})")
        pareto = []
        for m in mean_full.index:
            dominated = any(
                (mean_full.loc[o, acc_col] >= mean_full.loc[m, acc_col]) and
                (mean_full.loc[o, stab_col] >= mean_full.loc[m, stab_col]) and
                (o != m) and (
                    mean_full.loc[o, acc_col] > mean_full.loc[m, acc_col] or
                    mean_full.loc[o, stab_col] > mean_full.loc[m, stab_col])
                for o in mean_full.index)
            if not dominated:
                pareto.append(m)
        cfg_manifest = {
            "datasets": meta["loaded"], "methods": meta["methods"], "k": meta["k"],
            "bootstrap_resamples": meta["nb"],
            "seeds": list(range(meta["base_seed"], meta["base_seed"] + meta["n_seeds"])),
            "strict_causal_mode": meta["strict_causal"],
        }

        # ---- Compact results in sub-tabs ----
        ov, ch, ds_tab, cz, ex = st.tabs(
            ["📊 Overview", "📈 Charts", "🗂 Per-dataset", "🎯 Causal", "⬇ Export"])

        with ov:
            if n_ds == 1:
                _scope = f"Results for **{ds_names[0]}**."
            else:
                _scope = (f"📊 These tables are **averaged across all {n_ds} selected "
                          f"datasets** ({', '.join(ds_names)}). For a per-dataset breakdown, "
                          f"open the **🗂 Per-dataset** sub-tab.")
            st.info(_scope)
            st.markdown("**Best per metric** — " + "; ".join(bullets) + ".")
            if "Proposed" in pareto:
                st.success(f"**Proposed is Pareto-optimal** on accuracy vs. stability "
                           f"(non-dominated: {', '.join(pareto)}).")
            else:
                st.info(f"Pareto-optimal (accuracy vs. stability): {', '.join(pareto)}.")
            hdr = (f"#### Mean ± std over {n_ds} datasets" if n_ds > 1
                   else f"#### Metrics — {ds_names[0]}")
            st.markdown(hdr)
            disp_mean = mean_full[sel_labels]
            st.markdown(render_table(disp_mean, best_per_column_mask(disp_mean),
                                     proposed_row="Proposed", display=text_full[sel_labels]),
                        unsafe_allow_html=True)
            st.caption("Green = best per metric (Proposed row tinted blue). "
                       "± = std across datasets×seeds. Runtime: lower is better.")
            if n_ds >= 2:
                st.markdown("**Average rank across datasets** (1 = best)")
                ranks = average_ranks(per_ds, selected_metrics).reindex(methods_used)
                rank_mask = np.zeros(ranks.shape, dtype=bool)
                for j in range(ranks.shape[1]):
                    rank_mask[:, j] = ranks.to_numpy()[:, j] == np.nanmin(ranks.to_numpy()[:, j])
                st.markdown(render_table(ranks, rank_mask, proposed_row="Proposed"),
                            unsafe_allow_html=True)
            with st.expander("📖 Metric definitions & run configuration"):
                for lbl in sel_labels:
                    st.markdown(f"- **{lbl}** — {METRIC_GLOSSARY.get(lbl, '')}")
                st.json(cfg_manifest)

        with ch:
            cc1, cc2 = st.columns(2)
            with cc1:
                if sel_labels:
                    st.caption("Bars = mean · whiskers = std")
                    st.pyplot(metric_bars(mean_full[sel_labels], std_full[sel_labels], sel_labels),
                              use_container_width=False)
            with cc2:
                if len(methods_used) >= 2:
                    axis_opts = list(axis_mean.columns)
                    a1, a2 = st.columns(2)
                    x_label = a1.selectbox(
                        "X", axis_opts,
                        index=axis_opts.index(acc_col) if acc_col in axis_opts else 0, key="pareto_x")
                    default_y = stab_col if stab_col in axis_opts else axis_opts[min(1, len(axis_opts) - 1)]
                    y_label = a2.selectbox("Y", axis_opts, index=axis_opts.index(default_y), key="pareto_y")
                    if x_label == y_label:
                        st.info("Pick two different metrics.")
                    else:
                        valid = axis_mean[[x_label, y_label]].dropna().index.tolist()
                        if len(valid) >= 2:
                            st.pyplot(pareto_scatter(axis_mean.loc[valid], axis_std.loc[valid],
                                                     x_label, y_label), use_container_width=False)
                            st.caption("Green = Pareto-optimal · red = Proposed.")
                        else:
                            st.info("Causal metrics need a ground-truth dataset (synthetic only).")

        with ds_tab:
            # Plain per-dataset tables: one block per dataset (methods × metrics).
            st.markdown("#### Per-dataset results")
            st.caption("One table per dataset · green = best method per metric · "
                       "Proposed row tinted blue.")
            for d in ds_names:
                sub = (per_ds[per_ds["dataset"] == d].set_index("method")
                       .reindex(methods_used)[selected_metrics])
                sub.columns = [METRIC_LABELS[m] for m in selected_metrics]
                sub.index.name = "method"
                with st.expander(f"📄 {d}", expanded=(n_ds <= 3)):
                    st.markdown(render_table(sub, best_per_column_mask(sub),
                                             proposed_row="Proposed"), unsafe_allow_html=True)

            if n_ds < 2:
                st.caption("Select 2+ datasets to also see the heatmap and significance tests.")
            else:
                st.markdown("#### Heatmap")
                hm_metric = st.selectbox("Metric", selected_metrics,
                                         format_func=lambda m: METRIC_LABELS[m], key="hm_metric")
                st.pyplot(metric_heatmap(per_ds, hm_metric), use_container_width=False)
                if n_ds >= 3 and "Proposed" in methods_used:
                    st.markdown("#### Statistical significance")
                    acc_mat = per_ds.pivot(index="dataset", columns="method", values="accuracy")[methods_used]
                    stab_mat = per_ds.pivot(index="dataset", columns="method", values="stability")[methods_used]
                    fa, pa = friedman_test(acc_mat.to_numpy())
                    fs, ps = friedman_test(stab_mat.to_numpy())
                    oka = "✓ significant" if pa < 0.05 else "not significant"
                    oks = "✓ significant" if ps < 0.05 else "not significant"
                    st.write(f"**Friedman** — accuracy χ²={fa:.2f}, p={pa:.3g} ({oka}); "
                             f"stability χ²={fs:.2f}, p={ps:.3g} ({oks}).")
                    ba = {m: acc_mat[m].to_numpy() for m in methods_used if m != "Proposed"}
                    bs = {m: stab_mat[m].to_numpy() for m in methods_used if m != "Proposed"}
                    wa = wilcoxon_vs_baselines(acc_mat["Proposed"].to_numpy(), ba)
                    ws = wilcoxon_vs_baselines(stab_mat["Proposed"].to_numpy(), bs)
                    sig = pd.DataFrame({
                        "Δacc": {m: round(wa[m]["delta"], 3) for m in ba},
                        "p(acc)": {m: round(wa[m]["p_value"], 3) for m in ba},
                        "Δstab": {m: round(ws[m]["delta"], 3) for m in bs},
                        "p(stab)": {m: round(ws[m]["p_value"], 3) for m in bs},
                    })
                    st.dataframe(sig, use_container_width=True)
                    st.caption("Wilcoxon vs. each baseline; positive Δ favours Proposed; p<0.05 significant.")

        with cz:
            if gt.empty:
                st.info("Causal plausibility needs a dataset with a **known ground-truth Markov "
                        "Blanket**, which only synthetic data provides — it isn't defined for the "
                        "real datasets here. The standalone `causal_plausibility_experiment.py` "
                        "script demonstrates it on a synthetic causal benchmark.")
            else:
                st.caption("Fraction of selected features that are **true causal drivers** "
                           "(the rest are stably-selected spurious correlates).")
                cp = (gt.pivot_table(index="method", columns="dataset", values="causal_plausibility")
                      .reindex([m for m in methods_used if m in gt["method"].unique()]))
                cp["Mean CP"] = cp.mean(axis=1)
                st.markdown(render_table(cp, best_per_column_mask(cp), proposed_row="Proposed"),
                            unsafe_allow_html=True)
                stat_methods = [m for m in ("LASSO", "mRMR", "PCA", "VAE") if m in cp.index]
                if "Proposed" in cp.index and stat_methods:
                    stab_by = gt[gt["method"].isin(stat_methods)].groupby("method")["stability"].mean()
                    most_stable = stab_by.idxmax()
                    prop_cp = cp.loc["Proposed", "Mean CP"]; base_cp = cp.loc[most_stable, "Mean CP"]
                    base_stab = stab_by.loc[most_stable]
                    prop_stab = gt[gt["method"] == "Proposed"]["stability"].mean()
                    m1, m2 = st.columns(2)
                    m1.metric(f"{most_stable} (most stable stat.)", f"CP {base_cp:.0%}",
                              f"stability {base_stab:.2f}", delta_color="off")
                    m2.metric("Proposed (causal)", f"CP {prop_cp:.0%}",
                              f"stability {prop_stab:.2f}", delta_color="off")
                    if prop_cp > base_cp:
                        st.success(f"**{most_stable}** is stable (Jaccard {base_stab:.2f}) yet only "
                                   f"{base_cp:.0%} of its picks are true drivers — stably selecting "
                                   f"spurious correlates. **Proposed** recovers {prop_cp:.0%}.")
                    else:
                        st.info(f"{most_stable} CP {base_cp:.0%} vs. Proposed {prop_cp:.0%} on this run; "
                                f"try a larger k.")
                with st.expander("Causal recall (share of the true Markov Blanket recovered)"):
                    cr = (gt.pivot_table(index="method", columns="dataset", values="causal_recall")
                          .reindex([m for m in methods_used if m in gt["method"].unique()]))
                    cr["Mean"] = cr.mean(axis=1)
                    st.markdown(render_table(cr, best_per_column_mask(cr), proposed_row="Proposed"),
                                unsafe_allow_html=True)

        with ex:
            latex = to_latex_table(mean_full[sel_labels], std_full[sel_labels],
                                   caption="Mean $\\pm$ std across datasets and seeds; best per column in bold.")
            e1, e2, e3 = st.columns(3)
            e1.download_button("⬇ Raw CSV", df.to_csv(index=False).encode(),
                               "causal_hfs_results.csv", "text/csv", use_container_width=True)
            e2.download_button("⬇ LaTeX (.tex)", latex.encode(),
                               "causal_hfs_table.tex", "text/plain", use_container_width=True)
            e3.download_button("⬇ Config (JSON)", json.dumps(cfg_manifest, indent=2).encode(),
                               "causal_hfs_config.json", "application/json", use_container_width=True)
            with st.expander("Preview LaTeX table"):
                st.code(latex, language="latex")


# --------------------------------------------------------------------------- #
# TAB 2 — Algorithm (static steps + live run trace)
# --------------------------------------------------------------------------- #
with tab_algo:
    st.subheader("How the algorithm works")
    st.caption("This is the **enhanced** version of the framework: the 7 base stages below, "
               "plus the improvements summarised here.")

    with st.expander("✨ What's new in the current version (enhancements over the base paper)",
                     expanded=True):
        st.markdown(
            "The default **Strict causal mode** applies enhancements ①–④; the wrapper (⑤) is opt-in.\n\n"
            "**① Markov-Blanket isolation** — before clustering, the pipeline restricts to the "
            "discovered Markov Blanket so only genuine causal candidates compete "
            "(`restrict_to_mb`). *Effect:* recovers the true drivers — causal plausibility went "
            "from ≈0 to **0.83** on a synthetic causal benchmark.\n\n"
            "**② Conditional / direct relevance** — the relevance score `R` now uses the "
            "*direct-effect* score `|partial-corr(f, y | strongest others)|` instead of marginal "
            "mutual information, so a spurious aggregate collapses once its parent drivers are "
            "conditioned on (`conditional_relevance`).\n\n"
            "**③ Relevance-based prototype selection** — each cluster keeps its most "
            "causally-relevant member (the driver of the group) rather than the high-degree / "
            "high-MI hub (`prototype_by=\"relevance\"`; use `\"predictive\"` to favour accuracy).\n\n"
            "**④ Full consensus budget** — the bootstrap consensus vote now uses the full budget. "
            "*Effect:* a clean Pareto gain — accuracy **+0.3%**, stability **+6%**, trustworthiness "
            "**+0.7%** vs. the previous default, nothing regresses.\n\n"
            "**⑤ Wrapper refinement (optional)** — a light forward-swap over the downstream KNN "
            "score, run once after selection (`wrapper_refine`). *Effect:* **+2–3% accuracy** "
            "(can trade some stability); adds a trace stage 8. Toggle it in the demo on the right.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### The pipeline steps")
        for s in CausalHFS.describe_pipeline():
            with st.container(border=True):
                st.markdown(f"**{s['stage']}**")
                st.latex(s["latex"])
                st.caption(s["detail"])

    with right:
        st.markdown("#### Watch it run, step by step")
        a_dataset = st.selectbox(
            "Dataset", [n for n in CATALOGUE if "(sklearn)" in n],
            key="algo_ds", format_func=_ds_label)
        a_k = st.number_input("k", 2, 40, 6, 1, key="algo_k")
        a_strict = st.checkbox("Strict causal mode (improved)", value=True, key="algo_strict")
        a_wrap = st.checkbox("Accuracy refinement (wrapper)", value=False, key="algo_wrap")
        if st.button("▶ Run with trace", key="algo_run", type="primary"):
            X, y, names, ncls, real_name, true_relevant = load_catalogue_dataset(a_dataset)
            kk = int(min(a_k, X.shape[1]))
            if a_strict:
                cfg = FrameworkConfig(n_representatives=kk, n_bootstrap=8, restrict_to_mb=True,
                                      mb_max_cond_set=5, conditional_relevance=True,
                                      prototype_by="relevance", wrapper_refine=a_wrap,
                                      prefilter_top=150, lam=0.7, alpha=0.4)
            else:
                cfg = FrameworkConfig(n_representatives=kk, n_bootstrap=8, mb_max_cond_set=2,
                                      wrapper_refine=a_wrap, prefilter_top=150)
            with st.status("Running the algorithm…", expanded=True) as status:
                def _cb(entry):
                    st.markdown(f"**{entry['stage']}** — {entry['detail']}")
                model = CausalHFS(cfg, step_callback=_cb).fit(X, y)
                status.update(label="Done — selection complete", state="complete")

            sel_names = [names[i] for i in model.selected_features_]
            st.success(f"Selected {len(sel_names)} features · "
                       f"Jaccard stability {model.stability_:.3f}")
            out = {"index": model.selected_features_, "feature": sel_names,
                   "causal_relevance": np.round(model.relevance_[model.selected_features_], 3)}
            if true_relevant is not None:
                from causal_hfs.evaluation import causal_plausibility
                cpv = causal_plausibility(model.selected_features_, true_relevant)
                st.info(f"Causal plausibility on this ground-truth dataset: **{cpv:.0%}** "
                        f"of selected features are true causal drivers.")
                out["is_true_driver"] = [i in set(true_relevant) for i in model.selected_features_]
            st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# TAB 3 — Simulate (guided, learner-friendly walkthrough)
# --------------------------------------------------------------------------- #
LEARNER_NOTES = {
    1: {"goal": "Put every feature on the same scale.",
        "plain": "Features come in different units — one might range 0–1000, another −1 to 1. "
                 "We rescale each to mean 0 and spread 1, like converting every price into the "
                 "same currency before comparing them.",
        "notice": "Raw values (left) vary wildly; after z-scoring (right) they all sit roughly "
                  "between −3 and 3.",
        "why": "Otherwise the later distance steps would be dominated by whichever feature "
               "happens to have the biggest numbers."},
    2: {"goal": "Find the features that truly drive the target.",
        "plain": "We look for the target's **Markov Blanket** — the small set of features that, "
                 "once known, make everything else irrelevant for predicting it. Think of it as "
                 "the target's immediate family: its direct causes and effects.",
        "notice": "Gold bars are the Markov-Blanket features. In causal mode we keep only these, "
                  "so the column count drops on the right.",
        "why": "This focuses the pipeline on genuine drivers rather than look-alike (spurious) "
               "features."},
    3: {"goal": "Map which features carry the same information.",
        "plain": "We connect two features with an edge when they're strongly correlated — i.e. "
                 "partly redundant. Tight clusters of nodes are groups of near-duplicates.",
        "notice": "Densely connected nodes say almost the same thing; lonely nodes carry unique "
                  "information.",
        "why": "We'll only need to keep one feature from each redundant group."},
    4: {"goal": "Turn 'how redundant?' into a distance.",
        "plain": "Correlations become distances: features that say the same thing are **close**, "
                 "features that differ are **far apart**. We blend statistical closeness with "
                 "causal closeness.",
        "notice": "Dark cells = very close (redundant) pairs; bright cells = distinct features.",
        "why": "This distance is exactly what the clustering step groups on."},
    5: {"goal": "Group the redundant features together.",
        "plain": "Starting with every feature on its own, we repeatedly merge the two closest "
                 "groups, growing a tree (dendrogram). Cutting the tree gives k groups of "
                 "similar features.",
        "notice": "Features that merge low on the tree are near-duplicates; branches that join "
                  "high up are quite different.",
        "why": "Each surviving branch becomes one 'slot' in the reduced dataset."},
    6: {"goal": "Pick one spokesperson per group.",
        "plain": "From each group we keep the single most informative, causally-relevant feature "
                 "and drop the rest. This is the actual dimensionality reduction — and every kept "
                 "column is still a real, named feature.",
        "notice": "Bold leaves are the chosen representatives; the table on the right now has "
                  "only k columns.",
        "why": "You shrink the data while keeping full interpretability — unlike PCA, which "
               "returns blended components you can't name."},
    7: {"goal": "Keep only the reliably-chosen features.",
        "plain": "We repeat the whole process on many random re-samples of the data and keep the "
                 "features that get picked again and again. Consistency = trustworthiness.",
        "notice": "Green bars clear the red τ line — they were selected in enough resamples to "
                  "count as stable.",
        "why": "This guards against features that only looked good by chance in one sample."},
}

with tab_sim:
    st.subheader("🎓 Learn the algorithm — one stage at a time")
    st.caption("Pick a small dataset and walk through how the method turns many features into a "
               "few interpretable ones. New here? Start with **Wine** or **Iris**.")
    with st.expander("📖 Key terms (for beginners)"):
        gcol1, gcol2 = st.columns(2)
        gcol1.markdown(
            "- **Feature** — one measured column/variable.\n"
            "- **Target** — the thing we predict (the label).\n"
            "- **Dimensionality reduction** — shrink the number of features while keeping the useful information.\n"
            "- **Feature selection** — reduction that *keeps original features* (unlike PCA, which makes new blended ones).\n"
            "- **Standardization (z-score)** — rescale a feature to mean 0, spread 1.\n"
            "- **Redundancy** — two features carrying nearly the same information (high correlation).\n"
            "- **Prototype / representative** — the one feature kept to stand in for a redundant group.")
        gcol2.markdown(
            "- **Markov Blanket** — the minimal set of features that makes the target independent of all the rest (its direct causes, effects & co-parents).\n"
            "- **Causal driver** — a feature that genuinely influences the target.\n"
            "- **Spurious correlate** — moves with the target but isn't causal.\n"
            "- **Bootstrap** — a random resample of the rows, used to test robustness.\n"
            "- **Consensus** — keep features chosen across many bootstraps.\n"
            "- **Jaccard stability** — how much the selected sets overlap across resamples (1 = identical every time).\n"
            "- **Trustworthiness** — how well the reduced data preserves the neighbours of the full data.")
    sc1, sc2, sc3 = st.columns([3, 1, 1])
    sim_options = [n for n in CATALOGUE if "(sklearn)" in n]
    _default = "Wine (sklearn)"
    sim_ds = sc1.selectbox("Dataset", sim_options,
                           index=sim_options.index(_default) if _default in sim_options else 0,
                           key="sim_ds", format_func=_ds_label,
                           help="Fewer features = clearer visuals.")
    sim_k = sc2.number_input("Keep k", 2, 40, 5, 1, key="sim_k")
    sim_strict = sc3.checkbox("Causal mode", value=True, key="sim_strict")
    if st.button("▶ Start / restart walkthrough", key="sim_run", type="primary"):
        with st.spinner("Running the pipeline…"):
            X, y, names, ncls, real_name, true_rel = load_catalogue_dataset(sim_ds)
            st.session_state["sim"] = run_simulation(X, y, names, int(sim_k), sim_strict, true_rel)
            st.session_state["sim_name"] = real_name
            st.session_state["sim_step"] = 1
            st.session_state["sim_playing"] = False

    sim = st.session_state.get("sim")
    if sim is None:
        st.info("👆 Choose a dataset (try **Wine** or **Iris**) and click **Start walkthrough**. "
                "You'll step through all 7 stages — each with a picture, a plain-English "
                "explanation, and a before/after peek at the data.")
    else:
        steps = CausalHFS.describe_pipeline()
        n = len(steps)
        model = sim["model"]
        step = max(1, min(n, int(st.session_state.get("sim_step", 1))))

        # progress + navigation
        st.progress(step / n, text=f"Stage {step} of {n}")
        st.caption(f"Dataset **{st.session_state.get('sim_name','')}** · "
                   f"{len(sim['names'])} features · Markov Blanket {len(model.markov_blanket_)} · "
                   f"keeping k={len(model.selected_features_)} · stability {model.stability_:.0%}")

        def _sim_go(d):
            st.session_state["sim_step"] = max(1, min(n, int(st.session_state.get("sim_step", 1)) + d))

        nav1, nav2, nav3 = st.columns([1, 2, 1])
        nav1.button("⬅ Previous", key="sim_prev", on_click=_sim_go, args=(-1,),
                    use_container_width=True, disabled=(step == 1))
        nav2.markdown(f"<div style='text-align:center;font-weight:600;padding-top:6px'>"
                      f"{steps[step-1]['stage']}</div>", unsafe_allow_html=True)
        nav3.button("Next ➡", key="sim_next", on_click=_sim_go, args=(1,),
                    use_container_width=True, disabled=(step == n))

        pcol, scol = st.columns([1, 2])
        if st.session_state.get("sim_playing", False):
            pcol.button("⏸ Stop", key="sim_stop", use_container_width=True,
                        on_click=lambda: st.session_state.update(sim_playing=False))
        else:
            pcol.button("▶ Auto-play", key="sim_play", use_container_width=True,
                        disabled=(step == n),
                        on_click=lambda: st.session_state.update(sim_playing=True))
        scol.select_slider("Auto-play speed", ["🐢 Slow", "Normal", "🐇 Fast"],
                           value="Normal", key="sim_speed")

        s = steps[step - 1]
        note = LEARNER_NOTES.get(step, {})
        st.markdown(f"### {s['stage'].split('. ', 1)[-1]}")
        st.markdown(f"🎯 **Goal:** {note.get('goal', '')}")
        st.markdown(note.get("plain", ""))

        colL, colR = st.columns([1, 1])
        with colL:
            fig = sim_figure(step, sim)
            if fig is not None:
                st.pyplot(fig, use_container_width=False)
        with colR:
            st.markdown("**👀 What to notice**")
            st.info(note.get("notice", ""))
            st.markdown(f"**Why it matters:** {note.get('why', '')}")
            with st.expander("🔢 Show the math"):
                st.latex(s["latex"])
                st.caption(s["detail"])

        # before/after data, with a note on what was dropped
        lb, ll, rb, rl = sim_data_preview(step, sim)
        if lb is not None:
            st.markdown("**🔎 The data, before → after this stage**")
            d1, d2 = st.columns(2)
            with d1:
                st.caption(f"Before — {ll}")
                st.dataframe(lb, use_container_width=True, height=150)
            with d2:
                st.caption(f"After — {rl}")
                st.dataframe(rb, use_container_width=True, height=150)
            if step in (2, 6):
                dropped = [c for c in lb.columns if c not in set(rb.columns)]
                if dropped:
                    st.caption(f"➖ Dropped here: {', '.join(map(str, dropped[:8]))}"
                               + (" …" if len(dropped) > 8 else ""))

        # celebratory wrap-up on the last stage
        if step == n:
            kept = ", ".join(str(sim["names"][i]) for i in model.selected_features_)
            msg = (f"🎉 **Done!** The pipeline reduced **{len(sim['names'])} → "
                   f"{len(model.selected_features_)} features**, keeping only named, "
                   f"interpretable ones: {kept}. Selection stability {model.stability_:.0%}.")
            if sim.get("true_relevant") is not None:
                from causal_hfs.evaluation import causal_plausibility
                cpv = causal_plausibility(model.selected_features_, sim["true_relevant"])
                msg += f" On this dataset **{cpv:.0%}** of them are true causal drivers."
            st.success(msg)
        st.caption("Use ⬅ / ➡ or ▶ Auto-play to move between stages.")

        # auto-advance while playing
        if st.session_state.get("sim_playing", False):
            if step < n:
                time.sleep({"🐢 Slow": 2.2, "Normal": 1.3, "🐇 Fast": 0.7}.get(
                    st.session_state.get("sim_speed", "Normal"), 1.3))
                st.session_state["sim_step"] = step + 1
                st.rerun()
            else:
                st.session_state["sim_playing"] = False


# --------------------------------------------------------------------------- #
# TAB 4 — Human-in-the-Loop
# --------------------------------------------------------------------------- #
with tab_hitl:
    st.subheader("Review causally ambiguous merges (Section 4.3)")
    hc1, hc2, hc3, hc4 = st.columns(4)
    h_dataset = hc1.selectbox("Dataset", [n for n in CATALOGUE if "(sklearn)" in n],
                              format_func=_ds_label)
    h_alpha = hc2.slider("Hybrid weight α", 0.0, 1.0, 0.5, 0.05, key="h_alpha")
    h_lam = hc3.slider("Causal weight λ", 0.0, 1.0, 0.5, 0.05, key="h_lam")
    h_k = hc4.number_input("k", 2, 40, 10, 1, key="h_k")
    h_tol = st.slider("Ambiguity tolerance", 0.0, 0.5, 0.05, 0.01, key="h_tol")
    if st.button("Analyse merges", key="h_run"):
        X, y, names, _, _, _ = load_catalogue_dataset(h_dataset)
        Xp = Preprocessor().fit_transform(X)
        an = CausalAnalyzer().fit(Xp, y, lam=h_lam)
        G = build_feature_graph(Xp)
        graph_centrality(G, X.shape[1])
        D = hybrid_distance(Xp, an.relevance_, alpha=h_alpha, enable_causal=True)
        kk = int(min(h_k, X.shape[1]))
        res = agglomerate(D, kk, record_merges=True)
        flagged = find_ambiguous_merges(res.merges, an.relevance_, an.mb_mask_, tol=h_tol)
        st.session_state["h_ctx"] = dict(names=names, flagged=flagged, mb=an.mb_,
                                         X=X, y=y, alpha=h_alpha, lam=h_lam, k=kk, tol=h_tol,
                                         D=D, R=an.relevance_)
        st.session_state["h_dec"] = {}

    ctx = st.session_state.get("h_ctx")
    if ctx is None:
        st.info("Pick a dataset and click **Analyse merges**.")
    else:
        names = ctx["names"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Features", len(names))
        m2.metric("Markov Blanket", len(ctx["mb"]))
        m3.metric("Ambiguous merges", len(ctx["flagged"]))
        oracle = rule_based_oracle()

        def _fmt(c):
            return ", ".join(names[i] for i in c)

        if not ctx["flagged"]:
            st.success("No ambiguous merges — the hybrid-distance criterion was decisive.")
        else:
            for mrg in ctx["flagged"]:
                with st.container(border=True):
                    a, b = st.columns([3, 1])
                    with a:
                        st.markdown(f"**Merge #{mrg.step}** — distance `{mrg.distance:.3f}` "
                                    f"vs next `{mrg.next_distance:.3f}` (gap `{mrg.distance_gap:.3f}`)")
                        st.write(f"• A: {_fmt(mrg.left_cluster)}")
                        st.write(f"• B: {_fmt(mrg.right_cluster)}")
                        st.write(f"• Causal-relevance gap: `{mrg.causal_gap:.3f}`")
                        if mrg.conflicting_causal:
                            st.warning("⚠ Conflicting causal roles (one side in the Markov Blanket).")
                    with b:
                        default = oracle(mrg)
                        ch = st.radio("Decision", ["Approve", "Veto"],
                                      index=0 if default else 1, key=f"h_dec_{mrg.step}")
                        st.session_state["h_dec"][mrg.step] = (ch == "Approve")

        if st.button("Apply decisions & select features", key="h_apply"):
            def cb(mrg):
                return st.session_state["h_dec"].get(mrg.step, True)
            cfg = FrameworkConfig(n_representatives=ctx["k"], alpha=ctx["alpha"], lam=ctx["lam"],
                                  ambiguity_tol=ctx["tol"], enable_hitl=True, n_bootstrap=12)
            model = CausalHFS(cfg, hitl_callback=cb).fit(ctx["X"], ctx["y"])
            st.success(f"Selected {len(model.selected_features_)} features · "
                       f"stability {model.stability_:.3f} · veto rate {model.veto_rate_:.0%}")
            st.dataframe(pd.DataFrame({
                "index": model.selected_features_,
                "feature": [names[i] for i in model.selected_features_],
                "causal_relevance": np.round(model.relevance_[model.selected_features_], 3),
            }), hide_index=True, use_container_width=True)
            if model.distance_matrix_ is not None and len(names) <= 60:
                fig = causal_dendrogram(model.distance_matrix_, model.relevance_, names,
                                        selected=model.selected_features_,
                                        title="Causal dendrogram (bold = selected)")
                st.pyplot(fig)
