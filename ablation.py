"""Ablation study and hyperparameter sensitivity (Sections 5.4 & 5.5).

* Ablation (Table 5): Full model vs. w/o Causality, w/o HITL, w/o Stability,
  reporting Jaccard selection stability.
* Sensitivity (Figure 4): sweep the hybrid-distance weight alpha from 0 (purely
  causal) to 1 (purely statistical) and report accuracy + stability.

Usage
-----
    python ablation.py --datasets Wine "Breast Cancer" --k 10
    python ablation.py --all-offline
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from causal_hfs import CausalHFS, FrameworkConfig
from causal_hfs.datasets import available_offline, load_dataset, make_synthetic
from causal_hfs.evaluation import knn_accuracy


VARIANTS = {
    "Full Model":      dict(enable_causal=True,  enable_hitl=True,  enable_stability=True),
    "w/o Causality":   dict(enable_causal=False, enable_hitl=True,  enable_stability=True),
    "w/o HITL":        dict(enable_causal=True,  enable_hitl=False, enable_stability=True),
    "w/o Stability":   dict(enable_causal=True,  enable_hitl=True,  enable_stability=False),
}


def run_ablation(datasets, k: int) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        kk = min(k, ds.X.shape[1])
        row = {"dataset": ds.name}
        for vname, flags in VARIANTS.items():
            cfg = FrameworkConfig(n_representatives=kk, n_bootstrap=15, **flags)
            model = CausalHFS(cfg).fit(ds.X, ds.y)
            row[vname] = round(model.stability_, 3)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("dataset")
    df.loc["Mean"] = df.mean(numeric_only=True).round(3)
    return df


def run_sensitivity(datasets, k: int, alphas=None) -> pd.DataFrame:
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for a in alphas:
        accs, stabs = [], []
        for ds in datasets:
            kk = min(k, ds.X.shape[1])
            cfg = FrameworkConfig(n_representatives=kk, alpha=a, n_bootstrap=15)
            model = CausalHFS(cfg).fit(ds.X, ds.y)
            accs.append(knn_accuracy(model.transform(ds.X), ds.y))
            stabs.append(model.stability_)
        rows.append({"alpha": a, "accuracy": round(np.mean(accs), 3),
                     "stability": round(np.mean(stabs), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--all-offline", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    datasets = []
    names: List[str] = args.datasets or (available_offline() if args.all_offline else [])
    for nm in names:
        try:
            datasets.append(load_dataset(nm))
        except Exception as exc:
            print(f"[skip] {nm}: {exc}")
    if not datasets:
        datasets = [make_synthetic()]

    ab = run_ablation(datasets, args.k)
    se = run_sensitivity(datasets, args.k)
    ab.to_csv(os.path.join(args.outdir, "ablation.csv"))
    se.to_csv(os.path.join(args.outdir, "sensitivity.csv"), index=False)

    print("=== Ablation (Jaccard stability, Table 5) ===")
    print(ab.to_string())
    print("\n=== Sensitivity to alpha (Figure 4) ===")
    print(se.to_string(index=False))


if __name__ == "__main__":
    main()
