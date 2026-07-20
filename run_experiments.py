"""Reproduce the paper's evaluation tables (Sections 5.1-5.3).

Runs the proposed Causal-HFS framework and all seven baselines across a set of
datasets, then emits:

    results_per_dataset.csv   - every (dataset, method) metric row
    results_mean.csv          - mean metrics per method (Table 2)
    results_accuracy.csv      - per-dataset KNN accuracy matrix (Table 4)
    results_significance.txt  - Friedman + Wilcoxon tests (Tables 3)

Usage
-----
    python run_experiments.py --datasets Wine "Breast Cancer" --k 10
    python run_experiments.py --all-offline          # sklearn built-ins only
    python run_experiments.py --csv mydata.csv --label target

By default only offline (sklearn-bundled) datasets run so the harness works with
no network. Pass explicit names or ``--all`` to include OpenML datasets.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from causal_hfs import CausalHFS, FrameworkConfig
from causal_hfs.baselines import BASELINES, build_baseline
from causal_hfs.datasets import (
    available_offline,
    load_csv,
    load_dataset,
    make_synthetic,
)
from causal_hfs.evaluation import (
    MethodResult,
    evaluate_method,
    friedman_test,
    wilcoxon_vs_baselines,
)

METHOD_ORDER = ["Proposed", "PCA", "VAE", "LASSO", "mRMR", "CausalDRIFT", "CAFE", "Random"]


def _proposed_builder(k: int, discrete: bool, enable_hitl: bool):
    def build(seed: int):
        cfg = FrameworkConfig(
            n_representatives=k, enable_hitl=enable_hitl, random_state=seed,
            n_bootstrap=10,
        )
        return CausalHFS(cfg, discrete_target=discrete)
    return build


def _baseline_builder(name: str, k: int, discrete: bool):
    def build(seed: int):
        return build_baseline(name, k=k, random_state=seed, discrete_target=discrete)
    return build


def evaluate_dataset(ds, k: int, n_bootstrap: int, enable_hitl: bool) -> List[MethodResult]:
    discrete = True
    kk = min(k, ds.X.shape[1])
    results: List[MethodResult] = []
    # Proposed
    results.append(
        evaluate_method("Proposed", _proposed_builder(kk, discrete, enable_hitl),
                        ds.X, ds.y, kk, n_bootstrap=n_bootstrap)
    )
    # Baselines
    for name in BASELINES:
        results.append(
            evaluate_method(name, _baseline_builder(name, kk, discrete),
                            ds.X, ds.y, kk, n_bootstrap=n_bootstrap)
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal-HFS evaluation harness")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Dataset names from the registry.")
    ap.add_argument("--all-offline", action="store_true",
                    help="Use only sklearn-bundled datasets (no network).")
    ap.add_argument("--all", action="store_true", help="Use all 12 registered datasets.")
    ap.add_argument("--csv", default=None, help="Path to a custom CSV dataset.")
    ap.add_argument("--label", default=None, help="Label column for --csv.")
    ap.add_argument("--synthetic", action="store_true", help="Run on synthetic data.")
    ap.add_argument("--k", type=int, default=10, help="Number of selected features k.")
    ap.add_argument("--bootstrap", type=int, default=15, help="Bootstrap iterations.")
    ap.add_argument("--hitl", action="store_true", help="Enable HITL in the proposed method.")
    ap.add_argument("--outdir", default="results", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- assemble datasets ----
    datasets = []
    if args.synthetic:
        datasets.append(make_synthetic())
    if args.csv:
        datasets.append(load_csv(args.csv, label_col=args.label))
    names: List[str] = []
    if args.all:
        from causal_hfs.datasets import DATASET_REGISTRY
        names = list(DATASET_REGISTRY)
    elif args.all_offline:
        names = available_offline()
    elif args.datasets:
        names = args.datasets
    for nm in names:
        try:
            datasets.append(load_dataset(nm))
            print(f"[loaded] {nm}: {datasets[-1].shape}")
        except Exception as exc:
            print(f"[skip] {nm}: {exc}")
    if not datasets:
        print("No datasets specified; falling back to synthetic + offline built-ins.")
        datasets.append(make_synthetic())
        for nm in available_offline():
            try:
                datasets.append(load_dataset(nm))
            except Exception as exc:
                print(f"[skip] {nm}: {exc}")

    # ---- run ----
    rows = []
    acc_matrix: Dict[str, Dict[str, float]] = {}
    stab_matrix: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        print(f"\n=== {ds.name} {ds.shape} ===")
        results = evaluate_dataset(ds, args.k, args.bootstrap, args.hitl)
        acc_matrix[ds.name] = {}
        stab_matrix[ds.name] = {}
        for r in results:
            rows.append({
                "dataset": ds.name, "method": r.method, "accuracy": r.accuracy,
                "stability": r.stability, "trustworthiness": r.trustworthiness,
                "runtime_s": r.runtime,
            })
            acc_matrix[ds.name][r.method] = r.accuracy
            stab_matrix[ds.name][r.method] = r.stability
            print(f"  {r.method:12s} acc={r.accuracy:.3f} "
                  f"stab={r.stability:.3f} trust={r.trustworthiness:.3f} "
                  f"t={r.runtime:.3f}s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "results_per_dataset.csv"), index=False)

    # ---- Table 2: mean metrics ----
    mean_df = (df.groupby("method")[["accuracy", "stability", "trustworthiness", "runtime_s"]]
               .mean().reindex([m for m in METHOD_ORDER if m in df["method"].unique()]))
    mean_df.to_csv(os.path.join(args.outdir, "results_mean.csv"))
    print("\n=== Mean metrics (Table 2) ===")
    print(mean_df.round(3).to_string())

    # ---- Table 4: per-dataset accuracy ----
    acc_df = pd.DataFrame(acc_matrix).T
    acc_df = acc_df[[m for m in METHOD_ORDER if m in acc_df.columns]]
    acc_df.to_csv(os.path.join(args.outdir, "results_accuracy.csv"))

    # ---- Significance tests (only meaningful with >=3 datasets) ----
    sig_lines: List[str] = []
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    if len(acc_df) >= 3 and len(methods) >= 3:
        acc_mat = acc_df[methods].to_numpy()
        stab_mat = pd.DataFrame(stab_matrix).T[methods].to_numpy()
        fa, pa = friedman_test(acc_mat)
        fs, ps = friedman_test(stab_mat)
        sig_lines.append(f"Friedman accuracy: chi2={fa:.3f}, p={pa:.4g}")
        sig_lines.append(f"Friedman stability: chi2={fs:.3f}, p={ps:.4g}\n")

        prop_acc = acc_df["Proposed"].to_numpy()
        prop_stab = pd.DataFrame(stab_matrix).T["Proposed"].to_numpy()
        base_acc = {m: acc_df[m].to_numpy() for m in methods if m != "Proposed"}
        base_stab = {m: pd.DataFrame(stab_matrix).T[m].to_numpy()
                     for m in methods if m != "Proposed"}
        wa = wilcoxon_vs_baselines(prop_acc, base_acc)
        ws = wilcoxon_vs_baselines(prop_stab, base_stab)
        sig_lines.append("Wilcoxon (Proposed vs baseline):")
        sig_lines.append(f"{'baseline':12s} {'dAcc':>8s} {'pAcc':>8s} {'dStab':>8s} {'pStab':>8s}")
        for m in base_acc:
            sig_lines.append(
                f"{m:12s} {wa[m]['delta']:+8.3f} {wa[m]['p_value']:8.3g} "
                f"{ws[m]['delta']:+8.3f} {ws[m]['p_value']:8.3g}"
            )
    else:
        sig_lines.append("Significance tests require >=3 datasets; skipped.")

    sig_text = "\n".join(sig_lines)
    with open(os.path.join(args.outdir, "results_significance.txt"), "w") as fh:
        fh.write(sig_text + "\n")
    print("\n=== Significance ===")
    print(sig_text)
    print(f"\nAll outputs written to {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
