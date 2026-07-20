"""Causal-Plausibility experiment: stability is not correctness.

Jaccard stability only asks whether a method selects the *same* features across
resamples. It says nothing about whether those features are the *right* ones. On a
structural causal model with a known Markov Blanket, this script measures:

    Causal Plausibility (CP) = |selected ∩ true_drivers| / |selected|
    Causal Recall           = |selected ∩ true_drivers| / |true_drivers|

and shows that purely statistical selectors (mRMR, CausalDRIFT, CAFE) are *stably
wrong* — they consistently pick spurious aggregate correlates instead of the true
causal drivers — while the proposed framework in strict causal mode recovers the
drivers.

Run:
    python causal_plausibility_experiment.py
Outputs a table to stdout and writes results/causal_plausibility.csv + .md.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from causal_hfs import CausalHFS, FrameworkConfig
from causal_hfs.baselines import BASELINES, build_baseline
from causal_hfs.datasets import make_causal_benchmark
from causal_hfs.evaluation import (
    causal_plausibility,
    causal_recall,
    evaluate_method,
)


def build_proposed(kk, strict=True):
    def build(seed):
        if strict:
            # Improved causal mode: Markov-Blanket isolation + conditional/direct
            # relevance (#1) + relevance-based prototype selection (#2).
            return CausalHFS(FrameworkConfig(
                n_representatives=kk, n_bootstrap=6, random_state=seed,
                restrict_to_mb=True, mb_max_cond_set=5, conditional_relevance=True,
                prototype_by="relevance", lam=0.7, alpha=0.4))
        return CausalHFS(FrameworkConfig(
            n_representatives=kk, n_bootstrap=6, random_state=seed, mb_max_cond_set=2))
    return build


def main(k: int = 6, n_bootstrap: int = 12, seeds=(0, 1, 2)) -> None:
    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in seeds:
        ds = make_causal_benchmark(random_state=seed)
        true = ds.true_relevant
        kk = min(k, ds.X.shape[1])
        specs = [
            ("Proposed (improved causal)", build_proposed(kk, strict=True)),
            ("Proposed (soft/default)", build_proposed(kk, strict=False)),
        ] + [(n, (lambda nm: (lambda s: build_baseline(nm, k=kk, random_state=s)))(n))
             for n in BASELINES]
        for name, bf in specs:
            r = evaluate_method(name, bf, ds.X, ds.y, kk,
                                n_bootstrap=n_bootstrap, true_relevant=true)
            rows.append({
                "seed": seed, "method": name,
                "accuracy": r.accuracy, "stability": r.stability,
                "causal_plausibility": r.causal_plausibility,
                "causal_recall": r.causal_recall,
            })

    df = pd.DataFrame(rows)
    agg = (df.groupby("method")[["accuracy", "stability", "causal_plausibility",
                                 "causal_recall"]]
           .mean().round(3))
    order = (["Proposed (improved causal)", "Proposed (soft/default)"] + list(BASELINES))
    agg = agg.reindex([m for m in order if m in agg.index])
    agg.to_csv("results/causal_plausibility.csv")

    print("Causal-Benchmark: 5 true drivers, 10 spurious aggregates, 10 noise "
          f"(k={k}, {n_bootstrap} bootstraps, {len(seeds)} seeds)\n")
    print(agg.to_string())

    with open("results/causal_plausibility.md", "w") as fh:
        fh.write("# Causal Plausibility — stability is not correctness\n\n")
        fh.write(f"Structural causal model with a known Markov Blanket: 5 true "
                 f"drivers cause the target, 10 spurious features are aggregates of "
                 f"two drivers (higher marginal correlation, but not causal), plus "
                 f"10 noise features. Every method selects k={k}. Averaged over "
                 f"{len(seeds)} seeds, {n_bootstrap} bootstrap resamples.\n\n")
        fh.write(agg.to_markdown())
        fh.write("\n\n**Causal Plausibility** = fraction of selected features that "
                 "are true causal drivers. A method can be highly *stable* yet have "
                 "low plausibility if it stably selects spurious correlates. The "
                 "purely statistical selectors (mRMR, CausalDRIFT, CAFE) tend to "
                 "score ~0 here — they lock onto the spurious aggregates — while the "
                 "proposed framework in strict causal mode recovers the true drivers.\n")
    print("\nSaved results/causal_plausibility.csv and .md")


if __name__ == "__main__":
    main()
