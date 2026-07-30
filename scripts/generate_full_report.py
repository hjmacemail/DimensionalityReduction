"""Generate the full-benchmark results CSV that the app's 'Full Report' tab displays.

Runs EVERY dataset in the catalogue (offline sklearn sets + the online UCI/OpenML
sets + the high-dimensional sets) with auto-suggested k (nested-CV), all methods, and
all metrics, then writes ``results/full_benchmark_results.csv``.

Run this ONCE on a machine with internet (the online datasets are downloaded):

    python scripts/generate_full_report.py

Then commit the updated results/full_benchmark_results.csv. The app's Full Report tab
displays it read-only — end users never run the benchmark themselves.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Make the causal_hfs package importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from causal_hfs import CausalHFS, FrameworkConfig
from causal_hfs.baselines import build_baseline
from causal_hfs.datasets import (DATASET_REGISTRY, Dataset, apply_common_schema,
                                  load_dataset)
from causal_hfs.evaluation import evaluate_method, select_k_nested

METHODS = ["Proposed", "PCA", "VAE", "LASSO", "mRMR", "CausalDRIFT", "CAFE", "Random"]
OUT = os.path.join(ROOT, "results", "full_benchmark_results.csv")


# --------------------------------------------------------------------------- #
# Loaders (mirror app.py's catalogue)
# --------------------------------------------------------------------------- #
def _sklearn(loader_name, name, ncls):
    from sklearn import datasets as skd
    d = getattr(skd, loader_name)()
    X, y, names = apply_common_schema(d.data, d.target,
                                      list(getattr(d, "feature_names", [])) or None,
                                      max_features=120, max_samples=600)
    return Dataset(name, X, y, names, ncls)


def _mnist():
    from sklearn.datasets import fetch_openml
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    Xf = np.asarray(d.data)
    yf = np.asarray(d.target, dtype=int)
    idx = np.random.default_rng(42).choice(Xf.shape[0], size=min(500, Xf.shape[0]), replace=False)
    X = np.ascontiguousarray(Xf[idx], dtype=np.float32)
    yv = yf[idx]
    del d, Xf, yf
    gc.collect()
    X, y, names = apply_common_schema(X, yv, None, max_features=784, max_samples=500)
    return Dataset("MNIST", X, y, names, 10)


def _olivetti():
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces()
    X = np.ascontiguousarray(d.data, dtype=np.float32)
    yv = np.asarray(d.target, dtype=int)
    del d
    gc.collect()
    X, y, names = apply_common_schema(X, yv, None, max_features=900, max_samples=400)
    return Dataset("Olivetti Faces", X, y, names, 40)


# (display name, loader) — display names are unique so nothing collides in the report.
CATALOGUE = [
    ("Iris (sklearn)", lambda: _sklearn("load_iris", "Iris", 3)),
    ("Wine (sklearn)", lambda: _sklearn("load_wine", "Wine", 3)),
    ("Breast Cancer (sklearn)", lambda: _sklearn("load_breast_cancer", "Breast Cancer", 2)),
    ("Digits (sklearn)", lambda: _sklearn("load_digits", "Digits", 10)),
]
for _nm in DATASET_REGISTRY:
    CATALOGUE.append((f"{_nm} (UCI)", (lambda n=_nm: load_dataset(n))))
CATALOGUE += [
    ("Isolet (617 feat)", lambda: load_dataset("Isolet", max_features=617, max_samples=500)),
    ("MNIST digits (784 feat)", _mnist),
    ("Olivetti Faces (900 feat)", _olivetti),
]


def main():
    rows = []
    skipped = []
    seed = 0
    t_all = time.time()
    for disp, loader in CATALOGUE:
        try:
            ds = loader()
        except Exception as exc:
            skipped.append((disp, str(exc)[:120]))
            print(f"SKIP {disp}: {str(exc)[:100]}", flush=True)
            continue
        X, y, p = ds.X, ds.y, ds.X.shape[1]
        wide, very_wide = p > 200, p > 600
        pf_base = 80 if very_wide else (110 if wide else 150)
        nb_eval = 8 if not wide else 3
        nb_cons = 8 if not wide else 3
        reps = 2 if wide else 3
        t = time.time()
        kk, _ = select_k_nested(X, y, k_max=min(40, p), n_repeats=reps,
                                tau_stability=0.6, random_state=seed)
        pf_top = max(pf_base, kk + 30)
        print(f"[{disp}] {X.shape}  auto-k={kk}  ({time.time()-t:.1f}s)", flush=True)
        for m in METHODS:
            if m == "Proposed":
                bf = lambda sd: CausalHFS(FrameworkConfig(
                    n_representatives=kk, n_bootstrap=nb_cons, random_state=sd,
                    mb_max_cond_set=3, rf_relevance=True, prototype_by="greedy",
                    improved_greedy=True, prefilter_top=pf_top))
            else:
                bf = (lambda nm: (lambda sd: build_baseline(nm, k=kk, random_state=sd)))(m)
            r = evaluate_method(m, bf, X, y, kk, n_bootstrap=nb_eval, random_state=seed)
            rows.append({"dataset": disp, "method": m, "seed": seed, "k": kk,
                         "accuracy": r.accuracy, "stability": r.stability,
                         "trustworthiness": r.trustworthiness, "runtime_s": r.runtime,
                         "causal_plausibility": r.causal_plausibility,
                         "causal_recall": r.causal_recall})
        pd.DataFrame(rows).to_csv(OUT, index=False)   # incremental save

    print(f"\nDone in {time.time()-t_all:.0f}s. {len(rows)} rows over "
          f"{len({r['dataset'] for r in rows})} datasets -> {OUT}")
    if skipped:
        print("Skipped:", ", ".join(f"{d} ({e})" for d, e in skipped))


if __name__ == "__main__":
    main()
