"""Dataset loading and the common schema (Section 4.1).

Raw files are parsed into a common schema (numeric features, label last), with
identifier columns dropped and missing values median-imputed. High-dimensional
sets are capped to their ``max_features`` (default 120) highest-variance features
and sub-sampled to at most ``max_samples`` (default 600) rows.

Loading strategy
----------------
* scikit-learn built-ins (Wine, Breast Cancer, ...) load fully offline.
* UCI datasets are fetched via ``sklearn.datasets.fetch_openml`` when network is
  available; failures degrade gracefully (the dataset is skipped, not fatal).
* Arbitrary user data can be supplied with :func:`load_csv` at any time.

The 12-dataset registry from Table 1 is declared in :data:`DATASET_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # (n_samples, n_features), numeric
    y: np.ndarray            # (n_samples,), label
    feature_names: List[str]
    n_classes: int
    notes: str = ""
    # Ground-truth causal structure (only known for synthetic datasets).
    # ``true_relevant`` are the indices of genuine causal drivers — the true
    # Markov Blanket of the target. ``redundant`` are spurious correlates
    # (statistically associated but not causal). Both are None for real data.
    true_relevant: Optional[List[int]] = None
    redundant: Optional[List[int]] = None

    @property
    def shape(self):
        return self.X.shape

    @property
    def has_ground_truth(self) -> bool:
        return self.true_relevant is not None


# --------------------------------------------------------------------------- #
# Common-schema helpers
# --------------------------------------------------------------------------- #
def apply_common_schema(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_features: int = 120,
    max_samples: int = 600,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Median-impute, cap to highest-variance features, sub-sample rows."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, p = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(p)]

    # Median imputation of missing values.
    col_median = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_median, inds[1])

    # Cap to the highest-variance features.
    if p > max_features:
        var = np.nanvar(X, axis=0)
        keep = np.argsort(var)[::-1][:max_features]
        keep.sort()
        X = X[:, keep]
        feature_names = [feature_names[i] for i in keep]

    # Sub-sample rows (stratified where possible).
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        try:
            from sklearn.model_selection import train_test_split

            idx = np.arange(n)
            idx, _ = train_test_split(
                idx, train_size=max_samples, stratify=y, random_state=random_state
            )
        except Exception:
            idx = rng.choice(n, size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    return X, y, feature_names


def load_csv(
    path: str,
    label_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
    max_features: int = 120,
    max_samples: int = 600,
    name: str = "custom",
) -> Dataset:
    """Load a user-supplied CSV into the common schema.

    Parameters
    ----------
    path : CSV path.
    label_col : name of the target column; defaults to the last column.
    id_cols : identifier columns to drop.
    """
    df = pd.read_csv(path)
    if id_cols:
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
    if label_col is None:
        label_col = df.columns[-1]
    y_raw = df[label_col].to_numpy()
    Xdf = df.drop(columns=[label_col])

    # Coerce non-numeric feature columns via label encoding; '?' -> NaN.
    Xdf = Xdf.replace("?", np.nan)
    for c in Xdf.columns:
        if not np.issubdtype(Xdf[c].dtype, np.number):
            Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    X = Xdf.to_numpy(dtype=float)

    # Encode labels to integers.
    classes, y = np.unique(y_raw, return_inverse=True)
    X, y, names = apply_common_schema(
        X, y, list(Xdf.columns), max_features, max_samples
    )
    return Dataset(name, X, y, names, n_classes=len(classes),
                   notes=f"loaded from {path}")


# --------------------------------------------------------------------------- #
# Built-in loaders
# --------------------------------------------------------------------------- #
def _from_sklearn(loader) -> tuple:
    data = loader()
    names = list(getattr(data, "feature_names", [])) or None
    return data.data, data.target, names


def _sklearn_wine():
    from sklearn.datasets import load_wine
    return _from_sklearn(load_wine)


def _sklearn_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    return _from_sklearn(load_breast_cancer)


def _openml(name: str, version: str | int = "active"):
    """Fetch a dataset from OpenML (returns X, y, names) or raise."""
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    Xdf = Xdf.replace("?", np.nan)
    for c in Xdf.columns:
        if not np.issubdtype(Xdf[c].dtype, np.number):
            Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    X = Xdf.to_numpy(dtype=float)
    _, y = np.unique(ds.target.to_numpy(), return_inverse=True)
    return X, y, list(Xdf.columns)


@dataclass
class DatasetSpec:
    name: str
    loader: Callable
    n_classes: int
    notes: str
    offline: bool  # True if it loads without network


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "Zoo": DatasetSpec("Zoo", lambda: _openml("zoo"), 7, "Animal taxonomy", False),
    "Wine": DatasetSpec("Wine", _sklearn_wine, 3, "Chemical assay", True),
    "Glass": DatasetSpec("Glass", lambda: _openml("glass"), 6, "Forensic glass", False),
    "Ionosphere": DatasetSpec("Ionosphere", lambda: _openml("ionosphere"), 2, "Radar returns", False),
    "Breast Cancer": DatasetSpec("Breast Cancer", _sklearn_breast_cancer, 2, "WDBC diagnosis", True),
    "Crowdsourced": DatasetSpec("Crowdsourced", lambda: _openml("Crowdsourced-Mapping"), 6, "Land-cover mapping", False),
    "Sonar": DatasetSpec("Sonar", lambda: _openml("sonar"), 2, "Sonar mines/rocks", False),
    "Spambase": DatasetSpec("Spambase", lambda: _openml("spambase"), 2, "Email spam", False),
    "Arrhythmia": DatasetSpec("Arrhythmia", lambda: _openml("arrhythmia"), 13, "ECG (capped at 120 features)", False),
    "Isolet": DatasetSpec("Isolet", lambda: _openml("isolet"), 26, "Speech (capped at 120 features)", False),
    "Musk": DatasetSpec("Musk", lambda: _openml("musk"), 2, "Molecules (capped at 120 features)", False),
    "Madelon": DatasetSpec("Madelon", lambda: _openml("madelon"), 2, "NIPS-2003 synthetic", False),
}


def load_dataset(
    name: str, max_features: int = 120, max_samples: int = 600, random_state: int = 42
) -> Dataset:
    """Load one registered dataset into the common schema.

    Raises ``RuntimeError`` if a network-dependent dataset cannot be fetched.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Options: {list(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[name]
    try:
        X, y, names = spec.loader()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Could not load '{name}' ({exc}). It may require network access to "
            f"OpenML. Supply the data manually with load_csv()."
        ) from exc
    X, y, names = apply_common_schema(X, y, names, max_features, max_samples, random_state)
    return Dataset(spec.name, X, y, names, spec.n_classes, spec.notes)


def make_synthetic(
    n_samples: int = 300, n_features: int = 20, n_informative: int = 5,
    n_classes: int = 2, random_state: int = 42, n_redundant: int | None = None,
    name: str = "Synthetic",
) -> Dataset:
    """A self-contained synthetic dataset with a KNOWN causal structure.

    Built with ``shuffle=False`` so the column layout is deterministic:

        [0 : n_informative)                      -> true causal drivers (Markov Blanket)
        [n_informative : n_informative+n_redundant) -> redundant spurious correlates
                                                      (linear combos of the drivers)
        [n_informative+n_redundant : ]           -> pure noise

    The ground-truth Markov Blanket (``true_relevant``) is therefore the first
    ``n_informative`` columns; the redundant block is recorded separately so we
    can show that statistical selectors tend to grab these spurious correlates.
    """
    from sklearn.datasets import make_classification

    if n_redundant is None:
        n_redundant = min(5, n_features - n_informative)
    n_redundant = max(0, min(n_redundant, n_features - n_informative))

    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        shuffle=False, random_state=random_state,
    )
    names = [f"f{i}" for i in range(n_features)]
    true_relevant = list(range(n_informative))
    redundant = list(range(n_informative, n_informative + n_redundant))
    return Dataset(name, X, y, names, n_classes,
                   "sklearn make_classification (known causal structure)",
                   true_relevant=true_relevant, redundant=redundant)


def make_madelon_like(
    n_samples: int = 480, n_features: int = 50, n_informative: int = 5,
    n_redundant: int = 15, random_state: int = 3,
) -> Dataset:
    """A Madelon-style dataset: few true drivers, many spurious correlates, rest noise.

    Mirrors the design of the NIPS-2003 Madelon benchmark (5 informative features,
    a block of redundant linear combinations, and a large noise block). Ideal for
    the Causal-Plausibility test: a purely statistical selector can *stably* pick
    the redundant correlates instead of the 5 true causal drivers.
    """
    return make_synthetic(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_classes=2, random_state=random_state,
        name="Madelon-like",
    )


def make_causal_benchmark(
    n_samples: int = 500, n_drivers: int = 5, n_noise: int = 10,
    driver_noise: float = 0.4, spurious_noise: float = 0.25, random_state: int = 0,
) -> Dataset:
    """A structural causal model with a KNOWN Markov Blanket, built to expose the
    difference between *stable* and *correct* selection.

    Structure::

        driver_0..driver_{d-1}     ~ N(0, 1)          # the TRUE causal drivers (MB of y)
        y = 1[ sum(drivers) + noise > median ]        # y is caused by the drivers
        spurious_ij = driver_i + driver_j + noise     # aggregates of TWO drivers
        noise_0..noise_{m-1}       ~ N(0, 1)          # irrelevant

    Each ``spurious_ij`` correlates with the target *more strongly* than any single
    driver (it sums two causal signals), so purely statistical selectors rank the
    spurious aggregates first and select them **stably but wrongly**. They are not
    in the Markov Blanket, though: conditioning on their two parent drivers renders
    them independent of ``y``. ``true_relevant`` therefore lists only the drivers.
    """
    rng = np.random.default_rng(random_state)
    D = rng.normal(size=(n_samples, n_drivers))
    ylin = D.sum(axis=1) + rng.normal(scale=driver_noise, size=n_samples)
    y = (ylin > np.median(ylin)).astype(int)
    pairs = [(i, j) for i in range(n_drivers) for j in range(i + 1, n_drivers)]
    A = np.column_stack([D[:, i] + D[:, j] + rng.normal(scale=spurious_noise, size=n_samples)
                         for i, j in pairs])
    N = rng.normal(size=(n_samples, n_noise))
    X = np.hstack([D, A, N])
    names = ([f"driver_{i}" for i in range(n_drivers)]
             + [f"spurious_{i}{j}" for i, j in pairs]
             + [f"noise_{i}" for i in range(n_noise)])
    ds = Dataset("Causal-Benchmark", X, y, names, 2,
                 "SEM: drivers cause y; spurious = driver aggregates; plus noise")
    ds.true_relevant = list(range(n_drivers))
    ds.redundant = list(range(n_drivers, n_drivers + len(pairs)))
    return ds


def available_offline() -> List[str]:
    """Names of datasets that load without network access."""
    return [n for n, s in DATASET_REGISTRY.items() if s.offline]
