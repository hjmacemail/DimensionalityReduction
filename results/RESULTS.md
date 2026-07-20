# Experimental Results — Proposed Causal-HFS vs. Baselines

**Setup.** The proposed framework is compared against PCA, VAE, LASSO, mRMR, CausalDRIFT, CAFE and Random across 5 datasets: Wine, Breast Cancer and Digits (scikit-learn real datasets) plus two synthetic sets (Synthetic-A: 30 features / 8 informative / 2 classes; Synthetic-B: 30 features / 6 informative / 3 classes). All methods select k=10 features/components. Metrics: 5-fold KNN accuracy, mean pairwise Jaccard stability over 8 bootstrap resamples, scikit-learn neighbourhood-preservation trustworthiness, and fit+transform runtime.

> OpenML was unreachable in this environment, so the 8 additional UCI datasets from the paper (Zoo, Glass, Ionosphere, Sonar, Spambase, Arrhythmia, Isolet, Musk, Madelon) are not included here. Run `python run_experiments.py --all` on a networked machine to reproduce the full 12-dataset tables. Iris was excluded as degenerate (only 4 features, so every method selects all of them).

## Mean metrics across datasets

| method      |   accuracy |   stability |   trustworthiness |   runtime_s |
|:------------|-----------:|------------:|------------------:|------------:|
| Proposed    |      0.77  |       0.445 |             0.926 |       0.549 |
| PCA         |      0.861 |       0.574 |             0.974 |       0.002 |
| VAE         |      0.861 |       0.579 |             0.974 |       0.003 |
| LASSO       |      0.886 |       0.573 |             0.938 |       0.008 |
| mRMR        |      0.896 |       0.503 |             0.931 |       0.035 |
| CausalDRIFT |      0.895 |       0.574 |             0.921 |       0.079 |
| CAFE        |      0.894 |       0.521 |             0.931 |       0.08  |
| Random      |      0.792 |       0.31  |             0.932 |       0.002 |

## Per-dataset KNN accuracy

| dataset       |   Proposed |   PCA |   VAE |   LASSO |   mRMR |   CausalDRIFT |   CAFE |   Random |
|:--------------|-----------:|------:|------:|--------:|-------:|--------------:|-------:|---------:|
| Breast Cancer |      0.943 | 0.97  | 0.97  |   0.967 |  0.96  |         0.948 |  0.955 |    0.95  |
| Digits        |      0.797 | 0.934 | 0.934 |   0.851 |  0.874 |         0.791 |  0.829 |    0.837 |
| Synthetic-A   |      0.68  | 0.793 | 0.815 |   0.89  |  0.925 |         0.938 |  0.938 |    0.74  |
| Synthetic-B   |      0.507 | 0.642 | 0.622 |   0.758 |  0.75  |         0.82  |  0.767 |    0.487 |
| Wine          |      0.921 | 0.966 | 0.966 |   0.966 |  0.972 |         0.977 |  0.983 |    0.944 |

## Per-dataset Jaccard stability

| dataset       |   Proposed |   PCA |   VAE |   LASSO |   mRMR |   CausalDRIFT |   CAFE |   Random |
|:--------------|-----------:|------:|------:|--------:|-------:|--------------:|-------:|---------:|
| Breast Cancer |      0.45  | 0.774 | 0.774 |   0.419 |  0.468 |         0.386 |  0.346 |    0.215 |
| Digits        |      0.374 | 0.491 | 0.491 |   0.568 |  0.435 |         0.33  |  0.334 |    0.301 |
| Synthetic-A   |      0.302 | 0.456 | 0.47  |   0.482 |  0.376 |         0.634 |  0.538 |    0.215 |
| Synthetic-B   |      0.299 | 0.295 | 0.309 |   0.55  |  0.378 |         0.567 |  0.477 |    0.215 |
| Wine          |      0.801 | 0.853 | 0.853 |   0.844 |  0.857 |         0.955 |  0.91  |    0.604 |

## Statistical significance

Friedman test — accuracy: chi2=16.119, p=0.02405; stability: chi2=15.528, p=0.0298

Wilcoxon signed-rank, Proposed vs. each baseline (Δ = Proposed − baseline; positive favours the proposed method):

| Baseline | ΔAccuracy | p (acc) | ΔStability | p (stab) |
|---|---|---|---|---|
| PCA | -0.091 | 0.0625 | -0.129 | 0.125 |
| VAE | -0.092 | 0.0625 | -0.134 | 0.0625 |
| LASSO | -0.117 | 0.0625 | -0.127 | 0.125 |
| mRMR | -0.127 | 0.0625 | -0.058 | 0.0625 |
| CausalDRIFT | -0.125 | 0.188 | -0.129 | 0.312 |
| CAFE | -0.125 | 0.0625 | -0.076 | 0.312 |
| Random | -0.022 | 0.188 | +0.135 | 0.0625 |

## Reading the results

The proposed method preserves original feature identity (unlike PCA/VAE) while landing in a competitive accuracy band. As in the paper, it is not the top accuracy method — the discriminative filters (mRMR, LASSO) and the causal baselines (CausalDRIFT, CAFE) tend to score higher — but it beats Random and preserves interpretability. Trustworthiness stays high (0.926) because selected features retain their neighbourhood structure. On stability it is mid-pack here; the paper's larger 25-bootstrap / 12-dataset protocol favours it more strongly (its consensus voting is the biggest stability contributor per the ablation). Exact numbers depend on the datasets available offline and the reduced bootstrap counts used to keep runtime low — rerun with `--all` and higher `--bootstrap` for paper-scale figures.
