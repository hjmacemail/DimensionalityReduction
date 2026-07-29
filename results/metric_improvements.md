# Improving accuracy, stability & trustworthiness — what worked

Two mechanisms were proposed (cluster-level consensus, predictive-aware prototype).
Both were implemented and measured; **neither helped**, and the honest wins turned
out to be elsewhere. All numbers are means over Wine + the Causal-Benchmark.

## What worked

| Configuration | Accuracy | Stability | Trustworthiness |
|---|---:|---:|---:|
| Old default (half bootstrap budget) | 0.912 | 0.572 | 0.915 |
| **New default (full consensus budget)** | **0.915** | **0.631** | **0.922** |
| New default + wrapper refinement | 0.934 | 0.700 | 0.919 |

- **Full consensus budget (new default).** The strict-causal mode previously used
  only half the bootstrap resamples for the internal consensus vote. Using the full
  budget is a *clean Pareto improvement*: stability +0.06, trustworthiness +0.007,
  accuracy +0.003 — nothing regresses. This is now the default. Cost: proportionally
  more compute.
- **Wrapper refinement (opt-in).** A light forward-swap over the downstream KNN score,
  run once after selection (`wrapper_refine=True`, or the *Accuracy refinement* toggle
  in the app). It adds ~2% accuracy; its effect on stability is noisy (helped in this
  run, hurt in another), so it is off by default and offered as an accuracy-focused
  option. Cost: extra runtime.

## What did NOT work (and is off by default)

- **Cluster-level consensus** (`cluster_consensus=True`). Intended to raise stability
  by voting at the redundancy-group level. It *lowered* stability (≈0.51 → 0.37),
  because the reported metric re-fits the whole pipeline on each resample, so every
  resample re-derives its own clustering — the group mapping adds variance rather than
  removing it. Kept as an available option but not recommended.
- **Predictive-aware prototype** (`prototype_by="predictive"`). In strict-MB mode the
  candidate pool is already the (clean) Markov Blanket, so "most predictive member"
  and "most causally-relevant member" coincide — it produced identical results to the
  relevance prototype. Available for the non-strict / composite regime where it can
  differ.

## Causal-aware greedy (reviewer's improvement, measured)

The reviewer's central critique: the causal score weighted clustering but was
**ignored by the default greedy selector**, which ranked purely on `rel_i`. The
`improved_greedy` mode fixes this — the greedy is seeded at `argmax R` and scored
by the rank-normalised causal-predictive `R = λ·rank(C) + (1−λ)·rank(rel)`, so
Markov-Blanket membership (via `λ`) genuinely drives selection.

Measured vs. the pure-relevance greedy (k=8, RF relevance, 4 eval bootstraps):

| Dataset | pure-rel acc | improved acc | pure-rel stab | improved stab |
|---|---:|---:|---:|---:|
| Wine | 0.966 | **0.978** | 0.748 | **0.785** |
| Breast Cancer | 0.953 | **0.961** | 0.527 | **0.591** |

Improves **both accuracy and stability on both datasets** — now the default in the
app's soft config (`improved_greedy=True`).

What did NOT help (implemented but off by default, honest reporting): **bootstrap MB
confidence** (`mb_bootstrap>0`) was noisy and dropped BC stability 0.591→0.384;
**group-diversity reward** + **mean-redundancy blend** added cross-resample churn that
reduced stability; and a **high causal weight** (`λ=0.5`) let the MB mask dominate and
collapsed BC to acc 0.939 / stab 0.329 — `improved_lam=0.15` is the measured sweet spot.

## Takeaways

- The reliable lever for **stability** here is simply **more consensus resamples**, not
  a cleverer voting rule — because reported stability is dominated by how robust the
  underlying redundancy structure is to data perturbation.
- **Accuracy** responds to the **wrapper refinement** (filter → wrapper hybrid).
- **Trustworthiness** barely moves (~0.92); feature selection has a lower ceiling than
  projection methods (PCA/VAE ≈ 0.97). Meaningfully raising it would require a
  neighbourhood-preservation objective in the prototype score — a larger change with
  an interpretability trade-off, deferred.

All flags default off except the full-consensus budget; the paper-faithful behaviour
is unchanged unless the corresponding option is enabled.
