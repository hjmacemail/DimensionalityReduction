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
