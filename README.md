# Causality-Aware Stable Hierarchical Feature Selection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
[![CI](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/ci.yml)

> Replace `YOUR-APP-NAME` (Streamlit URL) and `YOUR-USERNAME/YOUR-REPO` in the badges
> above after you create the GitHub repo and deploy — see **Deploy on the web** below.

A reference implementation of the framework described in *"A Causality-Aware Stable
Hierarchical Feature Selection Framework via Markov Blanket Discovery and Consensus
Selection."* It reduces dimensionality by **selecting original features** (not
projecting them), combining Markov-Blanket causal discovery, hybrid
causal–statistical hierarchical clustering, bootstrap consensus stability, and an
optional Human-in-the-Loop review of ambiguous merges.

## Why this framework

Projection methods (PCA, VAE) predict well but produce latent dimensions with no
direct meaning. Classic selectors (LASSO, mRMR) keep feature identity but are
often unstable under resampling. This framework aims to balance **stability,
causal relevance, interpretability, and accuracy** while preserving the semantic
identity of every selected variable.

## The seven-stage pipeline

| Stage | Description | Module |
|------|-------------|--------|
| 1. Preprocessing | median imputation, z-score, feature transposition | `preprocessing.py` |
| 2. Causal discovery | Markov Blanket (IAMB) + mutual information → relevance `R` (Eq. 1) | `causal.py` |
| 3. Structural mapping | sparsified weighted feature graph, `w = |corr|` (Eq. 2) | `graph.py` |
| 4. Hybrid distance | `d = α·d_stat + (1−α)·d_causal` (Eqs. 3–4) | `distance.py` |
| 5. Agglomeration | average-linkage clustering to `k` clusters (Eq. 5) | `clustering.py` |
| 6. Representative extraction | composite prototype score `S` (Eq. 6) | `clustering.py` |
| 7. Robustness validation | bootstrap consensus voting + Jaccard stability | `consensus.py` |
| + | Human-in-the-Loop merge review | `hitl.py`, `app.py` |

## Install

```bash
pip install -r requirements.txt          # torch is optional (VAE baseline)
```

## Quick start

```python
from causal_hfs import CausalHFS, FrameworkConfig
from causal_hfs.datasets import load_dataset

ds = load_dataset("Breast Cancer")                # sklearn-bundled, offline
model = CausalHFS(FrameworkConfig(n_representatives=12)).fit(ds.X, ds.y)

print(model.selected_features_)   # indices of chosen representatives
print(model.stability_)           # Jaccard stability across bootstraps
X_reduced = model.transform(ds.X)
```

## Run the benchmark (Tables 2–4)

```bash
python run_experiments.py --all-offline --k 10       # offline sklearn datasets
python run_experiments.py --all --k 10               # all 12 (needs OpenML)
python run_experiments.py --csv mydata.csv --label target
```

Outputs land in `results/`: per-dataset metrics, mean-metric table, per-dataset
accuracy matrix, and Friedman + Wilcoxon significance tests.

## Ablation & sensitivity (Tables 5, Figure 4)

```bash
python ablation.py --all-offline --k 10
```

## Human-in-the-Loop app (Section 4.3)

```bash
streamlit run app.py
```

Pick a dataset and α; the app surfaces causally ambiguous merges (two clusters,
their hybrid distances, causal-relevance gap, and a conflicting-role warning),
lets you approve or veto each, then folds the decisions into the final selection
and draws the causal dendrogram (Figure 6).

## Bringing your own data

Any CSV with numeric/categorical features and a label column works:

```python
from causal_hfs.datasets import load_csv
ds = load_csv("path.csv", label_col="target", id_cols=["id"])
```

Categorical columns are label-encoded, `?` is treated as missing and
median-imputed, high-dimensional data is capped to the 120 highest-variance
features, and rows are sub-sampled to ≤600 (Section 4.1 schema).

## Baselines

`PCA`, `VAE` (PyTorch, with an SVD fallback), `LASSO`, `mRMR`, `CausalDRIFT`
(causal MB+MI ranking), `CAFE` (causal-aware greedy selection with a HITL oracle),
and `Random`. All are configured to the same `k` for fair comparison.

## Implementation notes

* **Markov Blanket** uses IAMB with a Fisher-Z partial-correlation CI test — the
  paper's "lightweight PC-skeleton approximation."
* **Causal distance** (not given a closed form in the paper) is defined as the
  normalised absolute difference in causal-relevance scores `R`, matching the
  signals the HITL interface displays.
* `CausalDRIFT` and `CAFE` are described only as named baselines in the paper;
  they are implemented here as reasonable causal selectors and documented as such.
* **Trustworthiness** is the scikit-learn neighbourhood-preservation score between
  the original standardised space and the reduced representation.

Exact numeric values depend on dataset versions, capping/sub-sampling seeds, and
bootstrap counts, so they will not match the paper's tables to three decimals, but
the qualitative behaviour (stability gains over PCA/Random, interpretable
selections, selective HITL intervention) is reproduced.

## Deploy on the web (GitHub → Streamlit Community Cloud)

The interactive app (`app.py`) deploys for free from a GitHub repo:

1. **Push to GitHub**
   ```bash
   cd "Dimensionality Reduction"
   git init
   git add .
   git commit -m "Causal-HFS platform"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. **Deploy** — go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, choose your repo/branch, set **Main file path** to
   `app.py`, and **Deploy**. Streamlit installs `requirements.txt` and serves the app
   at a public URL.

Notes:

* The repo root must contain `app.py` **and** the `causal_hfs/` package (it does).
* `torch` is commented out of `requirements.txt` to keep the cloud build light; the
  VAE baseline falls back to an SVD autoencoder. Uncomment it for the neural VAE.
* Streamlit Cloud has internet, so the download-based datasets (UCI/OpenML, MNIST,
  Olivetti, Isolet) load fine; the four scikit-learn built-ins work with no network.
* `.streamlit/config.toml` sets a light theme and headless server; secrets (if any)
  go in the Streamlit Cloud dashboard, never in the repo.

**Alternatives:** the same repo runs on **Hugging Face Spaces** (choose the Streamlit
SDK, entry point `app.py`) or any container host (**Render**, **Railway**, **Fly.io**)
with `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

# DimensionalityReduction
# DimensionalityReduction
