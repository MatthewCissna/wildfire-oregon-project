# Oregon Wildfire ML System

A reproducible, production-quality wildfire analysis and prediction system for
Oregon, built to **outperform published baselines** through better validation and
features rather than just bigger models. Three deliverables:

1. **Risk heatmap** — a per-cell fire-probability surface over Oregon (gradient-boosted
   trees), rendered as an interactive [folium](https://python-visualization.github.io/folium/)
   map and a static high-resolution map.
2. **Fire-count prediction** — expected number of fires per region (county / ecoregion)
   per season, with uncertainty intervals (Poisson / negative-binomial / GBM count).
3. **Fire detection** — a transfer-learning CNN (EfficientNet/ResNet) that classifies
   satellite image patches as **fire / no-fire**, using FIRMS/MODIS active-fire labels
   aligned to Sentinel-2 imagery.

### Why this beats weaker published models (the whole point)

- **Spatially & temporally honest validation.** Block CV, leave-one-ecoregion-out,
  and forward-chaining in time — never random splits, which leak and inflate scores.
- **Honest rare-event metrics.** PR-AUC, precision/recall at operating thresholds,
  and Brier score — not accuracy or ROC-AUC, which flatter on imbalanced data.
- **Ignition-cause signal.** NIFC records distinguish lightning vs. human ignition;
  most models ignore this. We engineer human-ignition proxies (roads, power lines).
- **Real baselines** benchmarked on identical splits, and SHAP to confirm the model
  learns fire physics, not artifacts.

See [`docs/`](docs/) for the methodology, data-source justification, and literature
comparison, and [`RESULTS.md`](RESULTS.md) for measured performance vs. baselines.

---

## Quick start

> **Prerequisites:** Windows/macOS/Linux, [`uv`](https://docs.astral.sh/uv/) installed,
> and (for the CNN) an NVIDIA GPU. Python itself is managed by `uv`.

```powershell
# 1. Install uv (once), if you don't have it:
#    PowerShell:  irm https://astral.sh/uv/install.ps1 | iex
#    macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment and install core deps (ingest, features, tree models, viz):
uv sync

# 3. (Optional) Add the fire-detection CNN deps — pulls the CUDA build of PyTorch:
uv sync --extra cnn

# 4. Smoke-test the WHOLE pipeline on synthetic data (no Earth Engine needed yet):
uv run python scripts/run_pipeline.py --synthetic --quick
```

Step 4 ingests synthetic data, builds features, trains the risk + count models,
evaluates them with spatial CV, and writes a heatmap to `outputs/maps/` — proving
the pipeline end-to-end before any real data is pulled.

### Interactive website

A self-contained atlas — clickable risk map with per-cell detail, results, data
explorer, and a paper — lives in [`site/`](site/). Build its data and serve it:

```powershell
uv run python scripts/build_site.py                 # refresh site/data from outputs
uv run python -m http.server 8000 --directory site  # open http://localhost:8000
```

(Or just double-click `site/index.html`.) See [`site/README.md`](site/README.md).

**Auto-deploy to GitHub Pages + weekly automated updates.** The repo ships with
two GitHub Actions workflows:

- [`.github/workflows/pages.yml`](.github/workflows/pages.yml) — deploys `site/`
  to GitHub Pages on every push to `main`.
- [`.github/workflows/update.yml`](.github/workflows/update.yml) — runs every
  Monday during fire season, pulls the latest MODIS labels via Earth Engine
  (using a service-account key from GitHub Secrets), refreshes
  `site/data/predictions.{json,js}` with the actuals, and commits — which
  triggers `pages.yml` and the live site updates itself.

One-time setup: see [`docs/github_deploy_setup.md`](docs/github_deploy_setup.md).

### Set up Earth Engine (for real data)

The only manual step. Follow **[`docs/earth_engine_setup.md`](docs/earth_engine_setup.md)**
(~15 min, free for students). When done, put your Cloud project ID in
[`configs/config.yaml`](configs/config.yaml):

```yaml
earth_engine:
  project_id: "ee-yourname-wildfire"
```

then verify:

```powershell
uv run python scripts/00_check_earth_engine.py
```

---

## Running the pipeline (stage by stage)

Each stage is an idempotent script under [`scripts/`](scripts/). They read
[`configs/config.yaml`](configs/config.yaml) and write to `data/` and `outputs/`.

| # | Stage | Script | Output |
|---|-------|--------|--------|
| 0 | Check Earth Engine auth | `scripts/00_check_earth_engine.py` | console |
| 1 | Ingest (GEE + NIFC + OSM) | `scripts/01_ingest.py` | `data/raw/`, `data/interim/` |
| 2 | Build feature grid | `scripts/02_features.py` | `data/processed/features.parquet` |
| 3 | Extract image patches (CNN) | `scripts/03_patches.py` | `data/patches/` |
| 4 | Train risk + count models | `scripts/04_train_tabular.py` | `outputs/models/` |
| 5 | Train detection CNN | `scripts/05_train_cnn.py` | `outputs/models/cnn/` |
| 6 | Evaluate → RESULTS.md | `scripts/06_evaluate.py` | `RESULTS.md`, `outputs/metrics/` |
| 7 | Visualize (maps) | `scripts/07_visualize.py` | `outputs/maps/`, `outputs/figures/` |
| 8 | (optional) Optuna tuning | `scripts/08_tune.py` | `outputs/models/*_best_params.json` |

Add `--synthetic` to any stage to use the offline fallback. The orchestrator
`scripts/run_pipeline.py` chains them (`--quick` uses a small sample / few epochs).

**Hyperparameter tuning (optional).** `uv run python scripts/08_tune.py --target risk`
runs an Optuna study that optimizes **spatial-block CV PR-AUC** (not a leaky random
split) and saves the best params; the next `04_train_tabular.py` run picks them up
automatically. Add `--target cnn` to tune the detector (needs the `cnn` extra).

**Notebooks.** `notebooks/01–04` walk through the project with plots — data
exploration & ecoregions, features & class imbalance, modeling & honest validation
(incl. the climatology-collapse demo and SHAP), and the maps + CNN. Launch with
`uv run jupyter lab`.

---

## Project structure

```
wildfire-oregon-project/
├── configs/config.yaml        # all settings (GEE project id, grid, datasets, model params)
├── docs/
│   ├── earth_engine_setup.md  # student GEE onboarding (do this once)
│   ├── data_sources.md        # dataset comparison & justification
│   └── literature.md          # related work + where we improve
├── data/                      # raw / interim / processed / patches  (git-ignored)
├── outputs/                   # maps / models / metrics / figures     (git-ignored)
├── notebooks/                 # exploratory notebooks
├── scripts/                   # numbered, runnable pipeline stages
├── src/wildfire/
│   ├── config.py              # typed config loader (env overrides)
│   ├── ingest/                # Earth Engine pulls, NIFC, OSM, synthetic fallback
│   ├── features/              # spatiotemporal grid + fire-domain features
│   ├── models/                # risk GBM, count model, detection CNN, baselines
│   ├── eval/                  # spatial/temporal CV, rare-event metrics, SHAP
│   └── viz/                   # folium + static maps
├── tests/                     # pytest smoke tests
├── pyproject.toml             # deps (uv) + tooling
└── RESULTS.md                 # metrics vs. baselines & literature
```

> **Note on `src/wildfire/`:** the requested `ingest/features/models/eval/viz`
> layout lives inside a proper installable `wildfire` package, so imports are clean
> (`from wildfire.features import grid`) and there's no `sys.path` hacking.

---

## Reproducibility

- **`uv.lock`** pins every transitive dependency; it is committed. `uv sync`
  reproduces the exact environment.
- A single **random seed** (`project.random_seed` in config) threads through
  sampling, splits, and model training.
- No credentials are stored in the repo. The Earth Engine project ID is config; the
  OAuth token lives in your user profile and is git-ignored.

## Hardware notes

- Tabular models (risk, count) run fine on CPU.
- The detection CNN expects an NVIDIA GPU. The CUDA build of PyTorch is pinned via
  the `cu121` index in `pyproject.toml`; switch to `cu124`/`cpu` there if needed.
  For full training without a local GPU, the patch dataset and training script also
  run on Google Colab.

## License

MIT (see `pyproject.toml`). Data sources retain their respective licenses — see
[`docs/data_sources.md`](docs/data_sources.md).
