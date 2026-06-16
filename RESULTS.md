# RESULTS — Oregon Wildfire ML System

*Source: **synthetic** data (quick sample); 800 grid cells, base fire rate 3.87%.*

> Numbers below are produced by `scripts/06_evaluate.py` from the saved metrics — re-run after training to refresh. With Earth Engine configured and a full run, rerun the pipeline without `--quick` to populate real-data results.


## 1. Risk model vs. baselines (identical CV splits)

Primary metric is **PR-AUC** (rare-event); higher is better. **lift** = PR-AUC ÷ base rate. **recall@20%** = fraction of real fires caught if we flag the top 20% riskiest cells. **Brier** = calibration (lower better).


### CV scheme: `forward_chaining`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.115 | 3.0x | 0.526 | 0.0354 | 0.734 |
| logistic_weather | 0.379 | 10.0x | 0.836 | 0.1322 | 0.912 |
| risk_gbm **(ours)** | 0.499 | 13.1x | 0.899 | 0.0270 | 0.937 |

### CV scheme: `spatial_block`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.039 | 1.0x | 0.247 | 0.0374 | 0.500 |
| logistic_weather | 0.374 | 11.9x | 0.828 | 0.1233 | 0.908 |
| risk_gbm **(ours)** | 0.486 | 15.4x | 0.886 | 0.0257 | 0.934 |

**Reading it:** the climatology baseline (which only exploits *where* fires recur) typically collapses toward no-skill (PR-AUC ≈ base rate, ROC-AUC ≈ 0.5) under `spatial_block` CV — the clearest demonstration of why random splits mislead. Our GBM, using engineered fire-domain features + ignition-cause priors, should lead on PR-AUC and calibration on **both** schemes.


## 2. What drives the risk model (SHAP)

Top features by mean |SHAP| — these should be physical fire drivers, not artifacts:

| # | feature | mean |SHAP| |
|---|---|---|
| 1 | `erc` | 1.3705 |
| 2 | `lightning_density` | 0.4700 |
| 3 | `days_since_rain` | 0.2890 |
| 4 | `precip` | 0.2694 |
| 5 | `fuel_load` | 0.2016 |
| 6 | `bi` | 0.1614 |
| 7 | `doy_sin` | 0.1569 |
| 8 | `erc_roll2` | 0.1423 |
| 9 | `pdsi` | 0.1389 |
| 10 | `ndvi_roll8` | 0.1250 |
| 11 | `erc_roll8` | 0.1146 |
| 12 | `elevation` | 0.1092 |

Figure: `outputs/figures/shap_importance.png`.


## 3. Fire-count model (per region / season)

| Model | MAE | RMSE | Poisson deviance | 95% PI coverage |
|---|---|---|---|---|
| negbin | 9.481 | 14.728 | 4.419 | 0.72 |
| poisson | 9.834 | 15.196 | 4.672 | 0.70 |

Forward-chaining (by year) CV. The negative-binomial model handles overdispersion, giving calibrated (wider) intervals than Poisson. Per-region predictions: `outputs/metrics/region_count_predictions.csv`; map: `outputs/maps/fire_count_map.html`.


## 4. Fire-detection CNN (held-out spatial blocks)

Backbone: `efficientnet_b0` (transfer learning, multi-band input).

| PR-AUC | lift | ROC-AUC | recall@20% | Brier |
|---|---|---|---|---|
| 0.733 | 1.8x | 0.810 | 0.420 | 0.2523 |

Evaluated on spatial blocks unseen in training (no leakage). Model: `outputs/models/cnn/fire_detector.pt`.


## 5. Deliverables

- **Risk heatmap:** `outputs/maps/risk_heatmap.html` (interactive), `outputs/figures/risk_map.png` (static).
- **Fire-count prediction:** `outputs/maps/fire_count_map.html`, `outputs/metrics/region_count_predictions.csv`.
- **Fire detection:** `outputs/models/cnn/fire_detector.pt`, `outputs/metrics/cnn_metrics.json`.


## 6. Caveats

- These numbers are from the **synthetic** generator (latent fire model), which validates the methodology end-to-end but is **not** a claim about real Oregon skill. Configure Earth Engine and run without `--quick` for real-data results.

- All metrics use spatially/temporally honest CV; do not compare directly to papers' random-split numbers (see `docs/literature.md`).
