# RESULTS — Oregon Wildfire ML System

*Source: **synthetic** data (full run); 7,224 grid cells, base fire rate 3.02%.*

> Numbers below are produced by `scripts/06_evaluate.py` from the saved metrics — re-run after training to refresh. With Earth Engine configured and a full run, rerun the pipeline without `--quick` to populate real-data results.


## 1. Risk model vs. baselines (identical CV splits)

Primary metric is **PR-AUC** (rare-event); higher is better. **lift** = PR-AUC ÷ base rate. **recall@20%** = fraction of real fires caught if we flag the top 20% riskiest cells. **Brier** = calibration (lower better).


### CV scheme: `forward_chaining`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.067 | 2.2x | 0.470 | 0.0286 | 0.753 |
| logistic_weather | 0.333 | 11.0x | 0.818 | 0.1326 | 0.901 |
| risk_gbm **(ours)** | 0.476 | 15.8x | 0.890 | 0.0209 | 0.935 |

### CV scheme: `spatial_block`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.030 | 1.0x | 0.378 | 0.0293 | 0.500 |
| logistic_weather | 0.335 | 11.2x | 0.816 | 0.1320 | 0.901 |
| risk_gbm **(ours)** | 0.476 | 15.9x | 0.889 | 0.0208 | 0.934 |

**Reading it:** the climatology baseline (which only exploits *where* fires recur) typically collapses toward no-skill (PR-AUC ≈ base rate, ROC-AUC ≈ 0.5) under `spatial_block` CV — the clearest demonstration of why random splits mislead. Our GBM, using engineered fire-domain features + ignition-cause priors, should lead on PR-AUC and calibration on **both** schemes.


## 2. What drives the risk model (SHAP)

Top features by mean |SHAP| — these should be physical fire drivers, not artifacts:

| # | feature | mean |SHAP| |
|---|---|---|
| 1 | `erc` | 1.1545 |
| 2 | `lightning_density` | 0.5629 |
| 3 | `doy_sin` | 0.4433 |
| 4 | `days_since_rain` | 0.3748 |
| 5 | `doy_cos` | 0.3321 |
| 6 | `bi` | 0.1999 |
| 7 | `pdsi` | 0.1884 |
| 8 | `erc_roll2` | 0.1644 |
| 9 | `fuel_load` | 0.1487 |
| 10 | `ndvi_anom` | 0.1478 |
| 11 | `precip` | 0.1309 |
| 12 | `month` | 0.1089 |

Figure: `outputs/figures/shap_importance.png`.


## 3. Fire-count model (per region / season)

| Model | MAE | RMSE | Poisson deviance | 95% PI coverage |
|---|---|---|---|---|
| negbin | 125.581 | 309.255 | 22.468 | 0.98 |
| poisson | 28.738 | 42.492 | 4.656 | 0.61 |

Forward-chaining (by year) CV. The negative-binomial model handles overdispersion, giving calibrated (wider) intervals than Poisson. Per-region predictions: `outputs/metrics/region_count_predictions.csv`; map: `outputs/maps/fire_count_map.html`.


## 4. Fire-detection CNN (held-out spatial blocks)

Backbone: `efficientnet_b0` (transfer learning, multi-band input).

| PR-AUC | lift | ROC-AUC | recall@20% | Brier |
|---|---|---|---|---|
| 0.993 | 3.1x | 0.996 | 0.621 | 0.0124 |

Evaluated on spatial blocks unseen in training (no leakage). Model: `outputs/models/cnn/fire_detector.pt`.


> ⚠️ On **synthetic** patches the fire signature is cleaner than real imagery, so this score is optimistic — it confirms the pipeline, not real-world detection skill. Published Sentinel-2 detectors report F1 ≈ 0.88–0.97 (see `docs/literature.md`).


## 5. Deliverables

- **Risk heatmap:** `outputs/maps/risk_heatmap.html` (interactive), `outputs/figures/risk_map.png` (static).
- **Fire-count prediction:** `outputs/maps/fire_count_map.html`, `outputs/metrics/region_count_predictions.csv`.
- **Fire detection:** `outputs/models/cnn/fire_detector.pt`, `outputs/metrics/cnn_metrics.json`.


## 6. Caveats

- These numbers are from the **synthetic** generator (latent fire model), which validates the methodology end-to-end but is **not** a claim about real Oregon skill. Configure Earth Engine and run without `--quick` for real-data results.

- All metrics use spatially/temporally honest CV; do not compare directly to papers' random-split numbers (see `docs/literature.md`).
