# RESULTS — Oregon Wildfire ML System

*Source: **gee+nifc** data (full run); 7,224 grid cells, base fire rate 1.24%.*

> Numbers below are produced by `scripts/06_evaluate.py` from the saved metrics — re-run after training to refresh. With Earth Engine configured and a full run, rerun the pipeline without `--quick` to populate real-data results.


## 1. Risk model vs. baselines (identical CV splits)

Primary metric is **PR-AUC** (rare-event); higher is better. **lift** = PR-AUC ÷ base rate. **recall@20%** = fraction of real fires caught if we flag the top 20% riskiest cells. **Brier** = calibration (lower better).


### CV scheme: `forward_chaining`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.039 | 3.2x | 0.388 | 0.0136 | 0.630 |
| logistic_weather | 0.022 | 1.5x | 0.317 | 0.2613 | 0.622 |
| risk_gbm **(ours)** | 0.063 | 5.2x | 0.531 | 0.0134 | 0.768 |

### CV scheme: `spatial_block`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.012 | 1.0x | 0.143 | 0.0123 | 0.500 |
| logistic_weather | 0.019 | 1.5x | 0.337 | 0.2400 | 0.608 |
| risk_gbm **(ours)** | 0.048 | 3.9x | 0.561 | 0.0121 | 0.770 |

**Reading it:** the climatology baseline (which only exploits *where* fires recur) typically collapses toward no-skill (PR-AUC ≈ base rate, ROC-AUC ≈ 0.5) under `spatial_block` CV — the clearest demonstration of why random splits mislead. Our GBM, using engineered fire-domain features + ignition-cause priors, should lead on PR-AUC and calibration on **both** schemes.


**Real-data interpretation (important).** We deliberately **exclude `fire_lag1`** (did this cell burn last step?). On real MODIS data fires *persist* week-to-week, so that single feature dominates and inflates PR-AUC to ~0.65 — but it makes the model a fire-**continuation** predictor, not a risk model. The numbers above are the honest **environmental** skill (weather / fuel / terrain / drought only).

Predicting weekly fire in ~36 km² cells from environment alone is genuinely hard, so absolute PR-AUC is modest (base rate ~1.2%). But on the full 24-year pull **with MODIS NDVI added**, vegetation state (`ndvi_roll8`, `ndvi_anom`, `ndvi`) joins the top SHAP drivers and the GBM clearly beats both baselines on **both** CV schemes — PR-AUC ~0.05–0.06, ROC-AUC ~0.77, recall@20% ~0.55. The result is strongest **spatially**: under leave-region-out CV the GBM beats climatology ~4× (climatology collapses to no-skill there) — flagging the top 20% riskiest cells catches ~55% of fires in regions the model never trained on. That spatial generalization is the honest, defensible result; chasing a higher headline with persistence or random splits would be exactly the failure mode this project avoids.


## 2. What drives the risk model (SHAP)

Top features by mean |SHAP| — these should be physical fire drivers, not artifacts:

| # | feature | mean |SHAP| |
|---|---|---|
| 1 | `doy_sin` | 0.3197 |
| 2 | `ndvi_roll8` | 0.2568 |
| 3 | `ndvi_anom` | 0.2037 |
| 4 | `elevation` | 0.1767 |
| 5 | `ndvi` | 0.1631 |
| 6 | `precip_sum8` | 0.1260 |
| 7 | `doy_cos` | 0.1070 |
| 8 | `rmin_roll8` | 0.0966 |
| 9 | `slope` | 0.0812 |
| 10 | `fuel_x_dryness` | 0.0736 |
| 11 | `precip` | 0.0726 |
| 12 | `month` | 0.0721 |

Figure: `outputs/figures/shap_importance.png`.


## 3. Fire-count model (per region / season)

| Model | MAE | RMSE | Poisson deviance | 95% PI coverage |
|---|---|---|---|---|
| negbin | 181.056 | 262.414 | 135.121 | 0.98 |
| poisson | 157.586 | 216.333 | 115.572 | 0.11 |

Forward-chaining (by year) CV. The negative-binomial model handles overdispersion, giving calibrated (wider) intervals than Poisson. Per-region predictions: `outputs/metrics/region_count_predictions.csv`; map: `outputs/maps/fire_count_map.html`.


## 4. Fire-detection CNN (held-out spatial blocks)

Backbone: `efficientnet_b0` (transfer learning, multi-band input).

| PR-AUC | lift | ROC-AUC | recall@20% | Brier |
|---|---|---|---|---|
| 0.807 | 1.3x | 0.733 | 0.306 | 0.3073 |

Evaluated on spatial blocks unseen in training (no leakage). Model: `outputs/models/cnn/fire_detector.pt`.


> **Real Sentinel-2 burn-scar detector** (1,500 patches from 5 fire-season years, MODIS burned-area labels, post-peak S2 median composite, EfficientNet transfer learning, spatial-block split). Honest methodology note: a first attempt with **active-fire** labels + a full-season median scored near chance (ROC-AUC ~0.59) because a median composite **averages the transient fire away**. Switching to **burned-area** labels + a post-peak composite — so the model sees persistent burn scars — lifted it to ROC-AUC ~0.73. It detects burn scars with real skill, though probabilities are under-calibrated (high Brier) and it's below specialized segmentation SOTA (F1 ≈ 0.88–0.97), which use pixel-level labels and time-matched imagery — the clear next step.


## 5. Deliverables

- **Risk heatmap:** `outputs/maps/risk_heatmap.html` (interactive), `outputs/figures/risk_map.png` (static).
- **Fire-count prediction:** `outputs/maps/fire_count_map.html`, `outputs/metrics/region_count_predictions.csv`.
- **Fire detection:** `outputs/models/cnn/fire_detector.pt`, `outputs/metrics/cnn_metrics.json`.


## 6. Caveats

- All metrics use spatially/temporally honest CV; do not compare directly to papers' random-split numbers (see `docs/literature.md`).
