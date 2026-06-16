# RESULTS — Oregon Wildfire ML System

*Source: **gee+nifc** data (full run); 7,224 grid cells, base fire rate 1.23%.*

> Numbers below are produced by `scripts/06_evaluate.py` from the saved metrics — re-run after training to refresh. With Earth Engine configured and a full run, rerun the pipeline without `--quick` to populate real-data results.


## 1. Risk model vs. baselines (identical CV splits)

Primary metric is **PR-AUC** (rare-event); higher is better. **lift** = PR-AUC ÷ base rate. **recall@20%** = fraction of real fires caught if we flag the top 20% riskiest cells. **Brier** = calibration (lower better).


### CV scheme: `forward_chaining`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.034 | 2.8x | 0.355 | 0.0137 | 0.608 |
| logistic_weather | 0.022 | 1.5x | 0.316 | 0.2606 | 0.620 |
| risk_gbm **(ours)** | 0.039 | 3.2x | 0.463 | 0.0135 | 0.730 |

### CV scheme: `spatial_block`

| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |
|---|---|---|---|---|---|
| climatology | 0.012 | 1.0x | 0.145 | 0.0122 | 0.500 |
| logistic_weather | 0.020 | 1.6x | 0.354 | 0.2366 | 0.627 |
| risk_gbm **(ours)** | 0.039 | 3.2x | 0.531 | 0.0120 | 0.756 |

**Reading it:** the climatology baseline (which only exploits *where* fires recur) typically collapses toward no-skill (PR-AUC ≈ base rate, ROC-AUC ≈ 0.5) under `spatial_block` CV — the clearest demonstration of why random splits mislead. Our GBM, using engineered fire-domain features + ignition-cause priors, should lead on PR-AUC and calibration on **both** schemes.


**Real-data interpretation (important).** We deliberately **exclude `fire_lag1`** (did this cell burn last step?). On real MODIS data fires *persist* week-to-week, so that single feature dominates and inflates PR-AUC to ~0.65 — but it makes the model a fire-**continuation** predictor, not a risk model. The numbers above are the honest **environmental** skill (weather / fuel / terrain / drought only).

Predicting weekly fire in ~36 km² cells from environment alone is genuinely hard, so absolute PR-AUC is modest (base rate ~1.2%). The model's real value shows up **spatially**: under leave-region-out CV it beats climatology by ~3× (which collapses to no-skill there), with ROC-AUC ≈ 0.75 and recall@20% ≈ 0.53 — i.e. flagging the top 20% riskiest cells catches about half of all fires in regions the model never trained on. That spatial generalization is the honest, defensible result; chasing a higher headline number with persistence or random splits would be exactly the failure mode this project is built to avoid.


## 2. What drives the risk model (SHAP)

Top features by mean |SHAP| — these should be physical fire drivers, not artifacts:

| # | feature | mean |SHAP| |
|---|---|---|
| 1 | `doy_sin` | 0.3570 |
| 2 | `elevation` | 0.1978 |
| 3 | `precip_sum8` | 0.1551 |
| 4 | `doy_cos` | 0.1298 |
| 5 | `slope` | 0.1131 |
| 6 | `fuel_x_dryness` | 0.1061 |
| 7 | `rmin_roll8` | 0.1039 |
| 8 | `fuel_load` | 0.0989 |
| 9 | `bi_roll8` | 0.0903 |
| 10 | `pdsi` | 0.0879 |
| 11 | `precip` | 0.0800 |
| 12 | `wind_roll8` | 0.0484 |

Figure: `outputs/figures/shap_importance.png`.


## 3. Fire-count model (per region / season)

| Model | MAE | RMSE | Poisson deviance | 95% PI coverage |
|---|---|---|---|---|
| negbin | 168.722 | 240.165 | 133.717 | 0.95 |
| poisson | 165.260 | 237.613 | 145.498 | 0.13 |

Forward-chaining (by year) CV. The negative-binomial model handles overdispersion, giving calibrated (wider) intervals than Poisson. Per-region predictions: `outputs/metrics/region_count_predictions.csv`; map: `outputs/maps/fire_count_map.html`.


## 4. Fire-detection CNN (held-out spatial blocks)

Backbone: `efficientnet_b0` (transfer learning, multi-band input).

| PR-AUC | lift | ROC-AUC | recall@20% | Brier |
|---|---|---|---|---|
| 0.993 | 3.1x | 0.996 | 0.621 | 0.0124 |

Evaluated on spatial blocks unseen in training (no leakage). Model: `outputs/models/cnn/fire_detector.pt`.


> ⚠️ The detector is trained on **synthetic** patches (the live Sentinel-2 patch export is implemented but not yet assembled end-to-end). The fire signature there is cleaner than real imagery, so this score is optimistic — it confirms the pipeline, not real-world detection skill. Published Sentinel-2 detectors report F1 ≈ 0.88–0.97 (see `docs/literature.md`).


## 5. Deliverables

- **Risk heatmap:** `outputs/maps/risk_heatmap.html` (interactive), `outputs/figures/risk_map.png` (static).
- **Fire-count prediction:** `outputs/maps/fire_count_map.html`, `outputs/metrics/region_count_predictions.csv`.
- **Fire detection:** `outputs/models/cnn/fire_detector.pt`, `outputs/metrics/cnn_metrics.json`.


## 6. Caveats

- All metrics use spatially/temporally honest CV; do not compare directly to papers' random-split numbers (see `docs/literature.md`).
