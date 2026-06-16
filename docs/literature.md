# Literature Review & Where We Improve

A focused review of recent wildfire-prediction and fire-detection work, the methods
and metrics they report, and the specific places this project does better. The
recurring theme: **many published models are evaluated in ways that overstate their
real-world skill**, and most underuse **ignition cause**. Those are our two edges.

> Scope note: this is a working bibliography for the project, not an exhaustive
> survey. Metrics are quoted as reported by the authors; they are **not** directly
> comparable across different study areas, label definitions, and resolutions —
> which is exactly why we benchmark our own baselines on identical splits (see
> [`../RESULTS.md`](../RESULTS.md)) rather than comparing raw numbers across papers.

---

## 1. Wildfire occurrence / risk modeling

| Work | Method | Reported performance | Validation | Gap we address |
|---|---|---|---|---|
| ELM2.1-XGBfire1.0, *Geosci. Model Dev.* 2025 | XGBoost fire model embedded in a land-surface model (PNNL) | Improved burned-area skill vs. process model | Process-model coupling; regional | No ignition-cause split; we add lightning vs human |
| Daily Wildfire Expansion Rate, *Fire* 2023 (MDPI) | XGBoost / RF / MLP on weather + topo + fuel | ~90% next-day growth-direction accuracy | Train/test split on fire-days | Accuracy on near-balanced target; we use PR-AUC for the *rare* occurrence problem |
| Geo-spatial ML for robust wildfire occurrence, *Sci. Reports* 2025 | RF / boosting on geospatial stack | High AUC | **Uses spatial CV** (good) | We add temporal forward-chaining + ignition cause + calibration |
| Gradient boosting + extreme-value theory, *PMC* 2023 | GBM with EVT tail model | Better tail calibration | Spatiotemporal | We adopt its calibration spirit (prior correction, Brier) |
| ML pipeline survey, *Sci. Total Environ.* 2025 | Survey of the wildfire-ML pipeline | — | — | Confirms inconsistent validation/metrics across the field |

**Takeaways we operationalize:**
- Occurrence is a **rare-event** problem; accuracy/ROC-AUC flatter models. We report
  **PR-AUC, recall@flag-rate, and Brier** as primary.
- The better recent papers already use spatial CV — we go further with **both**
  spatial-block **and** temporal forward-chaining, and we benchmark a climatology
  baseline that *only* exploits spatial recurrence to quantify how much dynamic
  (weather/fuel) skill we actually add.

## 2. Validation rigor & data leakage (our central edge)

- **Kapoor & Narayanan, "Leakage and the reproducibility crisis in ML-based
  science," *Patterns* 2023** — catalogs how leakage (including spatial/temporal)
  inflates reported performance across scientific ML. Directly motivates our design.
- **Random forest & spatial cross-validation, *Environ. Syst. Res.* 2024** — shows
  spatial CV yields *larger, more honest* errors than standard k-fold; standard
  random splits overestimate performance under spatial autocorrelation.
- **Spatial-leakage practices** reported in the literature we adopt: keep all of a
  region (or year) in a single fold; buffer train/test; chronological splits to
  preserve temporal dependence.

**What we do:** `wildfire.eval.cv` implements `forward_chaining` (expanding-window
time CV), `spatial_block` (GroupKFold on H3-parent blocks), and
`leave_one_block_out`. Raw `lon`/`lat`/`year` are excluded as features so trees
can't memorize position/time. This is the difference between a number that looks
good in a notebook and one that holds up on a new region or a future season.

## 3. Ignition cause (the underused signal)

Most occurrence models stop at weather + fuel + topography. Yet **lightning- vs
human-caused** ignitions have very different spatial/temporal signatures (human
ignitions cluster near roads/power/WUI and on weekends; lightning tracks convective
activity). NIFC / FPA-FOD records carry cause, and we turn them into per-cell
historical lightning/human densities plus OSM road/power proximity proxies
(`wildfire.ingest.nifc`, `wildfire.ingest.osm`). SHAP confirms these features carry
real weight (see RESULTS), which is precisely the signal most baselines discard.

## 4. Fire detection from satellite imagery

| Work | Data | Method | Reported | Note |
|---|---|---|---|---|
| de Almeida Pereira et al. 2021 | Landsat-8 | Large-scale active-fire dataset + CNN (MultiScale-Net) | High F1 vs thresholds | Established the CNN-vs-threshold gap |
| Sentinel-2 active-fire framework, 2023 | Sentinel-2 | DL segmentation | **F1 up to ~88%**, +19% over threshold algorithms | Confirms DL > spectral thresholds |
| SEN2FIRE benchmark, 2024 | Sentinel-2 | Segmentation + domain adaptation | Up to **F1 ~97%** with domain adaptation | Benchmark dataset; domain shift matters |
| Sentinel-2 active-fire segmentation (CNN vs Transformers), 2024 | Sentinel-2 | UNet/DeepLabV3+/SegFormer/Mask2Former | SOTA comparisons | Architecture/loss/band ablations |

**What we do:** transfer learning from a pretrained EfficientNet/ResNet (`timm`),
first conv adapted to a **multi-band + spectral-index** stack (NDVI/NBR/NDMI/BAI),
with **spatial-block** train/val/test splits and augmentation. Our emphasis is the
same discipline as the tabular side — no spatial leakage between splits — plus an
honest PR-AUC/recall report on held-out blocks rather than patch-shuffled accuracy.
A documented path swaps the patch-classification head for segmentation to match the
SEN2FIRE-style benchmarks.

## 5. Forecasting & uncertainty

- **Uncertainty-Aware DL for Wildfire Danger Forecasting, arXiv 2025** and
  **Deep Learning for Wildfire Risk Prediction, arXiv 2024 (2405.01607)** integrate
  remote sensing + environmental drivers and stress *calibrated* outputs. We mirror
  this with King–Zeng prior correction (calibrated probabilities after rare-event
  undersampling) and negative-binomial prediction intervals for the count model.

---

## Where this project measurably improves

1. **Validation that doesn't leak.** Spatial-block + temporal forward-chaining CV,
   with a climatology baseline to isolate dynamic skill. Most weaker models report
   random-split numbers that don't transfer.
2. **Ignition-cause features.** Lightning/human history + infrastructure proximity,
   shown to matter via SHAP — signal most occurrence models omit.
3. **Honest rare-event metrics + calibration.** PR-AUC, recall@flag-rate, Brier, and
   prior-corrected probabilities — not accuracy/ROC-AUC on an imbalanced target.
4. **Multimodal + reproducible.** Tabular risk/count **and** an image-recognition CNN
   in one reproducible, config-driven pipeline (uv-locked), with a synthetic
   fallback so results are regenerable end-to-end.

## Sources

- ELM2.1-XGBfire1.0, GMD 2025: <https://gmd.copernicus.org/articles/18/4103/2025/>
- Daily Wildfire Expansion Rate, Fire 2023: <https://www.mdpi.com/2571-6255/6/8/319>
- Geo-spatial ML for robust wildfire occurrence, Sci. Reports 2025: <https://www.nature.com/articles/s41598-025-94002-4>
- Gradient boosting + extreme-value theory, 2023: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10115709/>
- ML pipeline survey, 2025: <https://www.sciencedirect.com/science/article/pii/S1574954125003346>
- RF & spatial cross-validation, Environ. Syst. Res. 2024: <https://link.springer.com/article/10.1186/s40068-024-00352-9>
- Leakage & reproducibility crisis (Kapoor & Narayanan), Patterns 2023: <https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9>
- Active Fire Detection in Landsat-8 (large-scale dataset + DL), 2021: <https://www.researchgate.net/publication/348403289>
- SEN2FIRE benchmark, 2024: <https://www.researchgate.net/publication/383812879>
- Sentinel-2 active-fire segmentation (CNN vs Transformers), 2024: <https://www.researchgate.net/publication/383135469>
- Deep Learning for Wildfire Risk Prediction, arXiv 2024: <https://arxiv.org/html/2405.01607v5>
- Uncertainty-Aware DL for Wildfire Danger Forecasting, arXiv 2025: <https://arxiv.org/pdf/2509.25017>
