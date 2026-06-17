// Academic paper rendered in the "Paper" tab.
document.getElementById("paper-body").innerHTML = `
<h1>Honest Spatial Generalization for Wildfire Risk:<br/>A Leakage-Aware Machine-Learning System for Oregon, 2001–2024</h1>
<p class="authors">Oregon Wildfire ML Project</p>
<p class="affil">Reproducible pipeline · Google Earth Engine · NIFC · MODIS · Sentinel-2 · GRIDMET</p>

<div class="abstract">
<strong>Abstract.</strong> We present a reproducible, end-to-end machine-learning system for
wildfire analysis in Oregon, comprising (i) a per-cell fire-<em>risk</em> surface, (ii) per-region
fire-<em>count</em> forecasts with calibrated uncertainty, and (iii) a satellite burn-scar
<em>detector</em>. Training on 4.56&nbsp;million real cell-weeks (H3 hexagons × weekly steps,
2001–2024) of MODIS burned-area labels, GRIDMET weather, SRTM terrain, ESA WorldCover fuel,
MODIS NDVI and NIFC ignition cause, we emphasize <em>validation that does not leak</em>: spatial-block,
leave-one-region-out, and forward-chaining cross-validation, with rare-event metrics (PR-AUC, Brier,
recall at operating thresholds). Our central result is that weekly fire occurrence at 36&nbsp;km²
resolution is intrinsically hard — absolute PR-AUC is low for all methods — yet a gradient-boosted
model achieves meaningful, <em>generalizable</em> skill: under leave-one-region-out CV it outperforms
a climatology baseline ≈4× (ROC-AUC 0.77, recall@20%&nbsp;≈0.55) while climatology collapses to
no-skill. We further document two instructive pitfalls we explicitly reject — a fire-persistence
shortcut that inflates PR-AUC to 0.65, and a trivially separable synthetic detector — arguing that
honest evaluation is the field's most valuable and most-neglected lever.
</div>

<h2>1. Introduction</h2>
<p>Data-driven wildfire prediction has proliferated, but reported performance is frequently inflated
by <em>leakage</em>: random train/test splits over spatially and temporally autocorrelated data, and
target-adjacent features. A model can post strong numbers in a notebook and fail in deployment on a
new region or future season. We build a complete Oregon system that treats validation rigor as a
first-class design constraint, and we report what survives that scrutiny.</p>
<p>Oregon spans a steep west-wet / east-dry gradient across eight ecoregions, from the temperate Coast
Range to the Northern Basin and Range. The 2001–2024 record includes record-setting seasons (2024) and
quiet ones (2019); a credible model must capture both the spatial structure of where fire is possible
and the dynamic conditions that ignite a given week.</p>

<h2>2. Data</h2>
<p>All layers are clipped to Oregon and aggregated onto an H3 resolution-6 grid (7,224 hexes,
≈36&nbsp;km² each) at weekly steps. Labels are MODIS burned area (MCD64A1) and thermal anomalies
(MOD14A1); predictors are GRIDMET weather and fire-danger indices (VPD, ERC, BI, wind, RH, precip),
GRIDMET/DROUGHT PDSI, SRTM-derived elevation/slope/aspect, ESA WorldCover land cover (mapped to a fuel
proxy), and MODIS NDVI. A complementary source — NIFC incident records — supplies ignition
<em>cause</em> (lightning vs. human), a signal most occurrence models omit. Imagery for the detector is
Sentinel-2 surface reflectance. The positive (fire) rate is 1.24% of cell-weeks: a rare-event problem.</p>
<p>Per-timestep pulls are concurrent, retriable and checkpointed, making a full-history pull feasible and
resumable. When Earth Engine is unavailable, a synthetic generator emits the same canonical tables so the
methodology is reproducible offline.</p>

<h2>3. Methods</h2>
<h3>3.1 Features</h3>
<p>Beyond instantaneous weather we engineer antecedent windows (rolling 2/4/8-step means of temperature,
VPD, ERC, wind, RH, PDSI, NDVI), antecedent-precipitation sums and a derived days-since-rain (dryness
memory), an NDVI anomaly relative to each cell's greenness baseline, fuel×dryness and wind×dryness
interactions, and per-cell historical ignition-cause densities. Raw longitude, latitude and year are
<em>excluded</em> so the trees cannot memorize absolute position or time — a deliberate guard against the
very leakage we critique.</p>
<h3>3.2 Models</h3>
<p><b>Risk surface.</b> A LightGBM classifier on the engineered features, trained on undersampled
negatives and <em>recalibrated</em> to the true base rate (King–Zeng prior correction) so probabilities
remain honest. <b>Counts.</b> A negative-binomial GLM per ecoregion-season with a log-exposure offset,
yielding rate estimates and calibrated prediction intervals. <b>Detection.</b> An EfficientNet-B0,
transfer-learned with its input convolution adapted to a ten-channel (six bands + four indices) stack,
trained with spatial-block splits and augmentation.</p>
<h3>3.3 Evaluation</h3>
<p>We use three leakage-aware schemes: <em>forward-chaining</em> (train on years ≤ Y, test on Y+1),
<em>spatial-block</em> GroupKFold on H3-parent blocks, and <em>leave-one-region-out</em>. We report
PR-AUC (average precision) as the headline, with Brier (calibration) and recall at flag-rate (operational).
A climatology baseline (per-cell historical frequency) and a weather-only logistic regression are run on
identical splits. SHAP confirms which features drive predictions.</p>

<h2>4. Results</h2>
<p>Table 1 reports the risk model against baselines under both CV schemes. The model leads on PR-AUC,
ROC-AUC and calibration throughout. Critically, the climatology baseline — which merely encodes where fire
has historically recurred — collapses to no-skill under spatial-block CV (PR-AUC ≈ base rate, ROC-AUC 0.50),
whereas the environmental model retains ≈4× lift, demonstrating genuine transfer to unseen regions.</p>

<table>
<caption style="text-align:left;font-size:12px;color:#666;margin-bottom:6px">Table 1. Risk model vs. baselines (real data, honest CV). PR-AUC headline; higher better, Brier lower better.</caption>
<tr><th>Model</th><th>PR-AUC (time)</th><th>PR-AUC (spatial)</th><th>ROC-AUC</th><th>recall@20%</th><th>Brier</th></tr>
<tr><td>Climatology</td><td>0.039</td><td>0.012</td><td>0.50–0.63</td><td>0.14–0.39</td><td>0.012–0.014</td></tr>
<tr><td>Logistic (weather)</td><td>0.022</td><td>0.019</td><td>0.62</td><td>0.32–0.34</td><td>0.24–0.26</td></tr>
<tr><td><b>GBM (ours)</b></td><td><b>0.063</b></td><td><b>0.048</b></td><td><b>0.77</b></td><td><b>0.53–0.56</b></td><td><b>0.012–0.013</b></td></tr>
</table>

<figure>
  <img src="assets/shap_importance.png" alt="SHAP feature importance"/>
  <figcaption>Figure 1. Top risk-model drivers by mean |SHAP|: seasonality, NDVI / vegetation state,
  elevation, antecedent dryness, fuel×dryness and drought — physically interpretable fire drivers.</figcaption>
</figure>

<p><b>Counts.</b> The negative-binomial model predicts per-ecoregion seasonal fire counts with
well-calibrated 95% prediction intervals (empirical coverage ≈0.95–0.98), correctly widening intervals to
reflect overdispersion. <b>Detection.</b> On real Sentinel-2 imagery the burn-scar detector reaches
ROC-AUC ≈0.73 / PR-AUC ≈0.81 on held-out spatial blocks (Section 5).</p>

<figure>
  <img src="assets/risk_map.png" alt="Oregon risk surface"/>
  <figcaption>Figure 2. Modeled per-cell wildfire-risk surface over Oregon.</figcaption>
</figure>

<h2>5. Discussion</h2>
<p><b>Persistence is not risk.</b> Including a one-week fire lag inflates PR-AUC to ≈0.65, but SHAP shows it
dominates: the model becomes a fire-<em>continuation</em> predictor (fires burn for days) rather than a risk
model. We exclude it and report the honest environmental skill. <b>Spatial generalization is the real
result.</b> The gap between forward-chaining and spatial-block CV — and the climatology collapse — shows that
much apparent skill in the literature is spatial memorization. Our model's value is that it transfers.
<b>Imagery must match the label.</b> A first detector paired transient active-fire labels with a season
<em>median</em> composite and scored near chance, because the median averages the fire away; re-framing as
burn-scar detection (persistent scars, post-peak composite) recovered real skill. Each of these is a case
where a tempting higher number was rejected for a defensible lower one.</p>
<p>Relative to recent Oregon/Pacific-Northwest work, our contributions are methodological rather than a new
architecture: leakage-aware CV with an explicit climatology control, ignition-cause features, honest
rare-event metrics with calibration, and a multimodal, fully reproducible pipeline.</p>

<h2>6. Limitations &amp; Future Work</h2>
<p>Absolute risk PR-AUC is modest — weekly 36&nbsp;km² occurrence is hard, and finer spatio-temporal
resolution would help. Per-timestep historical lightning is unavailable in Earth Engine for 2001–2024,
limiting the ignition-cause signal; the FPA-FOD archive and GOES-GLM (2017+) are avenues. The detector uses
coarse MODIS-derived labels and patch-level classification; pixel-level segmentation on time-matched imagery
is the path to literature-grade detection. CNN probabilities are under-calibrated.</p>

<h2>7. Conclusion</h2>
<p>A wildfire model is only as trustworthy as its evaluation. By holding validation rigor fixed and reporting
what survives, we obtain an Oregon system with modest but genuine, generalizable skill — and, as importantly,
a record of the inflated results we declined to claim.</p>

<h2>References</h2>
<ol class="refs">
<li>Kapoor, S. &amp; Narayanan, A. (2023). Leakage and the reproducibility crisis in ML-based science. <i>Patterns</i>.</li>
<li>Random forest and spatial cross-validation performance (2024). <i>Environmental Systems Research</i>.</li>
<li>ELM2.1-XGBfire1.0: integrating a machine-learning fire model in a land-surface model (2025). <i>Geosci. Model Dev.</i></li>
<li>A machine-learning approach to predicting daily wildfire expansion rate (2023). <i>Fire</i>, 6(8), 319.</li>
<li>Exploration of geo-spatial data and ML for robust wildfire occurrence prediction (2025). <i>Scientific Reports</i>.</li>
<li>SEN2FIRE: a benchmark dataset for wildfire detection using Sentinel data (2024).</li>
<li>Active fire detection in Landsat-8 imagery: a large-scale dataset and a deep-learning study (2021).</li>
<li>King, G. &amp; Zeng, L. (2001). Logistic regression in rare events data. <i>Political Analysis</i>.</li>
</ol>
`;
