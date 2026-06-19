// Academic paper rendered in the "Paper" tab.
document.getElementById("paper-body").innerHTML = `
<h1>Leakage-Aware Spatial Generalization for Wildfire Risk:<br/>A Machine-Learning System for Oregon, 2001–2024</h1>
<p class="authors">Oregon Wildfire ML Project</p>
<p class="affil">Reproducible pipeline · Google Earth Engine · NIFC · MODIS · Sentinel-2 · GRIDMET</p>

<div class="abstract">
<strong>Abstract.</strong> We present a reproducible, end-to-end machine-learning system for
wildfire analysis in Oregon. It has three parts: a per-cell fire-<em>risk</em> surface, per-region
fire-<em>count</em> forecasts with calibrated uncertainty, and a satellite burn-scar
<em>detector</em>. We train on 4.56&nbsp;million real cell-weeks (H3 hexagons × weekly steps,
2001–2024) of MODIS burned-area labels, GRIDMET weather, SRTM terrain, ESA WorldCover fuel,
MODIS NDVI and NIFC ignition cause, and we hold the validation to a standard that doesn't leak:
spatial-block, leave-one-region-out, and forward-chaining cross-validation, scored with rare-event
metrics (PR-AUC, Brier, recall at operating thresholds). The central result is that weekly fire
occurrence at 36&nbsp;km² resolution is hard, with low absolute PR-AUC for every method, yet a
gradient-boosted model still shows skill that <em>generalizes</em>: under leave-one-region-out CV it
beats a climatology baseline by about 4× (ROC-AUC 0.77, recall@20%&nbsp;≈0.55) while climatology
falls to no-skill. We also report two results we found and rejected: a fire-persistence shortcut
that inflates PR-AUC to 0.65, and a synthetic detector that was trivially separable. The point we
want to make is narrow but practical: in this setting, the evaluation design decides which numbers
you can believe.
</div>

<h2>1. Introduction</h2>
<p>Data-driven wildfire prediction has grown quickly, but reported performance is often inflated by
<em>leakage</em>: random train/test splits over data that is correlated in space and time, and
features that sit too close to the target. A model can post strong numbers in a notebook and then
fail on a new region or a future season. We build a complete Oregon system, fix the validation rules
up front, and report only what survives them.</p>
<p>Oregon runs a steep west-wet to east-dry gradient across eight ecoregions, from the temperate Coast
Range to the Northern Basin and Range. The 2001–2024 record includes record-setting seasons (2024) and
quiet ones (2019). A credible model has to capture both the spatial structure of where fire is possible
and the conditions that set off a given week.</p>

<h2>2. Data</h2>
<p>All layers are clipped to Oregon and aggregated onto an H3 resolution-6 grid (7,224 hexes,
≈36&nbsp;km² each) at weekly steps. Labels are MODIS burned area (MCD64A1) and thermal anomalies
(MOD14A1); predictors are GRIDMET weather and fire-danger indices (VPD, ERC, BI, wind, RH, precip),
GRIDMET/DROUGHT PDSI, SRTM-derived elevation/slope/aspect, ESA WorldCover land cover (mapped to a fuel
proxy), and MODIS NDVI. NIFC incident records add ignition <em>cause</em> (lightning vs. human),
a signal most occurrence models leave out. Imagery for the detector is Sentinel-2 surface reflectance.
The positive (fire) rate is 1.24% of cell-weeks, so this is a rare-event problem.</p>
<p>Per-timestep pulls run concurrently, retry on failure and checkpoint to disk, which makes a
full-history pull feasible and resumable. When Earth Engine is unavailable, a synthetic generator
writes the same canonical tables, so the methodology still runs offline.</p>

<h2>3. Methods</h2>
<h3>3.1 Features</h3>
<p>Beyond instantaneous weather we engineer antecedent windows (rolling 2/4/8-step means of temperature,
VPD, ERC, wind, RH, PDSI, NDVI), antecedent-precipitation sums and a derived days-since-rain (dryness
memory), an NDVI anomaly relative to each cell's greenness baseline, fuel×dryness and wind×dryness
interactions, and per-cell historical ignition-cause densities. Raw longitude, latitude and year are
<em>excluded</em>, which keeps the trees from memorizing absolute position or time. That is a direct
guard against the leakage this paper is about.</p>
<h3>3.2 Models</h3>
<p><b>Risk surface.</b> A LightGBM classifier on the engineered features, trained on undersampled
negatives and <em>recalibrated</em> to the true base rate (King–Zeng prior correction) so probabilities
stay calibrated. <b>Counts.</b> A negative-binomial GLM per ecoregion-season with a log-exposure offset,
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
ROC-AUC and calibration in every case. The climatology baseline, which just encodes where fire has
recurred in the past, drops to no-skill under spatial-block CV (PR-AUC ≈ base rate, ROC-AUC 0.50),
while the environmental model keeps about 4× lift. That gap is the model transferring to regions it
was not trained on.</p>

<table>
<caption style="text-align:left;font-size:12px;color:#666;margin-bottom:6px">Table 1. Risk model vs. baselines (real data, leakage-aware CV). PR-AUC headline; higher better, Brier lower better.</caption>
<tr><th>Model</th><th>PR-AUC (time)</th><th>PR-AUC (spatial)</th><th>ROC-AUC</th><th>recall@20%</th><th>Brier</th></tr>
<tr><td>Climatology</td><td>0.039</td><td>0.012</td><td>0.50–0.63</td><td>0.14–0.39</td><td>0.012–0.014</td></tr>
<tr><td>Logistic (weather)</td><td>0.022</td><td>0.019</td><td>0.62</td><td>0.32–0.34</td><td>0.24–0.26</td></tr>
<tr><td><b>GBM (ours)</b></td><td><b>0.063</b></td><td><b>0.048</b></td><td><b>0.77</b></td><td><b>0.53–0.56</b></td><td><b>0.012–0.013</b></td></tr>
</table>

<figure>
  <img src="assets/shap_importance.png" alt="SHAP feature importance"/>
  <figcaption>Figure 1. Top risk-model drivers by mean |SHAP|: seasonality, NDVI / vegetation state,
  elevation, antecedent dryness, fuel×dryness and drought, all physically interpretable fire drivers.</figcaption>
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
<p>A one-week fire lag pushes PR-AUC to ≈0.65, but SHAP shows it takes over the model: with that
feature in, the model is really predicting fire <em>continuation</em> (fires burn for days), not
risk. We drop it and report the environmental skill on its own. The more interesting number is the
gap between forward-chaining and spatial-block CV, together with the climatology collapse. It
suggests that a lot of the apparent skill in the literature is spatial memorization, and that the
value of this model is that it carries to new ground.</p>
<p>The detector taught us a related lesson about matching imagery to the label. A first version
paired transient active-fire labels with a season <em>median</em> composite and scored near chance,
because the median averages the fire away. Reframing the task as burn-scar detection (persistent
scars, post-peak composite) recovered real skill. In each of these cases a higher number was on the
table and we took the lower, defensible one instead.</p>
<p>Set against recent Oregon and Pacific-Northwest work, the contribution here is methodological
rather than a new architecture: leakage-aware CV with an explicit climatology control, ignition-cause
features, rare-event metrics reported with calibration, and a multimodal pipeline that runs
end to end and reproduces.</p>

<h2>6. Limitations &amp; Future Work</h2>
<p>Absolute risk PR-AUC is modest. Weekly occurrence at 36&nbsp;km² is hard, and finer spatial and
temporal resolution would likely help. Per-timestep historical lightning is not available in Earth
Engine for 2001–2024, which limits the ignition-cause signal; the FPA-FOD archive and GOES-GLM
(2017 onward) are two ways to fill that in. The detector relies on coarse MODIS-derived labels and
classifies whole patches; pixel-level segmentation on time-matched imagery is the route to
literature-grade detection. The CNN's probabilities are also under-calibrated.</p>

<h2>7. Conclusion</h2>
<p>How much you can trust a wildfire model comes down to how it was evaluated. By fixing the
validation rules first and reporting only what got through them, we end up with an Oregon system that
has modest but real skill that generalizes, plus a written record of the larger numbers we chose not
to claim.</p>

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
