// Academic paper rendered in the "Paper" tab.
document.getElementById("paper-body").innerHTML = `
<h1>Leakage-aware spatial generalization for wildfire risk:<br/>a machine-learning system for Oregon, 2001–2024</h1>
<p class="authors">Oregon Wildfire ML Project</p>
<p class="affil">Reproducible pipeline. Data: Google Earth Engine, NIFC, MODIS, Sentinel-2, GRIDMET, ESA WorldCover, SRTM.</p>

<div class="abstract">
<strong>Abstract.</strong> We describe a reproducible machine-learning system for wildfire analysis in
Oregon and report what it can and cannot do once the evaluation is held to a strict standard. The
system has three parts: a per-cell fire-<em>risk</em> surface, per-region seasonal fire-<em>count</em>
forecasts with calibrated prediction intervals, and a satellite burn-scar <em>detector</em>. Training
uses 4.56 million cell-week records on an H3 hexagon grid spanning 2001 to 2024, with MODIS
burned-area labels, GRIDMET weather and fire-danger indices, SRTM terrain, ESA WorldCover land cover,
MODIS NDVI, and NIFC ignition cause. We validate with spatial-block, leave-one-region-out, and
forward-chaining cross-validation, and we score with metrics suited to a rare event: precision-recall
AUC, Brier score, and recall at operating thresholds. The central result is that weekly fire occurrence
at a 36&nbsp;km² resolution is genuinely hard, and absolute PR-AUC stays low for every method we tried.
A gradient-boosted model still generalizes: under leave-one-region-out cross-validation it beats a
climatology baseline by about four times (ROC-AUC 0.77, recall near 0.55 at the 20% flag rate), while
the climatology baseline falls to no skill. We also document two results that looked strong and were set
aside: a fire-persistence shortcut that inflates PR-AUC to 0.65, and a synthetic detector that was too
easy to separate. The practical point is narrow. In this setting, the evaluation design decides which
numbers you can trust.
</div>

<h2>1. Introduction</h2>
<p>Data-driven wildfire prediction has grown quickly, and so has the gap between reported and
real-world performance. A common cause is leakage: random train/test splits applied to data that is
correlated in space and time, or features that sit too close to the target (Kapoor and Narayanan,
2023). A model can post strong numbers in a notebook and then fail on a new region or a future season.
For spatial and temporal data the standard correction is blocked cross-validation, because random
splits let neighboring observations leak between the training and test sets (Roberts et al., 2017).</p>
<p>Oregon is a useful test case. Moisture drops sharply from the wet Coast Range to the arid Northern
Basin and Range, across eight ecoregions, so a model has to handle very different fire regimes inside
one state. The 2001–2024 record includes severe seasons such as 2024 and quiet ones such as 2019, a
spread consistent with the documented effect of warming on fuel aridity and burned area in the western
United States (Abatzoglou and Williams, 2016). A credible model has to capture both where fire is
possible and the conditions that set off a given week. We build a complete Oregon system, fix the
validation rules before training, and report only what survives them.</p>

<h2>2. Data</h2>
<p>Every layer is clipped to Oregon and summarized on an H3 resolution-6 hexagon grid (Uber
Technologies), about 7,224 cells of roughly 36&nbsp;km² each, at weekly steps. Fire labels come from
MODIS Collection 6 burned area (MCD64A1; Giglio et al., 2018) and MODIS thermal anomalies (MOD14A1).
Predictors are GRIDMET weather and fire-danger indices, including vapor pressure deficit, the Energy
Release Component, the Burning Index, wind, relative humidity, and precipitation (Abatzoglou, 2013),
together with GRIDMET/DROUGHT PDSI, SRTM-derived elevation, slope and aspect, ESA WorldCover land cover
mapped to a fuel proxy, and MODIS NDVI. NIFC incident records add ignition cause, separating lightning
from human starts, a signal most occurrence models leave out. Imagery for the detector is Sentinel-2
surface reflectance. Fire occurs in 1.24% of cell-weeks, so this is a rare-event problem.</p>
<p>The ingest is built to be resumable. Per-timestep pulls run concurrently, retry on failure, and
checkpoint to disk, which makes a full-history pull feasible to restart after an interruption. When
Earth Engine is unavailable, a synthetic generator writes the same tables so the pipeline still runs
offline.</p>

<h2>3. Methods</h2>
<h3>3.1 Features</h3>
<p>Beyond instantaneous weather, we add antecedent windows (rolling 2-, 4-, and 8-step means of
temperature, VPD, ERC, wind, relative humidity, PDSI, and NDVI), antecedent-precipitation sums, and a
derived days-since-rain term that carries dryness memory. We compute an NDVI anomaly against each
cell's own greenness baseline, fuel×dryness and wind×dryness interactions, and per-cell historical
ignition-cause densities. We exclude raw longitude, latitude, and year, which keeps the trees from
memorizing absolute position or time. That exclusion is a deliberate guard against the leakage this
paper is about.</p>
<h3>3.2 Models</h3>
<p>The risk surface is a LightGBM classifier (Ke et al., 2017) trained on undersampled negatives and
recalibrated to the true base rate with the King–Zeng prior correction (King and Zeng, 2001), so the
output probabilities stay calibrated. Counts are modeled per ecoregion-season with a negative-binomial
GLM and a log-exposure offset, which yields rate estimates and prediction intervals that widen under
overdispersion. Detection uses EfficientNet-B0 (Tan and Le, 2019) with its input convolution adapted to
a ten-channel stack of six Sentinel-2 bands and four spectral indices, trained with spatial-block
splits and augmentation. Hyper-parameters are tuned with Optuna against the spatial-block objective
rather than a random split (Akiba et al., 2019).</p>
<h3>3.3 Evaluation</h3>
<p>We use three leakage-aware schemes: forward-chaining in time (train on years ≤ Y, test on Y+1),
spatial-block GroupKFold on H3 parent cells, and leave-one-region-out across ecoregions (Roberts et
al., 2017). PR-AUC (average precision) is the headline, with Brier score for calibration and recall at
a fixed flag rate for operational use. A climatology baseline (per-cell historical frequency) and a
weather-only logistic regression run on identical splits. We read feature attributions with SHAP
(Lundberg and Lee, 2017) to check that the model uses fire-relevant signals rather than artifacts.</p>

<h2>4. Results</h2>
<p>Table 1 reports the risk model against baselines under both cross-validation schemes. The model
leads on PR-AUC, ROC-AUC, and calibration in every case. The climatology baseline, which encodes where
fire has recurred, drops to no skill under spatial-block cross-validation (PR-AUC near the base rate,
ROC-AUC 0.50), while the environmental model keeps about a fourfold lift. That gap is the model
transferring to regions it was not trained on.</p>

<table>
<caption style="text-align:left;font-size:12px;color:#666;margin-bottom:6px">Table 1. Risk model vs. baselines (real data, leakage-aware CV). PR-AUC headline; higher better, Brier lower better.</caption>
<tr><th>Model</th><th>PR-AUC (time)</th><th>PR-AUC (spatial)</th><th>ROC-AUC</th><th>recall@20%</th><th>Brier</th></tr>
<tr><td>Climatology</td><td>0.039</td><td>0.012</td><td>0.50–0.63</td><td>0.14–0.39</td><td>0.012–0.014</td></tr>
<tr><td>Logistic (weather)</td><td>0.022</td><td>0.019</td><td>0.62</td><td>0.32–0.34</td><td>0.24–0.26</td></tr>
<tr><td><b>GBM (ours)</b></td><td><b>0.063</b></td><td><b>0.048</b></td><td><b>0.77</b></td><td><b>0.53–0.56</b></td><td><b>0.012–0.013</b></td></tr>
</table>

<figure>
  <img src="assets/shap_importance.png" alt="SHAP feature importance"/>
  <figcaption>Figure 1. Top risk-model drivers by mean |SHAP|: seasonality, NDVI and vegetation state,
  elevation, antecedent dryness, fuel×dryness, and drought. Each is a physically interpretable fire
  driver (Lundberg and Lee, 2017).</figcaption>
</figure>

<p>The count model predicts per-ecoregion seasonal fire counts with well-calibrated 95% prediction
intervals (empirical coverage about 0.95 to 0.98), widening the intervals where the data is
overdispersed. On real Sentinel-2 imagery the burn-scar detector reaches ROC-AUC about 0.73 and PR-AUC
about 0.81 on held-out spatial blocks (Section 5).</p>

<p>Two operational layers sit on top of these models. A weekly fire-danger check places each ODF
district's predicted risk and its ERC reading on the same five-step caution scale (none to extreme),
where the ERC class follows the climatological-percentile method of the National Fire-Danger Rating
System (Bradshaw et al., 1984); across district-weeks the model class and the ERC class agree exactly
28% of the time and within one step 75% of the time, which is reasonable agreement given that one
rating is risk-based and the other is index-based. A live watch pulls recent FIRMS thermal detections
and runs the burn-scar detector on fresh Sentinel-2 imagery to flag whether a hot cell actually looks
burned.</p>

<figure>
  <img src="assets/risk_map.png" alt="Oregon risk surface"/>
  <figcaption>Figure 2. Modeled per-cell wildfire-risk surface over Oregon.</figcaption>
</figure>

<h2>5. Discussion</h2>
<p>A one-week fire lag pushes PR-AUC to about 0.65, but SHAP shows that this single feature takes over:
with it in the model, the prediction is really fire continuation, since fires burn for several days,
rather than next-period risk. We drop it and report the environmental skill on its own. The more
informative comparison is the gap between forward-chaining and spatial-block cross-validation, together
with the climatology collapse. It suggests that a large part of the apparent skill in some published
work is spatial memorization, and that the value of this model is that it carries to new ground
(Kapoor and Narayanan, 2023; Roberts et al., 2017).</p>
<p>The detector taught a related lesson about matching imagery to the label. A first version paired
transient active-fire labels with a season-median composite and scored near chance, because the median
averages the fire away. Reframing the task as burn-scar detection, with burned-area labels and a
post-peak composite, recovered real skill. In each case a higher number was available and we reported
the lower, defensible one instead.</p>
<p>Set against recent Oregon and Pacific-Northwest work, and against explainability-focused wildfire
studies that compare models on single regions (Sengupta and Woodford, 2025), the contribution here is
methodological rather than a new architecture: leakage-aware cross-validation with an explicit
climatology control, ignition-cause features, rare-event metrics reported with calibration, and a
multimodal pipeline that runs end to end and reproduces.</p>

<h2>6. Limitations and future work</h2>
<p>Absolute risk PR-AUC is modest. Weekly occurrence at 36&nbsp;km² is hard, and finer spatial and
temporal resolution would likely help. Per-timestep historical lightning is not available in Earth
Engine for 2001–2024, which limits the ignition-cause signal; the FPA-FOD archive and GOES-GLM (2017
onward) are two ways to fill that gap. The detector relies on coarse MODIS-derived labels and
classifies whole patches, so pixel-level segmentation on time-matched imagery is the route to
literature-grade detection. The detector's probabilities are also under-calibrated.</p>

<h2>7. Conclusion</h2>
<p>How much you can trust a wildfire model comes down to how it was evaluated. By fixing the validation
rules first and reporting only what survives them, we end up with an Oregon system that has modest but
real skill that generalizes, along with a written record of the larger numbers we chose not to claim.</p>

<h2>References</h2>
<ol class="refs">
<li>Abatzoglou, J.T. (2013). Development of gridded surface meteorological data for ecological applications and modelling. <i>International Journal of Climatology</i>, 33(1), 121–131. <a href="https://doi.org/10.1002/joc.3413" target="_blank" rel="noopener">doi:10.1002/joc.3413</a></li>
<li>Abatzoglou, J.T. &amp; Williams, A.P. (2016). Impact of anthropogenic climate change on wildfire across western US forests. <i>PNAS</i>, 113(42), 11770–11775. <a href="https://doi.org/10.1073/pnas.1607171113" target="_blank" rel="noopener">doi:10.1073/pnas.1607171113</a></li>
<li>Akiba, T., Sano, S., Yanase, T., Ohta, T. &amp; Koyama, M. (2019). Optuna: a next-generation hyperparameter optimization framework. <i>Proc. 25th ACM SIGKDD</i>, 2623–2631. <a href="https://doi.org/10.1145/3292500.3330701" target="_blank" rel="noopener">doi:10.1145/3292500.3330701</a></li>
<li>Bradshaw, L.S., Deeming, J.E., Burgan, R.E. &amp; Cohen, J.D. (1984). The 1978 National Fire-Danger Rating System: technical documentation. <i>USDA Forest Service General Technical Report INT-169</i>, Intermountain Forest and Range Experiment Station, Ogden, UT. <a href="https://research.fs.usda.gov/treesearch/29615" target="_blank" rel="noopener">research.fs.usda.gov/treesearch/29615</a></li>
<li>Giglio, L., Boschetti, L., Roy, D.P., Humber, M.L. &amp; Justice, C.O. (2018). The Collection 6 MODIS burned area mapping algorithm and product. <i>Remote Sensing of Environment</i>, 217, 72–85. <a href="https://doi.org/10.1016/j.rse.2018.08.005" target="_blank" rel="noopener">doi:10.1016/j.rse.2018.08.005</a></li>
<li>Kapoor, S. &amp; Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. <i>Patterns</i>, 4(9), 100804. <a href="https://doi.org/10.1016/j.patter.2023.100804" target="_blank" rel="noopener">doi:10.1016/j.patter.2023.100804</a></li>
<li>Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. &amp; Liu, T.-Y. (2017). LightGBM: a highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing Systems</i>, 30, 3146–3154. <a href="https://doi.org/10.5555/3294996.3295074" target="_blank" rel="noopener">doi:10.5555/3294996.3295074</a></li>
<li>King, G. &amp; Zeng, L. (2001). Logistic regression in rare events data. <i>Political Analysis</i>, 9(2), 137–163. <a href="https://doi.org/10.1093/oxfordjournals.pan.a004868" target="_blank" rel="noopener">doi:10.1093/oxfordjournals.pan.a004868</a></li>
<li>Lundberg, S.M. &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>Advances in Neural Information Processing Systems</i>, 30, 4768–4777. <a href="https://doi.org/10.5555/3295222.3295230" target="_blank" rel="noopener">doi:10.5555/3295222.3295230</a></li>
<li>Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J.J., Schröder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F. &amp; Dormann, C.F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. <i>Ecography</i>, 40(8), 913–929. <a href="https://doi.org/10.1111/ecog.02881" target="_blank" rel="noopener">doi:10.1111/ecog.02881</a></li>
<li>Sengupta, A. &amp; Woodford, B.J. (2025). Recent advances in explainable machine learning models for wildfire prediction. <i>Applied Computing and Geosciences</i>, 27, 100266. <a href="https://doi.org/10.1016/j.acags.2025.100266" target="_blank" rel="noopener">doi:10.1016/j.acags.2025.100266</a></li>
<li>Tan, M. &amp; Le, Q.V. (2019). EfficientNet: rethinking model scaling for convolutional neural networks. <i>Proc. 36th International Conference on Machine Learning (ICML)</i>, 6105–6114. <a href="https://arxiv.org/abs/1905.11946" target="_blank" rel="noopener">arXiv:1905.11946</a></li>
<li>Uber Technologies. H3: a hexagonal hierarchical geospatial indexing system. <a href="https://h3geo.org" target="_blank" rel="noopener">h3geo.org</a></li>
</ol>
`;
