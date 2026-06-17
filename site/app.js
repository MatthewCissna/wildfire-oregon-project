/* Oregon Wildfire ML — interactive atlas */
(function () {
  "use strict";
  const META = window.WF_META, CELLS = window.WF_CELLS;
  const YEARS = CELLS.years;
  const $ = (s, r) => (r || document).querySelector(s);
  const el = (h) => { const d = document.createElement("div"); d.innerHTML = h; return d.firstElementChild; };
  const fmt = (v, n = 2) => (v == null || isNaN(v)) ? "—" : (+v).toFixed(n);

  /* ---------- color ramp (blue → yellow → orange → red) ---------- */
  const STOPS = [[44, 123, 182], [255, 255, 191], [253, 174, 97], [215, 25, 28]];
  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    const s = t * (STOPS.length - 1), i = Math.floor(s), f = s - i;
    const a = STOPS[i], b = STOPS[Math.min(i + 1, STOPS.length - 1)];
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
  }
  function domainFor(metric) {
    if (metric === "risk_pct") return [0, 1];
    const v = CELLS.features.map(f => f.properties[metric]).filter(x => x != null && isFinite(x)).sort((a, b) => a - b);
    const q = p => v[Math.floor(p * (v.length - 1))];
    return [q(0.02), q(0.98)];
  }

  /* ---------- tabs ---------- */
  $("#tabs").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return; showTab(b.dataset.tab);
  });
  function showTab(name) {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".tab").forEach(s => s.classList.toggle("active", s.id === name));
    if (name === "map" && !map) initMap();
    if (name === "map" && map) setTimeout(() => map.invalidateSize(), 60);
  }

  /* ---------- overview ---------- */
  function buildOverview() {
    $("#ov-rows").textContent = (META.manifest.n_panel_rows || 0).toLocaleString();
    const sb = (META.schemes.spatial_block || []).find(r => r.model === "risk_gbm") || {};
    const cards = [
      { big: (META.manifest.n_panel_rows / 1e6).toFixed(2) + "M", lbl: "Real cell-weeks", note: YEARS[0] + "–" + YEARS[YEARS.length - 1] + " · weekly" },
      { big: (META.manifest.n_cells || 0).toLocaleString(), lbl: "Oregon grid cells", note: "H3 res-6 · ~36 km² each" },
      { big: (sb.pr_lift ? sb.pr_lift + "×" : "—"), lbl: "Risk lift vs climatology", note: "leave-region-out CV" },
      { big: META.cnn ? fmt(META.cnn.roc_auc, 2) : "—", lbl: "Burn-scar detector ROC-AUC", note: "real Sentinel-2 · held-out blocks" },
    ];
    $("#ov-cards").innerHTML = cards.map(c =>
      `<div class="card"><div class="big">${c.big}</div><div class="lbl">${c.lbl}</div><div class="note">${c.note}</div></div>`).join("");
    $("#ov-shap").innerHTML = barsHTML(META.shap.slice(0, 8).map(d => ({ name: d.feature, v: d.value })));
    $("#ov-yearchart").appendChild(yearChart(META.state_fires_by_year, 760, 150));
    $("#foot-meta").textContent = `source: ${META.manifest.source} · ${(META.manifest.positive_rate * 100).toFixed(2)}% positive`;
  }

  function barsHTML(items) {
    const max = Math.max(...items.map(i => i.v)) || 1;
    return `<div class="bars">` + items.map(i =>
      `<div class="row"><div class="name" title="${i.name}">${i.name}</div>
       <div class="track"><div class="fill" style="width:${(i.v / max * 100).toFixed(1)}%"></div></div>
       <div class="val">${i.v.toFixed(3)}</div></div>`).join("") + `</div>`;
  }

  function yearChart(values, w, h) {
    const pad = { l: 30, r: 8, t: 8, b: 20 }, iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const max = Math.max(...values, 1), bw = iw / values.length;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("width", "100%");
    let html = "";
    [0, 0.5, 1].forEach(g => { const y = pad.t + ih * (1 - g);
      html += `<line x1="${pad.l}" y1="${y}" x2="${w - pad.r}" y2="${y}" stroke="#2b3440"/>
               <text x="${pad.l - 5}" y="${y + 3}" fill="#6b7785" font-size="9" text-anchor="end">${Math.round(max * g)}</text>`; });
    values.forEach((v, i) => {
      const bh = (v / max) * ih, x = pad.l + i * bw, y = pad.t + ih - bh;
      const t = v / max;
      html += `<rect x="${x + 1}" y="${y}" width="${bw - 2}" height="${bh}" fill="${ramp(t)}" rx="1"><title>${YEARS[i]}: ${v}</title></rect>`;
      if (i % 3 === 0) html += `<text x="${x + bw / 2}" y="${h - 6}" fill="#6b7785" font-size="9" text-anchor="middle">${String(YEARS[i]).slice(2)}</text>`;
    });
    svg.innerHTML = html; return svg;
  }

  /* ---------- map ---------- */
  let map, layer, selected, state = { metric: "risk_pct", eco: "" };
  function initMap() {
    if (typeof L === "undefined") {  // Leaflet (CDN) not loaded yet / offline
      $("#leaflet").innerHTML = '<div style="padding:30px;color:#9aa7b4">Map library is loading… ' +
        'if this persists you are offline — the interactive map needs internet for Leaflet &amp; basemap tiles. ' +
        'All other tabs work offline.</div>';
      return;
    }
    map = L.map("leaflet", { preferCanvas: true, zoomControl: true }).setView([44.0, -120.6], 6);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: "© OpenStreetMap © CARTO", subdomains: "abcd", maxZoom: 12, opacity: .8
    }).addTo(map);
    drawLayer();
    const ecos = [...new Set(CELLS.features.map(f => f.properties.eco))].sort();
    $("#ecofilter").innerHTML = `<option value="">All</option>` + ecos.map(e => `<option>${e}</option>`).join("");
    $("#metric").addEventListener("change", e => { state.metric = e.target.value; drawLayer(); });
    $("#ecofilter").addEventListener("change", e => { state.eco = e.target.value; drawLayer(); });
  }
  function styleFn(dom) {
    return f => {
      const p = f.properties;
      if (state.eco && p.eco !== state.eco) return { stroke: false, fillOpacity: 0 };
      let v = p[state.metric], t = v == null ? 0 : (v - dom[0]) / (dom[1] - dom[0]);
      return { color: "#0b0e12", weight: .2, fillColor: ramp(t), fillOpacity: .8 };
    };
  }
  function drawLayer() {
    const dom = domainFor(state.metric);
    if (layer) map.removeLayer(layer);
    layer = L.geoJSON(CELLS, {
      style: styleFn(dom),
      onEachFeature: (feat, lyr) => {
        lyr.on("click", () => { selectCell(lyr, feat.properties); });
        lyr.on("mouseover", () => lyr.setStyle({ weight: 1.2, color: "#fff" }));
        lyr.on("mouseout", () => { if (lyr !== selected) layer.resetStyle(lyr); });
      }
    }).addTo(map);
    window._wfLayer = layer;  // exposed for debugging / power users
    drawLegend(dom);
  }
  function selectCell(lyr, p) {
    if (selected) layer.resetStyle(selected);
    selected = lyr; lyr.setStyle({ weight: 2, color: "#ffffff" }); lyr.bringToFront();
    showDetail(p);
  }
  function drawLegend(dom) {
    const labels = { risk_pct: "Modeled risk (percentile)", fires_rate: "Fire rate (% of weeks)", fuel: "Fuel load", elev: "Elevation (m)", vpd: "Mean VPD (kPa)", ndvi: "Mean NDVI" };
    $("#legend").innerHTML = `<div>${labels[state.metric] || state.metric}</div>
      <div class="bar" style="background:linear-gradient(90deg,rgb(44,123,182),rgb(255,255,191),rgb(253,174,97),rgb(215,25,28))"></div>
      <div class="ends"><span>${state.metric === "risk_pct" ? "low" : fmt(dom[0], 1)}</span><span>${state.metric === "risk_pct" ? "high" : fmt(dom[1], 1)}</span></div>`;
  }

  /* ---------- detail panel ---------- */
  function showDetail(p) {
    const kv = (rows) => `<div class="kv">` + rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("") + `</div>`;
    const riskMark = `<div class="riskgauge"><div class="track"></div><div class="marker"><i style="left:${(p.risk_pct * 100).toFixed(0)}%"></i></div></div>`;
    const climate = [
      ["Max temp", fmt(p.tmax, 1) + " °C"], ["Min RH", fmt(p.rmin, 0) + " %"],
      ["VPD", fmt(p.vpd, 2) + " kPa"], ["Wind", fmt(p.wind, 1) + " m/s"],
      ["Precip", fmt(p.precip, 1) + " mm"], ["ERC", fmt(p.erc, 0)],
      ["Burning index", fmt(p.bi, 0)], ["PDSI (drought)", fmt(p.pdsi, 2)],
      ["NDVI", fmt(p.ndvi, 2)], ["Days since rain", fmt(p.days_since_rain, 0)],
    ];
    const d = $("#detail");
    d.innerHTML =
      `<h3>Cell ${p.id.slice(0, 8)}…</h3>
       <span class="eco-pill">${p.eco}</span>
       <h4>Modeled wildfire risk</h4>
       <div class="kv"><div class="k">Risk (calibrated)</div><div class="v">${fmt(p.risk * 100, 2)}%</div>
         <div class="k">Statewide percentile</div><div class="v">${(p.risk_pct * 100).toFixed(0)}ᵗʰ</div></div>
       ${riskMark}
       <h4>Location &amp; terrain</h4>
       ${kv([["Latitude", fmt(p.lat, 3)], ["Longitude", fmt(p.lon, 3)], ["Elevation", fmt(p.elev, 0) + " m"], ["Slope", fmt(p.slope, 1) + "°"], ["Aspect", fmt(p.aspect, 0) + "°"]])}
       <h4>Land cover &amp; fuel</h4>
       ${kv([["Land cover", p.landcover], ["Fuel load (0–1)", fmt(p.fuel, 2)]])}
       <h4>Climate normals (2001–2024 mean)</h4>
       ${kv(climate)}
       <h4>Fire history</h4>
       ${kv([["Burned cell-weeks", p.fires_total], ["Fire rate", fmt(p.fires_rate, 2) + " %"]])}
       <div id="cellyears"></div>`;
    $("#cellyears", d).appendChild(yearChart(p.fires_by_year, 340, 120));
  }

  /* ---------- results ---------- */
  function schemeTable(rows) {
    const best = Math.max(...rows.map(r => r.pr_auc || 0));
    const head = `<table class="t"><tr><th>Model</th><th>PR-AUC</th><th>Lift</th><th>recall@20%</th><th>Brier</th><th>ROC-AUC</th></tr>`;
    return head + rows.map(r => {
      const cls = (r.model === "risk_gbm" ? "ours " : "") + (r.pr_auc === best ? "best" : "");
      const name = r.model === "risk_gbm" ? "GBM (ours)" : r.model === "logistic_weather" ? "Logistic (weather)" : "Climatology";
      return `<tr class="${cls}"><td>${name}</td><td>${fmt(r.pr_auc, 3)}</td><td>${fmt(r.pr_lift, 1)}×</td><td>${fmt(r.recall20, 3)}</td><td>${fmt(r.brier, 4)}</td><td>${fmt(r.roc_auc, 3)}</td></tr>`;
    }).join("") + `</table>`;
  }
  function buildResults() {
    $("#tbl-fc").innerHTML = schemeTable(META.schemes.forward_chaining);
    $("#tbl-sb").innerHTML = schemeTable(META.schemes.spatial_block);
    $("#res-shap").innerHTML = barsHTML(META.shap.slice(0, 12).map(d => ({ name: d.feature, v: d.value })));
    if (META.cnn) {
      const c = META.cnn;
      $("#cnn-box").innerHTML = `<div class="cnnstats">
        <div class="s"><div class="n">${fmt(c.pr_auc, 2)}</div><div class="l">PR-AUC</div></div>
        <div class="s"><div class="n">${fmt(c.roc_auc, 2)}</div><div class="l">ROC-AUC</div></div>
        <div class="s"><div class="n">${fmt(c.recall_at_p20, 2)}</div><div class="l">recall@20%</div></div>
        <div class="s"><div class="n">${META.cnn_backbone || "CNN"}</div><div class="l">backbone</div></div></div>`;
    }
    const preds = (META.count && META.count.predictions) || [];
    $("#count-box").innerHTML = `<table class="t"><tr><th>Ecoregion</th><th>Predicted fires</th><th>95% interval</th></tr>` +
      preds.map(r => `<tr><td>${r.region}</td><td>${fmt(r.pred, 1)}</td><td>${fmt(r.lo, 0)} – ${fmt(r.hi, 0)}</td></tr>`).join("") + `</table>`;
  }

  /* ---------- data explorer ---------- */
  const SOURCES = [
    ["MODIS MCD64A1", "Burned area (label)", "500 m / monthly"],
    ["MODIS MOD14A1", "Active fire / thermal", "1 km / daily"],
    ["GRIDMET", "Weather + fire-danger (VPD, ERC, BI)", "~4 km / daily"],
    ["GRIDMET/DROUGHT", "PDSI drought", "~4 km"],
    ["SRTM", "Elevation → slope, aspect", "30 m"],
    ["ESA WorldCover", "Land cover → fuel proxy", "10 m"],
    ["MODIS MOD13A1", "NDVI vegetation", "500 m / 16-day"],
    ["Sentinel-2 SR", "Imagery (burn-scar CNN)", "10–20 m"],
    ["NIFC WFIGS", "Ignition cause (lightning/human)", "incident points"],
  ];
  function buildData() {
    $("#src-table").innerHTML = `<table class="t"><tr><th>Source</th><th>Role</th><th>Resolution</th></tr>` +
      SOURCES.map(s => `<tr><td>${s[0]}</td><td>${s[1]}</td><td><span class="tag">${s[2]}</span></td></tr>`).join("") + `</table>`;
    const e = META.ecoregions;
    $("#eco-table").innerHTML = `<table class="t click"><tr><th>Ecoregion</th><th>Cells</th><th>Mean risk</th><th>Fire rate</th><th>Burned wks</th><th>Mean elev</th></tr>` +
      e.map(r => `<tr data-eco="${r.name}"><td>${r.name}</td><td>${r.cells}</td><td>${fmt(r.mean_risk * 100, 2)}%</td><td>${fmt(r.fire_rate, 2)}%</td><td>${r.fires_total.toLocaleString()}</td><td>${fmt(r.mean_elev, 0)} m</td></tr>`).join("") + `</table>`;
    $("#eco-table").addEventListener("click", ev => {
      const tr = ev.target.closest("tr[data-eco]"); if (!tr) return;
      showTab("map"); if (!map) initMap();
      state.eco = tr.dataset.eco; $("#ecofilter").value = state.eco; drawLayer();
    });
  }

  /* ---------- go ---------- */
  buildOverview(); buildResults(); buildData();
})();
