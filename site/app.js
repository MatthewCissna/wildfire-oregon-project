/* Oregon Wildfire ML — interactive atlas */
(function () {
  "use strict";
  const META = window.WF_META, CELLS = window.WF_CELLS.cells, YEARS = window.WF_CELLS.years;
  const SURF = META.surfaces;
  const $ = (s, r) => (r || document).querySelector(s);
  const fmt = (v, n = 2) => (v == null || isNaN(v)) ? "—" : (+v).toFixed(n);
  const BASE_NOLBL = "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png";
  const BASE_LBL = "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png";

  function rampFromStops(stops, t) {
    t = Math.max(0, Math.min(1, t));
    const s = t * (stops.length - 1), i = Math.floor(s), f = s - i;
    const A = hex2rgb(stops[i]), B = hex2rgb(stops[Math.min(i + 1, stops.length - 1)]);
    return `rgb(${Math.round(A[0] + (B[0] - A[0]) * f)},${Math.round(A[1] + (B[1] - A[1]) * f)},${Math.round(A[2] + (B[2] - A[2]) * f)})`;
  }
  const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const RISK_STOPS = (SURF.metrics.risk || {}).stops || ["#000004", "#781c6d", "#ed6925", "#fcffa4"];

  /* ---------- tabs ---------- */
  $("#tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) showTab(b.dataset.tab); });
  function showTab(name) {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".tab").forEach(s => s.classList.toggle("active", s.id === name));
    if (name === "map") { initMap(); if (map) setTimeout(() => map.invalidateSize(), 60); }
    if (name === "results") { initDistMap(); if (dmap) setTimeout(() => dmap.invalidateSize(), 60); }
  }

  /* ---------- overview ---------- */
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
    [0, .5, 1].forEach(g => { const y = pad.t + ih * (1 - g);
      html += `<line x1="${pad.l}" y1="${y}" x2="${w - pad.r}" y2="${y}" stroke="#2b3440"/>
               <text x="${pad.l - 5}" y="${y + 3}" fill="#6b7785" font-size="9" text-anchor="end">${Math.round(max * g)}</text>`; });
    values.forEach((v, i) => {
      const bh = (v / max) * ih, x = pad.l + i * bw, y = pad.t + ih - bh;
      html += `<rect x="${x + 1}" y="${y}" width="${bw - 2}" height="${bh}" fill="${rampFromStops(RISK_STOPS, v / max)}" rx="1"><title>${YEARS[i]}: ${v}</title></rect>`;
      if (i % 3 === 0) html += `<text x="${x + bw / 2}" y="${h - 6}" fill="#6b7785" font-size="9" text-anchor="middle">${String(YEARS[i]).slice(2)}</text>`;
    });
    svg.innerHTML = html; return svg;
  }
  function buildOverview() {
    $("#ov-rows").textContent = (META.manifest.n_panel_rows || 0).toLocaleString();
    const sb = (META.schemes.spatial_block || []).find(r => r.model === "risk_gbm") || {};
    const cards = [
      { big: (META.manifest.n_panel_rows / 1e6).toFixed(2) + "M", lbl: "Real cell-weeks", note: YEARS[0] + "–" + YEARS[YEARS.length - 1] + " · weekly" },
      { big: (META.manifest.n_cells || 0).toLocaleString(), lbl: "Oregon grid cells", note: "H3 res-6 · ~36 km² each" },
      { big: (sb.pr_lift ? sb.pr_lift + "×" : "—"), lbl: "Risk lift vs climatology", note: "leave-region-out CV" },
      { big: META.cnn ? fmt(META.cnn.roc_auc, 2) : "—", lbl: "Burn-scar detector ROC-AUC", note: "real Sentinel-2 · held-out blocks" },
    ];
    $("#ov-cards").innerHTML = cards.map(c => `<div class="card"><div class="big">${c.big}</div><div class="lbl">${c.lbl}</div><div class="note">${c.note}</div></div>`).join("");
    $("#ov-shap").innerHTML = barsHTML(META.shap.slice(0, 8).map(d => ({ name: d.feature, v: d.value })));
    $("#ov-yearchart").appendChild(yearChart(META.state_fires_by_year, 760, 150));
    $("#foot-meta").textContent = `source: ${META.manifest.source} · ${(META.manifest.positive_rate * 100).toFixed(2)}% positive`;
  }

  /* ---------- nearest cell ---------- */
  function nearest(lat, lng) {
    let best = null, bd = Infinity;
    for (let i = 0; i < CELLS.length; i++) { const c = CELLS[i], dx = c.lon - lng, dy = c.lat - lat, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = c; } }
    return best;
  }
  let last = 0;
  function throttle(fn, ms) { return (...a) => { const t = Date.now(); if (t - last > ms) { last = t; fn(...a); } }; }

  /* ---------- map ---------- */
  let map, overlay, marker, distOutline, ecoOutline, state = { metric: "risk" };
  function initMap() {
    if (map) return;
    if (typeof L === "undefined") { $("#leaflet").innerHTML = '<div style="padding:30px;color:#9aa7b4">Map library loading… needs internet for Leaflet &amp; basemap tiles. Other tabs work offline.</div>'; return; }
    map = L.map("leaflet", { preferCanvas: true, minZoom: 5, maxZoom: 11, zoomControl: true }).setView([44.0, -120.5], 6.4);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      { subdomains: "abcd", attribution: "© OpenStreetMap © CARTO" }).addTo(map);

    const sel = $("#metric");
    sel.innerHTML = Object.entries(SURF.metrics).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join("");
    setSurface("risk");
    sel.addEventListener("change", e => setSurface(e.target.value));
    $("#opacity").addEventListener("input", e => overlay && overlay.setOpacity(e.target.value / 100));
    $("#distToggle").addEventListener("change", e => toggleDistricts(e.target.checked));
    $("#ecoToggle").addEventListener("change", e => toggleEco(e.target.checked));

    map.on("mousemove", throttle(e => probe(e), 55));
    map.on("mouseout", () => $("#hovertip").hidden = true);
    map.on("click", e => { const c = nearest(e.latlng.lat, e.latlng.lng); selectCell(c); });
  }
  function setSurface(metric) {
    state.metric = metric; const m = SURF.metrics[metric];
    if (overlay) map.removeLayer(overlay);
    overlay = L.imageOverlay(m.png, SURF.bounds, { opacity: $("#opacity").value / 100, interactive: false }).addTo(map);
    overlay.bringToFront();
    if (distOutline) distOutline.bringToFront();
    drawLegend(m);
  }
  function drawLegend(m) {
    const grad = `linear-gradient(90deg,${m.stops.join(",")})`;
    $("#legend").innerHTML = `<div>${m.label}${m.unit ? " (" + m.unit + ")" : ""}</div>
      <div class="bar" style="background:${grad}"></div>
      <div class="ends"><span>${fmt(m.domain[0], 1)}</span><span>${fmt(m.domain[1], 1)}</span></div>`;
  }
  function probe(e) {
    const c = nearest(e.latlng.lat, e.latlng.lng); if (!c) return;
    const tip = $("#hovertip"); tip.hidden = false;
    tip.innerHTML = `<b>${c.eco}</b><div class="kv2"><span>Risk</span><span>${fmt(c.risk * 100, 2)}%</span>
      <span>Fire rate</span><span>${fmt(c.fires_rate, 2)}%</span><span>Elevation</span><span>${fmt(c.elev, 0)} m</span>
      <span>NDVI</span><span>${fmt(c.ndvi, 2)}</span></div><div class="tiphint">click for full detail</div>`;
    const wrap = $(".mapwrap").getBoundingClientRect();
    let x = e.originalEvent.clientX - wrap.left + 14, y = e.originalEvent.clientY - wrap.top + 14;
    if (x > wrap.width - 190) x -= 200; if (y > wrap.height - 150) y -= 150;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function selectCell(c) {
    if (!c) return;
    if (marker) map.removeLayer(marker);
    marker = L.circleMarker([c.lat, c.lon], { radius: 9, color: "#fff", weight: 2, fillColor: "#ff7a18", fillOpacity: .9, className: "pulse" }).addTo(map);
    showDetail(c);
  }
  function toggleDistricts(on) {
    if (on && META.districts_geo) {
      distOutline = L.geoJSON(META.districts_geo, { style: { color: "#ffe3b3", weight: 1.3, fill: false, opacity: .85 },
        onEachFeature: (f, l) => l.bindTooltip(f.properties.district, { sticky: true, className: "lbltip" }) }).addTo(map);
    } else if (distOutline) { map.removeLayer(distOutline); distOutline = null; }
  }
  function toggleEco(on) {
    if (on && META.ecoregions_geo) {
      ecoOutline = L.geoJSON(META.ecoregions_geo, { style: { color: "#7fd1ff", weight: 1, fill: false, opacity: .7, dashArray: "4 3" },
        onEachFeature: (f, l) => l.bindTooltip(f.properties.ecoregion, { sticky: true, className: "lbltip" }) }).addTo(map);
    } else if (ecoOutline) { map.removeLayer(ecoOutline); ecoOutline = null; }
  }

  /* ---------- detail panel ---------- */
  function showDetail(p) {
    const kv = rows => `<div class="kv">` + rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("") + `</div>`;
    const gauge = `<div class="riskgauge"><div class="track"></div><div class="marker"><i style="left:${(p.risk_pct * 100).toFixed(0)}%"></i></div></div>`;
    const climate = [["Max temp", fmt(p.tmax, 1) + " °C"], ["Min RH", fmt(p.rmin, 0) + " %"], ["VPD", fmt(p.vpd, 2) + " kPa"],
      ["Wind", fmt(p.wind, 1) + " m/s"], ["Precip", fmt(p.precip, 1) + " mm"], ["ERC", fmt(p.erc, 0)],
      ["Burning index", fmt(p.bi, 0)], ["PDSI (drought)", fmt(p.pdsi, 2)], ["NDVI", fmt(p.ndvi, 2)], ["Days since rain", fmt(p.days_since_rain, 0)]];
    const d = $("#detail");
    d.innerHTML = `<h3>Cell ${p.id.slice(0, 8)}…</h3><span class="eco-pill">${p.eco}</span>
      <h4>Modeled wildfire risk</h4>
      <div class="kv"><div class="k">Risk (calibrated)</div><div class="v">${fmt(p.risk * 100, 2)}%</div>
        <div class="k">Statewide percentile</div><div class="v">${(p.risk_pct * 100).toFixed(0)}ᵗʰ</div></div>${gauge}
      <h4>Location &amp; terrain</h4>${kv([["Latitude", fmt(p.lat, 3)], ["Longitude", fmt(p.lon, 3)], ["Elevation", fmt(p.elev, 0) + " m"], ["Slope", fmt(p.slope, 1) + "°"], ["Aspect", fmt(p.aspect, 0) + "°"]])}
      <h4>Land cover &amp; fuel</h4>${kv([["Land cover", p.landcover], ["Fuel load (0–1)", fmt(p.fuel, 2)]])}
      <h4>Climate normals (2001–2024 mean)</h4>${kv(climate)}
      <h4>Fire history</h4>${kv([["Burned cell-weeks", p.fires_total], ["Fire rate", fmt(p.fires_rate, 2) + " %"]])}<div id="cellyears"></div>`;
    $("#cellyears", d).appendChild(yearChart(p.fires_by_year, 340, 120));
  }

  /* ---------- results + district map ---------- */
  function schemeTable(rows) {
    const best = Math.max(...rows.map(r => r.pr_auc || 0));
    return `<table class="t"><tr><th>Model</th><th>PR-AUC</th><th>Lift</th><th>recall@20%</th><th>Brier</th><th>ROC-AUC</th></tr>` +
      rows.map(r => { const cls = (r.model === "risk_gbm" ? "ours " : "") + (r.pr_auc === best ? "best" : "");
        const name = r.model === "risk_gbm" ? "GBM (ours)" : r.model === "logistic_weather" ? "Logistic (weather)" : "Climatology";
        return `<tr class="${cls}"><td>${name}</td><td>${fmt(r.pr_auc, 3)}</td><td>${fmt(r.pr_lift, 1)}×</td><td>${fmt(r.recall20, 3)}</td><td>${fmt(r.brier, 4)}</td><td>${fmt(r.roc_auc, 3)}</td></tr>`; }).join("") + `</table>`;
  }
  function buildResults() {
    $("#tbl-fc").innerHTML = schemeTable(META.schemes.forward_chaining);
    $("#tbl-sb").innerHTML = schemeTable(META.schemes.spatial_block);
    $("#res-shap").innerHTML = barsHTML(META.shap.slice(0, 12).map(d => ({ name: d.feature, v: d.value })));
    if (META.cnn) { const c = META.cnn;
      $("#cnn-box").innerHTML = `<div class="cnnstats">
        <div class="s"><div class="n">${fmt(c.pr_auc, 2)}</div><div class="l">PR-AUC</div></div>
        <div class="s"><div class="n">${fmt(c.roc_auc, 2)}</div><div class="l">ROC-AUC</div></div>
        <div class="s"><div class="n">${fmt(c.recall_at_p20, 2)}</div><div class="l">recall@20%</div></div>
        <div class="s"><div class="n">${META.cnn_backbone || "CNN"}</div><div class="l">backbone</div></div></div>`; }
    const preds = (META.count && META.count.districts) || [];
    $("#count-box").innerHTML = `<table class="t click" id="dtable"><tr><th>ODF district</th><th>Pred. fires</th><th>95% interval</th></tr>` +
      preds.map(r => `<tr data-d="${r.district}"><td>${r.district}</td><td>${fmt(r.pred, 0)}</td><td>${fmt(r.lo, 0)} – ${fmt(r.hi, 0)}</td></tr>`).join("") + `</table>`;
  }
  let dmap, dLayer;
  function initDistMap() {
    if (dmap || typeof L === "undefined" || !META.districts_geo) {
      if (!META.districts_geo && !dmap) $("#distmap").innerHTML = '<div style="padding:20px;color:#9aa7b4">District layer unavailable (needs the ODF boundary fetch at build time).</div>';
      return;
    }
    dmap = L.map("distmap", { zoomControl: true, minZoom: 5, maxZoom: 9 }).setView([44.0, -120.5], 6);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { subdomains: "abcd" }).addTo(dmap);
    const vals = META.districts_geo.features.map(f => f.properties.pred_count || 0), max = Math.max(...vals) || 1;
    dLayer = L.geoJSON(META.districts_geo, {
      style: f => ({ fillColor: rampFromStops(RISK_STOPS, (f.properties.pred_count || 0) / max), color: "#0b0e12", weight: 1, fillOpacity: .8 }),
      onEachFeature: (f, l) => {
        const p = f.properties;
        l.bindTooltip(`<b>${p.district}</b><br>${Math.round(p.pred_count)} fires (95% ${Math.round(p.pred_lo)}–${Math.round(p.pred_hi)})`, { sticky: true, className: "lbltip" });
        l.on("mouseover", () => l.setStyle({ weight: 2.5, color: "#fff" }));
        l.on("mouseout", () => dLayer.resetStyle(l));
      }
    }).addTo(dmap);
    $("#dtable").addEventListener("mouseover", ev => { const tr = ev.target.closest("tr[data-d]"); if (!tr) return;
      dLayer.eachLayer(l => { if (l.feature.properties.district === tr.dataset.d) { l.setStyle({ weight: 3, color: "#fff" }); l.bringToFront(); } else dLayer.resetStyle(l); }); });
  }

  /* ---------- data explorer ---------- */
  const SOURCES = [["MODIS MCD64A1", "Burned area (label)", "500 m / monthly"], ["MODIS MOD14A1", "Active fire / thermal", "1 km / daily"],
    ["GRIDMET", "Weather + fire-danger (VPD, ERC, BI)", "~4 km / daily"], ["GRIDMET/DROUGHT", "PDSI drought", "~4 km"],
    ["SRTM", "Elevation → slope, aspect", "30 m"], ["ESA WorldCover", "Land cover → fuel proxy", "10 m"],
    ["MODIS MOD13A1", "NDVI vegetation", "500 m / 16-day"], ["Sentinel-2 SR", "Imagery (burn-scar CNN)", "10–20 m"],
    ["NIFC WFIGS", "Ignition cause (lightning/human)", "incident points"], ["ODF", "Fire-protection districts", "12 districts"]];
  function buildData() {
    $("#src-table").innerHTML = `<table class="t"><tr><th>Source</th><th>Role</th><th>Resolution</th></tr>` +
      SOURCES.map(s => `<tr><td>${s[0]}</td><td>${s[1]}</td><td><span class="tag">${s[2]}</span></td></tr>`).join("") + `</table>`;
    $("#eco-table").innerHTML = `<table class="t click"><tr><th>Ecoregion</th><th>Cells</th><th>Mean risk</th><th>Fire rate</th><th>Burned wks</th><th>Mean elev</th></tr>` +
      META.ecoregions.map(r => `<tr data-lat="${r.lat}" data-lon="${r.lon}"><td>${r.name}</td><td>${r.cells}</td><td>${fmt(r.mean_risk * 100, 2)}%</td><td>${fmt(r.fire_rate, 2)}%</td><td>${r.fires_total.toLocaleString()}</td><td>${fmt(r.mean_elev, 0)} m</td></tr>`).join("") + `</table>`;
    $("#eco-table").addEventListener("click", ev => { const tr = ev.target.closest("tr[data-lat]"); if (!tr) return;
      showTab("map"); initMap(); if (map) { map.setView([+tr.dataset.lat, +tr.dataset.lon], 8); setTimeout(() => map.invalidateSize(), 60); } });
  }

  buildOverview(); buildResults(); buildData();
})();
