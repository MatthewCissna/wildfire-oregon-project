/* Oregon Wildfire ML — interactive atlas */
(function () {
  "use strict";
  const META = window.WF_META, CELLS = window.WF_CELLS.cells, YEARS = window.WF_CELLS.years;
  const SURF = META.surfaces;
  const ID2I = {}; CELLS.forEach((c, i) => { ID2I[c.id] = i; });  // cell id -> hex index
  const $ = (s, r) => (r || document).querySelector(s);
  const fmt = (v, n = 2) => (v == null || isNaN(v)) ? "—" : (+v).toFixed(n);
  const DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

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
    if (name === "livewatch") { buildLiveWatch(); if (lwMap) setTimeout(() => lwMap.invalidateSize(), 60); }
    if (name === "forecast") { initForecast(); if (fcMap) setTimeout(() => fcMap.invalidateSize(), 60); }
    if (name === "danger") { buildDanger(); }
    if (name === "tracker") { buildTracker(); if (trMap) setTimeout(() => trMap.invalidateSize(), 60); }
    if (name === "results") { initDistMap(); if (dmap) setTimeout(() => dmap.invalidateSize(), 60); }
  }

  /* ---------- nearest cell (only on click — no mousemove) ---------- */
  function nearest(lat, lng) {
    let best = null, bd = Infinity;
    for (let i = 0; i < CELLS.length; i++) { const c = CELLS[i], dx = c.lon - lng, dy = c.lat - lat, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = c; } }
    return best;
  }

  /* ---------- bars + small year chart ---------- */
  function barsHTML(items) {
    const max = Math.max(...items.map(i => i.v)) || 1;
    return `<div class="bars">` + items.map(i =>
      `<div class="row"><div class="name" title="${i.name}">${i.name}</div>
       <div class="track"><div class="fill" style="width:${(i.v / max * 100).toFixed(1)}%"></div></div>
       <div class="val">${i.v.toFixed(3)}</div></div>`).join("") + `</div>`;
  }
  function yearChart(values, w, h, labels) {
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
      const lbl = labels ? labels[i] : YEARS[i];
      html += `<rect x="${x + 1}" y="${y}" width="${bw - 2}" height="${bh}" fill="${rampFromStops(RISK_STOPS, v / max)}" rx="1"><title>${lbl}: ${v}</title></rect>`;
      const step = labels ? 4 : 3;
      if (i % step === 0) html += `<text x="${x + bw / 2}" y="${h - 6}" fill="#6b7785" font-size="9" text-anchor="middle">${String(lbl).slice(labels ? 5 : 2)}</text>`;
    });
    svg.innerHTML = html; return svg;
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
    $("#ov-cards").innerHTML = cards.map(c => `<div class="card"><div class="big">${c.big}</div><div class="lbl">${c.lbl}</div><div class="note">${c.note}</div></div>`).join("");
    $("#ov-shap").innerHTML = barsHTML(META.shap.slice(0, 8).map(d => ({ name: d.feature, v: d.value })));
    $("#ov-yearchart").appendChild(yearChart(META.state_fires_by_year, 760, 150));
    $("#foot-meta").textContent = `source: ${META.manifest.source} · ${(META.manifest.positive_rate * 100).toFixed(2)}% positive`;
  }

  /* ---------- risk map (true H3 hexagons, click-only) ---------- */
  const HEX_BORDER = "rgba(6,9,13,.45)";
  let map, hexLayer, hexPolys = [], cityLayer, distOutline, marker, state = { metric: "risk" };
  // Normalize a cell's raw metric value to 0–1 against the same 2nd–98th pct range
  // the legend uses (lo/hi shipped once per metric in META.hexes.lohi).
  function hexNorm(cell, metric) {
    const lohi = META.hexes && META.hexes.lohi[metric];
    if (!lohi || !cell) return null;
    const raw = cell[metric];
    if (raw == null || isNaN(raw)) return null;
    return Math.min(1, Math.max(0, (raw - lohi[0]) / (lohi[1] - lohi[0] + 1e-12)));
  }
  function initMap() {
    if (map) return;
    if (typeof L === "undefined") {
      $("#leaflet").innerHTML = '<div style="padding:30px;color:#9aa7b4">Map library loading… needs internet for Leaflet &amp; tiles.</div>';
      return;
    }
    map = L.map("leaflet", { preferCanvas: true, minZoom: 5, maxZoom: 11, zoomControl: true }).setView([44.0, -120.5], 6.4);
    L.tileLayer(DARK_TILES, { subdomains: "abcd", attribution: "© OpenStreetMap © CARTO" }).addTo(map);

    const sel = $("#metric");
    sel.innerHTML = Object.entries(SURF.metrics).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join("");
    buildHexes();
    setSurface("risk");
    sel.addEventListener("change", e => setSurface(e.target.value));
    $("#citiesToggle").addEventListener("change", e => toggleCities(e.target.checked));
    $("#distToggle").addEventListener("change", e => toggleDistricts(e.target.checked));

    toggleCities(true);
    // Click in a gap between tiles: still snap to the nearest cell.
    map.on("click", e => { const c = nearest(e.latlng.lat, e.latlng.lng); selectCell(c); });
  }
  // Draw every H3 hexagon once as a canvas polygon; recolor on metric change.
  // hexPolys[i] corresponds to CELLS[i], so a click hits exactly that cell.
  function buildHexes() {
    const H = META.hexes;
    if (!H) return;
    const renderer = L.canvas({ padding: 0.5 });
    hexLayer = L.layerGroup();
    H.poly.forEach((verts, i) => {
      const cell = CELLS[i];
      const poly = L.polygon(verts, {
        renderer, weight: 0.4, color: HEX_BORDER, fillColor: "#222", fillOpacity: 0.85,
      });
      poly._cell = cell;
      poly.on("click", e => { L.DomEvent.stopPropagation(e); selectCell(cell); });
      poly.on("mouseover", () => poly.setStyle({ weight: 1.7, color: "#fff" }));
      poly.on("mouseout", () => poly.setStyle({ weight: 0.4, color: HEX_BORDER }));
      hexPolys.push(poly);
      hexLayer.addLayer(poly);
    });
    hexLayer.addTo(map);
  }
  function setSurface(metric) {
    state.metric = metric;
    const m = SURF.metrics[metric];
    hexPolys.forEach(poly => {
      const v = hexNorm(poly._cell, metric);
      if (v == null) poly.setStyle({ fillOpacity: 0, stroke: false });
      else poly.setStyle({ fillColor: rampFromStops(m.stops, v), fillOpacity: 0.85, stroke: true, weight: 0.4, color: HEX_BORDER });
    });
    if (cityLayer) cityLayer.eachLayer(l => l.bringToFront && l.bringToFront());
    if (distOutline) distOutline.bringToFront();
    if (marker) marker.bringToFront();
    drawLegend(m, "#legend");
  }
  function drawLegend(m, sel) {
    const grad = `linear-gradient(90deg,${m.stops.join(",")})`;
    $(sel).innerHTML = `<div>${m.label}${m.unit ? " (" + m.unit + ")" : ""}</div>
      <div class="bar" style="background:${grad}"></div>
      <div class="ends"><span>${fmt(m.domain[0], 1)}</span><span>${fmt(m.domain[1], 1)}</span></div>`;
  }
  function toggleCities(on) {
    if (on) {
      if (cityLayer) return;
      cityLayer = L.layerGroup();
      META.cities.forEach(ct => {
        const r = ct.pop > 100000 ? 7 : ct.pop > 25000 ? 5 : 3.5;
        const mk = L.circleMarker([ct.lat, ct.lon], {
          radius: r, color: "#ffffff", weight: 1.5, fillColor: "#ffe3b3", fillOpacity: 0.95,
        }).bindTooltip(ct.name, { direction: "top", offset: [0, -4], permanent: ct.pop > 80000, className: "citytip" });
        mk.on("click", e => { L.DomEvent.stopPropagation(e); selectCity(ct); });
        cityLayer.addLayer(mk);
      });
      cityLayer.addTo(map);
    } else if (cityLayer) { map.removeLayer(cityLayer); cityLayer = null; }
  }
  function toggleDistricts(on) {
    if (on && META.districts_geo) {
      distOutline = L.geoJSON(META.districts_geo, {
        style: { color: "#ffe3b3", weight: 1.3, fill: false, opacity: 0.85 },
        onEachFeature: (f, l) => l.bindTooltip(f.properties.district, { sticky: true, className: "citytip" })
      }).addTo(map);
    } else if (distOutline) { map.removeLayer(distOutline); distOutline = null; }
  }
  function selectCity(ct) {
    const c = nearest(ct.lat, ct.lon); if (!c) return;
    selectCell({ ...c, _city: ct.name });
    map.setView([ct.lat, ct.lon], Math.max(map.getZoom(), 7));
  }
  function selectCell(c) {
    if (!c) return;
    if (marker) map.removeLayer(marker);
    marker = L.circleMarker([c.lat, c.lon], { radius: 9, color: "#fff", weight: 2, fillColor: "#e8742b", fillOpacity: .9 }).addTo(map);
    showDetail(c);
  }

  /* ---------- detail panel ---------- */
  function showDetail(p) {
    const kv = rows => `<div class="kv">` + rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("") + `</div>`;
    const gauge = `<div class="riskgauge"><div class="track"></div><div class="marker"><i style="left:${(p.risk_pct * 100).toFixed(0)}%"></i></div></div>`;
    const climate = [["Max temp", fmt(p.tmax, 1) + " °C"], ["Min RH", fmt(p.rmin, 0) + " %"], ["VPD", fmt(p.vpd, 2) + " kPa"],
      ["Wind", fmt(p.wind, 1) + " m/s"], ["Precip", fmt(p.precip, 1) + " mm"], ["ERC", fmt(p.erc, 0)],
      ["Burning index", fmt(p.bi, 0)], ["PDSI (drought)", fmt(p.pdsi, 2)], ["NDVI", fmt(p.ndvi, 2)], ["Days since rain", fmt(p.days_since_rain, 0)]];
    const headTitle = p._city ? p._city : (p.near_city || "Cell " + p.id.slice(0, 8) + "…");
    const subTitle = p._city ? `Nearest grid cell · ${p.eco}` : `${p.eco} · ${p.near_km} km from ${p.near_city}`;
    const d = $("#detail");
    d.innerHTML = `<h3>${headTitle}</h3><div class="muted small" style="margin-bottom:8px">${subTitle}</div>
      <h4>Modeled wildfire risk</h4>
      <div class="kv"><div class="k">Risk (calibrated)</div><div class="v">${fmt(p.risk * 100, 2)}%</div>
        <div class="k">Statewide percentile</div><div class="v">${(p.risk_pct * 100).toFixed(0)}ᵗʰ</div></div>${gauge}
      <h4>Location</h4>${kv([["Nearest city", `${p.near_city} (${p.near_km} km)`], ["Latitude", fmt(p.lat, 3)], ["Longitude", fmt(p.lon, 3)], ["Elevation", fmt(p.elev, 0) + " m"], ["Slope", fmt(p.slope, 1) + "°"], ["Aspect", fmt(p.aspect, 0) + "°"]])}
      <h4>Land cover &amp; fuel</h4>${kv([["Land cover", p.landcover], ["Fuel load (0–1)", fmt(p.fuel, 2)]])}
      <h4>Climate normals (2001–2024 mean)</h4>${kv(climate)}
      <h4>Fire history</h4>${kv([["Burned cell-weeks", p.fires_total], ["Fire rate", fmt(p.fires_rate, 2) + " %"]])}<div id="cellyears"></div>`;
    $("#cellyears", d).appendChild(yearChart(p.fires_by_year, 340, 120));
  }

  /* ---------- forecast tab (true hexagons, recolored per week) ---------- */
  const FC_BORDER = "rgba(6,9,13,.4)";
  let fcMap, fcHexPolys = [], fcWeekIdx = 0, fcCityLayer;
  function initForecast() {
    if (fcMap || !META.forecast) {
      if (!META.forecast) $("#fc-headline").innerHTML = '<div class="panel">No forecast available — run scripts/build_site.py.</div>';
      return;
    }
    const F = META.forecast;
    $("#fc-year").textContent = F.target_year;
    const nw = F.next_week;
    $("#fc-headline").innerHTML = `
      <div class="fc-card"><div class="big">${nw.label}</div><div class="lbl">Next forecast week</div></div>
      <div class="fc-card"><div class="big">${Math.round(nw.expected_state)}</div><div class="lbl">Expected fires that week</div></div>
      <div class="fc-card"><div class="big">${nw.max_risk.toFixed(2)}%</div><div class="lbl">Max single-cell risk</div></div>
      <div class="fc-card"><div class="big">${F.n_weeks}</div><div class="lbl">Forecast weeks (May–Oct)</div></div>`;
    $("#fc-slider").max = F.weeks.length - 1;

    fcMap = L.map("fc-map", { preferCanvas: true, minZoom: 5, maxZoom: 10 }).setView([44, -120.5], 6.3);
    L.tileLayer(DARK_TILES, { subdomains: "abcd" }).addTo(fcMap);
    buildForecastHexes();
    fcCityLayer = L.layerGroup();
    META.cities.filter(c => c.pop > 50000).forEach(ct => {
      L.circleMarker([ct.lat, ct.lon], { radius: 4, color: "#fff", weight: 1, fillColor: "#ffe3b3", fillOpacity: .9 })
        .bindTooltip(ct.name, { direction: "top", offset: [0, -4], className: "citytip" }).addTo(fcCityLayer);
    });
    fcCityLayer.addTo(fcMap);
    setForecastWeek(0);

    $("#fc-slider").addEventListener("input", e => setForecastWeek(+e.target.value));
    $("#fc-prev").addEventListener("click", () => setForecastWeek(Math.max(0, fcWeekIdx - 1)));
    $("#fc-next").addEventListener("click", () => setForecastWeek(Math.min(F.weeks.length - 1, fcWeekIdx + 1)));

    // Build the seasonal curve chart.
    const curveVals = F.predicted.weekly_curve.map(w => Math.round(w.expected));
    const labels = F.predicted.weekly_curve.map(w => w.date.slice(5));
    $("#fc-curve").innerHTML = "";
    $("#fc-curve").appendChild(yearChart(curveVals, 980, 180, labels));
  }
  // The same H3 hexagons as the risk map, on their own Leaflet instance.
  function buildForecastHexes() {
    const H = META.hexes;
    if (!H) return;
    const renderer = L.canvas({ padding: 0.5 });
    const layer = L.layerGroup();
    H.poly.forEach((verts, i) => {
      const poly = L.polygon(verts, { renderer, weight: 0.3, color: FC_BORDER, fillColor: "#111", fillOpacity: 0.85 });
      poly._idx = i;
      poly.on("click", e => { L.DomEvent.stopPropagation(e); selectForecastCell(i); });
      poly.on("mouseover", () => poly.setStyle({ weight: 1.5, color: "#fff" }));
      poly.on("mouseout", () => poly.setStyle({ weight: 0.3, color: FC_BORDER }));
      fcHexPolys.push(poly);
      layer.addLayer(poly);
    });
    layer.addTo(fcMap);
  }
  function selectForecastCell(i) {
    const cell = CELLS[i], F = META.forecast;
    const rk = (F.hex.weeks[fcWeekIdx][i] / 100 * F.hex.vmax_pct).toFixed(2);
    fcHexPolys[i].bindPopup(
      `<b>${cell.near_city}</b> <span class="muted">(${cell.near_km} km)</span><br>${cell.eco}<br>` +
      `Forecast risk this week: <b>${rk}%</b>`, { className: "citytip" }).openPopup();
  }
  function setForecastWeek(i) {
    fcWeekIdx = i; const F = META.forecast, wk = F.weeks[i], col = F.hex.weeks[i];
    $("#fc-slider").value = i;
    $("#fc-label").textContent = `${wk.label} 2025 · week ${i + 1} of ${F.weeks.length}`;
    fcHexPolys.forEach(poly => {
      const t = col[poly._idx] / 100;
      if (!t) poly.setStyle({ fillColor: "#0c0f14", fillOpacity: 0.06, stroke: false });
      else poly.setStyle({ fillColor: rampFromStops(RISK_STOPS, t), fillOpacity: 0.85, stroke: true, weight: 0.3, color: FC_BORDER });
    });
    if (fcCityLayer) fcCityLayer.eachLayer(l => l.bringToFront && l.bringToFront());
    drawLegend({ label: "Risk", unit: "%", stops: RISK_STOPS, domain: [0, F.hex.vmax_pct] }, "#fc-legend");
    $("#fc-stats").innerHTML = `<div class="kv">
      <div class="k">Expected statewide fires</div><div class="v">${Math.round(wk.expected_state)}</div>
      <div class="k">Max single-cell risk</div><div class="v">${wk.max_risk.toFixed(2)}%</div>
      <div class="k">Mean risk</div><div class="v">${wk.mean_risk.toFixed(3)}%</div></div>`;
    const dists = Object.entries(wk.district_expected).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const max = dists[0] ? dists[0][1] : 1;
    $("#fc-dists").innerHTML = `<div class="bars">` + dists.map(([n, v]) =>
      `<div class="row"><div class="name" title="${n}">${n}</div>
        <div class="track"><div class="fill" style="width:${(v / max * 100).toFixed(1)}%"></div></div>
        <div class="val">${Math.round(v)}</div></div>`).join("") + `</div>`;
  }

  /* ---------- predictions tracker ---------- */
  // Hex map of the locked seasonal forecast: every cell coloured by predicted risk,
  // the 40 highest-risk cells outlined (green once they're confirmed burned).
  let trMap, trHexPolys = [];
  function initTrackerMap() {
    if (trMap || typeof L === "undefined" || !META.hexes || !(META.forecast && META.forecast.hex)) return;
    const H = META.hexes, FH = META.forecast.hex, season = FH.season;
    const P = window.WF_PREDICTIONS || META.predictions;
    const topIds = new Set(((P && P.predicted && P.predicted.top_cells) || []).map(c => c.id));
    const burned = {};
    if (P && P.actuals && P.actuals.top_cells) P.actuals.top_cells.forEach(c => { burned[c.id] = c.actual_weeks_burned; });
    trMap = L.map("tr-map", { preferCanvas: true, minZoom: 5, maxZoom: 10 }).setView([44, -120.5], 6.2);
    L.tileLayer(DARK_TILES, { subdomains: "abcd" }).addTo(trMap);
    const renderer = L.canvas({ padding: 0.5 });
    const layer = L.layerGroup();
    H.poly.forEach((verts, i) => {
      const cell = CELLS[i], t = (season[i] || 0) / 100;
      const isTop = topIds.has(cell.id), didBurn = burned[cell.id] > 0;
      const poly = L.polygon(verts, {
        renderer, fillColor: rampFromStops(RISK_STOPS, t), fillOpacity: t ? 0.85 : 0.05,
        color: isTop ? (didBurn ? "#36d399" : "#ffffff") : "rgba(6,9,13,.4)", weight: isTop ? 1.8 : 0.3,
      });
      poly.on("click", e => {
        L.DomEvent.stopPropagation(e);
        const rk = (t * FH.season_max_pct).toFixed(2);
        const actLine = (P && P.actuals) ? `<br>${didBurn ? `burned ${burned[cell.id]} wk in ${P.target_year}` : (isTop ? "no fire in " + P.target_year : "")}` : "";
        poly.bindPopup(
          `<b>${cell.near_city}</b> <span class="muted">(${cell.near_km} km)</span><br>${cell.eco}<br>` +
          `Seasonal pred. risk: <b>${rk}%</b>${isTop ? " · top-40" : ""}${actLine}`, { className: "citytip" }).openPopup();
      });
      if (isTop) poly.bringToFront();
      trHexPolys.push(poly);
      layer.addLayer(poly);
    });
    layer.addTo(trMap);
    drawLegend({ label: "Seasonal predicted risk", unit: "%", stops: RISK_STOPS, domain: [0, FH.season_max_pct] }, "#tr-legend");
    setTimeout(() => trMap.invalidateSize(), 60);
  }
  function buildTracker() {
    initTrackerMap();
    const P = window.WF_PREDICTIONS || META.predictions;
    if (!P) { $("#tr-status").innerHTML = '<div class="muted">No locked predictions yet — run scripts/build_site.py.</div>'; return; }
    const A = P.actuals;
    const status = A
      ? `<div><span class="badge ok">verified</span> &nbsp;Actuals pulled ${A.verified_at_utc.slice(0, 10)} for ${A.target_year} (state actual: ${A.state_actual_fires} fires).</div>`
      : `<div><span class="badge wait">awaiting actuals</span> &nbsp;Predictions locked at ${P.locked_at_utc.slice(0, 10)}. Run <code>uv run python scripts/verify_predictions.py</code> after the ${P.target_year} fire season to compare.</div>`;
    $("#tr-status").innerHTML = status;

    // ---- statewide ----
    const sp = P.predicted.state_expected_fires, sa = A ? A.state_actual_fires : null;
    const stateRow = `<table class="t"><tr><th>Metric</th><th>Predicted</th><th>Actual</th><th>Error</th></tr>
      <tr><td>Total burned cell-weeks (${P.target_year} season)</td><td>${Math.round(sp)}</td><td>${sa != null ? sa : "—"}</td><td>${sa != null ? fmtErr(sp, sa) : "—"}</td></tr></table>`;
    $("#tr-state").innerHTML = stateRow;

    // ---- per district ----
    const aMap = A && A.districts ? Object.fromEntries(A.districts.map(d => [d.district, d.actual_fires])) : {};
    const drows = P.predicted.districts.map(d => {
      const act = aMap[d.district];
      return `<tr><td>${d.district}</td><td>${Math.round(d.expected_fires)}</td><td>${act != null ? act : "—"}</td><td>${act != null ? fmtErr(d.expected_fires, act) : "—"}</td></tr>`;
    }).join("");
    $("#tr-districts").innerHTML = `<table class="t"><tr><th>ODF District</th><th>Pred. fires</th><th>Actual</th><th>Error</th></tr>${drows}</table>`;

    // ---- top cells ----
    const cellAct = A && A.top_cells ? Object.fromEntries(A.top_cells.map(c => [c.id, c.actual_weeks_burned])) : {};
    const crows = P.predicted.top_cells.map((c, i) => {
      const act = cellAct[c.id];
      const hit = act != null ? (act > 0 ? `<span class="badge ok">${act} weeks burned</span>` : `<span class="badge miss">no fire</span>`) : "—";
      return `<tr><td>${i + 1}</td><td>${c.near_city} (${c.near_km} km)</td><td>${c.eco}</td><td>${c.pred_risk.toFixed(2)}%</td><td>${hit}</td></tr>`;
    }).join("");
    $("#tr-cells").innerHTML = `<table class="t"><tr><th>#</th><th>Nearest city</th><th>Ecoregion</th><th>Pred. risk</th><th>Actual</th></tr>${crows}</table>`;

    // ---- weekly curve ----
    $("#tr-curve").innerHTML = "";
    const wk = P.predicted.weekly_curve;
    const aw = A && A.weekly_curve ? Object.fromEntries(A.weekly_curve.map(w => [w.date, w.actual])) : {};
    const pred = wk.map(w => Math.round(w.expected));
    const labels = wk.map(w => w.date.slice(5));
    const actual = A ? wk.map(w => aw[w.date] != null ? aw[w.date] : 0) : null;
    $("#tr-curve").appendChild(dualLineChart(labels, pred, actual, 980, 200));
  }
  function fmtErr(pred, act) {
    const e = pred - act, pct = act === 0 ? "—" : `${(100 * e / act).toFixed(0)}%`;
    const cls = Math.abs(e) <= Math.max(2, act * 0.2) ? "ok" : "miss";
    return `<span class="badge ${cls}">${e >= 0 ? "+" : ""}${Math.round(e)}${pct !== "—" ? " (" + pct + ")" : ""}</span>`;
  }
  function dualLineChart(labels, pred, actual, w, h) {
    const pad = { l: 36, r: 8, t: 12, b: 22 }, iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const series = actual ? pred.concat(actual) : pred;
    const max = Math.max(...series, 1);
    const sx = i => pad.l + (i / Math.max(1, labels.length - 1)) * iw;
    const sy = v => pad.t + ih - (v / max) * ih;
    const path = arr => arr.map((v, i) => `${i ? "L" : "M"}${sx(i)},${sy(v)}`).join("");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("width", "100%");
    let html = "";
    [0, .5, 1].forEach(g => { const y = pad.t + ih * (1 - g);
      html += `<line x1="${pad.l}" y1="${y}" x2="${w - pad.r}" y2="${y}" stroke="#2b3440"/>
               <text x="${pad.l - 5}" y="${y + 3}" fill="#6b7785" font-size="10" text-anchor="end">${Math.round(max * g)}</text>`; });
    html += `<path d="${path(pred)}" fill="none" stroke="#e8742b" stroke-width="2.2"/>`;
    if (actual) html += `<path d="${path(actual)}" fill="none" stroke="#7fd1ff" stroke-width="2.2" stroke-dasharray="4 3"/>`;
    labels.forEach((l, i) => { if (i % 4 === 0) html += `<text x="${sx(i)}" y="${h - 6}" fill="#6b7785" font-size="10" text-anchor="middle">${l}</text>`; });
    html += `<g font-size="11" fill="#cdd7e0">
      <rect x="${w - 200}" y="6" width="194" height="22" fill="#161b22" stroke="#2b3440" rx="4"/>
      <line x1="${w - 192}" y1="17" x2="${w - 172}" y2="17" stroke="#e8742b" stroke-width="2.2"/>
      <text x="${w - 168}" y="20">predicted</text>
      <line x1="${w - 110}" y1="17" x2="${w - 90}" y2="17" stroke="#7fd1ff" stroke-width="2.2" stroke-dasharray="4 3"/>
      <text x="${w - 86}" y="20">${actual ? "actual" : "(awaiting)"}</text></g>`;
    svg.innerHTML = html; return svg;
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
      if (!META.districts_geo && !dmap) $("#distmap").innerHTML = '<div style="padding:20px;color:#9aa7b4">District layer unavailable.</div>';
      return;
    }
    dmap = L.map("distmap", { zoomControl: true, minZoom: 5, maxZoom: 9 }).setView([44.0, -120.5], 6);
    L.tileLayer(DARK_TILES, { subdomains: "abcd" }).addTo(dmap);
    const vals = META.districts_geo.features.map(f => f.properties.pred_count || 0), max = Math.max(...vals) || 1;
    dLayer = L.geoJSON(META.districts_geo, {
      style: f => ({ fillColor: rampFromStops(RISK_STOPS, (f.properties.pred_count || 0) / max), color: "#0b0e12", weight: 1, fillOpacity: .8 }),
      onEachFeature: (f, l) => {
        const p = f.properties;
        l.bindTooltip(`<b>${p.district}</b><br>${Math.round(p.pred_count)} fires (95% ${Math.round(p.pred_lo)}–${Math.round(p.pred_hi)})`, { sticky: true, className: "citytip" });
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

  /* ---------- live fire watch ---------- */
  const FIRMS_STOPS = ["#3a1505", "#e8742b", "#ffd98a"];
  let lwMap, lwDetLayer;
  function cnnTag(d) {
    if (d.cnn_prob == null) return '<span class="badge">not imaged</span>';
    const pct = (d.cnn_prob * 100).toFixed(0) + "%";
    return d.cnn_prob >= 0.5
      ? `<span class="badge ok">burn-scar confirmed · ${pct}</span>`
      : `<span class="badge wait">low burn signal · ${pct}</span>`;
  }
  function livePopup(d) {
    return `<b>${d.near_city || "Cell"}</b> ${d.near_km != null ? `<span class="muted">(${d.near_km} km)</span>` : ""}<br>
      <span class="muted">${d.eco || ""}</span><br>
      Brightness <b>${d.t21 != null ? d.t21 + " K" : "—"}</b> · FIRMS conf ${d.confidence != null ? d.confidence : "—"}${d.frp != null ? ` · FRP ${d.frp}` : ""}<br>
      Detected ${d.acq_date || "—"}<br>${cnnTag(d)}
      ${d.thumb ? `<br><img class="lw-thumb" src="${d.thumb}" alt="Sentinel-2 patch" onerror="this.style.display='none'"/>` : ""}`;
  }
  function buildLiveWatch() {
    const LV = window.WF_LIVE;
    if (!LV) { $("#lw-status").innerHTML = '<div class="muted">No scan yet — run <code>scripts/live_fire_scan.py</code>.</div>'; return; }
    const demo = LV.mode !== "live";
    const when = (LV.generated_utc || "").slice(0, 16).replace("T", " ");
    $("#lw-status").innerHTML = `<div class="lw-statline">
      <span class="badge ${demo ? "wait" : "ok"}">${demo ? "demo data" : "live"}</span>
      <span><b>${LV.n_active}</b> active cell(s) · <b>${LV.n_confirmed}</b> CNN-confirmed burn(s) · source ${LV.source} · scanned ${when} UTC</span></div>
      ${demo ? '<div class="muted small" style="margin-top:8px">Demo mode: synthetic FIRMS detections seeded in high-risk cells, each scored by the <b>real trained burn-scar CNN</b>. The scheduled job swaps this for live Earth Engine FIRMS plus Sentinel-2 imagery and thumbnails.</div>' : ""}`;
    $("#lw-source").textContent = demo ? "Showing demo data until the scheduled scan runs." : "";
    $("#lw-count").textContent = `· ${LV.detections.length}`;
    initLiveMap(LV);
    renderLiveList(LV);
  }
  function initLiveMap(LV) {
    if (!lwMap) {
      if (typeof L === "undefined") { $("#lw-map").innerHTML = '<div style="padding:20px;color:#9aa7b4">Map needs internet for Leaflet.</div>'; return; }
      lwMap = L.map("lw-map", { preferCanvas: true, minZoom: 5, maxZoom: 11 }).setView([44.0, -120.5], 6.3);
      L.tileLayer(DARK_TILES, { subdomains: "abcd" }).addTo(lwMap);
      const r0 = L.canvas({ padding: 0.5 }), base = L.layerGroup();
      META.hexes.poly.forEach(v => L.polygon(v, { renderer: r0, weight: 0.2, color: "rgba(70,80,92,.35)", fill: false }).addTo(base));
      base.addTo(lwMap);
      lwDetLayer = L.layerGroup().addTo(lwMap);
      drawLegend({ label: "FIRMS brightness", unit: "K", stops: FIRMS_STOPS, domain: [310, 420] }, "#lw-legend");
    }
    lwDetLayer.clearLayers();
    const r = L.canvas({ padding: 0.5 });
    LV.detections.forEach(d => {
      const i = ID2I[d.id], sev = d.t21 != null ? Math.min(1, Math.max(0, (d.t21 - 310) / 110)) : 0.6;
      if (i != null) L.polygon(META.hexes.poly[i], { renderer: r, weight: 0.7, color: "#ffce8a", fillColor: rampFromStops(FIRMS_STOPS, sev), fillOpacity: 0.85 }).addTo(lwDetLayer);
      L.circleMarker([d.lat, d.lon], { radius: 6, color: "#fff", weight: 1.4, fillColor: "#ff4d3d", fillOpacity: 0.95 })
        .bindPopup(livePopup(d), { className: "citytip", maxWidth: 240 }).addTo(lwDetLayer);
    });
  }
  function renderLiveList(LV) {
    if (!LV.detections.length) { $("#lw-list").innerHTML = '<div class="muted">No active detections in the current window. Quiet skies.</div>'; return; }
    $("#lw-list").innerHTML = LV.detections.map((d, k) => `
      <div class="lw-item" data-k="${k}">
        ${d.thumb ? `<img class="lw-thumb" src="${d.thumb}" alt="" onerror="this.style.display='none'"/>` : `<div class="lw-thumb lw-thumb-ph">no img</div>`}
        <div class="lw-meta">
          <div class="lw-title">${d.near_city || "Cell"} ${d.near_km != null ? `<span class="muted">${d.near_km} km</span>` : ""}</div>
          <div class="muted small">${d.eco || ""}${d.acq_date ? " · " + d.acq_date : ""}</div>
          <div class="lw-nums"><span>${d.t21 != null ? d.t21 + " K" : "—"}</span><span>conf ${d.confidence != null ? d.confidence : "—"}</span>${d.frp != null ? `<span>FRP ${d.frp}</span>` : ""}</div>
          <div>${cnnTag(d)}</div>
        </div>
      </div>`).join("");
    if (!$("#lw-list").dataset.bound) {
      $("#lw-list").dataset.bound = "1";
      $("#lw-list").addEventListener("click", ev => {
        const it = ev.target.closest(".lw-item"); if (!it || !lwMap) return;
        const d = (window.WF_LIVE.detections || [])[+it.dataset.k]; if (!d) return;
        lwMap.setView([d.lat, d.lon], Math.max(lwMap.getZoom(), 8));
      });
    }
  }

  /* ---------- fire danger check ---------- */
  const DG_COLORS = ["#2f7d4f", "#6bbf4a", "#f2c43d", "#e8742b", "#d23b3b"];
  const DG_LABELS = ["None", "Low", "Moderate", "High", "Extreme"];
  const AGREE_COLORS = ["#2f7d4f", "#c4a13a", "#c0432f"]; // exact, off-by-one, off>=2
  function dgBadge(i) { return `<span class="dbadge" style="background:${DG_COLORS[i]}">${DG_LABELS[i]}</span>`; }
  function buildDanger() {
    const D = META.danger;
    if (!D || !D.districts || !D.districts.length) { $("#dg-cards").innerHTML = '<div class="muted">No danger comparison available — rebuild the site.</div>'; return; }
    const host = $("#danger");
    if (host.dataset.built) return;
    host.dataset.built = "1";
    const s = D.summary;
    $("#dg-cards").innerHTML = [
      { big: (s.exact_rate * 100).toFixed(0) + "%", lbl: "Exact agreement", note: "model class = conditions class" },
      { big: (s.within1_rate * 100).toFixed(0) + "%", lbl: "Within one class", note: s.n_district_weeks + " district-weeks" },
      { big: s.mean_abs_class_diff.toFixed(2), lbl: "Mean class gap", note: "avg |model − conditions|" },
      { big: D.districts.length, lbl: "ODF districts", note: "vs ERC climatology" },
    ].map(c => `<div class="card"><div class="big">${c.big}</div><div class="lbl">${c.lbl}</div><div class="note">${c.note}</div></div>`).join("");
    $("#dg-week").textContent = "· " + D.current_week_label;

    // This week, per district.
    const cur = D.districts.map(d => {
      const m = d.current.model, cn = d.current.cond, diff = Math.abs(m - cn);
      const gap = diff === 0 ? '<span class="badge ok">match</span>'
        : `<span class="badge ${diff === 1 ? "wait" : "miss"}">${m > cn ? "model hotter" : "model cooler"} · ${diff}</span>`;
      return `<tr><td>${d.district}</td><td>${dgBadge(m)}</td><td>${dgBadge(cn)}</td><td>${d.current.erc != null ? d.current.erc : "—"}</td><td>${gap}</td></tr>`;
    }).join("");
    $("#dg-current").innerHTML = `<table class="t"><tr><th>District</th><th>Model says</th><th>Conditions say</th><th>ERC</th><th>Gap</th></tr>${cur}</table>`;

    // Season agreement grid.
    $("#dg-legend-grid").innerHTML = ["match", "off by one", "off by 2+"]
      .map((t, i) => `<span class="dg-key"><i style="background:${AGREE_COLORS[i]}"></i>${t}</span>`).join("");
    const weeks = D.districts[0].weeks;
    let grid = `<div class="dg-grid" style="grid-template-columns:150px repeat(${weeks.length},minmax(10px,1fr))">`;
    grid += `<div class="dg-corner"></div>` + weeks.map(w => `<div class="dg-col">${w.label.replace(/^[A-Za-z]+ /, "")}</div>`).join("");
    D.districts.forEach(d => {
      grid += `<div class="dg-rowlbl" title="${d.district}: ${(d.agreement_rate * 100).toFixed(0)}% exact">${d.district}</div>`;
      d.weeks.forEach(w => {
        const diff = Math.abs(w.model - w.cond), col = AGREE_COLORS[Math.min(2, diff)];
        grid += `<div class="dg-cell" style="background:${col}" title="${d.district} · ${w.label}&#10;Model: ${DG_LABELS[w.model]} (${w.model_pct}%)&#10;Conditions: ${DG_LABELS[w.cond]} — ERC ${w.erc != null ? w.erc : "—"} (${w.cond_pct}%)"></div>`;
      });
    });
    grid += `</div>`;
    $("#dg-grid").innerHTML = grid;

    // Confusion matrix.
    const cm = s.confusion, maxc = Math.max(1, ...cm.flat());
    let conf = `<table class="t dg-confusion"><tr><th>cond ↓ \\ model →</th>${DG_LABELS.map(l => `<th>${l}</th>`).join("")}</tr>`;
    cm.forEach((row, i) => {
      conf += `<tr><th>${DG_LABELS[i]}</th>` + row.map((v, j) => {
        const a = v / maxc, rgb = i === j ? "47,125,79" : "200,90,55";
        return `<td style="background:rgba(${rgb},${(0.10 + 0.82 * a).toFixed(2)})">${v}</td>`;
      }).join("") + `</tr>`;
    });
    conf += `</table>`;
    $("#dg-confusion").innerHTML = conf;
  }

  buildOverview(); buildResults(); buildData();
})();
