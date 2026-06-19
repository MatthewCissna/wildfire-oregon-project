"""Weekly fire-danger comparison: the model's caution vs. conditions-based caution.

The question this answers: for each ODF fire-protection district and each forecast
week, does the model's risk-based caution level line up with the fire-danger level
you'd assign from observed conditions?

Two ratings on the same five-step scale (none / low / moderate / high / extreme):

* **Conditions class** — the standard fire-danger approach. NFDRS sets adjective
  fire-danger from where today's fire-danger index sits in its *climatological*
  distribution. We use GRIDMET's Energy Release Component (ERC), the index Oregon
  agencies lean on, and take the district's weekly-mean ERC percentile against that
  district's full 2001-2024 record. Percentile breakpoints follow the NFDRS-style
  adjective bins.
* **Model class** — the district's predicted fire activity for that week (from the
  forecast) placed on the same percentile scale, across the forecast season.

Agreement is then an honest check: exact match, off by one step, or further apart,
plus a confusion matrix across every district-week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The user-facing scale. Index 0..4 maps to these labels.
CLASSES = ["none", "low", "moderate", "high", "extreme"]

# Percentile breakpoints between the five classes (NFDRS-style adjective bins: the
# bottom fifth is "none/very-low", the top reserved for "extreme").
BREAKPOINTS = [20.0, 40.0, 65.0, 88.0]


def classify_percentile(pct: float) -> int:
    """Map a 0-100 percentile to a class index 0..4."""
    for i, b in enumerate(BREAKPOINTS):
        if pct < b:
            return i
    return len(BREAKPOINTS)


def _percentile_of(value: float, dist: np.ndarray) -> float:
    """Percentile (0-100) of ``value`` within ``dist`` (empirical CDF)."""
    if value is None or not np.isfinite(value) or len(dist) == 0:
        return 0.0
    return float((dist <= value).mean() * 100.0)


def compute_fire_danger(cfg, features: pd.DataFrame, forecast: dict, dmap: pd.DataFrame) -> dict:
    """Build the per-district, per-week danger comparison.

    Args:
        features: the spatiotemporal panel (needs ``cell_id``, ``date``, ``erc``).
        forecast: the dict from ``build_forecast`` (its ``weeks`` carry per-district
            expected fires per week).
        dmap: DataFrame[cell_id, district] from ``assign_districts``.

    Returns a compact dict shipped to the site as ``WF_META.danger``.
    """
    weeks = forecast.get("weeks") or []
    if not weeks:
        return {"classes": CLASSES, "districts": [], "summary": {}}

    # ---- climatological weekly ERC per district ----
    df = features[["cell_id", "date", "erc"]].dropna(subset=["erc"]).copy()
    df = df.merge(dmap, on="cell_id", how="left")
    df = df[df["district"].notna()]
    df["date"] = pd.to_datetime(df["date"])
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)

    # Weekly district ERC = mean ERC over the district's cells for each date.
    wk = df.groupby(["district", "date", "woy"], observed=True)["erc"].mean().reset_index()
    # Climatology: average weekly ERC per (district, week-of-year) across all years.
    clim = wk.groupby(["district", "woy"], observed=True)["erc"].mean()
    # Full historical weekly-ERC distribution per district (percentile reference).
    erc_dist = {d: g["erc"].to_numpy() for d, g in wk.groupby("district", observed=True)}

    # Order forecast weeks once; pull each week's week-of-year.
    wk_meta = []
    for w in weeks:
        d = pd.Timestamp(w["date"])
        wk_meta.append((w, d, int(d.isocalendar().week)))

    # Which forecast week is closest to "now" — used for the per-district snapshot.
    now_woy = int(pd.Timestamp.utcnow().isocalendar().week)
    cur_idx = min(range(len(wk_meta)), key=lambda i: abs(wk_meta[i][2] - now_woy))
    current_label = weeks[cur_idx]["label"]

    # The 12 named ODF protection districts (drop the catch-all unprotected area,
    # whose ERC climatology and model signal span too much terrain to be meaningful).
    from wildfire.ingest.districts import OUTSIDE
    districts = sorted({k for w in weeks for k in (w.get("district_expected") or {}) if k != OUTSIDE})

    out_districts = []
    confusion = [[0] * len(CLASSES) for _ in range(len(CLASSES))]
    n_dw = exact = within1 = 0
    abs_diff_sum = 0

    for dist in districts:
        ref = erc_dist.get(dist, np.array([]))
        # The model's per-week expected fires for this district across the season.
        model_series = np.array([
            float((w.get("district_expected") or {}).get(dist, 0.0)) for w in weeks
        ])

        wk_rows = []
        for i, (w, d, woy) in enumerate(wk_meta):
            erc_val = clim.get((dist, woy))
            if erc_val is None or not np.isfinite(erc_val):
                # Nearest available week-of-year for this district.
                avail = [woy2 for (dd, woy2) in clim.index if dd == dist]
                erc_val = clim.get((dist, min(avail, key=lambda x: abs(x - woy)))) if avail else None
            cond_pct = _percentile_of(erc_val, ref)
            model_pct = _percentile_of(model_series[i], model_series)
            cond_cls = classify_percentile(cond_pct)
            model_cls = classify_percentile(model_pct)

            wk_rows.append({
                "date": w["date"], "label": w["label"],
                "erc": round(float(erc_val), 1) if erc_val is not None else None,
                "cond": cond_cls, "model": model_cls,
                "cond_pct": round(cond_pct, 1), "model_pct": round(model_pct, 1),
            })
            confusion[cond_cls][model_cls] += 1
            diff = abs(cond_cls - model_cls)
            n_dw += 1
            exact += int(diff == 0)
            within1 += int(diff <= 1)
            abs_diff_sum += diff

        d_exact = sum(1 for r in wk_rows if r["cond"] == r["model"])
        out_districts.append({
            "district": dist,
            "weeks": wk_rows,
            "agreement_rate": round(d_exact / max(1, len(wk_rows)), 3),
            "current": wk_rows[cur_idx] if cur_idx < len(wk_rows) else wk_rows[-1],
        })

    # Rank districts by how well the model tracks conditions (worst first is useful too,
    # but the site sorts; keep stable alphabetical here).
    out_districts.sort(key=lambda d: d["district"])

    summary = {
        "n_district_weeks": n_dw,
        "exact": exact, "within1": within1,
        "exact_rate": round(exact / max(1, n_dw), 3),
        "within1_rate": round(within1 / max(1, n_dw), 3),
        "mean_abs_class_diff": round(abs_diff_sum / max(1, n_dw), 3),
        "confusion": confusion,
    }
    return {
        "classes": CLASSES,
        "breakpoints_pct": BREAKPOINTS,
        "index": "ERC (Energy Release Component, GRIDMET)",
        "current_week_label": current_label,
        "districts": out_districts,
        "summary": summary,
    }
