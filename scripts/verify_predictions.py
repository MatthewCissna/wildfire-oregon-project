"""Verify the locked-in predictions against actual MODIS observations.

Reads ``site/data/predictions.json`` (predictions locked at build time),
pulls real MODIS fire labels for the target year via Earth Engine (resumable;
reuses the existing per-timestep cache), aggregates them to the same units the
predictions live in (statewide total, per district, per top-cell), and fills the
``actuals`` block of the predictions file. Re-running build_site.py picks up
the updated file and the Tracker tab shows predicted vs. actual side-by-side.

    uv run python scripts/verify_predictions.py            # uses target_year from file
    uv run python scripts/verify_predictions.py --year 2025
"""

from __future__ import annotations

import argparse
import json
import sys

import geopandas as gpd
import pandas as pd

from wildfire.config import REPO_ROOT, load_config
from wildfire.utils import init_console


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None,
                    help="target year; defaults to predictions.json's target_year")
    args = ap.parse_args()

    init_console()
    cfg = load_config()
    pred_path = REPO_ROOT / "site" / "data" / "predictions.json"
    if not pred_path.exists():
        print("❌ No site/data/predictions.json yet. Run scripts/build_site.py first.")
        return 1
    pred = json.loads(pred_path.read_text())
    year = args.year or int(pred["target_year"])

    if not cfg.ee_project:
        print("⚠️  No Earth Engine project configured; cannot pull real actuals.")
        print("   See docs/earth_engine_setup.md, then re-run.")
        return 2

    try:
        from wildfire.features.grid import build_grid, time_index
        from wildfire.ingest.earth_engine import pull_fire_labels
        from wildfire.ingest.ee_auth import initialize_ee
    except Exception as exc:
        print(f"❌ Could not import GEE ingest: {exc}")
        return 3

    initialize_ee(cfg)
    grid = build_grid(cfg)
    months = set(cfg.get("time.fire_season_months", [5, 6, 7, 8, 9, 10]))
    dates = pd.date_range(f"{year}-04-28", f"{year}-11-10", freq="W-MON")
    dates = pd.DatetimeIndex([d for d in dates if d.month in months])
    print(f"Pulling actual MODIS labels for {year}: {len(dates)} weeks ...")
    labels = pull_fire_labels(cfg, grid, dates, max_workers=8)

    # State total: total burned cell-weeks across the season.
    state_actual = int(labels["fire"].sum())
    # Per-district actuals.
    actual_districts = None
    try:
        from wildfire.ingest.districts import assign_districts

        dmap = assign_districts(grid, cfg)
        merged = labels.merge(dmap, on="cell_id", how="left")
        actual_districts = (
            merged.groupby("district")["fire"].sum().astype(int).sort_values(ascending=False)
            .reset_index().rename(columns={"fire": "actual_fires"}).to_dict(orient="records")
        )
    except Exception as exc:
        print(f"   (district aggregation skipped: {exc})")

    # Top-cell actuals.
    cell_act = labels.groupby("cell_id")["fire"].sum().to_dict()
    top_cells_actual = [
        {"id": c["id"], "predicted": c["pred_risk"], "actual_weeks_burned": int(cell_act.get(c["id"], 0))}
        for c in pred["predicted"]["top_cells"]
    ]
    hits = sum(1 for c in top_cells_actual if c["actual_weeks_burned"] > 0)

    # Weekly state curve (actual).
    weekly_actual = (
        labels.groupby(pd.to_datetime(labels["date"]))["fire"].sum().reset_index()
        .rename(columns={"date": "date_dt"})
    )
    weekly_actual["date"] = weekly_actual["date_dt"].dt.strftime("%Y-%m-%d")
    weekly_actual_map = dict(zip(weekly_actual["date"], weekly_actual["fire"].astype(int)))

    pred["actuals"] = {
        "verified_at_utc": pd.Timestamp.utcnow().isoformat(),
        "target_year": year,
        "state_actual_fires": state_actual,
        "districts": actual_districts,
        "top_cells": top_cells_actual,
        "top_cell_hit_count": hits,
        "weekly_curve": [{"date": d, "actual": v} for d, v in weekly_actual_map.items()],
    }
    pred_path.write_text(json.dumps(pred, indent=2))
    # Also refresh the website-loaded JS so the Tracker tab updates without a full rebuild.
    (REPO_ROOT / "site" / "data" / "predictions.js").write_text(
        "window.WF_PREDICTIONS=" + json.dumps(pred, separators=(",", ":")) + ";",
        encoding="utf-8",
    )
    print(f"\n✅ Verification written -> {pred_path}")
    print(f"   state predicted : {pred['predicted']['state_expected_fires']:.0f} fires")
    print(f"   state actual    : {state_actual}")
    print(f"   top-cell hits   : {hits}/{len(top_cells_actual)} (top cells that did burn)")
    print("   site/data/predictions.js refreshed; commit + push to update the live site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
