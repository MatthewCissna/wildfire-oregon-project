"""Live fire scan: FIRMS active-fire detections, confirmed by the burn-scar CNN.

Pulls the last few days of FIRMS thermal detections over Oregon (Earth Engine by
default; the NASA NRT API if ``FIRMS_MAP_KEY`` is set), reduces them to the H3 grid,
then for the hottest cells pulls a recent Sentinel-2 patch and runs the trained
detection CNN to flag whether the imagery looks like a real burn scar. The result is
written to ``site/data/live_scan.{json,js}`` and the Live Fire Watch tab renders it.

When Earth Engine isn't configured on this machine, it falls back to a clearly
labeled **demo**: synthetic FIRMS cells seeded in high-risk areas, still run through
the *real* trained CNN, so the image-recognition step is genuine even offline. The
scheduled GitHub Action runs the live path with the service-account key.

    uv run python scripts/live_fire_scan.py
    uv run python scripts/live_fire_scan.py --days 3 --max-confirm 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from wildfire.config import REPO_ROOT, load_config
from wildfire.utils import init_console


def _load_cell_index() -> dict:
    """Map cell_id -> {near_city, near_km, eco, risk} from the site's cells.js."""
    p = REPO_ROOT / "site" / "data" / "cells.js"
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8").strip().rstrip(";")
    blob = json.loads(txt[txt.index("=") + 1:])
    return {c["id"]: c for c in blob.get("cells", [])}


def _cnn_confirm(cfg, active: pd.DataFrame, *, end_date: str, max_confirm: int, live: bool):
    """Return {cell_id: (cnn_prob, thumb_url)} for the hottest active cells.

    Live path pulls recent Sentinel-2 patches from EE; demo path runs the real CNN on
    synthetic burn-signature patches so a probability is still produced offline.
    """
    from wildfire.models import detector as det_mod

    out: dict[str, tuple] = {}
    if not det_mod.available(cfg):
        return out, "missing"

    hottest = active.sort_values("t21", ascending=False).head(max_confirm)
    points = [(r["cell_id"], float(r["lat"]), float(r["lon"])) for _, r in hottest.iterrows()]
    if not points:
        return out, "none"

    det = det_mod.load_detector(cfg)

    if live:
        from wildfire.ingest.patches import pull_live_s2_patches

        patches, thumbs = pull_live_s2_patches(cfg, points, end_date=end_date, window_days=20)
        order = [cid for cid, _, _ in points if cid in patches]
        if order:
            X = np.stack([patches[cid] for cid in order], axis=0)
            probs = det_mod.predict_patches(det, X)
            for cid, p in zip(order, probs):
                out[cid] = (float(p), thumbs.get(cid))
        return out, "sentinel2"

    # Demo: synthetic burn-signature patches through the real CNN.
    from wildfire.ingest.patches import _make_patch_batch, channel_names, n_channels

    rng = np.random.default_rng(cfg.seed)
    size = int(cfg.get("patches.size_px", 64))
    c = n_channels(cfg)
    X = np.concatenate([_make_patch_batch(1, size, c, fire=True, rng=rng) for _ in points], axis=0)
    probs = det_mod.predict_patches(det, X)
    for (cid, _, _), p in zip(points, probs):
        out[cid] = (float(p), None)
    return out, "synthetic-patch"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2, help="FIRMS look-back window in days")
    ap.add_argument("--max-confirm", type=int, default=30, help="hottest cells to CNN-confirm")
    ap.add_argument("--synthetic", action="store_true", help="force the offline demo path")
    args = ap.parse_args()

    init_console()
    cfg = load_config()
    site = REPO_ROOT / "site"
    surf = gpd.read_parquet(cfg.path_for("data_processed") / "risk_surface.parquet")
    cell_idx = _load_cell_index()
    map_key = os.environ.get("FIRMS_MAP_KEY") or None
    end_date = (dt.datetime.utcnow().date()).isoformat()

    mode, active, cnn_method = "demo", None, "synthetic-patch"
    if not args.synthetic:
        try:
            from wildfire.features.grid import build_grid
            from wildfire.ingest.firms import firms_active_cells

            grid = build_grid(cfg)
            active = firms_active_cells(cfg, grid, days=args.days, map_key=map_key)
            mode = "live"
            print(f"   FIRMS live: {len(active)} active cell(s) in the last {args.days} day(s)")
        except Exception as exc:
            print(f"   ! live FIRMS/EE unavailable ({exc}); using synthetic demo")
            active = None

    if active is None:
        from wildfire.ingest.firms import synthetic_firms

        active = synthetic_firms(cfg, surf)
        mode = "demo"
        print(f"   demo FIRMS: {len(active)} seeded active cell(s)")

    # CNN confirmation on the hottest cells.
    try:
        confirm, cnn_method = _cnn_confirm(
            cfg, active, end_date=end_date, max_confirm=args.max_confirm, live=(mode == "live")
        )
    except Exception as exc:
        print(f"   ! CNN confirmation skipped ({exc})")
        confirm, cnn_method = {}, "skipped"

    # Assemble per-detection records, richest first.
    dets = []
    for _, r in active.sort_values("t21", ascending=False).iterrows():
        cid = r["cell_id"]
        info = cell_idx.get(cid, {})
        prob, thumb = confirm.get(cid, (None, None))
        dets.append({
            "id": cid,
            "lat": round(float(r["lat"]), 3), "lon": round(float(r["lon"]), 3),
            "near_city": info.get("near_city"), "near_km": info.get("near_km"),
            "eco": info.get("eco"), "risk": info.get("risk"),
            "t21": None if pd.isna(r["t21"]) else round(float(r["t21"]), 1),
            "confidence": None if pd.isna(r["confidence"]) else round(float(r["confidence"]), 0),
            "frp": None if pd.isna(r.get("frp")) else round(float(r["frp"]), 1),
            "n_det": int(r["n_det"]) if pd.notna(r.get("n_det")) else None,
            "acq_date": r.get("acq_date"),
            "cnn_prob": None if prob is None else round(prob, 3),
            "thumb": thumb,
        })

    n_conf = sum(1 for d in dets if d["cnn_prob"] is not None and d["cnn_prob"] >= 0.5)
    source = active["source"].iloc[0] if len(active) else ("synthetic-demo" if mode == "demo" else "none")
    payload = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "mode": mode,                 # "live" or "demo"
        "source": source,             # firms backend used
        "cnn_method": cnn_method,     # how the CNN prob was produced
        "window_days": args.days,
        "n_active": len(dets),
        "n_confirmed": int(n_conf),
        "detections": dets,
    }

    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "live_scan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (site / "data" / "live_scan.js").write_text(
        "window.WF_LIVE=" + json.dumps(payload, separators=(",", ":")) + ";", encoding="utf-8")
    print(f"\n✅ Live scan written ({mode}): {len(dets)} active cell(s), "
          f"{n_conf} CNN-confirmed burn(s), cnn={cnn_method}")
    print("   -> site/data/live_scan.js  (commit + push to update the live site)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
