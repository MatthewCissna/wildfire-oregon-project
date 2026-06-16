"""Stage 1 — ingest data and write the canonical tables to data/interim.

    uv run python scripts/01_ingest.py --synthetic        # offline fallback
    uv run python scripts/01_ingest.py --synthetic --quick # tiny, fast
    uv run python scripts/01_ingest.py --gee              # live Earth Engine + NIFC + OSM
"""

from __future__ import annotations

import argparse

from wildfire.config import load_config
from wildfire.ingest.datasets import materialize
from wildfire.utils import init_console


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest wildfire data.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="use the offline synthetic generator")
    src.add_argument("--gee", action="store_true", help="pull live Earth Engine + NIFC + OSM")
    ap.add_argument("--quick", action="store_true", help="small sample for a fast smoke run")
    args = ap.parse_args()

    init_console()
    cfg = load_config()

    # Default to synthetic unless --gee is explicitly requested (and possible).
    use_synthetic = not args.gee
    if args.gee and not cfg.ee_project:
        print("⚠️  --gee requested but no Earth Engine project configured.")
        print("   Falling back to synthetic. See docs/earth_engine_setup.md.")
        use_synthetic = True

    out = materialize(cfg, synthetic=use_synthetic, quick=args.quick)
    m = out["manifest"]
    print("\n✅ Ingest complete")
    print(f"   source        : {m['source']}")
    print(f"   cells         : {m['n_cells']:,}")
    print(f"   panel rows    : {m['n_panel_rows']:,}")
    print(f"   fire events   : {m['n_events']:,}")
    print(f"   positive rate : {m['positive_rate']*100:.2f}%")
    print("\n   Next:  uv run python scripts/02_features.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
