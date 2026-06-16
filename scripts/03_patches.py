"""Stage 3 — build the image-patch dataset for the detection CNN.

    uv run python scripts/03_patches.py --synthetic         # offline
    uv run python scripts/03_patches.py --synthetic --quick # small
    uv run python scripts/03_patches.py --gee --year 2021   # live S2 export (async)
"""

from __future__ import annotations

import argparse

from wildfire.config import load_config
from wildfire.ingest import patches
from wildfire.utils import init_console


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--gee", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()

    init_console()
    cfg = load_config()

    if args.gee and cfg.ee_project:
        tasks = patches.export_s2_patches_ee(cfg, year=args.year)
        print(f"✅ Started {len(tasks)} GEE export task(s) for {args.year}.")
        print("   Monitor in the Earth Engine Tasks tab; then assemble the TFRecords.")
        return 0

    data = patches.synthetic_patches(cfg, quick=args.quick)
    path = patches.save_patches(cfg, data)
    print("\n✅ Patches built")
    print(f"   patches : {len(data['y']):,}  ({int(data['y'].sum())} fire / {int((data['y']==0).sum())} no-fire)")
    print(f"   shape   : {data['X'].shape}  channels={data['channels']}")
    print(f"   saved   : {path}")
    print("\n   Next:  uv run python scripts/05_train_cnn.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
