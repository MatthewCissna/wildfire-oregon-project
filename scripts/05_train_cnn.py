"""Stage 5 — train the fire-detection CNN on the patch dataset.

    uv run python scripts/05_train_cnn.py            # full
    uv run python scripts/05_train_cnn.py --quick    # 2 epochs, small

Requires the cnn extra:  uv sync --extra cnn
"""

from __future__ import annotations

import argparse
import json

from wildfire.config import load_config
from wildfire.utils import init_console


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    init_console()
    cfg = load_config()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("❌ PyTorch is not installed. Run:  uv sync --extra cnn")
        return 1

    from wildfire.ingest.patches import load_patches
    from wildfire.models.cnn import save_cnn, train_cnn

    data = load_patches(cfg)
    result = train_cnn(cfg, data, quick=args.quick)
    path = save_cnn(cfg, result)

    metrics_path = cfg.path_for("metrics") / "cnn_metrics.json"
    metrics_path.write_text(json.dumps(
        {"backbone": result.backbone, "test_metrics": result.test_metrics, "history": result.history},
        indent=2, default=float,
    ))

    m = result.test_metrics
    print("\n✅ CNN trained (held-out spatial blocks)")
    print(f"   backbone   : {result.backbone}")
    print(f"   PR-AUC     : {m['pr_auc']:.3f}  (lift {m['pr_auc_lift']:.1f}x)")
    print(f"   ROC-AUC    : {m['roc_auc']:.3f}")
    print(f"   recall@20% : {m['recall_at_p20']:.3f}")
    print(f"   Brier      : {m['brier']:.4f}")
    print(f"   saved      : {path}")
    print(f"   metrics    : {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
