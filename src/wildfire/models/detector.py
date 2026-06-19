"""Load the trained burn-scar detection CNN and run inference on patches.

Thin wrapper around the weights saved by :func:`wildfire.models.cnn.save_cnn`
(``outputs/models/cnn/fire_detector.pt``). Used by the live fire scan to confirm
FIRMS active-fire detections against the satellite imagery, and reusable anywhere
single-shot inference is needed. Torch/timm are imported lazily so importing this
module is cheap and the ``cnn`` extra is only required when you actually predict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class Detector:
    model: object
    channels: list
    mean: np.ndarray
    std: np.ndarray
    device: str
    test_metrics: dict


def detector_path(cfg: Config) -> "object":
    return cfg.path_for("models") / "cnn" / "fire_detector.pt"


def available(cfg: Config | None = None) -> bool:
    """True if the trained detector weights are on disk."""
    cfg = cfg or load_config()
    return detector_path(cfg).exists()


def load_detector(cfg: Config | None = None, device: str | None = None) -> Detector:
    """Rebuild the CNN architecture and load the trained weights for inference."""
    import torch

    from wildfire.models.cnn import resolve_device

    cfg = cfg or load_config()
    path = detector_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained detector at {path}. Train it with scripts/05_train_cnn.py."
        )
    import timm

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    channels = list(ckpt["channels"])
    dev = device or resolve_device(cfg.get("cnn.device", "auto"))

    # Build the architecture from the checkpoint's own backbone with pretrained=False —
    # we immediately load the trained weights, so there's no need to fetch ImageNet
    # weights from the hub (which would make CI depend on network access).
    bb = ckpt.get("backbone", cfg.get("cnn.backbone", "efficientnet_b0"))
    model = timm.create_model(bb, pretrained=False, in_chans=len(channels), num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    return Detector(model=model, channels=channels, mean=mean, std=std,
                    device=dev, test_metrics=ckpt.get("test_metrics", {}))


def predict_patches(det: Detector, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """Return per-patch burn-scar probabilities for ``X`` shaped (N, C, H, W)."""
    import torch

    if X.ndim != 4:
        raise ValueError(f"expected (N,C,H,W) patches, got shape {X.shape}")
    mean = det.mean.reshape(-1, 1, 1)
    std = det.std.reshape(-1, 1, 1)
    out = []
    amp_device = "cuda" if det.device == "cuda" else "cpu"
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = (X[i:i + batch].astype(np.float32) - mean) / std
            t = torch.from_numpy(np.ascontiguousarray(xb)).float().to(det.device)
            with torch.amp.autocast(amp_device, enabled=(det.device == "cuda")):
                prob = torch.softmax(det.model(t), dim=1)[:, 1]
            out.append(prob.float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])
