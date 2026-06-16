"""Fire-detection CNN — transfer learning on multi-band satellite patches.

* **Backbone:** a pretrained EfficientNet/ResNet from ``timm``, with the first conv
  adapted to the patch's channel count (bands + indices) via ``in_chans``.
* **Spatial-block split:** train/val/test are split by ``block_id`` so no block
  appears in two sets — the same leakage discipline as the tabular models.
* **Augmentation:** flips and 90° rotations (label-preserving for nadir imagery).
* **Honest metrics:** PR-AUC / recall@threshold on the held-out spatial test blocks.

Torch is imported lazily so the rest of the repo doesn't require the ``cnn`` extra.
Install it with ``uv sync --extra cnn``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def spatial_block_split(meta, seed: int = 42, fracs=(0.7, 0.15, 0.15)):
    """Split indices by ``block_id`` into train/val/test (no block in two sets)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    blocks = np.array(sorted(meta["block_id"].unique()))
    rng.shuffle(blocks)
    n = len(blocks)
    n_tr = int(n * fracs[0])
    n_va = int(n * (fracs[0] + fracs[1]))
    sets = {"train": set(blocks[:n_tr]), "val": set(blocks[n_tr:n_va]), "test": set(blocks[n_va:])}
    idx = {k: meta.index[meta["block_id"].isin(v)].to_numpy() for k, v in sets.items()}
    return idx


def _make_dataset_class():
    """Build the Dataset class lazily (after torch import)."""
    import torch
    from torch.utils.data import Dataset

    class PatchDataset(Dataset):
        def __init__(self, X, y, mean, std, augment=False):
            self.X = X
            self.y = y
            self.mean = mean.reshape(-1, 1, 1)
            self.std = std.reshape(-1, 1, 1)
            self.augment = augment

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            x = (self.X[i] - self.mean) / self.std
            if self.augment:
                if np.random.rand() < 0.5:
                    x = x[:, :, ::-1]
                if np.random.rand() < 0.5:
                    x = x[:, ::-1, :]
                k = np.random.randint(0, 4)
                if k:
                    x = np.rot90(x, k, axes=(1, 2))
            return torch.from_numpy(np.ascontiguousarray(x)).float(), int(self.y[i])

    return PatchDataset


def build_model(cfg: Config, in_chans: int):
    import timm

    return timm.create_model(
        cfg.get("cnn.backbone", "efficientnet_b0"),
        pretrained=bool(cfg.get("cnn.pretrained", True)),
        in_chans=in_chans,
        num_classes=2,
    )


@dataclass
class CNNResult:
    state_dict: dict
    backbone: str
    channels: list
    mean: np.ndarray
    std: np.ndarray
    history: list = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)


def train_cnn(cfg: Config | None = None, data: dict | None = None, *, quick: bool = False) -> CNNResult:
    """Train the detection CNN on patches; return weights + test metrics."""
    import torch
    from torch.utils.data import DataLoader

    from wildfire.eval.metrics import classification_metrics
    from wildfire.utils import seed_everything

    cfg = cfg or load_config()
    seed_everything(cfg.seed)
    if data is None:
        from wildfire.ingest.patches import load_patches

        data = load_patches(cfg)

    X, y, meta = data["X"], data["y"], data["meta"]
    idx = spatial_block_split(meta, seed=cfg.seed)
    device = resolve_device(cfg.get("cnn.device", "auto"))

    # Per-channel normalization from the TRAIN split only.
    tr = idx["train"]
    mean = X[tr].mean(axis=(0, 2, 3))
    std = X[tr].std(axis=(0, 2, 3)) + 1e-6

    PatchDataset = _make_dataset_class()
    bs = int(cfg.get("cnn.batch_size", 64))
    nw = 0 if quick else int(cfg.get("cnn.num_workers", 4))
    loaders = {}
    for split in ("train", "val", "test"):
        ds = PatchDataset(X[idx[split]], y[idx[split]], mean, std, augment=(split == "train"))
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=(split == "train"), num_workers=nw)

    model = build_model(cfg, in_chans=X.shape[1]).to(device)
    # Class weights for imbalance.
    pos = float((y[tr] == 1).mean())
    w = torch.tensor([pos, 1 - pos], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("cnn.lr", 3e-4)),
        weight_decay=float(cfg.get("cnn.weight_decay", 1e-4)),
    )
    epochs = 2 if quick else int(cfg.get("cnn.epochs", 25))
    amp_device = "cuda" if device == "cuda" else "cpu"
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    history = []
    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast(amp_device, enabled=use_amp):
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(yb)
        val_scores, val_y = _infer(model, loaders["val"], device)
        val_m = classification_metrics(val_y, val_scores)
        history.append({"epoch": ep, "train_loss": running / len(idx["train"]), **val_m})
        logger.info("epoch %d  loss=%.4f  val PR-AUC=%.3f", ep, history[-1]["train_loss"], val_m["pr_auc"])
        if val_m["pr_auc"] > best_val:
            best_val = val_m["pr_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_scores, test_y = _infer(model, loaders["test"], device)
    test_m = classification_metrics(test_y, test_scores)
    logger.info("TEST (held-out blocks)  PR-AUC=%.3f  recall@20%%=%.3f", test_m["pr_auc"], test_m["recall_at_p20"])

    return CNNResult(
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        backbone=cfg.get("cnn.backbone", "efficientnet_b0"),
        channels=data["channels"],
        mean=mean, std=std, history=history, test_metrics=test_m,
    )


def _infer(model, loader, device):
    import torch

    model.eval()
    amp_device = "cuda" if device == "cuda" else "cpu"
    scores, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            with torch.amp.autocast(amp_device, enabled=(device == "cuda")):
                prob = torch.softmax(model(xb), dim=1)[:, 1]
            scores.append(prob.float().cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(scores), np.concatenate(ys)


def save_cnn(cfg: Config, result: CNNResult) -> str:
    import torch

    out_dir = cfg.path_for("models") / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fire_detector.pt"
    torch.save(
        {
            "state_dict": result.state_dict,
            "backbone": result.backbone,
            "channels": result.channels,
            "mean": result.mean,
            "std": result.std,
            "test_metrics": result.test_metrics,
            "history": result.history,
        },
        path,
    )
    return str(path)
