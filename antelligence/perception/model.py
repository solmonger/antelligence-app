"""SegResNet of the MONAI brats_mri_segmentation bundle (v0.5.4, pinned HF revision).

Only ``models/model.pt`` is ever loaded, with ``torch.load(weights_only=True)``. The
TorchScript artefact ``models/model.ts`` is flagged by the HF scanner and is refused here.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import torch

from . import MODEL_ID, MODEL_REVISION

HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}"
WEIGHTS_FILE = "models/model.pt"
CONFIG_FILE = "configs/inference.json"


class UnsafeWeightsError(ValueError):
    """Refused to load an artefact that is not a plain weights file."""


def hyperparameters_from_config(config: dict) -> dict:
    """network_def hyperparameters from configs/inference.json (minus the _target_ key)."""
    nd = dict(config["network_def"])
    if nd.pop("_target_") != "SegResNet":
        raise ValueError("bundle network is not SegResNet")
    return nd


def build_segresnet(hparams: dict) -> torch.nn.Module:
    from monai.networks.nets import SegResNet

    return SegResNet(**hparams)


def load_weights(model: torch.nn.Module, path: str | Path, map_location="cpu") -> torch.nn.Module:
    """Load a state dict from ``model.pt`` with weights_only=True. Refuses ``.ts`` files."""
    path = Path(path)
    if path.suffix.lower() == ".ts" or path.name.lower() == "model.ts":
        raise UnsafeWeightsError(f"refusing TorchScript artefact {path.name}; load models/model.pt only")
    state = torch.load(str(path), map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]  # MONAI CheckpointLoader layout {"model": state_dict}
    model.load_state_dict(state)
    model.eval()
    return model


def _download(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:  # noqa: S310 - pinned https URL
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            h.update(chunk)
    tmp.replace(dest)
    return h.hexdigest()


def fetch_bundle(models_dir: str | Path) -> tuple[Path, Path]:
    """Download inference.json and model.pt (public repo, pinned revision) into ``models_dir``."""
    models_dir = Path(models_dir)
    cfg = models_dir / "inference.json"
    pt = models_dir / "model.pt"
    if not cfg.exists():
        _download(f"{HF_BASE}/{CONFIG_FILE}", cfg)
    if not pt.exists():
        digest = _download(f"{HF_BASE}/{WEIGHTS_FILE}", pt)
        (models_dir / "model.pt.sha256").write_text(digest + "\n")
    return cfg, pt


def load_bundle_model(models_dir: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    cfg_path, pt_path = fetch_bundle(models_dir)
    hp = hyperparameters_from_config(json.loads(Path(cfg_path).read_text()))
    model = load_weights(build_segresnet(hp), pt_path)
    return model.to(device).eval()
