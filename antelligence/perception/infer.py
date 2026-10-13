"""Bundle-style inference: nonzero z-score, sliding window, logits -> sigmoid, flip TTA.

Output channel order of the bundle is (TC, WT, ET) -- see Lambdad in inference.json.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .msd import BUNDLE_CHANNELS

ROI_SIZE = (240, 240, 160)
OVERLAP = 0.5
BUNDLE_OUT = ("TC", "WT", "ET")  # channel order of the bundle's network output
ALL_FLIPS: tuple[tuple[int, ...], ...] = tuple(
    c for r in range(4) for c in itertools.combinations((0, 1, 2), r)
)  # 8 subsets of the 3 spatial axes


def zero_channel(image: np.ndarray, names: list[str], channel: str = "FLAIR") -> np.ndarray:
    """missing_flair variant: return a copy with ``channel`` zeroed (``names`` = layout of image)."""
    out = image.copy()
    out[names.index(channel)] = 0
    return out


def normalise_nonzero(x: np.ndarray) -> np.ndarray:
    """MONAI NormalizeIntensity(nonzero=True, channel_wise=True): z-score over nonzero voxels only.

    A channel with no nonzero voxels (e.g. a zeroed FLAIR) stays all zero.
    """
    out = x.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        ch = out[c]
        nz = ch != 0
        if not nz.any():
            continue
        v = ch[nz]
        mean, std = float(v.mean()), float(v.std())
        if std == 0.0:
            std = 1.0
        ch[nz] = (v - mean) / std
    return out


def prepare_input(image_msd: np.ndarray, reorder: list[int], zero_flair_names: list[str] | None = None) -> np.ndarray:
    """MSD (C,X,Y,Z) -> bundle channel order, optionally zeroing FLAIR first, then normalised."""
    x = image_msd
    if zero_flair_names is not None:
        x = zero_channel(x, zero_flair_names, "FLAIR")
    x = x[list(reorder)]
    return normalise_nonzero(x)


@torch.no_grad()
def predict_logits(model, x: torch.Tensor, device, roi_size=ROI_SIZE, overlap=OVERLAP) -> torch.Tensor:
    """x: (1,C,X,Y,Z) on CPU. Sliding-window logits (1,3,X,Y,Z) on CPU."""
    from monai.inferers import sliding_window_inference

    return sliding_window_inference(
        x, roi_size=roi_size, sw_batch_size=1, predictor=model, overlap=overlap,
        sw_device=device, device=torch.device("cpu"),
    )


@torch.no_grad()
def predict_tta(model, x: np.ndarray | torch.Tensor, device="cpu", roi_size=ROI_SIZE, overlap=OVERLAP,
                flips=ALL_FLIPS) -> tuple[np.ndarray, np.ndarray]:
    """Flip TTA. x: (C,X,Y,Z) normalised, bundle order. Returns (mean_prob, std_prob), each (3,X,Y,Z).

    Each pass: flip input -> logits -> sigmoid -> unflip. mean = probability, std = uncertainty.
    Channel order is the bundle's (TC, WT, ET). Running sums keep memory at ~2 volumes.
    """
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32).unsqueeze(0)
    s1 = s2 = None
    for axes in flips:
        dims = tuple(a + 2 for a in axes)
        xin = torch.flip(xt, dims) if dims else xt
        logits = predict_logits(model, xin, device, roi_size, overlap)
        p = torch.sigmoid(logits)
        if dims:
            p = torch.flip(p, dims)
        p = p[0].double()
        s1 = p if s1 is None else s1 + p
        s2 = p * p if s2 is None else s2 + p * p
    n = len(flips)
    mean = s1 / n
    var = torch.clamp(s2 / n - mean * mean, min=0.0)
    return mean.float().numpy(), var.sqrt().float().numpy()


def to_msd_truth_order(p_bundle: np.ndarray) -> np.ndarray:
    """Bundle output (TC, WT, ET) -> (WT, TC, ET), the order used everywhere downstream."""
    i = {n: k for k, n in enumerate(BUNDLE_OUT)}
    return p_bundle[[i["WT"], i["TC"], i["ET"]]]


def select_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["BUNDLE_CHANNELS", "predict_tta", "prepare_input", "select_device"]
