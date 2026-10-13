"""Crop to the tumour neighbourhood: bbox(p >= 0.3) + 30 mm margin, clipped to the brain, at 2 mm."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

P_THRESHOLD = 0.3
MARGIN_MM = 30.0


def resample(arr: np.ndarray, spacing_mm: tuple[float, float, float], target_mm: float = 2.0,
             order: int = 1) -> np.ndarray:
    """Resample a 3D array to ``target_mm`` isotropic. Area-style average for order 1 (grid_mode)."""
    zoom = tuple(s / target_mm for s in spacing_mm)
    if all(abs(z - 1.0) < 1e-9 for z in zoom):
        return arr.copy()
    src = arr.astype(np.float32) if arr.dtype == bool else arr
    out = ndimage.zoom(src, zoom, order=order, mode="nearest", grid_mode=True)
    return out.astype(bool) if arr.dtype == bool else out


def compute_crop(p: np.ndarray, brain: np.ndarray, spacing_mm: float, threshold: float = P_THRESHOLD,
                 margin_mm: float = MARGIN_MM) -> tuple[slice, slice, slice] | None:
    """Box around ``p >= threshold`` grown by margin_mm, clipped to the brain-mask bounding box.

    Returns None when no voxel reaches the threshold.
    """
    hit = p >= threshold
    if not hit.any():
        return None
    margin = int(np.ceil(margin_mm / spacing_mm))
    brain_idx = np.nonzero(brain)
    slices = []
    for ax in range(3):
        h = np.nonzero(hit.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        lo, hi = int(h.min()) - margin, int(h.max()) + margin + 1
        if brain_idx[0].size:
            lo = max(lo, int(brain_idx[ax].min()))
            hi = min(hi, int(brain_idx[ax].max()) + 1)
        lo, hi = max(lo, 0), min(hi, p.shape[ax])
        slices.append(slice(lo, hi))
    return tuple(slices)  # type: ignore[return-value]


def truth_share_outside(truth_wt: np.ndarray, box: tuple[slice, slice, slice] | None) -> float:
    """Fraction of truth-WT voxels that fall outside the crop (1.0 if there is no crop, 0.0 if no truth)."""
    total = int(truth_wt.sum())
    if total == 0:
        return 0.0
    if box is None:
        return 1.0
    return float(1.0 - truth_wt[box].sum() / total)


def truth_voxels_outside(truth_wt: np.ndarray, box: tuple[slice, slice, slice] | None) -> int:
    """Count of truth-WT voxels (2 mm grid) that fall outside the crop (all of them if there is no crop)."""
    total = int(truth_wt.sum())
    if box is None:
        return total
    return total - int(truth_wt[box].sum())
