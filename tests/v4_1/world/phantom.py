"""Small deterministic synthetic phantoms for the tumor_mri tests. No real data."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from antelligence.worlds.tumor_mri import Crop, WorldConfig


def make_phantom(n: int = 36, seed: int = 7, outside_truth: int = 40, flair: bool = False) -> Crop:
    """Ellipsoid brain filling the crop (touches the faces), two spherical tumours, imperfect perception."""
    g = np.indices((n, n, n)).astype(np.float64)
    c = (n - 1) / 2.0
    brain = (((g[0] - c) / (c + 0.5)) ** 2 + ((g[1] - c) / (c + 0.5)) ** 2 + ((g[2] - c) / (c + 0.5)) ** 2) <= 1.0
    truth = np.zeros((n, n, n), dtype=bool)
    for centre, rad in (((c + 1, c - 1, c), 5.0), ((c - 7, c + 6, c - 3), 3.0)):
        truth |= ((g[0] - centre[0]) ** 2 + (g[1] - centre[1]) ** 2 + (g[2] - centre[2]) ** 2) <= rad ** 2
    truth &= brain
    rng = np.random.default_rng(seed)
    smooth = ndimage.gaussian_filter(truth.astype(np.float64), 1.2)
    p = np.clip(smooth * 1.6 + rng.normal(0, 0.02, truth.shape), 0.0, 1.0) * brain
    unc = np.clip(4.0 * p * (1.0 - p), 0.0, 1.0)
    arrays = dict(brain=brain, p_tumour=p, unc=unc, truth=truth)
    if flair:
        arrays["flair"] = rng.random(truth.shape).astype(np.float32)
        arrays["t1c"] = rng.random(truth.shape).astype(np.float32)
    return Crop.from_arrays(arrays, outside_crop_truth_voxels=outside_truth, case_id="phantom")


def phantom_config(**kw) -> WorldConfig:
    base = dict(total_drug=400.0, dose_threshold=6.0, explore_radius_vox=8, entry_shell_mm=4.0)
    base.update(kw)
    return WorldConfig(**base)
