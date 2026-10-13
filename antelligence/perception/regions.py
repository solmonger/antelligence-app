"""Regions of interest from a calibrated tumour map, emitted as INTERFACES `knowledge` payloads.

Every number in a payload is an integer in fixed-point units (_ppm, _dmm, _mm3). Coordinates
are voxel indices of the grid the map lives on (the stored 2 mm crop) times spacing, in 0.1 mm.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import MODEL_PROV

P_THRESHOLD = 0.3
MIN_VOLUME_MM3 = 100
SPLIT_ABOVE_MM3 = 20_000
TARGET_SUPERVOXEL_MM3 = 5_000
KMEANS_SEED = 41
CLS_THRESHOLD = 0.5

PAYLOAD_KEYS = ("region_id", "centroid_dmm", "cls", "p_tumour_ppm", "volume_mm3", "unc_ppm", "source", "prov")


def _ppm(x: float) -> int:
    return int(min(1_000_000, max(0, round(float(x) * 1_000_000))))


def kmeans_coords(coords: np.ndarray, k: int, seed: int = KMEANS_SEED, iters: int = 50) -> np.ndarray:
    """Deterministic k-means (k-means++ init, fixed seed) on voxel coordinates. Returns labels."""
    n = coords.shape[0]
    k = max(1, min(k, n))
    rng = np.random.RandomState(seed)
    x = coords.astype(np.float64)
    centers = [x[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        tot = d2.sum()
        probs = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
        centers.append(x[rng.choice(n, p=probs)])
    c = np.asarray(centers)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(0)
    return labels


def _classify(sel: tuple, p_tc: np.ndarray | None, p_et: np.ndarray | None) -> str:
    if p_et is not None and float(p_et[sel].mean()) >= CLS_THRESHOLD:
        return "ET"
    if p_tc is not None and float(p_tc[sel].mean()) >= CLS_THRESHOLD:
        return "TC"
    return "WT"


def extract_regions(p: np.ndarray, spacing_mm: float, unc: np.ndarray | None = None,
                    p_tc: np.ndarray | None = None, p_et: np.ndarray | None = None,
                    threshold: float = P_THRESHOLD, prov: str = MODEL_PROV,
                    min_volume_mm3: int = MIN_VOLUME_MM3, split_above_mm3: int = SPLIT_ABOVE_MM3,
                    target_mm3: int = TARGET_SUPERVOXEL_MM3, seed: int = KMEANS_SEED
                    ) -> tuple[list[dict], np.ndarray]:
    """Connected components (6-connectivity) of p >= threshold -> knowledge payload dicts.

    Components under ``min_volume_mm3`` are dropped; those over ``split_above_mm3`` are split into
    ~``target_mm3`` supervoxels. Returns (payloads, label_map) where label_map holds region_id per
    voxel (0 = background). Order is deterministic: by descending volume, then centroid.
    """
    vox_mm3 = float(spacing_mm) ** 3
    cc, n_cc = ndimage.label(p >= threshold, structure=ndimage.generate_binary_structure(3, 1))
    pieces: list[np.ndarray] = []  # each a tuple-of-index-arrays
    for comp in range(1, n_cc + 1):
        idx = np.nonzero(cc == comp)
        vol = idx[0].size * vox_mm3
        if vol < min_volume_mm3:
            continue
        if vol > split_above_mm3:
            k = max(2, int(round(vol / target_mm3)))
            lab = kmeans_coords(np.stack(idx, axis=1), k, seed)
            for j in range(lab.max() + 1):
                m = lab == j
                if m.any():
                    pieces.append(tuple(a[m] for a in idx))
        else:
            pieces.append(idx)

    def key(idx):
        return (-idx[0].size, *(round(float(a.mean()), 6) for a in idx))

    pieces.sort(key=key)
    payloads: list[dict] = []
    label_map = np.zeros(p.shape, dtype=np.int32)
    for rid, idx in enumerate(pieces, start=1):
        label_map[idx] = rid
        centroid = [int(round(float(a.mean()) * spacing_mm * 10)) for a in idx]
        payloads.append({
            "region_id": rid,
            "centroid_dmm": centroid,
            "cls": _classify(idx, p_tc, p_et),
            "p_tumour_ppm": _ppm(p[idx].mean()),
            "volume_mm3": int(round(idx[0].size * vox_mm3)),
            "unc_ppm": _ppm(unc[idx].mean()) if unc is not None else 0,
            "source": "perception",
            "prov": prov,
        })
    return payloads, label_map
