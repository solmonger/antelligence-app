"""Where an ant goes next inside a zone: the nearest unprobed spot (v4.3 A2r1; reused by the B2 probe rule).

Pure geometry on the scan map ``p_tumour`` (perception, which every ant sees; never truth). Two steps:

* :func:`zone_candidates` lists the zone's voxels with scan p >= ``p_min``. A zone is the equal-volume sphere of a perceived
  region plus 6 mm (``ScanInfo.zone_radius_mm``). Only a bounding box around the zone is looked at, never the whole crop.
  The list depends on the scan alone, so ``ScanInfo`` caches it once for all ants.
* :func:`nearest_new_spot` picks, from that list, the voxel nearest to the ant (Euclidean, in mm) that is at least
  ``min_gap_mm`` from every known spot. Ties go to the lexicographically smallest voxel.

Distances are compared in squared voxel units (integers), so ties are exact; the gap test carries the usual 1e-9 slack.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

Vox = Tuple[int, int, int]
CHUNK = 256            # the nearest candidates tested first against the known spots (grown x4 until one is free)


def zone_candidates(p: np.ndarray, centre_vox: Sequence[float], radius_mm: float, voxel_mm: float,
                    p_min: float = 0.5) -> np.ndarray:
    """(n, 3) int array of the voxels inside the zone sphere with ``p >= p_min``, in C order. Bounding box only."""
    r_vox = radius_mm / voxel_mm
    lo = [max(0, int(np.floor(c - r_vox)) - 1) for c in centre_vox]
    hi = [min(s, int(np.ceil(c + r_vox)) + 2) for c, s in zip(centre_vox, p.shape)]
    if any(h <= l for l, h in zip(lo, hi)):
        return np.zeros((0, 3), dtype=np.int64)
    sub = p[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    idx = np.argwhere(sub >= p_min)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    idx = idx + np.asarray(lo, dtype=np.int64)
    d2 = ((idx - np.asarray(centre_vox, dtype=np.float64)) ** 2).sum(axis=1)
    return idx[d2 * voxel_mm * voxel_mm <= radius_mm * radius_mm * (1 + 1e-9) ** 2]


def nearest_new_spot(cands: np.ndarray, from_pos: Sequence[float], known: Sequence[Sequence[float]], voxel_mm: float,
                     min_gap_mm: float) -> Optional[Vox]:
    """The candidate nearest to ``from_pos`` that is >= ``min_gap_mm`` from every ``known`` spot; ``None`` when none is left.

    Ties in the distance to the ant go to the lexicographically smallest voxel. Candidates are tested nearest first, in chunks,
    so the cost is small when the nearest free spot is close."""
    if len(cands) == 0:
        return None
    pos = np.asarray(from_pos, dtype=np.float64)
    d2 = ((cands - pos) ** 2).sum(axis=1)
    gap2 = (min_gap_mm / voxel_mm) ** 2 * (1 - 1e-9) ** 2
    kn = np.asarray([tuple(k) for k in known], dtype=np.float64).reshape(-1, 3)
    n, k = len(cands), CHUNK
    while True:
        # every candidate no farther than the k-th nearest (ties included), so a hit below is the nearest overall
        cut = np.inf if k >= n else np.partition(d2, k - 1)[k - 1]
        idx = np.flatnonzero(d2 <= cut)
        pts = cands[idx]
        if len(kn):
            far = (((pts[:, None, :] - kn[None, :, :]) ** 2).sum(axis=2) >= gap2).all(axis=1)
        else:
            far = np.ones(len(idx), dtype=bool)
        if far.any():
            idx, pts = idx[far], pts[far]
            best = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0], d2[idx]))[0]       # primary d2, then x, y, z
            v = pts[best]
            return (int(v[0]), int(v[1]), int(v[2]))
        if k >= n:
            return None
        k *= 4
