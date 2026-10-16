"""v4.31 A.2: geometric off-target measures. MEASURE ONLY: nothing here enters the score and no penalty is added.

Why geometry and not milligrams. A literature pass found no maximum tolerated concentration, exposure or dose for normal brain parenchyma, for any agent in any species: the
absence is structural, because infusion trials escalate the concentration going in and never measure it in tissue. So 'mg outside the tumour' has nothing to score against. What does exist
is the phase III PRECISE catheter protocol (Mueller et al. 2011, J Neurooncol 101:267-277): a release point at least 25 mm below the surface, a deep sulcus or the cavity wall, at least 5 mm
from the pia, and at least 10 mm from the cavity wall or the ependyma. Those are distances on a segmented volume, so they are scored here.

What the pipeline has, and what it does not.
* The brain mask is real (nonzero in any channel). Its boundary is taken as the pial surface. It does NOT resolve deep sulci, so the 'deep sulcus' exemption cannot be credited, and rule 1 is the
  conservative geometric version (distance to the mask surface). Crop faces are not a surface: the mask is extended by edge replication so a crop face never counts as pia.
* **There are NO ventricle labels in the pipeline.** MSD Task01 labels are background, edema, non-enhancing and enhancing tumour; the brain mask includes CSF. There is no resection cavity either
  (the scans are pre-operative). The ependymal measures are therefore computed from an intensity PROXY (dark on FLAIR and T1, interior, outside the tumour, in large components). It is
  unvalidated and scanner dependent (volumes of 4 to 65 mL across the cases probed), so it is reported as a sensitivity, never as the headline.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import ndimage

DEPTH_MM, PIA_MM, EPENDYMA_MM = 25.0, 5.0, 10.0           # PRECISE: >= 25 mm below the surface, >= 5 mm from the pia, >= 10 mm from the ependyma / cavity wall
PIA_SHARE_MM, EPENDYMA_SHARE_MM = 5.0, 10.0               # exposure landing within these of a pial / ependymal surface


def surface_distance_mm(brain: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Distance (mm) from every brain voxel to the nearest non-brain voxel (the pial surface); 0 outside the brain. Crop faces continue the mask (edge replication)."""
    padded = np.pad(brain, 1, mode="edge")
    return ndimage.distance_transform_edt(padded, sampling=spacing_mm)[1:-1, 1:-1, 1:-1]


def ventricle_proxy(flair: np.ndarray, t1: np.ndarray, brain: np.ndarray, wt: np.ndarray, spacing_mm: float, *, interior_mm: float = 6.0, min_voxels: int = 500) -> np.ndarray:
    """UNVALIDATED intensity proxy for ventricular CSF (see the module docstring). Dark on FLAIR (< 0.4 x brain median) and on T1 (< 0.6 x brain median), at least ``interior_mm`` from the
    brain surface, outside the tumour, in connected components of at least ``min_voxels`` voxels."""
    if not brain.any():
        return np.zeros(brain.shape, dtype=bool)
    edt = surface_distance_mm(brain, spacing_mm)
    cand = brain & (edt >= interior_mm) & (flair < 0.4 * float(np.median(flair[brain]))) & (t1 < 0.6 * float(np.median(t1[brain]))) & ~wt
    lab, n = ndimage.label(cand)
    if n == 0:
        return np.zeros(brain.shape, dtype=bool)
    sizes = ndimage.sum(cand, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
    return np.isin(lab, keep)


def distance_to_mask_mm(mask: np.ndarray, spacing_mm: float) -> Optional[np.ndarray]:
    """Distance (mm) from every voxel to the nearest voxel of ``mask``; None if the mask is empty."""
    if not mask.any():
        return None
    return ndimage.distance_transform_edt(~mask, sampling=spacing_mm)


def _at(field: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Trilinear sample of ``field`` at release lattice corners (a corner u sits between voxels u-1 and u, i.e. at voxel coordinate u - 0.5)."""
    coords = (np.asarray(lattice, dtype=np.float64) - 0.5).T
    return ndimage.map_coordinates(field, coords, order=1, mode="nearest")


def precise_rules(lattice: np.ndarray, masses: Sequence[float], surf_mm: np.ndarray, vent_mm: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """Share of delivered drug (by mass) whose release point breaks each PRECISE-style rule. ``ependyma`` is None when there is no (proxy) ventricle in the crop."""
    m = np.asarray(masses, dtype=np.float64)
    tot = float(m.sum())
    if tot <= 0.0 or len(m) == 0:
        return {"depth_lt_25mm": 0.0, "pia_lt_5mm": 0.0, "ependyma_lt_10mm_proxy": None if vent_mm is None else 0.0}
    d_s = _at(surf_mm, lattice)
    out: Dict[str, Optional[float]] = {"depth_lt_25mm": float(m[d_s < DEPTH_MM].sum() / tot), "pia_lt_5mm": float(m[d_s < PIA_MM].sum() / tot)}
    out["ependyma_lt_10mm_proxy"] = None if vent_mm is None else float(m[_at(vent_mm, lattice) < EPENDYMA_MM].sum() / tot)
    return out


def exposure_shares(dose: np.ndarray, brain: np.ndarray, surf_mm: np.ndarray, vent_mm: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """Share of the total exposure that lands within 5 mm of the pial surface (every non-brain voxel counts as at the surface) and within 10 mm of the ependymal (proxy) surface."""
    tot = float(dose.sum())
    if tot <= 0.0:
        return {"within_5mm_of_pia": 0.0, "within_10mm_of_ependyma_proxy": None if vent_mm is None else 0.0}
    near_pia = (~brain) | (surf_mm < PIA_SHARE_MM)
    out: Dict[str, Optional[float]] = {"within_5mm_of_pia": float(dose[near_pia].sum() / tot)}
    out["within_10mm_of_ependyma_proxy"] = None if vent_mm is None else float(dose[vent_mm < EPENDYMA_SHARE_MM].sum() / tot)
    return out
