"""v4.31 Phase A analysis: coverage-versus-threshold curves, viability weights, the +/-1-voxel band, and run evaluation.

Nothing here runs a model, reads a key or touches the network. Pure numpy/scipy on a dose map and the scorer-side truth channels.

Definitions (all fixed in advance, none tuned):

* **Plain coverage** at threshold T: truth voxels with dose >= T, over (truth voxels in the crop + truth voxels outside the crop), the
  v4.1 definition (``Scorer.coverage``).
* **Viability-weighted score**: enhancing tumour (ET) 1.0; the necrotic-plus-non-enhancing part of label 2 (TC and not ET) 0.3 for
  LOCAL delivery (0 for blood-borne delivery; Phase A is local, so 0.3), the share of necrotic-region biopsies that contained viable
  tumour (Eidel 2017); oedema (WT and not TC) 0.35 or 0.5 (two variants, the modelled FLAIR-edge density range). Sum of weights over
  covered voxels over the sum of weights over all truth voxels. Truth outside the crop is counted uncovered with the case's mean weight.
* **+/-1-voxel band**: the truth mask eroded and dilated by one voxel (3x3x3 structuring element, so 1 voxel in every direction
  including diagonals). Erosion keeps voxels that touch the crop face (the tumour continues outside); dilated-in voxels count as
  oedema. Human raters agree on these masks only to Dice 0.74-0.85 (sub-regions), so any effect smaller than the band is not a result.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from antelligence.worlds.tumor_mri.physics import DrugField

# Threshold grid for the units world, umol*h/L: 10^-1 .. 10^7 (eight decades), four points per decade.
THRESHOLDS_UMOL_H_L: Tuple[float, ...] = tuple(float(10.0 ** e) for e in np.arange(-1.0, 7.0001, 0.25))
# Threshold grid for the legacy world, in its own arbitrary dose units (D_t = 5 sits inside it).
LEGACY_D_T = 5.0   # the frozen v4.3 dose threshold, in the legacy world's arbitrary dose units; it is a grid point so the legacy curve is exact there
THRESHOLDS_LEGACY: Tuple[float, ...] = tuple(sorted({float(10.0 ** e) for e in np.arange(-2.0, 4.0001, 0.25)} | {LEGACY_D_T}))
# S' plans against one threshold per decade; S' planned for T and scored at T is the attainable ceiling at T.
PLAN_THRESHOLDS_UMOL_H_L: Tuple[float, ...] = tuple(float(10.0 ** e) for e in range(0, 7))

W_ET, W_LABEL2_LOCAL, W_OEDEMA = 1.0, 0.3, (0.35, 0.5)
BANDS = ("eroded", "nominal", "dilated")
_STRUCT = np.ones((3, 3, 3), dtype=bool)


def band_masks(truth: np.ndarray) -> Dict[str, np.ndarray]:
    """Truth WT eroded / as is / dilated by one voxel (see the module docstring)."""
    t = np.asarray(truth, dtype=bool)
    return {"eroded": ndimage.binary_erosion(t, structure=_STRUCT, iterations=1, border_value=1),
            "nominal": t,
            "dilated": ndimage.binary_dilation(t, structure=_STRUCT, iterations=1, border_value=0)}


def voxel_weights(mask: np.ndarray, wt: np.ndarray, tc: Optional[np.ndarray], et: Optional[np.ndarray],
                  oedema_w: float, label2_w: float = W_LABEL2_LOCAL) -> np.ndarray:
    """Per-voxel viability weight on ``mask`` (zero elsewhere). Without TC/ET channels every voxel gets the oedema-free weight 1.0."""
    w = np.zeros(mask.shape, dtype=np.float64)
    if tc is None or et is None:
        w[mask] = 1.0
        return w
    tc, et = np.asarray(tc, bool), np.asarray(et, bool)
    w[mask & et] = W_ET
    w[mask & tc & ~et] = label2_w
    w[mask & ~tc] = oedema_w            # oedema, and voxels dilated in beyond the original WT
    return w


def curve(dose: np.ndarray, mask: np.ndarray, weights: Optional[np.ndarray], thresholds: Sequence[float],
          outside_crop_voxels: int = 0) -> np.ndarray:
    """Covered share at every threshold (dose >= T). ``weights`` None gives the plain unweighted share."""
    vals = dose[mask].ravel()
    w = np.ones(vals.shape, dtype=np.float64) if weights is None else weights[mask].ravel()
    order = np.argsort(vals, kind="stable")
    vals, w = vals[order], w[order]
    cum = np.concatenate([[0.0], np.cumsum(w)])
    total = cum[-1]
    mean_w = total / len(vals) if len(vals) else 0.0
    denom = total + outside_crop_voxels * mean_w
    if denom <= 0.0:
        return np.zeros(len(thresholds))
    idx = np.searchsorted(vals, np.asarray(thresholds, dtype=np.float64), side="left")
    return (total - cum[idx]) / denom


def evaluate_dose(dose: np.ndarray, truth: np.ndarray, channels: Optional[Mapping[str, np.ndarray]], outside_crop_voxels: int,
                  thresholds: Sequence[float]) -> Dict[str, Any]:
    """All curves for one run: {band: {plain, w35, w50}} (lists over ``thresholds``)."""
    bands = band_masks(truth)
    tc = None if not channels else channels.get("tc")
    et = None if not channels else channels.get("et")
    out: Dict[str, Any] = {}
    for name, m in bands.items():
        out[name] = {"plain": curve(dose, m, None, thresholds, outside_crop_voxels).tolist(),
                     "w35": curve(dose, m, voxel_weights(m, truth, tc, et, W_OEDEMA[0]), thresholds, outside_crop_voxels).tolist(),
                     "w50": curve(dose, m, voxel_weights(m, truth, tc, et, W_OEDEMA[1]), thresholds, outside_crop_voxels).tolist(),
                     "n_voxels": int(m.sum())}
    return out


def at_threshold(curves: Mapping[str, Any], thresholds: Sequence[float], t: float, kind: str = "plain") -> Dict[str, float]:
    """Interpolate (log-threshold) each band's curve at ``t``."""
    lt = np.log10(np.asarray(thresholds, dtype=np.float64))
    return {b: float(np.interp(math.log10(t), lt, np.asarray(curves[b][kind]))) for b in BANDS}


# --- the legacy stepper with off-target accounting for the band masks ----------------------------------------------------


class BandedDrugField(DrugField):
    """The legacy :class:`DrugField`, byte for byte in its arithmetic, that also accumulates the decayed mass inside extra masks, so the
    off-target share can be reported under the +/-1-voxel band. ``dose`` is identical to a plain DrugField (a test pins that)."""

    def __init__(self, *args: Any, bands: Mapping[str, np.ndarray], **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.bands = {k: np.asarray(v, dtype=bool) for k, v in bands.items()}
        self.decayed_in_band = {k: 0.0 for k in self.bands}

    def step(self, region: Optional[np.ndarray] = None) -> None:
        if self.washout is not None or self.uptake is not None:
            raise ValueError("BandedDrugField is for the legacy F-off path")
        before = float(self.conc.sum())
        if before > 0.0:
            from scipy.ndimage import gaussian_filter
            self.conc = gaussian_filter(self.conc, self.diffuse_sigma_vox, mode="constant", cval=0.0, truncate=3.0)
            np.maximum(self.conc, 0.0, out=self.conc)
            after = float(self.conc.sum())
            self.lost_boundary += before - after
            if region is not None:
                self.decayed_in_region += float(self.conc[region].sum()) * self.decay
            for k, m in self.bands.items():
                self.decayed_in_band[k] += float(self.conc[m].sum()) * self.decay
            self.conc *= 1.0 - self.decay
            self.decayed += after * self.decay
        self.dose += self.conc

    def on_target(self, band: str) -> float:
        return self.decayed_in_band[band] + float(self.conc[self.bands[band]].sum())

    def off_target_share(self, band: str) -> float:
        if self.released <= 0.0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - self.on_target(band) / self.released))


def off_target_by_band(world: Any) -> Dict[str, Dict[str, float]]:
    """Off-target drug under the +/-1-voxel band for a finished world, in mg for the units world and as a share for both.

    ``on_target`` follows ``TumorMRIWorld.off_target_share``: the mask is truth WT inside the brain (so drug in non-brain voxels inside
    the crop counts as off-target). Units: cleared inside the mask plus still present inside it. Legacy: the same, from the banded stepper."""
    f = world.field
    brain = world._percept.brain
    out: Dict[str, Dict[str, float]] = {}
    masks = {k: (m & brain) for k, m in band_masks(world._scorer.truth).items()}
    for name, m in masks.items():
        if isinstance(f, BandedDrugField):
            share = f.off_target_share(name)
            out[name] = {"off_target_share": share}
        else:
            on = f.on_target_mg(m)
            released = f.released
            out[name] = {"off_target_mg": released - on, "off_target_share": (released - on) / released if released > 0 else 0.0,
                         "on_target_mg": on}
    return out


def digest(obj: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
