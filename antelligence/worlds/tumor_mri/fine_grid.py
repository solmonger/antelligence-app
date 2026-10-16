"""v4.31 A.2: the native 1 mm label grid for SCORING. Additive; the 2 mm store adapter is untouched.

The A.2 design: at k_ref (L = 0.505 mm) a 2 mm voxel is four decay lengths wide and holds 69% of a release's exposure, so the scorer refines. The scorer is analytic, so
its grid is a numerical choice. **Perception, targets and release points stay on the 2 mm grid; only the scorer refines.**

* **Truth (WT, TC, ET) is native 1 mm**, straight from the MSD label NIfTIs. The existing ``raw/`` bundles are at 2 mm (120 x 120 x 78), so 1 mm truth is NOT recoverable from
  them; it is recovered from the original Task01 label files, which need no perception and no model.
* **The brain mask is native 1 mm** (nonzero in any channel, as the pipeline defines it). FLAIR and T1 are kept at 1 mm for the ventricle proxy (geometry.py).
* No perception (p_tumour, uncertainty) is interpolated onto the fine grid: that would invent information.
* A release is made at a 2 mm voxel centre. On a 1 mm or 0.5 mm lattice that is the corner shared by eight voxels; it is placed there (x, y exactly; z snapped to the nearest
  lattice corner, because the pipeline's z resample factor is 155/78 = 1.987, not 2).

Development cases only; a real store is refused inside a test (a temp-dir store is the synthetic escape hatch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_FINE_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.31/data/cases_1mm_a2")
RESAMPLE_TARGET_MM = 2.0


def axis_scales(shape_1mm: Sequence[int], target_mm: float = RESAMPLE_TARGET_MM) -> Tuple[float, ...]:
    """Per-axis 1 mm voxels per 2 mm voxel, exactly as ``scipy.ndimage.zoom(..., grid_mode=True)`` realises the 2 mm resample:
    output size = round(input / target), scale = input / output (2.0 on x and y, 155 / 78 = 1.987 on a 155-slice z)."""
    out = [int(round(n / target_mm)) for n in shape_1mm]
    return tuple(n / o for n, o in zip(shape_1mm, out))


def fine_box(origin_2mm: Sequence[int], shape_2mm: Sequence[int], shape_1mm: Sequence[int]) -> Tuple[slice, slice, slice]:
    """The 1 mm voxels whose centres fall inside the 2 mm crop box (origin and shape on the 2 mm full-volume grid)."""
    sc = axis_scales(shape_1mm)
    sl = []
    for ax in range(3):
        lo = int(np.ceil(origin_2mm[ax] * sc[ax] - 0.5))
        hi = int(np.ceil((origin_2mm[ax] + shape_2mm[ax]) * sc[ax] - 0.5))
        sl.append(slice(max(lo, 0), min(hi, shape_1mm[ax])))
    return tuple(sl)  # type: ignore[return-value]


def make_fine_arrays(image: np.ndarray, label: np.ndarray, channels: Sequence[str], truth_fn, crop2_truth: np.ndarray, origin_2mm: Sequence[int],
                     shape_2mm: Sequence[int]) -> Dict[str, object]:
    """The 1 mm scoring arrays for one case: truth (3, X, Y, Z), brain mask, FLAIR, T1, the box on the full 1 mm volume, and two checks.

    ``image`` is (C, X, Y, Z) at 1 mm in MSD channel order named by ``channels``; ``truth_fn(label)`` gives (3, X, Y, Z) bool WT, TC, ET; ``crop2_truth`` is the stored 2 mm crop truth
    (channel 0 = WT) and ``origin_2mm`` / ``shape_2mm`` its place on the 2 mm full-volume grid. ``agreement`` is the Dice of the 1 mm truth sampled back onto the 2 mm voxel centres
    (scipy's nearest tie rule floor(x + 0.5)) against the stored 2 mm truth: it must be 1, or the grids are misaligned."""
    shape1 = image.shape[1:]
    box = fine_box(origin_2mm, shape_2mm, shape1)
    truth_full = truth_fn(label)
    brain_full = np.any(image != 0, axis=0)
    names = [c.upper() for c in channels]
    out: Dict[str, object] = {
        "brain_mask": brain_full[box], "truth": truth_full[(slice(None),) + box],
        "flair": image[names.index("FLAIR")][box].astype(np.float32), "t1": image[names.index("T1")][box].astype(np.float32),
        "box": np.asarray([[s.start, s.stop] for s in box]), "origin_2mm": np.asarray(origin_2mm), "shape_2mm": np.asarray(shape_2mm),
        "full_shape": np.asarray(shape1), "spacing_mm": np.asarray([1.0, 1.0, 1.0])}
    wt = truth_full[0]
    out["outside"] = int(wt.sum()) - int(wt[box].sum())
    sc = axis_scales(shape1)
    idx = [np.clip(np.floor((np.arange(shape_2mm[ax], dtype=np.float64) + origin_2mm[ax] + 0.5) * sc[ax] - 0.5 + 0.5).astype(int), 0, shape1[ax] - 1) for ax in range(3)]
    back = wt[np.ix_(*idx)]
    ref = crop2_truth[0] if crop2_truth.ndim == 4 else crop2_truth
    out["agreement"] = 2.0 * float((back & ref).sum()) / max(float(back.sum() + ref.sum()), 1.0)
    return out


def save_fine_case(case: str, arrays: Dict[str, object], root: Union[str, Path] = DEFAULT_FINE_ROOT) -> Path:
    d = Path(root) / case
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "case.npz", **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)})
    (d / "metrics.json").write_text(json.dumps({"case": case, "grid_mm": 1.0, "truth_wt_outside_crop_voxels": int(arrays["outside"]),   # type: ignore[arg-type]
                                                "agreement_dice_vs_2mm": float(arrays["agreement"]),                                    # type: ignore[arg-type]
                                                "note": "truth, brain, FLAIR and T1 native 1 mm; no perception on this grid"}))
    return d / "case.npz"


class FineCase:
    """The 1 mm scoring arrays of one development case, and the map from 2 mm release positions to a scoring lattice."""

    def __init__(self, case: str, arrays: Dict[str, np.ndarray], outside: int) -> None:
        self.case = case
        self.truth = arrays["truth"].astype(bool)               # (3, X, Y, Z) WT, TC, ET at 1 mm
        self.brain = arrays["brain_mask"].astype(bool)
        self.flair, self.t1 = arrays["flair"], arrays["t1"]
        self.box_lo = arrays["box"][:, 0].astype(int)
        self.origin_2mm = arrays["origin_2mm"].astype(int)
        self.full_shape = tuple(int(v) for v in arrays["full_shape"])
        self.outside = int(outside)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.truth.shape[1:])  # type: ignore[return-value]

    def at_spacing(self, spacing_mm: float):
        """(truth (3,...), brain, outside_crop_truth_voxels) on a lattice of ``spacing_mm``: 1.0 is native; 0.5 and 0.25 are nearest-neighbour upsamples of the 1 mm labels (the native
        resolution; no finer truth exists, stated as the limitation of those grids: they test the scorer's resolution of the exposure gradient, not the labels)."""
        for f in (2, 4):
            if abs(spacing_mm - 1.0 / f) < 1e-9:
                up = lambda a, f=f: np.repeat(np.repeat(np.repeat(a, f, axis=-3), f, axis=-2), f, axis=-1)      # noqa: E731
                return up(self.truth), up(self.brain), self.outside * f ** 3
        if abs(spacing_mm - 1.0) < 1e-9:
            return self.truth, self.brain, self.outside
        raise ValueError(f"scoring spacing {spacing_mm} mm: only 1.0 (native), 0.5 and 0.25 (upsampled) come from the fine labels; 2.0 is the 2 mm store")

    def release_lattice(self, positions_2mm: Sequence[Sequence[int]], spacing_mm: float) -> np.ndarray:
        """Integer lattice-corner index, on the ``spacing_mm`` scoring grid, of releases made at 2 mm voxel centres (positions in the 2 mm crop grid). x and y land exactly on a
        corner; z is snapped to the nearest corner (<= half a scoring voxel)."""
        sc = axis_scales(self.full_shape)
        pos = np.asarray(positions_2mm, dtype=np.float64).reshape(-1, 3)
        p_mm = np.empty_like(pos)
        for ax in range(3):
            p_mm[:, ax] = ((self.origin_2mm[ax] + pos[:, ax]) + 0.5) * sc[ax] - self.box_lo[ax]          # position in 1 mm voxel units, edge-based, within the box
        return np.rint(p_mm / spacing_mm).astype(int)


def load_fine_case(case: str, *, root: Union[str, Path, None] = None, repo_root: Union[str, Path, None] = None) -> FineCase:
    """The fine arrays of a DEVELOPMENT case (same gate as the 2 mm adapter). A real store is refused inside a test."""
    from antelligence.experiments.v41_gate import GateError, load_checked_split
    from antelligence.experiments.v41_split import which_split
    from antelligence.worlds.tumor_mri.store_adapter import REPO_ROOT, DevCaseInTestError, is_temp_store, under_pytest

    repo = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        kind = which_split(load_checked_split(repo), case)
    except KeyError as exc:
        raise GateError(str(exc)) from exc
    if kind != "dev":
        raise GateError(f"case {case!r} is in the {kind} set; the fine-grid loader serves development cases only")
    base = Path(root) if root is not None else DEFAULT_FINE_ROOT
    if under_pytest() and not is_temp_store(base):
        raise DevCaseInTestError(f"development case {case!r} requested inside a test from {base}: tests use the synthetic phantom only")
    with np.load(base / case / "case.npz") as z:
        a = {k: z[k] for k in z.files}
    metrics = json.loads((base / case / "metrics.json").read_text())
    return FineCase(case, a, int(metrics["truth_wt_outside_crop_voxels"]))
