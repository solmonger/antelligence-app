"""Crop data for the 3D drug-delivery world.

A :class:`Crop` is the perception store's per-case record resampled to 2 mm:
``brain`` mask, ``p_tumour`` and ``unc`` (perception outputs), the optional
``flair`` / ``t1c`` images and ``truth`` (whole-tumour ground truth). The world
splits it at construction: perception arrays go to the observation builder,
``truth`` goes only behind the scorer (see ``world.Scorer``).

Perception store format (``.npz``): keys ``brain, p_tumour, unc, flair, t1c, truth``;
optional scalars ``outside_crop_truth_voxels`` and ``voxel_mm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

VOXEL_MM = 2.0
REQUIRED_KEYS = ("brain", "p_tumour", "unc", "truth")
OPTIONAL_KEYS = ("flair", "t1c")


class CropError(ValueError):
    """The arrays do not form a valid crop."""


@dataclass(frozen=True)
class Crop:
    brain: np.ndarray  # bool
    p_tumour: np.ndarray  # float in [0, 1]
    unc: np.ndarray  # float in [0, 1]
    truth: np.ndarray  # bool, whole tumour inside the crop; scorer-only
    flair: Optional[np.ndarray] = None
    t1c: Optional[np.ndarray] = None
    outside_crop_truth_voxels: int = 0  # truth WT voxels of the case that fall outside the crop
    voxel_mm: float = VOXEL_MM
    case_id: str = "synthetic"

    def __post_init__(self) -> None:
        shape = np.shape(self.brain)
        if len(shape) != 3:
            raise CropError("arrays must be 3D")
        for name in ("p_tumour", "unc", "truth") + OPTIONAL_KEYS:
            arr = getattr(self, name)
            if arr is not None and np.shape(arr) != shape:
                raise CropError(f"{name} shape {np.shape(arr)} != brain shape {shape}")
        if self.outside_crop_truth_voxels < 0:
            raise CropError("outside_crop_truth_voxels must be >= 0")
        object.__setattr__(self, "brain", np.asarray(self.brain, dtype=bool))
        object.__setattr__(self, "truth", np.asarray(self.truth, dtype=bool))
        object.__setattr__(self, "p_tumour", np.asarray(self.p_tumour, dtype=np.float64))
        object.__setattr__(self, "unc", np.asarray(self.unc, dtype=np.float64))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.brain.shape)  # type: ignore[return-value]

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], *, outside_crop_truth_voxels: Optional[int] = None,
                    voxel_mm: Optional[float] = None, case_id: str = "synthetic") -> "Crop":
        missing = [k for k in REQUIRED_KEYS if k not in arrays]
        if missing:
            raise CropError(f"missing arrays: {missing}")
        if outside_crop_truth_voxels is None:
            outside_crop_truth_voxels = int(arrays["outside_crop_truth_voxels"]) if "outside_crop_truth_voxels" in arrays else 0
        if voxel_mm is None:
            voxel_mm = float(arrays["voxel_mm"]) if "voxel_mm" in arrays else VOXEL_MM
        return cls(brain=arrays["brain"], p_tumour=arrays["p_tumour"], unc=arrays["unc"], truth=arrays["truth"],
                   flair=arrays.get("flair"), t1c=arrays.get("t1c"),
                   outside_crop_truth_voxels=int(outside_crop_truth_voxels), voxel_mm=float(voxel_mm), case_id=case_id)

    @classmethod
    def from_npz(cls, path: Union[str, Path], *, outside_crop_truth_voxels: Optional[int] = None,
                 case_id: Optional[str] = None) -> "Crop":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return cls.from_arrays(arrays, outside_crop_truth_voxels=outside_crop_truth_voxels,
                               case_id=case_id or path.stem)


@dataclass(frozen=True)
class Candidate:
    """A perceived region a squad may be sent to. Built from perception only."""

    label: str  # "A".."F"
    region_id: int
    centroid_vox: Tuple[float, float, float]
    p_tumour: float
    unc: float
    volume_mm3: int

    def to_dict(self) -> dict:
        return {"label": self.label, "region_id": self.region_id,
                "centroid_vox": [round(float(c), 3) for c in self.centroid_vox],
                "p_tumour": round(float(self.p_tumour), 6), "unc": round(float(self.unc), 6),
                "volume_mm3": int(self.volume_mm3)}


def candidates_from_perception(p_tumour: np.ndarray, unc: np.ndarray, brain: np.ndarray, *, voxel_mm: float = VOXEL_MM,
                               threshold: float = 0.5, min_voxels: int = 8, max_candidates: int = 6) -> List[Candidate]:
    """Connected regions (26-connectivity) of p >= threshold inside the brain, labelled A.. by descending mass."""
    mask = (p_tumour >= threshold) & brain
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    counts = ndimage.sum(mask, lab, idx)
    mass = ndimage.sum(np.where(mask, p_tumour, 0.0), lab, idx)
    order = [i for i in np.argsort(-mass, kind="stable") if counts[i] >= min_voxels][:max_candidates]
    out: List[Candidate] = []
    for rank, i in enumerate(order):
        region = idx[i]
        weights = np.where(lab == region, p_tumour, 0.0)
        centroid = ndimage.center_of_mass(weights)
        sel = lab == region
        out.append(Candidate(label="ABCDEF"[rank], region_id=int(region), centroid_vox=tuple(float(c) for c in centroid),
                             p_tumour=float(p_tumour[sel].mean()), unc=float(unc[sel].mean()),
                             volume_mm3=int(round(counts[i] * voxel_mm ** 3))))
    return out


def sample_entry_sites(brain: np.ndarray, p_tumour: np.ndarray, seed: int, *, n: int = 8, voxel_mm: float = VOXEL_MM,
                       shell_mm: float = 4.0, min_dist_mm: float = 10.0,
                       perceived_threshold: float = 0.3) -> List[Tuple[int, int, int]]:
    """``n`` distinct brain voxels on the crop's outer shell, >= ``min_dist_mm`` from any perceived region.

    The shell is the brain voxels within ``shell_mm`` of the brain boundary (the crop face counts as boundary,
    because the mask is padded with background). If fewer than ``n`` voxels qualify the shell is widened one
    voxel at a time. Deterministic in ``seed``.
    """
    padded = np.pad(brain, 1, constant_values=False)
    depth = ndimage.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
    perceived = (p_tumour >= perceived_threshold) & brain
    if perceived.any():
        far = ndimage.distance_transform_edt(~perceived) * voxel_mm >= min_dist_mm
    else:
        far = np.ones_like(brain)
    ok = brain & far
    shell_vox = max(1.0, shell_mm / voxel_mm)
    max_depth = float(depth.max()) if brain.any() else 0.0
    while True:
        pool = np.flatnonzero((depth <= shell_vox) & ok)
        if len(pool) >= n or shell_vox > max_depth:
            break
        shell_vox += 1.0
    if len(pool) < n:
        raise CropError(f"only {len(pool)} entry voxels satisfy the constraints; need {n}")
    pick = np.random.default_rng(int(seed)).choice(pool, size=n, replace=False)
    return [tuple(int(c) for c in np.unravel_index(i, brain.shape)) for i in pick]
