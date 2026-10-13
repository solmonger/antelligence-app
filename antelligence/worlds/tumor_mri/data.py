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

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

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


REGION_THRESHOLD = 0.3  # calibrated p_tumour
REGION_MIN_MM3 = 100.0
REGION_MAX_MM3 = 20000.0  # 20 cm^3: larger components are split


@dataclass(frozen=True)
class Region:
    """A perceived region (L1 definition): 6-connected component of calibrated p >= 0.3, >= 100 mm^3, split above 20 cm^3.

    Mirrors the INTERFACES ``knowledge`` payload (region_id, centroid, p_tumour, unc, volume_mm3); the centroid is in
    crop voxel coordinates here (``from_knowledge`` converts from ``centroid_dmm``).
    """

    region_id: int
    centroid_vox: Tuple[float, float, float]
    p_tumour: float
    unc: float
    volume_mm3: int

    @classmethod
    def from_knowledge(cls, payload: Mapping[str, Any], voxel_mm: float = VOXEL_MM) -> "Region":
        c = payload["centroid_dmm"]
        return cls(region_id=int(payload["region_id"]),
                   centroid_vox=tuple(float(x) / 10.0 / voxel_mm for x in c),  # type: ignore[arg-type]
                   p_tumour=int(payload["p_tumour_ppm"]) / 1e6, unc=int(payload.get("unc_ppm", 0)) / 1e6,
                   volume_mm3=int(payload["volume_mm3"]))


@dataclass(frozen=True)
class Candidate:
    """A perceived region offered to squads under a label A..F. Built from perception only."""

    label: str  # "A".."F"
    region_id: int
    centroid_vox: Tuple[float, float, float]
    p_tumour: float
    unc: float
    volume_mm3: int

    @classmethod
    def from_region(cls, label: str, r: Region) -> "Candidate":
        return cls(label, r.region_id, r.centroid_vox, r.p_tumour, r.unc, r.volume_mm3)

    def to_dict(self) -> dict:
        return {"label": self.label, "region_id": self.region_id,
                "centroid_vox": [round(float(c), 3) for c in self.centroid_vox],
                "p_tumour": round(float(self.p_tumour), 6), "unc": round(float(self.unc), 6),
                "volume_mm3": int(self.volume_mm3)}


def _split(coords: np.ndarray, max_voxels: int) -> List[np.ndarray]:
    """Recursively bisect at the median along the longest extent until each part has <= max_voxels (deterministic)."""
    if len(coords) <= max_voxels:
        return [coords]
    axis = int(np.argmax(coords.max(axis=0) - coords.min(axis=0)))
    order = np.argsort(coords[:, axis], kind="stable")
    half = len(order) // 2
    return _split(coords[order[:half]], max_voxels) + _split(coords[order[half:]], max_voxels)


def perceived_mask(p_tumour: np.ndarray, brain: np.ndarray, voxel_mm: float = VOXEL_MM) -> np.ndarray:
    """Voxels of all perceived regions (before splitting, which does not change the union)."""
    lab, n = ndimage.label((p_tumour >= REGION_THRESHOLD) & brain)  # default structure = 6-connectivity
    if n == 0:
        return np.zeros_like(brain)
    counts = ndimage.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
    keep = np.concatenate([[False], counts * voxel_mm ** 3 >= REGION_MIN_MM3])
    return keep[lab]


def perceived_regions(p_tumour: np.ndarray, unc: np.ndarray, brain: np.ndarray, voxel_mm: float = VOXEL_MM) -> List[Region]:
    """L1's region definition applied to the perception arrays. Ordered by component then split order (deterministic).

    The >20 cm^3 split rule here (median bisection along the longest axis) is a stand-in until L1's regions are
    passed in through ``TumorMRIWorld(regions=...)``.
    """
    lab, n = ndimage.label((p_tumour >= REGION_THRESHOLD) & brain)
    out: List[Region] = []
    max_vox = int(REGION_MAX_MM3 // voxel_mm ** 3)
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        if len(coords) * voxel_mm ** 3 < REGION_MIN_MM3:
            continue
        for part in _split(coords, max_vox):
            idx = tuple(part.T)
            w = p_tumour[idx]
            out.append(Region(region_id=len(out), centroid_vox=tuple(float(c) for c in (part * w[:, None]).sum(0) / w.sum()),
                              p_tumour=float(w.mean()), unc=float(unc[idx].mean()),
                              volume_mm3=int(round(len(part) * voxel_mm ** 3))))
    return out


def select_candidates(regions: Sequence[Region], max_candidates: int = 6) -> List[Candidate]:
    """Fallback candidate pool A..F: the regions ranked by p_tumour x volume (S's ranking without the distance term)."""
    ranked = sorted(regions, key=lambda r: (-r.p_tumour * r.volume_mm3, r.region_id))[:max_candidates]
    return [Candidate.from_region("ABCDEF"[i], r) for i, r in enumerate(ranked)]


def candidates_from_perception(p_tumour: np.ndarray, unc: np.ndarray, brain: np.ndarray, *,
                               voxel_mm: float = VOXEL_MM, max_candidates: int = 6) -> List[Candidate]:
    return select_candidates(perceived_regions(p_tumour, unc, brain, voxel_mm), max_candidates)


def entry_sites_with_info(brain: np.ndarray, p_tumour: np.ndarray, seed: int, *, n: int = 8, voxel_mm: float = VOXEL_MM,
                          shell_mm: float = 4.0, min_dist_mm: float = 10.0
                          ) -> Tuple[List[Tuple[int, int, int]], float]:
    """``n`` distinct brain voxels on the crop's outer shell, >= ``min_dist_mm`` from any perceived region.

    The shell is the brain voxels within ``shell_mm`` of the brain boundary (the crop face counts as boundary, because
    the mask is padded with background). If fewer than ``n`` voxels qualify the shell is widened one voxel at a time;
    the widening is returned (mm) and warned about. Deterministic in ``seed``.
    """
    padded = np.pad(brain, 1, constant_values=False)
    depth = ndimage.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
    perceived = perceived_mask(p_tumour, brain, voxel_mm)
    if perceived.any():
        far = ndimage.distance_transform_edt(~perceived) * voxel_mm >= min_dist_mm
    else:
        far = np.ones_like(brain)
    ok = brain & far
    start = max(1.0, shell_mm / voxel_mm)
    shell_vox = start
    max_depth = float(depth.max()) if brain.any() else 0.0
    while True:
        pool = np.flatnonzero((depth <= shell_vox) & ok)
        if len(pool) >= n or shell_vox > max_depth:
            break
        shell_vox += 1.0
    if len(pool) < n:
        raise CropError(f"only {len(pool)} entry voxels satisfy the constraints; need {n}")
    widened = (shell_vox - start) * voxel_mm
    if widened > 0:
        warnings.warn(f"entry shell widened by {widened:g} mm (only {n} sites fit after widening)", stacklevel=2)
    pick = np.random.default_rng(int(seed)).choice(pool, size=n, replace=False)
    return [tuple(int(c) for c in np.unravel_index(i, brain.shape)) for i in pick], float(widened)


def sample_entry_sites(brain: np.ndarray, p_tumour: np.ndarray, seed: int, **kw: Any) -> List[Tuple[int, int, int]]:
    return entry_sites_with_info(brain, p_tumour, seed, **kw)[0]
