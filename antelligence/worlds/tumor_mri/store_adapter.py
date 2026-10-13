"""Load a perception case from the L1 store into the tumor_mri world's :class:`Crop` and :class:`Region` list.

Store layout (written by ``antelligence.perception.store.save_case`` / ``pipeline.finalize_case``):
``<root>/<case>/case.npz`` holds ``brain_mask, p_tumour, unc, flair, t1c, truth`` (truth is ``(3, X, Y, Z)`` in the
order WT, TC, ET; the world scores the whole tumour, channel 0) plus ``spacing_mm`` (3,), all on the 2 mm crop grid;
``<root>/<case>/metrics.json`` holds the ``knowledge`` region payloads (INTERFACES section 4, centroids in the crop grid
per section 8) and, for dev cases, ``truth_wt_outside_crop_voxels``.

Split gate: the store keeps truth for every case, so reading it is gated. Dev cases load freely. Eval cases load only
when :func:`antelligence.experiments.v41_gate.assert_eval_allowed` passes (committed preregistration, confirmed anchor
receipt, G1 file). Reserve and unknown cases never load. There is no bypass argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from antelligence.experiments.v41_gate import GateError, assert_eval_allowed, load_checked_split
from antelligence.experiments.v41_split import which_split
from antelligence.perception import store as l1_store
from antelligence.worlds.tumor_mri.config import perceived_volume_vox
from antelligence.worlds.tumor_mri.data import Crop, Region

REPO_ROOT = Path(__file__).resolve().parents[3]
WT = 0  # truth channel order from L1: WT, TC, ET
OUTSIDE_KEY = "truth_wt_outside_crop_voxels"


class AdapterError(RuntimeError):
    """The stored case is incomplete or inconsistent."""


@dataclass(frozen=True)
class LoadedCase:
    crop: Crop
    regions: List[Region]
    split: str  # "dev" or "eval"

    @property
    def v_perceived(self) -> float:
        """V_perceived (voxels): sum of calibrated p_tumour over the crop; the per_perceived_volume drug basis."""
        return perceived_volume_vox(self.crop.p_tumour)


def _outside_voxels(case: str, metrics: dict, cases_root: Path, raw_dir: Optional[Path]) -> int:
    if OUTSIDE_KEY in metrics:
        return int(metrics[OUTSIDE_KEY])
    raw = Path(raw_dir) if raw_dir is not None else cases_root.parent / "raw"
    if not (raw / f"{case}.npz").is_file():
        raise AdapterError(f"{case}: {OUTSIDE_KEY} is not in metrics.json and no raw bundle at {raw}")
    from antelligence.perception import pipeline
    return pipeline.outside_crop_voxels_from_raw(case, raw, cases_root)


def load_case(case: str, *, root: Union[str, Path, None] = None, repo_root: Union[str, Path, None] = None,
              prereg_path: Union[str, Path, None] = None, anchor_receipt: Union[str, Path, None] = None,
              g1_file: Union[str, Path, None] = None, raw_dir: Union[str, Path, None] = None) -> LoadedCase:
    """Return the case's crop and perceived regions. Raises :class:`GateError` for held-out cases without the lock."""
    repo = Path(repo_root) if repo_root is not None else REPO_ROOT
    if prereg_path is None or anchor_receipt is None or g1_file is None:
        # dev cases need no lock paths; an eval case with missing paths must fail inside the gate, not here
        try:
            kind = which_split(load_checked_split(repo), case)
        except KeyError as exc:
            raise GateError(str(exc)) from exc
        if kind != "dev":
            raise GateError(f"case {case!r} is in the {kind} set; the adapter loads it only through the v41_gate lock "
                            "(pass prereg_path, anchor_receipt and g1_file)")
        split = kind
    else:
        split = assert_eval_allowed(case, repo, prereg_path, anchor_receipt, g1_file)
    cases_root = Path(root) if root is not None else l1_store.DEFAULT_ROOT
    arrays = l1_store.load_case_arrays(case, cases_root)
    metrics_path = cases_root / case / "metrics.json"
    if not metrics_path.is_file():
        raise AdapterError(f"{case}: metrics.json missing at {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    truth = np.asarray(arrays["truth"])
    if truth.ndim != 4 or truth.shape[0] != 3:
        raise AdapterError(f"{case}: truth must be (3, X, Y, Z) in WT, TC, ET order; got {truth.shape}")
    spacing = arrays.get("spacing_mm")
    voxel_mm = float(np.asarray(spacing).ravel()[0]) if spacing is not None else 2.0
    if abs(voxel_mm - 2.0) > 1e-6:
        raise AdapterError(f"{case}: crop grid must be 2 mm, got {voxel_mm}")
    outside = _outside_voxels(case, metrics, cases_root, Path(raw_dir) if raw_dir is not None else None)
    crop = Crop(brain=arrays["brain_mask"], p_tumour=arrays["p_tumour"], unc=arrays["unc"], truth=truth[WT],
                flair=arrays["flair"], t1c=arrays["t1c"], outside_crop_truth_voxels=outside, voxel_mm=voxel_mm,
                case_id=case)
    regions = [Region.from_knowledge(p, voxel_mm) for p in metrics["regions"]]
    return LoadedCase(crop=crop, regions=regions, split=split)
