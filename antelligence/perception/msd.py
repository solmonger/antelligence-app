"""Medical Segmentation Decathlon Task01 (BrainTumour) reader.

Channel order and label values are READ from ``dataset.json`` and never hard-coded. The
bundle (MONAI brats_mri_segmentation) expects channels T1c, T1, T2, FLAIR; MSD ships
FLAIR, T1w, t1gd, T2w, so the reorder indices must come out as [2, 1, 3, 0]. We derive
them from the json and assert it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Channel order the bundle was trained with (inference.json / brats_mri_segmentation v0.5.4).
BUNDLE_CHANNELS = ("T1c", "T1", "T2", "FLAIR")
EXPECTED_REORDER = (2, 1, 3, 0)

_CHANNEL_ALIASES = {
    "flair": "FLAIR",
    "t1w": "T1",
    "t1": "T1",
    "t1gd": "T1c",
    "t1c": "T1c",
    "t1ce": "T1c",
    "t2w": "T2",
    "t2": "T2",
}


class MSDError(ValueError):
    """dataset.json or a case violates what the pipeline asserts about MSD Task01."""


def canonical_channel(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if key not in _CHANNEL_ALIASES:
        raise MSDError(f"unrecognised MRI modality name in dataset.json: {name!r}")
    return _CHANNEL_ALIASES[key]


def read_dataset_json(root: str | Path) -> dict:
    path = Path(root) / "dataset.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def channel_names(dataset: dict) -> list[str]:
    """Canonical channel names in MSD channel order (index = position in the 4D NIfTI)."""
    modality = dataset["modality"]
    order = sorted(modality, key=lambda k: int(k))
    if [int(k) for k in order] != list(range(len(order))):
        raise MSDError(f"modality keys are not 0..n-1: {list(modality)}")
    return [canonical_channel(modality[k]) for k in order]


def bundle_reorder_indices(names: list[str]) -> list[int]:
    """Indices that turn MSD channel order into bundle order (T1c, T1, T2, FLAIR)."""
    if sorted(names) != sorted(BUNDLE_CHANNELS):
        raise MSDError(f"channels {names} do not match bundle channels {list(BUNDLE_CHANNELS)}")
    return [names.index(c) for c in BUNDLE_CHANNELS]


def assert_msd_channel_order(dataset: dict) -> list[int]:
    """Return the reorder indices, asserting the documented MSD layout gives [2, 1, 3, 0]."""
    idx = bundle_reorder_indices(channel_names(dataset))
    if tuple(idx) != EXPECTED_REORDER:
        raise MSDError(f"reorder indices {idx} != expected {list(EXPECTED_REORDER)}")
    return idx


@dataclass(frozen=True)
class LabelDef:
    """Label values (from dataset.json) behind the three truth channels."""

    wt: tuple[int, ...]
    tc: tuple[int, ...]
    et: tuple[int, ...]


def label_definition(dataset: dict) -> LabelDef:
    """WT = any tumour label; TC = non-enhancing + enhancing; ET = enhancing.

    Resolved by label *name*; for MSD Task01 this is WT = label>0, TC = {2,3}, ET = {3}.
    """
    labels = {int(k): str(v).lower() for k, v in dataset["labels"].items()}
    tumour = {k: v for k, v in labels.items() if k != 0}
    et = tuple(sorted(k for k, v in tumour.items() if "enhancing" in v and "non" not in v))
    non_et = tuple(sorted(k for k, v in tumour.items() if "non" in v and "enhancing" in v))
    if not et or not non_et:
        raise MSDError(f"cannot resolve ET / non-enhancing labels from {labels}")
    wt = tuple(sorted(tumour))
    tc = tuple(sorted(set(et) | set(non_et)))
    return LabelDef(wt=wt, tc=tc, et=et)


def truth_channels(label: np.ndarray, ldef: LabelDef) -> np.ndarray:
    """(3, X, Y, Z) bool truth in the order WT, TC, ET."""
    return np.stack([np.isin(label, ldef.wt), np.isin(label, ldef.tc), np.isin(label, ldef.et)])


def list_case_ids(root: str | Path) -> list[str]:
    ds = read_dataset_json(root)
    return [Path(item["image"]).name.removesuffix(".nii.gz") for item in ds["training"]]


def case_paths(root: str | Path, case_id: str) -> tuple[Path, Path]:
    root = Path(root)
    img = root / "imagesTr" / f"{case_id}.nii.gz"
    lab = root / "labelsTr" / f"{case_id}.nii.gz"
    return img, lab


@dataclass
class Case:
    case_id: str
    image: np.ndarray  # (C, X, Y, Z) float32, MSD channel order
    label: np.ndarray | None  # (X, Y, Z) int16
    channels: list[str]  # canonical names, MSD order
    spacing_mm: tuple[float, float, float]
    affine: np.ndarray


def load_case(root: str | Path, case_id: str, with_label: bool = True) -> Case:
    import nibabel as nib

    ds = read_dataset_json(root)
    names = channel_names(ds)
    img_path, lab_path = case_paths(root, case_id)
    img_nii = nib.load(str(img_path))
    data = np.asarray(img_nii.dataobj, dtype=np.float32)
    if data.ndim != 4 or data.shape[-1] != len(names):
        raise MSDError(f"{img_path.name}: expected 4D with {len(names)} channels, got {data.shape}")
    image = np.moveaxis(data, -1, 0)
    label = None
    if with_label:
        label = np.asarray(nib.load(str(lab_path)).dataobj).astype(np.int16)
        if label.shape != image.shape[1:]:
            raise MSDError(f"{case_id}: label shape {label.shape} != image {image.shape[1:]}")
    zooms = img_nii.header.get_zooms()[:3]
    return Case(case_id, image, label, names, tuple(float(z) for z in zooms), img_nii.affine)


def brain_mask(image: np.ndarray) -> np.ndarray:
    """Brain = nonzero in any channel."""
    return np.any(image != 0, axis=0)
