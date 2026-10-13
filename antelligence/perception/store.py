"""Per-case npz store under data/v4_1/cases/<case>/ (all arrays at 2 mm, inside the crop)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ARRAYS = ("brain_mask", "p_tumour", "unc", "flair", "t1c", "truth")
PATCH = 64
DEFAULT_ROOT = Path(os.environ.get("ANTELLIGENCE_V41_CASES",
                                   Path(__file__).resolve().parents[2] / "data" / "v4_1" / "cases"))


def case_file(case: str, root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT) / case / "case.npz"


def save_case(case: str, root: str | Path | None = None, **arrays: np.ndarray) -> Path:
    """Write brain_mask, p_tumour, unc, flair, t1c, truth (+ optional extras like p_tc, p_et, origin)."""
    missing = [a for a in ARRAYS if a not in arrays]
    if missing:
        raise KeyError(f"missing arrays: {missing}")
    path = case_file(case, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


def load_case_arrays(case: str, root: str | Path | None = None) -> dict[str, np.ndarray]:
    with np.load(case_file(case, root)) as z:
        return {k: z[k] for k in z.files}


def local_patch(case: str, xyz, root: str | Path | None = None, arrays: dict | None = None) -> np.ndarray:
    """Axial FLAIR + T1c patch, shape (64, 64, 2), centred on voxel ``xyz`` (crop-grid indices).

    Channels: [..., 0] = FLAIR, [..., 1] = T1c. Outside the crop the patch is zero padded.
    """
    a = arrays if arrays is not None else load_case_arrays(case, root)
    x, y, z = (int(v) for v in xyz)
    fl, t1 = a["flair"], a["t1c"]
    if not 0 <= z < fl.shape[2]:
        raise IndexError(f"z={z} outside crop depth {fl.shape[2]}")
    out = np.zeros((PATCH, PATCH, 2), dtype=np.float32)
    half = PATCH // 2
    x0, y0 = x - half, y - half
    xs, xe = max(x0, 0), min(x0 + PATCH, fl.shape[0])
    ys, ye = max(y0, 0), min(y0 + PATCH, fl.shape[1])
    if xs < xe and ys < ye:
        sl = (slice(xs - x0, xe - x0), slice(ys - y0, ye - y0))
        out[sl[0], sl[1], 0] = fl[xs:xe, ys:ye, z]
        out[sl[0], sl[1], 1] = t1[xs:xe, ys:ye, z]
    return out
