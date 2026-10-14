"""SimLook: a simulated close-look sensor built from d1's recorded looks (v4.2 goal 5.1, F1).

What it is. d1 (the local close-look model) was run during the v4.1 step-7 SW-B calibration; each look was logged
with its position and value. Those looks, labelled with scorer-side truth, give an empirical distribution per *cell*
(true label x distance band x scan-p band). SimLook draws a fixed reading field per case from those distributions, so
an ant standing at a voxel reads what d1 would plausibly say there, with d1's mistakes concentrated where d1 is bad
(the tumour edge).

Values are already temperature-calibrated in the source (do not apply T again).

Reading field (per case, fixed):
  1. one uniform per voxel from ``rng("f1_field", case_id)`` (INTERFACES section 1);
  2. Gaussian smoothing, sigma 2 voxels;
  3. rank re-uniformisation (the smoothed noise is no longer uniform);
  4. the inverse empirical CDF of the voxel's cell, from the cell table.
Every ant, tick, seed and arm reads the same value at a voxel: the field is a pure function of
(case_id, truth, scan p, table). It is injected only through ``look_fn``; it needs the scorer-side truth to know each
voxel's cell, so build it in the harness, never in a policy.

Cell merging (a cell with fewer than ``MIN_CELL`` looks): first drop the scan-p band (pool the distance band over scan
bands); if the pooled band is still short, merge adjacent distance bands. Every merge is logged in the table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from antelligence.worlds.tumor_mri.physics import Vox, rng

SCHEMA = "cells_v4_2"
MIN_CELL = 50
N_LEVELS = 201                      # quantile levels 0, 0.005, ..., 1
SCAN_CUTS = (0.2, 0.5, 0.8)
SMOOTH_SIGMA_VOX = 2.0
# Signed distance bands, ordered from far outside to deep inside (adjacent bands are neighbours for merging).
BANDS = ("out_gt10", "out_4_10", "out_0_4", "in_0_4", "in_gt4")
N_BANDS, N_SCAN = len(BANDS), len(SCAN_CUTS) + 1
LEVELS = np.linspace(0.0, 1.0, N_LEVELS)


def distance_band_index(truth: np.ndarray, voxel_mm: float) -> np.ndarray:
    """Band index (0..4, see ``BANDS``) per voxel. Inside depth = distance (mm) to the nearest non-tumour voxel;
    outside distance = distance (mm) to the nearest tumour voxel. A voxel at the boundary is one voxel (2 mm) away."""
    truth = np.asarray(truth, dtype=bool)
    out = np.zeros(truth.shape, dtype=np.int8)
    if not truth.any():
        return out  # no tumour: everything is far outside
    d_out = distance_transform_edt(~truth) * voxel_mm
    d_in = distance_transform_edt(truth) * voxel_mm
    out[~truth & (d_out <= 10.0)] = 1
    out[~truth & (d_out <= 4.0)] = 2
    out[truth] = 3
    out[truth & (d_in > 4.0)] = 4
    return out


def scan_band_index(p: np.ndarray) -> np.ndarray:
    """Scan-p band 0..3 with cuts 0.2, 0.5, 0.8 (a value on a cut goes to the upper band)."""
    return np.searchsorted(np.asarray(SCAN_CUTS), np.asarray(p, dtype=np.float64), side="right").astype(np.int8)


def cell_key(band: int, scan: int) -> str:
    return f"d{band}s{scan}"


def _quantiles(values: np.ndarray) -> List[float]:
    return [round(float(q), 1) for q in np.quantile(np.asarray(values, dtype=np.float64), LEVELS)]


def build_cell_table(bands: Sequence[int], scans: Sequence[int], look_ppm: Sequence[float], *, source: str = "boards",
                     meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Cell table (counts + quantiles, no MRI) from labelled looks. ``bands``/``scans`` are band indices, ``look_ppm`` the
    d1 look values (ppm). Cells below ``MIN_CELL`` resolve to a pooled entry; every merge is logged in ``merge_log``."""
    b, s, v = (np.asarray(x) for x in (bands, scans, look_ppm))
    if not len(v):
        raise ValueError("no looks")
    n = np.zeros((N_BANDS, N_SCAN), dtype=int)
    for i in range(N_BANDS):
        for j in range(N_SCAN):
            n[i, j] = int(((b == i) & (s == j)).sum())
    log: List[Dict[str, Any]] = []
    groups: List[List[int]] = [[i] for i in range(N_BANDS)]

    def gn(g: List[int]) -> int:
        return int(sum(n[i].sum() for i in g))

    while True:  # merge adjacent distance bands while a (scan-pooled) group is short
        short = [k for k, g in enumerate(groups) if gn(g) < MIN_CELL]
        if not short or len(groups) == 1:
            break
        k = min(short, key=lambda k: (gn(groups[k]), k))
        nbrs = [m for m in (k - 1, k + 1) if 0 <= m < len(groups)]
        m = min(nbrs, key=lambda m: (gn(groups[m]), m))
        lo, hi = min(k, m), max(k, m)
        log.append({"merge": "distance_bands", "bands": [BANDS[i] for i in groups[lo] + groups[hi]],
                    "n_before": [gn(groups[lo]), gn(groups[hi])]})
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]

    entries: Dict[str, Dict[str, Any]] = {}
    cells: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        gname = "+".join(f"d{i}" for i in g)
        in_g = np.isin(b, g)
        pooled_key = gname
        for i in g:
            for j in range(N_SCAN):
                key = cell_key(i, j)
                label = "WT" if i >= 3 else "not WT"
                if n[i, j] >= MIN_CELL:
                    entries[key] = {"n": int(n[i, j]), "quantiles_ppm": _quantiles(v[(b == i) & (s == j)])}
                    cells[key] = {"band": BANDS[i], "scan_band": j, "label": label, "n_raw": int(n[i, j]), "entry": key}
                else:
                    if pooled_key not in entries:
                        entries[pooled_key] = {"n": int(in_g.sum()), "quantiles_ppm": _quantiles(v[in_g])}
                    cells[key] = {"band": BANDS[i], "scan_band": j, "label": label, "n_raw": int(n[i, j]), "entry": pooled_key}
                    log.append({"merge": "scan_band_dropped", "cell": key, "n_raw": int(n[i, j]), "into": pooled_key,
                                "pooled_n": int(in_g.sum())})
    return {"schema": SCHEMA, "source": source, "min_cell": MIN_CELL, "n_looks": int(len(v)), "bands": list(BANDS),
            "scan_cuts": list(SCAN_CUTS), "quantile_levels": N_LEVELS, "cells": cells, "entries": entries,
            "merge_log": log, **({"meta": dict(meta)} if meta else {})}


def save_table(table: Mapping[str, Any], path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")


def load_table(path: Union[str, Path]) -> Dict[str, Any]:
    t = json.loads(Path(path).read_text())
    if t.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} table")
    return t


def uniform_field(shape: Sequence[int], case_id: str) -> np.ndarray:
    """Steps 1-3: seeded uniforms -> Gaussian smoothing (sigma 2 voxels) -> rank re-uniformisation, values in (0, 1)."""
    u = rng("f1_field", case_id).random(tuple(shape))
    sm = gaussian_filter(u, SMOOTH_SIGMA_VOX, mode="reflect").ravel()
    order = np.argsort(sm, kind="stable")
    q = np.empty(sm.shape, dtype=np.float64)
    q[order] = (np.arange(sm.size, dtype=np.float64) + 0.5) / sm.size
    return q.reshape(tuple(shape))


class SimLook:
    """Fixed per-case close-look field. ``look_fn``-compatible: ``simlook(squad, pos, tick) -> float`` in [0, 1].

    ``truth`` (whole-tumour mask) and ``p_tumour`` (the scan's calibrated p) only choose each voxel's cell; they are
    scorer-side inputs, so construct this in the harness. The value never depends on squad, tick, seed or process."""

    def __init__(self, case_id: str, truth: np.ndarray, p_tumour: np.ndarray, voxel_mm: float, table: Mapping[str, Any]) -> None:
        if table.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} table")
        if np.shape(truth) != np.shape(p_tumour):
            raise ValueError("truth and p_tumour must have the same shape")
        self.case_id, self.source, self.table = case_id, table.get("source"), table
        self.shape = tuple(np.shape(truth))
        cell = distance_band_index(truth, voxel_mm).astype(np.int16) * N_SCAN + scan_band_index(p_tumour)
        q = uniform_field(self.shape, case_id)
        values = np.zeros(self.shape, dtype=np.float64)
        for i in range(N_BANDS):
            for j in range(N_SCAN):
                m = cell == i * N_SCAN + j
                if m.any():
                    entry = table["entries"][table["cells"][cell_key(i, j)]["entry"]]
                    values[m] = np.interp(q[m], LEVELS, np.asarray(entry["quantiles_ppm"], dtype=np.float64)) / 1e6
        self._field = np.clip(values, 0.0, 1.0)

    def __call__(self, squad: int, pos: Vox, tick: int) -> float:
        return float(self._field[tuple(int(c) for c in pos)])

    def at(self, pos: Vox) -> float:
        return self(0, pos, 0)

    def describe(self) -> Dict[str, Any]:
        return {"sensor": "SimLook (simulated close look built from d1's recorded looks)", "source": self.source,
                "n_looks": self.table.get("n_looks"), "case_id": self.case_id}
