"""Shared builders for the Q1 tests: synthetic turn inputs, the phantom case, a fake close look, a harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from antelligence.experiments.v42_harness import CaseData, V42Harness, phantom_case
from antelligence.worlds.tumor_mri import WorldConfig
from antelligence.worlds.tumor_mri.colony_policies import ScanInfo, TurnInputs
from antelligence.worlds.tumor_mri.config import FactsConfig
from antelligence.worlds.tumor_mri.data import Region

PHANTOM = Path(__file__).resolve().parents[2] / "v4_1" / "world" / "fixtures" / "phantom.npz"


def phantom() -> CaseData:
    return phantom_case(PHANTOM)


def fake_look_factory(case: CaseData):
    """A fake close look: the scan's own p, a pure function of the voxel (so identical for every ant, tick and arm)."""
    p = case.crop.p_tumour
    return lambda squad, pos, tick: float(p[tuple(int(c) for c in pos)])


def noisy_look_factory(case: CaseData):
    """A fake close look that disagrees with the scan in places: half scan p, half a fixed per-voxel pseudo-random draw."""
    p = case.crop.p_tumour

    def look(squad, pos, tick):
        x, y, z = (int(c) for c in pos)
        u = ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) % 1000 / 1000.0
        return float(0.5 * p[x, y, z] + 0.5 * u)
    return look


def phantom_cfg(ticks: int = 60, **facts: Any) -> WorldConfig:
    cfg = WorldConfig(total_drug=400.0, dose_threshold=6.0, ticks=ticks)
    return cfg.with_(facts=FactsConfig(**facts)) if facts else cfg


def harness(ticks: int = 60, *, fake: bool = True, noisy: bool = False, with_board: bool = True, **facts: Any) -> V42Harness:
    cfg = phantom_cfg(ticks, **({"f1_simlook": True, **facts} if not fake else facts))
    factory = None if not fake else (noisy_look_factory if noisy else fake_look_factory)
    return V42Harness(cfg, look_fn_factory=factory, with_board=with_board)


REGIONS = (Region(0, (10.0, 10.0, 10.0), 0.8, 0.1, 4000), Region(1, (30.0, 10.0, 10.0), 0.7, 0.1, 3000))
SCAN = ScanInfo(REGIONS, 2.0)


def cand(label: str, rid: int, pos, p: float, dist: float, visited: bool = False) -> Dict[str, Any]:
    return {"label": label, "region_id": rid, "centroid_vox": list(pos), "p_tumour": p, "unc": 0.1, "volume_mm3": 4000,
            "dist_mm": dist, "visited": visited}


def inputs(**kw: Any) -> TurnInputs:
    base: Dict[str, Any] = dict(
        tick=0, round=0, pos=(10, 10, 10), voxel_mm=2.0, payload_fraction=1.0, scan_p=0.7, look=0.8, dose_rel=0.0,
        candidates=[cand("A", 0, (10, 10, 10), 0.8, 0.0), cand("B", 1, (30, 10, 10), 0.7, 40.0)], reading_log=[], uptake=None,
        visited_regions=frozenset({0, 1}), zone_need={0: "high", 1: "medium"})
    base.update(kw)
    return TurnInputs(**base)


def log_look(pos, value, tick=0):
    return {"tick": tick, "pos_vox": list(pos), "kind": "look", "value": value, "band": 0, "word": None}


def mark_row(kind, pos, *, strength=1.0, authors=(3,), strong=False, disputed=False):
    return {"kind": kind, "pos_vox": list(pos), "strength": strength, "distinct_authors": len(authors), "strong": strong,
            "disputed": disputed, "authors": list(authors)}
