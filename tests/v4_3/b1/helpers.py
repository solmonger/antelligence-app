"""Shared builders for the B1 tests: synthetic TC / ET channels on the phantom, configs, harnesses, digests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import yaml
from scipy import ndimage

from antelligence.experiments.v42_harness import CaseData, V42Harness, bundle_of as v42_bundle_of
from antelligence.kernel.colony.episode import ColonyConfig
from antelligence.worlds.tumor_mri import WorldConfig
from antelligence.worlds.tumor_mri.config import FactsConfig
from tests.v4_2.q1.helpers import noisy_look_factory, phantom

REPO = Path(__file__).resolve().parents[3]


def channels(crop) -> Dict[str, np.ndarray]:
    """Synthetic scorer-side TC / ET for the phantom: TC = WT eroded once, ET = WT eroded twice (a rim of 1 voxel is
    ET-free here on purpose: the shell is core, the inner part is ET). Nothing real is involved."""
    tc = ndimage.binary_erosion(crop.truth, iterations=1)
    et = ndimage.binary_erosion(crop.truth, iterations=2)
    return {"tc": tc, "et": et}


def phantom_with_channels() -> CaseData:
    case = phantom()
    return replace(case, scorer_channels=channels(case.crop))


def load_yaml_configs(rel: str, ticks: int = 60):
    doc = yaml.safe_load((REPO / rel).read_text())
    return (WorldConfig.from_dict(doc["world"]).with_(ticks=ticks),
            ColonyConfig.from_dict({**doc.get("colony", {}), "quiet_rule": False}), doc)


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def record_digest(rec: Mapping[str, Any]) -> str:
    """Everything the bundle, the cards and the scripts report, as one hash."""
    return digest({"bundle": v42_bundle_of(rec), "cards": rec.get("cards_final"), "counters": rec.get("script_counters"),
                   "board_entries": rec.get("board_entries"), "alarms": rec.get("queen_alarms")})


def v42_harness_for(rel: str, ticks: int = 60) -> V42Harness:
    w, c, _ = load_yaml_configs(rel, ticks)
    return V42Harness(w, c, look_fn_factory=noisy_look_factory)
