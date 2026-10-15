"""Shared builders for the B2 tests: the foraging rules, scripted turns, uptake readings, an episode runner on the phantom."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from antelligence.experiments.v43_harness import V43Harness
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_policies import ScanInfo, decide_turn
from tests.v4_2.q1.helpers import REGIONS, inputs, noisy_look_factory
from tests.v4_3.a2.helpers import estimate, state
from tests.v4_3.b1.helpers import load_yaml_configs, phantom_with_channels

VM = 2.0
RF = AntRules.v43_foraging()
POS = (10, 10, 10)


def pmap(only_centre: bool = False) -> np.ndarray:
    p = np.full((45, 25, 25), 0.1)
    if only_centre:
        p[10, 10, 10] = 0.9
    else:
        p[4:17, 4:17, 4:17] = 0.9          # zone A (centroid (10, 10, 10), zone radius 7.9 voxels)
        p[24:37, 4:17, 4:17] = 0.9         # zone B
    return p


def scan(p: Optional[np.ndarray] = None) -> ScanInfo:
    return ScanInfo(REGIONS, VM, p if p is not None else pmap())


def uptake(pos, value: float, tick: int = 5) -> Dict[str, Any]:
    word = "good" if value >= 0.7 else "poor" if value < 0.4 else "middling"
    return {"tick": tick, "pos_vox": list(pos), "kind": "uptake", "value": value, "band": 0, "word": word}


def turn(st, sc, tick: int, pos=POS, ant: int = 0, **kw):
    kw.setdefault("round", tick // 5)
    return decide_turn(inputs(tick=tick, pos=pos, **kw), st, sc, ant)


def fresh(rules: AntRules = RF, p: Optional[np.ndarray] = None):
    return state(rules), scan(p)


def harness(rules: AntRules = RF, ticks: int = 120, rel: str = "config/v4_3/world.yaml") -> V43Harness:
    w, c, _ = load_yaml_configs(rel, ticks)
    return V43Harness(w, c, rules, estimate() if rules.use_combined else None, look_fn_factory=noisy_look_factory)


def run(rules: AntRules, arm: str, seed: int = 1, ticks: int = 120, rel: str = "config/v4_3/world.yaml"):
    return harness(rules, ticks, rel).run_episode(arm, phantom_with_channels(), seed)
