"""v4.3 arms and their control hooks, resolved EXPLICITLY (never through the shared v4.2 registry).

v4.2's ``v42_arms`` registers its controls in the global ``ARM_CONTROLS`` under the same arm names, and the global tables can be
edited by any test after import. ``V43Harness`` therefore asks this module for an arm's definition (:func:`resolve_arm`) and its
control hooks (:func:`controls_for`), both read from tables that live here. :func:`register_all` also makes sure the global
``ARMS`` table holds every v4.3 arm name (script-side name lookups such as ``v43_run.run_one`` use it); it never writes the
global ``ARM_CONTROLS``, so importing both modules cannot make one set of controls win over the other. Call it explicitly
(tests, drivers); nothing is registered at import time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from antelligence.experiments.v42_arms import WorldTruth
from antelligence.experiments.v42_harness import ARMS, ArmControls, ArmDef
from antelligence.experiments.v43_controls import DisplaceMarksV43, PerfectMarksV43
from antelligence.experiments.v42_controls import SLOW_U

CONTROL_ARMS = ("S-marks-displaced", "S-talk-shuffled", "S-marks-perfect")
BASE_ARMS = ("S", "S-solo", "S-marks", "S-talk", "S-marks-quiet", "S'")
SHARING_ARMS = ("S-marks", "S-talk")
EXEMPT_ARMS = ("S", "S-solo", "S'")                 # may run on a development case before the freeze (goal 4, gate A)

V43_ARMS: Dict[str, ArmDef] = {a.name: a for a in (
    ArmDef("S", "script_s", "S"),
    ArmDef("S-solo", "ant", "S-solo"),
    ArmDef("S-marks", "ant", "S-marks"),
    ArmDef("S-talk", "ant", "S-talk"),
    ArmDef("S-marks-quiet", "ant", "S-marks-quiet", quiet_rule=True, verdict=False),
    ArmDef("S'", "oracle", "S"),
    ArmDef("S-marks-displaced", "ant", "S-marks-displaced", verdict=True),
    ArmDef("S-talk-shuffled", "ant", "S-talk-shuffled", verdict=True),
    ArmDef("S-marks-perfect", "ant", "S-marks-perfect", verdict=False),
)}


def resolve_arm(name: str) -> ArmDef:
    if name not in V43_ARMS:
        raise KeyError(f"unknown arm {name!r}; v4.3 arms: {sorted(V43_ARMS)}")
    return V43_ARMS[name]


def needs_freeze(arm: str) -> bool:
    """True for every sharing arm and every control: anything but S, S-solo and S'."""
    resolve_arm(arm)
    return arm not in EXEMPT_ARMS


def _displaced(world: Any, case: Any, seed: int, world_cfg: Any, colony_cfg: Any) -> Dict[str, Any]:
    d = DisplaceMarksV43(world.zones(), np.argwhere(case.crop.brain), world.voxel_mm, case.case_id, seed)
    return {"mark_pos_fn": d, "describe": lambda: {"displaced": len(d.log), "uptake_to_random_zone": sum(r["kind"] in ("good_uptake", "slow_uptake") for r in d.log),
                                                    "log_head": d.log[:5]}}


def _perfect(world: Any, case: Any, seed: int, world_cfg: Any, colony_cfg: Any) -> Dict[str, Any]:
    pm = PerfectMarksV43(WorldTruth(world), fast_delta=0.5 * float(world_cfg.facts.f3_decay), slow_u=SLOW_U)
    return {"wrap_policy_for": pm.wrap_policy_for, "wrap_auto_rule": pm.wrap_auto_rule,
            "describe": lambda: {"rewritten": pm.rewritten, "dropped": pm.dropped, "fast_delta": pm.fast_delta}}


V43_CONTROLS: Dict[str, ArmControls] = {"S-marks-displaced": _displaced, "S-marks-perfect": _perfect}


def controls_for(name: str) -> Optional[ArmControls]:
    """The v4.3 control hooks of an arm, or None (S-talk-shuffled needs none: the driver shuffles, ``ArmSpec.shuffled``)."""
    return V43_CONTROLS.get(name)


def register_all() -> Tuple[str, ...]:
    """Make the global ``ARMS`` table hold every v4.3 arm (add what is missing; never replace). Idempotent. Returns the names."""
    for name, arm in V43_ARMS.items():
        ARMS.setdefault(name, arm)
    return tuple(V43_ARMS)
