"""v4.2 H1: the three control arms, registered with the Q1 harness through ``register_arm`` / ``register_controls``.

* ``S-marks-displaced``  S-marks, every mark moved to a random brain voxel of its zone (M1 ``DisplaceMarks`` as ``mark_pos_fn``).
* ``S-talk-shuffled``    S-talk, every ask goes to a random other ant (K1 ``ArmSpec.shuffled``; the driver does it, no hook).
* ``S-marks-perfect``    diagnostic, not a verdict arm: judgement marks and automatic sensor marks rewritten from truth
                         (M1 ``PerfectMarks``; ``fast_delta = 0.5 * f3_decay``, ``slow_u = 0.7``).

The ant's observation keys are those of its parent arm (``colony_obs.ALLOWED``); the runtime barrier check stays on. Truth
(tumour label, u, delta) is read from the world's scorer side here, in the harness, never handed to a policy.
Importing this module registers the arms; it is idempotent.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from antelligence.experiments.v42_controls import SLOW_U, DisplaceMarks, PerfectMarks
from antelligence.experiments.v42_harness import ARMS, ArmDef, register_arm, register_controls

CONTROL_ARMS = ("S-marks-displaced", "S-talk-shuffled", "S-marks-perfect")


class WorldTruth:
    """M1's ``Truth`` protocol over the world's scorer-side arrays (label from the crop, u / delta from the drug field)."""

    def __init__(self, world: Any) -> None:
        self.truth = world.scorer_truth().truth
        self.uptake = world.field.uptake
        self.washout = world.field.washout

    def label(self, pos) -> bool:
        return bool(self.truth[tuple(pos)])

    def u(self, pos) -> float:
        return 1.0 if self.uptake is None else float(self.uptake[tuple(pos)])

    def delta(self, pos) -> float:
        return 0.0 if self.washout is None else float(self.washout[tuple(pos)])


def _displaced(world, case, seed, world_cfg, colony_cfg) -> Dict[str, Any]:
    d = DisplaceMarks(world.zones(), np.argwhere(case.crop.brain), world.voxel_mm, case.case_id, seed)
    return {"mark_pos_fn": d, "describe": lambda: {"displaced": len(d.log), "log_head": d.log[:5]}}


def _perfect(world, case, seed, world_cfg, colony_cfg) -> Dict[str, Any]:
    pm = PerfectMarks(WorldTruth(world), fast_delta=0.5 * float(world_cfg.facts.f3_decay), slow_u=SLOW_U)
    return {"wrap_policy_for": pm.wrap_policy_for, "wrap_auto_rule": pm.wrap_auto_rule,
            "describe": lambda: {"rewritten": pm.rewritten, "dropped": pm.dropped, "fast_delta": pm.fast_delta}}


def register_all() -> None:
    for name, kind_verdict in (("S-marks-displaced", True), ("S-talk-shuffled", True), ("S-marks-perfect", False)):
        if name not in ARMS:
            register_arm(ArmDef(name, "ant", name, verdict=kind_verdict))
    register_controls("S-marks-displaced", _displaced)
    register_controls("S-marks-perfect", _perfect)


register_all()
