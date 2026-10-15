"""Shared builders for the A2 tests: a fixed in-code coefficient dict (the real file is merged later), rule states, harnesses."""

from __future__ import annotations

from typing import Any, Dict, Optional

from antelligence.experiments.v43_harness import V43Harness
from antelligence.kernel.colony.thresholds import Thresholds
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_policies import AntState
from antelligence.worlds.tumor_mri.combined import CombinedEstimate
from tests.v4_2.q1.helpers import fake_look_factory, noisy_look_factory, phantom_cfg

COEF: Dict[str, Any] = {"schema": "combined_v4_3", "features": ["1", "logit_scan", "logit_look", "logit_scan*logit_look"],
                        "coef": [-0.2, 0.8, 1.2, 0.1], "clip": [0.001, 0.999], "fit": {"note": "test fixture, not a fit"}}
R42 = AntRules.v42()
R43 = AntRules.v43()


def estimate() -> CombinedEstimate:
    return CombinedEstimate.from_dict(COEF)


def state(rules: AntRules = R43, *, no_verify: bool = False) -> AntState:
    st = AntState(thresholds=Thresholds(**rules.threshold_kwargs()), rules=rules)
    if no_verify:
        st.thresholds.theta["verify"] = 1.0
    return st


def log_est(pos, est: float, tick: int = 0, *, raw: float = 0.95, scan_p: float = 0.6) -> Dict[str, Any]:
    """A reading-log look whose raw value differs from its combined estimate, so a test proves which one a rule read."""
    return {"tick": tick, "pos_vox": list(pos), "kind": "look", "value": raw, "band": 0, "word": None, "scan_p": scan_p, "est": est}


def v43_harness(rules: AntRules = R43, ticks: int = 60, *, noisy: bool = True, est: Optional[CombinedEstimate] = None,
                with_board: bool = True) -> V43Harness:
    cfg = phantom_cfg(ticks)
    if est is None and rules.use_combined:
        est = estimate()
    return V43Harness(cfg, None, rules, est, look_fn_factory=noisy_look_factory if noisy else fake_look_factory,
                      with_board=with_board)
