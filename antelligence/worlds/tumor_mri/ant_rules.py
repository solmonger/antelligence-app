"""``AntRules``: every constant the colony ant's one decision function uses (v4.3 design readings 1 to 10).

``AntRules.v42()`` reproduces the v4.2 ant EXACTLY (it is what the legacy golden tests pin). ``AntRules.v43()`` is the
Step A fix: the ant acts on the combined estimate ``c`` instead of the raw close look, releases only at ``c >= 0.6``, lets a
fold-in or an answer only LOWER the estimate, bounds verify, and moves a job threshold only when its previous turn resolved.

Frozen, hashable, JSON-plain. Domain constants live here; the decision function in ``colony_policies`` reads them and never
looks at an arm's name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Tuple

VERSIONS = ("v42", "v43")
ANSWER_RULES = ("v42", "v43")


@dataclass(frozen=True)
class AntRules:
    version: str = "v42"
    use_combined: bool = False            # c = the combined estimate (reading-log ``est``) instead of the raw look
    # --- release -----------------------------------------------------------------------------------------------------
    release_c: float = 0.5                # dose job releases only at c >= this (v4.3: 0.6)
    explore_release_c: float = 0.65       # the explore job releases only at c >= this (v4.3: 0.6)
    explore_scan_min: float = 0.0         # the explore job also needs scan p >= this (v4.3: 0.5; 0.0 = no condition)
    washout_release_c: float = 0.65       # a fast washout here skips the release when c is below this
    # --- judgement marks ---------------------------------------------------------------------------------------------
    tumour_here_c: float = 0.65           # tumour_here: c >= this and scan p < tumour_here_scan_max (v4.3: 0.8)
    tumour_here_scan_max: float = 0.5
    tumour_here_any_c: Optional[float] = 0.8   # v4.2 only: ... or c >= this anywhere (v4.3: None, no such branch)
    not_tumour_c: float = 0.3             # not_tumour: c <= this and scan p >= not_tumour_scan_min (v4.3: 0.2)
    not_tumour_scan_min: float = 0.5
    # --- verify ------------------------------------------------------------------------------------------------------
    grey: Tuple[float, float] = (0.35, 0.65)
    verify_per_spot_max: Optional[int] = None      # at most this many verify turns per spot per ant (v4.3: 2; None = no cap)
    verify_share_max: Optional[float] = None       # (verify turns + 1) <= share * (decision turns + 1) (v4.3: 0.25)
    # --- thresholds --------------------------------------------------------------------------------------------------
    theta_step_up: float = 0.05
    theta_step_down: float = 0.1          # v4.3: 0.05
    theta_floor: float = 0.1
    theta_ceil: float = 1.0
    resolved_only: bool = False           # v4.3: theta moves only when the previous turn RESOLVED
    # --- talk and cards ----------------------------------------------------------------------------------------------
    fold_drop_c: float = 0.5              # a target whose folded estimate is below this is not eligible this turn (v4.3: 0.6)
    fold_can_permit: bool = True          # v4.3: False, the folded estimate can only lower c
    answer_rule: str = "v42"              # message answers (see ``kernel.colony.messages.answer_question``)
    card_min_resolved: int = 1            # a card says "usually right" only with at least this many resolved claims

    def __post_init__(self) -> None:
        if self.version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}, got {self.version!r}")
        if self.answer_rule not in ANSWER_RULES:
            raise ValueError(f"answer_rule must be one of {ANSWER_RULES}, got {self.answer_rule!r}")
        object.__setattr__(self, "grey", (float(self.grey[0]), float(self.grey[1])))
        if not 0.0 <= self.grey[0] <= self.grey[1] <= 1.0:
            raise ValueError("grey must be (lo, hi) inside [0, 1]")
        if not 0.0 < self.theta_floor <= self.theta_ceil:
            raise ValueError("theta bounds need 0 < floor <= ceil")
        if self.verify_per_spot_max is not None and self.verify_per_spot_max < 1:
            raise ValueError("verify_per_spot_max must be >= 1 or None")
        if self.verify_share_max is not None and not 0.0 < self.verify_share_max <= 1.0:
            raise ValueError("verify_share_max must be in (0, 1] or None")
        if self.card_min_resolved < 1:
            raise ValueError("card_min_resolved must be >= 1")

    # --- the two presets ---------------------------------------------------------------------------------------------
    @classmethod
    def v42(cls) -> "AntRules":
        """The v4.2 ant, exactly. Every field is its default."""
        return cls()

    @classmethod
    def v43(cls) -> "AntRules":
        """The v4.3 Step A fixed ant."""
        return cls(version="v43", use_combined=True, release_c=0.6, explore_release_c=0.6, explore_scan_min=0.5,
                   washout_release_c=0.65, tumour_here_c=0.8, tumour_here_scan_max=0.5, tumour_here_any_c=None,
                   not_tumour_c=0.2, not_tumour_scan_min=0.5, grey=(0.35, 0.65), verify_per_spot_max=2,
                   verify_share_max=0.25, theta_step_up=0.05, theta_step_down=0.05, theta_floor=0.1, theta_ceil=1.0,
                   resolved_only=True, fold_drop_c=0.6, fold_can_permit=False, answer_rule="v43", card_min_resolved=3)

    # --- (de)serialisation -------------------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["grey"] = list(self.grey)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AntRules":
        """``{"version": "v43"}`` is the v4.3 preset; any other key overrides one constant of the preset named by ``version``
        (``v42`` when absent). Unknown keys are an error."""
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown ant_rules keys: {sorted(extra)}")
        version = d.get("version", "v42")
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
        base = cls.v43() if version == "v43" else cls.v42()
        over = {k: (tuple(v) if k == "grey" else v) for k, v in d.items()}
        return replace(base, **over)

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    # --- derived -----------------------------------------------------------------------------------------------------
    def threshold_kwargs(self) -> Dict[str, float]:
        return {"step_up": self.theta_step_up, "step_down": self.theta_step_down, "floor": self.theta_floor,
                "ceil": self.theta_ceil}
