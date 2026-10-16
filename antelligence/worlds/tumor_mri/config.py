"""World configuration (config/v4_1/world.yaml; v4.2 adds the ``facts`` block, config/v4_2/world.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import yaml

AMOUNT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)  # levels 1..5, share of the squad's remaining payload
DRUG_MODES = ("fixed", "per_perceived_volume")
F2_MODELS = ("v42", "foraging")
UPTAKE_LEVEL_NAMES = ("low", "mid", "high")     # the preset values live in ``foraging.UPTAKE_LEVELS`` (named assumptions, never tuned)
PHYSICS_MODELS = ("legacy", "units")            # v4.31: "units" = the real-units world (physical.py); "legacy" is every earlier config, byte for byte
K_SETS = ("low", "high")                        # the carried clearance pair (physical.K_CLEAR_LOW_S, K_CLEAR_HIGH_S)
# Keys a units config must not set: the loader refuses them (they are derived from the physical block, or do not exist in real units).
UNITS_REFUSED_KEYS = ("drug_scale_k", "total_drug", "drug_mode", "release_sigma_mm", "diffuse_sigma_vox", "decay", "dose_threshold")


def perceived_volume_vox(p_tumour: Any) -> float:
    """V_perceived: sum of calibrated p_tumour over the crop, in voxels."""
    return float(np.asarray(p_tumour, dtype=np.float64).sum())


@dataclass(frozen=True)
class FactsConfig:
    """Hidden local facts F1-F3 (v4.2 goal section 5, INTERFACES 11). Every flag defaults off; all off == v4.1."""
    f1_simlook: bool = False
    f1_source: str = "boards"   # boards | local_d1 | log2000 (recorded, never inferred)
    f2: bool = False
    f2_source: str = "A"        # A (TC & ~ET from the store) | B (seeded smooth field)
    u_core: float = 1.0         # {0.3, 0.5, 0.7} when on
    f2_share: float = 0.0       # {0.2, 0.35}, Source B only
    f3: bool = False
    f3_decay: float = 0.0       # {0.04, 0.08}
    f3_share: float = 0.0       # {0.1, 0.2}
    # v4.3 (B1). Both are omitted from ``to_dict`` at their defaults, so a v4.2 config keeps its dict, hash and record.
    f2_model: str = "v42"       # v42 (the two-level F2 above) | foraging (the v4.3 uptake field, ``foraging.py``)
    uptake_level: str = "mid"   # low | mid | high; read only when f2 is on and f2_model is foraging

    def __post_init__(self) -> None:
        if self.f2_model not in F2_MODELS:
            raise ValueError(f"f2_model must be one of {F2_MODELS}, got {self.f2_model!r}")
        if self.uptake_level not in UPTAKE_LEVEL_NAMES:
            raise ValueError(f"uptake_level must be one of {UPTAKE_LEVEL_NAMES}, got {self.uptake_level!r}")
        if self.f1_source not in ("boards", "local_d1", "log2000"):
            raise ValueError("f1_source must be boards | local_d1 | log2000")
        if self.f2_source not in ("A", "B"):
            raise ValueError("f2_source must be A or B")
        if not 0.0 < self.u_core <= 1.0:
            raise ValueError("u_core must be in (0, 1]")
        if not 0.0 <= self.f2_share <= 1.0 or not 0.0 <= self.f3_share <= 1.0:
            raise ValueError("shares must be in [0, 1]")
        if self.f2 and self.f2_source == "B" and self.f2_share <= 0.0:
            raise ValueError("Source B needs f2_share > 0")
        if not 0.0 <= self.f3_decay < 1.0:
            raise ValueError("f3_decay must be in [0, 1)")
        if self.f3 and (self.f3_decay <= 0.0 or self.f3_share <= 0.0):
            raise ValueError("f3 on needs f3_decay > 0 and f3_share > 0")

    @property
    def any_on(self) -> bool:
        return bool(self.f1_simlook or self.f2 or self.f3)

    @property
    def foraging(self) -> bool:
        """True when F2 is on and uses the v4.3 foraging uptake field."""
        return bool(self.f2 and self.f2_model == "foraging")

    def to_dict(self) -> Dict[str, Any]:
        """All fields, except that ``f2_model`` / ``uptake_level`` are left out at their defaults (v4.2 dict unchanged).
        A foraging config always states its level, so a header names the field it ran."""
        d = asdict(self)
        if self.f2_model == "v42":
            del d["f2_model"]
            if self.uptake_level == "mid":
                del d["uptake_level"]
        return d


@dataclass(frozen=True)
class PhysicsConfig:
    """v4.31 real-units block (read only when ``physics_model`` is ``units``). Nothing here is a tuned result.

    * ``k_set``: which end of the carried clearance pair (low 1.2e-4 /s, high 6.0e-4 /s).
    * ``tick_seconds``: the declared tick (default 1800 s).
    * ``n_sub``: Gauss-Legendre sub-samples per axis for the voxel-averaged exposure kernel.
    * ``kernel_tol_umol_h_L``: the table radius is chosen so the whole budget released at one point adds less than this beyond it.
    * ``policy_threshold_umol_h_L``: the exposure level the SCRIPTS (S, S') plan against. Plumbing for the policies, not a result:
      every coverage figure is reported as a curve over thresholds, never at this one number alone.
    """
    k_set: str = "low"
    tick_seconds: float = 1800.0
    n_sub: int = 5
    kernel_tol_umol_h_L: float = 0.1
    policy_threshold_umol_h_L: float = 0.0

    def __post_init__(self) -> None:
        if self.k_set not in K_SETS:
            raise ValueError(f"physics.k_set must be one of {K_SETS}, got {self.k_set!r}")
        if not self.tick_seconds > 0 or self.n_sub < 2 or not self.kernel_tol_umol_h_L > 0:
            raise ValueError("physics: tick_seconds > 0, n_sub >= 2, kernel_tol_umol_h_L > 0")
        if self.policy_threshold_umol_h_L < 0:
            raise ValueError("physics.policy_threshold_umol_h_L must be >= 0")


@dataclass(frozen=True)
class WorldConfig:
    n_squads: int = 8
    ticks: int = 120
    decision_every: int = 5
    queen_every: int = 30  # queen hook at tick 0 and every `queen_every`
    total_drug: float = 1000.0  # drug_mode "fixed": the per-case budget (equal shares per squad)
    drug_mode: str = "fixed"  # "fixed" | "per_perceived_volume": total = drug_scale_k x V_perceived (perception only)
    drug_scale_k: float = 0.0  # drug per perceived tumour voxel; required (> 0) in per_perceived_volume mode
    dose_threshold: float = 5.0  # D_t, placeholder until tuned; coverage counts truth voxels with dose >= D_t
    release_sigma_mm: float = 2.0
    diffuse_sigma_vox: float = 1.0
    decay: float = 0.02
    entry_shell_mm: float = 4.0
    entry_min_dist_mm: float = 10.0
    visit_radius_mm: float = 6.0  # a squad within this of a region centroid has visited the region
    explore_unc_quantile: float = 0.95  # 'highest-uncertainty' voxels: unc >= this quantile of brain unc
    path_weight: float = 1.5  # weighted A*: path cost <= weight * optimal
    path_max_expand: int = 40000
    amount_fractions: Tuple[float, ...] = AMOUNT_FRACTIONS
    facts: FactsConfig = FactsConfig()
    # v4.31 (additive): both are left out of ``to_dict`` for a legacy config, so every v4.1-v4.3 dict, hash and record is unchanged.
    physics_model: str = "legacy"
    physics: PhysicsConfig = PhysicsConfig()

    def __post_init__(self) -> None:
        if self.physics_model not in PHYSICS_MODELS:
            raise ValueError(f"physics_model must be one of {PHYSICS_MODELS}, got {self.physics_model!r}")
        if self.physics_model == "units":
            self._init_units()
        if self.n_squads < 1 or self.ticks < 1 or self.decision_every < 1 or self.queen_every < 1:
            raise ValueError("counts must be positive")
        if self.total_drug <= 0 or self.dose_threshold <= 0:
            raise ValueError("total_drug and dose_threshold must be positive")
        if self.drug_mode not in DRUG_MODES:
            raise ValueError(f"drug_mode must be one of {DRUG_MODES}, got {self.drug_mode!r}")
        if self.drug_mode == "per_perceived_volume" and self.drug_scale_k <= 0:
            raise ValueError("drug_scale_k must be positive in per_perceived_volume mode")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if len(self.amount_fractions) != 5:
            raise ValueError("five amount levels")
        object.__setattr__(self, "amount_fractions", tuple(float(f) for f in self.amount_fractions))

    @property
    def rounds(self) -> int:
        return self.ticks // self.decision_every

    # --- v4.31 real units ------------------------------------------------------------------------------------------------

    @property
    def units(self) -> bool:
        return self.physics_model == "units"

    def physical_params(self):
        """The :class:`PhysicalParams` of a units config (derived; nothing is hand-copied). Raises for a legacy config."""
        if not self.units:
            raise ValueError("physical_params() needs physics_model: units")
        from antelligence.worlds.tumor_mri.physical import PhysicalParams
        return PhysicalParams.carmustine_wafer(self.physics.k_set, tick_seconds=self.physics.tick_seconds, ticks=self.ticks)

    def _init_units(self) -> None:
        """Real-units world: facts off, a positive policy threshold, and the legacy physics fields replaced by values DERIVED from
        the physical block (the budget is the fixed 61.6 mg cap, never scaled by perceived volume). They exist only so shared code
        that reads ``cfg.decay`` etc. keeps working; ``to_dict`` leaves them out."""
        if self.facts.any_on:
            raise ValueError("physics_model: units needs F1/F2/F3 off (Phase A changes physics, not hidden facts)")
        if self.physics.policy_threshold_umol_h_L <= 0:
            raise ValueError("physics.policy_threshold_umol_h_L must be > 0 in a units config (what the scripts plan against; results are curves)")
        p = self.physical_params()
        for name, val in (("drug_mode", "fixed"), ("total_drug", p.budget_mg), ("drug_scale_k", 0.0), ("release_sigma_mm", 0.0),
                          ("diffuse_sigma_vox", p.diffuse_sigma_vox), ("decay", p.decay),
                          ("dose_threshold", float(self.physics.policy_threshold_umol_h_L))):
            object.__setattr__(self, name, val)

    def resolve_total_drug(self, p_tumour: Any) -> float:
        """The case's total drug. ``fixed``: ``total_drug``. ``per_perceived_volume``: ``drug_scale_k`` x V_perceived, with
        V_perceived the sum of calibrated ``p_tumour`` over the crop (voxels). Perception only: truth is never an input."""
        if self.drug_mode == "fixed":
            return float(self.total_drug)
        return float(self.drug_scale_k) * perceived_volume_vox(p_tumour)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["amount_fractions"] = list(self.amount_fractions)
        if self.facts.any_on:  # the v4.1 config (and its hash) is unchanged when every fact is off
            d["facts"] = self.facts.to_dict()
        else:
            d.pop("facts", None)
        if self.units:  # v4.31: the legacy physics keys are derived, not configuration
            for k in UNITS_REFUSED_KEYS:
                d.pop(k, None)
            return d
        d.pop("physics_model", None)  # a legacy config's dict (and hash) is exactly what v4.1-v4.3 recorded
        d.pop("physics", None)
        if self.drug_mode == "per_perceived_volume":
            del d["total_drug"]  # the budget is per case (k x V_perceived); a fixed total would mislead
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldConfig":
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown world config keys: {sorted(extra)}")
        if d.get("drug_mode") == "per_perceived_volume" and "total_drug" in d:
            raise ValueError("total_drug must not be set when drug_mode is per_perceived_volume (the budget is k x V_perceived)")
        model = d.get("physics_model", "legacy")
        if model == "units":
            bad = sorted(set(d) & set(UNITS_REFUSED_KEYS))
            if bad:
                raise ValueError(f"physics_model: units refuses {bad}: the budget is the fixed 61.6 mg cap (never scaled by perceived volume, "
                                 "never tuned, so drug_scale_k does not exist here) and spread, loss and release width are derived from the physical block")
        elif "physics" in d:
            raise ValueError("a physics block needs physics_model: units")
        def conv(k: str, v: Any) -> Any:
            if k == "amount_fractions":
                return tuple(v)
            if k == "facts":
                return v if isinstance(v, FactsConfig) else FactsConfig(**(v or {}))
            if k == "physics":
                return v if isinstance(v, PhysicsConfig) else PhysicsConfig(**(v or {}))
            return v

        return cls(**{k: conv(k, v) for k, v in d.items()})

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "WorldConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data.get("world", data))

    def with_(self, **kw: Any) -> "WorldConfig":
        return replace(self, **kw)
