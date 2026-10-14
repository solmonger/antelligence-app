"""World configuration (config/v4_1/world.yaml; v4.2 adds the ``facts`` block, config/v4_2/world.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import yaml

AMOUNT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)  # levels 1..5, share of the squad's remaining payload
DRUG_MODES = ("fixed", "per_perceived_volume")


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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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
            d["facts"] = asdict(self.facts)
        else:
            d.pop("facts", None)
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
        def conv(k: str, v: Any) -> Any:
            if k == "amount_fractions":
                return tuple(v)
            if k == "facts":
                return v if isinstance(v, FactsConfig) else FactsConfig(**(v or {}))
            return v

        return cls(**{k: conv(k, v) for k, v in d.items()})

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "WorldConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data.get("world", data))

    def with_(self, **kw: Any) -> "WorldConfig":
        return replace(self, **kw)
