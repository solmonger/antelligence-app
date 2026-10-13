"""World configuration (config/v4_1/world.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml

AMOUNT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)  # levels 1..5, share of the squad's remaining payload


@dataclass(frozen=True)
class WorldConfig:
    n_squads: int = 8
    ticks: int = 120
    decision_every: int = 5
    queen_every: int = 30  # queen hook at tick 0 and every `queen_every`
    total_drug: float = 1000.0  # placeholder until tuned (scripts/v41_tune_world.py); equal shares per squad
    dose_threshold: float = 5.0  # D_t, placeholder until tuned; coverage counts truth voxels with dose >= D_t
    release_sigma_mm: float = 2.0
    diffuse_sigma_vox: float = 1.0
    decay: float = 0.02
    entry_shell_mm: float = 4.0
    entry_min_dist_mm: float = 10.0
    perceived_threshold: float = 0.3
    visit_radius_mm: float = 6.0  # a squad within this of a region centroid has visited the region
    explore_unc_quantile: float = 0.95  # 'highest-uncertainty' voxels: unc >= this quantile of brain unc
    path_weight: float = 1.5  # weighted A*: path cost <= weight * optimal
    path_max_expand: int = 40000
    amount_fractions: Tuple[float, ...] = AMOUNT_FRACTIONS

    def __post_init__(self) -> None:
        if self.n_squads < 1 or self.ticks < 1 or self.decision_every < 1 or self.queen_every < 1:
            raise ValueError("counts must be positive")
        if self.total_drug <= 0 or self.dose_threshold <= 0:
            raise ValueError("total_drug and dose_threshold must be positive")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if len(self.amount_fractions) != 5:
            raise ValueError("five amount levels")
        object.__setattr__(self, "amount_fractions", tuple(float(f) for f in self.amount_fractions))

    @property
    def rounds(self) -> int:
        return self.ticks // self.decision_every

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["amount_fractions"] = list(self.amount_fractions)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldConfig":
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown world config keys: {sorted(extra)}")
        return cls(**{k: (tuple(v) if k == "amount_fractions" else v) for k, v in d.items()})

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "WorldConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data.get("world", data))

    def with_(self, **kw: Any) -> "WorldConfig":
        return replace(self, **kw)
