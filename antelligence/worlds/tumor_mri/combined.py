"""The v4.3 combined estimate ``c = P(tumour | scan p, close look)`` (design readings 2).

``c = sigmoid(a + b*L(s) + c*L(l) + d*L(s)*L(l))`` with ``L(x) = logit(clip(x, 0.001, 0.999))``, ``s`` the scan p at the
ant's own voxel and ``l`` its own close look. The coefficients come from a fitted file (workstream A1) with the schema::

    {"schema": "combined_v4_3", "features": ["1", "logit_scan", "logit_look", "logit_scan*logit_look"],
     "coef": [a, b, c, d], "clip": [0.001, 0.999], "fit": {...}}

The estimate takes exactly two numbers (the scan p and the look at the ant's own voxel). Nothing else may enter: the
information barrier depends on it. Pure arithmetic, no I/O beyond ``from_json``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COEFFICIENTS_PATH = REPO_ROOT / "docs" / "research" / "v4.3-20261015" / "combined" / "coefficients.json"
SCHEMA = "combined_v4_3"
FEATURES = ("1", "logit_scan", "logit_look", "logit_scan*logit_look")


class CombinedEstimateError(ValueError):
    pass


def _logit(x: float, lo: float, hi: float) -> float:
    x = min(max(float(x), lo), hi)
    return math.log(x / (1.0 - x))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class CombinedEstimate:
    """``c = CombinedEstimate(...)(scan_p, look)``. Immutable after construction."""

    def __init__(self, coef: Sequence[float], clip: Sequence[float] = (0.001, 0.999), fit: Optional[Mapping[str, Any]] = None) -> None:
        coef = [float(c) for c in coef]
        if len(coef) != 4:
            raise CombinedEstimateError(f"coef needs 4 numbers (a, b, c, d), got {len(coef)}")
        if not all(math.isfinite(c) for c in coef):
            raise CombinedEstimateError("coef must be finite")
        clip = [float(c) for c in clip]
        if len(clip) != 2 or not 0.0 < clip[0] < clip[1] < 1.0:
            raise CombinedEstimateError(f"clip must be [lo, hi] with 0 < lo < hi < 1, got {clip}")
        self._coef = tuple(coef)
        self._clip = (clip[0], clip[1])
        self._fit = dict(fit or {})

    @property
    def coef(self) -> tuple:
        return self._coef

    @property
    def clip(self) -> tuple:
        return self._clip

    def __call__(self, scan_p: float, look: float) -> float:
        lo, hi = self._clip
        ls, ll = _logit(scan_p, lo, hi), _logit(look, lo, hi)
        a, b, c, d = self._coef
        return _sigmoid(a + b * ls + c * ll + d * ls * ll)

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "features": list(FEATURES), "coef": list(self._coef), "clip": list(self._clip),
                "fit": dict(self._fit)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CombinedEstimate":
        if d.get("schema") != SCHEMA:
            raise CombinedEstimateError(f"schema must be {SCHEMA!r}, got {d.get('schema')!r}")
        if "features" in d and list(d["features"]) != list(FEATURES):
            raise CombinedEstimateError(f"features must be {list(FEATURES)}, got {list(d['features'])}")
        if "coef" not in d:
            raise CombinedEstimateError("missing coef")
        return cls(d["coef"], d.get("clip", (0.001, 0.999)), d.get("fit"))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "CombinedEstimate":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def sha(self) -> str:
        """sha256 of the canonical coefficient body (schema, features, coef, clip); the fit block is not part of it."""
        body = {"schema": SCHEMA, "features": list(FEATURES), "coef": list(self._coef), "clip": list(self._clip)}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
