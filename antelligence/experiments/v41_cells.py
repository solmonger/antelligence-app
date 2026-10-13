"""v4.1 per-run cell schema and validator (goal 4.7). Documented in docs/v4.1/CELLS.md."""

from __future__ import annotations

import re
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple

ARMS = ("S", "S-prime", "SW-B", "SW-noBoard", "FW", "SW-D", "SW-noQueen", "SW-staticQueen", "SW-Bonsai",
        "SW-Byz-local", "SW-Byz-chain", "SW-B-missingFLAIR")
SPLITS = ("dev", "eval")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Field(NamedTuple):
    name: str
    kind: str  # str | int | float | frac | hex64
    nullable: bool
    doc: str


FIELDS: Tuple[Field, ...] = (
    Field("run_id", "hex64", False, "sha256(JCS(run header)), INTERFACES 6"),
    Field("arm", "str", False, "one of ARMS"),
    Field("case", "str", False, "MSD case id (never goes on-chain)"),
    Field("seed", "int", False, "run seed"),
    Field("split", "str", False, "dev or eval"),
    Field("wall_s", "float", False, "wall time of the run, seconds"),
    Field("ticks_to_50", "int", True, "ticks until 50% of the final score; null = never reached"),
    Field("seconds_to_50", "float", True, "wall seconds until 50% of the final score; null = never reached"),
    Field("queen_calls", "int", False, "queen model calls"),
    Field("board_size", "int", False, "entries on the board at run end"),
    Field("anchors", "int", False, "anchor points posted (INTERFACES 3)"),
    Field("visibility_latency_ms", "int", True, "mode D: write to visible latency; null outside mode D"),
    Field("coverage", "frac", False, "primary outcome: fraction of tumour region covered, in [0,1]; 1 point = 0.01"),
    Field("off_target", "frac", False, "fraction of drug released off target"),
    Field("defer_rate", "frac", False, "fraction of decisions deferred to the slow path"),
    Field("citations", "int", False, "cited board entries"),
    Field("unread_frac", "frac", True, "fraction of briefed entries not read; null for arms without a board"),
    Field("brief_age_median", "float", True, "median age in ticks of briefed entries; null without a board"),
    Field("router_top_share", "frac", True, "share of routed calls going to the most used model; null without router"),
    Field("perception_ece", "frac", True, "perception expected calibration error; null if not measured"),
    Field("cost_usd", "float", False, "paid API spend of the run, dollars"),
    Field("tokens_in", "int", False, "model tokens in, all paid and pod calls of the run"),
    Field("tokens_out", "int", False, "model tokens out"),
    Field("pod_minutes", "float", False, "pod minutes attributed to the run"),
    Field("gas", "int", True, "gas used on chain 300 by the run; null when nothing was posted on-chain"),
    Field("alarms", "int", False, "alarm entries written (defer-rate, timeout, identity, other)"),
    Field("audit_conformal_coverage", "frac", True, "conformal coverage achieved on the 10% audit sample; null if no audit sample"),
    Field("citation_validity", "frac", True, "fraction of cited entry ids that exist on the board and precede the decision"),
    Field("uncited_decisions", "int", False, "decisions that cite no entry"),
    Field("perception_dice", "frac", True, "perception Dice against the truth mask; null if not measured"),
    Field("locallook_ece", "frac", True, "d1 local-look ECE; null if not measured"),
    Field("truth_outside_crop", "frac", True, "share of truth tumour voxels outside the perception crop"),
)
FIELD_NAMES = tuple(f.name for f in FIELDS)
_OPTIONAL = {"error"}


def _ok(f: Field, v: Any) -> Optional[str]:
    if v is None:
        return None if f.nullable else "must not be null"
    if f.kind == "str":
        return None if isinstance(v, str) and v else "must be a non-empty string"
    if f.kind == "hex64":
        return None if isinstance(v, str) and _HEX64.match(v) else "must be 64 lowercase hex chars"
    if isinstance(v, bool):
        return "must be a number, not bool"
    if f.kind == "int":
        return None if isinstance(v, int) else "must be an int"
    if not isinstance(v, (int, float)) or v != v:
        return "must be a finite number"
    if f.kind == "frac":
        return None if 0 <= v <= 1 else "must be in [0, 1]"
    return None if v >= 0 else "must be >= 0"


def validate_cell(cell: Mapping[str, Any]) -> List[str]:
    """Return a list of problems (empty = valid). Unknown keys are errors."""
    errs = []
    for f in FIELDS:
        if f.name not in cell:
            errs.append(f"{f.name}: missing")
            continue
        msg = _ok(f, cell[f.name])
        if msg:
            errs.append(f"{f.name}: {msg}")
    for k in cell:
        if k not in FIELD_NAMES and k not in _OPTIONAL:
            errs.append(f"{k}: unknown field")
    if cell.get("arm") not in ARMS and "arm" in cell:
        errs.append(f"arm: {cell.get('arm')!r} is not a v4.1 arm")
    if cell.get("split") not in SPLITS and "split" in cell:
        errs.append(f"split: {cell.get('split')!r} must be one of {SPLITS}")
    return errs


def validate_cells(cells: Sequence[Mapping[str, Any]]) -> List[str]:
    out = []
    seen = set()
    for i, c in enumerate(cells):
        out += [f"cell {i}: {e}" for e in validate_cell(c)]
        key = (c.get("arm"), c.get("case"), c.get("seed"))
        if key in seen:
            out.append(f"cell {i}: duplicate (arm, case, seed) {key}")
        seen.add(key)
    return out


def assert_valid(cells: Sequence[Mapping[str, Any]]) -> None:
    errs = validate_cells(cells)
    if errs:
        raise ValueError("invalid cells:\n" + "\n".join(errs[:20]))
