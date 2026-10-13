"""v4.1 harness config: load, validate, hash. Shared by the queen and the sandbox loop.

``config/v4_1/harness.yaml`` holds ``harness`` (the tunable values) and ``allowed_keys`` (JSON-pointer
prefixes the sandbox may edit). Validation is schema-strict so an edit can never smuggle in a value
outside the ranges goal 4.6 allows.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import yaml

from antelligence.kernel.canonical import canonical_json

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "v4_1" / "harness.yaml"
BRIEF_ENTRY_TYPES = ("knowledge", "evidence", "decision", "alarm")
BRIEF_COUNT = (6, 12)
CADENCE = (30, 40)
MAX_NOTES = 8
MAX_NOTE_CHARS = 200
MAX_TEMPLATE_CHARS = 280
TEMPLATE_FIELDS = {"objective_template": {"n_regions", "n_squads", "coverage_pct"}}


class HarnessError(ValueError):
    """The harness config violates its schema."""


def _num(v: Any, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def _template_ok(text: Any, fields: set) -> bool:
    """Plain ``{name}`` fields from ``fields`` only: no format spec, conversion, attribute/index access or positional field."""
    import string
    if not isinstance(text, str) or not text or len(text) > MAX_TEMPLATE_CHARS:
        return False
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError:
        return False
    for _, name, spec, conv in parsed:
        if name is None:
            continue
        if name not in fields or spec or conv is not None:
            return False
    return True


def validate_harness(h: Mapping[str, Any]) -> None:
    """Raise :class:`HarnessError` unless ``h`` (the ``harness`` section) is exactly well-formed."""
    if not isinstance(h, Mapping) or set(h) != {"router", "briefs", "queen", "playbook_notes"}:
        raise HarnessError("harness needs exactly router, briefs, queen, playbook_notes")
    r, b, q, notes = h["router"], h["briefs"], h["queen"], h["playbook_notes"]
    if not isinstance(r, Mapping) or set(r) != {"slm_conf_min", "fw_conf_min", "cost_penalty"}:
        raise HarnessError("router keys")
    if not (_num(r["slm_conf_min"], 0, 1) and _num(r["fw_conf_min"], 0, 1) and _num(r["cost_penalty"], 0, 10)):
        raise HarnessError("router thresholds out of range")
    if not isinstance(b, Mapping) or set(b) != {"count", "entry_types", "objective_template"}:
        raise HarnessError("briefs keys")
    lo, hi = BRIEF_COUNT
    if not isinstance(b["count"], int) or isinstance(b["count"], bool) or not lo <= b["count"] <= hi:
        raise HarnessError(f"briefs.count must be an int in {lo}..{hi}")
    types = b["entry_types"]
    if not isinstance(types, list) or not types or len(set(types)) != len(types) or any(t not in BRIEF_ENTRY_TYPES for t in types):
        raise HarnessError(f"briefs.entry_types must be a non-empty unique subset of {BRIEF_ENTRY_TYPES}")
    for key, fields in TEMPLATE_FIELDS.items():
        if not _template_ok(b[key], fields):
            raise HarnessError(f"briefs.{key} must be a short template using only {sorted(fields)}")
    if not isinstance(q, Mapping) or set(q) != {"cadence"}:
        raise HarnessError("queen keys")
    c = q["cadence"]
    if not isinstance(c, int) or isinstance(c, bool) or not CADENCE[0] <= c <= CADENCE[1]:
        raise HarnessError(f"queen.cadence must be an int in {CADENCE[0]}..{CADENCE[1]}")
    if not isinstance(notes, list) or len(notes) > MAX_NOTES or any(not isinstance(n, str) or not n or len(n) > MAX_NOTE_CHARS for n in notes):
        raise HarnessError(f"playbook_notes: at most {MAX_NOTES} strings of at most {MAX_NOTE_CHARS} chars")


MAX_YAML_BYTES = 64 * 1024


class _NoAliasLoader(yaml.SafeLoader):
    """SafeLoader that refuses anchors and aliases (no shared or recursive structures)."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            raise HarnessError("YAML aliases are not allowed")
        if getattr(self.peek_event(), "anchor", None) is not None:
            raise HarnessError("YAML anchors are not allowed")
        return super().compose_node(parent, index)


def load_yaml_text(text: str) -> Any:
    if len(text.encode("utf-8")) > MAX_YAML_BYTES:
        raise HarnessError(f"harness YAML over {MAX_YAML_BYTES} bytes")
    return yaml.load(text, Loader=_NoAliasLoader)  # noqa: S506 - SafeLoader subclass


def load_harness_doc(path: Union[str, Path] = DEFAULT_PATH) -> Dict[str, Any]:
    p = Path(path)
    if p.stat().st_size > MAX_YAML_BYTES:
        raise HarnessError(f"harness YAML over {MAX_YAML_BYTES} bytes")
    doc = load_yaml_text(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"harness", "allowed_keys"}:
        raise HarnessError("harness.yaml needs exactly harness and allowed_keys")
    validate_harness(doc["harness"])
    keys = doc["allowed_keys"]
    if not isinstance(keys, list) or not all(isinstance(k, str) and k.startswith("/") for k in keys):
        raise HarnessError("allowed_keys must be a list of JSON pointers")
    return doc


def load_harness(path: Union[str, Path] = DEFAULT_PATH) -> Dict[str, Any]:
    """The base harness (static queen): a fresh deep copy of the ``harness`` section."""
    return copy.deepcopy(load_harness_doc(path)["harness"])


def load_allowed_keys(path: Union[str, Path] = DEFAULT_PATH) -> List[str]:
    return list(load_harness_doc(path)["allowed_keys"])


def harness_sha256(h: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(h)).encode("utf-8")).hexdigest()
