"""Canonical JSON and content hashing shared by every kernel record."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class CanonicalError(ValueError):
    """A value cannot be represented as canonical JSON."""


def _check(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalError(f"{path}: non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError(f"{path}: object keys must be strings")
            _check(item, f"{path}.{key}")
        return
    raise CanonicalError(f"{path}: unsupported type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, finite numbers only."""
    _check(value, "$")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def content_hash(value: Any) -> str:
    """sha256 hex digest of the canonical form of ``value``."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def plain(value: Any) -> Any:
    """Return a detached, JSON-plain deep copy (tuples become lists)."""
    return json.loads(canonical_json(value))


# ---------------------------------------------------------------------------
# Strict RFC 8785 (JCS) for the v4.1 board: integers only, UTF-16 key order.
# ---------------------------------------------------------------------------

MAX_SAFE_INT = (1 << 53) - 1


def _jcs(value: Any, path: str, out: list) -> None:
    if value is None:
        out.append("null")
    elif isinstance(value, bool):
        out.append("true" if value else "false")
    elif isinstance(value, int):
        if not -MAX_SAFE_INT <= value <= MAX_SAFE_INT:
            raise CanonicalError(f"{path}: integer outside +-(2^53-1)")
        out.append(str(value))
    elif isinstance(value, float):
        raise CanonicalError(f"{path}: floats are not allowed in board entries")
    elif isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise CanonicalError(f"{path}: string is not valid unicode") from exc
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _jcs(item, f"{path}[{index}]", out)
        out.append("]")
    elif isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise CanonicalError(f"{path}: object keys must be strings")
        out.append("{")
        for index, key in enumerate(sorted(value, key=lambda k: k.encode("utf-16-be", "surrogatepass"))):
            if index:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _jcs(value[key], f"{path}.{key}", out)
        out.append("}")
    else:
        raise CanonicalError(f"{path}: unsupported type {type(value).__name__}")


def jcs_json(value: Any) -> str:
    """RFC 8785 canonical JSON restricted to integers (no floats), as v4.1 requires."""
    out: list = []
    _jcs(value, "$", out)
    return "".join(out)


def jcs_bytes(value: Any) -> bytes:
    return jcs_json(value).encode("utf-8")
