"""Worker override rate (pre-registered diagnostic, decision 2026-10-13-conformity-label-sprime-resolved, rule e).

The share of decisions where the worker's chosen target, resolved to a region, differs from the queen's assignment
for that squad at that time. Labels are resolved before they are compared (rule a): a candidate letter is its region;
``S`` (stay) is the squad's *current* target's region, so a worker that keeps the queen's target did not override it;
``X`` (explore) is its own region, ``"X"``, which never equals an assignment.

Resolution is delegated to the canonical resolver ``v41_resolve`` (S15); this module only maps its keys back to the
override helper's region ids (hold -> None, explore -> "X").
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from antelligence.experiments.v41_resolve import (EXPLORE_KEY, HOLD_KEY, ResolveError, current_key, region_key,
                                                  resolve_label)

STAY, EXPLORE = "S", "X"
Region = Union[int, str]   # a perceived region id, or "X" for explore


def resolve_target_region(label: str, current_label: Optional[str], candidates: Sequence[Mapping[str, Any]]
                          ) -> Optional[Region]:
    """Region a target label refers to, or None when it refers to no region (stay with no target, unknown letter).

    ``current_label`` is the squad's current target label as the world reports it (a candidate letter, ``S`` or None);
    ``candidates`` are the squad's observed candidates (``label`` and ``region_id``)."""
    by_label = {str(c["label"]): int(c["region_id"]) for c in candidates}
    cur_region = by_label.get(str(current_label)) if current_label not in (None, STAY, EXPLORE) else None
    current = current_key(None, cur_region)   # unchanged S16 rule: S while exploring/untargeted refers to no region
    try:
        key = resolve_label(label, {l: region_key(r) for l, r in by_label.items()}, current)
    except ResolveError:
        return None
    if key == EXPLORE_KEY:
        return EXPLORE
    return None if key == HOLD_KEY else int(key[1:])


def queen_region(route: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The queen's assignment: the first region of the squad's latest route (None when it has none)."""
    ids = (route or {}).get("region_ids") or []
    return int(ids[0]) if ids else None


def is_override(worker_label: str, current_label: Optional[str], candidates: Sequence[Mapping[str, Any]],
                assigned_region: Optional[int]) -> Optional[bool]:
    """True when the worker's resolved target differs from the queen's assignment; None when there is no assignment
    (the decision is not comparable and is left out of the rate)."""
    if assigned_region is None:
        return None
    return resolve_target_region(worker_label, current_label, candidates) != assigned_region


def override_rate(count: int, comparable: int) -> Optional[float]:
    return (count / comparable) if comparable else None


def pooled_override(cells: Sequence[Mapping[str, Any]], arm: str) -> Tuple[int, int, Optional[float]]:
    """(overrides, comparable decisions, rate) pooled over an arm's runs: the decision is the unit of the rate."""
    rows = [c for c in cells if c["arm"] == arm and c.get("override_decisions")]
    n = sum(int(c["override_decisions"]) for c in rows)
    k = sum(int(c.get("override_count") or 0) for c in rows)
    return k, n, override_rate(k, n)
