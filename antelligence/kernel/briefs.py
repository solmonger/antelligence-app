"""Brief assembly for v4.1 squads: what a squad reads, and the receipt of what it read.

``build_brief`` picks 6-12 board entries (or all of them when fewer exist), renders them under a
token cap, and returns the exact entry ids in brief order (the read receipt that goes into
``decision.brief_ids``). ``build_brief_board_off`` is the ablation: the squad sees only the task,
perception's region signals and its own queen assignment, never other squads' evidence, decisions
or local looks. Citation checks and the board-use metrics live here too.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from antelligence.kernel.board import Board, Entry
from antelligence.kernel.canonical import jcs_json

SCHEMA_TEXT = (
    "Answer three questions: target (one of the candidate labels, S to stay, X to skip), "
    "release (yes or no), amount (level 1-5). Give a probability for every option. "
    "Cite by full entry id only entries from this brief."
)
BOUNDARIES_TEXT = (
    "Use only what is on this brief. Do not assume facts about other squads or hidden ground truth. "
    "An id not listed here is not citable."
)
DEFAULT_CAP = (6, 12)
DEFAULT_MAX_TOKENS = 1500


class BriefError(ValueError):
    """A brief cannot be built within its rules."""


def estimate_tokens(text: str, tokenizer: Optional[Callable[[str], int]] = None) -> int:
    """Token count: the supplied tokenizer, else the chars/4 rule (rounded up)."""
    if tokenizer is not None:
        return int(tokenizer(text))
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class Brief:
    squad: int
    objective: str
    schema: str
    boundaries: str
    entries: Tuple[Entry, ...]
    brief_ids: Tuple[str, ...]   # read receipt, in brief order
    text: str
    tokens: int


def _render_entry(e: Entry) -> str:
    return f"[{e.id}] t={e.tick} {e.type} by {e.author['agent']}: {jcs_json(e.payload)}"


def _render(objective: str, assignment: str, entries: Sequence[Entry]) -> str:
    lines = [f"OBJECTIVE: {objective}", f"ASSIGNMENT: {assignment}", f"SCHEMA: {SCHEMA_TEXT}", f"BOUNDARIES: {BOUNDARIES_TEXT}", "ENTRIES:"]
    lines += [_render_entry(e) for e in entries]
    return "\n".join(lines)


def _route_parts(route_entry: Any) -> Tuple[Mapping[str, Any], Optional[int]]:
    """A route may be an appended Entry or a bare payload dict (a draft, not yet on the board)."""
    if isinstance(route_entry, Entry):
        if route_entry.type != "route":
            raise BriefError("route_entry is not a route entry")
        return route_entry.payload, route_entry.seq
    if isinstance(route_entry, Mapping):
        return route_entry, None
    raise BriefError("route_entry must be a route Entry or a route payload dict")


def _assignment(payload: Mapping[str, Any]) -> str:
    return f"squad {payload['squad']} regions {list(payload['region_ids'])} priority {payload['priority']}"


def _objective(board: Board, upto: Optional[int]) -> Tuple[str, Optional[Entry]]:
    tasks = [e for e in board.by_type("task") if upto is None or e.seq <= upto]
    if not tasks:
        return "(no task on the board)", None
    return tasks[-1].payload["objective"], tasks[-1]


def _finish(
    squad: int, objective: str, assignment: str, chosen: List[Entry], max_tokens: int,
    tokenizer: Optional[Callable[[str], int]], min_keep: int,
) -> Brief:
    """Drop lowest-priority entries (list tail) until under the token cap, never below ``min_keep``."""
    while True:
        text = _render(objective, assignment, chosen)
        tokens = estimate_tokens(text, tokenizer)
        if tokens <= max_tokens:
            break
        if len(chosen) <= min_keep:
            raise BriefError(f"brief needs {tokens} tokens with {len(chosen)} entries; cap is {max_tokens}")
        chosen = chosen[:-1]
    return Brief(
        squad=squad, objective=objective, schema=SCHEMA_TEXT, boundaries=BOUNDARIES_TEXT, entries=tuple(chosen),
        brief_ids=tuple(e.id for e in chosen), text=text, tokens=tokens,
    )


def build_brief(
    board: Board,
    squad: int,
    route_entry: Any,
    cap: Tuple[int, int] = DEFAULT_CAP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tokenizer: Optional[Callable[[str], int]] = None,
) -> Brief:
    """Board-on brief for ``squad``: task, its route, knowledge on its regions, then recent evidence.

    Priority order (also the drop order when over the token cap, last dropped first):
    task, the squad's route entry, knowledge for route regions in priority order, other knowledge,
    evidence/decisions/alarms newest first. Only entries up to the route's seq are visible.
    """
    lo, hi = cap
    if not 1 <= lo <= hi:
        raise BriefError("cap must satisfy 1 <= min <= max")
    payload, upto = _route_parts(route_entry)
    if payload["squad"] != squad:
        raise BriefError("route entry belongs to a different squad")
    visible = [e for e in board.entries() if upto is None or e.seq <= upto]
    objective, task = _objective(board, upto)

    ordered: List[Entry] = []
    seen = set()

    def take(e: Entry) -> None:
        if e.id not in seen:
            seen.add(e.id)
            ordered.append(e)

    if task is not None:
        take(task)
    if isinstance(route_entry, Entry):
        take(route_entry)
    knowledge = [e for e in visible if e.type == "knowledge"]
    for region in payload["region_ids"]:
        for e in reversed(knowledge):
            if e.payload["region_id"] == region:
                take(e)
    for e in reversed(knowledge):
        take(e)
    for e in reversed(visible):
        if e.type in ("evidence", "decision", "alarm"):
            take(e)
    chosen = ordered[:hi]
    min_keep = min(lo, len(chosen))
    return _finish(squad, objective, _assignment(payload), chosen, max_tokens, tokenizer, min_keep)


def build_brief_board_off(
    board: Board,
    squad: int,
    route_entry: Any,
    cap: Tuple[int, int] = DEFAULT_CAP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tokenizer: Optional[Callable[[str], int]] = None,
) -> Brief:
    """Board-off brief: task statement, perception's region signals, and this squad's own assignment.

    Never includes other squads' evidence, decisions, local looks (knowledge with source local_look),
    or other squads' routes.
    """
    _, hi = cap
    payload, upto = _route_parts(route_entry)
    if payload["squad"] != squad:
        raise BriefError("route entry belongs to a different squad")
    objective, task = _objective(board, upto)
    chosen: List[Entry] = []
    if task is not None:
        chosen.append(task)
    if isinstance(route_entry, Entry):
        chosen.append(route_entry)
    signals = [
        e for e in board.by_type("knowledge")
        if e.author["agent"] == "perception" and e.payload["source"] == "perception"
        and (upto is None or e.seq <= upto)
    ]
    wanted = list(payload["region_ids"])
    signals.sort(key=lambda e: (wanted.index(e.payload["region_id"]) if e.payload["region_id"] in wanted else len(wanted), -e.seq))
    chosen += signals
    chosen = chosen[:hi]
    return _finish(squad, objective, _assignment(payload), chosen, max_tokens, tokenizer, min_keep=min(1, len(chosen)))


# ---------------------------------------------------------------------------
# Citations and metrics
# ---------------------------------------------------------------------------

class CitationCheck(NamedTuple):
    valid: List[str]
    invalid: List[str]


def check_citations(cites: Iterable[str], brief_ids: Iterable[str]) -> CitationCheck:
    """A citation is valid only if the id was in the brief the worker read."""
    allowed = set(brief_ids)
    valid: List[str] = []
    invalid: List[str] = []
    for cite in cites:
        (valid if cite in allowed else invalid).append(cite)
    return CitationCheck(valid, invalid)


def citation_validity_rate(board: Board) -> Optional[float]:
    """Valid cites / all cites over decision entries, judged against each decision's own read receipt."""
    total = good = 0
    for e in board.by_type("decision"):
        result = check_citations(e.cites, e.payload["brief_ids"])
        good += len(result.valid)
        total += len(result.valid) + len(result.invalid)
    return None if total == 0 else good / total


def uncited_decisions(board: Board) -> List[str]:
    """Ids of decisions that cite nothing."""
    return [e.id for e in board.by_type("decision") if not e.cites]


_READABLE = ("task", "knowledge", "evidence", "route", "decision")


def never_read_share(board: Board) -> Optional[float]:
    """Share of briefable entries (task/knowledge/evidence/route/decision) that appear in no read receipt."""
    read = {i for d in board.by_type("decision") for i in d.payload["brief_ids"]}
    pool = [e for e in board.entries() if e.type in _READABLE]
    return None if not pool else sum(1 for e in pool if e.id not in read) / len(pool)


def median_entry_age_in_briefs(board: Board) -> Optional[float]:
    """Median of (decision tick - entry tick) over every id in every decision's read receipt."""
    ages = [d.tick - board.get(i).tick for d in board.by_type("decision") for i in d.payload["brief_ids"] if i in board]
    return None if not ages else float(statistics.median(ages))
