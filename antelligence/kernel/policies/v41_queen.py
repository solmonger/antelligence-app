"""V41Queen: the thin v4.1 orchestrator (goal 4.6).

Her only tools are board writes (``task``, ``route``, ``alarm`` entries; see :class:`QueenTools`). She
never decides a squad's action: she advises which regions each squad should look at, which (at most six)
regions appear as candidates A..F, and the brief each squad reads. Every squad can always pick ``S`` or
``X`` ("advise, never gate"); the squad count is fixed at construction and a reply cannot change it.

Inputs (never truth): perceived regions (INTERFACES ``knowledge`` payload fields, integers), squad
positions and payloads, a coverage ESTIMATE from p_tumour-weighted dose, and, in the board-on arm only,
the latest ``evidence`` entries. Board-off gets no squad evidence and builds board-off briefs.

Ledgers follow Magentic-One: a task ledger (facts, guesses, plan) and a progress ledger (per-squad
status, coverage estimates, stalls). Two consecutive queen steps without estimated progress force a
re-plan (the next call carries a re-plan directive). Caps per run: 6 calls including re-plans, input
at most 6k tokens (truncated deterministically), output at most 800 tokens. An invalid reply, a failed
call or a spent call cap falls back to S-rule assignments (each squad stays on its previous route, else
its nearest perceived region) and writes an ``alarm`` entry.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.kernel.board import Board, Entry
from antelligence.kernel.briefs import Brief, build_brief, build_brief_board_off, estimate_tokens
from antelligence.kernel.canonical import canonical_json
from antelligence.kernel.decision import DecisionAbort
from antelligence.kernel.policies.v41_harness import load_harness, validate_harness
from antelligence.providers.base import ChatRequest, Provider

MAX_CALLS = 6
MAX_INPUT_TOKENS = 6000
MAX_OUTPUT_TOKENS = 800
MAX_CANDIDATES = 6
LABELS = "ABCDEF"
ALWAYS_AVAILABLE = ("S", "X")
STALL_EPS_PPM = 5000          # estimated coverage must rise by 0.5 points to count as progress
STALL_LIMIT = 2
FALLBACK_PRIORITY = 3
MAX_EVIDENCE_PER_SQUAD = 2
DEFAULT_THINKING_OFF: Mapping[str, Any] = {"reasoning": {"enabled": False}}  # live-probed 2026-10-13: 0 reasoning tokens on Nous
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

SYSTEM = (
    "You are the queen of a squad swarm treating a brain tumour. You cannot see the tumour, only the "
    "perceived regions, where the squads are, and a coverage estimate. You never choose a squad's action; "
    "you advise which regions each squad should look at and which regions to offer as candidates. "
    "Reply with one JSON object and nothing else: "
    '{"plan": "<=200 chars", "guesses": ["<=3 short strings"], '
    '"routes": [{"squad": <int>, "region_ids": [<region_id>, ...], "priority": <1-5>}], '
    '"candidates": [<region_id>, ... at most 6]}. Use only the squad numbers and region ids given.'
)


class QueenError(RuntimeError):
    """A queen rule was broken (a tool misuse or an impossible input)."""


class QueenReplyError(ValueError):
    """The reply does not satisfy the schema."""


class QueenToolError(QueenError):
    """The queen tried a board write outside her tool set."""


@dataclass(frozen=True)
class SquadState:
    """What the queen may know about a squad: where it is and how much drug it has left."""

    squad: int
    pos_vox: Tuple[int, int, int]
    payload_ppm: int = 1_000_000


class QueenTools:
    """The queen's whole tool set: board writes of ``task``, ``route`` and ``alarm`` entries as ``queen``."""

    ALLOWED = ("task", "route", "alarm")

    def __init__(self, board: Board) -> None:
        self._board = board

    def write(self, entry_type: str, payload: Mapping[str, Any], tick: int, cites: Sequence[str] = ()) -> Entry:
        if entry_type not in self.ALLOWED:
            raise QueenToolError(f"the queen may write only {self.ALLOWED}, not {entry_type!r}")
        return self._board.append(entry_type, "queen", payload, tick, cites)


@dataclass
class TaskLedger:
    facts: List[str] = field(default_factory=list)
    guesses: List[str] = field(default_factory=list)
    plan: str = ""


@dataclass
class ProgressLedger:
    coverage_ppm: List[int] = field(default_factory=list)   # estimate at each queen step
    stalls: int = 0
    replans: int = 0
    status: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class QueenStep:
    tick: int
    task: Entry
    routes: Tuple[Entry, ...]
    briefs: Mapping[int, Brief]
    candidate_region_ids: Tuple[int, ...]   # label order A..F
    options: Tuple[str, ...]                # candidate labels plus S and X, always
    fell_back: bool
    replan: bool
    called: bool
    calls_used: int
    alarm: Optional[Entry] = None


def coverage_estimate_ppm(regions: Sequence[Mapping[str, Any]], dose_fraction: Mapping[int, float]) -> int:
    """p_tumour-weighted dose coverage estimate in ppm. ``dose_fraction``: region_id -> dose / threshold (clipped to 0..1).

    Weights are p_tumour x volume from perception. The truth mask is never an input.
    """
    total = sum(r["p_tumour_ppm"] * r["volume_mm3"] for r in regions)
    if total <= 0:
        return 0
    got = sum(r["p_tumour_ppm"] * r["volume_mm3"] * min(1.0, max(0.0, float(dose_fraction.get(r["region_id"], 0.0)))) for r in regions)
    return int(round(1_000_000 * got / total))


def _weight(r: Mapping[str, Any]) -> int:
    return int(r["p_tumour_ppm"]) * int(r["volume_mm3"])


def rank_regions(regions: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Deterministic order: p x volume descending, region_id ascending."""
    return sorted(regions, key=lambda r: (-_weight(r), int(r["region_id"])))


def _view_region(r: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: r[k] for k in ("region_id", "centroid_dmm", "cls", "p_tumour_ppm", "volume_mm3", "unc_ppm") if k in r}


class V41Queen:
    def __init__(
        self,
        provider: Provider,
        model: str,
        board: Board,
        *,
        n_squads: int,
        harness: Optional[Mapping[str, Any]] = None,
        board_on: bool = True,
        thinking_params: Optional[Mapping[str, Any]] = DEFAULT_THINKING_OFF,
        max_calls: int = MAX_CALLS,
    ) -> None:
        if n_squads < 1:
            raise ValueError("n_squads must be positive")
        self.provider = provider
        self.model = model
        self.board = board
        self.tools = QueenTools(board)
        self.n_squads = int(n_squads)       # fixed for the run; no reply can change it
        self.harness = dict(harness) if harness is not None else load_harness()
        validate_harness(self.harness)
        self.board_on = bool(board_on)
        self.thinking_params = dict(thinking_params) if thinking_params else None
        self.max_calls = min(int(max_calls), MAX_CALLS)   # the cap can be tightened, never raised
        self.calls = 0
        self.task_ledger = TaskLedger()
        self.progress = ProgressLedger()
        self.previous_routes: Dict[int, Dict[str, Any]] = {}
        self.last_prompt: Optional[str] = None     # for audit and tests: exactly what the model saw
        self.steps = 0

    def describe(self) -> Dict[str, Any]:
        return {"policy": "v41_queen", "model": self.model, "provider": self.provider.describe(), "n_squads": self.n_squads,
                "board_on": self.board_on, "max_calls": self.max_calls, "thinking": self.thinking_params,
                "cadence": self.harness["queen"]["cadence"]}

    # -- prompt --------------------------------------------------------------------------------
    def _evidence(self) -> List[Dict[str, Any]]:
        if not self.board_on:
            return []
        out: List[Dict[str, Any]] = []
        per_squad: Dict[int, int] = {}
        for e in reversed(self.board.by_type("evidence")):
            s = e.payload["squad"]
            if per_squad.get(s, 0) >= MAX_EVIDENCE_PER_SQUAD:
                continue
            per_squad[s] = per_squad.get(s, 0) + 1
            p = e.payload
            out.append({"tick": e.tick, "squad": s, "pos_vox": p["pos_vox"], "look_p_ppm": p["look_p_ppm"],
                        "p_tumour_ppm": p["p_tumour_ppm"], "dose_ug": p["dose_ug"]})
        out.reverse()   # oldest first, so truncation drops the oldest from the front
        return out

    def _body(self, tick: int, regions: Sequence[Mapping[str, Any]], squads: Sequence[SquadState], coverage_ppm: int,
              replan: bool) -> Dict[str, Any]:
        ranked = rank_regions(regions)
        return {
            "tick": tick, "step": self.steps, "arm": "board_on" if self.board_on else "board_off", "replan": replan,
            "coverage_est_ppm": coverage_ppm, "stalls": self.progress.stalls,
            "task_ledger": {"facts": list(self.task_ledger.facts), "guesses": list(self.task_ledger.guesses), "plan": self.task_ledger.plan},
            "progress_ledger": {"coverage_est_ppm_history": list(self.progress.coverage_ppm),
                                "squad_status": [self.progress.status[s] for s in sorted(self.progress.status)]},
            "squads": [{"squad": s.squad, "pos_vox": list(s.pos_vox), "payload_ppm": s.payload_ppm} for s in sorted(squads, key=lambda s: s.squad)],
            "regions": [_view_region(r) for r in ranked],
            "evidence": self._evidence(),
            "previous_routes": [self.previous_routes[s] for s in sorted(self.previous_routes)],
            "playbook": list(self.harness["playbook_notes"]),
        }

    def _render(self, body: Mapping[str, Any]) -> str:
        return canonical_json(body)

    def _fit(self, body: Dict[str, Any]) -> str:
        """Truncate deterministically until system + user fit the input cap: oldest evidence, then playbook tail,
        guesses, history, previous routes, then lowest-ranked regions (never below one)."""
        def tokens() -> int:
            return estimate_tokens(SYSTEM) + estimate_tokens(self._render(body))
        stages = (
            lambda: body["evidence"].pop(0) if body["evidence"] else None,
            lambda: body["playbook"].pop() if body["playbook"] else None,
            lambda: body["task_ledger"]["guesses"].pop() if body["task_ledger"]["guesses"] else None,
            lambda: body["progress_ledger"]["coverage_est_ppm_history"].pop(0) if len(body["progress_ledger"]["coverage_est_ppm_history"]) > 2 else None,
            lambda: body["task_ledger"]["facts"].pop() if body["task_ledger"]["facts"] else None,
            lambda: body["previous_routes"].pop() if body["previous_routes"] else None,
            lambda: body["regions"].pop() if len(body["regions"]) > 1 else None,
        )
        for drop in stages:
            while tokens() > MAX_INPUT_TOKENS and drop() is not None:
                pass
        if tokens() > MAX_INPUT_TOKENS:
            raise QueenError(f"queen input cannot fit {MAX_INPUT_TOKENS} tokens")
        return self._render(body)

    def request(self, tick: int, regions: Sequence[Mapping[str, Any]], squads: Sequence[SquadState], coverage_ppm: int,
                replan: bool = False) -> ChatRequest:
        user = self._fit(self._body(tick, regions, squads, coverage_ppm, replan))
        self.last_prompt = user
        return ChatRequest(model=self.model, messages=({"role": "system", "content": SYSTEM}, {"role": "user", "content": user}),
                           max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0, seed=(tick + 7919 * self.steps) % 2_147_483_647,
                           response_format={"type": "json_object"}, extra_body=self.thinking_params)

    # -- reply ---------------------------------------------------------------------------------
    def parse(self, content: str, regions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        text = content.strip()
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1)
        try:
            reply = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QueenReplyError(f"not JSON ({exc.msg})") from exc
        if not isinstance(reply, dict) or not isinstance(reply.get("routes"), list):
            raise QueenReplyError("reply needs a 'routes' list")
        known = {int(r["region_id"]) for r in regions}
        routes: Dict[int, Dict[str, Any]] = {}
        for item in reply["routes"]:
            if not isinstance(item, dict):
                continue
            squad, ids, prio = item.get("squad"), item.get("region_ids"), item.get("priority", FALLBACK_PRIORITY)
            if not (isinstance(squad, int) and not isinstance(squad, bool) and 0 <= squad < self.n_squads and isinstance(ids, list)):
                continue   # unknown squads are ignored: the queen cannot add squads
            if not (isinstance(prio, int) and not isinstance(prio, bool) and 1 <= prio <= 5):
                prio = FALLBACK_PRIORITY
            keep = list(dict.fromkeys(i for i in ids if isinstance(i, int) and not isinstance(i, bool) and i in known))
            if keep and squad not in routes:
                routes[squad] = {"region_ids": keep, "priority": prio}
        if not routes:
            raise QueenReplyError("no valid route")
        cand_raw = reply.get("candidates")
        candidates = list(dict.fromkeys(i for i in cand_raw if isinstance(i, int) and not isinstance(i, bool) and i in known))[:MAX_CANDIDATES] \
            if isinstance(cand_raw, list) else []
        plan = reply.get("plan")
        guesses = reply.get("guesses")
        return {"routes": routes, "candidates": candidates,
                "plan": plan[:200] if isinstance(plan, str) else "",
                "guesses": [g[:120] for g in guesses[:3] if isinstance(g, str)] if isinstance(guesses, list) else []}

    # -- fallback ------------------------------------------------------------------------------
    def s_rule_routes(self, regions: Sequence[Mapping[str, Any]], squads: Sequence[SquadState]) -> Dict[int, Dict[str, Any]]:
        """Stay: keep the previous route; with none, the nearest perceived region (ties to lower region_id)."""
        out: Dict[int, Dict[str, Any]] = {}
        ranked = rank_regions(regions)
        for s in sorted(squads, key=lambda s: s.squad):
            if s.squad in self.previous_routes:
                prev = self.previous_routes[s.squad]
                out[s.squad] = {"region_ids": list(prev["region_ids"]), "priority": prev["priority"]}
                continue
            def dist(r: Mapping[str, Any]) -> Tuple[int, int]:
                return (sum((c // 20 - p) ** 2 for c, p in zip(r["centroid_dmm"], s.pos_vox)), int(r["region_id"]))
            out[s.squad] = {"region_ids": [int(min(ranked, key=dist)["region_id"])] if ranked else [], "priority": FALLBACK_PRIORITY}
        return out

    # -- step ----------------------------------------------------------------------------------
    def _update_stalls(self, coverage_ppm: int) -> None:
        hist = self.progress.coverage_ppm
        if hist:
            self.progress.stalls = 0 if coverage_ppm - hist[-1] >= STALL_EPS_PPM else self.progress.stalls + 1
        hist.append(coverage_ppm)

    async def step_async(self, tick: int, regions: Sequence[Mapping[str, Any]], squads: Sequence[SquadState],
                         coverage_ppm: int) -> QueenStep:
        """One queen step: call (if the cap allows), write task + route entries with their briefs, return the plan."""
        if sorted(s.squad for s in squads) != list(range(self.n_squads)):
            raise QueenError("squads must be exactly 0..n_squads-1; the queen cannot change the squad count")
        if not regions:
            raise QueenError("no perceived regions")
        regions = [dict(r) for r in regions]
        self._update_stalls(coverage_ppm)
        replan = self.progress.stalls >= STALL_LIMIT
        parsed: Optional[Dict[str, Any]] = None
        why = ""
        called = False
        if self.calls >= self.max_calls:
            why = f"call cap {self.max_calls} spent"
        else:
            request = self.request(tick, regions, squads, coverage_ppm, replan)
            self.calls += 1
            called = True
            try:
                response = await self.provider.complete(request)
                if response.completion_tokens > MAX_OUTPUT_TOKENS or response.finish_reason == "length":
                    raise QueenReplyError("output over the 800-token cap")
                parsed = self.parse(response.content, regions)
            except DecisionAbort:      # a budget hard stop ends the run; it is never a queen fallback
                raise
            except Exception as exc:   # invalid reply, provider failure and per-run call/token trips land here
                why = f"{type(exc).__name__}: {str(exc)[:90]}"
        if replan and called:
            self.progress.replans += 1
            self.progress.stalls = 0
        self.steps += 1
        fell_back = parsed is None
        if parsed is None:
            routes = self.s_rule_routes(regions, squads)
            candidates = [int(r["region_id"]) for r in rank_regions(regions)[:MAX_CANDIDATES]]
        else:
            fb = self.s_rule_routes(regions, squads)
            routes = {s: parsed["routes"].get(s, fb[s]) for s in range(self.n_squads)}
            candidates = parsed["candidates"] or [int(r["region_id"]) for r in rank_regions(regions)[:MAX_CANDIDATES]]
            self.task_ledger.plan = parsed["plan"]
            self.task_ledger.guesses = parsed["guesses"]
        ranked = rank_regions(regions)
        self.task_ledger.facts = [f"{len(regions)} perceived regions; top region {ranked[0]['region_id']}",
                                  f"coverage estimate {coverage_ppm // 10000}%"]
        alarm = None
        if fell_back:
            alarm = self.tools.write("alarm", {"kind": "other", "value_ppm": 1_000_000, "note": f"queen fallback: {why}"[:140]}, tick)

        cadence = self.harness["queen"]["cadence"]
        objective = self.harness["briefs"]["objective_template"].format(
            n_regions=len(regions), n_squads=self.n_squads, coverage_pct=coverage_ppm // 10000)[:280]
        task = self.tools.write("task", {"objective": objective, "squads": list(range(self.n_squads)), "until_tick": tick + cadence}, tick)
        count = self.harness["briefs"]["count"]
        types = tuple(self.harness["briefs"]["entry_types"])
        briefs: Dict[int, Brief] = {}
        route_entries: List[Entry] = []
        for s in range(self.n_squads):
            draft = {"squad": s, "region_ids": routes[s]["region_ids"], "priority": routes[s]["priority"], "brief_ids": []}
            if self.board_on:
                brief = build_brief(self.board, s, draft, cap=(6, count), entry_types=types)
            else:
                brief = build_brief_board_off(self.board, s, draft, cap=(6, count))
            briefs[s] = brief
            entry = self.tools.write("route", {**draft, "brief_ids": list(brief.brief_ids)}, tick, cites=[task.id])
            route_entries.append(entry)
            self.previous_routes[s] = {"squad": s, "region_ids": list(draft["region_ids"]), "priority": draft["priority"]}
            self.progress.status[s] = {"squad": s, "region_ids": list(draft["region_ids"]), "priority": draft["priority"],
                                       "payload_ppm": next(q.payload_ppm for q in squads if q.squad == s)}
        options = tuple(LABELS[: len(candidates)]) + ALWAYS_AVAILABLE
        return QueenStep(tick=tick, task=task, routes=tuple(route_entries), briefs=briefs, candidate_region_ids=tuple(candidates),
                         options=options, fell_back=fell_back, replan=replan and called, called=called, calls_used=self.calls, alarm=alarm)

    def step(self, tick: int, regions: Sequence[Mapping[str, Any]], squads: Sequence[SquadState], coverage_ppm: int) -> QueenStep:
        """Synchronous wrapper (the world's queen hook is synchronous)."""
        return asyncio.run(self.step_async(tick, regions, squads, coverage_ppm))
