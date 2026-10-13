"""v4.1 episode harness (goal 4.7): one seeded episode of any arm in goal section 6 on one case.

``V41Harness.run(arm, case, seed)`` wires the world (``worlds.tumor_mri``), the board and briefs (``kernel.board``,
``kernel.briefs``), the queen (``kernel.policies.v41_queen``), the router and the conformal deferral
(``experiments.v41_router`` / ``v41_conformal``), the worker backends (``providers.decision_backends``) and the
outside anchors (``anchoring.Anchorer``). It returns an :class:`EpisodeResult`: the cell record
(docs/v4.1/CELLS.md), the provenance bundle, the per-decision log, the run log, the board and the anchor records.

Round, as run here (goal 4.2/4.5/4.7), for every squad at the same time (observations are taken before any squad's
decision is applied):

  local look (d1)  ->  evidence entry (board on)  ->  brief  ->  d1 answers  ->  router pick  ->  conformal
  deferral (when d1 was picked)  ->  decision entry carrying the read receipt  ->  world step

The world is injected through a loader, ``load_case(case_id) -> (Crop, regions)`` (``regions`` may be ``None`` to use
the world's own region finder). The ``missing_flair`` variant calls ``load_case(case_id, variant="missing_flair")``.

Choices worth knowing (each is a small interpretation of the goal; see ``docs/v4.1/FRONTEND-NOTES.md`` for the board):

* Briefs are rebuilt every round from the squad's current queen route, so a squad reads this round's evidence. The
  queen's route entries (written at her steps) still carry the briefs of her step.
* Board-off (SW-noBoard): squads never write evidence or decisions to a store, never cite, and read only the task,
  perception's region signals and their own assignment. A private queen log (task, knowledge, route, alarm) exists
  only because the queen's tools are board writes; it holds nothing a squad produced, is not anchored, and is not
  counted as the run's board (``board_size`` 0).
* Calibration mode (Phase 0 step 7): Qwen3.5-9B drives every decision. d1 and Qwen3.5-4B answer the same state
  counterfactually; their answers and S' labels are logged and never touch the world, the board or a brief.
* ``router.slm_conf_min`` escalates a Qwen3.5-4B answer whose top target probability is below it to the deferral
  target. ``router.fw_conf_min`` and ``router.cost_penalty`` have no effect here (no second FW model; the kNN router
  already prefers the cheaper model on ties).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from antelligence.providers.redact import scrub
from antelligence.anchoring.rekor import AnchorKey
from antelligence.experiments import v41_cells
from antelligence.experiments.v41_conformal import (ConformalDeferral, ConformalThresholds, DeferralMonitor,
                                                    audit_coverage, audit_selected, decision_labels)
from antelligence.experiments.v41_gate import assert_eval_allowed
from antelligence.experiments.v41_local_look import (apply_temperature, load_frozen_temperature, look_request,
                                                     LOOK_QUESTION)
from antelligence.experiments.v41_router import KNNRouter, router_features, sprime_labels
from antelligence.kernel import briefs as briefs_mod
from antelligence.kernel.board import (DEFAULT_AUTHORS, Board, Entry, RunKeys, anchor_sizes,
                                       generate_run_keys, pubkey_hex, run_id_of)
from antelligence.kernel.canonical import canonical_json, content_hash
from antelligence.kernel.decision import DecisionAbort, DecisionBackend, DecisionError, DecisionRequest, DecisionResponse
from antelligence.kernel.policies.v41_harness import harness_sha256, load_harness, validate_harness
from antelligence.kernel.policies.v41_queen import FALLBACK_CAUSES, SquadState, V41Queen, coverage_estimate_ppm
from antelligence.kernel.scheduler import RunResult, derive_seed
from antelligence.kernel.types import Intent, LocalView
from antelligence.perception.store import local_patch
from antelligence.providers.outage import OutagePolicy, QueenUnavailable
from antelligence.provenance.bundle import build_bundle
from antelligence.worlds.tumor_mri import (ACTION_DECIDE, Candidate, ScriptS, TumorMRIWorld, WorldConfig, build_request,
                                           decide_from_answers, oracle_for, run_episode, select_candidates)

QUEEN_MODEL = "deepseek/deepseek-v4.1-flash"
D1, Q4, Q9, BONSAI, FW = "d1", "qwen3.5-4b", "qwen3.5-9b", "bonsai-8b", "fw"
SCOPE = "tumor_mri-v4.1"
VERIFY_RADIUS_VOX = 6        # a squad this close to a claim can contradict or corroborate it
LIE_RADIUS_VOX = 4           # a liar claims tumour at a perceived-empty voxel this close to an honest squad
LIE_BELOW_PPM, LIE_ABOVE_PPM = 100_000, 900_000
BYZANTINE_SQUADS = 2
BRIEF_STATE_CHARS = 700

DECISION_LOG_FIELDS = (
    "run_id", "arm", "case", "seed", "squad", "round", "tick", "backend", "model", "revision", "hardware",
    "server_ms", "network_ms", "queue_ms", "tokens_in", "tokens_out", "cost_usd", "router_pick", "deferred",
    "escalated", "cited_ids", "brief_ids", "target", "release", "amount", "p_target_ppm", "p_release_ppm",
    "p_amount_ppm", "look_p_ppm", "prob_method", "identity", "calls", "counterfactual", "audit", "sprime",
    "error", "error_reason", "attempts", "notes", "failures", "kind", "features")
RUN_LOG_FIELDS = (
    "run_id", "arm", "case", "seed", "split", "wall_s", "ticks", "ticks_to_50", "seconds_to_50", "queen_calls",
    "queen_fallbacks", "queen_outage_retries", "board_size", "anchors", "anchor_latency_s", "visibility_latency_ms", "decisions", "rounds", "shared_store",
    "calibration", "variant", "alarms", "dropped_cites", "reputation", "d_timeouts", "decision_errors", "look_errors", "errors")
# What a replay from cache must reproduce: every cell field except the wall-clock ones and the anchor count
# (checked separately against ``anchor_sizes``).
METRIC_FIELDS = tuple(f for f in v41_cells.FIELD_NAMES
                      if f not in ("wall_s", "seconds_to_50", "visibility_latency_ms", "anchors"))
_TRACE_KEYS = ("squad", "round", "tick", "backend", "model", "revision", "router_pick", "deferred", "escalated",
               "cited_ids", "brief_ids", "target", "release", "amount", "p_target_ppm", "p_release_ppm",
               "p_amount_ppm", "look_p_ppm", "tokens_in", "tokens_out", "cost_usd", "error")
DECISION_RETRIES = 3          # goal 8: retry provider errors up to 3 times with the same request, then mark an error
DECISION_ERROR_ALARM = 0.05   # a run with more than this share of error decisions is flagged for the explained-null checklist
FALLBACK_TARGET, FALLBACK_RELEASE, FALLBACK_AMOUNT = "S", 0, 1   # the neutral decision: stay, release nothing
CORRECTIVE_NOTE = ("your previous reply was rejected ({reason}); give a full probability distribution for every "
                   "question, including amount when you do not release")


class _DecisionFailed(Exception):
    """A backend call still failed after the retries; the decision (not the run) becomes an error."""

    def __init__(self, worker: str, reason: str) -> None:
        super().__init__(f"{worker}: {reason}")
        self.worker, self.reason = worker, reason


class HarnessError(RuntimeError):
    """The episode cannot run as configured."""


class EpisodeNotStarted(RuntimeError):
    """``run_many`` did not start this episode because the queen provider stayed down (:class:`QueenUnavailable`) in
    another one. It has no cell and no cache entry; rerun it later."""


# --------------------------------------------------------------------------------------------------
# Arms (goal section 6)
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str                       # script | oracle | workers | fw
    queen: str = "none"             # none | promoted | base | static_rule
    board: str = "off"              # off | B | D
    anchored: bool = False
    variant: str = "full"
    byzantine: int = 0
    reputation: Optional[str] = None  # None | local | chain
    bonsai: bool = False

    @property
    def uses_queen(self) -> bool:
        return self.queen in ("promoted", "base")

    @property
    def board_on(self) -> bool:
        return self.board != "off"


ARM_SPECS: Dict[str, ArmSpec] = {s.name: s for s in (
    ArmSpec("S", "script"),
    ArmSpec("S-prime", "oracle"),
    ArmSpec("SW-B", "workers", "promoted", "B", anchored=True),
    ArmSpec("SW-noBoard", "workers", "promoted", "off"),
    ArmSpec("FW", "fw", "promoted", "B", anchored=True),
    ArmSpec("SW-D", "workers", "promoted", "D", anchored=True),
    ArmSpec("SW-noQueen", "workers", "static_rule", "B"),
    ArmSpec("SW-staticQueen", "workers", "base", "B"),
    ArmSpec("SW-Bonsai", "workers", "promoted", "B", bonsai=True),
    ArmSpec("SW-Byz-local", "workers", "promoted", "B", byzantine=BYZANTINE_SQUADS, reputation="local", anchored=True),
    ArmSpec("SW-Byz-chain", "workers", "promoted", "B", byzantine=BYZANTINE_SQUADS, reputation="chain", anchored=True),
    ArmSpec("SW-B-missingFLAIR", "workers", "promoted", "B", variant="missing_flair"),
)}
assert tuple(ARM_SPECS) == v41_cells.ARMS
CALIBRATION_ARMS = ("SW-B", "SW-noBoard")
ANCHORED_ARMS = tuple(n for n, s in ARM_SPECS.items() if s.anchored)
_STD = (D1, Q4, Q9)
# Router model set per arm (goal section 6), checked against the frozen router before an episode starts.
ROUTER_MODELS_BY_ARM: Dict[str, Tuple[str, ...]] = {
    **{a: _STD for a in ("SW-B", "SW-noBoard", "SW-D", "SW-noQueen", "SW-staticQueen", "SW-Byz-local", "SW-Byz-chain",
                         "SW-B-missingFLAIR")},
    "SW-Bonsai": (D1, Q4, BONSAI),
    "FW": (FW,),
}


# --------------------------------------------------------------------------------------------------
# Small pieces: workers, keys, reputation, board views, tallies
# --------------------------------------------------------------------------------------------------

def _model_of(backend: Any) -> str:
    while not hasattr(backend, "model") and hasattr(backend, "inner"):
        backend = backend.inner
    return str(backend.model)


@dataclass(frozen=True)
class Worker:
    """A decision backend and the model id its requests carry (``request.model`` must equal it)."""

    backend: DecisionBackend
    model: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(self, "model", _model_of(self.backend))


class KeyVault:
    """Per-run ed25519 author keys and the Rekor anchor key, by label. With a ``ledger/`` directory they persist, so a
    replay rebuilds the same header (and so the same entry ids and request hashes); without one they live in memory
    for this vault's lifetime. Private seeds are only ever written under a ``ledger/`` directory."""

    def __init__(self, ledger_dir: Optional[Union[str, Path]] = None) -> None:
        self.ledger_dir = Path(ledger_dir) if ledger_dir is not None else None
        self._mem: Dict[str, Tuple[RunKeys, AnchorKey]] = {}

    def get(self, label: str) -> Tuple[RunKeys, AnchorKey]:
        if label in self._mem:
            return self._mem[label]
        keys_file = anchor_file = None
        if self.ledger_dir is not None:
            keys_file, anchor_file = self.ledger_dir / f"{label}.keys.json", self.ledger_dir / f"{label}.anchor.pem"
        if keys_file is not None and keys_file.exists() and anchor_file.exists():
            seeds = {a: bytes.fromhex(s) for a, s in json.loads(keys_file.read_text()).items()}
            from cryptography.hazmat.primitives import serialization
            akey = AnchorKey(serialization.load_pem_private_key(anchor_file.read_bytes(), password=None))
            keys = RunKeys({a: pubkey_hex(s) for a, s in seeds.items()}, seeds)
        else:
            keys = generate_run_keys(DEFAULT_AUTHORS, self.ledger_dir, label)
            akey = AnchorKey()
            if self.ledger_dir is not None:
                akey.save(self.ledger_dir, label)
        self._mem[label] = (keys, akey)
        return keys, akey


def _reputation_rule():
    try:
        from chain import v4_1 as rule
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
        from chain import v4_1 as rule
    return rule


class ReputationLedger:
    """Author reputation in integer ppm under the shared rule (``chain.v4_1.reputation_update``): a contradicted claim
    halves it, a corroborated one adds 0.1 (cap 1.0), below 0.3 an author leaves briefs. One report per
    (author, claim) is applied (idempotent, like ``ReputationRegistry.report``). With a ``client`` every report is
    also queued for ``ReputationRegistry`` (the on-chain Byzantine arm); the local mirror gives the live score."""

    def __init__(self, pubkeys: Mapping[str, str], run_id: str, client: Any = None) -> None:
        self._rule = _reputation_rule()
        self.score: Dict[str, int] = {}
        self.pubkeys, self.run_id, self.client = dict(pubkeys), run_id, client
        self._seen: Set[Tuple[str, str]] = set()
        self.reports: List[Dict[str, Any]] = []
        self.jobs: List[int] = []

    def reputation(self, author: str) -> int:
        return self.score.get(author, self._rule.REP_START_PPM)

    def excluded(self, author: str) -> bool:
        return self._rule.is_excluded(self.reputation(author))

    def excluded_authors(self) -> frozenset:
        return frozenset(a for a in self.score if self.excluded(a))

    def report(self, author: str, claim_id: str, contradicted: bool, tick: int) -> None:
        if (author, claim_id) in self._seen:
            return
        self._seen.add((author, claim_id))
        self.score[author] = self._rule.reputation_update(self.reputation(author), contradicted)
        self.reports.append({"author": author, "claim_id": claim_id, "contradicted": contradicted, "tick": tick,
                             "after_ppm": self.score[author]})
        if self.client is not None:
            author_hash = hashlib.sha256(bytes.fromhex(self.pubkeys[author])).hexdigest()
            self.jobs.append(self.client.enqueue_report(self.run_id, author_hash, claim_id, contradicted))


class BriefView:
    """What squads may read: the board minus entries not yet visible (mode D) and minus excluded authors."""

    def __init__(self, board: Board, visible: Optional[Set[int]] = None, excluded: frozenset = frozenset()) -> None:
        self._board, self._visible, self._excluded = board, visible, excluded

    def entries(self) -> List[Entry]:
        return [e for e in self._board.entries()
                if (self._visible is None or e.seq in self._visible) and e.author["agent"] not in self._excluded]

    def by_type(self, entry_type: str) -> List[Entry]:
        return [e for e in self.entries() if e.type == entry_type]

    def get(self, key: Union[int, str]) -> Entry:
        entry = self._board.get(key)
        if entry not in self.entries():
            raise KeyError(key)
        return entry

    def __contains__(self, entry_id: object) -> bool:
        return any(e.id == entry_id for e in self.entries())

    def __len__(self) -> int:
        return len(self.entries())

    def __getattr__(self, name: str) -> Any:   # append, header, run_id ... go to the real board
        return getattr(self._board, name)


class _DeferredView(BriefView):
    """A BriefView whose visible set is looked up when read (the set is replaced every round)."""

    def __init__(self, board: Board, visible: Callable[[], Optional[Set[int]]]) -> None:
        super().__init__(board)
        self._visible_fn = visible

    def entries(self) -> List[Entry]:
        visible = self._visible_fn()
        return [e for e in self._board.entries() if visible is None or e.seq in visible]


class _HookedBoard(Board):
    """A board that calls ``on_append(board, entry)`` after every append, the queen's included, so anchor points fall
    exactly on sizes 50, 100, ..."""

    def __init__(self, *a: Any, on_append: Optional[Callable[[Board, Entry], None]] = None, **kw: Any) -> None:
        self._on_append = None
        super().__init__(*a, **kw)
        self._on_append = on_append

    def append(self, *a: Any, **kw: Any) -> Entry:
        entry = super().append(*a, **kw)
        if self._on_append is not None:
            self._on_append(self, entry)
        return entry


class _TallyProvider:
    """Counts the queen's tokens and cost; otherwise transparent."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.tokens_in, self.tokens_out, self.cost = inner, 0, 0, 0.0

    def describe(self) -> Dict[str, Any]:
        return self.inner.describe()

    async def complete(self, request: Any) -> Any:
        response = await self.inner.complete(request)
        self.tokens_in += int(response.prompt_tokens)
        self.tokens_out += int(response.completion_tokens)
        self.cost += float((response.extra or {}).get("cost_usd", 0.0) or 0.0)
        return response


def _ppm(x: float) -> int:
    return int(min(1_000_000, max(0, round(float(x) * 1_000_000))))


def _sha(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    cell: Dict[str, Any]
    bundle: Dict[str, Any]
    decision_log: List[Dict[str, Any]]
    run_log: Dict[str, Any]
    board: Optional[Board]
    anchors: List[Dict[str, Any]]
    run_dir: Optional[Path]
    trace_hash: str
    reputation: Optional[ReputationLedger] = None


@dataclass
class GateConfig:
    repo_root: Path
    prereg_path: Path
    anchor_receipt: Path
    g1_file: Path


def make_gate(cfg: GateConfig) -> Callable[[str], str]:
    """``v41_gate.assert_eval_allowed`` bound to the study's files; returns the case's split or raises GateError."""
    return lambda case: assert_eval_allowed(case, cfg.repo_root, cfg.prereg_path, cfg.anchor_receipt, cfg.g1_file)


# --------------------------------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------------------------------

@dataclass
class _Ctx:
    """Everything one episode owns. Nothing here is shared between episodes (they may run concurrently)."""

    spec: ArmSpec
    arm: str
    case: str
    seed: int
    split: str
    calibration: bool
    world: TumorMRIWorld
    crop: Any
    harness: Dict[str, Any]
    run_id: str
    header: Dict[str, Any]
    board: Optional[Board]
    shared: bool
    queen: Optional[V41Queen]
    queen_provider: Optional[_TallyProvider]
    anchorer: Any
    post_mode: Any
    reputation: Optional[ReputationLedger]
    router: Optional[KNNRouter]
    deferral: Optional[ConformalDeferral]
    target_name: str
    region_by_id: Dict[int, Any]
    region_payloads: List[Dict[str, Any]]
    liars: List[int]
    pubkeys: Dict[str, str]
    routes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    route_entries: Dict[int, Entry] = field(default_factory=dict)
    visible: Optional[Set[int]] = None
    posted_upto: int = 0
    looks: Dict[Tuple[int, int], Optional[float]] = field(default_factory=dict)
    log: List[Dict[str, Any]] = field(default_factory=list)
    monitor: DeferralMonitor = field(default_factory=DeferralMonitor)
    audits: List[Tuple[Mapping[str, Tuple[str, ...]], Mapping[str, str]]] = field(default_factory=list)
    picks: Counter = field(default_factory=Counter)
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    alarm_count: int = 0
    dropped_cites: int = 0
    all_cites: int = 0
    valid_cites: int = 0
    vis_latency: List[int] = field(default_factory=list)
    d_timeouts: int = 0
    errors: List[str] = field(default_factory=list)
    decision_errors: int = 0
    look_errors: int = 0
    look_rows: List[Dict[str, Any]] = field(default_factory=list)
    attempt_log: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    replay_attempts: Optional[Mapping[str, Any]] = None
    oracle: Any = None


class V41Harness:
    def __init__(self, *, load_case: Callable[..., Tuple[Any, Optional[Sequence[Any]]]], gate: Callable[[str], str],
                 workers: Optional[Mapping[str, Worker]] = None, queen_provider: Any = None,
                 queen_model: str = QUEEN_MODEL, world_config: Optional[WorldConfig] = None,
                 promoted_harness: Optional[Mapping[str, Any]] = None, base_harness: Optional[Mapping[str, Any]] = None,
                 router: Optional[KNNRouter] = None, bonsai_router: Optional[KNNRouter] = None,
                 thresholds: Optional[ConformalThresholds] = None,
                 bonsai_thresholds: Optional[ConformalThresholds] = None, look_temperature: Optional[float] = None,
                 anchorer_factory: Optional[Callable[[Board, AnchorKey, Path], Any]] = None,
                 allow_unanchored: bool = False, post_mode_factory: Optional[Callable[[], Any]] = None,
                 reputation_client: Any = None, outbox: Any = None, keys: Optional[KeyVault] = None,
                 out_dir: Optional[Union[str, Path]] = None, code_sha: str = "0" * 40,
                 hardware: Optional[Mapping[str, str]] = None, backends_kind: str = "config",
                 perception_stats: Optional[Callable[[str], Mapping[str, float]]] = None,
                 clock: Callable[[], float] = time.perf_counter, queen_outage: Optional[OutagePolicy] = None) -> None:
        self.load_case, self.gate = load_case, gate
        self.queen_outage = queen_outage if queen_outage is not None else OutagePolicy()
        self.workers = dict(workers or {})
        self.queen_provider, self.queen_model = queen_provider, queen_model
        self.world_config = world_config or WorldConfig()
        self.base_harness = dict(base_harness) if base_harness is not None else load_harness()
        self.promoted_harness = dict(promoted_harness) if promoted_harness is not None else dict(self.base_harness)
        validate_harness(self.base_harness)
        validate_harness(self.promoted_harness)
        self.router, self.bonsai_router = router, bonsai_router
        self.thresholds, self.bonsai_thresholds = thresholds, bonsai_thresholds
        # default = the frozen, sha-verified temperature; an explicit value is an override and is recorded as such
        self.frozen_look_temperature = load_frozen_temperature()
        self.look_temperature_source = "frozen" if look_temperature is None else "override"
        self.look_temperature = self.frozen_look_temperature if look_temperature is None else float(look_temperature)
        self.anchorer_factory, self.allow_unanchored = anchorer_factory, allow_unanchored
        self.post_mode_factory, self.reputation_client = post_mode_factory, reputation_client
        self.outbox, self.keys = outbox, keys or KeyVault()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.code_sha, self.hardware = code_sha, dict(hardware or {})
        if backends_kind not in v41_cells.BACKENDS_KINDS:
            raise HarnessError(f"backends_kind must be one of {v41_cells.BACKENDS_KINDS}, got {backends_kind!r}")
        self.backends_kind = backends_kind
        self.perception_stats, self.clock = perception_stats, clock

    @staticmethod
    def _router_kind(router: Any) -> str:
        return "none" if router is None else str(getattr(router, "kind", "frozen"))

    @property
    def router_kind(self) -> str:
        """frozen | none | fake for this harness's router (the Bonsai arm's router must agree)."""
        kinds = {self._router_kind(r) for r in (self.router, self.bonsai_router) if r is not None}
        if not kinds:
            return "none"
        return kinds.pop() if len(kinds) == 1 else ("fake" if "fake" in kinds else "none")

    def _check_isolation(self, split: str) -> None:
        """Fake backends and bootstrap/fake routers are for development cases only (goal 5 step 7). Raised before anything
        is loaded or called, and independent of the gate: a present prereg, receipt and G1 file do not lift it."""
        if split == "dev":
            return
        problems = []
        if self.backends_kind == "fake":
            problems.append("fake backends")
        if self.router_kind in ("none", "fake"):
            problems.append(f"router {self.router_kind!r}")
        if problems:
            raise HarnessError(f"{' and '.join(problems)} refused on a {split} case: development cases only")

    # ---- public --------------------------------------------------------------------------------

    async def run(self, arm: str, case: str, seed: int, *, calibration: bool = False,
                  anchor: bool = True, replay_attempts: Optional[Mapping[str, Any]] = None) -> EpisodeResult:
        """One episode. The gate is the first thing that happens; nothing is loaded or called before it passes."""
        if arm not in ARM_SPECS:
            raise HarnessError(f"unknown arm {arm!r}")
        spec = ARM_SPECS[arm]
        split = self.gate(case)
        if split not in ("dev", "eval"):
            raise HarnessError(f"gate returned {split!r}")
        self._check_isolation(split)
        if split == "eval" and not math.isclose(self.look_temperature, self.frozen_look_temperature, rel_tol=0, abs_tol=1e-9):
            raise HarnessError(f"look temperature override {self.look_temperature} differs from the frozen "
                               f"{self.frozen_look_temperature}: refused on an eval case (it must match the prereg)")
        if calibration and (spec.name not in CALIBRATION_ARMS or split != "dev"):
            raise HarnessError("calibration mode is for SW-B and SW-noBoard on development cases only")
        if spec.kind in ("workers", "fw"):
            self._check_ready(spec, calibration)
            if spec.anchored and anchor and self.anchorer_factory is None and not self.allow_unanchored:
                raise HarnessError(f"arm {spec.name} must be anchored: pass anchorer_factory (or allow_unanchored)")
        t0 = self.clock()
        ctx = self._build(spec, case, int(seed), split, calibration, anchor)
        ctx.replay_attempts = replay_attempts
        if spec.kind in ("script", "oracle"):
            wall_marks = self._run_scripted(ctx, t0)
        else:
            wall_marks = await self._run_swarm(ctx, t0)
        return self._finish(ctx, t0, wall_marks)

    async def run_many(self, requests: Sequence[Tuple[str, str, int]], *, concurrency: int = 4,
                       calibration: bool = False) -> List[Union[EpisodeResult, BaseException]]:
        """Episodes concurrently (asyncio) in ``requests`` order; each owns its world, board, queen and anchorer. A
        failed episode comes back as its exception so the others still finish.

        The first :class:`QueenUnavailable` (the queen provider stayed down past the maximum wait) stops the batch
        starting new episodes: those still waiting come back as :class:`EpisodeNotStarted`, those in flight finish or
        error on their own. A rerun resumes from the caches."""
        sem = asyncio.Semaphore(max(1, concurrency))
        stop = asyncio.Event()

        async def one(r: Tuple[str, str, int]) -> Union[EpisodeResult, BaseException]:
            async with sem:
                if stop.is_set():
                    return EpisodeNotStarted(f"{r[0]} {r[1]} seed {r[2]}: not started, the queen provider is unavailable")
                try:
                    return await self.run(r[0], r[1], r[2], calibration=calibration)
                except QueenUnavailable as exc:
                    stop.set()
                    return exc
                except BaseException as exc:  # noqa: BLE001  (recorded per episode)
                    return exc
        return list(await asyncio.gather(*(one(r) for r in requests)))

    async def replay(self, bundle: Mapping[str, Any]) -> Dict[str, Any]:
        """Re-run a bundle's episode (this harness must read the same caches offline and hold the same keys) and require
        the same trace hash, every metric field and the same anchor sizes."""
        spec = bundle["spec"]
        again = await self.run(spec["arm"], spec["case"], spec["seed"], calibration=spec["calibration"], anchor=False,
                                 replay_attempts=bundle["extra"].get("attempts") or {})
        recorded = bundle["extra"]["cell"]
        mismatched = [f for f in METRIC_FIELDS if recorded.get(f) != again.cell.get(f)
                      and not (f in v41_cells.QUEEN_FALLBACK_FIELDS and f not in recorded)]   # bundles from before the field existed
        if again.trace_hash != bundle["trace_hash"]:
            mismatched.append("trace_hash")
        if bundle["extra"]["anchor_sizes"] != anchor_sizes(again.cell["board_size"]) and bundle["extra"]["anchor_sizes"]:
            mismatched.append("anchor_sizes")
        return {"replay_ok": not mismatched, "mismatched_fields": mismatched, "trace_hash": again.trace_hash}

    # ---- construction ----------------------------------------------------------------------------

    def _build(self, spec: ArmSpec, case: str, seed: int, split: str, calibration: bool, anchor: bool) -> _Ctx:
        if spec.variant == "full":
            crop, regions = self.load_case(case)
        else:
            try:
                crop, regions = self.load_case(case, variant=spec.variant)
            except TypeError as exc:
                raise HarnessError(f"the case loader cannot produce variant {spec.variant!r}: {exc}") from exc
        harness = dict(self.promoted_harness if spec.queen == "promoted" and not calibration else self.base_harness)
        arm_label = f"{spec.name}:calibration" if calibration else spec.name
        label = f"{arm_label}-{case}-{seed}".replace(":", "_").replace("/", "_")
        world_cfg_sha = _sha({"world": self.world_config.to_dict(), "variant": spec.variant})
        scripted = spec.kind in ("script", "oracle")
        keys: Optional[RunKeys] = None
        akey: Optional[AnchorKey] = None
        if not scripted:
            keys, akey = self.keys.get(label)
        header = {"study": "v4.1", "arm": arm_label, "case": case, "seed": seed, "split": split,
                  "world_cfg_sha256": world_cfg_sha, "harness_cfg_sha256": harness_sha256(harness),
                  "pubkeys": dict(keys.pubkeys) if keys else {}, "anchor_pubkey": akey.fingerprint if akey else "0" * 64,
                  "code_sha": self.code_sha}
        run_id = run_id_of(header)
        ctx_kw: Dict[str, Any] = dict(spec=spec, arm=spec.name, case=case, seed=seed, split=split,
                                      calibration=calibration, harness=harness, run_id=run_id, header=header,
                                      crop=crop, queen=None, queen_provider=None, anchorer=None, post_mode=None,
                                      reputation=None, router=None, deferral=None, target_name=Q9)
        world = TumorMRIWorld(crop, self.world_config, seed, regions=regions)
        ctx_kw.update(world=world, region_by_id={r.region_id: r for r in world.regions}, liars=[],
                      pubkeys=dict(keys.pubkeys) if keys else {}, board=None, shared=False)
        ctx_kw["region_payloads"] = [self._knowledge_payload(r, world.voxel_mm) for r in world.regions]
        if spec.kind in ("script", "oracle"):
            return _Ctx(**ctx_kw)
        world.look_fn = lambda i, pos, tick: ctx.looks.get((i, tick))
        board_mode = "B" if spec.board_on else "A"
        ctx_holder: List[_Ctx] = []

        def on_append(board: Board, entry: Entry) -> None:
            if ctx_holder and ctx_holder[0].anchorer is not None:
                ctx_holder[0].anchorer.maybe_anchor(board)
            if entry.type == "alarm" and ctx_holder:
                ctx_holder[0].alarm_count += 1

        board = _HookedBoard(header, mode=board_mode, keys=keys.seeds if board_mode == "B" else None,
                             on_append=on_append)
        ctx_kw.update(board=board, shared=spec.board_on)
        if spec.uses_queen:
            if self.queen_provider is None:
                raise HarnessError(f"arm {spec.name} needs a queen_provider")
            tally = _TallyProvider(self.queen_provider)
            ctx_kw["queen_provider"] = tally
            outage = dataclasses.replace(self.queen_outage)       # own retry counters per episode
            queen_board: Any = board
            if spec.board == "D":      # mode D: the queen reads only what the chain has included
                queen_board = _DeferredView(board, lambda: ctx_holder[0].visible)
            ctx_kw["queen"] = V41Queen(tally, self.queen_model, queen_board, n_squads=self.world_config.n_squads,
                                       harness=harness, board_on=spec.board_on, outage=outage)
        if spec.kind == "workers" and not calibration:
            ctx_kw["router"] = self.bonsai_router if spec.bonsai else self.router
            thresholds = self.bonsai_thresholds if spec.bonsai else self.thresholds
            ctx_kw["deferral"] = ConformalDeferral(thresholds)
            ctx_kw["target_name"] = BONSAI if spec.bonsai else Q9
        if spec.byzantine:
            rng = random.Random(f"v4.1|byz|{seed}")
            ctx_kw["liars"] = sorted(rng.sample(range(self.world_config.n_squads), spec.byzantine))
            client = self.reputation_client if spec.reputation == "chain" else None
            if spec.reputation == "chain" and client is None:
                raise HarnessError("SW-Byz-chain needs reputation_client")
            ctx_kw["reputation"] = ReputationLedger(keys.pubkeys, run_id, client)
        if spec.board == "D":
            if self.post_mode_factory is None:
                raise HarnessError("SW-D needs post_mode_factory")
            ctx_kw["post_mode"] = self.post_mode_factory()
            ctx_kw["visible"] = set()
        if spec.anchored and anchor:
            if self.anchorer_factory is not None:
                run_dir = self._run_dir(run_id, create=True)
                # the outside verifier needs the public Rekor key next to the export (<run_id>.anchor_pub.pem)
                (run_dir / f"{run_id}.anchor_pub.pem").write_bytes(akey.pem)
                ctx_kw["anchorer"] = self.anchorer_factory(board, akey, run_dir)
        ctx = _Ctx(**ctx_kw)
        ctx_holder.append(ctx)
        return ctx

    def _check_ready(self, spec: ArmSpec, calibration: bool) -> None:
        need = {D1} if spec.kind == "workers" or spec.kind == "fw" else set()
        if spec.kind == "fw":
            need.add(FW)
        elif calibration:
            need |= {Q4, Q9, D1}
        else:
            router = self.bonsai_router if spec.bonsai else self.router
            thresholds = self.bonsai_thresholds if spec.bonsai else self.thresholds
            if router is None or thresholds is None:
                raise HarnessError(f"arm {spec.name} needs a frozen router and conformal thresholds")
            if tuple(router.models) != ROUTER_MODELS_BY_ARM[spec.name]:
                raise HarnessError(f"arm {spec.name} routes {ROUTER_MODELS_BY_ARM[spec.name]}, "
                                   f"but the router was trained on {tuple(router.models)}")
            need |= set(router.models) | {BONSAI if spec.bonsai else Q9}
        missing = sorted(need - set(self.workers))
        if missing:
            raise HarnessError(f"arm {spec.name} needs workers {missing}")

    def _run_dir(self, run_id: str, create: bool = False) -> Path:
        if self.out_dir is None:
            raise HarnessError("out_dir is required to write run files")
        path = self.out_dir / "runs" / run_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _knowledge_payload(r: Any, voxel_mm: float) -> Dict[str, Any]:
        return {"region_id": int(r.region_id), "centroid_dmm": [int(round(c * voxel_mm * 10)) for c in r.centroid_vox],
                "cls": "WT", "p_tumour_ppm": _ppm(r.p_tumour), "volume_mm3": int(r.volume_mm3),
                "unc_ppm": _ppm(r.unc), "source": "perception", "prov": "monai-brats+tta@perception"}

    # ---- scripted arms -----------------------------------------------------------------------------

    def _run_scripted(self, ctx: _Ctx, t0: float) -> Dict[str, Any]:
        world = ctx.world
        policy = oracle_for(world) if ctx.spec.kind == "oracle" else None
        metrics = run_episode(world, (lambda a: policy) if policy is not None else (lambda a: ScriptS()))
        return {"metrics": metrics, "t50_wall": None}

    # ---- swarm arms ----------------------------------------------------------------------------------

    async def _run_swarm(self, ctx: _Ctx, t0: float) -> Dict[str, Any]:
        world, cfg, spec = ctx.world, self.world_config, ctx.spec
        board = ctx.board
        assert board is not None
        for payload in ctx.region_payloads:
            board.append("knowledge", "perception", payload, 0)
        cadence = int(ctx.harness["queen"]["cadence"]) if spec.uses_queen else cfg.queen_every
        if ctx.calibration:
            ctx.oracle = oracle_for(world)
        t50_wall: Optional[float] = None
        for tick in range(cfg.ticks):
            if tick % cadence == 0:
                if spec.uses_queen:
                    await self._queen_step(ctx, tick)
                elif tick == 0 and spec.queen == "static_rule":
                    self._static_assignments(ctx, tick)
            if tick % cfg.decision_every == 0:
                await self._round(ctx, tick)
            world.step_environment(tick)
            if t50_wall is None and world.metrics()["time_to_50_coverage"] is not None:
                t50_wall = self.clock() - t0
        if ctx.post_mode is not None:
            await self._post_pending(ctx, cfg.ticks)
        return {"metrics": world.metrics(), "t50_wall": t50_wall}

    # -- queen / static assignments --

    def _squad_states(self, ctx: _Ctx) -> List[SquadState]:
        share = ctx.world.total_drug / self.world_config.n_squads  # per-case budget
        return [SquadState(i, tuple(int(c) for c in pos), _ppm(pay / share))
                for i, (pos, pay) in enumerate(zip(ctx.world.squad_positions(), ctx.world.squad_payloads()))]

    def _dose_fraction(self, ctx: _Ctx) -> Dict[int, float]:
        dose, thr = ctx.world.field.dose, self.world_config.dose_threshold
        out: Dict[int, float] = {}
        for rid, r in ctx.region_by_id.items():
            idx = tuple(int(min(max(round(c), 0), s - 1)) for c, s in zip(r.centroid_vox, ctx.world.shape))
            out[rid] = float(dose[idx]) / thr
        return out

    def _set_candidates(self, ctx: _Ctx, squad: int, route_ids: Sequence[int], offered: Sequence[int]) -> None:
        order = list(dict.fromkeys([i for i in route_ids if i in ctx.region_by_id]
                                   + [i for i in offered if i in ctx.region_by_id]))[:6]
        cands = [Candidate.from_region("ABCDEF"[k], ctx.region_by_id[rid]) for k, rid in enumerate(order)]
        ctx.world.set_candidates(cands, squad=squad)
        if route_ids and route_ids[0] in ctx.region_by_id:   # the queen's first region is the squad's assignment
            ctx.world.assign_route(squad, route_ids[0])

    async def _queen_step(self, ctx: _Ctx, tick: int) -> None:
        assert ctx.queen is not None and ctx.board is not None
        if not ctx.region_payloads:
            ctx.board.append("alarm", "harness", {"kind": "other", "value_ppm": 0, "note": "no perceived regions"}, tick)
            return
        cov = coverage_estimate_ppm(ctx.region_payloads, self._dose_fraction(ctx))
        step = await ctx.queen.step_async(tick, ctx.region_payloads, self._squad_states(ctx), cov)
        for entry in step.routes:
            s = entry.payload["squad"]
            ctx.routes[s] = {"squad": s, "region_ids": list(entry.payload["region_ids"]),
                             "priority": entry.payload["priority"]}
            ctx.route_entries[s] = entry
            self._set_candidates(ctx, s, entry.payload["region_ids"], step.candidate_region_ids)

    def _static_assignments(self, ctx: _Ctx, tick: int) -> None:
        """SW-noQueen: S's rule fixes each squad's regions once at tick 0 (highest p x volume / distance first)."""
        assert ctx.board is not None
        world = ctx.world
        n = self.world_config.n_squads
        ctx.board.append("task", "harness", {"objective": f"Cover the tumour: {len(ctx.region_payloads)} perceived regions, "
                                             f"{n} squads, static assignments.", "squads": list(range(n)),
                                             "until_tick": self.world_config.ticks}, tick)
        offered = [c.region_id for c in select_candidates(world.regions)]
        for s, pos in enumerate(world.squad_positions()):
            ranked = sorted(world.regions, key=lambda r: (-r.p_tumour * r.volume_mm3
                            / max(world.voxel_mm * float(np.linalg.norm(np.subtract(pos, r.centroid_vox))), 1.0), r.region_id))
            ids = [r.region_id for r in ranked[:2]]
            ctx.routes[s] = {"squad": s, "region_ids": ids, "priority": 3}
            ctx.route_entries[s] = ctx.board.append("route", "harness", {**ctx.routes[s], "brief_ids": []}, tick)
            self._set_candidates(ctx, s, ids, offered)

    # -- one decision round --

    def _view(self, ctx: _Ctx) -> BriefView:
        excluded = ctx.reputation.excluded_authors() if ctx.reputation else frozenset()
        return BriefView(ctx.board, ctx.visible, excluded)

    async def _post_pending(self, ctx: _Ctx, tick: int) -> None:
        assert ctx.board is not None
        pending = ctx.board.entries()[ctx.posted_upto:]
        ctx.posted_upto += len(pending)
        result = await asyncio.to_thread(ctx.post_mode.post_round, ctx.board.run_id, pending)
        ctx.visible = set(result.visible)
        ctx.vis_latency.extend(result.latency_ms.values())
        if result.timed_out:
            ctx.d_timeouts += 1
            ctx.board.append("alarm", "harness", {"kind": "timeout", "value_ppm": 1_000_000,
                                                  "note": f"{len(result.invisible)} entries not on chain after 60 s"}, tick)

    async def _round(self, ctx: _Ctx, tick: int) -> None:
        world, spec, board = ctx.world, ctx.spec, ctx.board
        assert board is not None
        rnd = tick // self.world_config.decision_every
        if ctx.post_mode is not None:
            await self._post_pending(ctx, tick)
        agents = world.agents()
        n = len(agents)
        # 1. local look (d1), concurrently
        looks = await asyncio.gather(*(self._look(ctx, i, tick) for i in range(n)))
        for i, look in enumerate(looks):
            ctx.looks[(i, tick)] = look
        obs = [world.observe(a, tick).data for a in agents]
        # 2. evidence (board on)
        evidence_ids: Dict[int, str] = {}
        if spec.board_on:
            for i in range(n):
                evidence_ids[i] = self._write_evidence(ctx, i, obs[i], looks[i], tick)
            for liar in ctx.liars:
                self._write_lie(ctx, liar, obs, looks, rnd, tick)
            if ctx.reputation is not None:
                self._check_claims(ctx, obs, looks, tick)
        # 3. briefs
        briefs = [self._brief(ctx, i) for i in range(n)]
        # 4. decisions, concurrently (S' labels for the calibration log are taken first, in squad order)
        sprimes: List[Optional[Dict[str, str]]] = [None] * n
        if ctx.oracle is not None:
            for i in range(n):
                view = LocalView(agent_id=f"squad-{i}", tick=tick, revision=world.revision, seed=ctx.seed,
                                 observation=dict(obs[i]))
                sprimes[i] = sprime_labels(ctx.oracle.decide(view).params, obs[i]["candidates"],
                                           voxel_mm=world.voxel_mm)
        records = await asyncio.gather(*(self._decide(ctx, i, obs[i], looks[i], briefs[i], tick, rnd, sprimes[i])
                                         for i in range(n)))
        # 5. in squad order: decision entries and world
        for i in range(n):
            rec = records[i]
            self._apply(ctx, i, rec, tick)

    async def _look(self, ctx: _Ctx, i: int, tick: int) -> Optional[float]:
        crop = ctx.crop
        if crop.flair is None or crop.t1c is None:
            return None
        pos = ctx.world.squad_positions()[i]
        patch = local_patch(ctx.case, pos, arrays={"flair": crop.flair, "t1c": crop.t1c})
        worker = self.workers[D1]
        req = look_request(worker.model, patch, seed=derive_seed(ctx.seed, f"squad-{i}", tick))
        trail: List[Dict[str, Any]] = []
        try:
            resp = await self._attempts(ctx, D1, req, trail, f"look|{i}|{tick}")
        except _DecisionFailed as failed:
            # the local look failed for good: no look (the squad decides without it), an error row and an alarm
            reason = f"look_error: {failed.reason}"
            ctx.look_errors += 1
            ctx.look_rows.append({"kind": "look", "run_id": ctx.run_id, "arm": ctx.arm, "case": ctx.case,
                                  "seed": ctx.seed, "squad": i, "round": tick // self.world_config.decision_every,
                                  "tick": tick, "backend": D1, "error": True, "error_reason": reason,
                                  "attempts": DECISION_RETRIES + 1, "notes": [t["note"] for t in trail if t["note"]],
                                  "failures": [{"attempt": t["attempt"], "error": t["error"], "billed": t["billed"]}
                                               for t in trail if t["error"]]})
            ctx.board.append("alarm", "harness", {"kind": "other", "value_ppm": 0, "note": reason[:140]}, tick)
            return None
        return float(apply_temperature(resp.answers[LOOK_QUESTION].p_yes, self.look_temperature))

    @staticmethod
    def _tally(ctx: _Ctx, resp: DecisionResponse) -> None:
        ctx.tokens_in += int(resp.input_tokens)
        ctx.tokens_out += int(resp.output_tokens)
        ctx.cost += float(resp.cost_usd)

    def _brief(self, ctx: _Ctx, squad: int) -> briefs_mod.Brief:
        assert ctx.board is not None
        h = ctx.harness["briefs"]
        draft = {**ctx.routes.get(squad, {"squad": squad, "region_ids": [], "priority": 3}), "brief_ids": []}
        own = ctx.route_entries.get(squad)
        if own is not None and ctx.visible is not None and own.seq not in ctx.visible:
            own = None   # mode D: a route the chain has not included is not yet readable
        if ctx.spec.board_on:
            return briefs_mod.build_brief(self._view(ctx), squad, draft, cap=(6, h["count"]),
                                          entry_types=tuple(h["entry_types"]), own_route=own)
        return briefs_mod.build_brief_board_off(ctx.board, squad, draft, cap=(6, h["count"]), own_route=own)

    @staticmethod
    def _compact(brief: briefs_mod.Brief, with_ids: bool) -> str:
        lines = []
        for e in brief.entries:
            p = e.payload
            if e.type == "task":
                body = str(p["objective"])[:70]
            elif e.type == "knowledge":
                body = f"region {p['region_id']} tumour {p['p_tumour_ppm'] // 10000}% size {p['volume_mm3']} mm3"
            elif e.type == "evidence":
                body = f"squad {p['squad']} looked: tumour {p['look_p_ppm'] // 10000}% at {p['pos_vox']}"
            elif e.type == "route":
                body = f"squad {p['squad']} regions {p['region_ids']}"
            elif e.type == "decision":
                body = f"squad {p['squad']} chose {p['target']} release {p['release']} level {p['amount_level']}"
            else:
                body = str(p.get("note", ""))[:40]
            lines.append(f"{e.id[:8] + ' ' if with_ids else ''}t{e.tick} {e.type}: {body}")
        return " | ".join(lines)[:BRIEF_STATE_CHARS]

    # -- evidence, lies, contradictions --

    def _write_evidence(self, ctx: _Ctx, i: int, o: Mapping[str, Any], look: Optional[float], tick: int) -> str:
        total = ctx.world.total_drug  # per-case budget (k x V_perceived in per_perceived_volume mode)
        share = total / self.world_config.n_squads
        p_here = _ppm(o["local_p_tumour"])
        payload = {"squad": i, "pos_vox": [int(c) for c in o["position_vox"]],
                   "look_p_ppm": p_here if look is None else _ppm(look), "p_tumour_ppm": p_here,
                   "dose_ug": _ppm(o["local_dose"] / total),
                   "released_ug": _ppm((share - o["payload"]) / total)}
        return ctx.board.append("evidence", f"squad-{i}", payload, tick).id

    def _write_lie(self, ctx: _Ctx, liar: int, obs: Sequence[Mapping[str, Any]], looks: Sequence[Optional[float]],
                   rnd: int, tick: int) -> None:
        """A false claim: p_tumour >= 0.9 at a voxel the perception scored < 0.1, near an honest squad."""
        honest = [s for s in range(len(obs)) if s not in ctx.liars]
        rng = random.Random(f"v4.1|lie|{ctx.seed}|{rnd}|{liar}")
        target = honest[(rnd + liar) % len(honest)]
        centre = obs[target]["position_vox"]
        crop = ctx.crop
        offsets = [(dx, dy, dz) for dx in range(-LIE_RADIUS_VOX, LIE_RADIUS_VOX + 1)
                   for dy in range(-LIE_RADIUS_VOX, LIE_RADIUS_VOX + 1) for dz in range(-LIE_RADIUS_VOX, LIE_RADIUS_VOX + 1)]
        rng.shuffle(offsets)
        for dx, dy, dz in offsets:
            v = (centre[0] + dx, centre[1] + dy, centre[2] + dz)
            if all(0 <= c < s for c, s in zip(v, crop.brain.shape)) and crop.brain[v] and _ppm(crop.p_tumour[v]) < LIE_BELOW_PPM:
                claim = rng.randint(LIE_ABOVE_PPM, 990_000)
                ctx.board.append("evidence", f"squad-{liar}", {
                    "squad": liar, "pos_vox": list(v), "look_p_ppm": claim, "p_tumour_ppm": claim, "dose_ug": 0,
                    "released_ug": 0}, tick)
                return

    def _check_claims(self, ctx: _Ctx, obs: Sequence[Mapping[str, Any]], looks: Sequence[Optional[float]],
                      tick: int) -> None:
        """Another squad's local look decides: a claim of tumour (>= 0.9) next to a squad whose look says otherwise
        (< 0.5) is contradicted; next to a squad whose look agrees (>= 0.5) it is corroborated."""
        rep = ctx.reputation
        for e in ctx.board.entries()[::-1]:
            if e.tick != tick:
                break
            if e.type != "evidence" or e.payload["look_p_ppm"] < LIE_ABOVE_PPM:
                continue
            author = e.author["agent"]
            best = None
            for j, o in enumerate(obs):
                if f"squad-{j}" == author or looks[j] is None or j in ctx.liars:
                    continue
                d = max(abs(a - b) for a, b in zip(o["position_vox"], e.payload["pos_vox"]))
                if d <= VERIFY_RADIUS_VOX and (best is None or (d, j) < best[:2]):
                    best = (d, j)
            if best is not None:
                rep.report(author, e.id, looks[best[1]] < 0.5, tick)

    # -- decision --

    async def _attempts(self, ctx: _Ctx, name: str, req: DecisionRequest, trail: Optional[List[Dict[str, Any]]],
                        slot: str) -> DecisionResponse:
        """One backend call under the decision failure policy (goal 8), identical for every backend, arm and question set.

        A :class:`DecisionError` is retried up to ``DECISION_RETRIES`` times with the same request; on retries 2 and 3 a
        verbalised backend also gets a one-line corrective note (not hashed, so a good reply is cached under the
        original request hash). Usage the provider billed for a rejected reply is tallied. After the last failure
        :class:`_DecisionFailed` carries the reason. A :class:`DecisionAbort` (spent budget) is never absorbed.

        ``slot`` names this decision or look in the run. The failed attempts a run experienced are kept in
        ``ctx.attempt_log[slot]`` (reason and billed usage) and go into the bundle; in a replay they are re-played from
        there (``ctx.replay_attempts``) without calling the backend, and only then does the backend (the offline cache)
        serve the stored reply. Nothing is read from a failures file shared with other runs."""
        backend = self.workers[name].backend
        verbalised = backend.describe().get("prob_method") == "verbalised"
        replayed = None if ctx.replay_attempts is None else list(ctx.replay_attempts.get(slot, []))
        reason = ""
        for attempt in range(DECISION_RETRIES + 1):
            note = CORRECTIVE_NOTE.format(reason=reason) if (attempt >= 2 and verbalised) else ""
            billed = None
            if replayed is not None and attempt < len(replayed):
                reason, billed = str(replayed[attempt]["error"]), replayed[attempt].get("billed")
            else:
                sent = dataclasses.replace(req, note=note) if note else req
                try:
                    resp = await backend.decide(sent)
                except DecisionAbort:
                    raise
                except DecisionError as exc:
                    reason, billed = scrub(str(exc))[:160], exc.billed
                else:
                    if trail is not None and note:
                        trail.append({"worker": name, "attempt": attempt, "error": None, "note": note})
                    self._tally(ctx, resp)
                    return resp
                if replayed is None:
                    ctx.attempt_log.setdefault(slot, []).append({"attempt": attempt, "error": reason, "billed": billed})
            if billed:
                ctx.tokens_in += int(billed.get("input_tokens", 0))
                ctx.tokens_out += int(billed.get("output_tokens", 0))
                ctx.cost += float(billed.get("cost_usd", 0.0))
            if trail is not None:
                trail.append({"worker": name, "attempt": attempt, "error": reason, "note": note, "billed": billed})
        raise _DecisionFailed(name, reason)

    async def _call(self, ctx: _Ctx, name: str, fields: Mapping[str, Any], text: str, brief: briefs_mod.Brief,
                    tick: int, squad: int, trail: Optional[List[Dict[str, Any]]] = None
                    ) -> Tuple[DecisionRequest, DecisionResponse]:
        worker = self.workers[name]
        req = build_request(worker.model, fields, text, seed=derive_seed(ctx.seed, f"squad-{squad}", tick))
        if ctx.spec.board_on:
            req = dataclasses.replace(req, brief_ids=tuple(brief.brief_ids))
        return req, await self._attempts(ctx, name, req, trail, f"decision|{squad}|{tick}|{name}")

    def _call_info(self, name: str, resp: DecisionResponse) -> Dict[str, Any]:
        t = dict(resp.timing)
        return {"worker": name, "model": resp.served_model, "revision": (resp.identity or {}).get("revision", "none"),
                "hardware": self.hardware.get(name, "unknown"), "server_ms": int(t.get("server_ms", 0)),
                "network_ms": int(t.get("network_ms", 0)), "queue_ms": int(t.get("queue_ms", 0)),
                "tokens_in": int(resp.input_tokens), "tokens_out": int(resp.output_tokens),
                "cost_usd": float(resp.cost_usd), "prob_method": resp.prob_method, "cached": bool(resp.cached)}

    async def _decide(self, ctx: _Ctx, i: int, fields: Mapping[str, Any], look: Optional[float],
                      brief: briefs_mod.Brief, tick: int, rnd: int, sprime: Optional[Mapping[str, str]]) -> Dict[str, Any]:
        """The squad's decision. If a backend call fails for good (see :meth:`_call`) the DECISION is an error, not the
        run: the squad takes the neutral fallback (target S, release no) and the run goes on."""
        trail: List[Dict[str, Any]] = []
        try:
            rec = await self._decide_models(ctx, i, fields, look, brief, tick, rnd, sprime, trail)
        except _DecisionFailed as failed:
            return self._error_record(failed, brief, fields, look, sprime, trail)
        rec["trail"] = trail
        return rec

    def _error_record(self, failed: _DecisionFailed, brief: briefs_mod.Brief, fields: Mapping[str, Any],
                      look: Optional[float], sprime: Optional[Mapping[str, str]], trail: List[Dict[str, Any]]
                      ) -> Dict[str, Any]:
        decision = {"target": FALLBACK_TARGET, "release": FALLBACK_RELEASE, "amount": FALLBACK_AMOUNT,
                    "p_target": 0.0, "p_release": 0.0, "p_amount": 0.0}
        final = DecisionResponse(answers={}, served_model="", request_hash="")
        return {"decision": decision, "final": final, "final_name": failed.worker, "pick": failed.worker, "deferred": 0,
                "escalated": 0, "calls": [], "counterfactual": None, "audit": None, "features": None,
                "sprime": dict(sprime) if sprime else None, "look": look, "brief": brief, "fields": fields,
                "trail": trail, "error": f"decision_error: {failed.reason}"}

    async def _decide_models(self, ctx: _Ctx, i: int, fields: Mapping[str, Any], look: Optional[float],
                             brief: briefs_mod.Brief, tick: int, rnd: int, sprime: Optional[Mapping[str, str]],
                             trail: List[Dict[str, Any]]) -> Dict[str, Any]:
        spec = ctx.spec
        text = self._compact(brief, with_ids=spec.board_on)
        used: List[Tuple[str, DecisionResponse]] = []
        counterfactual = audit = features = None
        deferred = escalated = 0
        key = f"{ctx.arm}|{ctx.case}|{ctx.seed}|{i}|{rnd}"

        async def go(name: str) -> Tuple[DecisionRequest, DecisionResponse]:
            return await self._call(ctx, name, fields, text, brief, tick, i, trail)

        if spec.kind == "fw":
            req, final = await go(FW)
            final_name = pick = FW
            used.append((FW, final))
        elif ctx.calibration:
            got = await asyncio.gather(go(Q9), go(D1), go(Q4), return_exceptions=True)
            for g in got:  # every call runs to its end first; the first failure then decides
                if isinstance(g, BaseException):
                    raise g
            (req, final), (r1, d1r), (r4, q4r) = got
            final_name = pick = Q9
            used.append((Q9, final))
            counterfactual = {D1: self._cf(r1, d1r), Q4: self._cf(r4, q4r)}
            counterfactual[D1]["call"] = self._call_info(D1, d1r)
            counterfactual[Q4]["call"] = self._call_info(Q4, q4r)
            features = router_features(fields, d1r.answers)  # logged so scripts/v41_fit.py can train the router
        else:
            req1, d1r = await go(D1)
            used.append((D1, d1r))
            features = router_features(fields, d1r.answers)
            pick = ctx.router.predict(features)
            ctx.picks[pick] += 1
            if pick not in self.workers:
                raise HarnessError(f"router picked {pick!r}, which is not a worker")
            if pick == D1:
                result = ctx.deferral.decide_answers(req1, d1r.answers)
                deferred = int(result.deferred)
                ctx.monitor.record(result.deferred)
                audited = audit_selected(key)
                req, final, final_name = req1, d1r, D1
                if deferred or audited:
                    treq, tr = await go(ctx.target_name)
                    if audited:
                        ctx.audits.append((result.sets, decision_labels(treq, tr.answers)))
                        audit = {"sets": {q: list(s) for q, s in result.sets.items()},
                                 "target_labels": decision_labels(treq, tr.answers), "used": bool(deferred),
                                 "call": self._call_info(ctx.target_name, tr)}
                    if deferred:
                        req, final, final_name = treq, tr, ctx.target_name
                        used.append((ctx.target_name, tr))
            else:
                ctx.monitor.record(False)
                req, final = await go(pick)
                final_name = pick
                used.append((pick, final))
                if pick == Q4 and max(final.answers["target"].probabilities.values()) < ctx.harness["router"]["slm_conf_min"]:
                    req, final = await go(ctx.target_name)
                    escalated, final_name = 1, ctx.target_name
                    used.append((ctx.target_name, final))
        decision = decide_from_answers(final.answers, req)
        return {"decision": decision, "final": final, "final_name": final_name, "pick": pick, "deferred": deferred,
                "escalated": escalated, "calls": [self._call_info(n, r) for n, r in used],
                "counterfactual": counterfactual, "audit": audit, "features": features,
                "sprime": dict(sprime) if sprime else None, "look": look, "brief": brief, "fields": fields}

    @staticmethod
    def _cf(req: DecisionRequest, resp: DecisionResponse) -> Dict[str, Any]:
        d = decide_from_answers(resp.answers, req)
        return {"target": d["target"], "release": d["release"], "amount": d["amount"], "model": resp.served_model,
                "probs": {q: dict(a.probabilities) for q, a in resp.answers.items()}}

    def _apply(self, ctx: _Ctx, i: int, rec: Dict[str, Any], tick: int) -> None:
        d, final, brief = rec["decision"], rec["final"], rec["brief"]
        ident = dict(final.identity or {})
        revision = ident.get("revision", "none") or "none"
        board_ids = set(brief.brief_ids)
        raw = list(final.cites) if ctx.spec.board_on else []
        seen = self._view(ctx) if ctx.spec.board_on else None
        on_board = [c for c in raw if seen is not None and c in seen]
        ctx.dropped_cites += len(raw) - len(on_board)
        ctx.all_cites += len(raw)
        ctx.valid_cites += sum(1 for c in raw if c in board_ids)
        entry_id = None
        payload = {"squad": i, "round": tick // self.world_config.decision_every, "target": d["target"],
                   "release": int(d["release"]), "amount_level": int(d["amount"]), "p_target_ppm": _ppm(d["p_target"]),
                   "p_release_ppm": _ppm(d["p_release"]), "p_amount_ppm": _ppm(d["p_amount"]),
                   "backend": rec["final_name"], "deferred": rec["deferred"], "router_pick": rec["pick"],
                   "brief_ids": list(brief.brief_ids) if ctx.spec.board_on else []}
        if ctx.spec.board_on:
            author = {"agent": f"squad-{i}", "model": final.served_model or "none", "rev": revision}
            entry_id = ctx.board.append("decision", author, payload, tick, cites=on_board).id
        error = rec.get("error")
        if error:
            ctx.decision_errors += 1
            ctx.board.append("alarm", "harness", {"kind": "other", "value_ppm": 0, "note": error[:140]}, tick)
        if final.alarm:
            ctx.board.append("alarm", "harness", dict(final.alarm), tick)
        world_outcome = ctx.world.apply(f"squad-{i}", Intent(ACTION_DECIDE, {"target": d["target"], "release": d["release"],
                                                                              "amount": d["amount"]}), tick)
        if not world_outcome.accepted:
            raise HarnessError(f"world refused squad-{i}'s decision at tick {tick}: {world_outcome.reason}")
        calls = rec["calls"]
        look = rec["look"]
        ctx.log.append({
            "run_id": ctx.run_id, "arm": ctx.arm, "case": ctx.case, "seed": ctx.seed, "squad": i, "round": payload["round"],
            "tick": tick, "backend": rec["final_name"], "model": final.served_model, "revision": revision,
            "hardware": self.hardware.get(rec["final_name"], "unknown"),
            "server_ms": sum(c["server_ms"] for c in calls), "network_ms": sum(c["network_ms"] for c in calls),
            "queue_ms": sum(c["queue_ms"] for c in calls), "tokens_in": sum(c["tokens_in"] for c in calls),
            "tokens_out": sum(c["tokens_out"] for c in calls), "cost_usd": sum(c["cost_usd"] for c in calls),
            "router_pick": rec["pick"], "deferred": rec["deferred"], "escalated": rec["escalated"],
            "cited_ids": raw, "brief_ids": list(brief.brief_ids) if ctx.spec.board_on else [],
            "target": d["target"], "release": int(d["release"]), "amount": int(d["amount"]),
            "p_target_ppm": payload["p_target_ppm"], "p_release_ppm": payload["p_release_ppm"],
            "p_amount_ppm": payload["p_amount_ppm"], "look_p_ppm": None if look is None else _ppm(look),
            "prob_method": final.prob_method, "identity": ident, "calls": calls,
            "counterfactual": rec["counterfactual"], "audit": rec["audit"], "features": rec.get("features"),
            "sprime": rec["sprime"], "entry_id": entry_id, "kind": "decision", "error": bool(error), "error_reason": error,
            "attempts": sum(1 for t in rec["trail"] if t["error"]) + (0 if error else 1),
            "notes": [t["note"] for t in rec["trail"] if t["note"]],
            "failures": [{"attempt": t["attempt"], "error": t["error"], "billed": t["billed"]}
                         for t in rec["trail"] if t["error"]]})

    # ---- finishing -------------------------------------------------------------------------------------

    def _router_sha(self, spec: ArmSpec) -> Optional[str]:
        router = self.bonsai_router if spec.bonsai else self.router
        if self.router_kind != "frozen" or router is None:
            return None
        return str(router.sha256)

    @staticmethod
    def _identities(ctx: "_Ctx") -> Optional[str]:
        ids = sorted({str((r.get("identity") or {}).get("served_model") or "none") for r in ctx.log if not r["error"]})
        return ",".join(ids) if ids else None

    def _finish(self, ctx: _Ctx, t0: float, marks: Mapping[str, Any]) -> EpisodeResult:
        metrics: Dict[str, Any] = marks["metrics"]
        board = ctx.board
        spec = ctx.spec
        anchors: List[Dict[str, Any]] = []
        if board is not None:
            alarm = ctx.monitor.finish() if ctx.deferral is not None else None
            if alarm:
                board.append("alarm", "harness", alarm, self.world_config.ticks)
            if ctx.anchorer is not None:
                ctx.anchorer.finalize(board)
                ctx.anchorer.refresh_chain()
                anchors = list(ctx.anchorer.records)
        wall_s = self.clock() - t0
        n_dec = len(ctx.log)
        defers = sum(r["deferred"] for r in ctx.log)
        shared = ctx.shared and board is not None
        board_ro: Any = None
        if shared:
            board_ro = BriefView(board, ctx.visible) if ctx.visible is not None else board
        stats = (self.perception_stats(ctx.case) if self.perception_stats else {}) or {}
        t50 = metrics.get("time_to_50_coverage")
        wall_t50 = marks.get("t50_wall")
        top_share = None
        if ctx.picks:
            top_share = max(ctx.picks.values()) / sum(ctx.picks.values())
        audit_cov = audit_coverage(ctx.audits).get("all") if ctx.audits else None
        cites_n = ctx.all_cites
        vis = int(statistics.median(ctx.vis_latency)) if ctx.vis_latency else None
        queen_tin = ctx.queen_provider.tokens_in if ctx.queen_provider else 0
        queen_tout = ctx.queen_provider.tokens_out if ctx.queen_provider else 0
        cell: Dict[str, Any] = {
            "run_id": ctx.run_id, "arm": ctx.arm, "case": ctx.case, "seed": ctx.seed, "split": ctx.split,
            "wall_s": float(wall_s), "ticks_to_50": t50, "seconds_to_50": None if wall_t50 is None else float(wall_t50),
            "queen_calls": ctx.queen.calls if ctx.queen else 0,
            **{f"queen_fallbacks_{c}": (ctx.queen.fallbacks[c] if ctx.queen else 0) for c in FALLBACK_CAUSES},
            "board_size": board.size() if shared else 0, "anchors": len(anchors), "visibility_latency_ms": vis,
            "coverage": float(metrics["coverage"]), "off_target": float(metrics["off_target_share"]),
            "defer_rate": (defers / n_dec) if n_dec else 0.0, "citations": cites_n,
            "unread_frac": briefs_mod.never_read_share(board_ro) if shared else None,
            "brief_age_median": briefs_mod.median_entry_age_in_briefs(board_ro) if shared else None,
            "router_top_share": top_share,
            "perception_ece": stats.get("perception_ece"),
            "cost_usd": float(ctx.cost + (ctx.queen_provider.cost if ctx.queen_provider else 0.0)),
            "tokens_in": ctx.tokens_in + queen_tin, "tokens_out": ctx.tokens_out + queen_tout, "pod_minutes": 0.0,
            "gas": None, "alarms": ctx.alarm_count,
            "audit_conformal_coverage": audit_cov,
            "citation_validity": (ctx.valid_cites / cites_n) if cites_n else None,
            "uncited_decisions": sum(1 for r in ctx.log if not r["cited_ids"]),
            "perception_dice": stats.get("perception_dice"), "locallook_ece": stats.get("locallook_ece"),
            "truth_outside_crop": float(metrics["truth_share_outside_crop"]),
            "backends_kind": self.backends_kind, "router_kind": self.router_kind,
            "router_sha256": self._router_sha(ctx.spec), "identities": self._identities(ctx),
            "decision_errors": ctx.decision_errors, "look_errors": ctx.look_errors,
            "decision_error_share": (ctx.decision_errors / n_dec) if n_dec else 0.0,
            "decision_error_alarm": int(bool(n_dec) and ctx.decision_errors / n_dec > DECISION_ERROR_ALARM),
            "look_temperature": self.look_temperature, "look_temperature_source": self.look_temperature_source,
        }
        problems = v41_cells.validate_cell(cell)
        if problems:
            raise HarnessError("cell failed validation: " + "; ".join(problems))
        trace_hash = content_hash({"decisions": [{k: r[k] for k in _TRACE_KEYS} for r in ctx.log],
                                   "metrics": {k: metrics[k] for k in ("coverage", "off_target_share", "released_total",
                                                                       "time_to_50_coverage")},
                                   "entry_sites": [list(s) for s in ctx.world.entry_sites]})
        run_log = {
            "run_id": ctx.run_id, "arm": ctx.arm, "case": ctx.case, "seed": ctx.seed, "split": ctx.split,
            "wall_s": wall_s, "ticks": int(metrics["ticks"]), "ticks_to_50": t50, "seconds_to_50": cell["seconds_to_50"],
            "queen_calls": cell["queen_calls"],
            "queen_fallbacks": {c: cell[f"queen_fallbacks_{c}"] for c in FALLBACK_CAUSES},
            "queen_outage_retries": ctx.queen.outage.retries if ctx.queen else 0,
            "board_size": cell["board_size"], "anchors": len(anchors),
            "anchor_latency_s": ({m: list(v) for m, v in ctx.anchorer.latency_s.items()} if ctx.anchorer else {}),
            "visibility_latency_ms": vis, "decisions": n_dec, "rounds": self.world_config.rounds,
            "shared_store": shared, "calibration": ctx.calibration, "variant": spec.variant,
            "alarms": ctx.alarm_count, "dropped_cites": ctx.dropped_cites,
            "identities": sorted({str((r.get("identity") or {}).get("served_model") or "none")
                                  for r in ctx.log if not r["error"]}),
            "decision_errors": ctx.decision_errors, "look_errors": ctx.look_errors,
            "backends_kind": self.backends_kind, "router_kind": self.router_kind,
            "look_temperature": self.look_temperature, "look_temperature_source": self.look_temperature_source,
            "reputation": dict(ctx.reputation.score) if ctx.reputation else {}, "d_timeouts": ctx.d_timeouts,
            "errors": [scrub(e) for e in list(ctx.errors) + ([f"{e['mode']}: {e['error']}" for e in ctx.anchorer.errors] if ctx.anchorer else [])]}
        bundle = self._bundle(ctx, cell, metrics, trace_hash, board if shared else None, anchors)
        run_dir = None
        if self.out_dir is not None:
            run_dir = self._write(ctx, cell, run_log, bundle, board)
        if self.outbox is not None:
            self.outbox.enqueue(bundle)
        return EpisodeResult(cell, bundle, ctx.log, run_log, board, anchors, run_dir, trace_hash, ctx.reputation)

    def _bundle(self, ctx: _Ctx, cell: Mapping[str, Any], metrics: Mapping[str, Any], trace_hash: str,
                board: Optional[Board], anchors: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        spec = {"study": "v4.1", "world": "tumor_mri", "arm": ctx.arm, "case": ctx.case, "seed": ctx.seed,
                "calibration": ctx.calibration, "variant": ctx.spec.variant, "split": ctx.split}
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None}
        result = RunResult(run_id=ctx.run_id, arm=ctx.arm, scope=SCOPE, ticks=int(metrics["ticks"]),
                           metrics={**numeric, "coverage_at_queen": {str(k): v for k, v in metrics["coverage_at_queen"].items()}},
                           trace_hash=trace_hash,
                           config_hash=content_hash({"world": ctx.header["world_cfg_sha256"],
                                                     "harness": ctx.header["harness_cfg_sha256"], "arm": ctx.arm}),
                           event_count=board.size() if board else len(ctx.log), policy_failures=0, signals_deposited=0,
                           signals_rejected=0)
        extra = {"cell": dict(cell), "anchor_sizes": anchor_sizes(board.size()) if (board and ctx.spec.anchored) else [],
                 "anchors": [dict(a) for a in anchors], "board_head": board.get(board.size() - 1).id if board and board.size() else None,
                 "board_root": board.root().hex() if board and board.size() else None,
                 "decision_log_sha256": content_hash([{k: r[k] for k in _TRACE_KEYS} for r in ctx.log]),
                 "shared_store": bool(board),
                 "attempts": {k: [dict(a) for a in v] for k, v in sorted(ctx.attempt_log.items())}}
        return build_bundle(result, spec, extra=extra)

    def _write(self, ctx: _Ctx, cell: Mapping[str, Any], run_log: Mapping[str, Any], bundle: Mapping[str, Any],
               board: Optional[Board]) -> Path:
        run_dir = self._run_dir(ctx.run_id, create=True)
        if board is not None:
            board.export_jsonl(run_dir / f"{ctx.run_id}.jsonl")
        (run_dir / "decisions.jsonl").write_text(
            "".join(canonical_json(r) + "\n" for r in [*ctx.log, *ctx.look_rows]), encoding="utf-8")
        (run_dir / "run_log.json").write_text(json.dumps(run_log, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "cell.json").write_text(json.dumps(cell, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "bundle.json").write_text(canonical_json(dict(bundle)) + "\n", encoding="utf-8")
        return run_dir


def append_cell(path: Union[str, Path], cell: Mapping[str, Any]) -> None:
    """One JSON line per run (``cells.jsonl``)."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(cell), sort_keys=True) + "\n")


def error_cell(arm: str, case: str, seed: int, split: str, error: BaseException, *,
               backends_kind: Optional[str] = None, router_kind: Optional[str] = None) -> Dict[str, Any]:
    """A failed run's cell: required fields zeroed or null, plus ``error`` (the analysis excludes it)."""
    cell: Dict[str, Any] = {f.name: (None if f.nullable else (0 if f.kind == "int" else 0.0)) for f in v41_cells.FIELDS}
    cell.update(run_id="0" * 64, arm=arm, case=case, seed=seed, split=split, backends_kind=backends_kind,
                router_kind=router_kind, error=scrub(f"{type(error).__name__}: {error}")[:300])
    return cell
