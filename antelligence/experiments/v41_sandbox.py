"""Sandbox self-improvement loop for the v4.1 queen (goal 4.6; Phase 0, development cases only).

Each round the queen proposes ONE edit to ``config/v4_1/harness.yaml``: a JSON patch limited to the allowed
keys, plus a numeric prediction. The edit runs on every dev-A case under both SW-B and SW-noBoard and is paired
against the current harness (whose results are cached). Score = mean paired coverage change across the two
arms, in coverage points. Keep when score >= +1 point, the edit wins >= 60% of the case-arm pairs that
differ, and the sign of the prediction is right; otherwise revert. At most 4 rounds. Every edit, kept or not,
becomes an ``edit`` board entry. Kept edits that do not lose on dev-B are promotion candidates (the lead
approves them at G1; nothing is promoted here). The proposer sees aggregates only, never per-case truth.

``evaluate(harness, arm, case_ids)`` is injected: it runs the worlds and returns ``{case_id: coverage}`` (0..1)
or an :class:`EvalResult`. Nothing here calls a model except an explicit :func:`llm_proposer`.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from antelligence.kernel.board import Board
from antelligence.kernel.canonical import canonical_json
from antelligence.kernel.policies.v41_harness import HarnessError, harness_sha256, load_allowed_keys, validate_harness
from antelligence.providers.base import ChatRequest, Provider
from antelligence.providers.outage import OutagePolicy, complete_with_outage_retry

ARMS = ("SW-B", "SW-noBoard")
MAX_ROUNDS = 4
MIN_GAIN_POINTS = 1.0
MIN_WIN_RATE = 0.6
DEV_A_SIZE = 30
DEV_B_SIZE = 10
PPM_PER_POINT = 10_000
EPS = 1e-9
FORBIDDEN_SEGMENTS = frozenset({"scorer", "physics", "verifier", "conformal", "budget", "budgets", "key", "keys", "cap", "caps",
                                "allowed_keys", "allowed-keys"})
OPS = ("replace", "add")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class PatchError(ValueError):
    """The proposed edit is malformed or touches a key it may not."""


@dataclass(frozen=True)
class EvalResult:
    coverage: Mapping[str, float]                  # case_id -> coverage fraction 0..1
    stats: Mapping[str, float] = field(default_factory=dict)   # aggregate router/deferral stats, if the runner has them


Evaluate = Callable[[Mapping[str, Any], str, Sequence[str]], Union[EvalResult, Mapping[str, float]]]
Proposer = Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]]


@dataclass(frozen=True)
class Proposal:
    patch: Tuple[Mapping[str, Any], ...]
    prediction: Mapping[str, Any]

    @property
    def keys(self) -> List[str]:
        return [op["path"] for op in self.patch]

    @property
    def patch_sha256(self) -> str:
        return hashlib.sha256(canonical_json({"patch": list(self.patch), "prediction": dict(self.prediction)}).encode()).hexdigest()


# --------------------------------------------------------------------------------------------------
# Patch parsing
# --------------------------------------------------------------------------------------------------

def _segments(path: str) -> List[str]:
    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise PatchError(f"bad path {path!r}")
    return [s.replace("~1", "/").replace("~0", "~") for s in path[1:].split("/")]


def _is_allowed(path: str, allowed_keys: Sequence[str]) -> bool:
    """Leaf paths only: the path must equal an allowed key exactly (no children, indices, '-', escapes or '..')."""
    return path in allowed_keys


def _no_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise PatchError(f"duplicate key {k!r} in proposal JSON")
        out[k] = v
    return out


def parse_proposal(raw: Any, allowed_keys: Sequence[str]) -> Proposal:
    """Validate shape, forbidden keys, allowed keys and the numeric prediction. Raise :class:`PatchError` otherwise."""
    if isinstance(raw, str):
        text = raw.strip()
        fenced = _FENCE.match(text)
        try:
            raw = json.loads(fenced.group(1) if fenced else text, object_pairs_hook=_no_duplicates)
        except json.JSONDecodeError as exc:
            raise PatchError(f"not JSON ({exc.msg})") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"patch", "prediction"}:
        raise PatchError("proposal needs exactly 'patch' and 'prediction'")
    ops = raw["patch"]
    if not isinstance(ops, list) or not ops or len(ops) > 4:
        raise PatchError("patch must be a list of 1-4 operations")
    clean: List[Dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, Mapping) or set(op) != {"op", "path", "value"} or op["op"] not in OPS:
            raise PatchError(f"each operation needs exactly op ({'/'.join(OPS)}), path, value")
        segs = _segments(op["path"])
        bad = [s for s in segs if s.lower() in FORBIDDEN_SEGMENTS]
        if bad:
            raise PatchError(f"forbidden key {bad[0]!r} (scorer, physics, verifier, conformal, budgets, keys, caps, allowed_keys)")
        if not _is_allowed(op["path"], allowed_keys):
            raise PatchError(f"{op['path']!r} is not an allowed key")
        clean.append({"op": op["op"], "path": op["path"], "value": copy.deepcopy(op["value"])})
    pred = raw["prediction"]
    if not isinstance(pred, Mapping) or set(pred) != {"metric", "delta_ppm"} or not isinstance(pred["metric"], str) or not pred["metric"]:
        raise PatchError("prediction needs {metric: str, delta_ppm: int}")
    d = pred["delta_ppm"]
    if not isinstance(d, int) or isinstance(d, bool) or d == 0:
        raise PatchError("prediction.delta_ppm must be a non-zero integer (a numeric, signed prediction is required)")
    return Proposal(patch=tuple(clean), prediction={"metric": pred["metric"], "delta_ppm": d})


def apply_patch(harness: Mapping[str, Any], proposal: Proposal, allowed_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Apply the patch to a deep copy and re-validate the whole harness. Re-checks ops and paths (defence in depth)."""
    if allowed_keys is None:
        allowed_keys = load_allowed_keys()
    out = copy.deepcopy(dict(harness))
    for op in proposal.patch:
        if op["op"] not in OPS or not _is_allowed(op["path"], allowed_keys) \
                or any(s.lower() in FORBIDDEN_SEGMENTS for s in _segments(op["path"])):
            raise PatchError(f"{op['path']!r} is not an allowed leaf key")
        segs = _segments(op["path"])
        node: Any = out
        for s in segs[:-1]:
            try:
                node = node[int(s)] if isinstance(node, list) else node[s]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise PatchError(f"path {op['path']!r} does not exist") from exc
        last = segs[-1]
        if isinstance(node, list):
            if op["op"] == "add" and last == "-":
                node.append(op["value"])
                continue
            try:
                idx = int(last)
                if op["op"] == "add":
                    node.insert(idx, op["value"])
                else:
                    node[idx] = op["value"]
            except (ValueError, IndexError) as exc:
                raise PatchError(f"bad list index in {op['path']!r}") from exc
        elif isinstance(node, dict):
            if last not in node:
                raise PatchError(f"{op['path']!r} does not exist; the patch may only change existing keys")
            node[last] = op["value"]
        else:
            raise PatchError(f"path {op['path']!r} does not exist")
    try:
        validate_harness(out)
    except HarnessError as exc:
        raise PatchError(f"patched harness is invalid: {exc}") from exc
    return out


# --------------------------------------------------------------------------------------------------
# Scoring and keep rule
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedScore:
    score_points: float          # mean over arms of the mean paired coverage change, in points
    wins: int
    losses: int
    ties: int
    win_rate: float              # wins / (wins + losses), 0.0 when nothing differs
    per_arm_points: Mapping[str, float]
    n_pairs: int


def paired_score(current: Mapping[str, Mapping[str, float]], candidate: Mapping[str, Mapping[str, float]],
                 arms: Sequence[str] = ARMS) -> PairedScore:
    """Pair candidate against current on the same (arm, case) cells. Coverage fractions become points (x100)."""
    wins = losses = ties = 0
    per_arm: Dict[str, float] = {}
    for arm in arms:
        cases = sorted(set(current[arm]) & set(candidate[arm]))
        if not cases:
            raise ValueError(f"no shared cases for arm {arm}")
        diffs = [(candidate[arm][c] - current[arm][c]) * 100.0 for c in cases]
        per_arm[arm] = sum(diffs) / len(diffs)
        for d in diffs:
            if d > EPS:
                wins += 1
            elif d < -EPS:
                losses += 1
            else:
                ties += 1
    differing = wins + losses
    return PairedScore(score_points=sum(per_arm.values()) / len(per_arm), wins=wins, losses=losses, ties=ties,
                       win_rate=(wins / differing) if differing else 0.0, per_arm_points=per_arm, n_pairs=wins + losses + ties)


def keep_rule(score: PairedScore, predicted_delta_ppm: int) -> bool:
    """Keep iff score >= +1 point, wins >= 60% of differing pairs, and the prediction's sign matches the score's."""
    sign_ok = (predicted_delta_ppm > 0) == (score.score_points > 0) and predicted_delta_ppm != 0 and score.score_points != 0
    return score.score_points >= MIN_GAIN_POINTS - EPS and score.win_rate >= MIN_WIN_RATE - EPS and sign_ok


# --------------------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundRecord:
    round: int
    keys: Tuple[str, ...]
    prediction: Mapping[str, Any]
    patch_sha256: str
    rejected: str                       # "" when the patch was valid
    score_points: Optional[float]
    win_rate: Optional[float]
    kept: bool
    edit_entry_id: str
    proposal: Any = None                              # the raw proposal as the queen sent it (JSON-able), valid or not
    diff: Tuple[Mapping[str, Any], ...] = ()          # per operation: op, path, before, after (valid proposals only)
    candidate_sha256: Optional[str] = None
    per_arm_points: Mapping[str, float] = field(default_factory=dict)   # dev-A paired change per arm, in points
    wins: int = 0
    losses: int = 0
    ties: int = 0


@dataclass(frozen=True)
class SandboxResult:
    base_harness: Mapping[str, Any]
    final_harness: Mapping[str, Any]
    rounds: Tuple[RoundRecord, ...]
    promotion_candidates: Tuple[int, ...]      # kept rounds that do not lose on dev-B
    dev_b_scores: Mapping[int, float]          # round -> dev-B paired points
    evaluations: int                           # evaluate() calls actually made (cache misses)
    dev_b_detail: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)   # round -> points, per arm, wins/losses/ties


class _Cache:
    def __init__(self, evaluate: Evaluate) -> None:
        self._evaluate = evaluate
        self._store: Dict[Tuple[str, str, Tuple[str, ...]], EvalResult] = {}
        self.misses = 0

    def get(self, harness: Mapping[str, Any], arm: str, cases: Sequence[str]) -> EvalResult:
        key = (harness_sha256(harness), arm, tuple(cases))
        if key not in self._store:
            self.misses += 1
            res = self._evaluate(copy.deepcopy(dict(harness)), arm, list(cases))
            if not isinstance(res, EvalResult):
                res = EvalResult(coverage=dict(res))
            self._store[key] = res
        return self._store[key]

    def coverage(self, harness: Mapping[str, Any], cases: Sequence[str]) -> Dict[str, Dict[str, float]]:
        return {arm: dict(self.get(harness, arm, cases).coverage) for arm in ARMS}


def aggregates(cache: _Cache, harness: Mapping[str, Any], cases: Sequence[str]) -> Dict[str, Any]:
    """What the queen may see: per-arm means and runner stats. No case ids, no per-case values."""
    out: Dict[str, Any] = {}
    for arm in ARMS:
        res = cache.get(harness, arm, cases)
        vals = list(res.coverage.values())
        out[arm] = {"n_cases": len(vals), "mean_coverage_points": round(100.0 * sum(vals) / len(vals), 3),
                    "stats": {k: round(float(v), 4) for k, v in sorted(res.stats.items())}}
    return out


def run_sandbox(
    board: Board,
    harness: Mapping[str, Any],
    allowed_keys: Sequence[str],
    dev_a: Sequence[str],
    dev_b: Sequence[str],
    evaluate: Evaluate,
    propose: Proposer,
    *,
    max_rounds: int = MAX_ROUNDS,
    on_round: Optional[Callable[["RoundRecord"], None]] = None,
) -> SandboxResult:
    """Run up to 4 rounds. ``propose`` gets aggregates and previous edits only and returns a raw proposal (or None to stop)."""
    validate_harness(harness)
    cache = _Cache(evaluate)
    base = copy.deepcopy(dict(harness))
    current = copy.deepcopy(base)
    history: List[Dict[str, Any]] = []
    records: List[RoundRecord] = []
    states: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}   # kept round -> (before, after)
    for rnd in range(1, min(max_rounds, MAX_ROUNDS) + 1):
        ctx = {"round": rnd, "harness": copy.deepcopy(current), "allowed_keys": list(allowed_keys),
               "aggregates": aggregates(cache, current, dev_a), "previous_edits": copy.deepcopy(history)}
        raw = propose(ctx)
        if raw is None:
            break
        rejected = ""
        score: Optional[PairedScore] = None
        kept = False
        try:
            proposal = parse_proposal(raw, allowed_keys)
            candidate = apply_patch(current, proposal, allowed_keys)
            if harness_sha256(candidate) == harness_sha256(current):
                raise PatchError("the edit changes nothing")
        except PatchError as exc:
            rejected = str(exc)[:140]
            proposal = None
            candidate = None
        if proposal is not None and candidate is not None:
            score = paired_score(cache.coverage(current, dev_a), cache.coverage(candidate, dev_a))
            kept = keep_rule(score, proposal.prediction["delta_ppm"])
            keys, pred, sha = proposal.keys, dict(proposal.prediction), proposal.patch_sha256
        else:
            keys, pred = _salvage_keys(raw), _salvage_prediction(raw)
            sha = hashlib.sha256(canonical_json(_jsonable(raw)).encode()).hexdigest()
        entry = board.append("edit", "harness", {"round": rnd, "patch_sha256": sha, "keys": keys, "prediction": pred,
                                                 "kept": 1 if kept else 0}, tick=rnd)
        records.append(RoundRecord(
            round=rnd, keys=tuple(keys), prediction=pred, patch_sha256=sha, rejected=rejected,
            score_points=None if score is None else score.score_points,
            win_rate=None if score is None else score.win_rate, kept=kept, edit_entry_id=entry.id,
            proposal=_jsonable(raw), diff=_diff(current, candidate, proposal) if (proposal and candidate) else (),
            candidate_sha256=harness_sha256(candidate) if candidate is not None else None,
            per_arm_points=dict(score.per_arm_points) if score else {},
            wins=score.wins if score else 0, losses=score.losses if score else 0, ties=score.ties if score else 0))
        if on_round is not None:
            on_round(records[-1])          # partial results survive a later abort
        history.append({"round": rnd, "keys": keys, "prediction": pred, "kept": kept, "rejected": rejected,
                        "score_points": None if score is None else round(score.score_points, 3)})
        if kept and candidate is not None:
            states[rnd] = (current, candidate)
            current = candidate
    dev_b_scores: Dict[int, float] = {}
    dev_b_detail: Dict[int, Dict[str, Any]] = {}
    for rnd, (before, after) in states.items():
        sc = paired_score(cache.coverage(before, dev_b), cache.coverage(after, dev_b))
        dev_b_scores[rnd] = sc.score_points
        dev_b_detail[rnd] = {"score_points": sc.score_points, "per_arm_points": dict(sc.per_arm_points), "wins": sc.wins,
                             "losses": sc.losses, "ties": sc.ties, "win_rate": sc.win_rate, "n_pairs": sc.n_pairs}
    candidates = tuple(r for r, s in sorted(dev_b_scores.items()) if s >= -EPS)
    return SandboxResult(base_harness=base, final_harness=current, rounds=tuple(records), promotion_candidates=candidates,
                         dev_b_scores=dev_b_scores, evaluations=cache.misses, dev_b_detail=dev_b_detail)


def _pointer_get(doc: Any, path: str) -> Any:
    node = doc
    for seg in _segments(path):
        node = node[int(seg)] if isinstance(node, list) else node[seg]
    return node


def _diff(before: Mapping[str, Any], after: Mapping[str, Any], proposal: Proposal) -> Tuple[Dict[str, Any], ...]:
    return tuple({"op": op["op"], "path": op["path"], "before": copy.deepcopy(_pointer_get(before, op["path"])),
                  "after": copy.deepcopy(_pointer_get(after, op["path"]))} for op in proposal.patch)


def _jsonable(raw: Any) -> Any:
    try:
        return json.loads(json.dumps(raw, default=str))
    except (TypeError, ValueError):
        return str(raw)


def _salvage_keys(raw: Any) -> List[str]:
    """Best-effort key list for the audit entry of a rejected edit."""
    out: List[str] = []
    if isinstance(raw, Mapping) and isinstance(raw.get("patch"), list):
        for op in raw["patch"][:4]:
            if isinstance(op, Mapping) and isinstance(op.get("path"), str):
                out.append(op["path"][:80])
    return out


def _salvage_prediction(raw: Any) -> Dict[str, Any]:
    p = raw.get("prediction") if isinstance(raw, Mapping) else None
    if isinstance(p, Mapping) and set(p) == {"metric", "delta_ppm"} and isinstance(p["metric"], str) \
            and isinstance(p["delta_ppm"], int) and not isinstance(p["delta_ppm"], bool):
        return {"metric": p["metric"], "delta_ppm": p["delta_ppm"]}
    return {"metric": "invalid", "delta_ppm": 0}


# --------------------------------------------------------------------------------------------------
# LLM proposer (the queen proposing). Only built when the lead wires a real provider; tests use FakeProvider.
# --------------------------------------------------------------------------------------------------

PROPOSER_SYSTEM = (
    "You tune a swarm harness. You see aggregate dev results only. Propose ONE edit as JSON and nothing else: "
    '{"patch": [{"op": "replace", "path": "/router/cost_penalty", "value": 0.2}], '
    '"prediction": {"metric": "coverage", "delta_ppm": 15000}}. Only paths in allowed_keys may change; '
    "delta_ppm is your expected change in mean coverage (10000 = 1 point), signed and non-zero."
)


def llm_proposer(provider: Provider, model: str, *, thinking_params: Optional[Mapping[str, Any]] = None,
                 run: Callable[[Awaitable[Any]], Any] = asyncio.run, outage: Optional[OutagePolicy] = None) -> Proposer:
    """A proposer backed by a provider. One call per round, 800-token output cap. ``run`` drives the coroutine (the
    sandbox script passes one that uses the harness's own event loop, so shared async locks stay on one loop).

    A provider outage (HTTP 401/429/5xx, transport error) is waited out with backoff and the same request is sent
    again (``outage`` sets the maximum wait, default 600 s); past it :class:`QueenUnavailable` propagates and ends the
    sandbox as an error. The proposer has no fallback."""
    policy = outage if outage is not None else OutagePolicy()
    def propose(ctx: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        request = ChatRequest(model=model, messages=({"role": "system", "content": PROPOSER_SYSTEM},
                                                     {"role": "user", "content": canonical_json(dict(ctx))}),
                              max_tokens=800, temperature=0.0, seed=ctx["round"], response_format={"type": "json_object"},
                              extra_body=thinking_params)
        response = run(_complete(provider, request, policy))
        text = response.content.strip()
        fenced = _FENCE.match(text)
        try:
            return json.loads(fenced.group(1) if fenced else text)
        except json.JSONDecodeError:
            return {"unparseable": text[:80]}   # rejected by parse_proposal; still logged as an edit entry
    return propose


async def _complete(provider: Provider, request: ChatRequest, policy: Optional[OutagePolicy] = None) -> Any:
    return await complete_with_outage_retry(provider, request, policy)
