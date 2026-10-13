#!/usr/bin/env python
"""Queen sandbox loop (goal 4.6, Phase 0 step 8): the queen proposes up to 4 edits to config/v4_1/harness.yaml, each run on
dev-A under SW-B and SW-noBoard against the current harness, kept or reverted by the keep rule, then dev-B promotion checks.

  .venv/bin/python scripts/v41_sandbox.py --router R.json --thresholds T.json --backends live --anchors mock \\
      --out docs/research/v4.1-20261013/phase0/sandbox.json --rounds 4 --concurrency 4

Development cases only: dev-A is the first 30 development cases in split order (seed 1), dev-B the next 10. Any other case
is refused before an episode starts. Episodes run through the real harness (``v41_harness``) with the candidate harness.yaml
as the promoted harness. ``evaluate(harness_cfg, arm, case_ids)`` first looks in a results cache keyed by (harness sha, arm,
case, seed); only misses run, so each round pays for the candidate alone (the episode reply cache under it makes repeated
requests free). The queen sees aggregates only (per-arm mean coverage, router and deferral stats, her previous edits).

Every edit is an ``edit`` board entry on a sandbox board, anchored with the run's anchorer (``--anchors mock`` per
decisions/2026-10-13-calibration-runs-mock-anchors). Nothing is promoted: the output lists candidate edits for G1.
Every queen call goes through the usage ledger and the Nous $7.50 stop; the projected cost is printed first.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v41_run  # noqa: E402  (scripts/v41_run.py: backend, router and mock-anchor wiring)
from antelligence.experiments import v41_budget, v41_prod, v4_budget  # noqa: E402
from antelligence.experiments.v4_wiring import NOUS_EXTRA  # noqa: E402
from antelligence.experiments.v41_harness import QUEEN_MODEL, HarnessError, V41Harness, _HookedBoard  # noqa: E402
from antelligence.experiments.v41_sandbox import (ARMS, DEV_A_SIZE, DEV_B_SIZE, MAX_ROUNDS, EvalResult, llm_proposer,  # noqa: E402
                                                  run_sandbox)
from antelligence.experiments.v41_split import load_split  # noqa: E402
from antelligence.kernel.canonical import canonical_json  # noqa: E402
from antelligence.kernel.decision import DecisionAbort  # noqa: E402
from antelligence.kernel.policies.v41_harness import harness_sha256, load_allowed_keys, load_harness  # noqa: E402
from antelligence.providers.ledger import PRICES  # noqa: E402
from antelligence.providers.redact import scrub  # noqa: E402

SEED = 1
QUEEN_CALLS_PER_EPISODE = 6          # goal 4.6 cap per run, re-plans included
STAT_KEYS = ("defer_rate", "router_top_share", "off_target", "decision_error_share", "queen_calls", "alarms")
DEFAULT_OUT = "docs/research/v4.1-20261013/phase0/sandbox.json"


class DevOnlyError(HarnessError):
    """A sandbox episode was requested on a case that is not a development case (or not in dev-A / dev-B)."""


# --------------------------------------------------------------------------------------------------
# Dev-only split
# --------------------------------------------------------------------------------------------------

def dev_sets(split: Mapping[str, Any], n_a: int = DEV_A_SIZE, n_b: int = DEV_B_SIZE,
             override_a: Optional[Sequence[str]] = None, override_b: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
    """dev-A = the first n_a dev-A cases in split order, dev-B = the first n_b dev-B cases. Overrides must be development
    cases of the matching set, otherwise :class:`DevOnlyError`."""
    if not (1 <= n_a <= DEV_A_SIZE and 1 <= n_b <= DEV_B_SIZE):
        raise DevOnlyError(f"dev-A is at most {DEV_A_SIZE} cases and dev-B at most {DEV_B_SIZE}")
    a_all, b_all = list(split["dev_a"])[:DEV_A_SIZE], list(split["dev_b"])[:DEV_B_SIZE]
    a = list(override_a) if override_a else a_all[:n_a]
    b = list(override_b) if override_b else b_all[:n_b]
    bad = sorted((set(a) - set(a_all)) | (set(b) - set(b_all)))
    if bad:
        raise DevOnlyError(f"refused: {bad[:3]} are not development cases of the right set (development cases only)")
    if set(a) & set(b):
        raise DevOnlyError("dev-A and dev-B overlap")
    return a, b


# --------------------------------------------------------------------------------------------------
# Results cache: (harness sha, arm, case, seed) -> coverage and aggregate-able stats
# --------------------------------------------------------------------------------------------------

class ResultsCache:
    """Append-only JSONL. Rows carry a context fingerprint (router, thresholds, backends, world, look temperature); rows
    made under another context are ignored, never reused."""

    def __init__(self, path: Optional[Path], context: str) -> None:
        self.path, self.context = path, context
        self._rows: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
        self.hits = 0
        self.runs = 0
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("ctx") == context:
                        self._rows[(row["harness"], row["arm"], row["case"], row["seed"])] = row

    def get(self, sha: str, arm: str, case: str, seed: int) -> Optional[Dict[str, Any]]:
        return self._rows.get((sha, arm, case, seed))

    def put(self, sha: str, arm: str, case: str, seed: int, coverage: float, stats: Mapping[str, float]) -> None:
        row = {"harness": sha, "arm": arm, "case": case, "seed": seed, "coverage": coverage, "stats": dict(stats),
               "ctx": self.context}
        self._rows[(sha, arm, case, seed)] = row
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def _cell_stats(cell: Mapping[str, Any]) -> Dict[str, float]:
    return {k: float(cell[k]) for k in STAT_KEYS if cell.get(k) is not None}


def _mean_stats(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in STAT_KEYS:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


class EventLoop:
    """One asyncio loop on a thread for the whole sandbox, so the harness's cached providers keep one loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def make_evaluate(h: V41Harness, cache: ResultsCache, loop: EventLoop, allowed_cases: Sequence[str], *, seed: int = SEED,
                  concurrency: int = 4) -> Callable[[Mapping[str, Any], str, Sequence[str]], EvalResult]:
    """The ``evaluate(harness_cfg, arm, case_ids)`` callback: SW-B or SW-noBoard episodes on dev cases under a candidate
    harness.yaml (as the promoted harness), seed 1. Cache hits make no episode. Any failed episode aborts the sandbox:
    a missing case would bias the paired score."""
    allowed = set(allowed_cases)

    def evaluate(harness_cfg: Mapping[str, Any], arm: str, case_ids: Sequence[str]) -> EvalResult:
        if arm not in ARMS:
            raise HarnessError(f"sandbox arms are {ARMS}, got {arm!r}")
        refused = [c for c in case_ids if c not in allowed]
        if refused:
            raise DevOnlyError(f"refused: {refused[:3]} are not sandbox development cases (development cases only)")
        sha = harness_sha256(harness_cfg)
        todo = [c for c in case_ids if cache.get(sha, arm, c, seed) is None]
        cache.hits += len(case_ids) - len(todo)
        if todo:
            h.promoted_harness = dict(harness_cfg)       # SW-B / SW-noBoard run the promoted harness (not calibration)
            results = loop.run(h.run_many([(arm, c, seed) for c in todo], concurrency=concurrency))
            failed = []
            for case, res in zip(todo, results):
                if isinstance(res, DecisionAbort):
                    raise res
                if isinstance(res, BaseException):
                    failed.append(f"{case}: {type(res).__name__}: {res}")
                    continue
                cache.put(sha, arm, case, seed, float(res.cell["coverage"]), _cell_stats(res.cell))
                cache.runs += 1
            if failed:
                raise HarnessError(scrub(f"{len(failed)} sandbox episode(s) failed under {arm}: " + "; ".join(failed[:3])))
        rows = [cache.get(sha, arm, c, seed) for c in case_ids]
        return EvalResult(coverage={c: r["coverage"] for c, r in zip(case_ids, rows)},
                          stats=_mean_stats([r["stats"] for r in rows]))
    return evaluate


# --------------------------------------------------------------------------------------------------
# Cost projection
# --------------------------------------------------------------------------------------------------

def max_episodes(rounds: int, n_a: int, n_b: int) -> int:
    """Worst case, nothing cached: base + one candidate per round, on dev-A and (every round kept) dev-B, both arms."""
    return 2 * (n_a + n_b) * (rounds + 1)


def measured_cost_per_queen_call(usage_ledger: Optional[Path]) -> Tuple[float, str]:
    """Mean ``cost_usd`` of Nous queen rows in the usage ledger; the goal's token caps priced at the table otherwise."""
    if usage_ledger is not None and Path(usage_ledger).exists():
        rows = [r for r in v4_budget.read_jsonl(usage_ledger)
                if r.get("role") == "queen" and v4_budget.provider_of(r) == "nous" and r.get("cost_usd")]
        if rows:
            return sum(float(r["cost_usd"]) for r in rows) / len(rows), f"measured over {len(rows)} ledgered queen calls"
    p_in, p_out = PRICES[QUEEN_MODEL]
    return (6000 * p_in + 800 * p_out) / 1_000_000, "upper bound: 6k in / 800 out at table prices (no queen rows in the ledger)"


def project_cost(rounds: int, n_a: int, n_b: int, *, usage_ledger: Optional[Path], pod_costs: Optional[Path],
                 paid: bool) -> Dict[str, Any]:
    episodes = max_episodes(rounds, n_a, n_b)
    calls = episodes * QUEEN_CALLS_PER_EPISODE + rounds           # queen steps inside episodes + one proposal per round
    per_call, basis = measured_cost_per_queen_call(usage_ledger) if paid else (0.0, "fake backends: nothing is paid")
    spent = v41_budget.spend_now(usage_ledger, pod_costs)["nous"] if (paid and usage_ledger and Path(usage_ledger).exists()) else 0.0
    projected = calls * per_call
    return {"rounds": rounds, "max_episodes": episodes, "queen_calls": calls, "cost_per_call_usd": per_call,
            "basis": basis, "projected_nous_usd": projected, "nous_spent_usd": spent,
            "nous_stop_usd": v41_budget.STOPS["nous"], "headroom_usd": v41_budget.STOPS["nous"] - spent,
            "fits": (not paid) or projected <= v41_budget.STOPS["nous"] - spent,
            "note": "worst case, nothing cached; RunPod pod time for the worker calls is billed by the pod script, not per call"}


def describe_projection(p: Mapping[str, Any]) -> str:
    return (f"projected sandbox cost: {p['queen_calls']} queen calls ({p['max_episodes']} episodes x {QUEEN_CALLS_PER_EPISODE} "
            f"+ {p['rounds']} proposals) x ${p['cost_per_call_usd']:.6f} per call ({p['basis']}) = "
            f"${p['projected_nous_usd']:.4f} Nous; spent ${p['nous_spent_usd']:.4f} of the ${p['nous_stop_usd']:.2f} stop "
            f"(headroom ${p['headroom_usd']:.4f}). Worst case: cached episodes cost nothing.")


# --------------------------------------------------------------------------------------------------
# Proposers
# --------------------------------------------------------------------------------------------------

_FAKE_EDITS = (("/router/cost_penalty", 0.25), ("/briefs/count", 10), ("/queen/cadence", 36), ("/router/slm_conf_min", 0.6))


def fake_proposer() -> Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    """Scripted proposer for ``--backends fake``: no model, a different allowed edit per round, a positive prediction."""
    def propose(ctx: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        path, value = _FAKE_EDITS[(ctx["round"] - 1) % len(_FAKE_EDITS)]
        return {"patch": [{"op": "replace", "path": path, "value": value}], "prediction": {"metric": "coverage", "delta_ppm": 15000}}
    return propose


# --------------------------------------------------------------------------------------------------
# Sandbox board (edit entries, anchored like any other board)
# --------------------------------------------------------------------------------------------------

def build_board(h: V41Harness, base_sha: str, world_sha: str, out_dir: Path) -> Tuple[Any, Any]:
    """A signed board for the ``edit`` entries, with the harness's anchorer (mock in the sandbox) when it has one."""
    keys, akey = h.keys.get("sandbox-v4.1")
    header = {"study": "v4.1", "arm": "sandbox", "case": "dev-A", "seed": SEED, "split": "dev", "world_cfg_sha256": world_sha,
              "harness_cfg_sha256": base_sha, "pubkeys": dict(keys.pubkeys), "anchor_pubkey": akey.fingerprint,
              "code_sha": h.code_sha}
    holder: List[Any] = []

    def on_append(board: Any, entry: Any) -> None:
        if holder:
            holder[0].maybe_anchor(board)
    board = _HookedBoard(header, mode="B", keys=keys.seeds, on_append=on_append)
    anchorer = None
    if h.anchorer_factory is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{board.run_id}.anchor_pub.pem").write_bytes(akey.pem)
        anchorer = h.anchorer_factory(board, akey, out_dir)
        holder.append(anchorer)
    return board, anchorer


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="1-4 (default 4; the Nous cut order allows 3)")
    ap.add_argument("--router", required=True, help="frozen kNN router JSON (Phase 0 step 7)")
    ap.add_argument("--thresholds", required=True, help="frozen conformal thresholds JSON (Phase 0 step 7)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / DEFAULT_OUT), help="sandbox.json to write")
    ap.add_argument("--backends", choices=("live", "fake"), default="live")
    ap.add_argument("--anchors", choices=("mock",), default="mock", help="the sandbox always uses mock anchors")
    ap.add_argument("--harness", default=str(REPO / "config/v4_1/harness.yaml"), help="the starting harness.yaml")
    ap.add_argument("--work-dir", help="episode exports, board export and results cache (default <repo>/data/v4_1/sandbox)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-a", type=int, default=DEV_A_SIZE, help=f"dev-A cases, at most {DEV_A_SIZE} (default all)")
    ap.add_argument("--n-b", type=int, default=DEV_B_SIZE, help=f"dev-B cases, at most {DEV_B_SIZE} (default all)")
    ap.add_argument("--dev-a", help="comma-separated override; every case must be a dev-A case")
    ap.add_argument("--dev-b", help="comma-separated override; every case must be a dev-B case")
    ap.add_argument("--loader", default=None)
    ap.add_argument("--cases-root")
    ap.add_argument("--world-yaml", default=str(REPO / "config/v4_1/world.yaml"))
    ap.add_argument("--pod-state", default=str(v41_prod.POD_STATE))
    ap.add_argument("--bonsai-url", default=v41_prod.DEFAULT_BONSAI_URL)
    ap.add_argument("--usage-ledger", default=str(v41_prod.USAGE_LEDGER))
    ap.add_argument("--pod-costs", default=str(v41_prod.POD_COSTS))
    ap.add_argument("--reply-cache", default=str(v41_prod.CACHE_DIR))
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--dry-projection", action="store_true", help="print the projected cost and exit")
    return ap


def _file_sha(path: Optional[str]) -> str:
    if not path or path in ("none", "fake"):
        return str(path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    sha = out.stdout.strip()
    return sha if len(sha) == 40 else "0" * 40


def worker_identities(h: V41Harness) -> Dict[str, Any]:
    """Each worker's model id, revision and runtime identity as the backend describes itself."""
    out: Dict[str, Any] = {}
    for name, w in sorted(h.workers.items()):
        describe = getattr(w.backend, "describe", None)
        out[name] = {"model": w.model, "describe": json.loads(canonical_json(describe())) if callable(describe) else None}
    return out


def results_context(h: V41Harness, args: argparse.Namespace, world_sha: str) -> str:
    """Fingerprint a results-cache row must match: code, frozen files, backends, worker identities, world, look T."""
    return hashlib.sha256(canonical_json({
        "code_sha": _git_head(), "router": _file_sha(args.router), "thresholds": _file_sha(args.thresholds),
        "backends": h.backends_kind, "workers": worker_identities(h), "world": world_sha,
        "look_temperature": h.look_temperature}).encode()).hexdigest()


def _run_args(args: argparse.Namespace, work: Path, dev_a: Sequence[str]) -> argparse.Namespace:
    argv = ["--backends", args.backends, "--anchors", "mock", "--router", args.router, "--thresholds", args.thresholds,
            "--out-dir", str(work), "--concurrency", str(args.concurrency), "--world-yaml", args.world_yaml,
            "--pod-state", args.pod_state, "--bonsai-url", args.bonsai_url, "--usage-ledger", args.usage_ledger,
            "--pod-costs", args.pod_costs, "--reply-cache", args.reply_cache, "--repo-root", args.repo_root]
    if args.loader:
        argv += ["--loader", args.loader]
    if args.cases_root:
        argv += ["--cases-root", args.cases_root]
    for arm in ARMS:
        argv += ["--run", f"{arm}:{dev_a[0]}:{args.seed}"]
    ns = v41_run.parser().parse_args(argv)
    ns.live_anchoring = None
    ns.mock_lab = None
    return ns


def _round_json(r: Any, dev_b: Optional[Mapping[str, Any]], candidate: bool) -> Dict[str, Any]:
    return {
        "round": r.round, "proposal": r.proposal, "prediction": dict(r.prediction), "diff": [dict(d) for d in r.diff],
        "rejected": r.rejected or None, "patch_sha256": r.patch_sha256, "candidate_harness_sha256": r.candidate_sha256,
        "dev_a": None if r.score_points is None else {
            "score_points": r.score_points, "win_rate": r.win_rate, "wins": r.wins, "losses": r.losses, "ties": r.ties,
            "per_arm_points": dict(r.per_arm_points), "n_pairs": r.wins + r.losses + r.ties},
        "kept": r.kept, "edit_entry_id": r.edit_entry_id,
        "dev_b": dict(dev_b) if dev_b else None, "promotion_candidate": candidate}


def main(argv: Optional[Sequence[str]] = None, *, get_env: Callable[[str], Optional[str]] = None,
         harness: Optional[V41Harness] = None, proposer: Optional[Callable[..., Any]] = None) -> int:
    import os
    get_env = get_env or os.environ.get
    args = parser().parse_args(argv)
    if not 1 <= args.rounds <= MAX_ROUNDS:
        raise SystemExit(f"--rounds must be 1-{MAX_ROUNDS}")
    if args.backends == "live" and args.router in ("none", "fake"):
        raise SystemExit("live sandbox runs need the frozen router and thresholds files (Phase 0 step 7), not 'none'/'fake'")
    split = load_split(Path(args.repo_root) / "config/v4_1/split.json")
    try:
        dev_a, dev_b = dev_sets(split, args.n_a, args.n_b, args.dev_a.split(",") if args.dev_a else None,
                                args.dev_b.split(",") if args.dev_b else None)
    except DevOnlyError as exc:
        raise SystemExit(str(exc))
    paid = args.backends == "live"
    usage = Path(args.usage_ledger) if paid else None
    projection = project_cost(args.rounds, len(dev_a), len(dev_b), usage_ledger=usage, pod_costs=Path(args.pod_costs), paid=paid)
    print(describe_projection(projection))
    if args.dry_projection:
        return 0
    if not projection["fits"]:
        print(f"refused: projected ${projection['projected_nous_usd']:.4f} exceeds the Nous headroom "
              f"${projection['headroom_usd']:.4f}; try --rounds 3", file=sys.stderr)
        return 2
    work = Path(args.work_dir) if args.work_dir else REPO / "data/v4_1/sandbox"
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    h = harness or v41_run.build_harness(_run_args(args, work, dev_a), get_env)
    base = load_harness(args.harness)
    allowed_keys = load_allowed_keys(args.harness)
    base_sha = harness_sha256(base)
    world_sha = hashlib.sha256(Path(args.world_yaml).read_bytes()).hexdigest() if Path(args.world_yaml).exists() else "0" * 64
    context = results_context(h, args, world_sha)
    cache = ResultsCache(work / "results-cache.jsonl", context)
    loop = EventLoop()
    board, anchorer = build_board(h, base_sha, world_sha, work / "sandbox-board")
    evaluate = make_evaluate(h, cache, loop, [*dev_a, *dev_b], seed=args.seed, concurrency=args.concurrency)
    if proposer is None:
        if paid:
            proposer = llm_proposer(h.queen_provider, QUEEN_MODEL, thinking_params=NOUS_EXTRA, run=loop.run)
        else:
            proposer = fake_proposer()
    status, error = "complete", None
    result = None
    partial: List[Any] = []                # rounds finished before an abort
    try:
        result = run_sandbox(board, base, allowed_keys, dev_a, dev_b, evaluate, proposer, max_rounds=args.rounds,
                             on_round=partial.append)
    except DecisionAbort as exc:           # the Nous stop: results so far are in the results cache, rerun to resume
        status, error = "aborted_budget", scrub(str(exc))
    except Exception as exc:               # any other failure: keep what finished, say why
        status, error = "aborted_error", scrub(f"{type(exc).__name__}: {exc}")
    finally:
        loop.close()
    anchors: List[Any] = []
    if anchorer is not None:
        anchorer.finalize(board)
        anchorer.refresh_chain()
        anchors = [dict(a) for a in anchorer.records]
    board.export_jsonl(work / "sandbox-board" / f"{board.run_id}.jsonl")
    doc: Dict[str, Any] = {
        "study": "v4.1", "step": "Phase 0 step 8 (queen sandbox loop, goal 4.6)", "status": status, "error": error,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "arms": list(ARMS), "backends": h.backends_kind, "anchors": "mock", "router_kind": h.router_kind,
        "router_sha256": _file_sha(args.router), "thresholds_sha256": _file_sha(args.thresholds),
        "dev_a": dev_a, "dev_b": dev_b, "queen_model": QUEEN_MODEL, "queen_extra_body": NOUS_EXTRA,
        "projection": projection, "base_harness_sha256": base_sha,
        "board": {"run_id": board.run_id, "entries": board.size(), "anchors": anchors},
        "episodes_run": cache.runs, "results_cache_hits": cache.hits,
        "promotion": "Nothing is promoted automatically. promotion_candidates are the candidate edits for G1; the lead approves.",
    }
    if result is not None:
        cands = set(result.promotion_candidates)
        doc.update({
            "final_harness_sha256": harness_sha256(result.final_harness), "evaluations": result.evaluations,
            "rounds": [_round_json(r, result.dev_b_detail.get(r.round), r.round in cands) for r in result.rounds],
            "promotion_candidates": [
                {"round": r.round, "proposal": r.proposal, "diff": [dict(d) for d in r.diff],
                 "dev_a_score_points": r.score_points, "dev_b_score_points": result.dev_b_scores[r.round]}
                for r in result.rounds if r.round in cands]})
    else:
        doc["rounds"] = [_round_json(r, None, False) for r in partial]
        doc["promotion_candidates"] = []
        doc["note"] = "aborted: completed rounds are listed without a dev-B check; nothing is a candidate"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sandbox {status}: {len(doc.get('rounds', []))} round(s), {cache.runs} episodes run, {cache.hits} cache hits, "
          f"{len(doc.get('promotion_candidates', []))} candidate edit(s) -> {out}")
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    from scripts.v41_pod import run_main_with_pod_down  # ends the run by bringing the pod down; --keep-pod skips it
    raise SystemExit(run_main_with_pod_down(main))
