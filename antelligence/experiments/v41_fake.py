"""Offline fakes for ``scripts/v41_run.py --backends fake`` (dry plumbing runs on real dev cases, no network, no cost).

Every fake answers the real typed questions with probabilities that vary with the request (a seeded hash of the request
hash and question id), so routing, deferral, the Qwen-4B escalation, cites and every board path are exercised. Replies
carry the identity ``fake`` so a fake run can never be mistaken for a real one in a cell. ``NullRouter`` is the
Phase 0 bootstrap router (``--router none``): route everything to Qwen3.5-9B, never defer (goal section 5 step 7).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Optional, Sequence

from antelligence.experiments.v41_conformal import ConformalThresholds
from antelligence.experiments.v41_harness import BONSAI, D1, FW, Q4, Q9, Worker
from antelligence.experiments.v41_router import BONSAI_ROUTER_MODELS, ROUTER_MODELS
from antelligence.kernel.canonical import content_hash
from antelligence.kernel.decision import Answer, DecisionRequest, DecisionResponse
from antelligence.providers.decision_backends import FakeBackend
from antelligence.providers.wrappers import FakeProvider

FAKE_IDENTITY = {"served_model": "fake", "revision": "fake", "runtime": "fake", "runtime_version": "fake"}
FAKE_PROB_METHOD = "fake"


def _h(*parts: Any) -> int:
    return int(content_hash([str(p) for p in parts])[:12], 16)


def varied_answers(request: DecisionRequest, salt: str = "") -> Dict[str, Answer]:
    """Deterministic answers whose confidence spans 0.35..0.95 (so conformal sets and the 4B escalation both fire)."""
    out: Dict[str, Answer] = {}
    for qid, q in request.questions.items():
        h = _h(request.request_hash, qid, salt)
        if q.type == "noul":
            p = round((h % 1001) / 1000, 3)
            out[qid] = Answer("noul", p, {"yes": p, "no": 1.0 - p})
            continue
        n = len(q.options)
        keys = q.options if q.type == "choice" else tuple(str(i) for i in range(n))
        top = h % n
        peak = 0.35 + 0.6 * (((h >> 8) % 1000) / 1000)
        peak = max(peak, 1.0 / n + 0.05)
        rest = (1.0 - peak) / (n - 1)
        probs = {k: (peak if i == top else rest) for i, k in enumerate(keys)}
        value: Any = keys[top] if q.type == "choice" else float(sum(i * probs[k] for i, k in enumerate(keys)))
        out[qid] = Answer(q.type, value, probs, round(peak, 3))
    return out


class FakeWorkerBackend(FakeBackend):
    """A FakeBackend whose replies carry identity ``fake``, a prob_method and hash-picked cites of the brief ids."""

    def __init__(self, name: str, salt: str = "") -> None:
        super().__init__(lambda r: varied_answers(r, salt or name), model=f"fake-{name}", name=name)

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        resp = await super().decide(request)
        cites = tuple(b for b in request.brief_ids if _h(request.request_hash, b) % 2 == 0)
        return dataclasses.replace(resp, identity=dict(FAKE_IDENTITY), prob_method=FAKE_PROB_METHOD, cites=cites,
                                   timing={"server_ms": 1, "network_ms": 0, "queue_ms": 0})


def build_fake_workers() -> Dict[str, Worker]:
    return {name: Worker(FakeWorkerBackend(name)) for name in (D1, Q4, Q9, BONSAI, FW)}


def fake_queen_reply(request: Any) -> str:
    """A valid queen reply built from the regions and squads in the prompt (reads what the real queen sees)."""
    body = json.loads(request.messages[1]["content"])
    ids = [r["region_id"] for r in body["regions"]]
    routes = [{"squad": s["squad"], "region_ids": ids[s["squad"] % max(1, len(ids)):][:2] or ids[:1], "priority": 2}
              for s in body["squads"]]
    return json.dumps({"plan": "split the squads", "guesses": ["core is the largest region"], "routes": routes,
                       "candidates": ids[:6]})


def build_fake_queen() -> FakeProvider:
    return FakeProvider(fake_queen_reply, model="fake-queen")


class NullRouter:
    """Phase 0 bootstrap router: every decision goes to ``target`` (Qwen3.5-9B drives). No fitted state."""

    kind = "none"
    sha256 = None

    def __init__(self, models: Sequence[str] = ROUTER_MODELS, target: str = Q9) -> None:
        self.models, self.target = tuple(models), target

    def predict(self, features: Sequence[float]) -> str:
        return self.target


def null_thresholds(arm: str = "SW-B", target: str = Q9) -> ConformalThresholds:
    """Thresholds that never defer (qhat 1.0): the bootstrap used with :class:`NullRouter`."""
    per = {q: {"alpha": 1.0, "qhat": 1.0, "n": 0, "dev_defer_rate": 0.0, "on_target": True}
           for q in ("target", "release", "amount")}
    return ConformalThresholds(arm, target, per, 1.0, 0.0, True)


def null_bootstrap(bonsai: bool = False):
    models = BONSAI_ROUTER_MODELS if bonsai else ROUTER_MODELS
    target = BONSAI if bonsai else Q9
    return NullRouter(models, target), null_thresholds("SW-Bonsai" if bonsai else "SW-B", target)


def fake_bootstrap(bonsai: bool = False):
    """A seeded random kNN router with mid thresholds (qhat 0.45): exercises routing, deferral and escalation on fakes."""
    import numpy as np
    from antelligence.experiments.v41_router import FEATURE_NAMES, KNNRouter, TrainRow
    models = BONSAI_ROUTER_MODELS if bonsai else ROUTER_MODELS
    target = BONSAI if bonsai else Q9
    rng = np.random.default_rng(3)
    rows = [TrainRow(list(rng.random(len(FEATURE_NAMES))), models[i % len(models)]) for i in range(30)]
    per = {q: {"alpha": 0.2, "qhat": 0.45, "n": 0, "dev_defer_rate": 0.2, "on_target": True}
           for q in ("target", "release", "amount")}
    router = KNNRouter.fit(rows, models, 5)
    router.kind = "fake"
    return router, ConformalThresholds("SW-Bonsai" if bonsai else "SW-B", target, per, 0.2, 0.2, True)


class MockPostMode:
    """Mode D without a chain: every batch is 'included' at once. It still runs :func:`compact_entry` on each entry, so a
    payload the real BoardLog encoder would reject fails here too (the same encoding the live mode posts)."""

    def __init__(self, latency_ms: int = 40) -> None:
        self.latency_ms, self._upto, self.visible, self.batches = latency_ms, 0, set(), 0

    def post_round(self, run_id: str, entries: Sequence[Any]):
        from antelligence.anchoring.chain_modes import RoundResult, compact_entry
        for e in entries:
            compact_entry(e)
        if entries:
            first = entries[0].seq
            if first != self._upto or [e.seq for e in entries] != list(range(first, first + len(entries))):
                raise ValueError(f"batch must continue at seq {self._upto}, got {[e.seq for e in entries]}")
            self._upto = first + len(entries)
            self.batches += 1
        self.visible |= {e.seq for e in entries}
        return RoundResult(set(self.visible), set(), False, {e.seq: self.latency_ms + e.seq % 7 for e in entries})


class MockReputationClient:
    """ReputationRegistry without a chain: idempotent per (run, author, claim), like the contract's queue."""

    def __init__(self) -> None:
        self.reports: Dict[tuple, bool] = {}

    def enqueue_report(self, run_id: str, author: str, claim_id: str, contradicted: bool) -> int:
        self.reports.setdefault((run_id, author, claim_id), bool(contradicted))
        return len(self.reports)


def verify_mock_exports(lab: Any, out_dir: Any, results: Sequence[Any]) -> Dict[str, Any]:
    """Run the outside verifier (all of B, OTS, R, C; final anchor required) over the mock-anchored runs of one process.

    The mock OTS calendars, Rekor log and registry only live as long as ``lab``, so this must run in the same process
    as the runs. Exports are gathered flat in ``<out_dir>/export`` (what ``v41_verify_board.py --export-dir`` expects);
    the report is written to ``<out_dir>/verify-mock-report.json``.
    """
    import hashlib
    import shutil
    from pathlib import Path

    from antelligence.anchoring.verify import OutsideVerifier
    from antelligence.kernel.canonical import jcs_bytes

    out = Path(out_dir)
    export = out / "export"
    export.mkdir(parents=True, exist_ok=True)
    runs = []
    for res in results:
        if isinstance(res, BaseException) or res.run_dir is None or res.board is None or not res.anchors:
            continue
        for f in Path(res.run_dir).iterdir():
            if f.name.startswith(res.cell["run_id"]) and f.suffix in (".jsonl", ".pem", ".ots", ".json"):
                shutil.copy2(f, export / f.name)
        h = res.board.header
        runs.append({"run_id": res.cell["run_id"], "pubkeys": h["pubkeys"], "anchor_pubkey": h["anchor_pubkey"]})
    if not runs:
        return {"verdict": "no anchored runs", "ok": False}
    plan = {"study": "v4.1", "runs": runs}
    (out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    study = lab.single_shot(out / "study-anchors")
    study.anchor("prereg", hashlib.sha256(lab.prereg_doc()).digest())
    study.anchor("runplan", hashlib.sha256(jcs_bytes(plan)).digest())
    lab.confirm_ots(export)
    lab.confirm_ots(out / "study-anchors")
    v = OutsideVerifier(export, plan, chain=lab.reader(), rekor=lab.rekor, ots_dir=export,
                        header_provider=lab.header_provider(), conditions={"B", "OTS", "R", "C"}, pins=lab.pins(),
                        prereg=lab.prereg_path(), study_pem=lab.study_key.pem, require_final_anchor=True)
    report = v.verify()
    report["study"] = v.verify_study(lab.prereg_path(), hashlib.sha256(jcs_bytes(plan)).digest(),
                                     prereg_ots=out / "study-anchors" / "prereg.ots", study_pem=lab.study_key.pem)
    if report["study"]["failed"]:
        report["verdict"] = "failed"
    elif report["study"]["unverified"] and report["verdict"] == "verified":
        report["verdict"] = "unverified"
    report["ok"] = report["verdict"] == "verified"
    (out / "verify-mock-report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    return report
