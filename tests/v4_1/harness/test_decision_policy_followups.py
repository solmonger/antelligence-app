"""Follow-ups to the decision failure policy: the d1 local look, budget hard stops on every live backend and the queen,
billed failed attempts, and a live-then-offline replay with a rescued and an errored decision. No network, no keys."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from antelligence.experiments import v41_cells
from antelligence.experiments.v41_budget import BudgetRefused
from antelligence.experiments.v41_harness import (BONSAI, D1, DECISION_RETRIES, FW, Q9, KeyVault, Worker)
from antelligence.experiments.v41_prod import BudgetGuard, GuardedChat, GuardedDecision, _stack
from antelligence.kernel.decision import DecisionAbort, DecisionError
from antelligence.kernel.policies.v41_queen import V41Queen  # noqa: F401  (the queen path under test)
from antelligence.provenance.bundle import verify_bundle_integrity
from antelligence.providers.base import BudgetExceeded
from antelligence.providers.decision_backends import (CachedDecision, D1Backend, FakeBackend, LlamaCppBackend,
                                                      NousFWBackend, SLMBackend, VLLMLogprobBackend, hashed_answers)
from antelligence.providers.ledger import LedgerWriter
from antelligence.providers.wrappers import Cached, FakeProvider
from tests.v4_1.harness.conftest import (CASE_DEV, build_rig, fake_workers, queen_provider, small_cfg)
from tests.v4_1.harness.test_decision_errors import FailingBackend, rig_with, run
from tests.v4_1.workers.helpers import BONSAI as BONSAI_ID, QWEN9, Recorder, make_request

N_DEC = 8 * small_cfg().rounds


class Stop:
    """A budget guard that is already at its hard stop."""

    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1
        raise BudgetRefused("nous hard stop: $7.60 spent >= $7.50")


# --- 1. the d1 local look is under the same policy --------------------------------------------------------------------

class LookFailing(FakeBackend):
    def __init__(self, fail_looks, billed=None):
        super().__init__(None, model="d1-fake", name=D1)
        self.fail_looks, self.billed, self.look_calls = fail_looks, billed, 0

    async def decide(self, request):
        if "look" in request.questions:
            self.look_calls += 1
            if self.fail_looks(request):
                raise DecisionError("d1: look reply invalid", billed=self.billed)
        return await super().decide(request)


def test_failed_look_is_retried_then_none_with_error_row_and_alarm(tmp_path):
    victim = {}

    def fail(request):
        victim.setdefault("seed", request.seed)
        return request.seed == victim["seed"]

    be = LookFailing(fail)
    res = run(rig_with(tmp_path, D1, be), "SW-B")
    assert res.cell.get("error") is None and len(res.decision_log) == N_DEC           # the run continued
    assert res.cell["look_errors"] == 1 and res.cell["decision_errors"] == 0           # counted apart
    assert res.run_log["look_errors"] == 1
    rows = [json.loads(line) for line in (res.run_dir / "decisions.jsonl").read_text().splitlines()]
    looks = [r for r in rows if r.get("kind") == "look"]
    assert len(looks) == 1 and looks[0]["error"] and looks[0]["error_reason"].startswith("look_error: ")
    assert looks[0]["attempts"] == DECISION_RETRIES + 1 and len(rows) == N_DEC + 1
    assert all(r["kind"] == "decision" for r in rows if r.get("kind") != "look")
    notes = [e.payload["note"] for e in res.board.by_type("alarm") if e.payload["note"].startswith("look_error")]
    assert len(notes) == 1
    assert be.look_calls >= DECISION_RETRIES + 1
    assert v41_cells.validate_cell(res.cell) == []


def test_look_budget_abort_is_not_absorbed(tmp_path):
    be = LookFailing(lambda r: True)
    be.fail_looks = lambda r: (_ for _ in ()).throw(DecisionAbort("hard stop"))
    with pytest.raises(DecisionAbort):
        run(rig_with(tmp_path, D1, be), "SW-B")
    assert be.look_calls == 8                                                         # one per squad: no retry


# --- 2. a budget hard stop is a DecisionAbort on every live backend and the queen -------------------------------------

def _real_backends():
    ok = lambda r: httpx.Response(200, json={})    # noqa: E731  never reached: the guard refuses first
    out = {}
    for name, make in {
        "d1": lambda t: D1Backend("http://pod:8102", retries=3, transport=t),
        "vllm": lambda t: VLLMLogprobBackend("http://pod:8100", *QWEN9, retries=3, transport=t),
        "llamacpp": lambda t: LlamaCppBackend("http://mac:8080", *BONSAI_ID, retries=3, transport=t),
        "nous_fw": lambda t: NousFWBackend("https://nous.test/v1", api_key="nk", retries=3, transport=t),
    }.items():
        rec = Recorder(ok)
        out[name] = (make(rec.transport()), rec)
    return out


@pytest.mark.parametrize("name", ["d1", "vllm", "llamacpp", "nous_fw"])
def test_budget_hard_stop_aborts_every_live_backend(tmp_path, name):
    backend, rec = _real_backends()[name]
    stop = Stop()
    worker = _stack(backend, stop, LedgerWriter(tmp_path / "usage.jsonl"), None)
    req = make_request(backend.model)
    with pytest.raises(DecisionAbort):
        asyncio.run(worker.backend.decide(req))
    assert rec.requests == [] and stop.checks == 1                                    # no call, no retry
    assert not (tmp_path / "usage.jsonl").exists()                                    # a refused call writes no row


def test_real_nous_budget_guard_raises_decision_abort(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps({"provider": "nous", "cost_usd": 7.6}) + "\n")
    guard = BudgetGuard("nous", usage, tmp_path / "pods.jsonl")
    with pytest.raises(DecisionAbort):
        asyncio.run(GuardedDecision(FakeBackend(model="m"), guard).decide(make_request("m")))
    assert issubclass(BudgetRefused, DecisionAbort)


def test_slm_backend_run_budget_is_a_decision_abort():
    class Over:
        def describe(self):
            return {}

        async def complete(self, request):
            raise BudgetExceeded("call budget spent")

    with pytest.raises(DecisionAbort):
        asyncio.run(SLMBackend(Over(), "m", retries=3).decide(make_request("m")))


@pytest.mark.parametrize("name,arm,calibration", [(D1, "SW-B", False), (Q9, "SW-B", True), (FW, "FW", False)])
def test_harness_does_not_retry_or_absorb_a_budget_stop(tmp_path, name, arm, calibration):
    inner = FakeBackend(model=f"{name}-fake", name=name)
    stop = Stop()
    workers = fake_workers()
    workers[name] = Worker(GuardedDecision(inner, stop), inner.model)
    with pytest.raises(DecisionAbort):
        asyncio.run(build_rig(tmp_path, workers=workers).harness.run(arm, CASE_DEV, 1, calibration=calibration))
    assert not any("look" not in r.questions for r in inner.requests)                  # no decision call went out


def test_queen_budget_stop_aborts_the_run(tmp_path):
    stop = Stop()
    rig = build_rig(tmp_path, queen=GuardedChat(queen_provider(), stop))
    with pytest.raises(DecisionAbort):
        run(rig, "SW-B")
    assert stop.checks == 1
    # an ordinary queen failure is still only a fallback
    def boom(request):
        raise RuntimeError("provider down")
    res = run(build_rig(tmp_path / "ok", queen=FakeProvider(boom, model="fake-queen")), "SW-B")
    assert res.cell.get("error") is None


# --- 3. failed attempts that the provider bills are tallied -------------------------------------------------------------

def _nous_handler(bad_first: int, billed_cost=0.001):
    state = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        spec = json.loads(content.split("\n")[0])["questions"]
        state["n"] += 1
        bad = state["n"] <= bad_first
        amount = {str(i): (0.0 if bad else 0.2) for i in range(len(spec["amount"]["levels"]))}
        options = list(spec["target"]["options"])
        reply = json.dumps({"target": {o: 1.0 / len(options) for o in options}, "release": 0.2, "amount": amount})
        return httpx.Response(200, json={
            "model": body["model"], "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 60, "cost": billed_cost}})
    return handle, state


def test_billed_failed_attempts_reach_cost_tokens_and_ledger(tmp_path):
    handle, state = _nous_handler(bad_first=2)
    backend = NousFWBackend("https://nous.test/v1", api_key="nk", retries=0, transport=httpx.MockTransport(handle))
    ledger = LedgerWriter(tmp_path / "usage.jsonl")
    workers = fake_workers()
    workers[FW] = _stack(backend, None, ledger, None)
    res = run(build_rig(tmp_path / "rig", workers=workers), "FW")
    assert res.cell.get("error") is None and res.cell["decision_errors"] == 0          # both failures were rescued
    rows = ledger.rows()
    failed = [r for r in rows if r.get("error")]
    assert len(failed) == 2 and all(r["cost_usd"] == pytest.approx(0.001) and r["input_tokens"] == 300 for r in failed)
    assert state["n"] == N_DEC + 2 and len(rows) == N_DEC + 2
    assert res.cell["cost_usd"] == pytest.approx(ledger.total_cost()) == pytest.approx((N_DEC + 2) * 0.001)
    base_handle, _ = _nous_handler(bad_first=0)
    base_workers = fake_workers()
    base_workers[FW] = _stack(NousFWBackend("https://nous.test/v1", api_key="nk", retries=0,
                                            transport=httpx.MockTransport(base_handle)), None,
                              LedgerWriter(tmp_path / "base.jsonl"), None)
    base = run(build_rig(tmp_path / "base", workers=base_workers), "FW")
    assert res.cell["tokens_in"] - base.cell["tokens_in"] == 2 * 300                    # the two rejected replies
    assert res.cell["tokens_out"] - base.cell["tokens_out"] == 2 * 60


def test_billed_failures_of_an_errored_decision_are_counted_too(tmp_path):
    handle, state = _nous_handler(bad_first=DECISION_RETRIES + 1)                     # the first decision fails all 4
    backend = NousFWBackend("https://nous.test/v1", api_key="nk", retries=0, transport=httpx.MockTransport(handle))
    ledger = LedgerWriter(tmp_path / "usage.jsonl")
    workers = fake_workers()
    workers[FW] = _stack(backend, None, ledger, None)
    res = run(build_rig(tmp_path / "rig", workers=workers), "FW")
    assert res.cell["decision_errors"] >= 1
    assert res.cell["cost_usd"] == pytest.approx(ledger.total_cost()) == pytest.approx(state["n"] * 0.001)


# --- 4. live then offline: a rescued and an errored decision replay to the same metrics --------------------------------

class Scripted(FakeBackend):
    """First decision (by seed): two billed failures, then fine (rescued). Second decision: always fails (errored)."""

    def __init__(self):
        super().__init__(None, model="fw-fake", name=FW)
        self.order, self.fails = [], {}

    def describe(self):
        return {**super().describe(), "prob_method": "verbalised"}

    async def decide(self, request):
        if "look" in request.questions:
            return await super().decide(request)
        if request.seed not in self.order and len(self.order) < 2:
            self.order.append(request.seed)
        if request.seed in self.order:
            k = self.fails[request.seed] = self.fails.get(request.seed, 0) + 1
            rescued = self.order.index(request.seed) == 0
            if not rescued or k <= 2:
                raise DecisionError("slm: probabilities sum to 0.000 (all-zero distribution)",
                                    billed={"input_tokens": 300, "output_tokens": 60, "cost_usd": 0.001})
        return await super().decide(request)


def test_live_then_offline_replay_reproduces_metrics(tmp_path):
    cache, vault = tmp_path / "cache", KeyVault()

    def workers(live: bool):
        w = fake_workers()
        out = {}
        for name, worker in w.items():
            inner = worker.backend
            if name == FW:
                inner = Scripted() if live else FakeBackend(
                    lambda r: (_ for _ in ()).throw(AssertionError("backend called during replay")), model="fw-fake",
                    name=FW)
            elif not live:
                inner = FakeBackend(lambda r: (_ for _ in ()).throw(AssertionError("backend called during replay")),
                                    model=worker.model)
            out[name] = Worker(CachedDecision(inner, cache / name, offline=not live), worker.model)
        return out

    live = build_rig(tmp_path / "live", vault=vault, workers=workers(True),
                     queen=Cached(queen_provider(), cache / "queen.jsonl"))
    res = run(live, "FW")
    assert res.cell["decision_errors"] == 1 and len(res.decision_log) == N_DEC
    assert sorted(r["attempts"] for r in res.decision_log if r["error"] or r["attempts"] > 1) == [3, 4]
    assert res.cell["cost_usd"] == pytest.approx(6 * 0.001)                           # 2 + 4 billed failed attempts
    assert verify_bundle_integrity(res.bundle)
    offline = build_rig(tmp_path / "replay", vault=vault, workers=workers(False),
                        queen=Cached(queen_provider(lambda r: (_ for _ in ()).throw(AssertionError("queen called"))),
                                     cache / "queen.jsonl", offline=True))
    verdict = asyncio.run(offline.harness.replay(res.bundle))
    assert verdict == {"replay_ok": True, "mismatched_fields": [], "trace_hash": res.trace_hash}
