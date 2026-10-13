"""Decision-level failure policy (goal 8: retry up to 3 times, then mark the DECISION as an error, not the run), the
strict verbalised parser and the reply cache. Fake backends, no network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from antelligence.experiments import v41_cells
from antelligence.experiments.v41_harness import (CORRECTIVE_NOTE, DECISION_ERROR_ALARM, DECISION_RETRIES, D1, FW, Q9,
                                                  Worker)
from antelligence.kernel.decision import DecisionAbort, DecisionError
from antelligence.providers.decision_backends import (CachedDecision, FakeBackend, NousFWBackend, _slm_prompt,
                                                      hashed_answers, parse_slm_json)
from tests.v4_1.harness.conftest import CASE_DEV, build_rig, fake_workers, small_cfg
from tests.v4_1.workers.helpers import make_request

N_DEC = 8 * small_cfg().rounds


def run(rig, arm="FW", case=CASE_DEV, seed=1, **kw):
    return asyncio.run(rig.harness.run(arm, case, seed, **kw))


class FailingBackend(FakeBackend):
    """Raises DecisionError for every call of the first decision question set it sees (identified by seed)."""

    def __init__(self, name, model, *, verbalised=False, fail_attempts=None, exc=DecisionError):
        super().__init__(None, model=model, name=name)
        self.verbalised, self.fail_attempts, self.exc = verbalised, fail_attempts, exc
        self.victim = None
        self.seen_notes = []

    def describe(self):
        return {**super().describe(), **({"prob_method": "verbalised"} if self.verbalised else {})}

    async def decide(self, request):
        if "look" in request.questions:
            return await super().decide(request)
        if self.victim is None:
            self.victim = request.seed
        if request.seed == self.victim:
            self.seen_notes.append(request.note)
            if self.fail_attempts is None or len(self.seen_notes) <= self.fail_attempts:
                raise self.exc("slm: probabilities sum to 0.000 (all-zero distribution)")
        return await super().decide(request)


def rig_with(tmp_path, name, backend):
    workers = fake_workers()
    workers[name] = Worker(backend)
    return build_rig(tmp_path, workers=workers)


# --- parser stays strict ------------------------------------------------------------------------------------------

REQ = make_request("m")


def _reply(**over):
    ok = {"target": {"A": 0.6, "B": 0.2, "C": 0.1, "S": 0.05, "X": 0.05}, "release": 0.1,
          "amount": {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}}
    ok.update(over)
    return json.dumps(ok).replace("NaN", "NaN")


@pytest.mark.parametrize("amount", [
    {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},                 # the live failure: not releasing, amount all zero
    {"0": 0.5, "1": -0.5, "2": 0.5, "3": 0.5, "4": 0.0},
    {"0": 0.5, "1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},        # sums to 2.5: outside the normalisation range
])
def test_parser_rejects_degenerate_distributions(amount):
    with pytest.raises(DecisionError, match="slm:"):
        parse_slm_json(REQ, _reply(amount=amount))


def test_parser_rejects_nan_and_names_the_zero_sum():
    with pytest.raises(DecisionError, match="not a finite number"):
        parse_slm_json(REQ, _reply().replace('"0": 0.2', '"0": NaN', 1))
    with pytest.raises(DecisionError, match=r"sum to 0\.000 \(all-zero distribution\)"):
        parse_slm_json(REQ, _reply(amount={"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}))
    ok = parse_slm_json(REQ, _reply())
    assert ok["amount"].probabilities["0"] == pytest.approx(0.2)


def test_prompt_asks_for_full_distributions_neutrally():
    prompt = _slm_prompt(REQ)
    assert "full probability distribution" in prompt and "amount" in prompt
    assert "Note:" not in prompt                                          # no note unless a retry asks for one
    import dataclasses
    assert prompt in _slm_prompt(dataclasses.replace(REQ, note="x")) and "Note: x" in _slm_prompt(
        dataclasses.replace(REQ, note="x"))
    assert REQ.request_hash == dataclasses.replace(REQ, note="x").request_hash   # the note is not hashed


def test_nous_backend_raises_on_zero_sum_reply():
    reply = {"model": "deepseek/deepseek-v4.1-flash", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
             "choices": [{"finish_reason": "stop", "message": {"content": _reply(
                 amount={"0": 0, "1": 0, "2": 0, "3": 0, "4": 0})}}]}
    be = NousFWBackend("https://nous.test/v1", api_key="nk", retries=0,
                       transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))
    with pytest.raises(DecisionError, match="sum to 0.000"):
        asyncio.run(be.decide(make_request("deepseek/deepseek-v4.1-flash")))


# --- decision-level policy --------------------------------------------------------------------------------------

def test_zero_sum_is_retried_then_decision_error_and_run_continues(tmp_path):
    fw = FailingBackend(FW, "fw-fake", verbalised=True)
    res = run(rig_with(tmp_path, FW, fw))
    assert res.cell.get("error") is None and len(res.decision_log) == N_DEC          # the run finished
    bad = [r for r in res.decision_log if r["error"]]
    assert len(bad) == 1 and sum(1 for r in res.decision_log if not r["error"]) == N_DEC - 1
    row = bad[0]
    assert (row["target"], row["release"]) == ("S", 0) and row["error_reason"].startswith("decision_error: ")
    assert "all-zero" in row["error_reason"] and row["attempts"] == DECISION_RETRIES + 1
    assert len(fw.seen_notes) == DECISION_RETRIES + 1
    alarms = [e for e in res.board.by_type("alarm") if e.payload["note"].startswith("decision_error")]
    assert len(alarms) == 1 and alarms[0].payload["kind"] == "other"
    assert res.cell["alarms"] >= 1 and res.cell["identities"] == "none"           # fakes carry no identity; the error row adds none


def test_corrective_note_only_on_retries_two_and_three(tmp_path):
    fw = FailingBackend(FW, "fw-fake", verbalised=True)
    res = run(rig_with(tmp_path, FW, fw))
    first, second, third, fourth = fw.seen_notes
    assert first == "" and second == ""                                               # the first retry repeats the request
    assert third == fourth != "" and "sum to 0.000" in third and third.startswith(CORRECTIVE_NOTE[:20])
    row = next(r for r in res.decision_log if r["error"])
    assert row["notes"] == [third, fourth]                                            # recorded in the decision log
    assert all(r["notes"] == [] for r in res.decision_log if not r["error"])


def test_note_that_fixes_the_reply_gives_a_normal_decision(tmp_path):
    fw = FailingBackend(FW, "fw-fake", verbalised=True, fail_attempts=2)
    res = run(rig_with(tmp_path, FW, fw))
    assert res.cell["decision_errors"] == 0 and not any(r["error"] for r in res.decision_log)
    row = next(r for r in res.decision_log if r["attempts"] == 3)
    assert len(row["notes"]) == 1 and fw.seen_notes[-1] != ""


def test_non_verbalised_backend_gets_no_note(tmp_path):
    fw = FailingBackend(FW, "fw-fake", verbalised=False)
    res = run(rig_with(tmp_path, FW, fw))
    assert fw.seen_notes == [""] * (DECISION_RETRIES + 1) and res.cell["decision_errors"] == 1


def test_decision_error_metric_in_cell(tmp_path):
    res = run(rig_with(tmp_path, FW, FailingBackend(FW, "fw-fake", verbalised=True)))
    c = res.cell
    assert c["decision_errors"] == 1 and c["decision_error_share"] == pytest.approx(1 / N_DEC)
    assert c["decision_error_alarm"] == int(1 / N_DEC > DECISION_ERROR_ALARM)
    assert v41_cells.validate_cell(c) == [] and res.run_log["decision_errors"] == 1
    clean = run(build_rig(tmp_path / "clean"))
    assert clean.cell["decision_errors"] == 0 and clean.cell["decision_error_share"] == 0.0
    assert clean.cell["decision_error_alarm"] == 0


def test_alarm_flag_above_five_percent(tmp_path):
    class Half(FailingBackend):
        async def decide(self, request):
            if "look" not in request.questions and request.seed % 2 == 0:
                raise DecisionError("slm: reply is not JSON")
            return await FakeBackend.decide(self, request)

    res = run(rig_with(tmp_path, FW, Half(FW, "fw-fake")))
    assert res.cell["decision_errors"] > 0.05 * N_DEC and res.cell["decision_error_alarm"] == 1
    assert 0 < res.cell["decision_error_share"] < 1 and len(res.decision_log) == N_DEC


@pytest.mark.parametrize("name,arm,calibration", [(D1, "SW-B", False), (Q9, "SW-B", True), (FW, "FW", False)])
def test_same_policy_for_every_backend(tmp_path, name, arm, calibration):
    """vLLM (Q9), d1 and the Nous FW arm: a raised DecisionError is retried 3 times, the decision becomes the neutral
    fallback, the run continues."""
    be = FailingBackend(name, f"{name}-fake")
    res = run(rig_with(tmp_path, name, be), arm, calibration=calibration)
    assert len(be.seen_notes) == DECISION_RETRIES + 1
    assert res.cell.get("error") is None and len(res.decision_log) == N_DEC
    bad = [r for r in res.decision_log if r["error"]]
    assert len(bad) == 1 and (bad[0]["target"], bad[0]["release"]) == ("S", 0) and bad[0]["attempts"] == 4
    assert res.cell["decision_errors"] == 1 and v41_cells.validate_cell(res.cell) == []


def test_budget_abort_is_not_absorbed(tmp_path):
    be = FailingBackend(FW, "fw-fake", exc=DecisionAbort)
    with pytest.raises(DecisionAbort):
        run(rig_with(tmp_path, FW, be))
    assert len(be.seen_notes) == 1                                                    # no retry either


# --- the cache never turns an invalid reply into a valid one ------------------------------------------------------

def test_cache_does_not_store_invalid_replies(tmp_path):
    be = FailingBackend(FW, "m")
    be.victim = REQ.seed
    cache = CachedDecision(be, tmp_path / "cache")
    for _ in range(2):
        with pytest.raises(DecisionError):
            asyncio.run(cache.decide(REQ))
    assert not list((tmp_path / "cache").glob("*")) and len(be.seen_notes) == 2        # nothing stored, both calls went out
    good = FakeBackend(hashed_answers, model="m")
    fresh = CachedDecision(good, tmp_path / "cache")
    assert not asyncio.run(fresh.decide(REQ)).cached and asyncio.run(fresh.decide(REQ)).cached


def test_cache_ignores_a_stored_invalid_reply(tmp_path):
    good = FakeBackend(hashed_answers, model="m")
    cache = CachedDecision(good, tmp_path / "cache")
    asyncio.run(cache.decide(REQ))
    path = tmp_path / "cache" / f"{REQ.request_hash}.json"
    data = json.loads(path.read_text())
    data["answers"]["amount"]["probabilities"] = {k: 0.0 for k in data["answers"]["amount"]["probabilities"]}
    path.write_text(json.dumps(data))                                                  # a zero-sum reply on disk
    again = asyncio.run(cache.decide(REQ))
    assert not again.cached and sum(again.answers["amount"].probabilities.values()) == pytest.approx(1.0)
    path.write_text("not json")
    assert not asyncio.run(cache.decide(REQ)).cached
    offline = CachedDecision(good, tmp_path / "cache2", offline=True)
    with pytest.raises(DecisionError, match="offline"):
        asyncio.run(offline.decide(REQ))
