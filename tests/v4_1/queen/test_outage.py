"""Queen and proposer provider outages: wait and retry the same request, never a silent fallback (mistake-2039).

Fake providers, injectable sleep and clock: nothing here waits or touches a network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from antelligence.kernel.decision import DecisionAbort
from antelligence.providers.base import BudgetExceeded, ChatRequest, ProviderError, ProviderUnavailable
from antelligence.providers.ledger import LedgeredChat, LedgerWriter
from antelligence.providers.openai_compat import OpenAICompatProvider
from antelligence.providers.outage import (DEFAULT_MAX_WAIT_S, EXIT_QUEEN_UNAVAILABLE, OutagePolicy, QueenUnavailable,
                                           complete_with_outage_retry)
from antelligence.providers.wrappers import FakeProvider
from tests.v4_1.queen.conftest import N_SQUADS, good_reply, make_queen, make_regions, make_squads


class Clock:
    """A fake time source whose sleep advances it, so backoff arithmetic is checked without waiting."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    async def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def policy(self, max_wait=DEFAULT_MAX_WAIT_S):
        return OutagePolicy(max_wait_s=max_wait, sleep=self.sleep, clock=self)


def outage_for(seconds, clock, then):
    """A responder that raises HTTP 401 until ``seconds`` of fake time have passed, then answers ``then``."""
    def respond(req):
        if clock.now < seconds:
            raise ProviderUnavailable("HTTP 401: invalid, blocked or out of funds", status=401)
        return then
    return respond


REQ = ChatRequest(model="m", messages=({"role": "user", "content": "hi"},))


# --- the retry helper --------------------------------------------------------------------------------------------------

def test_backoff_is_15_30_60_then_120_and_stops_at_the_max_wait():
    clock = Clock()
    provider = FakeProvider(lambda r: (_ for _ in ()).throw(ProviderUnavailable("HTTP 503", status=503)))
    with pytest.raises(QueenUnavailable) as ei:
        asyncio.run(complete_with_outage_retry(provider, REQ, clock.policy(600)))
    assert clock.sleeps == [15, 30, 60, 120, 120, 120, 120, 15]      # the last sleep is cut to the 600 s budget
    assert sum(clock.sleeps) == 600 and isinstance(ei.value, DecisionAbort)
    assert len(provider.requests) == len(clock.sleeps) + 1


def test_outage_shorter_than_max_wait_returns_the_same_answer_on_the_same_request():
    clock = Clock()
    provider = FakeProvider(outage_for(100, clock, "ok"), model="m")
    response = asyncio.run(complete_with_outage_retry(provider, REQ, clock.policy()))
    assert response.content == "ok" and clock.sleeps == [15, 30, 60]
    assert len({r.request_hash for r in provider.requests}) == 1 and len(provider.requests) == 4


@pytest.mark.parametrize("exc", [ProviderError("generation did not finish normally (length)"), BudgetExceeded("call budget"),
                                 RuntimeError("odd"), ProviderError("HTTP 400: bad request")])
def test_only_outages_are_retried(exc):
    clock = Clock()
    provider = FakeProvider(lambda r: (_ for _ in ()).throw(exc))
    with pytest.raises(type(exc)):
        asyncio.run(complete_with_outage_retry(provider, REQ, clock.policy()))
    assert clock.sleeps == [] and len(provider.requests) == 1


def test_zero_max_wait_means_no_retry():
    clock = Clock()
    provider = FakeProvider(lambda r: (_ for _ in ()).throw(ProviderUnavailable("HTTP 429", status=429)))
    with pytest.raises(QueenUnavailable):
        asyncio.run(complete_with_outage_retry(provider, REQ, clock.policy(0)))
    assert clock.sleeps == [] and len(provider.requests) == 1


def test_policy_validates():
    for bad in (dict(max_wait_s=-1), dict(initial_backoff_s=0), dict(factor=0.5), dict(max_backoff_s=1)):
        with pytest.raises(ValueError):
            OutagePolicy(**bad)
    assert EXIT_QUEEN_UNAVAILABLE == 75 and DEFAULT_MAX_WAIT_S == 600


# --- which provider errors are outages --------------------------------------------------------------------------------

def _compat(handler):
    return OpenAICompatProvider("https://example.invalid/v1", api_key="k-test", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status,outage", [(401, True), (429, True), (500, True), (502, True), (503, True),
                                           (400, False), (403, False), (404, False)])
def test_http_status_classification(status, outage):
    provider = _compat(lambda req: httpx.Response(status, text="no"))
    with pytest.raises(ProviderError) as ei:
        asyncio.run(provider.complete(REQ))
    assert isinstance(ei.value, ProviderUnavailable) == outage
    if outage:
        assert ei.value.status == status


def test_transport_error_is_an_outage():
    def boom(req):
        raise httpx.ConnectError("refused")
    with pytest.raises(ProviderUnavailable) as ei:
        asyncio.run(_compat(boom).complete(REQ))
    assert ei.value.status is None and "k-test" not in str(ei.value)


def test_malformed_200_is_not_an_outage():
    provider = _compat(lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError) as ei:
        asyncio.run(provider.complete(REQ))
    assert not isinstance(ei.value, ProviderUnavailable)


def test_ledger_wrapper_keeps_the_outage_class_and_logs_every_attempt(tmp_path):
    provider = LedgeredChat(_compat(lambda req: httpx.Response(401, text="blocked")),
                            LedgerWriter(tmp_path / "u.jsonl", redact=["k-test"]), role="queen", expected_model="m", price=(0.0, 0.0))
    clock = Clock()
    with pytest.raises(QueenUnavailable):
        asyncio.run(complete_with_outage_retry(provider, REQ, clock.policy(100)))
    rows = (tmp_path / "u.jsonl").read_text().splitlines()
    assert len(rows) == len(clock.sleeps) + 1 and all("error" in json.loads(r) for r in rows)   # the ledger sees each try


# --- the in-episode queen ---------------------------------------------------------------------------------------------

def test_short_outage_gives_the_same_plan_with_no_fallback_and_one_counted_call():
    clock = Clock()
    plain, _, _ = make_queen()
    expected = plain.step(0, make_regions(), make_squads(), 0)
    queen, provider, board = make_queen(outage_for(100, clock, good_reply()), outage=clock.policy())
    step = queen.step(0, make_regions(), make_squads(), 0)
    assert not step.fell_back and step.alarm is None and step.fallback_cause is None
    assert [e.payload["region_ids"] for e in step.routes] == [e.payload["region_ids"] for e in expected.routes]
    assert queen.calls == 1 and queen.outage.retries == 3 and len(provider.requests) == 4
    assert queen.fallbacks == {"provider": 0, "invalid_reply": 0, "cap": 0}
    assert not [e for e in board.entries() if e.type == "alarm"]


def test_retries_do_not_use_up_the_call_cap():
    clock = Clock()
    flaky = {"n": 0}

    def respond(req):
        flaky["n"] += 1
        if flaky["n"] % 3:
            raise ProviderUnavailable("HTTP 429", status=429)
        return good_reply()
    queen, _, _ = make_queen(respond, outage=clock.policy())
    for tick in range(0, 180, 30):
        assert not queen.step(tick, make_regions(), make_squads(), 100_000 * (tick // 30 + 1)).fell_back
    assert queen.calls == 6 and queen.fallbacks["cap"] == 0


def test_long_outage_raises_queen_unavailable_and_writes_no_fallback():
    clock = Clock()
    queen, _, board = make_queen(lambda r: (_ for _ in ()).throw(ProviderUnavailable("HTTP 401", status=401)),
                                 outage=clock.policy(600))
    with pytest.raises(QueenUnavailable):
        queen.step(0, make_regions(), make_squads(), 0)
    assert not [e for e in board.entries() if e.type == "alarm"] and sum(queen.fallbacks.values()) == 0


def test_invalid_replies_still_fall_back_and_are_counted_by_cause():
    queen, _, board = make_queen(lambda r: "not json")
    step = queen.step(0, make_regions(), make_squads(), 0)
    assert step.fell_back and step.fallback_cause == "invalid_reply" and queen.fallbacks["invalid_reply"] == 1
    assert step.alarm.payload["note"].startswith("queen fallback: QueenReplyError")
    # over the cap
    queen, _, _ = make_queen(max_calls=1)
    queen.step(0, make_regions(), make_squads(), 0)
    capped = queen.step(30, make_regions(), make_squads(), 0)
    assert capped.fell_back and capped.fallback_cause == "cap" and queen.fallbacks == {"provider": 0, "invalid_reply": 0, "cap": 1}
    # a budget trip is a cap, not a provider fault
    queen, _, _ = make_queen(lambda r: (_ for _ in ()).throw(BudgetExceeded("token budget")))
    assert queen.step(0, make_regions(), make_squads(), 0).fallback_cause == "cap"


def test_non_outage_provider_failure_is_a_provider_fallback():
    queen, provider, _ = make_queen(lambda r: (_ for _ in ()).throw(ProviderError("HTTP 400: bad request")))
    step = queen.step(0, make_regions(), make_squads(), 0)
    assert step.fell_back and step.fallback_cause == "provider" and queen.fallbacks["provider"] == 1
    assert len(provider.requests) == 1 and step.alarm.payload["note"].startswith("queen fallback: ProviderError")


def test_budget_hard_stop_is_not_retried_or_absorbed():
    from antelligence.experiments.v41_budget import BudgetRefused
    queen, provider, _ = make_queen(lambda r: (_ for _ in ()).throw(BudgetRefused("nous stop")), outage=Clock().policy())
    with pytest.raises(BudgetRefused):
        queen.step(0, make_regions(), make_squads(), 0)
    assert len(provider.requests) == 1


# --- the sandbox proposer ---------------------------------------------------------------------------------------------

def test_proposer_waits_out_a_short_outage_and_aborts_on_a_long_one():
    from antelligence.experiments.v41_sandbox import llm_proposer
    ctx = {"round": 1}
    clock = Clock()
    reply = json.dumps({"patch": [], "prediction": {"metric": "coverage", "delta_ppm": 5}})
    provider = FakeProvider(outage_for(45, clock, reply), model="m")
    got = llm_proposer(provider, "m", outage=clock.policy())(ctx)
    assert got["prediction"]["delta_ppm"] == 5 and clock.sleeps == [15, 30] and len(provider.requests) == 3
    assert len({r.request_hash for r in provider.requests}) == 1
    clock2 = Clock()
    down = FakeProvider(lambda r: (_ for _ in ()).throw(ProviderUnavailable("HTTP 503", status=503)), model="m")
    with pytest.raises(QueenUnavailable):
        llm_proposer(down, "m", outage=clock2.policy(60))(ctx)
