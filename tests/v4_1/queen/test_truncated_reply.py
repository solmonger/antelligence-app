"""A truncated (finish_reason=length) queen reply is an invalid reply, not a provider failure (S22).

Fake providers only: no network, no keys."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from antelligence.experiments import v41_fallbacks
from antelligence.experiments.v41_sandbox import llm_proposer
from antelligence.providers import ProviderTruncated as ExportedTruncated
from antelligence.providers.base import ChatRequest, ProviderError, ProviderTruncated, ProviderUnavailable
from antelligence.providers.ledger import LedgeredChat, LedgerWriter
from antelligence.providers.openai_compat import OpenAICompatProvider
from antelligence.providers.wrappers import FakeProvider
from tests.v4_1.queen.conftest import make_queen, make_regions, make_squads

ROOT = Path(__file__).resolve().parents[3]
REQ = ChatRequest(model="m", messages=({"role": "user", "content": "hi"},))


def _body(finish):
    return {"model": "m", "choices": [{"message": {"content": "{"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 800}}


def _compat(finish):
    return OpenAICompatProvider("https://example.invalid/v1", api_key="k-test",
                                transport=httpx.MockTransport(lambda req: httpx.Response(200, json=_body(finish))))


def test_length_raises_provider_truncated_which_is_a_provider_error_but_not_an_outage():
    assert ExportedTruncated is ProviderTruncated
    with pytest.raises(ProviderTruncated) as ei:
        asyncio.run(_compat("length").complete(REQ))
    assert isinstance(ei.value, ProviderError) and not isinstance(ei.value, ProviderUnavailable)
    assert ei.value.http_status == 200 and "(length)" in str(ei.value)       # SLMBackend still keys on "(length)"


def test_other_abnormal_finishes_stay_plain_provider_errors():
    with pytest.raises(ProviderError) as ei:
        asyncio.run(_compat("content_filter").complete(REQ))
    assert type(ei.value) is ProviderError


def test_ledger_wrapper_keeps_the_truncated_class(tmp_path):
    chat = LedgeredChat(_compat("length"), LedgerWriter(tmp_path / "u.jsonl"), role="queen", expected_model="m", price=(0.0, 0.0))
    with pytest.raises(ProviderTruncated):
        asyncio.run(chat.complete(REQ))


def test_queen_counts_length_as_invalid_reply_and_the_episode_stays_cacheable(tmp_path):
    from scripts import v41_sandbox as sb
    provider_calls = []

    def respond(req):
        provider_calls.append(req)
        raise ProviderTruncated("generation did not finish normally (length)", http_status=200)
    queen, provider, _ = make_queen(respond)
    step = queen.step(0, make_regions(), make_squads(), 0)
    assert step.fell_back and step.fallback_cause == "invalid_reply"
    assert queen.fallbacks == {"provider": 0, "invalid_reply": 1, "cap": 0} and len(provider.requests) == 1   # no retry
    note = step.alarm.payload["note"]
    assert note.startswith("queen fallback: ProviderTruncated") and not v41_fallbacks.is_provider_fallback_note(note)
    cache = sb.ResultsCache(tmp_path / "rc.jsonl", "ctx")
    cache.put("h", "SW-B", "c1", 1, 0.5, {}, queen_fallbacks=queen.fallbacks, run_id="r" * 64)    # not refused
    assert cache.get("h", "SW-B", "c1", 1) is not None


def test_a_real_provider_failure_is_still_a_provider_fallback():
    queen, _, _ = make_queen(lambda r: (_ for _ in ()).throw(ProviderError("HTTP 400: bad request", http_status=400)))
    step = queen.step(0, make_regions(), make_squads(), 0)
    assert step.fallback_cause == "provider" and queen.fallbacks["provider"] == 1
    assert v41_fallbacks.is_provider_fallback_note(step.alarm.payload["note"])


def test_legacy_length_alarm_is_not_provider_tainted():
    old = "queen fallback: ProviderError: ProviderError: generation did not finish normally (length)"
    assert not v41_fallbacks.is_provider_fallback_note(old)
    assert v41_fallbacks.is_provider_fallback_note("queen fallback: ProviderError: HTTP 400: bad request")


def test_proposer_treats_a_truncated_reply_as_an_invalid_proposal():
    provider = FakeProvider(lambda r: (_ for _ in ()).throw(ProviderTruncated("generation did not finish normally (length)")),
                            model="m")
    got = llm_proposer(provider, "m")({"round": 1})
    assert "unparseable" in got and len(provider.requests) == 1


def test_purge_ignores_length_fallbacks_and_still_drops_provider_ones(tmp_path):
    spec = importlib.util.spec_from_file_location("v41_purge_s22", ROOT / "scripts" / "v41_purge_provider_fallbacks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tmp_path / "work"
    notes = {"len_new": "queen fallback: ProviderTruncated: ProviderTruncated: generation did not finish normally (length)",
             "len_old": "queen fallback: ProviderError: ProviderError: generation did not finish normally (length)",
             "bad": "queen fallback: ProviderError: HTTP 400: bad request"}
    rows = []
    for i, (case, note) in enumerate(notes.items()):
        run_id = f"{i}".ljust(64, "0")
        d = work / "runs" / run_id
        d.mkdir(parents=True)
        (d / "cell.json").write_text(json.dumps({"run_id": run_id, "arm": "SW-B", "case": case, "seed": 1}))
        (d / "decisions.jsonl").write_text("")
        (d / f"{run_id}.jsonl").write_text(json.dumps({"harness_cfg_sha256": "h" * 20}) + "\n" + json.dumps(
            {"type": "alarm", "payload": {"note": note}}) + "\n")
        rows.append({"harness": "h" * 20, "arm": "SW-B", "case": case, "seed": 1, "coverage": 0.5, "stats": {},
                     "ctx": "c", "run_id": run_id})
    (work / "results-cache.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = mod.purge(work, apply=True)
    kept = [json.loads(line)["case"] for line in (work / "results-cache.jsonl").read_text().splitlines()]
    assert kept == ["len_new", "len_old"], report
