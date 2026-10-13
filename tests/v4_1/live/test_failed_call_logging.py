"""S18: every failed provider call records http_status and a redacted, truncated error_body (ledger row + error text)."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from antelligence.kernel.decision import DecisionError
from antelligence.providers.base import ChatRequest, ProviderError
from antelligence.providers.decision_backends import LedgeredDecision, NousFWBackend
from antelligence.providers.ledger import LedgeredChat, LedgerWriter
from antelligence.providers.openai_compat import OpenAICompatProvider
from antelligence.providers.redact import redact_body
from tests.v4_1.workers.helpers import BRIEF, Recorder, chat_reply, default_tops, make_request

KEY = "nk-FAKE-secret-value-123456"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"


def run(coro):
    return asyncio.run(coro)


def test_redact_body_scrubs_and_truncates():
    text = f"bad key sk-abcdefghijklmnop and Bearer abc.def.ghi and {JWT} and {'A' * 40} and {KEY} ok"
    out = redact_body(text, [KEY])
    for secret in ("sk-abcdefghijklmnop", "abc.def.ghi", JWT, "A" * 40, KEY):
        assert secret not in out
    assert "[redacted]" in out and "***" not in out and "ok" in out
    assert len(redact_body("x " * 1000)) == 500
    # a key straddling the cut cannot survive as a fragment
    assert "sk-" not in redact_body("y" * 495 + " sk-abcdefghijklmnop")


def test_openai_compat_error_carries_status_and_body():
    body = f'{{"error": "invalid token {KEY}", "detail": "{JWT}"}}'
    transport = httpx.MockTransport(lambda r: httpx.Response(401, text=body))
    prov = OpenAICompatProvider("https://nous.test/v1", api_key=KEY, transport=transport)
    with pytest.raises(ProviderError) as ei:
        run(prov.complete(ChatRequest("m", ({"role": "user", "content": "hi"},), max_tokens=8)))
    exc = ei.value
    assert exc.http_status == 401 and "HTTP 401" in str(exc)
    assert KEY not in str(exc) and JWT not in str(exc) and KEY not in exc.error_body and "[redacted]" in exc.error_body


def test_ledgered_chat_row_has_status_and_body(tmp_path):
    transport = httpx.MockTransport(lambda r: httpx.Response(401, text=f"Bearer {KEY} rejected " + "z" * 800))
    prov = OpenAICompatProvider("https://nous.test/v1", api_key=KEY, transport=transport)
    writer = LedgerWriter(tmp_path / "u.jsonl", redact=[KEY])
    chat = LedgeredChat(prov, writer, role="queen", expected_model="deepseek/deepseek-v4.1-flash")
    with pytest.raises(ProviderError) as ei:
        run(chat.complete(ChatRequest("deepseek/deepseek-v4.1-flash", ({"role": "user", "content": "hi"},), max_tokens=8)))
    assert ei.value.http_status == 401
    (row,) = writer.rows()
    assert row["http_status"] == 401 and 0 < len(row["error_body"]) <= 500
    assert KEY not in (tmp_path / "u.jsonl").read_text()


def _nous(handler, retries):
    rec = Recorder(handler)
    return NousFWBackend("https://nous.test/v1", api_key=KEY, retries=retries, backoff_s=0.0, transport=rec.transport()), rec


def test_every_retry_attempt_gets_a_row_and_the_final_error_has_status_and_body(tmp_path):
    req = make_request("deepseek/deepseek-v4.1-flash", brief_ids=BRIEF)
    be, rec = _nous(lambda r: httpx.Response(401, text=f"expired {KEY} {JWT}"), retries=2)
    writer = LedgerWriter(tmp_path / "u.jsonl", redact=[KEY])
    led = LedgeredDecision(be, writer)
    with pytest.raises(DecisionError) as ei:
        run(led.decide(req))
    assert len(rec.posts) == 3
    rows = writer.rows()
    assert len(rows) == 3                                     # two retried attempts + the call's own final row
    assert [r["error"].split(":")[0] for r in rows[:2]] == ["attempt 1/3", "attempt 2/3"]
    for row in rows:
        assert row["http_status"] == 401 and "expired" in row["error_body"] and len(row["error_body"]) <= 500
    assert ei.value.http_status == 401 and "HTTP 401: expired" in str(ei.value)
    blob = (tmp_path / "u.jsonl").read_text() + str(ei.value)
    assert KEY not in blob and JWT not in blob


def test_a_retry_that_succeeds_still_logs_the_failed_attempt(tmp_path):
    req = make_request("deepseek/deepseek-v4.1-flash", brief_ids=BRIEF)
    state = {"n": 0}

    def post(r):
        state["n"] += 1
        return httpx.Response(401, text="token expired") if state["n"] == 1 else httpx.Response(500, text="boom")
    be, rec = _nous(post, retries=1)
    writer = LedgerWriter(tmp_path / "u.jsonl")
    with pytest.raises(DecisionError):
        run(LedgeredDecision(be, writer).decide(req))
    statuses = [r.get("http_status") for r in writer.rows()]
    assert statuses[:2] == [401, 500]


def test_permanent_4xx_records_status_and_body(tmp_path):
    req = make_request("deepseek/deepseek-v4.1-flash", brief_ids=BRIEF)
    be, _ = _nous(lambda r: httpx.Response(400, text="bad request " + KEY), retries=2)
    writer = LedgerWriter(tmp_path / "u.jsonl")
    with pytest.raises(DecisionError) as ei:
        run(LedgeredDecision(be, writer).decide(req))
    assert ei.value.http_status == 400 and KEY not in str(ei.value)
    assert all(r["http_status"] == 400 for r in writer.rows()) and len(writer.rows()) == 1
