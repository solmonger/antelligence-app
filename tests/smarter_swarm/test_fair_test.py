"""Tests for the fair test harness (antelligence/experiments/fair_test.py) and its two scripts. No network, $0."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from antelligence.experiments import fair_test as ft
from antelligence.providers.base import ChatResponse, ProviderError, ProviderUnavailable

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs/research/fair-test-20261029"


PAID = (0.106, 0.513)  # a nonzero price, so the tests can check spend accounting (the real model is free)


@pytest.fixture(autouse=True)
def _priced(monkeypatch):
    monkeypatch.setattr(ft, "PRICE", PAID)


def test_the_frozen_model_is_free_and_released_before_every_forecast():
    source = (REPO / "antelligence/experiments/fair_test.py").read_text()
    assert "\nPRICE = (0.0, 0.0)" in source  # the fixture above only prices the tests
    assert ft.MODEL == "stepfun/step-3.7-flash:free" and ft.RELEASED == "2026-05-28"
    answers = [json.loads(line) for line in (DATA / "forecast_answers.jsonl").read_text().splitlines()]
    assert all(a["resolution_date"] > ft.RELEASED for a in answers)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _exam(qid: str = "q1") -> ft.Question:
    return ft.Question("exam", qid, "lsat-lr", "Which is right?\n\nOptions:\n(A)x\n(B)y")


def _forecast(qid: str = "f1") -> ft.Question:
    return ft.Question("forecast", qid, "politics", "Question: Will it happen?")


class Scripted:
    """A fake provider: ``reply(request) -> content``; can fail the first ``outages`` calls."""

    def __init__(self, reply, outages: int = 0, error: Exception = None) -> None:
        self.reply, self.outages, self.error, self.calls = reply, outages, error, 0

    async def complete(self, req):
        self.calls += 1
        if self.outages:
            self.outages -= 1
            raise ProviderUnavailable("down", status=503)
        if self.error is not None:
            raise self.error
        content = self.reply(req)
        return ChatResponse(content=content, model=req.model, request_hash=req.request_hash, prompt_tokens=1000,
                            completion_tokens=100, finish_reason="stop")


# ------------------------------------------------------------------- parsing and combining

@pytest.mark.parametrize("text,want", [
    ('{"reason": "r", "answer": "B"}', "B"),
    ('```json\n{"reason": "r", "answer": "c"}\n```', "C"),
    ('{"answer": "(D)"}', "D"),
    ('{"answer": "F"}', None),
    ('{"answer": "AB"}', None),
    ("no json here", None),
    ('{"answer": ', None),
    ("[1, 2]", None),
])
def test_parse_exam(text, want):
    assert ft.parse_reply("exam", text)[0] == want


@pytest.mark.parametrize("text,want", [
    ('{"reason": "r", "probability": 0.3}', 0.3),
    ('{"probability": "0.8"}', 0.8),
    ('{"probability": 70}', 0.7),
    ('{"probability": 170}', None),
    ('{"probability": -0.1}', None),
    ('{"probability": "NaN"}', None),
    ('{"probability": null}', None),
])
def test_parse_forecast(text, want):
    got = ft.parse_reply("forecast", text)[0]
    assert got == pytest.approx(want) if want is not None else got is None


def test_vote_and_mean():
    assert ft.vote(["A", "B", "B", None]) == "B"
    assert ft.vote(["C", "A", "A", "C"]) == "C"           # a tie goes to the one given first
    assert ft.vote([None, None]) is None
    assert ft.mean_prob([0.0, 1.0]) == pytest.approx(0.5)  # clipped to [0.01, 0.99] first
    assert ft.mean_prob([None]) == 0.5
    assert ft.mean_prob([0.2, None, 0.4]) == pytest.approx(0.3)


# ------------------------------------------------------------------- questions and prompts

def test_questions_file_must_not_carry_answers(tmp_path):
    p = tmp_path / "q.jsonl"
    for banned in ("label", "answer", "outcome", "solution", "other", "crowd_prob"):
        p.write_text(json.dumps({"id": "1", "question": "q", "options": ["(A)a"], banned: "A"}) + "\n")
        with pytest.raises(ValueError, match=banned):
            ft.load_questions(p, "exam")
    p.write_text(json.dumps({"id": "1", "question": "q", "options": ["(A)a"]}) + "\n" +
                 json.dumps({"id": "1", "question": "q2", "options": ["(A)a"]}) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        ft.load_questions(p, "exam")
    with pytest.raises(ValueError):
        ft.load_questions(p, "poll")


def test_rendering(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"id": "e1", "task": "lsat-ar", "passage": "P.", "question": "Q?",
                             "options": ["(A)a", "(B)b"]}) + "\n")
    (q,) = ft.load_questions(p, "exam")
    assert q.group == "lsat-ar" and q.text == "P.\n\nQ?\n\nOptions:\n(A)a\n(B)b"
    f = tmp_path / "f.jsonl"
    f.write_text(json.dumps({"id": "f1", "field": "science", "question": "Will X?", "background": "B.",
                             "open_date": "2026-07-01", "close_date": "2026-09-30"}) + "\n")
    (fq,) = ft.load_questions(f, "forecast")
    assert fq.group == "science" and "Will X?" in fq.text and "2026-09-30" in fq.text


def test_each_ant_sees_the_other_four_not_itself():
    q = _exam()
    peers = [(j, "ABCDE"[j], f"reason {j}") for j in range(5) if j != 2]
    text = ft.seen_messages(q, 2, peers)[1]["content"]
    assert "reason 2" not in text and all(f"reason {j}" in text for j in (0, 1, 3, 4))
    assert ft.LENSES["exam"][2] in text
    plain = ft.plain_messages(q)[1]["content"]
    assert not any(lens in plain for lens in ft.LENSES["exam"])
    assert len({ft.lens_messages(q, i)[1]["content"] for i in range(ft.K)}) == ft.K


def test_requests_are_deterministic_and_distinct():
    q = _exam()
    a = ft.request(ft.plain_messages(q), q.id, "single", 0)
    assert a.request_hash == ft.request(ft.plain_messages(q), q.id, "single", 0).request_hash
    hashes = {ft.request(ft.plain_messages(q), q.id, "single", i).request_hash for i in range(ft.SINGLE_SAMPLES)}
    assert len(hashes) == ft.SINGLE_SAMPLES  # ten real samples, not one answer cached ten times
    assert a.payload()["reasoning"] == {"effort": "low"} and a.model == ft.MODEL


# ------------------------------------------------------------------- the runner

def _letter(req):
    return '{"reason": "r", "answer": "A"}'


def test_runner_makes_twenty_calls_per_question_and_resumes_free(tmp_path):
    prov = Scripted(_letter)
    r1 = ft.Runner(prov, tmp_path / "c.jsonl", max_usd=1.0, backoff=())
    rows = asyncio.run(r1.run([_exam("a"), _exam("b")]))
    assert prov.calls == 40 and [len(rows[0][k]) for k in ("single", "swarm", "seen")] == [10, 5, 5]
    assert rows[0]["swarm"] == ["A"] * 5
    spent = r1.spent
    assert spent == pytest.approx(40 * (1000 * 0.106 + 100 * 0.513) / 1e6)
    prov2 = Scripted(_letter)
    r2 = ft.Runner(prov2, tmp_path / "c.jsonl", max_usd=1.0, backoff=())
    assert asyncio.run(r2.run([_exam("a"), _exam("b")])) == rows
    assert prov2.calls == 0 and r2.spent == pytest.approx(spent)


def test_runner_stops_at_the_cap(tmp_path):
    r = ft.Runner(Scripted(_letter), tmp_path / "c.jsonl", max_usd=0.01, backoff=())
    with pytest.raises(ft.SpendCap):
        asyncio.run(r.run([_exam(str(i)) for i in range(5)], concurrency=1))
    assert r.spent <= 0.01


def test_outage_is_retried_and_a_lasting_one_caches_nothing(tmp_path):
    flaky = Scripted(_letter, outages=2)
    r2 = ft.Runner(flaky, tmp_path / "b.jsonl", max_usd=1, backoff=(0, 0))
    assert asyncio.run(r2.run_question(_exam()))["swarm"] == ["A"] * 5
    down = Scripted(_letter, outages=99)
    r3 = ft.Runner(down, tmp_path / "c.jsonl", max_usd=1, backoff=(0,))
    with pytest.raises(ProviderUnavailable):
        asyncio.run(r3.run_question(_exam()))
    assert not (tmp_path / "c.jsonl").exists() and r3.spent == pytest.approx(0) and r3.reserved == pytest.approx(0)


class Replying:
    """A fake provider that returns a full ChatResponse built by ``make(req)``."""

    def __init__(self, make) -> None:
        self.make, self.calls = make, 0

    async def complete(self, req):
        self.calls += 1
        return self.make(req)


def _resp(req, content='{"answer": "A"}', model=None, finish="stop", extra=None):
    return ChatResponse(content=content, model=model or req.model, request_hash=req.request_hash,
                        prompt_tokens=1000, completion_tokens=100, finish_reason=finish, extra=extra or {})


def test_a_broken_provider_stops_the_run_fast_and_says_why(tmp_path):
    # The Oct 29 failure: every call refused, 6,820 times, and the run still "completed" with nothing to score.
    prov = Scripted(_letter, error=ProviderError("HTTP 400: unknown parameter"))
    r = ft.Runner(prov, tmp_path / "c.jsonl", max_usd=1, backoff=(), max_fail_streak=5)
    with pytest.raises(ft.BrokenRun, match="unknown parameter"):
        asyncio.run(r.run([_exam(str(i)) for i in range(50)], concurrency=1))
    assert prov.calls <= 15  # stops after the first question's batch, not after 1,000 calls
    rows = [json.loads(line) for line in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert rows and all(row["error"] == "ProviderError" and "unknown parameter" in row["reason"] for row in rows)


NO_CREDIT = ('HTTP 404: {"status":404,"message":"Model \'deepseek/deepseek-v4.1-flash\' requires available credits. '
             'Your account balance is too low to use paid models"}')


def test_an_empty_balance_stops_at_the_first_call_and_caches_nothing(tmp_path):
    # The Oct 29 smoke check: the portal answered 404 "requires available credits" to every call.
    prov = Scripted(_letter, error=ProviderError(NO_CREDIT, http_status=404))
    r = ft.Runner(prov, tmp_path / "c.jsonl", max_usd=1, backoff=())
    with pytest.raises(ft.OutOfCredits, match="Out of credits"):
        asyncio.run(r.run([_exam(str(i)) for i in range(5)], concurrency=1))
    assert prov.calls <= 15 and not (tmp_path / "c.jsonl").exists() and r.reserved == pytest.approx(0)
    assert ft.out_of_credits(ProviderError("x", http_status=402))
    assert not ft.out_of_credits(ProviderError("HTTP 404: model not found", http_status=404))


def test_unparseable_replies_also_stop_the_run(tmp_path):
    prov = Replying(lambda req: _resp(req, content="I think the answer is A."))
    r = ft.Runner(prov, tmp_path / "c.jsonl", max_usd=1, backoff=())
    with pytest.raises(ft.BrokenRun, match="not in the asked format"):
        asyncio.run(r.run_question(_exam()))
    assert r.spent > 0  # those replies were billed, and counted


def test_a_few_bad_replies_do_not_stop_the_run(tmp_path):
    n = {"i": 0}

    def make(req):
        n["i"] += 1
        return _resp(req, content="nonsense" if n["i"] % 4 == 0 else '{"answer": "B"}')
    r = ft.Runner(Replying(make), tmp_path / "c.jsonl", max_usd=1, backoff=())
    rows = asyncio.run(r.run([_exam(str(i)) for i in range(3)], concurrency=1))
    assert len(rows) == 3 and sum(a is None for row in rows for a in row["single"]) > 0


def test_failed_calls_in_the_cache_are_asked_again(tmp_path):
    cache = tmp_path / "c.jsonl"
    bad = ft.Runner(Scripted(_letter, error=ProviderError("refused")), cache, max_usd=1, backoff=(), max_fail_streak=99)
    assert asyncio.run(bad.run_question(_exam()))["single"] == [None] * 10
    good_prov = Scripted(_letter)
    good = ft.Runner(good_prov, cache, max_usd=1, backoff=())
    assert good.ignored == 20 and not good.cache
    assert asyncio.run(good.run_question(_exam()))["single"] == ["A"] * 10
    assert good_prov.calls == 20
    again = Scripted(_letter)
    assert asyncio.run(ft.Runner(again, cache, max_usd=1, backoff=()).run_question(_exam()))["swarm"] == ["A"] * 5
    assert again.calls == 0  # usable replies are reused


def test_served_model_names():
    assert ft.served_ok("stepfun/step-3.7-flash")      # the portal may report the free route without its tag
    assert ft.served_ok("stepfun/step-3.7-flash:free")
    assert ft.served_ok("stepfun-ai/Step-3.7-Flash")
    assert ft.served_ok("step-3.7-flash-0528")
    assert not ft.served_ok("stepfun/step-5-preview")
    assert not ft.served_ok("qwen/qwen3-8b")
    assert not ft.served_ok("")
    assert ft.served_ok("DeepSeek-V4.1-Flash", "deepseek/deepseek-v4.1-flash")
    assert not ft.served_ok("deepseek/deepseek-v4.1-pro", "deepseek/deepseek-v4.1-flash")


def test_reply_checks_keep_the_cost(tmp_path):
    other = ft.Runner(Replying(lambda req: _resp(req, model="qwen/qwen3-8b")), tmp_path / "a.jsonl", max_usd=1,
                      backoff=(), max_fail_streak=99)
    assert asyncio.run(other.run_question(_exam()))["single"] == [None] * 10
    assert other.failed == 20 and other.spent > 0
    cut = ft.Runner(Replying(lambda req: _resp(req, finish="length")), tmp_path / "b.jsonl", max_usd=1, backoff=(),
                    max_fail_streak=99)
    asyncio.run(cut.run_question(_exam()))
    rows = [json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()]
    assert {row["error"] for row in rows} == {"NotFinished"} and all(row["cost_usd"] > 0 for row in rows)
    renamed = ft.Runner(Replying(lambda req: _resp(req, model="stepfun-ai/Step-3.7-Flash", extra={"cost": 0.0001})),
                        tmp_path / "c.jsonl", max_usd=1, backoff=())
    assert asyncio.run(renamed.run_question(_exam()))["seen"] == ["A"] * 5
    assert renamed.spent == pytest.approx(20 * 0.0001)  # the portal's own cost wins over our estimate
    reloaded = ft.Runner(Replying(lambda req: _resp(req)), tmp_path / "b.jsonl", max_usd=1, backoff=())
    assert reloaded.spent == pytest.approx(cut.spent)  # money already spent on failed replies still counts


# ------------------------------------------------------------------- scoring

def _row(qid, kind, single, swarm, seen=None, group="g"):
    return {"id": qid, "kind": kind, "group": group, "single": single, "swarm": swarm, "seen": seen or swarm}


def test_score_exam_pass_and_fail():
    key = {f"q{i}": "A" for i in range(40)}
    better = [_row(f"q{i}", "exam", ["B"] * 10, ["A"] * 5) if i % 2 else _row(f"q{i}", "exam", ["A"] * 10, ["A"] * 5)
              for i in range(40)]
    res = ft.score(better, key)
    p = res["primary"]["exam"]
    assert p["swarm5"] == 1.0 and p["single5"] == 0.5 and p["pass"] and res["verdict"] == "supported"
    same = [_row(f"q{i}", "exam", ["A"] * 10, ["A"] * 5) for i in range(40)]
    assert ft.score(same, key)["verdict"] == "not supported"  # no gain is no pass
    assert ft.score(same, key)["secondary"]["exam"]["per_lens"] == [1.0] * 5


def test_score_forecast_uses_brier_with_lower_better():
    key = {f"f{i}": ("yes" if i % 2 else "no") for i in range(30)}
    rows = []
    for i in range(30):
        y = 0.9 if i % 2 else 0.1
        rows.append(_row(f"f{i}", "forecast", [0.5] * 10, [y] * 5))
    res = ft.score(rows, key, crowd={"f1": 0.8})
    p = res["primary"]["forecast"]
    assert p["metric"] == "brier" and p["swarm5"] < p["single5"] and p["gain"] > 0 and p["pass"]
    assert res["secondary"]["forecast"]["crowd"]["n"] == 1


def test_mixed_verdict():
    ek = {f"q{i}": "A" for i in range(20)}
    fk = {f"f{i}": "yes" for i in range(20)}
    rows = [_row(f"q{i}", "exam", ["B"] * 10, ["A"] * 5) for i in range(20)]
    rows += [_row(f"f{i}", "forecast", [0.9] * 10, [0.2] * 5) for i in range(20)]
    assert ft.score(rows, {**ek, **fk})["verdict"] == "mixed"


def test_a_run_with_no_usable_answers_gets_no_verdict():
    key = {f"q{i}": "A" for i in range(10)}
    rows = [_row(f"q{i}", "exam", [None] * 10, [None] * 5) for i in range(10)]
    res = ft.score(rows, key)
    assert res["usable_share"] == 0 and res["verdict"].startswith("invalid run")
    half = [_row(f"q{i}", "exam", ["A"] * 10, ["A", None, None, None, None]) for i in range(10)]
    assert ft.score(half, key)["verdict"].startswith("invalid run")  # 80% usable is still too few


def test_estimate_counts_twenty_calls():
    est = ft.estimate([_exam("a"), _forecast("b")])
    assert est["calls"] == 40 and est["usd"] >= 0 and est["input_tokens"] > 0


# ------------------------------------------------------------------- scripts and the frozen data

def _agieval_like(raw):
    raw.mkdir()
    prep = _load("fair_test_prep")
    for task in prep.TASKS:
        rows = [{"passage": None, "question": f"{task} q{i}", "options": ["(A)a", "(B)b", "(C)c", "(D)d"],
                 "label": "ABCD"[i % 4], "answer": None, "other": {"solution": "secret"}} for i in range(8)]
        rows.append({"passage": None, "question": "free response", "options": None, "label": None, "answer": "7"})
        (raw / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return prep


def test_prep_keeps_answers_out_of_the_questions(tmp_path):
    prep = _agieval_like(tmp_path / "raw")
    out = tmp_path / "out"
    assert prep.main(["--out", str(out), "--per-task", "5", "--raw-dir", str(tmp_path / "raw")]) == 0
    qtext = (out / "exam_questions.jsonl").read_text()
    assert "secret" not in qtext and '"label"' not in qtext
    qs = ft.load_questions(out / "exam_questions.jsonl", "exam")
    answers = [json.loads(line) for line in (out / "exam_answers.jsonl").read_text().splitlines()]
    assert len(qs) == 20 == len(answers) and {q.id for q in qs} == {a["id"] for a in answers}
    with pytest.raises(ValueError):
        prep.main(["--out", str(out), "--per-task", "9", "--raw-dir", str(tmp_path / "raw")])


def test_cli_freeze_dry_run_and_score(tmp_path, capsys):
    prep = _agieval_like(tmp_path / "raw")
    data = tmp_path / "data"
    prep.main(["--out", str(data), "--per-task", "3", "--raw-dir", str(tmp_path / "raw")])
    cli = _load("fair_test")
    with pytest.raises(SystemExit, match="freeze"):
        cli.main(["run", "--dry-run", "--data", str(data), "--kinds", "exam"])
    assert cli.main(["freeze", "--data", str(data)]) == 0
    assert cli.main(["plan", "--data", str(data), "--kinds", "exam"]) == 0
    with pytest.raises(SystemExit, match="--live"):
        cli.main(["run", "--data", str(data), "--kinds", "exam"])
    assert cli.main(["run", "--dry-run", "--data", str(data), "--kinds", "exam"]) == 0
    rows = (data / "rows_dry.jsonl").read_text().splitlines()
    assert len(rows) == 12
    assert cli.main(["score", "--data", str(data), "--rows", str(data / "rows_dry.jsonl")]) == 0
    result = json.loads((data / "result_dry.json").read_text())
    assert result["primary"]["exam"]["n"] == 12 and result["verdict"] in ("supported", "mixed", "not supported")
    with (data / "exam_answers.jsonl").open("a") as fh:   # any change after the freeze voids the run
        fh.write("\n")
    with pytest.raises(SystemExit, match="changed since freeze"):
        cli.main(["score", "--data", str(data), "--rows", str(data / "rows_dry.jsonl")])


def test_smoke_check(tmp_path, capsys, monkeypatch):
    prep = _agieval_like(tmp_path / "raw")
    data = tmp_path / "data"
    prep.main(["--out", str(data), "--per-task", "2", "--raw-dir", str(tmp_path / "raw")])
    cli = _load("fair_test")
    cli.main(["freeze", "--data", str(data)])
    assert cli.main(["smoke", "--dry-run", "--data", str(data)]) == 0
    assert "SMOKE CHECK PASSED" in capsys.readouterr().out
    monkeypatch.setattr(cli, "live_provider", lambda: Scripted(_letter, error=ProviderError("HTTP 404: no such model")))
    assert cli.main(["smoke", "--live", "--data", str(data)]) == 2
    out = capsys.readouterr().out
    assert "SMOKE CHECK FAILED" in out and "no such model" in out
    monkeypatch.setattr(cli, "live_provider", lambda: Scripted(_letter, error=ProviderError(NO_CREDIT, http_status=404)))
    assert cli.main(["smoke", "--live", "--data", str(data)]) == 2
    out = capsys.readouterr().out
    assert "STOPPED: Out of credits" in out and "SMOKE CHECK FAILED" in out


def test_cli_scores_forecasts_and_the_late_subset(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    qs, ans = [], []
    for i in range(6):
        qs.append({"id": f"f{i}", "question": f"Will {i} happen?", "background": "b", "open_date": "2026-06-01",
                   "close_date": "2026-10-01", "field": "science"})
        ans.append({"id": f"f{i}", "outcome": "yes" if i % 2 else "no", "crowd_prob": 0.5,
                    "resolution_date": "2026-05-01" if i < 3 else "2026-10-01", "source_url": "x"})
    (data / "forecast_questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qs))
    (data / "forecast_answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ans))
    cli = _load("fair_test")
    assert cli.main(["freeze", "--data", str(data)]) == 0
    assert cli.main(["run", "--dry-run", "--data", str(data), "--kinds", "forecast"]) == 0
    assert cli.main(["score", "--data", str(data), "--rows", str(data / "rows_dry.jsonl")]) == 0
    res = json.loads((data / "result_dry.json").read_text())
    late = res["secondary"]["forecast_resolved_after_release"]
    assert res["primary"]["forecast"]["n"] == 6 and late["n"] == 3 and "pass" not in late
    assert res["secondary"]["forecast"]["crowd"]["crowd_brier"] == 0.25


def test_committed_forecast_set_keeps_answers_apart():
    qs = ft.load_questions(DATA / "forecast_questions.jsonl", "forecast")
    answers = [json.loads(line) for line in (DATA / "forecast_answers.jsonl").read_text().splitlines()]
    assert len(qs) == len(answers) >= 100 and {q.id for q in qs} == {a["id"] for a in answers}
    assert {a["outcome"] for a in answers} == {"yes", "no"}
    assert all("2026-07-01" <= a["resolution_date"] <= "2026-10-29" for a in answers)


def test_committed_exam_set_matches_its_manifest():
    manifest = json.loads((DATA / "exam_manifest.json").read_text())
    for name in ("exam_questions.jsonl", "exam_answers.jsonl"):
        assert ft.sha256_file(DATA / name) == manifest[name]
    qs = ft.load_questions(DATA / "exam_questions.jsonl", "exam")
    assert len(qs) == 240 and {q.group for q in qs} == {"lsat-ar", "lsat-lr", "logiqa-en", "aqua-rat"}


# ------------------------------------------------------------------- the live stack, against a fake portal

def _portal(served="stepfun-ai/Step-3.7-Flash", status=200, finish="stop", seen=None):
    import httpx

    def handler(request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "reasoning is not a valid parameter"}})
        text = body["messages"][-1]["content"]
        content = '{"reason": "r", "answer": "A"}' if '"answer"' in text else '{"reason": "r", "probability": 0.7}'
        return httpx.Response(200, json={"id": "x", "model": served, "choices": [
            {"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 60}})
    return httpx.MockTransport(handler)


def test_live_stack_accepts_the_portal_spelling_and_logs_spend(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("NOUS_API_KEY=sk-test-not-real\n")
    monkeypatch.setenv("ANTELLIGENCE_NOUS_ENV_FILE", str(env))
    cli = _load("fair_test")
    seen = []
    ledger = tmp_path / "ledger.jsonl"
    prov = cli.live_provider(ledger=ledger, transport=_portal(seen=seen))
    r = ft.Runner(prov, tmp_path / "c.jsonl", max_usd=1, backoff=())
    row = asyncio.run(r.run_question(_exam()))
    assert row["single"] == ["A"] * 10 and r.failed == 0
    assert seen[0]["reasoning"] == {"effort": "low"} and seen[0]["model"] == ft.MODEL == "stepfun/step-3.7-flash:free"
    assert seen[0]["max_tokens"] == 4000
    logged = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(logged) == 20 and all(x["cost_usd"] > 0 and "error" not in x for x in logged)
    assert r.spent == pytest.approx(sum(x["cost_usd"] for x in logged))
    assert "sk-test-not-real" not in ledger.read_text() + (tmp_path / "c.jsonl").read_text()


def test_the_old_provider_setup_refused_a_differently_spelled_model_name(tmp_path):
    # The Oct 29 run's stack, checking the name exactly: a portal that spells the model differently fails every call
    # while still billing it. The new stack accepts the spelling; a different model is still refused (above).
    from antelligence.providers.openai_compat import OpenAICompatProvider

    old = OpenAICompatProvider("https://portal.test/v1", api_key="k", allowed_models=[ft.MODEL],
                               extra_body=ft.REASONING, transport=_portal())
    r = ft.Runner(old, tmp_path / "c.jsonl", max_usd=1, backoff=())
    with pytest.raises(ft.BrokenRun, match="differs from requested"):
        asyncio.run(r.run_question(_exam()))


def test_live_stack_reports_a_rejected_request(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("NOUS_API_KEY=sk-test-not-real\n")
    monkeypatch.setenv("ANTELLIGENCE_NOUS_ENV_FILE", str(env))
    cli = _load("fair_test")
    prov = cli.live_provider(ledger=tmp_path / "l.jsonl", transport=_portal(status=400))
    r = ft.Runner(prov, tmp_path / "c.jsonl", max_usd=1, backoff=())
    with pytest.raises(ft.BrokenRun, match="not a valid parameter"):
        asyncio.run(r.run_question(_exam()))


def test_models_lists_what_the_key_can_call(tmp_path, monkeypatch, capsys):
    import httpx

    env = tmp_path / ".env"
    env.write_text("NOUS_API_KEY=sk-test-not-real\n")
    monkeypatch.setenv("ANTELLIGENCE_NOUS_ENV_FILE", str(env))
    cli = _load("fair_test")
    # The Oct 30 listing: the paid route and the free route of the same model. Only the free one is the test's.
    listing = {"data": [{"id": "stepfun/step-3.7-flash", "pricing": {"prompt": "0.0000002000", "completion": "0.0000011500"}},
                        {"id": "stepfun/step-3.7-flash:free", "pricing": {"prompt": "0", "completion": "0"}},
                        {"id": "deepseek/deepseek-v4.1-flash"}]}
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, json=listing))
    assert cli.main(["models"]) == 0
    lines = capsys.readouterr().out.splitlines()
    marked = [line for line in lines if "the fair test calls this" in line]
    assert len(marked) == 1 and marked[0].startswith("stepfun/step-3.7-flash:free ")
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, json={"data": [{"id": "x/other"}]}))
    cli.main(["models"])
    assert "is not in this list" in capsys.readouterr().out
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(401, text="bad key sk-test-not-real"))
    assert cli.main(["models"]) == 2
    out = capsys.readouterr().out
    assert "HTTP 401" in out and "sk-test-not-real" not in out


def test_account_shows_key_and_free_route_without_a_model_call(tmp_path, monkeypatch, capsys):
    import httpx

    env = tmp_path / ".env"
    env.write_text("NOUS_API_KEY=sk-nous-abcdefghijklmnop\n")
    monkeypatch.setenv("ANTELLIGENCE_NOUS_ENV_FILE", str(env))
    cli = _load("fair_test")
    asked = []

    def get(url, **kw):
        asked.append(url)
        if url.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "stepfun/step-3.7-flash:free",
                                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        if url.endswith("/key"):
            return httpx.Response(200, json={"data": {"label": "sk-nous-abcdefghijklmnop", "usage": 0, "limit": None,
                                                      "is_free_tier": True}})
        return httpx.Response(404, text="not found")
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("account must not call a model"))
    assert cli.main(["account"]) == 0
    out = capsys.readouterr().out
    assert str(env) in out and "starts sk-nou..." in out and "sk-nous-abcdefghijklmnop" not in out
    assert "is_free_tier" in out and "stepfun/step-3.7-flash:free: listed, free" in out
    assert not any("chat/completions" in u for u in asked)
