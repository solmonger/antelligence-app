"""Tests for the decision-model add-on (antelligence/experiments/dm_addon.py and scripts/dm_addon.py). No network, $0."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import fair_test as ft
from antelligence.kernel.decision import (MAX_STATE_CHARS, Answer, DecisionError, DecisionRequest, DecisionResponse,
                                          Question)

REPO = Path(__file__).resolve().parents[2]
FAIR = REPO / "docs/research/fair-test-20261029"
DATA = REPO / "docs/research/dm-addon-20261030"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _exam(qid: str = "q1", group: str = "lsat-lr") -> ft.Question:
    return ft.Question("exam", qid, group, "Which is right?\n\nOptions:\n(A)x\n(B)y\n(C)z")


def _forecast(qid: str = "f1") -> ft.Question:
    return ft.Question("forecast", qid, "politics", f"Question: Will {qid} happen?")


OPTS = {"A": "x", "B": "y", "C": "z"}


def _choice(probs):
    return {"choice": max(probs, key=probs.get), "probs": probs, "conf": None}


# ------------------------------------------------------------------ requests


def test_long_request_has_room_and_the_kernel_default_is_unchanged():
    long_state = "x" * 5000
    q = {"a": Question("noul", "Is it?")}
    with pytest.raises(ValueError, match=str(MAX_STATE_CHARS)):
        DecisionRequest("m", long_state, q)
    assert dm.LongRequest("m", long_state, q).state == long_state
    with pytest.raises(ValueError):
        dm.LongRequest("m", "x" * (dm.STATE_LIMIT + 1), q)
    # the limit is a class attribute, so it changes neither the payload nor the hash
    short = DecisionRequest("m", "hi", q)
    assert dm.LongRequest("m", "hi", q).request_hash == short.request_hash


def test_exam_request_has_clean_options_in_a_seeded_order():
    raw = {"options": ["(A)first", "(B) second", "(C)third", "(D)fourth"]}
    assert dm.option_texts(raw) == {"A": "first", "B": "second", "C": "third", "D": "fourth"}
    q = _exam()
    r1 = dm.dm_request("jev-1.13.0", q, "dm1", OPTS)
    r2 = dm.dm_request("jev-1.13.0", q, "dm1", OPTS)
    assert r1.request_hash == r2.request_hash                   # same seed, same order
    wire = r1.payload()["questions"]["answer"]
    assert wire["type"] == "choice" and set(wire["criteria"]) == {"A", "B", "C"}
    assert wire["criteria"]["B"] == "y" and r1.state == q.text
    orders = {tuple(dm.dm_request("m", _exam(f"q{i}"), "dm1", OPTS).questions["answer"].options) for i in range(20)}
    assert len(orders) > 1                                       # not always A first
    with pytest.raises(ValueError, match="options"):
        dm.dm_request("m", q, "dm1", None)


def test_forecast_request_is_a_yes_no_question():
    r = dm.dm_request("m", _forecast(), "dm1")
    wire = r.payload()["questions"]["answer"]
    assert wire == {"type": "noul", "instructions": "Will this question resolve YES?", "criteria": dm.NOUL_CRITERIA}


def test_read_shows_every_ant_with_a_trimmed_reason():
    ants = (dm.Ant("A", "because " + "a" * 500), dm.Ant(None, ""), dm.Ant("C", "short"))
    state = dm.read_state(_exam(), ants)
    assert "3 people answered this on their own" in state
    assert "Person 1 answered A: because" in state and "a" * 301 not in state
    assert "Person 2 answered nothing usable." in state and "Person 3 answered C: short" in state
    f = dm.read_state(_forecast(), (dm.Ant(0.7, "base rate"),))
    assert "Person 1 gave 0.70 chance of YES: base rate" in f
    with pytest.raises(ValueError, match="ants"):
        dm.dm_request("m", _exam(), "read", OPTS)
    with pytest.raises(ValueError):
        dm.dm_request("m", _exam(), "vote", OPTS)


# ------------------------------------------------------------------ combining


def test_vote_with_dm_breaks_a_tie_and_adds_nothing_when_it_failed():
    ants = ["A", "A", "B", "B", None]
    assert dm.vote_with_dm("exam", ants, _choice({"A": 0.3, "B": 0.6, "C": 0.1})) == "B"
    assert dm.vote_with_dm("exam", ants, None) == "A"            # a tie goes to the earlier letter
    assert dm.vote_with_dm("exam", ["C", "C", "C", "A", "B"], _choice({"A": 0.9, "B": 0.05, "C": 0.05})) == "C"
    assert dm.vote_with_dm("exam", [None] * 5, None) is None
    assert dm.vote_with_dm("forecast", [0.2, 0.4], {"p": 0.9}) == pytest.approx(0.5)
    assert dm.vote_with_dm("forecast", [0.2, 0.4], None) == pytest.approx(0.3)
    assert dm.vote_with_dm("forecast", [0.0, 1.0], {"p": 1.0}) == pytest.approx((0.01 + 0.99 + 0.99) / 3)


def test_confidence_and_dm_value():
    assert dm.confidence("exam", _choice({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})) == pytest.approx(0)
    assert dm.confidence("exam", _choice({"A": 1.0, "B": 0.0})) == pytest.approx(1)
    assert dm.confidence("exam", {"choice": "A", "probs": {"A": 0.6, "B": 0.4}, "conf": 0.33}) == 0.33
    assert dm.confidence("forecast", {"p": 0.9}) == pytest.approx(0.8)
    assert dm.confidence("forecast", None) == 0
    assert dm.dm_value("forecast", None) == 0.5 and dm.dm_value("forecast", {"p": 1.0}) == 0.99
    assert dm.dm_value("exam", None) is None


def _row(qid, kind, ants, baselines=None, group="lsat-lr"):
    calls = {k: 5 for k in (baselines or {})}
    return dm.SwarmRow(qid, kind, group, tuple(dm.Ant(a) for a in ants), baselines or {}, calls)


def test_cheap_bar_is_the_cheapest_that_keeps_the_ants_quality():
    # The model is right whenever it is sure (confidence 1), and wrong when unsure; the ants are always right.
    key, rows = {}, []
    for i in range(10):
        sure = i < 6
        key[f"q{i}"] = "A"
        ans = _choice({"A": 1.0, "B": 0.0}) if sure else _choice({"A": 0.4, "B": 0.6})
        rows.append((_row(f"q{i}", "exam", ["A"] * 5), ans))
    got = dm.pick_bar("exam", rows, key)
    assert got["bar"] > 0.2 and got["asked_share"] == pytest.approx(0.4)
    assert dm.cheap_answer("exam", ["A"] * 5, rows[0][1], got["bar"]) == ("A", False)
    assert dm.cheap_answer("exam", ["A"] * 5, rows[9][1], got["bar"]) == ("A", True)
    # a model that is never right is never trusted
    bad = [(r, _choice({"A": 0.0, "B": 1.0})) for r, _ in rows]
    assert dm.pick_bar("exam", bad, key) == {"bar": 1.01, "asked_share": 1.0}


def test_tune_half_is_fixed_and_about_half():
    ids = [f"x{i}" for i in range(400)]
    first = [dm.tune_half(i) for i in ids]
    assert first == [dm.tune_half(i) for i in ids] and 150 < sum(first) < 250


# ------------------------------------------------------------------ exports


class FakeChat:
    """Lens-aware fake: ant i answers 'ABCDE'[i % 2] with reason 'r<i>'; forecasts 0.1 * (i + 1)."""

    async def complete(self, req):
        from antelligence.providers.base import ChatResponse

        text = req.messages[-1]["content"]
        lens = next((i for i, l in enumerate(ft.LENSES["exam"] + ft.LENSES["forecast"]) if l in text), 0) % 5
        body = ({"reason": f"r{lens}", "answer": "AB"[lens % 2]} if '"answer"' in text
                else {"reason": f"r{lens}", "probability": 0.1 * (lens + 1)})
        return ChatResponse(content=json.dumps(body), model=req.model, request_hash=req.request_hash,
                            prompt_tokens=10, completion_tokens=5, finish_reason="stop")


def _fair_run(tmp_path, qs):
    runner = ft.Runner(FakeChat(), tmp_path / "cache.jsonl", max_usd=1.0, price=(0.0, 0.0))
    rows = asyncio.run(runner.run(qs))
    return rows, dm.read_jsonl(tmp_path / "cache.jsonl")


def test_export_fair_test_finds_every_reason_and_refuses_a_mismatch(tmp_path):
    qs = [_exam("e1"), _forecast("f1")]
    rows, cache = _fair_run(tmp_path, qs)
    out, stats = dm.export_fair_test(qs, rows, cache)
    assert stats == {"reasons_found": 10, "reasons_missing": 0}
    e = next(r for r in out if r.id == "e1")
    assert [a.answer for a in e.ants] == ["A", "B", "A", "B", "A"] and [a.reason for a in e.ants][:2] == ["r0", "r1"]
    assert e.baselines["swarm5"] == "A" and e.calls == {"single1": 1, "single5": 5, "single10": 10, "swarm5": 5,
                                                        "seen10": 10}
    f = next(r for r in out if r.id == "f1")
    assert [a.answer for a in f.ants] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert dm.SwarmRow.from_json(json.loads(json.dumps(e.to_json()))) == e
    rows[0]["swarm"][0] = "C"                                     # rows from another run
    with pytest.raises(ValueError, match="same run"):
        dm.export_fair_test(qs, rows, cache)
    _, stats = dm.export_fair_test(qs, [r for r in rows if r["id"] == "f1"], [])
    assert stats == {"reasons_found": 0, "reasons_missing": 5}


def test_export_shared_read_uses_the_alone_arm():
    rows = [{"id": "e1", "group": "aqua-rat", "reader": "B", "alone": ["A", "A", "B", None, "A", "C", "B", "B"],
             "notes": ["B"] * 8, "both": ["C"] * 8}]
    (r,) = dm.export_shared_read(rows)
    assert len(r.ants) == 8 and r.ants[3].answer is None and r.ants[0].reason == ""
    assert r.baselines == {"reader": "B", "alone8": "A", "notes8": "B", "both8": "C"}
    assert r.calls == {"reader": 1, "alone8": 8, "notes8": 9, "both8": 9}


# ------------------------------------------------------------------- running


class Scripted:
    """A fake decision backend: ``reply(request)`` gives an Answer or raises."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    async def decide(self, request):
        self.calls += 1
        answers = {qid: self.reply(request, q) for qid, q in request.questions.items()}
        return DecisionResponse(answers, request.model, request.request_hash, input_tokens=100, cost_usd=0.001)


def _yes(request, q):
    return Answer("noul", 0.8, {"yes": 0.8, "no": 0.2}) if q.type == "noul" else \
        Answer("choice", q.options[0], {o: (1.0 if o == q.options[0] else 0.0) for o in q.options}, 0.9)


def test_runner_caches_resumes_and_caps_spend(tmp_path):
    qs = [_exam("e1"), _forecast("f1")]
    swarm = {"e1": _row("e1", "exam", ["A"] * 5), "f1": _row("f1", "forecast", [0.5] * 5)}
    backend = Scripted(_yes)
    runner = dm.DMRunner(backend, tmp_path / "c.jsonl", max_usd=1.0, price=(0.042, 0.0))
    out = asyncio.run(runner.run(qs, swarm, "m", {"e1": OPTS}))
    assert backend.calls == 4 and runner.spent == pytest.approx(0.004)
    assert out[0]["dm1"]["conf"] == 0.9 and out[1]["read"] == {"p": 0.8}
    again = dm.DMRunner(backend, tmp_path / "c.jsonl", max_usd=1.0, price=(0.042, 0.0))
    assert asyncio.run(again.run(qs, swarm, "m", {"e1": OPTS})) == out and backend.calls == 4
    assert again.spent == pytest.approx(0.004)                   # earlier spend counts against the cap
    capped = dm.DMRunner(Scripted(_yes), tmp_path / "d.jsonl", max_usd=0.0, price=(0.042, 0.0))
    with pytest.raises(ft.SpendCap):
        asyncio.run(capped.run(qs, swarm, "m", {"e1": OPTS}))


def test_runner_asks_a_failed_call_again_and_stops_a_broken_run(tmp_path):
    def boom(request, q):
        raise DecisionError("HTTP 500: down", http_status=500)

    qs = [_forecast(f"f{i}") for i in range(4)]
    swarm = {q.id: _row(q.id, "forecast", [0.5] * 5) for q in qs}
    runner = dm.DMRunner(Scripted(boom), tmp_path / "c.jsonl", max_usd=1.0, max_fail_streak=3)
    with pytest.raises(dm.BrokenRun, match="3 calls in a row"):
        asyncio.run(runner.run(qs, swarm, "m", {}))
    rows = dm.read_jsonl(tmp_path / "c.jsonl")
    assert rows and all(r["error"] == "DecisionError" for r in rows)
    fixed = Scripted(_yes)
    asyncio.run(dm.DMRunner(fixed, tmp_path / "c.jsonl", max_usd=1.0).run(qs, swarm, "m", {}))
    assert fixed.calls == 8                                       # nothing failed was reused


def test_fake_dm_answers_in_the_contract():
    req = dm.dm_request("m", _exam(), "dm1", OPTS)
    resp = asyncio.run(dm.FakeDM().decide(req))
    a = resp.answers["answer"]
    assert a.value in OPTS and sum(a.probabilities.values()) == pytest.approx(1)


# -------------------------------------------------------------------- scoring


def _scored(dm_right: bool, n: int = 60, dm1_usable: bool = True):
    """60 exam + 60 forecast questions. The ants and single5 are right half the time; Jev, when ``dm_right``, is
    sure and right everywhere, so its vote breaks every tie the right way."""
    swarm, key, answers = [], {}, {}
    for i in range(n):
        qid = f"e{i}"
        key[qid] = "A"
        ants = ["B", "B", "A", "A", "C"]                         # the plain vote breaks the tie to B, wrongly
        swarm.append(_row(qid, "exam", ants, {"single5": "A" if i % 2 else "B", "swarm5": ft.vote(ants)},
                          group="lsat-lr" if i % 2 else "aqua-rat"))
        good = _choice({"A": 0.9, "B": 0.05, "C": 0.05}) if dm_right else _choice({"A": 0.05, "B": 0.9, "C": 0.05})
        answers[qid] = {"dm1": good if dm1_usable else None, "read": good}
        fid = f"f{i}"
        key[fid] = "yes" if i % 2 else "no"
        ants_f = [0.5] * 5
        swarm.append(_row(fid, "forecast", ants_f, {"single5": 0.5, "swarm5": 0.5}))
        p = (0.95 if i % 2 else 0.05) if dm_right else (0.05 if i % 2 else 0.95)
        answers[fid] = {"dm1": {"p": p}, "read": {"p": p}}
    return swarm, key, {"jev": answers}


def test_score_supported_when_the_mix_clearly_wins():
    swarm, key, dms = _scored(True)
    res = dm.score("fair-test", swarm, dms, key, crowd={"f1": 0.6})
    p = res["primary"]
    assert p["verdict"] == "supported" and p["exam"]["pass"] and p["forecast"]["pass"]
    ex = res["kinds"]["exam"]
    assert ex["means"]["jev:dm1"] == 1.0 and ex["means"]["jev:vote"] == 1.0 and ex["means"]["swarm5"] == 0.0
    assert ex["llm_calls"]["jev:dm1"] == 0 and ex["llm_calls"]["jev:vote"] == 5
    assert set(ex["by_group"]) == {"logic", "puzzles"}
    cheap = ex["dms"]["jev"]["cheap"]
    assert cheap["test_asked_share"] == 0 and cheap["llm_calls_per_question"] == 0   # sure and right: never asks
    assert "crowd_brier" in res["kinds"]["forecast"]


def test_score_not_supported_when_the_model_is_wrong_and_invalid_when_unusable():
    swarm, key, dms = _scored(False)
    assert dm.score("fair-test", swarm, dms, key)["primary"]["verdict"] == "not supported"
    swarm, key, dms = _scored(True, dm1_usable=False)
    assert dm.score("fair-test", swarm, dms, key)["primary"]["verdict"].startswith("invalid run")
    # another source gets no verdict
    sr = [dm.SwarmRow(r.id, r.kind, r.group, r.ants, {"alone5": ft.combine(r.kind, [a.answer for a in r.ants])},
                      {"alone5": 5}) for r in _scored(True)[0] if r.kind == "exam"]
    res = dm.score("shared-read", sr, _scored(True)[2], key)
    assert res["primary"] is None and "vote_vs_alone5" in res["kinds"]["exam"]["dms"]["jev"]


# --------------------------------------------------------------------- script


def test_jev_on_the_whole_fair_test_stays_well_under_five_cents():
    """Worst case: every ant gives a full 300-character reason, so ``read`` is as long as it can be."""
    cli = _load("dm_addon")
    qs = cli.questions()
    swarm = {q.id: _row(q.id, q.kind, ["A"] * 5 if q.kind == "exam" else [0.5] * 5) for q in qs}
    swarm = {k: dm.SwarmRow(r.id, r.kind, r.group, tuple(dm.Ant(a.answer, "w" * 300) for a in r.ants), {}, {})
             for k, r in swarm.items()}
    est = dm.estimate(qs, swarm, cli.options(), dm.DMS["jev"])
    assert est["calls"] == 2 * len(qs) == 682 and est["usd"] < 0.02


def test_cli_end_to_end_dry_run(tmp_path, capsys, monkeypatch):
    cli = _load("dm_addon")
    data = tmp_path / "dm"
    monkeypatch.setattr(cli, "DATA", data)
    qs = cli.questions()[:3] + [q for q in cli.questions() if q.kind == "forecast"][:3]
    rows, _ = _fair_run(tmp_path, qs)
    dm.write_jsonl(tmp_path / "rows.jsonl", rows)
    with pytest.raises(SystemExit, match="freeze"):
        cli.main(["plan", "--data", str(data)])
    assert cli.main(["freeze", "--data", str(data)]) == 0
    assert cli.main(["export", "--data", str(data), "--fair-rows", str(tmp_path / "rows.jsonl"),
                     "--fair-cache", str(tmp_path / "cache.jsonl")]) == 0
    assert cli.main(["plan", "--data", str(data), "--dm", "jev"]) == 0
    with pytest.raises(SystemExit, match="--live"):
        cli.main(["run", "--data", str(data)])
    assert cli.main(["smoke", "--data", str(data), "--dry-run"]) == 0
    for name in ("jev", "d1-3b"):
        assert cli.main(["run", "--data", str(data), "--dm", name, "--dry-run"]) == 0
    assert cli.main(["score", "--data", str(data), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "verdict (Jev vote vs single5 and swarm5)" in out and "d1-3b:vote" in out
    res = json.loads((data / "run" / "result_fair-test_dry.json").read_text())
    assert set(res["kinds"]["exam"]["dms"]) == {"jev", "d1-3b"}
    # any change to the frozen plan voids the run
    monkeypatch.setattr(dm, "PASS", {"exam_min_gain": 0.0, "forecast_min_brier_gain": 0.0})
    with pytest.raises(SystemExit, match="changed since it was frozen"):
        cli.main(["score", "--data", str(data), "--dry-run"])


def test_jev_key_is_read_from_the_env_file_only(tmp_path, monkeypatch):
    cli = _load("dm_addon")
    env = tmp_path / ".env"
    env.write_text("TYPESAFE_API_KEY=ts-secret\n")
    monkeypatch.setenv(cli.TYPESAFE_ENV_FILE_VAR, str(env))
    assert cli.read_typesafe_key() == "ts-secret"
    env.write_text("OTHER=1\n")
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        cli.read_typesafe_key()


def test_the_committed_plan_is_frozen():
    cli = _load("dm_addon")
    assert json.loads((DATA / "prereg.json").read_text()) == cli.fingerprint()
