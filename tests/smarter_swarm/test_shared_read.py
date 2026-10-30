"""Tests for the shared-read test (antelligence/experiments/shared_read.py) and its script. No network, $0."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import sys
from pathlib import Path

import httpx
import pytest

from antelligence.experiments import fair_test as ft
from antelligence.experiments import shared_read as sr
from antelligence.providers.base import ChatResponse, ProviderError, ProviderUnavailable
from antelligence.providers.openai_compat import OpenAICompatProvider

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs/research/fair-test-20261029"


def _script():
    spec = importlib.util.spec_from_file_location("shared_read_script", REPO / "scripts/shared_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shared_read_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def items():
    return sr.load_items(DATA / "exam_questions.jsonl")


# ------------------------------------------------------------------ questions and prompts

def test_loads_the_frozen_240_and_shows_exactly_what_the_fair_test_shows(items):
    assert len(items) == 240
    qs = {q.id: q for q in ft.load_questions(DATA / "exam_questions.jsonl", "exam")}
    for it in items:
        assert it.full == qs[it.id].text
    # the "alone" arm is the fair test's own lens prompt, so the two tests' ants are the same ants
    it = items[0]
    plain = sr.ModelSpec("lfm3:1.7b", "lfm")
    assert sr.ant_messages(it, "alone", 1, plain) == ft.lens_messages(qs[it.id], 1)


def test_refuses_a_questions_file_that_carries_answers(tmp_path):
    row = json.loads((DATA / "exam_questions.jsonl").read_text().splitlines()[0])
    row["label"] = "A"
    bad = tmp_path / "q.jsonl"
    bad.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        sr.load_items(bad)


def test_notes_arm_hides_the_passage_and_both_arm_keeps_it(items):
    it = next(i for i in items if i.passage)
    spec = sr.ModelSpec("gemma3:4b", "gemma")
    notes = "The order must satisfy six rules; watch the rule about O."
    only = sr.ant_messages(it, "notes", 0, spec, notes)[1]["content"]
    both = sr.ant_messages(it, "both", 0, spec, notes)[1]["content"]
    assert it.passage not in only and notes in only and it.stem in only and it.options[0] in only
    assert "you do not see the passage" in only
    assert it.passage in both and notes in both
    with pytest.raises(ValueError):
        sr.ant_messages(it, "notes", 0, spec, None)
    with pytest.raises(ValueError):
        sr.ant_messages(it, "nonsense", 0, spec, notes)


def test_a_model_suffix_and_extra_body_reach_the_request(items):
    q = sr.ant_messages(items[0], "alone", 0, sr.ModelSpec("qwen3:4b", "qwen", suffix="/no_think"))[1]["content"]
    g = sr.ant_messages(items[0], "alone", 0, sr.ModelSpec("gemma4:e2b", "gemma"))[1]["content"]
    assert q.endswith(" /no_think") and not g.endswith("/no_think")
    spec = sr.ModelSpec("granite4.2:3b", "granite", extra_body={"reasoning_effort": "none"})
    req = sr.make_request(spec, sr.ant_messages(items[0], "alone", 0, spec), items[0].id, "alone", 0, 100)
    assert req.payload()["reasoning_effort"] == "none" and req.payload()["model"] == "granite4.2:3b"


def test_reader_is_told_never_to_pick(items):
    text = sr.notes_messages(items[0])[1]["content"]
    assert "Do not say which option is correct" in text and items[0].full in text


def test_default_models_come_from_the_config_fit_36gb_and_cover_four_families():
    cfg = json.loads(sr.MODELS_FILE.read_text())
    assert sr.READER.tag == cfg["reader"]["tag"] == "qwen3.8:27b"
    assert [s.tag for s in sr.ANTS] == [a["tag"] for a in cfg["ants"]]
    assert len({s.family for s in sr.ANTS}) == 4 and len(sr.ant_roster()) == 8
    assert all(0 < s.gb <= 21 for s in (sr.READER, *sr.ANTS))  # one at a time, with headroom, on 36 GB
    assert sr.READER.extra_body == {"reasoning_effort": "none"}


@pytest.mark.parametrize("cfg,msg", [
    ({"reader": {"tag": "r", "family": "x"}, "ants": []}, "no ants"),
    ({"reader": {"tag": "r", "family": "x"}, "ants": [{"tag": "a", "family": "y"}, {"tag": "b", "family": "y"}]},
     "different family"),
    ({"reader": {"tag": "a", "family": "x"}, "ants": [{"tag": "a", "family": "y"}]}, "different models"),
    ({"reader": {"tag": "r", "family": "x", "colour": 1}, "ants": [{"tag": "a", "family": "y"}]}, "unknown"),
])
def test_bad_model_files_are_refused(tmp_path, cfg, msg):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=msg):
        sr.load_models(path)


# -------------------------------------------------------------------- parsing

@pytest.mark.parametrize("text,want", [
    ('{"reason": "x", "answer": "C"}', "C"),
    ('<think>\nif {a} then {b}\n</think>\n{"reason": "x", "answer": "b"}', "B"),
    ('<think>\n\n</think>\n\n```json\n{"reason": "because {this}", "answer": "D"}\n```', "D"),
    ('First guess {"answer": "Z"} then final {"reason": "r", "answer": "A"}', "A"),
    ('reasoning only, ends without tag</think>{"reason": "r", "answer": "E"}', "E"),
    ("I think it is C.", None),
    ("", None),
])
def test_parse_answer(text, want):
    assert sr.parse_answer(text)[0] == want


@pytest.mark.parametrize("notes,flag", [
    ("The answer is (C) because the rules force it.", True),
    ("Option B is correct.", True),
    ("(D) is the best fit.", True),
    ("Rule out A first, since O must be first or fifth.", True),
    ("The question asks for a valid order. Rules: T sits next to F or R; O is first or fifth. Trap: the "
     "'at least two between' rule cuts both ways.", False),
    ("A train travels 60 km in 1.5 hours; find its speed. Watch the units.", False),
    ("The best answer is a careful reading of rule 3; check whether the rule is right for each case.", False),
    ("Eliminate any order where O is not first or fifth.", False),
])
def test_leak_flag(notes, flag):
    assert sr.leaks(notes) is flag


def test_same_model_allows_only_ollamas_latest_suffix():
    assert sr.same_model("phi4-mini:latest", "phi4-mini")
    assert sr.same_model("QWEN3:4b", "qwen3:4b")
    assert not sr.same_model("qwen3:8b", "qwen3:4b")
    assert not sr.same_model("", "qwen3:4b")


# -------------------------------------------------------------------- scoring

def test_error_alikeness_is_one_for_clones_and_low_for_independent_ants():
    truths = ["A"] * 400
    rng = random.Random(1)
    clone = [["A" if rng.random() < 0.6 else "B"] * 4 for _ in truths]
    assert sr.error_alikeness(clone, truths) == pytest.approx(1.0)
    indep = [["A" if rng.random() < 0.6 else "B" for _ in range(4)] for _ in truths]
    assert abs(sr.error_alikeness(indep, truths)) < 0.1


def test_same_wrong_counts_matching_wrong_picks():
    assert sr.same_wrong([["B", "B", "A"]], ["A"]) == 1.0
    assert sr.same_wrong([["B", "C", "A", None]], ["A"]) == 0.0
    assert sr.same_wrong([["A", "A"]], ["A"]) == 0.0  # nobody wrong: nothing to compare


def _synthetic(n=200, *, p_alone=0.55, p_both=0.70, shared=False, leak_every=0, usable=1.0, seed=7):
    """Rows with 8 ants. ``shared`` makes the notes arms fail together (one coin per item)."""
    rng = random.Random(seed)
    rows, key = [], {}
    for i in range(n):
        qid = f"q{i}"
        key[qid] = "A"

        def ant(p, coin=None):
            if rng.random() > usable:
                return None
            return "A" if (coin if coin is not None else rng.random()) < p else rng.choice("BCDE")

        coin = rng.random() if shared else None
        both = [ant(p_both, coin) for _ in range(8)]
        rows.append({"id": qid, "group": "g1" if i % 2 else "g2", "reader_notes": "n",
                     "leak": bool(leak_every and i % leak_every == 0), "reader": "A" if rng.random() < 0.6 else "B",
                     "alone": [ant(p_alone) for _ in range(8)], "notes": list(both), "both": both})
    return rows, key


FAM = ["qwen", "qwen", "llama", "llama", "gemma", "gemma", "phi", "phi"]


def test_score_says_helps_when_notes_lift_independent_ants():
    rows, key = _synthetic()
    res = sr.score(rows, key, FAM, boot_reps=300)
    assert res["verdict"]["shared_read_helps"] == "yes"
    assert res["verdict"]["ants_think_alike"] == "no"
    assert res["all"]["arms"]["both"]["ant_mean"] > res["all"]["arms"]["alone"]["ant_mean"]
    assert set(res["all"]["lift_by_family"]) == {"qwen", "llama", "gemma", "phi"}
    assert set(res["by_group"]) == {"g1", "g2"}


def test_score_catches_ants_that_fail_together():
    rows, key = _synthetic(p_both=0.55, shared=True)
    res = sr.score(rows, key, FAM, boot_reps=300)
    assert res["verdict"]["ants_think_alike"] == "yes"
    assert res["all"]["arms"]["both"]["error_alikeness"] > 0.5


def test_score_refuses_a_broken_run_and_warns_on_leaky_notes():
    rows, key = _synthetic(usable=0.5)
    assert "invalid" in sr.score(rows, key, FAM, boot_reps=100)["verdict"]
    rows, key = _synthetic(leak_every=4)
    res = sr.score(rows, key, FAM, boot_reps=100)
    assert "warning" in res["verdict"] and res["no_leaks"]["n"] == 150


def test_plan_counts_calls():
    p = sr.plan(240)
    assert p == {"items": 240, "ants": 8, "reader_calls": 480, "ant_calls": 5760, "calls": 6240}


# -------------------------------------------------------------------- running

class Fake:
    """Answers right ("A") with notes, a random letter without; logs the model of every call in order."""

    def __init__(self, fail=None, served=None):
        self.log, self.fail, self.served = [], fail or (lambda req: None), served

    async def complete(self, req):
        self.log.append(req.model)
        err = self.fail(req)
        if err:
            raise err
        last = req.messages[-1]["content"]
        if '"answer"' in last:
            ans = "A" if "Notes from a reader" in last else random.Random(req.request_hash).choice("ABCDE")
            body = f'<think>\n\n</think>\n{{"reason": "r", "answer": "{ans}"}}'
        else:
            body = "The question asks for a valid order; key rules are listed; the trap is the spacing rule."
        return ChatResponse(content=body, model=self.served or req.model, request_hash=req.request_hash,
                            prompt_tokens=10, completion_tokens=10, finish_reason="stop", elapsed_s=0.5)


def _go(runner, its, ants=sr.ANTS):
    async def both():
        reading = await sr.run_reader(runner, its)
        return sr.rows(its, reading, await sr.run_ants(runner, its, reading, ants))
    return asyncio.run(both())


def test_runs_one_model_at_a_time_and_resumes_from_the_cache(items, tmp_path):
    its = items[:3]
    fake = Fake()
    rows = _go(sr.Runner(fake, tmp_path / "c.jsonl"), its)
    assert len(fake.log) == sr.plan(3)["calls"]
    order = [m for i, m in enumerate(fake.log) if i == 0 or fake.log[i - 1] != m]
    assert order == [sr.READER.tag] + [s.tag for s in sr.ANTS]  # each model's calls in one block
    assert all(r["both"] == ["A"] * 8 and r["notes"] == ["A"] * 8 for r in rows)
    again = Fake()
    assert _go(sr.Runner(again, tmp_path / "c.jsonl"), its) == rows and again.log == []


def test_a_failed_call_is_asked_again_and_a_streak_stops_the_run(items, tmp_path):
    flaky = Fake(fail=lambda req: ProviderError("bad reply") if req.model == "granite4.2:3b" else None)
    with pytest.raises(sr.BrokenRun, match="in a row"):
        _go(sr.Runner(flaky, tmp_path / "c.jsonl", concurrency=1), items[:3])
    fixed = Fake()
    rows = _go(sr.Runner(fixed, tmp_path / "c.jsonl"), items[:3])
    assert "granite4.2:3b" in fixed.log and sr.READER.tag not in fixed.log  # only the failed part is asked again
    assert all(a is not None for r in rows for a in r["both"])


def test_a_missing_model_stops_at_once_with_the_pull_command(items, tmp_path):
    gone = Fake(fail=lambda req: ProviderError('HTTP 404: {"error":"model \\"lfm3:1.7b\\" not found, try '
                                               'pulling it first"}', http_status=404)
                if req.model.startswith("lfm3") else None)
    with pytest.raises(sr.MissingModel, match="ollama pull lfm3:1.7b"):
        _go(sr.Runner(gone, tmp_path / "c.jsonl"), items[:2])


def test_a_different_served_model_is_refused(items, tmp_path):
    wrong = Fake(served="gemma4:e4b")
    with pytest.raises(sr.BrokenRun, match="served"):
        _go(sr.Runner(wrong, tmp_path / "c.jsonl", concurrency=1), items[:3])
    loose = Fake(served="/models/mlx/whatever")
    _go(sr.Runner(loose, tmp_path / "d.jsonl", check_names=False), items[:1])


def test_an_outage_stops_without_caching(items, tmp_path):
    down = Fake(fail=lambda req: ProviderUnavailable("transport error", status=None))
    runner = sr.Runner(down, tmp_path / "c.jsonl", backoff=(0,))
    with pytest.raises(ProviderUnavailable):
        _go(runner, items[:1])
    assert not (tmp_path / "c.jsonl").exists()


def test_end_to_end_over_http_like_ollama(items, tmp_path):
    """The real HTTP client against a stand-in that replies the way Ollama's OpenAI endpoint does."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        last = body["messages"][-1]["content"]
        content = ('<think>\n\n</think>\n\n{"reason": "r", "answer": "C"}' if '"answer"' in last
                   else "<think>\n\n</think>\n\nThe question asks which order is possible. Key rules: six conditions.")
        name = body["model"] + (":latest" if ":" not in body["model"] else "")
        return httpx.Response(200, json={"id": "chatcmpl-1", "object": "chat.completion", "model": name,
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": content}}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": 20}})

    prov = OpenAICompatProvider("http://localhost:11434/v1", require_model_match=False,
                                transport=httpx.MockTransport(handler))
    rows = _go(sr.Runner(prov, tmp_path / "c.jsonl"), items[:1])
    assert rows[0]["reader"] == "C" and rows[0]["both"] == ["C"] * 8 and not rows[0]["leak"]
    assert all(b["temperature"] == sr.TEMPERATURE and isinstance(b["seed"], int) for b in seen)
    assert len(seen) == sr.plan(1)["calls"]


def test_script_dry_run_end_to_end(tmp_path, capsys):
    mod = _script()
    out = str(tmp_path / "dry")
    assert mod.main(["plan", "--limit", "8"]) == 0
    assert mod.main(["bench", "--dry-run", "--out", out]) == 0
    assert mod.main(["run", "--dry-run", "--limit", "8", "--out", out]) == 0
    assert mod.main(["score", "--out", out]) == 0
    text = capsys.readouterr().out
    assert "Reader notes for" in text and "Verdict:" in text
    score = json.loads((tmp_path / "dry" / "score.json").read_text())
    assert score["n"] == 8 and set(score["by_group"]) == {"aqua-rat", "logiqa-en", "lsat-ar", "lsat-lr"}
    meta = json.loads((tmp_path / "dry" / "meta.json").read_text())
    assert meta["dry_run"] and len(meta["families"]) == 8


def test_pick_takes_tasks_in_turn(items):
    mod = _script()
    got = mod.pick(items, 8)
    assert [i.group for i in got] == ["aqua-rat", "logiqa-en", "lsat-ar", "lsat-lr"] * 2
    assert mod.pick(items, 0) == list(items)


def test_check_says_what_to_pull_and_which_models_think(monkeypatch, capsys):
    mod = _script()
    have = [sr.READER.tag] + [s.tag for s in sr.ANTS[:-1]]

    def fake_get(url, **kw):
        return httpx.Response(200, json={"data": [{"id": t} for t in have]}, request=httpx.Request("GET", url))

    def fake_post(url, **kw):
        assert url == "http://localhost:11434/api/show"
        caps = ["completion", "thinking"] if kw["json"]["model"] == sr.READER.tag else ["completion"]
        return httpx.Response(200, json={"capabilities": caps}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    assert mod.main(["check"]) == 1
    out = capsys.readouterr().out
    assert f"ollama pull {sr.ANTS[-1].tag}" in out
    assert 'a thinking model; we send {"reasoning_effort": "none"}' in out


def test_memory_guard_stops_before_loading_a_model_that_does_not_fit(items, tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod, "free_gb", lambda: 20.0)   # room for an ant, not for the 21 GB reader plus 6 GB
    posts = []
    monkeypatch.setattr(httpx, "post", lambda url, **kw: posts.append((url, kw["json"])) or
                        httpx.Response(200, json={}, request=httpx.Request("POST", url)))
    args = mod.argparse.Namespace(dry_run=False, headroom_gb=6.0, base_url="http://localhost:11434/v1")
    with pytest.raises(mod.MemoryTight, match="Only 20.0 GB free before loading qwen3.8:27b"):
        mod.guard(args, sr.READER)
    mod.guard(args, sr.ANTS[0])                           # an ant fits
    mod.unload(args, sr.ANTS[0])
    assert posts == [("http://localhost:11434/api/generate", {"model": sr.ANTS[0].tag, "keep_alive": 0})]
    mod.unload(mod.argparse.Namespace(dry_run=False, base_url="http://localhost:8080/v1"), sr.ANTS[0])
    assert len(posts) == 1                                # not Ollama: no unload call
    off = mod.argparse.Namespace(dry_run=False, headroom_gb=0.0, base_url="http://localhost:11434/v1")
    mod.guard(off, sr.READER)                             # 0 turns the guard off


def test_free_gb_reads_this_machine():
    mod = _script()
    got = mod.free_gb()
    assert got is None or got > 0


def test_unload_and_guard_run_around_every_model(items, tmp_path, monkeypatch, capsys):
    """The run checks memory before, and unloads after, the reader and each ant model, in that order."""
    mod = _script()
    events = []
    monkeypatch.setattr(mod, "guard", lambda a, spec: events.append(("guard", spec.tag)))
    monkeypatch.setattr(mod, "unload", lambda a, spec: events.append(("unload", spec.tag)))
    mod.main(["run", "--dry-run", "--limit", "2", "--out", str(tmp_path / "o")])
    want = []
    for spec in (sr.READER, *sr.ANTS):
        want += [("guard", spec.tag), ("unload", spec.tag)]
    assert events == want


def test_bench_flags_a_model_that_is_still_thinking(tmp_path, monkeypatch, capsys):
    mod = _script()

    class Thinker(mod.FakeModel):
        async def complete(self, req):
            r = await super().complete(req)
            slow = req.model == sr.ANTS[1].tag
            return ChatResponse(content=r.content, model=r.model, request_hash=r.request_hash,
                                prompt_tokens=r.prompt_tokens, completion_tokens=900 if slow else 60,
                                finish_reason="stop", elapsed_s=2.0)

    monkeypatch.setattr(mod, "FakeModel", Thinker)
    assert mod.main(["bench", "--dry-run", "--out", str(tmp_path / "b")]) == 3
    out = capsys.readouterr().out
    assert f"WARNING: {sr.ANTS[1].tag} wrote far more" in out and "Rough time for all 240" in out


def test_the_drop_in_prompt_names_the_real_commands_and_rules():
    text = (REPO / "docs/research/shared-read-20261030/CLAUDE_CODE_PROMPT.md").read_text()
    for cmd in ("check", "bench", "run --limit 60", "score", "freeze"):
        assert f"scripts/shared_read.py {cmd}" in text
    assert "claude/project-thread-zif8rm" in text and "Push nothing" in text and "Spend nothing" in text
    assert "memory_pressure" in text and "Never stop, kill or restart any process you didn't start" in text


def test_the_committed_plan_is_frozen_and_unchanged():
    """prereg.json fingerprints the questions, answers, harness, script, plan and pass lines. Any change after the
    freeze must be a deliberate re-freeze (and, after a real run, a new test)."""
    mod = _script()
    assert mod.PREREG.exists()
    assert mod.frozen_changes() == []


def test_score_warns_when_the_test_changed_since_the_freeze(tmp_path, monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod, "PREREG", tmp_path / "prereg.json")
    assert mod.main(["freeze"]) == 0 and mod.frozen_changes() == []
    monkeypatch.setattr(sr, "PASS", {**sr.PASS, "helps_min_gain": 0.0})
    assert mod.frozen_changes() == ["pass"]
    out = str(tmp_path / "dry")
    mod.main(["run", "--dry-run", "--limit", "4", "--out", out])
    mod.main(["score", "--out", out])
    assert "WARNING: the test changed" in capsys.readouterr().out
    assert json.loads((tmp_path / "dry" / "score.json").read_text())["frozen_plan_unchanged"] is False
