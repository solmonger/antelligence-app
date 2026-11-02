"""Tests for the harness v0: shapes, sandbox, toolbox, recipes, scorecard, learner, importer, retries, script. $0."""

from __future__ import annotations

import asyncio
import errno
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from antelligence.experiments import fresh_questions as fq
from antelligence.experiments import shared_read as sr
from antelligence.harness import learner as ln
from antelligence.harness import sandbox
from antelligence.harness import scorecard as sc
from antelligence.harness.core import Step, Task, bucket, features, margin, vote
from antelligence.harness.fakes import FakeProvider
from antelligence.harness.importer import import_sets
from antelligence.harness.planner import Planner, batch_order, sweep_batch, sweep_tools
from antelligence.harness.recipes import CodeThen, Ctx, ReaderCheck, Seq, Single, Vote, by_name, cheap_first, menu
from antelligence.harness.resilience import Patience, read_text, transient
from antelligence.harness.tools import (AntInfo, LiveBox, Missing, ReplayBox, code_problem, extract_code, load_steps,
                                        read_passes, save_steps)
from antelligence.providers.base import ProviderError, ProviderUnavailable

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "docs/research/fresh-20261031/models.json"
CFG = sr.load_config(MODELS)


def _script():
    spec = importlib.util.spec_from_file_location("_t_harness_cli", REPO / "scripts/harness.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def dev(seed: int = 3, per_task: int = 2):
    qs, ans, _ = fq.make_set(seed, per_task)
    return [Task.from_raw(q) for q in qs], {a["id"]: a["label"] for a in ans}


def fake_box(tasks, key, tmp_path: Path, code: bool = True, name: str = "cache.jsonl") -> LiveBox:
    skill = {CFG.reader.tag: 0.8, **{s.tag: 0.5 for s in CFG.ants}}
    prov = FakeProvider(tasks, key, skill, {CFG.reader.tag: CFG.reader.gb, **{s.tag: s.gb for s in CFG.ants}})
    runner = sr.Runner(prov, tmp_path / name, concurrency=4, check_names=False, max_fail_streak=10 ** 6)
    return LiveBox(runner, CFG.reader, CFG.ants, parser=CFG.parser, code=code)


# ----------------------------------------------------------------------- core

def test_task_refuses_answers_and_renders_like_the_fair_test():
    with pytest.raises(ValueError):
        Task.from_raw({"id": "x", "question": "q", "options": ["(A)1"], "label": "A"})
    tasks, _ = dev()
    t = tasks[0]
    item = sr.Item(t.id, t.kind, t.passage, t.stem, t.options)
    assert t.full == item.full
    assert t.letters == tuple("ABCDE")
    assert t.option_text("A") == t.options[0][3:]


def test_features_tell_the_dev_kinds_apart_without_reading_the_tag():
    tasks, _ = dev(per_task=6)
    by_kind = {}
    for t in tasks:
        by_kind.setdefault(t.kind, set()).add(bucket(t))
    assert by_kind["arith"] == {"num"}
    assert all("categorical" in b for b in by_kind["syllogism"])
    assert all("order" in b for b in by_kind["order"])
    assert not any("categorical" in b for b in by_kind["order"] | by_kind["assign"])
    assert features(tasks[0])["ask"] in ("could", "must", "value", "cannot", "other")


def test_vote_and_margin():
    assert vote(["B", "A", "A", None]) == "A"
    assert vote(["B", "A"]) == "B"           # tie: first given
    assert vote([None]) is None
    assert margin(["A", "A", "B"]) == 1
    assert margin(["A", "A"]) == 2
    assert margin([None, None]) == 0


def test_step_round_trips_and_weights():
    s = Step("code", "B", 10, 5, 1.5, passes=("B",), weight=10, payload={"x": 1})
    assert Step.from_json(json.loads(json.dumps(s.to_json()))) == s
    assert s.tokens == 15 and s.weighted_tokens == 150
    assert Step("x", None, 3, 1).weighted_tokens == 4


# -------------------------------------------------------------------- sandbox

def test_sandbox_runs_a_brute_force_program():
    code = ("import itertools, json\n"
            "ok = [p for p in itertools.permutations('FGH') if p.index('F') < p.index('G')]\n"
            "print(json.dumps({'A': len(ok) == 3, 'B': len(ok) == 6}))\n")
    r = sandbox.run(code)
    assert r.ok, r.error
    assert read_passes(r.stdout, "AB") == ("A",)


@pytest.mark.parametrize("code,why", [
    ("import os\nprint(1)", "import"),
    ("from subprocess import run", "import"),
    ("open('x')", "open"),
    ("print(().__class__)", "__class__"),
    ("eval('1')", "eval"),
    ("print(getattr(1, 'real'))", "getattr"),
    ("def f(:\n  pass", "syntax"),
])
def test_sandbox_refuses_unsafe_or_broken_programs(code, why):
    r = sandbox.run(code)
    assert not r.ok and r.error.startswith("refused") and why in r.error


def test_sandbox_allows_the_main_guard_and_reports_crashes_and_timeouts():
    assert sandbox.run("if __name__ == '__main__':\n    print('{\"A\": true}')").ok
    crash = sandbox.run("print(1/0)")
    assert not crash.ok and "ZeroDivisionError" in crash.error
    slow = sandbox.run("while True:\n    pass", timeout_s=1)
    assert not slow.ok and "longer" in slow.error


def test_sandbox_child_gets_no_environment(monkeypatch):
    monkeypatch.setenv("SECRET_FOR_TEST", "x")
    r = sandbox.run("import json\nprint(json.dumps({'A': True}))")
    assert r.ok


# ---------------------------------------------------------------- code helpers

def test_extract_code_and_read_passes():
    assert extract_code("Here:\n```python\nprint(1)\n```\nthanks") == "print(1)"
    assert extract_code("<think>hmm</think>```py\nx=1\nprint(x)\n```") == "x=1\nprint(x)"
    assert extract_code("no code here") == ""
    assert read_passes('noise\n{"A": false, "B": true, "(C)": true}', "ABCDE") == ("B", "C")
    assert read_passes('{"answer": "B"}', "ABCDE") is None
    assert read_passes("", "AB") is None


def test_code_problem_names_what_went_wrong():
    ok = sandbox.Result(True, "", "", 0.1)
    assert code_problem(ok, ("B",), "print()") == ""
    assert "exactly one" in code_problem(ok, ("A", "B"), "print()")
    assert "none" in code_problem(ok, (), "print()")
    assert "JSON" in code_problem(ok, None, "print()")
    assert "python block" in code_problem(ok, None, "")
    assert "crashed" in code_problem(sandbox.Result(False, "", "it crashed: x", 0.1), None, "print()")


# ---------------------------------------------------------------- live toolbox

def test_live_box_bills_every_step_and_counts_shared_notes_once(tmp_path):
    tasks, key = dev()
    box = fake_box(tasks, key, tmp_path)
    t = tasks[0]

    async def go():
        ctx = Ctx(box, t)
        await ctx.call("ant0:notes")
        await ctx.call("ant1:both")
        return ctx

    ctx = asyncio.run(go())
    assert ctx.order == ["notes", "ant0:notes", "ant1:both"]     # notes asked once, first
    assert all(s.tokens > 0 and s.latency_s > 0 for s in ctx.steps)
    assert ctx.done["notes"].payload["notes"] and ctx.done["notes"].answer is None
    assert ctx.done["ant0:notes"].weight == box.roster[0].weight


def test_live_box_code_checker_retries_until_one_option_passes(tmp_path):
    tasks, key = dev(per_task=4)
    box = fake_box(tasks, key, tmp_path)
    steps = [asyncio.run(box.run("code", t, {})) for t in tasks]
    assert all(s.calls >= 1 for s in steps)
    settled = [s for s in steps if s.answer is not None]
    assert settled and all(s.passes == (s.answer,) and s.sure == 1.0 for s in settled)
    retried = [s for s in steps if s.calls > 1]
    assert retried, "the fake gets some wrong, so some must be sent back"
    unsettled = [s for s in steps if s.answer is None]
    assert all(s.calls == 3 and s.note.startswith("not settled") for s in unsettled)


def test_live_box_has_and_missing(tmp_path):
    tasks, key = dev()
    box = fake_box(tasks, key, tmp_path, code=False)
    assert box.has("reader") and box.has("ant7:both") and not box.has("ant8:alone")
    assert not box.has("code") and not box.has("dm:jev:dm1")
    with pytest.raises(Missing):
        asyncio.run(box.run("code", tasks[0], {}))


def test_live_box_reuses_the_shared_read_cache(tmp_path):
    tasks, key = dev()
    box = fake_box(tasks, key, tmp_path)
    asyncio.run(box.run("reader", tasks[0], {}))
    again = fake_box(tasks, key, tmp_path)
    asyncio.run(again.run("reader", tasks[0], {}))
    assert again.runner.calls == 0                  # read back from the cache, no new call


# ------------------------------------------------------------------- recipes

ROSTER = tuple(AntInfo(i, f"m{i}", f"f{i}", 0, float(i + 1)) for i in range(4))


def replay(answers, reader="R", code=None, passes=None):
    steps = {"t": {f"ant{i}:alone": Step(f"ant{i}:alone", a, 10, 1, 1.0, weight=1) for i, a in enumerate(answers)}}
    steps["t"]["reader"] = Step("reader", reader, 100, 10, 10.0, weight=10)
    if code is not None or passes is not None:
        steps["t"]["code"] = Step("code", code, 200, 50, 20.0, calls=2, weight=10, passes=passes)
    return ReplayBox(steps, ROSTER)


TASK = Task("t", "p", "q", ("(A)a", "(B)b", "(C)c", "(D)d", "(E)e"))


def run(recipe, box):
    return asyncio.run(sc.run_one(recipe, box, TASK, {"t": "A"}))


def test_seq_stops_when_two_agree_and_escalates_when_split():
    order = (0, 1, 2, 3)
    agree = run(Seq("s", (), "alone", 2, None, 4, order), replay(["A", "A", "B", "B"]))
    assert agree.answer == "A" and agree.tools == ["ant0:alone", "ant1:alone"]
    split = run(Seq("s", (), "alone", 2, "reader", 4, order), replay(["A", "B", "C", "D"]))
    assert split.answer == "R" and split.tools[-1] == "reader"
    # with stop 3 after A, B the best reachable margin is 2: it hands up early instead of asking more ants
    early = run(Seq("s", (), "alone", 3, "reader", 4, order), replay(["A", "B", "A", "A"]))
    assert early.tools == ["ant0:alone", "ant1:alone", "reader"]


def test_seq_without_escalation_stops_when_the_leader_cant_change():
    out = run(Seq("s", (), "alone", 4, None, 4, (0, 1, 2, 3)), replay(["A", "A", "A", "B"]))
    assert out.answer == "A" and len(out.tools) == 3


def test_vote_code_then_and_reader_check():
    assert run(Vote("v", (), "alone", 4), replay(["B", "A", "A", "C"])).answer == "A"
    assert run(CodeThen("c", (), "reader"), replay([None] * 4, code="C", passes=("C",))).answer == "C"
    fall = run(CodeThen("c", (), "reader"), replay([None] * 4, code=None, passes=("A", "B")))
    assert fall.answer == "R" and fall.tools == ["code", "reader"]
    keep = run(ReaderCheck("rc"), replay([None] * 4, reader="B", code=None, passes=("A", "B")))
    assert keep.answer == "B"
    fix = run(ReaderCheck("rc"), replay([None] * 4, reader="B", code="A", passes=("A",)))
    assert fix.answer == "A"


def test_bill_totals_every_step_once():
    o = run(Seq("s", (), "alone", 2, "reader", 4, (0, 1, 2, 3)), replay(["A", "B", "C", "D"]))
    # after A, B, C the 4th ant can't make any answer 2 clear, so it hands up without asking it
    assert o.tools == ["ant0:alone", "ant1:alone", "ant2:alone", "reader"]
    assert o.tokens == 3 * 11 + 110 and o.calls == 4 and o.seconds == 13.0
    assert o.weighted_tokens == 3 * 11 + 1100 and o.right == 0.0


def test_menu_is_unique_and_cheap_first():
    names = [r.name for r in menu(ROSTER, ["dm:jev:dm1"])]
    assert len(names) == len(set(names)) and "reader" in names and "seq:alone:m2>reader" in names
    assert cheap_first(tuple(reversed(ROSTER))) == [0, 1, 2, 3]


def test_replay_box_raises_missing_for_unsaved_work():
    with pytest.raises(Missing):
        run(Single("code", ("code",), "code"), replay(["A"] * 4))


# ----------------------------------------------------------------- scorecard

def test_verdict_needs_accuracy_and_both_savings():
    assert sc.verdict(0.01, 0.05, 0.5, 0.5) == "beats it on all three"
    assert sc.verdict(0.01, 0.05, 1.5, 0.5).startswith("more accurate")
    assert sc.verdict(-0.05, -0.01, 0.5, 0.5) == "less accurate"
    assert sc.verdict(-0.02, 0.03, 0.5, 0.9) == "ties it on accuracy, and is cheaper on both"
    assert sc.verdict(-0.02, 0.03, 0.5, 1.1) == "ties it on accuracy, not cheaper on both"


def test_pareto_drops_dominated_recipes():
    table = {"a": {"accuracy": 0.8, "tokens": 100, "seconds": 10}, "b": {"accuracy": 0.7, "tokens": 200, "seconds": 20},
             "c": {"accuracy": 0.6, "tokens": 50, "seconds": 5}}
    assert sc.pareto(table) == ["a", "c"]


def test_log_round_trips(tmp_path):
    log = sc.Log(tmp_path / "o.jsonl")
    o = run(Single("reader", ("reader",), "reader"), replay(["A"] * 4, reader="A"))
    log.add(o)
    assert log.read() == [o]


# ------------------------------------------------------------------- learner

def synthetic(n_per: int = 30, seed: int = 1):
    """Two kinds of task. 'num' tasks: recipe 'x' is always right and cheap. 'text' tasks: 'y' is. The reader is
    right 70% of the time on both."""
    rng = random.Random(seed)
    tasks, table = [], {"reader": {}, "x": {}, "y": {}}
    for i in range(2 * n_per):
        num = i % 2 == 0
        t = Task(f"s{1 + i % 3}:t{i}", "", "What is it?" if num else "Which must be true?",
                 tuple(f"({l}){k}" for l, k in zip("ABCDE", ("1", "2", "3", "4", "5") if num else "vwxyz")),
                 "n" if num else "t")
        tasks.append(t)

        def o(name, right, tok, sec):
            return sc.Outcome(t.id, t.kind, bucket(t), name, None, float(right), tok, tok, sec, 0.0, 1)

        table["reader"][t.id] = o("reader", rng.random() < 0.7, 1000, 10.0)
        table["x"][t.id] = o("x", num or rng.random() < 0.3, 200, 2.0)
        table["y"][t.id] = o("y", (not num) or rng.random() < 0.3, 300, 3.0)
    return tasks, table


def test_learner_picks_the_winner_per_kind():
    tasks, table = synthetic()
    pol = ln.learn(table, tasks)
    picks = {bucket(t): pol.pick(t) for t in tasks}
    assert picks["num"] == "x" and picks["text:must"] == "y"
    assert pol.ref["tokens"] == 1000
    assert ln.Policy.from_json(json.loads(json.dumps(pol.to_json()))).pick(tasks[1]) == pol.pick(tasks[1])


def test_learner_keeps_the_default_for_tiny_buckets():
    tasks, table = synthetic(n_per=3)
    pol = ln.learn(table, tasks, min_n=8)
    assert pol.buckets == {}


def test_k_fold_and_held_out_route_every_task_unseen():
    tasks, table = synthetic()
    routed = ln.k_fold(table, tasks, k=5)
    assert sorted(o.task for o in routed) == sorted(t.id for t in tasks)
    assert all(o.recipe == "harness" for o in routed)
    acc = sum(o.right for o in routed) / len(routed)
    assert acc > 0.9
    pol, out = ln.held_out(table, [t for t in tasks if not t.id.startswith("s3")],
                           [t for t in tasks if t.id.startswith("s3")])
    assert out and all(o.task.startswith("s3") for o in out)


def test_common_compares_recipes_on_the_same_tasks():
    tasks, table = synthetic()
    del table["y"][tasks[0].id]
    names, keep = ln.common(table, tasks)
    assert set(names) == {"reader", "x", "y"} and tasks[0].id not in {t.id for t in keep}
    table["z"] = {tasks[1].id: table["x"][tasks[1].id]}
    names, _ = ln.common(table, tasks)
    assert "z" not in names


# --------------------------------------------------------------- the planner

def test_sweep_batch_groups_tools_by_model(tmp_path):
    tasks, key = dev()
    box = fake_box(tasks, key, tmp_path)
    groups = batch_order(box)
    assert groups[0] == ["reader", "notes", "code"]
    assert len(groups) == 1 + len({a.tag for a in box.roster})
    steps = asyncio.run(sweep_batch(box, tasks))
    assert all(set(per) == set(sweep_tools(box)) for per in steps.values())


def test_planner_answers_explores_and_logs(tmp_path):
    tasks, key = dev(per_task=3)
    box = fake_box(tasks, key, tmp_path)
    pol = ln.Policy("seq:alone:m2>reader", {"num": "code>reader"})
    log = sc.Log(tmp_path / "out.jsonl")
    planner = Planner(pol, by_name(menu(box.roster)), box, explore=0.5, log=log)
    outs = asyncio.run(planner.answer_all(tasks, key))
    assert len(outs) == len(tasks) == len(log.read())
    assert planner.swept and all(o.recipe.endswith("+sweep") for o in outs if o.task in planner.swept)
    assert {o.recipe.split("+")[0] for o in outs if bucket(next(t for t in tasks if t.id == o.task)) == "num"} \
        == {"harness:code>reader"}
    swept = next(iter(planner.swept.values()))
    assert set(swept) == set(sweep_tools(box))


# ----------------------------------------------------------------- importer

def _saved_set(tmp_path: Path, seed: int, name: str):
    """A set saved exactly the way the shared-read test saves one, from a fake sweep."""
    s = _script()
    run = tmp_path / name / "run"
    tasks, key = s.make_dev(tmp_path / f"dev-{name}", seed, 2)
    prov = s.fake_provider(tasks, key, CFG)
    box = LiveBox(sr.Runner(prov, run / "cache.jsonl", concurrency=4, check_names=False, max_fail_streak=10 ** 6),
                  CFG.reader, CFG.ants, parser=CFG.parser, code=False)
    run.mkdir(parents=True, exist_ok=True)
    steps = asyncio.run(sweep_batch(box, tasks))
    s.write_jsonl(run / "questions.jsonl", [s.task_row(t) for t in tasks])
    s.write_jsonl(run / "answers.jsonl", [{"id": i, "label": v} for i, v in key.items()])
    s.write_jsonl(run / "rows.jsonl", [{
        "id": t.id, "group": t.kind, "reader": steps[t.id]["reader"].answer,
        "reader_notes": steps[t.id]["notes"].payload["notes"], "leak": False,
        **{arm: [steps[t.id][f"ant{i}:{arm}"].answer for i in range(8)] for arm in ("alone", "notes", "both")}}
        for t in tasks])
    return run, tasks, key, steps


def test_import_matches_every_saved_call_exactly(tmp_path):
    run, tasks, key, steps = _saved_set(tmp_path, 5, "set1")
    itasks, ikey, isteps, roster, reports = import_sets({"set1": run}, CFG)
    assert reports[0]["priced"] == 1.0 and reports[0]["load_outliers"] == 0
    assert ikey == {f"set1:{k}": v for k, v in key.items()}
    for t in tasks:
        for tool, s in steps[t.id].items():
            got = isteps[f"set1:{t.id}"][tool]
            assert (got.answer, got.tokens_in, got.tokens_out) == (s.answer, s.tokens_in, s.tokens_out), tool


def test_import_fills_unmatched_calls_and_load_outliers(tmp_path):
    run, tasks, _, _ = _saved_set(tmp_path, 6, "set1")
    rows = [json.loads(l) for l in (run / "cache.jsonl").read_text().splitlines()]
    rows[0]["elapsed_s"] = 999.0                    # one call that was really a model load
    (run / "cache.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[2:] + rows[:1]))  # two lost
    _, _, isteps, _, reports = import_sets({"set1": run}, CFG)
    assert reports[0]["priced"] < 1.0 and reports[0]["load_outliers"] == 1
    notes = [s.note for per in isteps.values() for s in per.values()]
    assert "unpriced" in notes and any(n.startswith("load-time") for n in notes)


def test_import_refuses_a_roster_the_model_file_doesnt_know(tmp_path):
    run, *_ = _saved_set(tmp_path, 7, "set1")
    (run / "meta.json").write_text(json.dumps({"ants": [["not/a-model", "x", 0]]}))
    with pytest.raises(ValueError, match="model file"):
        import_sets({"set1": run}, CFG)


def test_steps_file_round_trips(tmp_path):
    steps = {"t": {"reader": Step("reader", "A", 1, 2, 0.5)}}
    save_steps(tmp_path / "s.json", steps, ROSTER, {"a": 1})
    got, roster, meta = load_steps(tmp_path / "s.json")
    assert got == steps and roster == ROSTER and meta == {"a": 1}


# --------------------------------------------------------------- resilience

def test_patience_waits_out_a_stalled_server_then_succeeds():
    waits = []

    async def sleep(s):
        waits.append(s)

    tries = {"n": 0}

    async def flaky():
        tries["n"] += 1
        if tries["n"] < 3:
            raise ProviderUnavailable("server not answering")
        return "ok"

    p = Patience(waits=(1, 2, 3), sleep=sleep, log=lambda m: None)
    assert asyncio.run(p.run("step", flaky)) == "ok"
    assert waits == [1, 2] and len(p.events) == 2


def test_patience_gives_up_and_never_retries_real_errors():
    async def sleep(s):
        pass

    async def bad():
        raise ProviderError("HTTP 400: bad request")

    async def always_down():
        raise OSError(errno.EIO, "Input/output error")

    p = Patience(waits=(1,), sleep=sleep, log=lambda m: None)
    with pytest.raises(ProviderError):
        asyncio.run(p.run("x", bad))
    assert p.events == []
    with pytest.raises(OSError):
        asyncio.run(p.run("x", always_down))
    assert transient(OSError(errno.EINTR, "x")) and not transient(OSError(errno.ENOENT, "x"))


def test_read_text_retries_a_stalled_disk(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    real = Path.read_text
    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EINTR, "Interrupted system call")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert read_text(f, waits=(0,), sleep=lambda s: None) == "hello"


# ------------------------------------------------------------------- script

def test_selftest_runs_the_whole_loop(tmp_path):
    s = _script()
    assert s.main(["selftest", "--out", str(tmp_path / "st")]) == 0


def test_mac_dry_run_imports_learns_and_smokes(tmp_path, capsys):
    night = tmp_path / "night"
    for i, name in enumerate(("set1", "set2", "set3")):
        _saved_set(night, 20 + i, name)
    s = _script()
    out = tmp_path / "mac"
    assert s.main(["mac", "--night", str(night), "--out", str(out), "--smoke", "1", "--dry-run", "--seed", "9",
                   "--code"]) == 0
    text = (out / "RESULT.txt").read_text()
    assert text.startswith("HARNESS RESULT")
    assert "learned on set1+set2" in text and "smoke sweep" in text and "planner live" in text
    assert "DEV NUMBERS ONLY" in text
    assert (out / "saved" / "policy.json").exists() and (out / "smoke" / "steps.json").exists()

    # diagnose explains the same run from its saved files, replaying the code checker with no model calls
    cache = out / "smoke" / "cache.jsonl"
    stale = {"request_hash": "0" * 64, "model": CFG.reader.tag, "content": "", "error": "ProviderTruncated",
             "reason": "an earlier run's call with another prompt"}
    cache.write_text(cache.read_text() + json.dumps(stale) + "\n")
    saved = cache.read_text()
    assert s.main(["diagnose", "--run", str(out)]) == 0
    diag = (out / "DIAGNOSE.txt").read_text()
    assert diag.startswith("HARNESS DIAGNOSE")
    assert "failed replies by model" in diag and "blank answers by tool" in diag
    assert "code checker, try by try" in diag and "calls missing from the log" not in diag
    head = diag.splitlines()[1]
    assert "failed: 0" in head and "the rest are from earlier runs" in head   # the stale row isn't this run's
    import re

    same, total = map(int, re.search(r"replay matched (\d+)/(\d+)", head).groups())
    assert same == total > 0                                                   # the replay is the run, exactly
    assert "settled on" in diag or "marked" in diag
    assert (out / "smoke" / "cache.jsonl").read_text() == saved   # the run's own files are left alone


def test_learn_command_writes_a_policy(tmp_path):
    night = tmp_path / "night"
    run, *_ = _saved_set(night, 30, "set1")
    s = _script()
    assert s.main(["import", "--set", f"set1={run}", "--out", str(tmp_path / "data")]) == 0
    assert s.main(["learn", "--data", str(tmp_path / "data"), "--folds", "3", "--out", str(tmp_path / "l")]) == 0
    pol = json.loads((tmp_path / "l" / "policy.json").read_text())
    assert pol["default"] and "ref" in pol


def test_trim_loads_counts_model_load_waits_at_the_median():
    from antelligence.harness.tools import trim_loads

    def st(sec):
        return Step(tool="reader", answer="A", tokens_in=10, tokens_out=1, latency_s=sec)

    steps = {f"t{i}": {"reader": st(2.0)} for i in range(5)}
    steps["t5"] = {"reader": st(95.0)}   # a model load
    steps["t6"] = {"reader": st(25.0)}   # slow, but under the 30s floor
    out, n = trim_loads(steps)
    assert n == 1
    assert out["t5"]["reader"].latency_s == 2.0
    assert "load-time outlier" in out["t5"]["reader"].note
    assert out["t6"]["reader"].latency_s == 25.0
    assert steps["t5"]["reader"].latency_s == 95.0   # the input is left alone


# ------------------------------------------------------- blank replies (Mac diagnose, Nov 2)

def _opt_task():
    return Task("q1", "", "How many hours?", ("(A)12 hours", "(B)15 hours", "(C)$1,200", "(D)None of these",
                                               "(E)3/4"), "arith")


def test_read_answer_maps_an_option_text_answer_to_its_letter():
    from antelligence.harness.tools import read_answer

    t = _opt_task()
    assert read_answer('{"reason": "x", "answer": "B"}', t) == "B"
    assert read_answer('{"reason": "rates", "answer": "15"}', t) == "B"          # LFM2.5's habit
    assert read_answer('{"reason": "x", "answer": "1200"}', t) == "C"
    assert read_answer('{"reason": "x", "answer": "0.75"}', t) == "E"
    assert read_answer('{"reason": "x", "answer": "(D)"}', t) == "D"
    assert read_answer('{"reason": "x", "answer": "None of these"}', t) == "D"   # it is an option here
    assert read_answer('{"reason": "x", "answer": "None"}', t) is None
    assert read_answer('{"reason": "x", "answer": "None of the above"}', t) is None
    assert read_answer('{"reason": "x", "answer": "7"}', t) is None               # no option says 7
    two = Task("q2", "", "?", ("(A)15 hours", "(B)15 minutes"), "arith")
    assert read_answer('{"answer": "15"}', two) is None                          # two options share it: no guess
    assert read_answer("", t) is None


class _Scripted:
    """A provider that answers from a function of the request and keeps every request it saw."""

    def __init__(self, fn):
        self.fn, self.seen = fn, []

    async def complete(self, req):
        from antelligence.providers.base import ChatResponse

        self.seen.append(req)
        content, finish = self.fn(req, len(self.seen))
        return ChatResponse(content=content, model=req.model, request_hash=req.request_hash, prompt_tokens=100,
                            completion_tokens=10, finish_reason=finish, elapsed_s=1.0)

    def describe(self):
        return {"provider": "scripted"}


def _scripted_box(tmp_path, fn, **kw):
    prov = _Scripted(fn)
    runner = sr.Runner(prov, tmp_path / "c.jsonl", check_names=False, max_fail_streak=10 ** 6)
    return LiveBox(runner, CFG.reader, CFG.ants, parser=CFG.parser, **kw), prov


def test_a_reply_with_no_option_gets_one_short_re_ask_and_both_are_billed(tmp_path):
    t = _opt_task()

    def fn(req, n):
        if req.max_tokens == 30:                                 # the re-ask
            return '{"answer": "B"}', "stop"
        return '{"reason": "none fit", "answer": "None"}', "stop"

    box, prov = _scripted_box(tmp_path, fn)
    s = asyncio.run(box.run("ant0:alone", t, {}))
    assert s.answer == "B" and s.calls == 2 and s.tokens_in == 200 and s.tokens_out == 20
    assert "fixed by a short re-ask" in s.note
    ask = prov.seen[-1].messages
    assert ask[-2]["role"] == "assistant" and "none fit" in ask[-2]["content"]
    assert "exactly one of the options" in ask[-1]["content"]

    (tmp_path / "off").mkdir()
    off, prov2 = _scripted_box(tmp_path / "off", fn, repair=False)
    s = asyncio.run(off.run("ant0:alone", t, {}))
    assert s.answer is None and s.calls == 1 and len(prov2.seen) == 1


def test_a_cut_off_reply_is_billed_and_re_asked(tmp_path):
    t = _opt_task()

    def fn(req, n):
        if req.max_tokens == 30:
            return '{"answer": "A"}', "stop"
        return '{"reason": "first the rate is 12 per hour, then', "length"

    box, _ = _scripted_box(tmp_path, fn)
    s = asyncio.run(box.run("reader", t, {}))
    assert s.answer == "A" and s.calls == 2 and s.note.startswith("cut off")


def test_the_code_checker_is_told_to_write_less_when_cut_off(tmp_path):
    t = _opt_task()
    good = '```python\nimport json\nprint(json.dumps({"A": False, "B": True, "C": False, "D": False, "E": False}))\n```'

    def fn(req, n):
        return ("```python\nimport itertools\nfor x in", "length") if n == 1 else (good, "stop")

    box, prov = _scripted_box(tmp_path, fn)
    s = asyncio.run(box.run("code", t, {}))
    assert s.answer == "B" and s.calls == 2 and s.tokens_in == 200
    assert box.code_trace[t.id][0].startswith("cut off")
    retry = prov.seen[1].messages
    assert "cut off at the length limit" in retry[-1]["content"] and "under 60 lines" in retry[-1]["content"]


def test_seq_votes_inside_what_the_code_checker_narrowed_to():
    roster = tuple(AntInfo(i, f"m{i}", "f", 0, 1.0) for i in range(3))
    t = _opt_task()
    st = lambda tool, a, **kw: Step(tool, a, 1, 1, 1.0, **kw)  # noqa: E731
    box = ReplayBox({t.id: {"ant0:alone": st("ant0:alone", "C"), "ant1:alone": st("ant1:alone", "A"),
                            "ant2:alone": st("ant2:alone", "C"),
                            "code": st("code", None, passes=("A", "B"))}}, roster)
    seq = Seq("s", ("ant0:alone", "ant1:alone", "ant2:alone", "code"), "alone", 3, "code", 3, (0, 1, 2))
    assert asyncio.run(seq.run(Ctx(box, t))) == "A"      # C has more votes, but the code ruled C out


def test_import_reads_a_blank_answer_from_its_saved_reply(tmp_path):
    run, tasks, key, steps = _saved_set(tmp_path, 7, "set1")
    t = tasks[0]
    item = sr.Item(t.id, t.kind, t.passage, t.stem, t.options)
    h = sr.make_request(CFG.reader, sr.reader_answer_messages(item, CFG.reader), t.id, "reader", 0,
                        __import__("antelligence.experiments.fair_test", fromlist=["x"]).MAX_TOKENS).request_hash
    rows = [json.loads(line) for line in (run / "cache.jsonl").read_text().splitlines()]
    for r in rows:
        if r["request_hash"] == h:
            r.update(content=json.dumps({"reason": "x", "answer": t.option_text(key[t.id])}), error="Unusable")
    (run / "cache.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    saved = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
    saved[0]["reader"] = None
    (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in saved))
    _, _, isteps, _, reports = import_sets({"set1": run}, CFG)
    assert isteps[f"set1:{t.id}"]["reader"].answer == key[t.id]
    assert reports[0]["reparsed"] == 1


def test_the_code_checker_gives_up_after_two_cut_off_programs(tmp_path):
    t = _opt_task()
    box, prov = _scripted_box(tmp_path, lambda req, n: ("```python\nfor a in range(9):\n    for b in", "length"))
    s = asyncio.run(box.run("code", t, {}))
    assert s.answer is None and s.calls == 2 and len(prov.seen) == 2
    assert all(x.startswith("cut off") for x in box.code_trace[t.id]) and "ends:" in box.code_trace[t.id][0]


def test_reask_counts_ignore_notes_added_after_the_re_ask():
    s = _script()
    st = lambda note: Step("ant0:alone", "A", 1, 1, 1.0, note=note)  # noqa: E731
    steps = {"a": {"x": st("cut off; fixed by a short re-ask for the letter; load-time outlier counted at the "
                           "median (107s)")},
             "b": {"x": st("no option named; still blank after a short re-ask for the letter")},
             "c": {"x": st("")}}
    assert s.reask_counts(steps) == (2, 1)


def test_live_runs_bench_the_code_checker_unless_asked(monkeypatch):
    s = _script()
    seen = []
    monkeypatch.setattr(s, "cmd_sweep", lambda ns: seen.append(ns.code) or 0)
    assert s.main(["sweep", "--dev", "d", "--out", "o"]) == 0
    assert s.main(["sweep", "--dev", "d", "--out", "o", "--code"]) == 0
    assert seen == [False, True]


def test_import_bills_a_thrown_away_cut_off_reply_at_its_length_limit(tmp_path):
    from antelligence.experiments import fair_test as ft

    run, tasks, key, steps = _saved_set(tmp_path, 8, "set1")
    t = tasks[0]
    item = sr.Item(t.id, t.kind, t.passage, t.stem, t.options)
    h = sr.make_request(CFG.reader, sr.reader_answer_messages(item, CFG.reader), t.id, "reader", 0,
                        ft.MAX_TOKENS).request_hash
    rows = [json.loads(line) for line in (run / "cache.jsonl").read_text().splitlines()]
    rows = [r if r["request_hash"] != h else {"request_hash": h, "model": r["model"], "content": "",
                                               "error": "ProviderTruncated",
                                               "reason": "generation did not finish normally (length)"}
            for r in rows]
    (run / "cache.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, _, isteps, _, reports = import_sets({"set1": run}, CFG)
    got = isteps[f"set1:{t.id}"]["reader"]
    assert got.tokens_out == ft.MAX_TOKENS and got.tokens_in > 0 and got.latency_s > 0
    assert got.note.startswith("cut off") and reports[0]["cut_off_billed"] == 1 and reports[0]["priced"] == 1.0
