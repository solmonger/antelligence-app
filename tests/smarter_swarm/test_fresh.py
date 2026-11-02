"""Tests for the fresh-question test: the question maker, the letter check, the script and the MLX stand-in. $0."""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import re
import socket
import sys
import textwrap
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import dm_leak as leak
from antelligence.experiments import fair_test as ft
from antelligence.experiments import fresh_questions as fq

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs/research/fresh-20261031"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- the questions


@pytest.fixture(scope="module")
def made():
    return fq.make_set(20261031, per_task=12)


def test_every_question_has_exactly_one_right_option_by_brute_force(made):
    qs, ans, specs = made
    assert len(qs) == 48 and {q["task"] for q in qs} == set(fq.TASKS)
    for q, a, s in zip(qs, ans, specs):
        assert fq.check(q, s), q["id"]
        assert a["label"] == s["label"] and len(q["options"]) == 5
        assert [o[:3] for o in q["options"]] == ["(A)", "(B)", "(C)", "(D)", "(E)"]
        assert len({o[3:] for o in q["options"]}) == 5


def test_questions_are_in_the_fair_test_format_and_carry_no_answers(made, tmp_path):
    qs, _, _ = made
    dm.write_jsonl(tmp_path / "q.jsonl", qs)
    loaded = ft.load_questions(tmp_path / "q.jsonl", "exam")      # refuses any answer field
    assert len(loaded) == len(qs) and loaded[0].text.endswith(qs[0]["options"][-1])
    assert dm.load_options(tmp_path / "q.jsonl")[qs[0]["id"]]["A"] == qs[0]["options"][0][3:]


def test_a_seed_gives_the_same_set_and_another_seed_a_new_one(made):
    again = fq.make_set(20261031, per_task=12)
    other = fq.make_set(1, per_task=12)
    assert again[0] == made[0]
    assert not {q["passage"] for q in other[0]} & {q["passage"] for q in made[0]}


def test_right_letters_are_spread_out():
    _, ans, _ = fq.make_set(5, per_task=40)
    counts = {x: sum(a["label"] == x for a in ans) for x in "ABCDE"}
    assert min(counts.values()) >= 16, counts                      # 160 questions, about 32 each


def test_the_quick_syllogism_check_matches_brute_force_over_every_model():
    rng = random.Random(3)
    statements = [(f, x, y) for f in fq.FORMS for x in range(4) for y in range(4) if x != y]
    for _ in range(25):
        premises = rng.sample(statements, rng.randint(2, 4))
        for imp in (False, True):
            models = fq.syl_models(premises, imp)
            assert fq.syl_sat(premises, imp) == bool(models)
            if models:
                for st in rng.sample(statements, 6):
                    assert fq.syl_follows(st, premises, imp) == all(fq.syl_holds(st, m) for m in models)


def test_a_textbook_syllogism():
    # All 0 are 1; all 1 are 2 => all 0 are 2. Some 0 are 2 follows only if every kind exists.
    p = [("all", 0, 1), ("all", 1, 2)]
    assert fq.syl_follows(("all", 0, 2), p, False) and fq.syl_follows(("all", 0, 2), p, True)
    assert not fq.syl_follows(("some", 0, 2), p, False) and fq.syl_follows(("some", 0, 2), p, True)


def test_order_rules_mean_what_they_say():
    pos = {"F": 1, "G": 2, "H": 3}
    assert fq.order_holds(("before", "F", "H"), pos) and not fq.order_holds(("before", "H", "F"), pos)
    assert fq.order_holds(("next", "F", "G"), pos) and not fq.order_holds(("next", "F", "H"), pos)
    assert fq.order_holds(("apart", "F", "H"), pos) and not fq.order_holds(("apart", "F", "G"), pos)
    assert fq.order_holds(("at_one", "G", frozenset({2, 5})), pos) and fq.order_holds(("not_at", "G", 1), pos)
    assert fq.order_text(("at_one", "G", frozenset({5, 2}))) == "G comes either second or fifth."
    sols = fq.order_solutions("FGH", [("next", "F", "G"), ("not_at", "F", 1)])
    assert sols == [("H", "F", "G")]


def test_assignment_puzzles_have_one_solution(made):
    qs, _, specs = made
    for s in (s for s in specs if s["task"] == "assign"):
        sols = fq.assign_solutions(s["people"], s["colours"], [tuple(c) for c in s["clues"]])
        assert len(sols) == 1


def _nums(text):
    return [int(x) for x in re.findall(r"\d+", text)]


def test_word_problem_answers_worked_out_again_from_the_text():
    rng = random.Random(11)
    for _ in range(30):
        text, question, right, _ = fq._arith_work(rng)
        a, b = _nums(text)
        assert right * (a + b) == a * b
        text, question, right, _ = fq._arith_speed(rng)
        d, u, v = _nums(text)
        assert right * (u + v) == 2 * u * v
        text, question, right, _ = fq._arith_mixture(rng)
        x, a, y, b = _nums(text)
        assert right * (x + y) == a * x + b * y
        text, question, right, _ = fq._arith_ages(rng)
        s, d = _nums(text)
        (n,) = _nums(question)
        assert right == (s + d) // 2 + n and (s + d) % 2 == 0
        text, question, right, _ = fq._arith_remainder(rng)
        r1, m1, r2, m2 = _nums(text)
        assert right % m1 == r1 and right % m2 == r2 and all(k % m1 != r1 or k % m2 != r2 for k in range(1, right))
        text, question, right, _ = fq._arith_percent(rng)
        p, x, y = _nums(text)
        assert right * 10000 == p * (100 + x) * (100 - y)
        text, question, right, _ = fq._arith_compound(rng)
        p, r = _nums(text)[:2]
        assert right * 10000 == p * (100 + r) ** 2


# ------------------------------------------------------------- letter check


RAW = {"id": "lsat-ar-0001", "task": "lsat-ar", "passage": "Some rules.", "question": "Which works?",
       "options": ["(A)one", "(B)two", "(C)three", "(D)four", "(E)five"]}


def test_moving_letters_leaves_no_option_where_it_was_and_maps_back():
    moved = leak.derangement("ABCDE", "q1")
    assert sorted(moved.values()) == list("ABCDE") and all(k != v for k, v in moved.items())
    assert leak.derangement("ABCDE", "q1") == moved and leak.derangement("ABCDE", "q2") != moved
    req, back = leak.probe_request("jev-1.13.0", RAW, "relabel")
    state = req.state
    assert "Some rules." in state and "Which works?" in state
    for old, new in leak.derangement("ABCDE", "lsat-ar-0001").items():
        text = dm.option_texts(RAW)[old]
        assert f"({new}){text}" in state and back[new] == old
    blind, _ = leak.probe_request("jev-1.13.0", RAW, "blind")
    assert "Some rules." not in blind.state and "Which works?" not in blind.state
    assert leak.back({"choice": "A"}, back) == back["A"] and leak.back(None, back) is None


def _leak_case(memoriser: bool, n: int = 60):
    raws = [{**RAW, "id": f"lsat-ar-{i:04d}", "task": "lsat-ar" if i % 2 else "lsat-lr"} for i in range(n)]
    key = {r["id"]: "C" for r in raws}
    dm1 = {r["id"]: "C" for r in raws}
    picks = {"relabel": {}, "blind": {}}
    for r in raws:
        _, back = leak.probe_request("m", r, "relabel")
        if memoriser:     # picks the letter it remembers, C, whatever option now stands there
            picks["relabel"][r["id"]] = back["C"]
            picks["blind"][r["id"]] = back["C"]
        else:             # reasons: finds option "three" wherever it now is; can't tell with options alone
            picks["relabel"][r["id"]] = "C"
            picks["blind"][r["id"]] = random.Random(r["id"]).choice("ABCDE")
    return raws, picks, key, dm1


def test_letter_check_catches_a_model_that_remembers_letters():
    res = leak.score(*_leak_case(True))
    assert res["means"]["dm1"] == 1.0 and res["means"]["relabel"] == 0.0
    assert res["relabel_drop"]["gain"] == pytest.approx(1.0) and res["verdict"].startswith("likely memorised")
    assert set(res["by_group"]) == {"logic", "puzzles"}


def test_letter_check_clears_a_model_that_reasons():
    res = leak.score(*_leak_case(False, n=200))
    assert res["means"]["relabel"] == 1.0 and res["relabel_drop"]["gain"] == 0.0
    assert not res["verdict"].startswith("likely memorised")


def test_letter_check_is_invalid_when_answers_are_missing():
    raws, picks, key, dm1 = _leak_case(False)
    picks["relabel"] = {k: None for k in picks["relabel"]}
    assert leak.score(raws, picks, key, dm1)["verdict"].startswith("invalid")


# ------------------------------------------------------------------- script


def test_the_committed_plan_is_frozen():
    cli = _load("fresh_test")
    assert json.loads((DATA / "prereg.json").read_text()) == cli.fingerprint()


def test_dry_run_end_to_end(tmp_path, capsys):
    cli = _load("fresh_test")
    data = tmp_path / "fresh"
    assert cli.main(["freeze", "--data", str(data)]) == 0
    assert cli.main(["make", "--data", str(data), "--dry-run", "--seed", "4", "--per-task", "5"]) == 0
    qs = (data / "dry-run/questions.jsonl").read_text()
    assert cli.main(["make", "--data", str(data), "--dry-run"]) == 0        # kept, not made again
    assert (data / "dry-run/questions.jsonl").read_text() == qs
    with pytest.raises(SystemExit, match="swarm"):
        cli.main(["dm", "--data", str(data), "--dry-run", "--dm", "jev"])
    assert cli.main(["swarm", "--data", str(data), "--dry-run", "--headroom-gb", "0"]) == 0
    for name in ("jev", "d1-3b"):
        assert cli.main(["dm", "--data", str(data), "--dry-run", "--dm", name]) == 0
    assert cli.main(["leak", "--data", str(data), "--dry-run", "--dm", "jev"]) == 0
    capsys.readouterr()
    assert cli.main(["score", "--data", str(data), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "FRESH TEST RESULT" in out and "VERDICT, Jev's lead" in out and "LETTER CHECK VERDICT" in out
    assert "jev:dm1" in out and "d1-3b:vote" in out and "reader" in out
    res = json.loads((data / "dry-run/result.json").read_text())
    assert res["swarm"]["n"] == 20 and res["leak"]["n"] == 240


def test_no_decision_model_is_asked_when_the_ants_are_empty(tmp_path):
    cli = _load("fresh_test")
    data = tmp_path / "fresh"
    cli.main(["freeze", "--data", str(data)])
    cli.main(["make", "--data", str(data), "--dry-run", "--seed", "4", "--per-task", "2"])
    d = data / "dry-run"
    qs = dm.read_jsonl(d / "questions.jsonl")
    dm.write_jsonl(d / "rows.jsonl", [{"id": q["id"], "group": q["task"], "reader": "A", "alone": [None] * 8,
                                       "notes": [None] * 8, "both": [None] * 8, "leak": False} for q in qs])
    with pytest.raises(SystemExit, match="usable"):
        cli.main(["dm", "--data", str(data), "--dry-run", "--dm", "jev"])


def test_caps_keep_jev_under_five_cents_with_run_one():
    cli = _load("fresh_test")
    assert 0.0223 + sum(cli.CAPS.values()) < 0.05
    with pytest.raises(SystemExit, match="cap"):
        cli.main(["dm", "--dry-run", "--max-usd", "1"])


def test_a_changed_plan_stops_the_run(tmp_path):
    cli = _load("fresh_test")
    data = tmp_path / "fresh"
    cli.main(["freeze", "--data", str(data)])
    p = json.loads((data / "prereg.json").read_text())
    p["caps"] = {"dm": 1, "leak": 1}
    (data / "prereg.json").write_text(json.dumps(p))
    with pytest.raises(SystemExit, match="changed"):
        cli.main(["make", "--data", str(data), "--dry-run"])


# --------------------------------------------------------------- stand-in

FAKE_MLX = textwrap.dedent('''
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    model, port = args[args.index("--model") + 1], int(args[args.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, obj):
            b = json.dumps(obj).encode(); self.send_response(200)
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self): self._send({"data": [{"id": model}]})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send({"model": model, "choices": [{"message": {"content": "served by " + model + " for " + body["model"]}}]})
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_stand_in_switches_models_and_frees_memory_on_unload(tmp_path):
    si = _load("mlx_standin")
    fake = tmp_path / "fake_mlx.py"
    fake.write_text(FAKE_MLX)
    backend = si.Backend([sys.executable, str(fake)], _free_port(), start_timeout_s=60)
    server = ThreadingHTTPServer(("127.0.0.1", 0), si.make_handler(backend, ["m/reader", "m/ant"]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
            assert [m["id"] for m in json.loads(r.read())["data"]] == ["m/reader", "m/ant"]
        out = _post(f"{base}/v1/chat/completions", {"model": "m/reader", "messages": []})
        assert out["choices"][0]["message"]["content"] == "served by m/reader for m/reader"
        first = backend.proc
        out = _post(f"{base}/v1/chat/completions", {"model": "m/ant", "messages": []})
        assert "served by m/ant" in out["choices"][0]["message"]["content"]
        assert first.poll() is not None                       # the reader's server was stopped first
        assert _post(f"{base}/api/generate", {"model": "m/ant", "keep_alive": 0}) == {"done": True}
        assert backend.proc is None
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(f"{base}/v1/chat/completions", {"model": "someone-elses", "messages": []})
        assert err.value.code == 404
    finally:
        server.shutdown()
        with backend.lock:
            backend.stop()
