"""The night run: routing by kind, field-notebook ants on exams, the memory watch and the resumable steps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import threading
import time
from pathlib import Path

import pytest

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import notebook_exam as nbx
from antelligence.experiments import route as rt
from antelligence.experiments import shared_read as sr

REPO = Path(__file__).resolve().parents[2]


def _night():
    spec = importlib.util.spec_from_file_location("_night_run_test", REPO / "scripts/night_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- routing

def _table(spec):
    """{group: {arm: [right, ...]}} -> a routing table."""
    t = {}
    for g, arms in spec.items():
        n = len(next(iter(arms.values())))
        for k in range(n):
            t[f"{g}-{k}"] = {"group": g, "right": {arm: float(v[k]) for arm, v in arms.items()}}
    return t


def test_table_scores_every_arm_like_the_addon():
    rows = [dm.SwarmRow("q1", "exam", "order", tuple(dm.Ant(a) for a in "AABBB"),
                        {"reader": "A", "alone5": "B", "notes5": "A", "both5": "B"},
                        {"reader": 1, "alone5": 5, "notes5": 6, "both5": 6})]
    dms = {"jev": {"q1": {"dm1": {"choice": "A", "probs": {"A": 0.9, "B": 0.1}}, "read": {"choice": "B", "probs": {}}}}}
    t = rt.table(rows, dms, {"q1": "A"})
    r = t["q1"]["right"]
    assert r["reader"] == 1 and r["alone5"] == 0 and r["notes5"] == 1
    assert r["jev:dm1"] == 1 and r["jev:read"] == 0
    assert r["jev:vote"] == 0   # 3 B votes against 2 A + 0.9 A


def test_routes_pick_the_best_arm_per_kind_and_ties_go_to_the_cheaper_arm():
    t = _table({"arith": {"reader": [1, 1, 1, 0], "jev:dm1": [1, 1, 1, 0], "notes8": [0, 0, 1, 0]},
                "syllogism": {"reader": [0, 1, 0, 0], "jev:dm1": [1, 1, 1, 1], "notes8": [1, 0, 1, 0]}})
    p = rt.pick_routes(t, rt.common_arms(t))
    assert p["routes"]["arith"]["arm"] == "reader"          # a tie with Jev goes to the free reader
    assert p["routes"]["syllogism"]["arm"] == "jev:dm1"
    assert p["single"]["arm"] == "jev:dm1"
    assert rt.common_arms(t) == ["reader", "notes8", "jev:dm1"]


def test_split_half_is_fair_where_in_sample_flatters():
    rng = random.Random(1)
    # ten arms of pure coin flips: picking the best per kind in-sample looks good; scored on the other half, it is
    # what it really is, about 50%, and so is the single best arm
    t = _table({g: {**{f"a{k}": [rng.random() < 0.5 for _ in range(40)] for k in range(10)}, "reader": [0] * 40}
                for g in ("order", "assign", "arith", "syllogism")})
    for v in t.values():
        v["right"]["reader"] = float(rng.random() < 0.5)
    s = rt.study(t, rt.common_arms(t))
    assert s["in_sample"]["routed"] > 0.6
    assert 0.44 < s["split_half"]["routed"] < 0.56
    assert 0.44 < s["split_half"]["single"] < 0.56
    lo, hi = s["split_half"]["routed_vs_reader"]["spread"]
    assert lo <= s["split_half"]["routed_vs_reader"]["gain"] <= hi


def test_split_half_finds_a_real_route():
    # syllogisms: Jev always right, the reader never; arithmetic the other way round. A router should get ~100%.
    t = _table({"arith": {"reader": [1] * 30, "jev:dm1": [0] * 30},
                "syllogism": {"reader": [0] * 30, "jev:dm1": [1] * 30}})
    s = rt.split_half(t, rt.common_arms(t), reps=20)
    assert s["routed"] == 1.0 and s["reader"] == 0.5 and s["single"] == 0.5


def test_held_out_uses_the_frozen_routes_and_reads_the_frozen_lines():
    frozen = {"routes": {"arith": {"arm": "reader"}, "syllogism": {"arm": "jev:dm1"}}, "single": {"arm": "reader"}}
    good = _table({"arith": {"reader": [1] * 40, "jev:dm1": [0] * 40},
                   "syllogism": {"reader": [0] * 40, "jev:dm1": [1] * 40}})
    res = rt.held_out(frozen, good)
    assert res["routed"] == 1.0 and res["reader"] == 0.5
    assert res["verdict"].startswith("helps")
    flat = _table({"arith": {"reader": [1, 0] * 20, "jev:dm1": [1, 0] * 20},
                   "syllogism": {"reader": [1, 0] * 20, "jev:dm1": [1, 0] * 20}})
    assert rt.held_out(frozen, flat)["verdict"].startswith("no gain")
    with pytest.raises(ValueError, match="no answers"):
        rt.held_out(frozen, _table({"arith": {"reader": [1, 0]}}))


# -------------------------------------------------------------------- notebook

GEMMA = sr.ModelSpec("g", "gemma")
QWEN = sr.ModelSpec("q", "qwen")
LFM = sr.ModelSpec("l", "lfm")
ROSTER = sr.ant_roster((GEMMA, QWEN, LFM))


def test_notebook_lists_every_other_model_once_and_the_ant_itself_last():
    first = [("A", "rule two fails"), ("A", ""), ("B", "x " * 50), (None, ""), ("C", "c"), ("C", "c")]
    book = nbx.notebook(first, ROSTER, QWEN)
    lines = book.splitlines()
    teammates = [l for l in lines if l.startswith("- Teammate")]
    assert len(teammates) == 2                      # gemma and lfm, once each, not their 4 tries
    assert "(gemma): first try A (\"rule two fails\"); second try A" in teammates[0]
    assert "(lfm): first try C" in teammates[1]
    own = next(l for l in lines if l.startswith("Your own work alone"))
    assert "first try B" in own and "second try: no answer" in own
    assert "..." in own                             # long reasons are cut
    assert "qwen" not in "\n".join(teammates)


def test_notebook_arms_need_notes_only_for_nb_notes_and_never_show_the_reader_answer():
    item = sr.Item("q1", "order", "", "Which order?", ("(A)x", "(B)y"))
    m = nbx.messages(item, "nb", 0, QWEN, "BOOK")
    assert "BOOK" in m[1]["content"] and "Notes from a reader" not in m[1]["content"]
    m2 = nbx.messages(item, "nb_notes", 0, QWEN, "BOOK", notes="the key rule")
    assert "the key rule" in m2[1]["content"]
    with pytest.raises(ValueError):
        nbx.messages(item, "nb_notes", 0, QWEN, "BOOK")
    with pytest.raises(ValueError):
        nbx.messages(item, "both", 0, QWEN, "BOOK")


def test_first_round_reasons_come_from_the_cache_only_when_the_answer_matches():
    item = sr.Item("q1", "order", "", "Which order?", ("(A)x", "(B)y"))
    row = {"id": "q1", "alone": ["A", "B", None, "A", "A", "B"]}
    cache = {}
    for col, (spec, lens) in enumerate(ROSTER):
        req = sr.make_request(spec, sr.ant_messages(item, "alone", lens, spec), "q1", f"alone:{spec.tag}", lens,
                              sr.ANT_MAX_TOKENS)
        said = "A" if col != 1 else "C"           # column 1's cache says C but the row says B: not the same run
        cache[req.request_hash] = {"content": json.dumps({"reason": f"why {col}", "answer": said}), "error": None}
    got = nbx.first_round([item], {"q1": row}, cache, ROSTER, 2)["q1"]
    assert got[0] == ("A", "why 0")
    assert got[1] == ("B", "")
    assert got[2] == (None, "")


def test_notebook_score_measures_copying_a_wrong_reader():
    key = {f"q{k}": "A" for k in range(30)}
    rows = []
    for k in range(30):
        wrong = k < 15                                   # the reader is wrong (says B) on half
        reader = "B" if wrong else "A"
        rows.append({"id": f"q{k}", "group": "order", "reader": reader,
                     "alone": ["A", "A", "C", "D"], "notes": [reader] * 4, "both": [reader] * 3 + ["A"],
                     "nb": ["A", "A", "A", "C"], "nb_notes": ["A", "A", "A", reader]})
    s = nbx.score(rows, key, boot_reps=200)
    assert s["arms"]["both"]["copies_wrong_reader"] == 0.75
    assert s["arms"]["nb_notes"]["copies_wrong_reader"] == 0.25
    assert s["copy_drop_vs_both"]["gain"] == 0.5
    assert s["verdict"]["stops_copying_reader"] == "yes"
    assert s["verdict"]["notebook_helps"] == "yes"
    assert s["arms"]["nb_notes"]["vote"] == 1.0 and s["arms"]["both"]["vote"] == 0.5


def test_notebook_score_is_invalid_when_answers_are_missing():
    key = {"q0": "A"}
    rows = [{"id": "q0", "group": "g", "reader": "A", "alone": ["A"], "notes": ["A"], "both": ["A"],
             "nb": [None], "nb_notes": [None]}]
    assert "invalid" in nbx.score(rows, key, boot_reps=50)["verdict"]


# ------------------------------------------------------------- memory watch

def test_watch_trips_on_swap_growth_and_critical_pressure_and_rebases(tmp_path):
    nr = _night()
    readings = iter([{"level": 1.0, "swap_gb": 1.0, "free_pct": 40.0, "fseventsd_gb": 0.5},
                     {"level": 1.0, "swap_gb": 3.5, "free_pct": 30.0, "fseventsd_gb": 22.0}])
    w = nr.Watch(tmp_path / "m.jsonl", read=lambda: next(readings))
    w.check()
    assert w.tripped is None
    w.check()
    assert w.tripped == "swap grew 2.5 GB" and w.max_fse == 22.0 and w.min_free == 30.0
    w.read = lambda: {"level": 1.0, "swap_gb": 3.5, "free_pct": 50.0, "fseventsd_gb": 1.0}
    w.rebase()
    assert w.tripped is None and w.base_swap == 3.5
    w.read = lambda: {"level": 4.0, "swap_gb": 3.5, "free_pct": 5.0, "fseventsd_gb": 1.0}
    w.check()
    assert w.tripped == "memory pressure critical"
    assert not w.normal()
    assert len((tmp_path / "m.jsonl").read_text().splitlines()) == 3


def test_a_memory_stop_kills_the_step_waits_and_retries_once(tmp_path):
    nr = _night()
    a = argparse.Namespace(dry_run=True, night=tmp_path, hours=1.0, mlx_server="", d1_python="", agieval="")
    w = nr.Watch(tmp_path / "logs/m.jsonl", read=lambda: {"level": 1.0, "swap_gb": None, "free_pct": None,
                                                         "fseventsd_gb": None})
    night = nr.Night(a, watch=w)
    night.logs.mkdir(parents=True)
    trip = threading.Timer(0.5, lambda: setattr(w, "tripped", "memory pressure critical"))
    trip.start()
    marker = tmp_path / "ran"
    cmd = [sys.executable, "-c",
           f"import pathlib,time; p=pathlib.Path({str(marker)!r}); n=int(p.read_text()) if p.exists() else 0; "
           f"p.write_text(str(n+1)); time.sleep(3 if n == 0 else 0)"]
    t0 = time.monotonic()
    assert night.step("demo", cmd, timeout_min=1)
    assert time.monotonic() - t0 < 3                 # the first attempt was stopped, not waited out
    assert marker.read_text() == "2"
    assert night.state["memory_stops"] == 1 and night.state["stages"]["demo"]["status"] == "done"
    assert night.step("demo", [sys.executable, "-c", "raise SystemExit(1)"])   # a done step is not run again
    saved = json.loads((tmp_path / "state_dry.json").read_text())
    assert saved["order"] == ["demo"]


def test_a_failing_step_is_marked_and_does_not_retry(tmp_path):
    nr = _night()
    a = argparse.Namespace(dry_run=True, night=tmp_path, hours=1.0, mlx_server="", d1_python="", agieval="")
    night = nr.Night(a, watch=nr.Watch(tmp_path / "logs/m.jsonl", read=lambda: {}))
    night.logs.mkdir(parents=True)
    assert not night.step("bad", [sys.executable, "-c", "raise SystemExit(3)"], timeout_min=1)
    assert night.state["stages"]["bad"]["status"] == "failed"
    assert "exit 3" in night.state["stages"]["bad"]["note"]


def test_night_plan_is_frozen_and_sets_run_under_the_first_sets_plan(tmp_path):
    nr = _night()
    assert nr.changed(tmp_path) == []
    d = nr.prepare_set(tmp_path, "set2")
    assert (d / "prereg.json").read_bytes() == (nr.FRESH / "prereg.json").read_bytes()
    assert nr.MLX_PORT != 8090                       # never the chat shim's port
    assert nr.JEV_NIGHT_CAP <= 0.05


def test_jev_cap_counts_only_tonights_sets(tmp_path):
    nr = _night()
    a = argparse.Namespace(dry_run=False, night=tmp_path, hours=1.0, mlx_server="x", d1_python="", agieval="")
    night = nr.Night(a, watch=nr.Watch(tmp_path / "m.jsonl", read=lambda: {}))
    for name, usd in (("set1", 0.9), ("set2", 0.004), ("set3", 0.003)):
        p = tmp_path / name / "run" / "cache_jev.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"cost_usd": usd}) + "\n")
    assert round(night.jev_spent(), 6) == 0.007
