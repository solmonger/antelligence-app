"""Tests for scripts/alone_vs_seen_check.py and scripts/fading_alone_check.py. Synthetic data, no model calls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


avs = _load("alone_vs_seen_check")
fac = _load("fading_alone_check")


def _cell(arm, task, seed, finals, correct, status="completed", domain="medical", model="qwen38-27b-q3k"):
    answer = max(set(finals), key=finals.count) if finals else None
    return {"model_key": model, "arm": arm, "world": "research_qa", "task_id": task, "seed": seed, "domain": domain,
            "status": status, "final_answers": finals, "answer": answer, "correct": correct}


def test_norm_strips_formatting():
    assert avs.norm(" 1,234 % ") == "1234" and avs.norm("Yes") == "yes" and avs.norm(None) is None
    assert avs.norm("  ") is None


def test_gold_comes_only_from_correct_cells():
    rows = [_cell("single", "t1", 0, ["yes"], True), _cell("single", "t2", 0, ["no"], False)]
    assert avs.gold_answers(rows) == {"t1": "yes"}


def test_compare_pair_counts_help_harm_unanimity_and_shared_errors():
    rows = [
        _cell("single", "t1", 0, ["yes"], True), _cell("single", "t2", 0, ["no"], True),
        # t1: blind team right with one dissent; seen team herds onto the right answer
        _cell("independent_vote", "t1", 0, ["yes", "yes", "no"], True),
        _cell("signal_board", "t1", 0, ["yes", "yes", "yes"], True),
        # t2: blind right; seen team herds onto the wrong answer (hurt), unanimous and wrong
        _cell("independent_vote", "t2", 0, ["no", "no", "yes"], True),
        _cell("signal_board", "t2", 0, ["yes", "yes", "yes"], False),
        # t3: seen cell broke (invalid) -> counts as wrong team answer, excluded from answer-level stats
        _cell("independent_vote", "t3", 0, ["no", "no", "no"], True),
        _cell("signal_board", "t3", 0, [], False, status="invalid"),
    ]
    res = avs.compare_pair(rows, "qwen38-27b-q3k", "independent_vote", "signal_board", avs.gold_answers(rows))
    assert res["pairs"] == 3 and res["pairs_both_completed"] == 2
    assert res["blind"]["team_accuracy"] == 1.0 and round(res["seen"]["team_accuracy"], 4) == round(1 / 3, 4)
    assert res["seen_hurt"] == 2 and res["seen_helped"] == 0
    assert res["seen_hurt_completed"] == 1
    assert res["seen"]["unanimous"] == 1.0 and res["seen"]["unanimous_and_wrong"] == 0.5
    # blind: one wrong agent per item, partner never wrong -> 0; seen t2: all three wrong -> 1
    assert res["blind"]["wrong_together"] == 0.0 and res["seen"]["wrong_together"] == 1.0


def test_alone_vs_seen_main_writes_json(tmp_path):
    rows = [_cell("single", "t1", 0, ["yes"], True),
            _cell("independent_vote", "t1", 0, ["yes"] * 3, True), _cell("signal_board", "t1", 0, ["yes"] * 3, True)]
    (tmp_path / "cells-qwen38-27b-q3k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "o.json"
    assert avs.main([str(tmp_path), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["gold_tasks"] == 1 and data["results"][0]["pairs"] == 1


def test_fading_alone_grid_is_paired_and_complete(tmp_path):
    out = tmp_path / "f.json"
    assert fac.main(["--ns", "16", "--seeds", "0,1", "--ticks", "10", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    rows = data["rows"]
    # 2 seeds x (2 fades x 2 policies + 1 solo) x 4 seed kinds
    assert len(rows) == 2 * 5 * 4
    assert all(r["extra_wrong_mean"] == 0 for r in rows if r["seeded"] == "none")
    names = {(c["a"], c["b"], c["seeded"]) for c in data["summary"]["comparisons"]}
    assert ("smart/alone", "fixed/herd", "none") in names


def test_boot_ci_brackets_the_mean():
    mean, lo, hi = fac.boot_ci([0.1, 0.2, 0.3, 0.4])
    assert lo <= mean <= hi
