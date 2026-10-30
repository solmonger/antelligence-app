"""Tests for scripts/cached_colony_check.py on tiny synthetic cells. No model calls, no repo data."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("cached_colony_check", REPO / "scripts" / "cached_colony_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cached_colony_check"] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _load()


def _row(model, domain, task, sample, answer, correct, status="verified"):
    return {"model": model, "domain": domain, "task_id": task, "sample": str(sample), "answer": answer,
            "correct": correct, "verification": {"status": status} if status else None}


def _write(tmp: Path, split: str, rows):
    for m in cc.MODELS:
        (tmp / f"{m}-{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["model"] == m))


def _cells(tmp: Path, *, qwen_first_verified: bool):
    """One item per domain. Qwen right, Haiku right, Phi and Gemma wrong (and agreeing with each other)."""
    rows = []
    for d in cc.DOMAINS:
        for s in cc.SAMPLES:
            ok_status = "verified" if qwen_first_verified else "failed"
            rows.append(_row("qwen", d, f"{d}:1", s, "yes", True, ok_status))
            rows.append(_row("haiku", d, f"{d}:1", s, "yes", True))
            rows.append(_row("phi", d, f"{d}:1", s, "no", False, "failed"))
            rows.append(_row("gemma", d, f"{d}:1", s, "no", False, None))
    _write(tmp, "evaluation", rows)
    _write(tmp, "development", rows)


def test_vote_ties_go_to_the_first_answer():
    rows = [_row("qwen", "finance", "t", 1, "a", True), _row("phi", "finance", "t", 1, "b", False)]
    assert cc.vote(rows) is True
    assert cc.vote(list(reversed(rows))) is False


def test_vote_ignores_missing_answers_and_normalises():
    rows = [_row("qwen", "finance", "t", 1, None, False), _row("phi", "finance", "t", 1, " Yes ", True),
            _row("gemma", "finance", "t", 1, "yes", True)]
    assert cc.vote(rows) is True
    assert cc.vote([_row("qwen", "finance", "t", 1, None, False)]) is False


def test_trust_weights_are_zero_for_coin_flippers(tmp_path):
    _cells(tmp_path, qwen_first_verified=True)
    w = cc.trust_weights(cc.load(tmp_path, "development"))
    assert w[("phi", "finance")] == 0.0 and w[("gemma", "medical")] == 0.0
    assert math.isclose(w[("qwen", "finance")], math.log(5 / 6 / (1 / 6)))


def test_earned_trust_beats_equal_votes_when_weak_members_agree(tmp_path):
    _cells(tmp_path, qwen_first_verified=True)
    res = cc.evaluate(cc.load(tmp_path, "evaluation"), cc.trust_weights(cc.load(tmp_path, "development")))
    for d in cc.DOMAINS:
        assert res[d]["mixed colony, equal votes"]["accuracy"] == 0.0  # phi + gemma outvote qwen
        assert res[d]["mixed colony, earned trust"]["accuracy"] == 1.0


def test_grow_on_hard_spends_one_call_when_the_first_answer_checks_out(tmp_path):
    _cells(tmp_path, qwen_first_verified=True)
    res = cc.evaluate(cc.load(tmp_path, "evaluation"), cc.trust_weights(cc.load(tmp_path, "development")))
    assert res["finance"]["grow on hard: +2 qwen"]["calls_per_item"] == 1.0
    assert res["finance"]["grow on hard: ask haiku"]["calls_per_item"] == 1.0


def test_grow_on_hard_recruits_when_the_check_fails(tmp_path):
    _cells(tmp_path, qwen_first_verified=False)
    res = cc.evaluate(cc.load(tmp_path, "evaluation"), cc.trust_weights(cc.load(tmp_path, "development")))
    assert res["medical"]["grow on hard: +2 qwen"]["calls_per_item"] == 3.0
    assert res["medical"]["grow on hard: ask haiku"]["calls_per_item"] == 2.0
    assert res["medical"]["_n_replays"] == {"items": 1, "replays": 4}


def test_main_writes_json(tmp_path):
    _cells(tmp_path, qwen_first_verified=True)
    out = tmp_path / "out.json"
    assert cc.main([str(tmp_path), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert set(data["results"]) == set(cc.DOMAINS)
    assert "qwen/finance" in data["trust_weights"]
