"""Tests for scripts/deep_on_split_check.py on synthetic cells (no model calls)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dos = _load("deep_on_split_check")

TRUTH = {"finance": "a", "medical": "yes"}


def _write(cells: Path, split: str, answers) -> None:
    """answers(model, domain, task, sample) -> answer string."""
    cells.mkdir(parents=True, exist_ok=True)
    for m in ("qwen", "haiku", "phi", "gemma"):
        rows = []
        for d in ("finance", "medical"):
            for t in ("t1", "t2"):
                for s in (1, 2, 3, 4):
                    a = answers(m, d, t, s)
                    rows.append({"model": m, "domain": d, "task_id": t, "sample": s, "answer": a,
                                 "correct": a == TRUTH[d], "verification": {"status": "verified"}})
        (cells / f"{m}-{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_deep_is_asked_only_where_the_cheap_crowd_splits(tmp_path):
    # Development: haiku always right, cheap models right 3 times in 4 -> haiku carries more trust.
    _write(tmp_path, "development", lambda m, d, t, s: TRUTH[d] if (m == "haiku" or s != 4) else "wrong")
    # Evaluation: qwen splits on sample 4 only; phi/gemma always wrong on t2; haiku always right.
    def ev(m, d, t, s):
        if m == "haiku":
            return TRUTH[d]
        if m == "qwen":
            return "wrong" if s == 4 else TRUTH[d]
        return "wrong" if t == "t2" else TRUTH[d]
    _write(tmp_path, "evaluation", ev)
    w = dos.trust_weights(dos.load(tmp_path, "development"))
    assert w[("haiku", "finance")] > w[("qwen", "finance")] > 0
    res = dos.evaluate(dos.load(tmp_path, "evaluation"), w)
    fin = res["finance"]
    assert fin["deep once"]["accuracy"] == 1.0
    # qwen x3 rotations: the crowd holds samples [k, ...] minus one; it splits whenever sample 4 is in the first three.
    split = fin["qwen x3: deep decides where split"]
    assert 0 < split["deep_calls_per_item"] < 1
    assert split["split_items"] == fin["qwen x3: cheap vote"]["split_items"]
    assert fin["qwen x3: deep is one vote everywhere"]["deep_calls_per_item"] == 1.0
    assert fin["qwen x3: cheap vote"]["cheap_calls_per_item"] == 3.0
    assert res["medical"]["_n_replays"] == {"items": 2, "replays": 8}


def test_unanimous_needs_one_real_answer():
    assert dos.unanimous([{"answer": "Yes"}, {"answer": " yes "}])
    assert not dos.unanimous([{"answer": "yes"}, {"answer": "no"}])
    assert not dos.unanimous([{"answer": None}, {"answer": None}])


def test_main_writes_json(tmp_path):
    _write(tmp_path / "c", "development", lambda m, d, t, s: TRUTH[d])
    _write(tmp_path / "c", "evaluation", lambda m, d, t, s: TRUTH[d])
    out = tmp_path / "o.json"
    assert dos.main([str(tmp_path / "c"), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["deep_stand_in"] == "haiku" and "finance" in data["results"]
