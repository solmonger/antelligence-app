"""Tests for the free local forecast run (antelligence/experiments/forecast_local.py, scripts/forecast_local.py).

No network and no model: the live path runs against tests/smarter_swarm/fake_mlx_server.py on 127.0.0.1."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import stat
import sys
import time
from pathlib import Path

import pytest

from antelligence.experiments import fair_test as ft
from antelligence.experiments import forecast_local as fl
from antelligence.providers.base import ChatResponse, ProviderError

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs/research/fair-test-20261029"
OUT = REPO / "docs/research/forecast-local-20261102"
FAKE = Path(__file__).with_name("fake_mlx_server.py")


def _script():
    spec = importlib.util.spec_from_file_location("forecast_local_cli", REPO / "scripts/forecast_local.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["forecast_local_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _models_file(tmp: Path, gb: float = 0.0) -> Path:
    cfg = json.loads(fl.MODELS_FILE.read_text())
    for m in [cfg["reader"], *cfg["ants"]]:
        m["gb"] = gb
    p = tmp / "models.json"
    p.write_text(json.dumps(cfg))
    return p


def _hub(tmp: Path, tags) -> Path:
    hub = tmp / "hub"
    for tag in tags:
        (hub / fl.cache_dir_name(tag) / "snapshots" / "abc123").mkdir(parents=True)
    return hub


def _launcher(tmp: Path) -> Path:
    """An executable that runs the fake server with this Python, the way mlx_lm.server is an executable."""
    sh = tmp / "mlx_lm.server"
    sh.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC)
    return sh


# ------------------------------------------------------------------ the plan

def test_models_are_the_night_runs_and_the_late_cut_is_the_newest_release():
    reader, ants = fl.load_models()
    night = json.loads((REPO / "docs/research/fresh-20261031/models.json").read_text()) \
        if (REPO / "docs/research/fresh-20261031/models.json").exists() else None
    assert reader.tag == "mlx-community/gemma-4-12B-it-4bit"
    assert [a.family for a in ants] == ["gemma", "qwen", "granite", "lfm"]
    if night:
        assert [a["tag"] for a in night["ants"]] == [a.tag for a in ants]
    assert max(m.released for m in (reader, *ants)) == fl.LATE_FROM
    answers = [json.loads(x) for x in (DATA / "forecast_answers.jsonl").read_text().splitlines()]
    assert all(r["resolution_date"] > reader.released for r in answers)   # Test A can't be scored on memory
    late = [r for r in answers if r["resolution_date"] >= fl.LATE_FROM]
    assert len(late) == 67


def test_a_model_newer_than_the_late_cut_is_refused(tmp_path):
    cfg = json.loads(fl.MODELS_FILE.read_text())
    cfg["ants"][0]["released"] = "2026-09-30"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="late cut"):
        fl.load_models(p)


def test_a_reader_released_after_the_first_forecast_is_refused(tmp_path):
    cfg = json.loads(fl.MODELS_FILE.read_text())
    cfg["reader"]["released"] = "2026-07-10"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="reader came out after"):
        fl.load_models(p)
    answers = [json.loads(x) for x in (DATA / "forecast_answers.jsonl").read_text().splitlines()]
    assert min(r["resolution_date"] for r in answers) == fl.FIRST_RESOLVED


def test_the_frozen_plan_matches_the_files():
    cli = _script()
    assert cli.changed(DATA, fl.MODELS_FILE, OUT / "prereg.json") == []
    frozen = json.loads((OUT / "prereg.json").read_text())
    assert frozen["pass"]["min_brier_gain"] == ft.PASS["forecast_min_brier_gain"] == 0.005
    assert frozen["lenses"] == list(ft.LENSES["forecast"])


def test_call_counts():
    assert fl.plan(101) == {"questions": 101, "reader_calls": 2020, "ant_calls": 404, "calls": 2424}


def test_requests_use_the_fair_test_prompts_and_each_model():
    reader, ants = fl.load_models()
    q = ft.load_questions(DATA / "forecast_questions.jsonl", "forecast")[0]
    first = fl.first_requests(q, reader)
    assert len(first) == 15 and len({r.request_hash for r in first}) == 15
    assert first[0].messages == ft.plain_messages(q) and first[10].messages == ft.lens_messages(q, 0)
    assert all(r.model == reader.tag and r.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
               for r in first)
    ant = fl.ant_request(q, ants[1])
    assert ant.model == ants[1].tag and ant.messages == ft.plain_messages(q)
    assert "outcome" not in json.dumps(ant.messages) and "crowd" not in json.dumps(ant.messages)


# ------------------------------------------------------------- model cache

def test_external_drive_paths_are_spotted():
    assert fl.on_external_disk(Path("/Volumes/WD_BLACK/huggingface/hub"))
    assert not fl.on_external_disk(Path.home() / "antelligence-models/hub")


def test_copy_models_copies_links_and_skips_what_is_done(tmp_path):
    reader, ants = fl.load_models()
    models = (reader, *ants)
    src = tmp_path / "src"
    for m in models:
        base = src / fl.cache_dir_name(m.tag)
        (base / "blobs").mkdir(parents=True)
        (base / "blobs" / "b1").write_bytes(b"x" * 1000)
        (base / "snapshots" / "s1").mkdir(parents=True)
        os.symlink("../../blobs/b1", base / "snapshots" / "s1" / "model.safetensors")
    logs = []
    sizes = fl.copy_models(src, tmp_path / "dst", models, min_free_gb=0, log=logs.append)
    assert all(v == 1000 for v in sizes.values()) and len(logs) == 5
    link = tmp_path / "dst" / fl.cache_dir_name(reader.tag) / "snapshots" / "s1" / "model.safetensors"
    assert link.is_symlink() and link.read_bytes() == b"x" * 1000
    assert fl.missing_models(tmp_path / "dst", models) == []
    logs.clear()
    fl.copy_models(src, tmp_path / "dst", models, min_free_gb=0, log=logs.append)
    assert logs == []                                       # already copied whole: nothing to do
    assert (src / fl.cache_dir_name(reader.tag) / "blobs" / "b1").exists()   # the source is untouched


def test_a_link_back_to_the_old_drive_becomes_a_real_file(tmp_path):
    reader, ants = fl.load_models()
    models = (reader, *ants)
    old_drive = tmp_path / "old_drive_blobs"
    old_drive.mkdir()
    (old_drive / "weights").write_bytes(b"w" * 500)
    src = tmp_path / "src"
    for m in models:
        snap = src / fl.cache_dir_name(m.tag) / "snapshots" / "s1"
        snap.mkdir(parents=True)
        os.symlink(str(old_drive / "weights"), snap / "model.safetensors")   # an absolute link outside the cache
    fl.copy_models(src, tmp_path / "dst", models, min_free_gb=0, log=lambda m: None)
    f = tmp_path / "dst" / fl.cache_dir_name(reader.tag) / "snapshots" / "s1" / "model.safetensors"
    assert not f.is_symlink() and f.read_bytes() == b"w" * 500
    (old_drive / "weights").unlink()                         # the old drive goes away: the copy still reads
    assert f.read_bytes() == b"w" * 500


def test_copy_models_refuses_to_fill_the_disk(tmp_path):
    reader, ants = fl.load_models()
    src = _hub(tmp_path, [m.tag for m in (reader, *ants)])
    with pytest.raises(OSError, match="must stay free"):
        fl.copy_models(src, tmp_path / "dst", (reader, *ants), min_free_gb=10 ** 9)


def test_copy_models_names_what_the_source_lacks(tmp_path):
    reader, ants = fl.load_models()
    src = _hub(tmp_path, [reader.tag])
    with pytest.raises(FileNotFoundError, match="granite"):
        fl.copy_models(src, tmp_path / "dst", (reader, *ants), min_free_gb=0)


# ----------------------------------------------------------------- running

class _Server:
    def __init__(self):
        self.model, self.started, self.stops = None, [], 0

    def start(self, tag):
        self.model = tag
        self.started.append(tag)

    def stop(self):
        self.model = None
        self.stops += 1


class _Fake:
    async def complete(self, req):
        p = 0.9 if req.model.endswith("Instruct-MLX-4bit") else 0.4
        return ChatResponse(content=json.dumps({"reason": "r", "probability": p}), model=req.model,
                            request_hash=req.request_hash, prompt_tokens=1, completion_tokens=1, finish_reason="stop")


def test_one_model_at_a_time_and_a_rerun_loads_nothing(tmp_path):
    reader, ants = fl.load_models()
    qs = ft.load_questions(DATA / "forecast_questions.jsonl", "forecast")[:3]
    srv = _Server()
    runner = ft.Runner(fl.OwnServer(_Fake(), srv), tmp_path / "c.jsonl", max_usd=0, price=(0, 0), max_call_usd=0)
    rows = asyncio.run(fl.LocalRun(runner, srv, reader, ants, log=lambda m: None).run(qs))
    assert srv.started == [reader.tag, *(a.tag for a in ants)]
    assert srv.model is None                                   # stopped at the end
    assert rows[0]["single"] == [0.4] * 10 and rows[0]["ants"]["lfm"] == 0.9 and len(rows[0]["seen"]) == 5
    srv2 = _Server()
    runner2 = ft.Runner(fl.OwnServer(_Fake(), srv2), tmp_path / "c.jsonl", max_usd=0, price=(0, 0), max_call_usd=0)
    rows2 = asyncio.run(fl.LocalRun(runner2, srv2, reader, ants, log=lambda m: None).run(qs))
    assert srv2.started == [] and runner2.calls == 0 and rows2 == rows


def test_the_memory_guard_runs_before_each_load(tmp_path):
    reader, ants = fl.load_models()
    qs = ft.load_questions(DATA / "forecast_questions.jsonl", "forecast")[:1]
    srv = _Server()
    runner = ft.Runner(fl.OwnServer(_Fake(), srv), tmp_path / "c.jsonl", max_usd=0, price=(0, 0), max_call_usd=0)

    def guard(m):
        if m.family == "granite":
            raise fl.MemoryTight("no room")

    with pytest.raises(fl.MemoryTight):
        asyncio.run(fl.LocalRun(runner, srv, reader, ants, guard=guard, log=lambda m: None).run(qs))
    assert "ibm-granite/granite-4.2-3b-q4-mlx" not in srv.started and srv.model is None


def test_own_server_refuses_a_call_for_another_model():
    reader, ants = fl.load_models()
    srv = _Server()
    srv.start(reader.tag)
    q = ft.load_questions(DATA / "forecast_questions.jsonl", "forecast")[0]
    with pytest.raises(ProviderError, match="server holds"):
        asyncio.run(fl.OwnServer(_Fake(), srv).complete(fl.ant_request(q, ants[0])))


# ----------------------------------------------------------------- scoring

def _row(i, single, swarm, ants, group="politics"):
    return {"id": f"q{i}", "kind": "forecast", "group": group, "single": single, "swarm": swarm, "seen": swarm,
            "ants": ants}


def test_a_mixed_crowd_that_is_right_more_often_passes_test_b_only():
    rows, key, late = [], {}, []
    for i in range(60):
        truth = "yes" if i % 2 else "no"
        wrongish = 0.4 if truth == "yes" else 0.6        # the reader leans the wrong way
        good = 0.9 if truth == "yes" else 0.1            # the ants lean the right way
        rows.append(_row(i, [wrongish] * 10, [wrongish] * 5, {f: good for f in ("gemma", "qwen", "granite", "lfm")}))
        key[f"q{i}"] = truth
        if i >= 20:
            late.append(f"q{i}")
    r = fl.score(rows, key, late)
    assert r["test_b_mixed_crowd"]["n"] == 40 and r["test_b_mixed_crowd"]["verdict"] == "supported"
    assert r["test_a_lenses"]["verdict"] == "not supported" and r["test_a_lenses"]["gain"] == 0
    assert r["secondary"]["per_model"]["lfm"] < r["secondary"]["per_model"]["reader"]
    text = fl.block(r)
    assert text.startswith("FORECAST RESULT") and "Test B mixed crowd (n 40)" in text and "SUPPORTED" in text


def test_too_few_usable_answers_is_an_invalid_run():
    rows, key = [], {}
    for i in range(20):
        rows.append(_row(i, [None] * 10, [None] * 5, {"gemma": 0.9, "qwen": None, "granite": None, "lfm": None}))
        key[f"q{i}"] = "yes"
    r = fl.score(rows, key, [f"q{i}" for i in range(20)])
    assert r["test_a_lenses"]["verdict"].startswith("invalid run")
    assert r["test_b_mixed_crowd"]["verdict"].startswith("invalid run") and not r["test_b_mixed_crowd"]["pass"]


# ------------------------------------------------------------- end to end

def test_dry_run_end_to_end(tmp_path):
    cli = _script()
    out = tmp_path / "out"
    assert cli.main(["freeze", "--out", str(out)]) == 0
    assert cli.main(["run", "--dry-run", "--out", str(out), "--limit", "6"]) == 0
    assert cli.main(["score", "--dry-run", "--out", str(out)]) == 0
    result = json.loads((out / "run_dry" / "result.json").read_text())
    assert result["test_a_lenses"]["n"] == 6 and result["plan_changed"] == []


def test_run_refuses_a_changed_plan(tmp_path):
    cli = _script()
    out = tmp_path / "out"
    cli.main(["freeze", "--out", str(out)])
    frozen = json.loads((out / "prereg.json").read_text())
    frozen["late_from"] = "2026-07-01"
    (out / "prereg.json").write_text(json.dumps(frozen))
    with pytest.raises(SystemExit, match="plan changed"):
        cli.main(["run", "--dry-run", "--out", str(out), "--limit", "1"])


def test_live_run_against_a_fake_mlx_server(tmp_path, capsys):
    cli = _script()
    models = _models_file(tmp_path)
    reader, ants = fl.load_models(models)
    hub = _hub(tmp_path, [m.tag for m in (reader, *ants)])
    out = tmp_path / "out"
    port = str(_free_port())
    args = ["--out", str(out), "--models", str(models)]
    live = ["--mlx-server", str(_launcher(tmp_path)), "--hf-cache", str(hub), "--port", port, "--headroom-gb", "0"]
    assert cli.main(["freeze", *args]) == 0
    assert cli.main(["check", *args, *live]) == 0
    assert "CHECK PASSED" in capsys.readouterr().out
    assert cli.main(["run", "--live", *args, *live, "--limit", "2"]) == 0
    meta = json.loads((out / "run" / "meta.json").read_text())
    assert meta["loaded"] == [reader.tag, *(a.tag for a in ants)] and meta["failed"] == 0
    rows = [json.loads(x) for x in (out / "run" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(p is not None for r in rows for p in r["single"] + r["swarm"] + r["seen"])
    assert set(rows[0]["ants"]) == {"gemma", "qwen", "granite", "lfm"}
    cache = [json.loads(x) for x in (out / "run" / "cache.jsonl").read_text().splitlines()]
    assert all(c["served_model"] in {reader.tag, *(a.tag for a in ants)} for c in cache)
    assert fl.port_free(int(port))                           # the server it started is gone
    assert cli.main(["run", "--live", *args, *live, "--limit", "2"]) == 0
    assert json.loads((out / "run" / "meta.json").read_text())["loaded"] == []   # all cached: nothing loaded


def test_check_accepts_a_cache_anywhere_but_names_a_missing_folder(tmp_path, capsys):
    """The cache stays wherever it is (Eshaan keeps it on the external WD_BLACK drive). A folder that isn't there is
    named as such, not as a frozen drive."""
    cli = _script()
    models = _models_file(tmp_path)
    out = tmp_path / "out"
    args = ["--out", str(out), "--models", str(models)]
    cli.main(["freeze", *args])
    rc = cli.main(["check", *args, "--mlx-server", str(_launcher(tmp_path)), "--hf-cache", "/Volumes/WD_BLACK/hub",
                   "--port", str(_free_port()), "--headroom-gb", "0"])
    text = capsys.readouterr().out
    assert rc == 1 and "doesn't exist or can't be read" in text and "external" not in text


def test_check_names_the_models_missing_from_the_cache(tmp_path, capsys):
    cli = _script()
    models = _models_file(tmp_path)
    reader, ants = fl.load_models(models)
    hub = _hub(tmp_path, [reader.tag, ants[0].tag])
    out = tmp_path / "out"
    args = ["--out", str(out), "--models", str(models)]
    cli.main(["freeze", *args])
    rc = cli.main(["check", *args, "--mlx-server", str(_launcher(tmp_path)), "--hf-cache", str(hub),
                   "--port", str(_free_port()), "--headroom-gb", "0"])
    text = capsys.readouterr().out
    assert rc == 1 and f"not in {hub}" in text and ants[1].tag in text and reader.tag not in text.split("not in")[1]


def test_a_drive_that_answers_lists_and_one_that_hangs_is_abandoned(tmp_path, monkeypatch):
    assert fl.drive_problem(tmp_path) is None
    assert "doesn't exist" in fl.drive_problem(tmp_path / "nope")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hang = fake_bin / "ls"
    hang.write_text("#!/bin/sh\nexec sleep 30\n")         # a read stuck on a frozen drive
    hang.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    t0 = time.monotonic()
    why = fl.drive_problem(tmp_path, timeout_s=0.5)
    assert "didn't answer within" in why and time.monotonic() - t0 < 5


def test_a_drive_that_stops_answering_mid_run_stops_before_the_next_load(tmp_path, monkeypatch, capsys):
    cli = _script()
    models = _models_file(tmp_path)
    reader, ants = fl.load_models(models)
    hub = _hub(tmp_path, [m.tag for m in (reader, *ants)])
    out = tmp_path / "out"
    args = ["--out", str(out), "--models", str(models)]
    cli.main(["freeze", *args])
    answers = iter([None, "the model folder went away"])     # fine at the check, gone before the first load
    monkeypatch.setattr(fl, "drive_problem", lambda *_a, **_k: next(answers))
    rc = cli.main(["run", "--live", *args, "--mlx-server", str(_launcher(tmp_path)), "--hf-cache", str(hub),
                   "--port", str(_free_port()), "--headroom-gb", "0", "--limit", "1"])
    text = capsys.readouterr().out
    assert rc == 2 and f"before loading {reader.tag}: the model folder went away" in text


def test_a_server_that_dies_on_start_stops_the_run_cleanly(tmp_path, monkeypatch, capsys):
    cli = _script()
    models = _models_file(tmp_path)
    reader, ants = fl.load_models(models)
    hub = _hub(tmp_path, [m.tag for m in (reader, *ants)])
    out = tmp_path / "out"
    args = ["--out", str(out), "--models", str(models)]
    cli.main(["freeze", *args])
    monkeypatch.setenv("FAKE_MLX_EXIT", "7")
    rc = cli.main(["run", "--live", *args, "--mlx-server", str(_launcher(tmp_path)), "--hf-cache", str(hub),
                   "--port", str(_free_port()), "--headroom-gb", "0", "--limit", "1"])
    assert rc == 2 and "exited with code 7" in capsys.readouterr().out


def test_check_needs_the_server_and_the_cache_named(tmp_path, capsys):
    cli = _script()
    models = _models_file(tmp_path)
    out = tmp_path / "out"
    args = ["--out", str(out), "--models", str(models)]
    cli.main(["freeze", *args])
    assert cli.main(["check", *args, "--port", str(_free_port()), "--headroom-gb", "0"]) == 1
    text = capsys.readouterr().out
    assert "give --mlx-server" in text and "give --hf-cache" in text
