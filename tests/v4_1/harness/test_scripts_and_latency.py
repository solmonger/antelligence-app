"""scripts/v41_run.py and scripts/v41_latency.py on fakes, and the latency helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from antelligence.experiments import v41_latency
from antelligence.experiments.v41_gate import GateError
from antelligence.experiments.v41_harness import D1, Q4, Q9
from antelligence.experiments.v41_live import LiveConfigError, build_workers, resolve_loader
from antelligence.providers.decision_backends import CachedDecision
from tests.v4_1.harness.conftest import CASE_DEV, CASE_EVAL, build_rig, fake_workers, make_loader

ROOT = Path(__file__).resolve().parents[3]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_script_writes_cells_and_reports_failures(tmp_path):
    run = script("v41_run")
    rig = build_rig(tmp_path, gate=lambda c: "dev" if c == CASE_DEV else (_ for _ in ()).throw(GateError("no G1")))
    cells = tmp_path / "cells.jsonl"
    code = run.main(["--config", "unused.json", "--out-dir", str(tmp_path / "o"), "--cells", str(cells),
                     "--run", f"SW-B:{CASE_DEV}:1", "--run", f"S:{CASE_DEV}:1", "--run", f"SW-B:{CASE_EVAL}:1"],
                    harness=rig.harness)
    rows = [json.loads(line) for line in cells.read_text().splitlines()]
    assert code == 1 and [r["arm"] for r in rows] == ["SW-B", "S", "SW-B"]
    assert "error" not in rows[0] and "no G1" in rows[2]["error"]
    from antelligence.experiments.v41_cells import validate_cells
    assert validate_cells(rows) == []


def test_run_script_rejects_unknown_arm(tmp_path):
    run = script("v41_run")
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    with pytest.raises(SystemExit):
        run.main(["--config", str(cfg), "--out-dir", str(tmp_path), "--run", f"SW-X:{CASE_DEV}:1"])


def test_harvest_requests_and_bench_backend(tmp_path):
    workers = fake_workers(latency_s=0.001)
    reqs = v41_latency.harvest_requests(make_loader(), [CASE_DEV], workers[D1].model, 40)
    assert len(reqs) == 40 and all(r.model == workers[D1].model for r in reqs)
    with pytest.raises(ValueError):
        v41_latency.harvest_requests(make_loader(), [CASE_DEV], "m", 10_000)
    out = {c: asyncio.run(v41_latency.bench_backend(workers[D1].backend, reqs, c)) for c in (1, 8)}
    assert out[1]["ok"] == out[8]["ok"] == 40 and out[8]["wall_s"] < out[1]["wall_s"]
    assert out[1]["elapsed_s_mean"] > 0


def test_board_mode_overhead_and_anchor_latency(tmp_path):
    modes = v41_latency.bench_board_modes(60, repeats=1)
    assert modes["B_ms_per_entry"] > modes["A_ms_per_entry"] > 0
    from antelligence.anchoring.tamper import Lab
    lab = Lab()
    lat = v41_latency.bench_anchors(lambda b, k, d: lab.anchorer(b, k, d), 120)
    assert {m: lat[m]["n"] for m in lat} == {"OTS": 3, "R": 3, "C": 3}


def test_latency_script_end_to_end_on_fakes(tmp_path):
    lat = script("v41_latency")
    out = tmp_path / "lat" / "results.json"
    code = lat.main(["--n-cases", "1", "--n", "16", "--out", str(out), "--anchors", "mock"],
                    workers={k: v for k, v in fake_workers().items() if k in (D1, Q4, Q9)}, loader=make_loader())
    res = json.loads(out.read_text())
    assert code == 0 and res["split"] == "dev" and set(res["backends"]) == {D1, Q4, Q9}
    assert set(res["backends"][D1]) == {"c1", "c8"} and res["board_modes"]["n_entries"] and res["anchors"]["OTS"]["n"] == 3


def test_live_builders_use_named_env_vars_only(tmp_path):
    cfg = {"cache_dir": str(tmp_path / "cache"), "d1": {"url": "http://127.0.0.1:1"},
           "qwen3.5-9b": {"url": "http://127.0.0.1:2", "hf_id": "org/q9", "revision": "r9"},
           "fw": {"api_key_env": "SOME_KEY"}}
    workers = build_workers(cfg, {"SOME_KEY": "dummy"}.get)
    assert set(workers) == {"d1", "qwen3.5-9b", "fw"} and isinstance(workers["d1"].backend, CachedDecision)
    assert workers["d1"].model and workers["qwen3.5-9b"].model.startswith("org/q9")
    with pytest.raises(LiveConfigError):
        build_workers(cfg, {}.get)
    with pytest.raises(LiveConfigError):
        resolve_loader("no.such.module:load_case")
