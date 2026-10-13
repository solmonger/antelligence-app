"""S9 dry plumbing run: --backends fake, --router none, mock-anchored verification, and the real-data bugs it found."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments import v41_cells, v41_fake
from antelligence.experiments.v41_harness import ARM_SPECS, D1, Q4, Q9
from antelligence.kernel.decision import Question, DecisionRequest
from tests.v4_1.harness.conftest import build_rig, make_loader
from tests.v4_1.harness.test_harness import run
from tests.v4_1.world.test_store_adapter import _write_case

ROOT = Path(__file__).resolve().parents[3]
DEV_CASE = "BRATS_016"      # dev_a[0]; the phantom loader stands in for the store


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


phantom_loader = make_loader()


def _request(state="a region", seed=0, model="m"):
    qs = {"look": Question("noul", "is it tumour?"),
          "target": Question("choice", "which region?", ("A", "B", "C")),
          "amount": Question("score", "how much?", ("none", "low", "mid", "high"))}
    return DecisionRequest.build(model, state, qs, seed)


def test_fake_backends_answer_real_questions_and_vary_with_state():
    workers = v41_fake.build_fake_workers()
    assert set(workers) == {"d1", "qwen3.5-4b", "qwen3.5-9b", "bonsai-8b", "fw"}
    outs = []
    for n in range(12):
        resp = asyncio.run(workers[Q4].backend.decide(_request(f"state {n}", seed=n)))
        assert resp.identity["served_model"] == "fake" and resp.prob_method == "fake"
        outs.append((resp.answers["look"].value, resp.answers["target"].value, max(resp.answers["target"].probabilities.values())))
    assert len({o[0] for o in outs}) > 6 and len({o[1] for o in outs}) > 1           # varies with state
    assert min(o[2] for o in outs) < 0.7 < max(o[2] for o in outs)                    # both sides of slm_conf_min
    assert asyncio.run(workers[D1].backend.decide(_request("s"))).answers["look"].value == \
        asyncio.run(workers[D1].backend.decide(_request("s"))).answers["look"].value  # deterministic


def test_fake_backend_cites_only_brief_ids():
    import dataclasses
    req = dataclasses.replace(_request(), brief_ids=tuple(f"id{i}" for i in range(20)))
    resp = asyncio.run(v41_fake.build_fake_workers()[Q9].backend.decide(req))
    assert resp.cites and set(resp.cites) <= set(req.brief_ids)


def test_null_router_sends_everything_to_the_9b_and_never_defers():
    router, thresholds = v41_fake.null_bootstrap()
    assert router.predict([0.0]) == Q9 and tuple(router.models) == ("d1", "qwen3.5-4b", "qwen3.5-9b")
    brouter, _ = v41_fake.null_bootstrap(bonsai=True)
    assert brouter.predict([0.0]) == "bonsai-8b"
    assert thresholds.qhat("target") == 1.0


def test_fake_dry_run_cli_every_arm_with_mock_verifier(tmp_path):
    """--backends fake --anchors mock --router none (and fake) over every arm: cells validate, the mock-anchored exports
    verify, SW-D goes through the real BoardLog encoder, and the run log records identity 'fake'."""
    mod = script("v41_run")
    out, cells = tmp_path / "o", tmp_path / "cells.jsonl"
    argv = ["--backends", "fake", "--anchors", "mock", "--router", "none", "--out-dir", str(out), "--cells", str(cells),
            "--loader", f"{__name__}:phantom_loader", "--world-yaml", str(tmp_path / "w.yaml")]
    (tmp_path / "w.yaml").write_text(json.dumps(_phantom_world().to_dict()))
    code = mod.main(argv + [a for arm in ARM_SPECS for a in ("--run", f"{arm}:{DEV_CASE}:1")])
    rows = [json.loads(line) for line in cells.read_text().splitlines()]
    assert code == 0 and [r["arm"] for r in rows] == list(ARM_SPECS)
    assert all("error" not in r for r in rows) and v41_cells.validate_cells(rows) == []
    report = json.loads((out / "verify-mock-report.json").read_text())
    assert report["verdict"] == "verified" and report["study"]["ok"], report["problems"]
    by = {r["arm"]: r for r in rows}
    assert by["SW-D"]["visibility_latency_ms"] is not None and by["SW-B"]["anchors"] > 0
    logs = [json.loads(p.read_text()) for p in (out / "runs").glob("*/run_log.json")]
    assert {l["arm"] for l in logs if l["decisions"]} >= {"SW-B", "SW-D", "FW"}
    assert all(l["identities"] in ([], ["fake"]) for l in logs)
    assert all(not l["errors"] for l in logs)


def _phantom_world():
    from tests.v4_1.harness.conftest import small_cfg
    return small_cfg()


def test_fake_router_exercises_routing_deferral_and_escalation(tmp_path):
    mod = script("v41_run")
    out = tmp_path / "o"
    (tmp_path / "w.yaml").write_text(json.dumps(_phantom_world().to_dict()))
    mod.main(["--backends", "fake", "--anchors", "mock", "--router", "fake", "--out-dir", str(out), "--loader",
              f"{__name__}:phantom_loader", "--world-yaml", str(tmp_path / "w.yaml"), "--run", f"SW-B:{DEV_CASE}:1"])
    rows = [json.loads(line) for p in (out / "runs").glob("*/decisions.jsonl") for line in p.read_text().splitlines()]
    assert {r["router_pick"] for r in rows} >= {"d1", "qwen3.5-4b"}
    assert sum(r["deferred"] for r in rows) > 0 and sum(r["escalated"] for r in rows) > 0


def test_fake_router_refused_without_fake_backends(tmp_path):
    mod = script("v41_run")
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    with pytest.raises(SystemExit):
        mod.main(["--config", str(cfg), "--router", "fake", "--out-dir", str(tmp_path), "--run", f"S:{DEV_CASE}:1"])


# --- real-data bugs ------------------------------------------------------------------------------------------------

def test_harness_loader_returns_crop_and_regions_and_serves_the_missing_flair_store(tmp_path):
    """Bug 1: the default --loader returned a LoadedCase (harness unpacks a tuple) and had no variant argument."""
    from antelligence.worlds.tumor_mri.store_adapter import make_harness_loader
    from antelligence.worlds.tumor_mri.data import Crop
    full, mf = tmp_path / "cases", tmp_path / "cases_missing_flair"
    _write_case(full, DEV_CASE, outside=17)
    _write_case(mf, DEV_CASE, outside=17)
    a = np.load(mf / DEV_CASE / "case.npz")
    np.savez(mf / DEV_CASE / "case.npz", **{**{k: a[k] for k in a.files}, "flair": np.zeros_like(a["flair"])})
    load = make_harness_loader(root=full)
    crop, regions = load(DEV_CASE)
    assert isinstance(crop, Crop) and regions and crop.flair.any()
    crop_mf, _ = load(DEV_CASE, variant="missing_flair")
    assert not crop_mf.flair.any()
    from antelligence.worlds.tumor_mri.store_adapter import AdapterError
    with pytest.raises(AdapterError):
        load(DEV_CASE, variant="nope")


def test_mode_d_encoder_importable_without_backend_on_sys_path(monkeypatch):
    """Bug 2: compact_entry did `from chain.v4_1 import ...` and failed with ModuleNotFoundError unless a live run had
    already put backend/ on sys.path (a mock-anchored SW-D died)."""
    from antelligence.anchoring.chain_modes import compact_entry
    monkeypatch.setattr(sys, "path", [p for p in sys.path if not p.rstrip("/").endswith("backend")])
    for name in [m for m in sys.modules if m == "chain" or m.startswith("chain.")]:
        monkeypatch.delitem(sys.modules, name)
    from antelligence.kernel.board import Board
    from antelligence.anchoring import mocks
    board, _, _ = mocks.synthetic_run(5, seed=1)
    assert len(compact_entry(board.entries()[0])) > 0


def test_anchored_run_exports_the_public_rekor_key_beside_the_board(tmp_path):
    """Bug 3: the verifier needs <run_id>.anchor_pub.pem next to the export; the harness never wrote it."""
    res = run(build_rig(tmp_path), "SW-B")
    pem = Path(res.run_dir) / f"{res.cell['run_id']}.anchor_pub.pem"
    assert pem.is_file() and pem.read_bytes().startswith(b"-----BEGIN PUBLIC KEY")


def test_mock_post_mode_enforces_contiguous_batches_and_encodes_entries():
    from antelligence.anchoring import mocks
    board, _, _ = mocks.synthetic_run(12, seed=2)
    pm = v41_fake.MockPostMode()
    entries = board.entries()
    r = pm.post_round("r", entries[:5])
    assert r.visible == set(range(5)) and not r.timed_out
    with pytest.raises(ValueError):
        pm.post_round("r", entries[7:9])


def test_mock_reputation_client_is_idempotent():
    c = v41_fake.MockReputationClient()
    c.enqueue_report("r", "a" * 64, "c" * 64, True)
    c.enqueue_report("r", "a" * 64, "c" * 64, False)
    assert c.reports == {("r", "a" * 64, "c" * 64): True}


# --- fake isolation (reviewer requirement) --------------------------------------------------------------------------

from antelligence.experiments.v41_analysis import IsolationError, analyze  # noqa: E402
from antelligence.experiments.v41_harness import GateConfig, HarnessError, make_gate  # noqa: E402
from tests.v4_1.analysis.conftest import make_cell, synth  # noqa: E402
from tests.v4_1.analysis.test_split_gate import _commit, world  # noqa: E402,F401


def _locked_rig(tmp_path, w, **kw):
    """A harness whose gate is the real one with a committed prereg, confirmed receipt and G1 file: eval is OPEN."""
    _commit(w)
    gate = make_gate(GateConfig(w["repo"], w["prereg"], w["receipt"], w["g1"]))
    return build_rig(tmp_path, gate=gate, **kw)


def test_eval_case_runs_with_live_style_harness_when_the_lock_is_present(tmp_path, world):  # noqa: F811
    eval_case = world["doc"]["eval"][0]
    res = run(_locked_rig(tmp_path, world, backends_kind="live"), "SW-B", case=eval_case)
    c = res.cell
    assert (c["backends_kind"], c["router_kind"]) == ("live", "frozen") and len(c["router_sha256"]) == 64
    assert v41_cells.validate_cell(c) == []
    assert json.loads((Path(res.run_dir) / "run_log.json").read_text())["router_kind"] == "frozen"


def test_fake_backends_refused_on_eval_even_with_lock_files(tmp_path, world):  # noqa: F811
    rig = _locked_rig(tmp_path, world, backends_kind="fake")
    with pytest.raises(HarnessError, match="fake backends"):
        run(rig, "SW-B", case=world["doc"]["eval"][0])
    with pytest.raises(HarnessError, match="fake backends"):
        run(rig, "S", case=world["doc"]["eval"][0])
    assert run(rig, "SW-B", case=world["doc"]["dev_a"][0]).cell["backends_kind"] == "fake"


@pytest.mark.parametrize("kind", ["none", "fake"])
def test_bootstrap_and_fake_routers_refused_on_eval_allowed_on_dev(tmp_path, world, kind):  # noqa: F811
    rig = _locked_rig(tmp_path, world, backends_kind="live")
    router, thr = v41_fake.null_bootstrap() if kind == "none" else v41_fake.fake_bootstrap()
    rig.harness.router, rig.harness.thresholds = router, thr
    with pytest.raises(HarnessError, match=f"router '{kind}'"):
        run(rig, "SW-B", case=world["doc"]["eval"][0])
    dev = run(rig, "SW-B", case=world["doc"]["dev_a"][0])
    assert dev.cell["router_kind"] == kind and dev.cell["router_sha256"] is None
    assert v41_cells.validate_cell(dev.cell) == []


def test_cli_fake_flags_are_refused_on_a_non_dev_case(tmp_path):
    mod = script("v41_run")
    (tmp_path / "w.yaml").write_text(json.dumps(_phantom_world().to_dict()))
    cells = tmp_path / "c.jsonl"
    code = mod.main(["--backends", "fake", "--anchors", "mock", "--router", "none", "--out-dir", str(tmp_path / "o"),
                     "--cells", str(cells), "--loader", f"{__name__}:phantom_loader", "--world-yaml", str(tmp_path / "w.yaml"),
                     "--run", "SW-B:BRATS_001:1"])      # not in the committed split: counts as non-dev
    row = json.loads(cells.read_text())
    assert code == 1 and "error" in row and row["backends_kind"] == "fake"


def test_validator_rules_for_kinds():
    ok = make_cell("SW-B", "C1", 1, 0.5)
    assert v41_cells.validate_cell(ok) == []
    assert v41_cells.validate_cell({**ok, "backends_kind": "mock"})
    assert v41_cells.validate_cell({**ok, "router_sha256": None})                       # frozen needs a hash
    assert v41_cells.validate_cell({**ok, "router_kind": "none", "router_sha256": "a" * 64})


def _with(cells, i, **kw):
    cells = list(cells)
    cells[i] = {**cells[i], **kw}
    return cells


@pytest.mark.parametrize("bad", [dict(backends_kind="fake"), dict(backends_kind="config"), dict(router_kind="none", router_sha256=None),
                                 dict(router_kind="fake", router_sha256=None), dict(identities="fake"),
                                 dict(identities="d1,fake"), dict(backends_kind=None)])
def test_eval_analysis_rejects_non_live_cells(bad):
    cells = synth(n_cases=12, seeds=1)
    assert analyze(cells)["n_cells"] == len(cells)
    with pytest.raises(IsolationError):
        analyze(_with(cells, 3, **bad))


def test_eval_analysis_rejects_fake_dev_cells_but_accepts_calibration_dev_cells():
    ev = synth(n_cases=12, seeds=1)
    dev = synth(n_cases=12, seeds=1, split="dev")
    boot = [{**c, "router_kind": "none", "router_sha256": None} for c in dev]          # Phase 0 calibration cells
    assert analyze(ev + boot)["n_cells"] == len(ev)
    with pytest.raises(IsolationError):
        analyze(ev + _with(dev, 0, backends_kind="fake"))


def test_dev_path_reads_router_none_and_fake_only_when_asked():
    dev = synth(n_cases=12, seeds=1, split="dev")
    boot = [{**c, "router_kind": "none", "router_sha256": None} for c in dev]
    assert analyze(boot, split="dev")["n_cells"] == len(boot)
    fake = [{**c, "backends_kind": "fake", "router_kind": "fake", "router_sha256": None, "identities": "fake"} for c in dev]
    with pytest.raises(IsolationError):
        analyze(fake, split="dev")
    assert analyze(fake, split="dev", allow_fake=True)["n_cells"] == len(fake)
    with pytest.raises(IsolationError):
        analyze([{**c, "split": "eval"} for c in fake], split="eval", allow_fake=True)


def test_fake_dry_run_cells_carry_fake_stamps_and_are_rejected_by_eval_analysis(tmp_path):
    mod = script("v41_run")
    (tmp_path / "w.yaml").write_text(json.dumps(_phantom_world().to_dict()))
    cells = tmp_path / "c.jsonl"
    mod.main(["--backends", "fake", "--anchors", "mock", "--router", "none", "--out-dir", str(tmp_path / "o"), "--cells", str(cells),
              "--loader", f"{__name__}:phantom_loader", "--world-yaml", str(tmp_path / "w.yaml"),
              "--run", f"SW-B:{DEV_CASE}:1", "--run", f"S:{DEV_CASE}:1"])
    rows = [json.loads(l) for l in cells.read_text().splitlines()]
    assert {r["backends_kind"] for r in rows} == {"fake"} and {r["router_kind"] for r in rows} == {"none"}
    assert rows[0]["identities"] == "fake" and rows[1]["identities"] is None
    assert v41_cells.isolation_problems(rows, allow_router_none=True)       # fake is never accepted unless asked
    with pytest.raises(IsolationError):
        analyze([{**r, "split": "eval"} for r in rows])
