"""C-fix: the freeze guard has ONE enforcement point (``V43Harness.run_episode``) and its hash covers every input that changes results.

Each reviewer finding has its test here. Nothing runs on a real development case: a development-labelled phantom (under a real
development id) stands in, and every freeze is written over a scratch copy of the code in a temp dir (never into this repo).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from antelligence.experiments import v43_drive as drive
from antelligence.experiments import v43_freeze as fz
from antelligence.experiments.v42_harness import load_table
from antelligence.experiments.v43_freeze import (CELL_TABLE, COEFFICIENTS, SPLIT_FILE, FreezeError, check_freeze, freeze_hash,
                                                 require_frozen_run, write_freeze)
from antelligence.experiments.v43_harness import V43Harness
from antelligence.worlds.tumor_mri.combined import CombinedEstimate
from tests.v4_3.b1.helpers import phantom_with_channels
from tests.v4_3.c.helpers import REPO, dev_ids, make_scratch_repo, script

SHARING = ("S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect", "S-marks-quiet")


def flip_byte(path: Path) -> None:
    b = bytearray(path.read_bytes())
    b[0] ^= 0x01
    path.write_bytes(bytes(b))


def dev_case():
    """The phantom under a real development id and label: what a development case looks like to the harness."""
    return replace(phantom_with_channels(), case_id=dev_ids()["dev_a"][0], split="dev")


def phantom():
    return replace(phantom_with_channels(), case_id="phantom", split="synthetic")


def make_harness(*, level: str = "mid", world_edit=None, colony_edit=None, estimate=None, **kw) -> V43Harness:
    """A harness built the way the drivers build it (real config, real coefficients), optionally with one thing changed."""
    wc, cc, rules = drive.load_configs(drive.DEFAULT_CONFIG)
    wc = wc.with_(facts=replace(wc.facts, uptake_level=level))
    if world_edit is not None:
        wc = world_edit(wc)
    if colony_edit is not None:
        cc = colony_edit(cc)
    est = estimate if estimate is not None else CombinedEstimate.from_json(REPO / COEFFICIENTS)
    return V43Harness(wc, cc, rules, est, **kw)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A scratch copy of the covered files, committed, with a CLEAN freeze written over it; the module constants point at both."""
    scratch = make_scratch_repo(tmp_path / "scratch")
    freeze = tmp_path / "freeze.json"
    doc = write_freeze(freeze, scratch)
    assert doc["dirty"] is False
    monkeypatch.setattr(fz, "REPO_ROOT", scratch)
    monkeypatch.setattr(fz, "FREEZE_FILE", freeze)
    return {"root": scratch, "file": freeze, "doc": doc}


@pytest.fixture
def no_world(monkeypatch):
    """Make any attempt to build a world fail: proves a refusal happens BEFORE the world is built."""
    def boom(self, *a, **k):
        raise AssertionError("the world was built before the freeze check")
    monkeypatch.setattr(V43Harness, "build_world", boom)


# --- defect 1: one enforcement point, inside the harness ---------------------------------------------------------------------

@pytest.mark.parametrize("arm", SHARING)
def test_run_episode_refuses_every_sharing_arm_and_control_on_a_dev_case_without_a_freeze(arm, tmp_path, monkeypatch, no_world):
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "none.json")
    with pytest.raises(FreezeError, match="does not exist"):
        make_harness().run_episode(arm, dev_case(), 1)


def test_s_s_solo_and_s_prime_need_no_freeze_on_a_dev_case(tmp_path, monkeypatch):
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "none.json")
    h = make_harness()
    for arm in ("S", "S-solo", "S'"):
        assert h.run_episode(arm, dev_case(), 1)["arm"] == arm


def test_the_phantom_never_needs_the_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "none.json")
    h = make_harness()
    for arm in ("S-marks", "S-marks-displaced", "S-talk-shuffled"):
        assert h.run_episode(arm, phantom(), 1)["arm"] == arm


def test_a_synthetic_label_on_a_real_dev_id_is_no_way_round_the_freeze(tmp_path, monkeypatch, no_world):
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "none.json")
    mislabelled = replace(dev_case(), split="synthetic")
    with pytest.raises(FreezeError):
        make_harness().run_episode("S-marks", mislabelled, 1)


def test_with_a_valid_freeze_the_sharing_arm_and_the_controls_run(frozen):
    h = make_harness()
    for arm in ("S-marks-displaced", "S-marks", "S-talk-shuffled"):
        assert h.run_episode(arm, dev_case(), 1)["arm"] == arm
    for level in ("low", "high"):                         # any of the three uptake levels is the same frozen config
        assert make_harness(level=level).run_episode("S-marks", dev_case(), 1)["arm"] == "S-marks"
    assert check_freeze(frozen["file"], frozen["root"])[0]


def test_a_one_byte_change_in_antelligence_makes_the_harness_refuse(frozen, no_world):
    flip_byte(frozen["root"] / "antelligence" / "kernel" / "colony" / "episode.py")
    with pytest.raises(FreezeError, match="changed since the freeze"):
        make_harness().run_episode("S-marks", dev_case(), 1)


def test_a_freeze_file_set_on_the_harness_is_used_instead_of_the_constant(frozen, tmp_path, monkeypatch):
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "none.json")
    h = make_harness()
    h.freeze_file = frozen["file"]
    assert h.run_episode("S-marks", dev_case(), 1)["arm"] == "S-marks"


def test_the_drive_layer_still_fails_fast_and_passes_the_file_to_the_harness(frozen, phantom_loader, tmp_path):
    task = {"kind": "episode", "arm": "S-marks", "case_id": dev_ids()["dev_a"][0], "seed": 5, "config": str(drive.DEFAULT_CONFIG),
            "freeze_file": str(frozen["file"]), "uptake_level": "mid"}
    assert drive.run_task(task)["arm"] == "S-marks"
    flip_byte(frozen["root"] / "antelligence" / "kernel" / "colony" / "episode.py")
    with pytest.raises(FreezeError, match="changed since the freeze"):
        drive.run_task(task)
    assert drive.run_task({**task, "arm": "S-solo"})["arm"] == "S-solo"
    with pytest.raises(FreezeError):
        drive.run_task({**task, "freeze_file": str(tmp_path / "none.json")})


def test_v43_run_refuses_a_sharing_arm_on_a_dev_case_before_loading_a_case(tmp_path, monkeypatch, capsys):
    run = script("v43_run")
    loaded = []
    monkeypatch.setattr(run, "load_dev_case", lambda cid, root=None: loaded.append(cid) or dev_case())
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "none.json")
    case = dev_ids()["dev_a"][0]
    for arms in ("S-marks", "S-marks,S-marks-displaced", "S-solo,S-talk-shuffled"):
        assert run.main(["--out", str(tmp_path / "o"), "--cases", case, "--arms", arms]) == 2
    assert run.main(["--out", str(tmp_path / "o"), "--cases", case, "--arms", "S-marks", "--freeze-file", str(tmp_path / "x.json")]) == 2
    assert loaded == [] and not (tmp_path / "o").exists() and "does not exist" in capsys.readouterr().err
    # S-solo (and S, S') do not need the freeze: the case is loaded and the run goes through
    assert run.main(["--out", str(tmp_path / "ok"), "--cases", case, "--arms", "S-solo,S,S-prime", "--seeds", "1"]) == 0
    assert loaded and (tmp_path / "ok" / "S-solo" / case / "seed1" / "record.json").is_file()


def test_v43_run_with_a_config_the_hash_never_covered_is_refused_by_the_harness(frozen, tmp_path, monkeypatch, capsys):
    """Finding 3: the script's own up-front check passes (valid freeze) but ``--config`` points at a config the freeze did not
    cover; the harness check catches it per episode."""
    run = script("v43_run")
    monkeypatch.setattr(run, "load_dev_case", lambda cid, root=None: dev_case())
    case = dev_ids()["dev_a"][0]
    other = tmp_path / "world.yaml"                      # same file, one constant moved
    other.write_text((REPO / "config/v4_3/world.yaml").read_text().replace("path_weight: 1.5", "path_weight: 1.6"))
    base = ["--out", str(tmp_path / "o"), "--cases", case, "--arms", "S-marks", "--seeds", "1", "--freeze-file", str(frozen["file"])]
    assert run.main([*base, "--config", str(other)]) == 2
    assert "allowed_world_cfg_sha" in capsys.readouterr().err and not (tmp_path / "o").exists()
    assert run.main(base) == 0                           # the frozen config itself runs
    assert run.main([*base, "--ticks", "30", "--out", str(tmp_path / "t")]) == 2        # a shortened run is a different world
    old = REPO / "config" / "v4_2" / "world.yaml"        # read-only: the v4.2 world
    assert run.main([*base, "--config", str(old), "--out", str(tmp_path / "v42")]) == 2


# --- defect 2/4: the hash covers every input that affects results ----------------------------------------------------------

def test_editing_the_cell_table_changes_the_hash(frozen):
    h = freeze_hash(frozen["root"])
    flip_byte(frozen["root"] / CELL_TABLE)
    assert freeze_hash(frozen["root"]) != h
    assert not check_freeze(frozen["file"], frozen["root"])[0]


def test_editing_the_coefficients_changes_the_hash(frozen):
    h = freeze_hash(frozen["root"])
    flip_byte(frozen["root"] / COEFFICIENTS)
    assert freeze_hash(frozen["root"]) != h


def test_editing_the_split_file_changes_the_hash(frozen):
    h = freeze_hash(frozen["root"])
    flip_byte(frozen["root"] / SPLIT_FILE)
    assert freeze_hash(frozen["root"]) != h


def test_an_untracked_new_file_under_antelligence_changes_the_hash(frozen):
    h = freeze_hash(frozen["root"])
    (frozen["root"] / "antelligence" / "untracked_new.py").write_text("x = 1\n")
    assert freeze_hash(frozen["root"]) != h
    ok, why = check_freeze(frozen["file"], frozen["root"])
    assert not ok and "changed since the freeze" in why
    for rel in ("config/v4_3/extra.yaml", "scripts/v43_new.py"):
        h2 = freeze_hash(frozen["root"])
        (frozen["root"] / rel).write_text("x\n")
        assert freeze_hash(frozen["root"]) != h2, rel


def test_pyc_files_and_pycache_do_not_change_the_hash(frozen):
    h = freeze_hash(frozen["root"])
    (frozen["root"] / "antelligence" / "__pycache__").mkdir()
    (frozen["root"] / "antelligence" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\0")
    (frozen["root"] / "antelligence" / "stray.pyc").write_bytes(b"\0")
    assert freeze_hash(frozen["root"]) == h


def test_a_gitignored_file_is_not_hashed(frozen):
    (frozen["root"] / ".gitignore").write_text("antelligence/scratch_ignored.py\n")
    h = freeze_hash(frozen["root"])
    (frozen["root"] / "antelligence" / "scratch_ignored.py").write_text("x\n")
    assert freeze_hash(frozen["root"]) == h


@pytest.mark.parametrize("rel", [CELL_TABLE, COEFFICIENTS, SPLIT_FILE])
def test_a_missing_explicit_file_is_an_error(frozen, rel):
    (frozen["root"] / rel).unlink()
    with pytest.raises(FreezeError, match="missing"):
        freeze_hash(frozen["root"])
    ok, why = check_freeze(frozen["file"], frozen["root"])
    assert not ok and "missing" in why


def test_an_untracked_file_makes_the_tree_dirty_for_the_writer(frozen, tmp_path):
    (frozen["root"] / "antelligence" / "untracked_new.py").write_text("x = 1\n")
    assert fz.uncommitted(frozen["root"]) == ["antelligence/untracked_new.py"]
    with pytest.raises(FreezeError, match="uncommitted"):
        write_freeze(tmp_path / "f2.json", frozen["root"])


# --- defect 2/3: the harness is checked by CONTENT, not by path -------------------------------------------------------------

def test_the_tree_hash_is_unchanged_in_every_case_below(frozen):
    assert check_freeze(frozen["file"], frozen["root"])[0]


def test_a_harness_with_a_different_config_is_refused(frozen, no_world):
    wc4, cc4, rules4 = drive.load_configs(REPO / "config" / "v4_2" / "world.yaml")      # read-only
    h = V43Harness(wc4, cc4, rules4, None)
    with pytest.raises(FreezeError, match="rules_sha"):
        h.run_episode("S-marks-displaced", dev_case(), 1)
    # the same config with one world constant moved, and with one colony constant moved
    with pytest.raises(FreezeError, match="allowed_world_cfg_sha"):
        make_harness(world_edit=lambda w: w.with_(path_weight=2.0)).run_episode("S-marks", dev_case(), 1)
    with pytest.raises(FreezeError, match="allowed_world_cfg_sha"):
        make_harness(world_edit=lambda w: w.with_(ticks=60)).run_episode("S-marks", dev_case(), 1)
    with pytest.raises(FreezeError, match="allowed_world_cfg_sha"):
        make_harness(world_edit=lambda w: w.with_(facts=replace(w.facts, f1_simlook=False))).run_episode("S-marks", dev_case(), 1)
    with pytest.raises(FreezeError, match="colony_cfg_sha"):
        make_harness(colony_edit=lambda c: replace(c, heartbeat_rounds=c.heartbeat_rounds + 1)).run_episode("S-marks", dev_case(), 1)
    assert check_freeze(frozen["file"], frozen["root"])[0]


def test_a_harness_with_different_coefficients_is_refused(frozen, no_world):
    doc = json.loads((REPO / COEFFICIENTS).read_text())
    doc["coef"] = [c + 0.01 for c in doc["coef"]]
    with pytest.raises(FreezeError, match="estimate_sha"):
        make_harness(estimate=CombinedEstimate.from_dict(doc)).run_episode("S-marks", dev_case(), 1)
    wc, cc, rules = drive.load_configs(drive.DEFAULT_CONFIG)
    assert rules.use_combined
    # the v4.3 ant cannot even be built without an estimate; a v4.2-rules harness has none and is refused on its rules
    assert check_freeze(frozen["file"], frozen["root"])[0]


def test_a_harness_with_a_different_cell_table_is_refused_by_content(frozen, tmp_path, no_world):
    table = json.loads((REPO / CELL_TABLE).read_text())
    edited = {**table, "extra": 1}
    with pytest.raises(FreezeError, match="cell_table_sha256"):
        make_harness(cell_table=edited).run_episode("S-marks", dev_case(), 1)
    path = tmp_path / "cells_other.json"                  # a different PATH holding different content
    path.write_text(json.dumps(edited))
    with pytest.raises(FreezeError, match="cell_table_sha256"):
        make_harness(cell_table_path=path).run_episode("S-marks", dev_case(), 1)
    assert check_freeze(frozen["file"], frozen["root"])[0]


def test_the_same_cell_table_by_another_path_or_in_memory_is_accepted(frozen, tmp_path):
    path = tmp_path / "copy.json"
    path.write_text((REPO / CELL_TABLE).read_text())      # identical content, another path
    assert make_harness(cell_table_path=path).run_episode("S-marks", dev_case(), 1)["arm"] == "S-marks"
    assert make_harness(cell_table=load_table(REPO / CELL_TABLE)).run_episode("S-marks", dev_case(), 1)["arm"] == "S-marks"


def test_an_injected_look_fn_factory_is_refused(frozen, no_world):
    with pytest.raises(FreezeError, match="look_fn_factory"):
        make_harness(look_fn_factory=lambda case: None).run_episode("S-marks", dev_case(), 1)


def test_a_dirty_recorded_or_old_style_freeze_is_refused_by_the_harness(frozen, tmp_path, no_world):
    doc = json.loads(frozen["file"].read_text())
    dirty = tmp_path / "dirty.json"
    dirty.write_text(json.dumps({**doc, "dirty": True}))
    h = make_harness()
    h.freeze_file = dirty
    with pytest.raises(FreezeError, match="dirty"):
        h.run_episode("S-marks", dev_case(), 1)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({k: v for k, v in doc.items() if k not in ("rules_sha", "cell_table_sha256")}))
    h.freeze_file = old
    with pytest.raises(FreezeError, match="rules_sha"):
        h.run_episode("S-marks", dev_case(), 1)
    with pytest.raises(FreezeError, match="dirty"):
        require_frozen_run(h, path=dirty)


def test_the_frozen_hashes_match_the_harness_the_drivers_build(frozen):
    """The three allowed world hashes are exactly the harness's at low / mid / high, so a verdict run at any level passes."""
    doc = frozen["doc"]
    for level in ("low", "mid", "high"):
        live = fz.live_hashes(make_harness(level=level), frozen["root"])
        assert live["world_cfg_sha"] == doc["allowed_world_cfg_sha"][level]
        assert live["colony_cfg_sha"] == doc["colony_cfg_sha"] and live["rules_sha"] == doc["rules_sha"]
        assert live["estimate_sha"] == doc["estimate_sha"]
    assert doc["cell_table_sha256"] == fz._file_sha(REPO / CELL_TABLE)


# --- defects 5, 6, 7 ---------------------------------------------------------------------------------------------------------

def test_replicate_cannot_lower_the_pair_requirement_from_the_command_line(tmp_path, capsys):
    rep = script("v43_replicate")
    with pytest.raises(SystemExit) as exc:
        rep.main(["--results", str(tmp_path), "--arm", "S-marks", "--expected-pairs", "4"])
    assert exc.value.code == 2 and "--expected-pairs" in capsys.readouterr().err
    from antelligence.experiments.v43_stats import REPLICATION_PAIRS
    assert REPLICATION_PAIRS == 80


def test_the_freeze_cli_documents_the_lead_dirty_rule_and_has_no_allow_dirty_at_check_time(capsys):
    cli = script("v43_freeze")
    doc = cli.__doc__
    assert "LEAD" in doc and '"dirty": true' in doc and "NO allow-dirty at check time" in doc
    with pytest.raises(SystemExit):
        cli.main(["check", "--allow-dirty"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(["write", "--help"])
    assert exc.value.code == 0 and '"dirty": true' in capsys.readouterr().out


def test_the_cli_refuses_a_dirty_recorded_freeze(frozen, tmp_path, capsys):
    cli = script("v43_freeze")
    (frozen["root"] / "antelligence" / "core_new.py").write_text("x = 1\n")
    dirty = tmp_path / "dirty.json"
    assert cli.main(["write", "--out", str(dirty), "--root", str(frozen["root"]), "--allow-dirty"]) == 0
    assert json.loads(dirty.read_text())["dirty"] is True
    assert cli.main(["check", "--freeze-file", str(dirty), "--root", str(frozen["root"])]) == 1
    assert "dirty" in capsys.readouterr().out


def test_v43_diagnose_imports_without_side_effects(tmp_path):
    """The v4.2 reader module it loads by path must not write anything at import: run the import in a subprocess from an empty
    directory and compare the v4.2 evidence folder before and after."""
    v42_docs = REPO / "docs" / "research" / "v4.2-20261014"

    def snapshot(root: Path):
        return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())

    before = snapshot(v42_docs)
    empty = tmp_path / "empty"
    empty.mkdir()
    code = ("import importlib.util, sys\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            f"spec = importlib.util.spec_from_file_location('v43_diagnose', {str(REPO / 'scripts' / 'v43_diagnose.py')!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "assert callable(mod.main) and mod.R is not None\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=empty, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stderr
    assert list(empty.iterdir()) == []                    # nothing appeared in the working directory
    assert snapshot(v42_docs) == before                   # nothing written under docs/research/v4.2-20261014
