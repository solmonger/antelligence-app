"""C: the freeze guard. Written against a throwaway git repo, so no tracked file of this repo is ever touched."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from antelligence.experiments import v43_freeze as fz
from antelligence.experiments.v43_freeze import (FreezeRefused, check_freeze, config_hashes, freeze_hash, freeze_needed, require_frozen,
                                                 tracked_files, uncommitted, write_freeze)
from tests.v4_3.c.helpers import REPO, script

CLI = script("v43_freeze")


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    for rel, text in {"antelligence/core.py": "x = 1\n", "antelligence/sub/more.py": "y = 2\n", "config/v4_3/world.yaml": "world: {}\n",
                      "config/v4_2/world.yaml": "old: true\n", "scripts/v43_run.py": "print('v43')\n", "scripts/v42_run.py": "print('v42')\n",
                      "scripts/other.py": "print('o')\n", "tests/t.py": "pass\n", "docs/readme.md": "hi\n"}.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "init")
    return r


def flip_byte(path: Path) -> None:
    b = bytearray(path.read_bytes())
    b[0] ^= 0x01
    path.write_bytes(bytes(b))


def test_tracked_set_is_antelligence_config_v4_3_and_v43_scripts_only(repo):
    assert tracked_files(repo) == ["antelligence/core.py", "antelligence/sub/more.py", "config/v4_3/world.yaml", "scripts/v43_run.py"]


def test_freeze_hash_is_stable_and_ignores_files_outside_the_set(repo):
    h = freeze_hash(repo)
    assert len(h) == 64 and freeze_hash(repo) == h
    for rel in ("scripts/other.py", "scripts/v42_run.py", "tests/t.py", "docs/readme.md", "config/v4_2/world.yaml"):
        (repo / rel).write_text("changed\n")
    (repo / "antelligence" / "untracked.py").write_text("not tracked\n")          # untracked files are not part of the freeze
    assert freeze_hash(repo) == h


@pytest.mark.parametrize("rel", ["antelligence/core.py", "antelligence/sub/more.py", "config/v4_3/world.yaml", "scripts/v43_run.py"])
def test_a_one_byte_change_in_any_frozen_file_changes_the_hash(repo, rel):
    h = freeze_hash(repo)
    flip_byte(repo / rel)
    assert freeze_hash(repo) != h


def test_a_deleted_tracked_file_changes_the_hash(repo):
    h = freeze_hash(repo)
    (repo / "antelligence" / "core.py").unlink()
    assert freeze_hash(repo) != h


def test_write_records_commit_hash_config_hashes_and_date(repo, tmp_path):
    out = tmp_path / "freeze.json"
    doc = write_freeze(out, repo)
    on_disk = json.loads(out.read_text())
    assert on_disk == doc
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert doc["commit"] == head and doc["freeze_hash"] == freeze_hash(repo)
    assert set(doc["config_sha256"]["files"]) == {"config/v4_3/world.yaml"} and len(doc["config_sha256"]["combined"]) == 64
    assert doc["config_sha256"] == config_hashes(repo) and doc["date"].endswith("Z") and doc["uncommitted"] == []


def test_check_passes_when_unchanged_and_fails_after_a_one_byte_change(repo, tmp_path):
    out = tmp_path / "freeze.json"
    write_freeze(out, repo)
    ok, why = check_freeze(out, repo)
    assert ok and "frozen at commit" in why
    assert require_frozen(out, repo)["freeze_hash"] == freeze_hash(repo)
    flip_byte(repo / "antelligence" / "core.py")
    ok, why = check_freeze(out, repo)
    assert not ok and "changed since the freeze" in why
    with pytest.raises(FreezeRefused):
        require_frozen(out, repo)
    flip_byte(repo / "antelligence" / "core.py")                                  # flip it back: the freeze holds again
    assert check_freeze(out, repo)[0]


def test_a_changed_config_is_named_in_the_refusal(repo, tmp_path):
    out = tmp_path / "freeze.json"
    write_freeze(out, repo)
    (repo / "config" / "v4_3" / "world.yaml").write_text("world: {k: 2}\n")
    ok, why = check_freeze(out, repo)
    assert not ok and "config/v4_3/world.yaml" in why


def test_missing_or_broken_freeze_files_refuse(repo, tmp_path):
    with pytest.raises(FreezeRefused, match="does not exist"):
        require_frozen(tmp_path / "none.json", repo)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(FreezeRefused, match="unreadable"):
        require_frozen(bad, repo)
    bad.write_text("{}")
    with pytest.raises(FreezeRefused, match="unreadable"):
        require_frozen(bad, repo)


def test_write_refuses_a_dirty_tree_unless_allowed(repo, tmp_path):
    out = tmp_path / "freeze.json"
    flip_byte(repo / "antelligence" / "core.py")
    assert uncommitted(repo) == ["antelligence/core.py"]
    with pytest.raises(FreezeRefused, match="uncommitted"):
        write_freeze(out, repo)
    assert not out.exists()
    doc = write_freeze(out, repo, allow_dirty=True)
    assert doc["uncommitted"] == ["antelligence/core.py"]


def test_which_arms_need_the_freeze():
    assert not freeze_needed(["S", "S-solo", "S'"])
    assert freeze_needed(["S", "S-solo", "S-marks"]) and freeze_needed(["S-marks-displaced"]) and freeze_needed(["S-talk-shuffled"])
    assert freeze_needed(["S-marks-perfect"]) and freeze_needed(["S-marks-quiet"]) and freeze_needed(["S-talk"])
    assert not freeze_needed(["S-marks"], on_dev_case=False)                 # a phantom-only run needs none


def test_the_cli_writes_checks_and_refuses(repo, tmp_path, capsys):
    out = tmp_path / "f.json"
    assert CLI.main(["write", "--out", str(out), "--root", str(repo)]) == 0
    assert CLI.main(["check", "--freeze-file", str(out), "--root", str(repo)]) == 0
    flip_byte(repo / "scripts" / "v43_run.py")
    assert CLI.main(["check", "--freeze-file", str(out), "--root", str(repo)]) == 1
    assert "MISMATCH" in capsys.readouterr().out
    assert CLI.main(["write", "--out", str(tmp_path / "g.json"), "--root", str(repo)]) == 2           # dirty tree
    assert CLI.main(["check", "--freeze-file", str(tmp_path / "none.json"), "--root", str(repo)]) == 1


def test_driver_refuses_after_a_one_byte_change_in_antelligence(repo, tmp_path, monkeypatch, phantom_loader):
    """The drive layer asks ``require_frozen`` before every sharing-arm or control task on a development case. Pointed at the
    throwaway repo, a one-byte change in ``antelligence/`` flips it from running to refusing; S-solo never asks."""
    from antelligence.experiments import v43_drive as drive
    from tests.v4_3.c.helpers import dev_ids
    freeze = tmp_path / "freeze.json"
    write_freeze(freeze, repo)
    monkeypatch.setattr(drive, "require_frozen", lambda path: require_frozen(path, repo))
    task = {"kind": "episode", "arm": "S-marks", "case_id": dev_ids()["dev_a"][0], "seed": 5, "config": str(drive.DEFAULT_CONFIG),
            "ticks": 20, "freeze_file": str(freeze), "uptake_level": "mid"}
    assert drive.run_task(task)["arm"] == "S-marks"
    flip_byte(repo / "antelligence" / "core.py")
    with pytest.raises(FreezeRefused, match="changed since the freeze"):
        drive.run_task(task)
    assert drive.run_task({**task, "arm": "S-solo"})["arm"] == "S-solo"


def test_this_repos_freeze_hash_covers_the_v43_code_and_not_the_v42_scripts():
    files = tracked_files(REPO)
    assert any(f.startswith("antelligence/experiments/v43_") for f in files)
    assert "scripts/v43_run.py" in files and "config/v4_3/world.yaml" in files
    assert not any(f.startswith("scripts/v42_") or f.startswith("config/v4_2") for f in files)
    h = freeze_hash(REPO)
    assert len(h) == 64 and freeze_hash(REPO) == h
    assert fz.DEFAULT_FREEZE_FILE == REPO / "docs" / "research" / "v4.3-20261015" / "freeze.json"
