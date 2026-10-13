"""Provider-caused queen fallbacks: the purge script and the v41_fit guard (mistake-2039). Synthetic files only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from antelligence.experiments import v41_fallbacks, v41_fit
from tests.v4_1.fit.test_fit import DEV, SPLIT, make_run

ROOT = Path(__file__).resolve().parents[3]
NOTE = "queen fallback: ProviderError: HTTP 401: invalid, blocked or out of funds"


def purge_module():
    spec = importlib.util.spec_from_file_location("v41_purge_t", ROOT / "scripts" / "v41_purge_provider_fallbacks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_export(run_dir: Path, run_id: str, harness_sha: str, notes=()):
    lines = [json.dumps({"study": "v4.1", "harness_cfg_sha256": harness_sha})]
    for i, note in enumerate(notes):
        lines.append(json.dumps({"run_id": run_id, "seq": i, "type": "alarm", "author": {"agent": "queen"},
                                 "payload": {"kind": "other", "value_ppm": 1000000, "note": note}, "tick": 0}))
    lines.append(json.dumps({"run_id": run_id, "seq": 99, "type": "task", "payload": {"objective": NOTE}, "tick": 0}))
    (run_dir / f"{run_id}.jsonl").write_text("\n".join(lines) + "\n")


def make_work(tmp_path):
    """Three cached episodes: one tainted (old row, no run_id), one clean, one tainted row that names its run_id."""
    work = tmp_path / "work"
    specs = [("SW-B", "case_bad", "h1", [NOTE, "queen fallback: QueenReplyError: not JSON"]),
             ("SW-B", "case_ok", "h1", ["queen fallback: QueenReplyError: not JSON"]),
             ("SW-noBoard", "case_new", "h2", [NOTE])]
    rows = []
    for arm, case, h, notes in specs:
        run_id = (arm[:2] + case).encode().hex().ljust(64, "0")[:64]
        d = work / "runs" / run_id
        d.mkdir(parents=True)
        (d / "cell.json").write_text(json.dumps({"run_id": run_id, "arm": arm, "case": case, "seed": 1}))
        (d / "decisions.jsonl").write_text("")
        write_export(d, run_id, h * 20, notes)
        rows.append({"harness": h * 20, "arm": arm, "case": case, "seed": 1, "coverage": 0.5, "stats": {}, "ctx": "c"})
    rows[2]["run_id"] = (specs[2][0][:2] + specs[2][1]).encode().hex().ljust(64, "0")[:64]
    cache = work / "results-cache.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json at all\n")
    return work, cache


def test_note_matching_is_exact_about_provider_errors():
    assert v41_fallbacks.is_provider_fallback_note(NOTE)
    assert v41_fallbacks.is_provider_fallback_note("queen fallback: ProviderUnavailable: HTTP 503")
    assert not v41_fallbacks.is_provider_fallback_note("queen fallback: QueenReplyError: not JSON")
    assert not v41_fallbacks.is_provider_fallback_note("queen fallback: call cap 6 spent")
    assert not v41_fallbacks.is_provider_fallback_note(None)


def test_purge_dry_run_changes_nothing_and_lists_the_tainted_rows(tmp_path, capsys):
    mod = purge_module()
    work, cache = make_work(tmp_path)
    before = cache.read_text()
    assert mod.main([str(work)]) == 0
    out = capsys.readouterr().out
    assert cache.read_text() == before and not (work / "results-cache.jsonl.bak").exists()
    assert "case_bad" in out and "case_new" in out and "case_ok" not in out and "dry run" in out


def test_purge_apply_drops_tainted_rows_keeps_the_rest_and_backs_up(tmp_path, capsys):
    mod = purge_module()
    work, cache = make_work(tmp_path)
    before = cache.read_text()
    assert mod.main([str(work), "--apply"]) == 0
    kept = cache.read_text().splitlines()
    assert (work / "results-cache.jsonl.bak").read_text() == before
    assert len(kept) == 2 and json.loads(kept[0])["case"] == "case_ok" and kept[1] == "not json at all"
    # a second apply has nothing to do and leaves the original backup alone
    assert mod.main([str(work), "--apply"]) == 0 and (work / "results-cache.jsonl.bak").read_text() == before


def test_purge_refuses_to_overwrite_the_first_backup(tmp_path):
    mod = purge_module()
    work, cache = make_work(tmp_path)
    (work / "results-cache.jsonl.bak").write_text("older original\n")
    assert mod.main([str(work), "--apply"]) == 2
    assert cache.read_text().count("\n") == 4 and (work / "results-cache.jsonl.bak").read_text() == "older original\n"


def test_purge_uses_the_recorded_split_and_the_cell_counter(tmp_path):
    mod = purge_module()
    work, cache = make_work(tmp_path)
    rows = [json.loads(l) for l in cache.read_text().splitlines() if l.startswith("{")]
    rows[1]["queen_fallbacks"] = {"provider": 2, "invalid_reply": 0, "cap": 0}          # the cache row itself says so
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = mod.purge(work)
    assert r["rows_dropped"] == 3 and r["rows_kept"] == 0


def test_purge_errors_on_a_missing_cache_or_dir(tmp_path):
    mod = purge_module()
    assert mod.main([str(tmp_path / "nope")]) == 1
    (tmp_path / "empty").mkdir()
    assert mod.main([str(tmp_path / "empty")]) == 1


# --- the fit guard ----------------------------------------------------------------------------------------------------

@pytest.fixture
def mixed(tmp_path):
    """Four dev cases x two arms, 20 rows each; the runs of the first case are tainted (one by cell, one by alarm)."""
    for c in DEV[:4]:
        for arm in ("SW-B", "SW-noBoard"):
            make_run(tmp_path, c, arm, rows=20)
    runs = {(json.loads((d / "cell.json").read_text())["arm"], json.loads((d / "cell.json").read_text())["case"]): d
            for d in (tmp_path / "runs").iterdir()}
    bad_cell = runs[("SW-B", DEV[0])]
    cell = json.loads((bad_cell / "cell.json").read_text())
    cell["queen_fallbacks_provider"] = 2
    (bad_cell / "cell.json").write_text(json.dumps(cell))
    bad_export = runs[("SW-noBoard", DEV[0])]
    write_export(bad_export, "e" * 64, "a" * 64, [NOTE])                  # an old cell with no counter, only the alarm
    return tmp_path, runs


def test_fit_excludes_runs_with_provider_fallbacks_by_default(mixed):
    root, _ = mixed
    rep = v41_fit.run_fit([root], tag="t", split_file=SPLIT)
    assert rep["inputs"]["run_dirs"] == 6 and rep["inputs"]["provider_fallback_runs"] == 2
    assert rep["inputs"]["provider_fallback_runs_excluded"] is True and rep["inputs"]["cases"] == 3
    assert rep["inputs"]["skipped"]["provider_fallback_run"] == 40


def test_fit_can_be_told_to_allow_them(mixed):
    root, _ = mixed
    rep = v41_fit.run_fit([root], tag="t", split_file=SPLIT, allow_provider_fallbacks=True)
    assert rep["inputs"]["run_dirs"] == 8 and rep["inputs"]["provider_fallback_runs"] == 2
    assert rep["inputs"]["provider_fallback_runs_excluded"] is False and "provider_fallback_run" not in rep["inputs"]["skipped"]


def test_fit_cells_file_drops_the_excluded_runs_too(mixed, tmp_path_factory):
    root, runs = mixed
    cells = tmp_path_factory.mktemp("c") / "cells.jsonl"
    rows = [json.loads((d / "cell.json").read_text()) | {"split": "dev"} for d in runs.values()]
    cells.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = v41_fit.run_fit([root], tag="t", cells_path=cells, split_file=SPLIT)
    assert rep["inputs"]["cells_file_lines"] == 6


def test_fit_refuses_when_every_run_is_tainted(tmp_path):
    for arm in ("SW-B", "SW-noBoard"):
        d = make_run(tmp_path, DEV[0], arm, rows=10)
        cell = json.loads((d / "cell.json").read_text())
        cell["queen_fallbacks_provider"] = 1
        (d / "cell.json").write_text(json.dumps(cell))
    with pytest.raises(v41_fit.FitError, match="provider-caused"):
        v41_fit.run_fit([tmp_path], tag="t", split_file=SPLIT)


def test_fit_script_has_the_override_flag(mixed, capsys):
    spec = importlib.util.spec_from_file_location("v41_fit_script", ROOT / "scripts" / "v41_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root, _ = mixed
    base = [str(root), "--tag", "t", "--no-freeze", "--report", str(root / "r.json")]
    mod.main(base)
    assert "2 run(s) excluded" in capsys.readouterr().out
    mod.main(base + ["--allow-provider-fallbacks"])
    assert "INCLUDED" in capsys.readouterr().out
