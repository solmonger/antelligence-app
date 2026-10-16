"""v4.31 guards: a development case cannot be read from a real store inside a test; the runner only touches dev cases and its own folders."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from antelligence.experiments.v41_gate import GateError
from antelligence.worlds.tumor_mri import store_adapter as sa

REPO = Path(__file__).resolve().parents[2]
SPLIT = json.loads((REPO / "config/v4_1/split.json").read_text())
DEV = SPLIT["dev_a"][0]                  # a real development case id (the id only; no data is read)


def _runner():
    spec = importlib.util.spec_from_file_location("v431_run", REPO / "scripts" / "v431_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)           # type: ignore[union-attr]
    return mod


def test_running_under_pytest_is_detected():
    assert sa.under_pytest()


def test_dev_case_from_the_default_store_is_refused_in_a_test():
    with pytest.raises(sa.DevCaseInTestError):
        sa.load_case(DEV)


def test_dev_case_from_a_real_store_path_is_refused_in_a_test():
    with pytest.raises(sa.DevCaseInTestError):
        sa.load_case(DEV, root="/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
    with pytest.raises(sa.DevCaseInTestError):
        sa.load_case(DEV, root=REPO / "data" / "v4_1" / "cases")


def test_a_symlink_into_a_real_store_does_not_launder_a_dev_case(tmp_path):
    link = tmp_path / "cases"
    link.symlink_to("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases") if Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases").exists() \
        else link.symlink_to(REPO / "config")
    with pytest.raises(sa.DevCaseInTestError):
        sa.load_case(DEV, root=link)


def test_dev_case_id_in_a_synthetic_temp_store_passes_the_guard_and_then_finds_no_data(tmp_path):
    """The guard protects real data, so a store the test wrote to a temp dir is allowed through (it then fails only because it is empty)."""
    with pytest.raises(FileNotFoundError):
        sa.load_case(DEV, root=tmp_path)
    assert sa.is_temp_store(tmp_path) and not sa.is_temp_store("/Volumes/WD_BLACK") and not sa.is_temp_store(REPO / "data")


def test_eval_and_unknown_cases_still_go_through_the_old_gate():
    with pytest.raises(GateError):
        sa.load_case(SPLIT["eval"][0])
    with pytest.raises(GateError):
        sa.load_case("NOT_A_CASE")
    assert not issubclass(GateError, sa.DevCaseInTestError)


def test_runner_lists_exactly_the_40_development_cases_and_no_eval_case():
    r = _runner()
    cases = r.dev_cases()
    ids = [c["case"] for c in cases]
    assert len(ids) == 40 and len(set(ids)) == 40
    assert sorted(ids) == sorted(SPLIT["dev_a"] + SPLIT["dev_b"])
    assert not set(ids) & set(SPLIT["eval"]) and not set(ids) & set(SPLIT["reserve"])
    assert {c["split"] for c in cases} == {"dev_a", "dev_b"}


def test_runner_refuses_outputs_outside_its_folders():
    r = _runner()
    for bad in ("/Volumes/WD_BLACK/antelligence-v4.1/runs/x", "/Volumes/WD_BLACK/antelligence-v4.2/runs/x", "/Volumes/WD_BLACK/antelligence-v4.3/runs/x",
                str(REPO / "docs" / "research" / "v4.1-20261013"), str(REPO / "docs" / "research" / "v4.3-20261015"), str(REPO / "config" / "v4_3"), "/tmp/x"):
        with pytest.raises(SystemExit):
            r.assert_out_allowed(Path(bad))
    r.assert_out_allowed(Path("/Volumes/WD_BLACK/antelligence-v4.31/runs/units_low"))
    r.assert_out_allowed(REPO / "docs" / "research" / "v4.31-20261016" / "phaseA")


def test_runner_never_imports_the_pod_or_live_backend_modules():
    src = (REPO / "scripts" / "v431_run.py").read_text() + (REPO / "antelligence/experiments/v431_phaseA.py").read_text()
    assert "v41_pod" not in src.replace("v41_pod or v41_prod", "") and "v41_prod" not in src.replace("v41_pod or v41_prod", "")
    for mod in ("requests", "httpx", "openai", "urllib.request", "socket"):
        assert f"import {mod}" not in src
