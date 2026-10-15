"""A tiny end-to-end run on the phantom: verdict -> results -> report, then the follow-on scripts, all in a temp dir.

The drive layer's case loader is replaced by the phantom (``phantom_loader``), registered under real development ids so the
driver's dev-case guard and freeze guard run exactly as they will for the lead. No development case is read or run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from antelligence.experiments import v43_drive as drive
from tests.v4_3.c.helpers import REPO, dev_ids, script, write_test_freeze

verdict = script("v43_verdict")
replicate = script("v43_replicate")
headroom = script("v43_headroom")
curve = script("v43_curve")
effect = script("v43_effect")
diagnose = script("v43_diagnose")
report = script("v43_report")

ALL_ARMS = "S,S-solo,S-marks,S-talk,S-marks-displaced,S-talk-shuffled,S',S-marks-perfect,S-marks-quiet"
# no --ticks: a shortened run on a (phantom labelled) development case is refused by the harness freeze check, as it must be
COMMON = ["--workers", "1", "--resamples", "200"]


def cases() -> str:
    d = dev_ids()
    return f"{d['dev_a'][0]},{d['dev_b'][0]}"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One module-wide verdict run (mid, seeds 5 and 6, every default arm) plus its freeze file. Built once."""
    from dataclasses import replace
    from tests.v4_3.b1.helpers import phantom_with_channels
    base = phantom_with_channels()
    drive.CASE_LOADER = lambda cid, root=None: replace(base, case_id=cid, split="dev")
    drive._CASES.clear()
    drive._HARNESS.clear()
    tmp = tmp_path_factory.mktemp("c_e2e")
    freeze = tmp / "freeze.json"
    write_test_freeze(freeze)
    out = tmp / "mid"
    rc = verdict.main(["--out", str(out), "--level", "mid", "--cases", cases(), "--seeds", "5,6", "--arms", ALL_ARMS,
                       "--freeze-file", str(freeze), "--cases-root", str(tmp), *COMMON])
    assert rc == 0
    yield {"tmp": tmp, "freeze": freeze, "out": out, "results": json.loads((out / "results.json").read_text())}
    drive.CASE_LOADER = None
    drive._CASES.clear()
    drive._HARNESS.clear()


def test_verdict_writes_results_report_stats_and_checks(world):
    out, res = world["out"], world["results"]
    for name in ("results.json", "REPORT.md", "stats.json", "verdict_checks.json"):
        assert (out / name).is_file(), name
    runs = res["runs"]
    assert len(runs) == 9 * 2 * 2 and {r["arm"] for r in runs} == set(ALL_ARMS.split(","))
    assert {r["split"] for r in runs} == {"dev-A", "dev-B"}
    meta = res["meta"]
    assert meta["level"] == "mid" and meta["seeds"] == [5, 6] and not meta["dry_overrides"]["ticks"]
    assert len(meta["config_sha256"]) == 64 and len(meta["ant_rules_sha"]) == 64 and len(meta["code_sha"]) == 64
    f = meta["freeze"]
    assert f["required"] and f["freeze_hash_recorded"] == f["freeze_hash_now"] and f["freeze_hash_recorded"]
    assert "ceiling_check" in meta and meta["incomplete"] == []
    assert all("probes" in r for r in runs if r["arm"] == "S-marks")
    text = (out / "REPORT.md").read_text()
    assert text.startswith("# v4.3 Phase 0 report (generated)")
    for h in ("## Primary metrics", "## Probes (mean per run)", "## Verdict statistics", "## Realised mean u per sub-region and level"):
        assert h in text, h
    assert "Source: `results.json`" in text
    stats = json.loads((out / "stats.json").read_text())
    assert set(stats["primary"]) == {"a", "b"}


def test_verdict_is_resumable_and_changes_nothing(world, capsys):
    out = world["out"]
    before = (out / "results.json").read_text()
    rc = verdict.main(["--out", str(out), "--level", "mid", "--cases", cases(), "--seeds", "5,6", "--arms", ALL_ARMS,
                       "--freeze-file", str(world["freeze"]), "--cases-root", str(world["tmp"]), *COMMON])
    log = capsys.readouterr().out
    assert rc == 0 and "ran " not in log and log.count("skip") == 36
    assert json.loads((out / "results.json").read_text())["runs"] == json.loads(before)["runs"]


def test_verdict_refuses_sharing_arms_without_a_freeze_but_runs_gate_a_arms(world, tmp_path, capsys):
    missing = tmp_path / "none.json"
    common = ["--cases", cases(), "--seeds", "5", "--freeze-file", str(missing), "--cases-root", str(world["tmp"]), *COMMON]
    out = tmp_path / "refused"
    assert verdict.main(["--out", str(out), "--arms", "S,S-solo,S-marks", *common]) == 2
    assert verdict.main(["--out", str(out), "--arms", "S-marks-displaced", *common]) == 2
    assert not out.exists() and "does not exist" in capsys.readouterr().err
    ok = tmp_path / "gate"
    assert verdict.main(["--out", str(ok), "--arms", "S,S-solo,S'", *common]) == 0       # gate-A style: no freeze needed
    assert {p.name for p in ok.iterdir() if p.is_dir()} == {"S", "S-solo", "S'"}


def test_verdict_refuses_eval_cases_extra_workers_and_v42_output(world, tmp_path):
    from antelligence.experiments.v41_gate import load_checked_split
    doc = load_checked_split(REPO)
    base = ["--freeze-file", str(world["freeze"]), "--cases-root", str(world["tmp"])]
    assert verdict.main(["--out", str(tmp_path / "e"), "--cases", doc["eval"][0], *base]) == 2
    assert verdict.main(["--out", str(tmp_path / "r"), "--cases", doc["reserve"][0], *base]) == 2
    assert verdict.main(["--out", str(tmp_path / "w"), "--cases", cases(), "--workers", "7", *base]) == 2
    for bad in (REPO / "docs" / "research" / "v4.2-20261014" / "x", REPO / "config" / "v4_2" / "x",
                REPO / "docs" / "research" / "v4.1-20261013" / "x", Path("/Volumes/WD_BLACK/antelligence-v4.2/runs/x")):
        assert verdict.main(["--out", str(bad), "--cases", cases(), *base]) == 2
        assert not bad.exists()


def test_drive_refuses_a_sharing_arm_on_a_dev_case_without_the_freeze(phantom_loader, tmp_path):
    from antelligence.experiments.v43_freeze import FreezeRefused
    d = dev_ids()
    task = {"kind": "episode", "case_id": d["dev_a"][0], "seed": 5, "config": str(drive.DEFAULT_CONFIG), "ticks": 20,
            "freeze_file": str(tmp_path / "no.json"), "uptake_level": "mid"}
    for arm in ("S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect", "S-marks-quiet"):
        with pytest.raises(FreezeRefused):
            drive.run_task({**task, "arm": arm})
    for arm in ("S", "S-solo", "S'"):
        assert drive.run_task({**task, "arm": arm})["arm"] == arm


def test_replicate_headroom_curve_effect_and_report_run_on_the_output(world, tmp_path):
    out, tmp = world["out"], world["tmp"]
    # replication: 2 cases x seeds 5 and 6 = 4 pairs here, so ask for 4 (the real run asks for 80 and fails on anything else)
    assert replicate.main(["--results", str(out), "--arm", "S-marks", "--seeds", "5,6", "--resamples", "200"], expected_pairs=4) == 0
    rep = json.loads((out / "replication.json").read_text())
    assert rep["n_pairs"] == 4 and rep["verdict"]["pairs_complete"] and rep["holm_p"] == pytest.approx(min(1.0, 2 * rep["raw_p"]))
    assert replicate.main(["--results", str(out), "--arm", "S-talk", "--seeds", "5,6", "--resamples", "200", "--out", str(tmp_path / "r2.json")]) == 0
    assert not json.loads((tmp_path / "r2.json").read_text())["verdict"]["pairs_complete"]            # 4 pairs, 80 expected
    assert headroom.main(["--results", str(out)]) == 0
    hr = json.loads((out / "headroom.json").read_text())
    assert hr["n_pairs"] == 4 and "pass" not in hr and 0.0 <= hr["share_pairs_ge_threshold"] <= 1.0
    # the curve needs a run at each level
    for lvl in ("low", "high"):
        assert verdict.main(["--out", str(tmp / lvl), "--level", lvl, "--cases", cases(), "--seeds", "5,6", "--arms", "S,S-solo,S-marks,S-talk,S'",
                             "--freeze-file", str(world["freeze"]), "--cases-root", str(tmp), *COMMON]) == 0
    assert curve.main(["--low", str(tmp / "low"), "--mid", str(out), "--high", str(tmp / "high"), "--out", str(tmp / "curve")]) == 0
    c = json.loads((tmp / "curve" / "curve.json").read_text())
    assert set(c["levels"]) == {"low", "mid", "high"}
    assert (tmp / "curve" / "curve.md").read_text().startswith("## Patchiness curve")
    assert curve.main(["--low", str(out), "--mid", str(out), "--high", str(out), "--out", str(tmp / "curve2")]) == 2   # level mismatch
    # message effect (replays; sharing arms need the freeze)
    assert effect.main(["--run-dir", str(out), "--out", str(tmp / "effect"), "--runs", "1", "--items", "3", "--workers", "1",
                        "--freeze-file", str(world["freeze"]), "--cases-root", str(tmp)]) == 0
    eff = json.loads((tmp / "effect" / "effect.json").read_text())
    assert set(eff) == {"S-marks", "S-talk"} and set(eff["S-marks"]["by_group"]) == {"all", "uptake_marks", "other_marks", "messages"}
    assert effect.main(["--run-dir", str(out), "--out", str(tmp / "effect2"), "--freeze-file", str(tmp / "none.json"),
                        "--cases-root", str(tmp)]) == 2
    # diagnostics from the saved runs, then the report with every section
    diag = diagnose.analyse(out, tmp, limit=0, extra_run_dirs=[], loader=drive.CASE_LOADER)
    assert diag["pairs"] == 4 and "S-marks" in diag["probes_per_run"] and diag["realised_u"]["levels"]["mid"]["rim"]["mean_u"] > 0.5
    for arm, g in diag["gap_to_s_prime_by_cause"].items():
        assert g["n_runs"] == 4
        parts = [g[k] for k in ("false_alarm", "poor_patch", "unreleased", "remainder")]
        assert sum(p for p in parts if p is not None) == pytest.approx(g["gap"])            # the pools and the remainder sum to the gap
    assert diag["gap_to_s_prime_by_cause"]["S"]["false_alarm"] is None                      # S has no log
    assert diag["gap_to_s_prime_by_cause"]["S-marks-displaced"]["poor_patch"] is None       # displaced marks sit at random places on purpose
    for arm, p in diag["poor_tissue_release"].items():
        if p["n_runs"]:
            assert p["max_abs_accounting_error"] < 0.01 * p["total_drug"]                    # dosed marks add up to the drug released
    with pytest.raises(SystemExit, match="refused"):
        diagnose.main(["--run-dir", str(out), "--cases-root", str(tmp), "--out", str(REPO / "docs" / "research" / "v4.2-20261014" / "d")])
    (tmp / "diag").mkdir()
    (tmp / "diag" / "diagnostics_numbers.json").write_text(json.dumps(diag, default=str))
    assert report.main([str(out / "results.json"), "--out", str(tmp / "REPORT2.md"), "--diagnostics", str(tmp / "diag" / "diagnostics_numbers.json"),
                        "--effect", str(tmp / "effect" / "effect.json"), "--cite", "docs/research/v4.3-20261015/verdict-mid/results.json"]) == 0
    text = (tmp / "REPORT2.md").read_text()
    for h in ("## Gap to S' by cause", "## Drug released on poor-uptake tissue", "## Message effect by kind", "## Probes"):
        assert h in text, h
    assert "docs/research/v4.3-20261015/verdict-mid/results.json" in text
