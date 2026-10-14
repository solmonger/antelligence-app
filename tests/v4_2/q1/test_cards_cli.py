"""Claim resolution for the name cards, and the resumable dev-only CLI."""

import importlib.util
import json
from pathlib import Path

from antelligence.experiments.v42_harness import Claim, resolve_claims
from tests.v4_2.q1.helpers import PHANTOM

REPO = Path(__file__).resolve().parents[3]


def load_cli():
    spec = importlib.util.spec_from_file_location("v42_run_cli", REPO / "scripts" / "v42_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_claims_uses_another_ants_later_look_within_4mm():
    c = [Claim(0, "tumour_here", (10, 10, 10), 5), Claim(0, "not_tumour", (10, 10, 10), 5), Claim(0, "need_more", (10, 10, 10), 5)]
    looks = [(0, (10, 10, 10), 10, 0.9),            # its own look never counts
             (1, (10, 10, 10), 5, 0.9),             # not later
             (2, (13, 10, 10), 10, 0.2),            # 6 mm away: out of the 4 mm radius
             (3, (11, 10, 10), 20, 0.7),
             (4, (10, 11, 10), 15, 0.4)]            # earlier than the ant-3 look, decides
    assert resolve_claims(c, looks, 2.0) == [False, True, False]
    assert resolve_claims(c, looks, 2.0, before_tick=15) == [None, None, None]
    assert resolve_claims(c, looks, 2.0, before_tick=16) == [False, True, False]


def test_answer_claims_follow_the_answer():
    looks = [(2, (10, 10, 10), 10, 0.9)]
    assert resolve_claims([Claim(1, "answer_yes", (10, 10, 10), 5), Claim(1, "answer_no", (10, 10, 10), 5)], looks, 2.0) == [True, False]


def run_cli(tmp_path, *extra):
    cli = load_cli()
    return cli.main(["--out", str(tmp_path), "--phantom", str(PHANTOM), "--ticks", "30", *extra]), cli


def test_cli_is_resumable_and_writes_bundles_and_cards(tmp_path, capsys):
    code, _ = run_cli(tmp_path, "--arms", "S,S-solo,S'", "--seeds", "1,2")
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("ran ") == 6 and "skip" not in out
    d = tmp_path / "S-solo" / "phantom" / "seed1"
    rec = json.loads((d / "record.json").read_text())
    bundle = json.loads((d / "bundle.json").read_text())
    assert rec["complete"] and rec["split"] == "synthetic" and rec["timing_s"]["wall"] > 0 and rec["queen_hook_unset"]
    assert bundle["extra"]["colony"]["arm"] == "S-solo" and bundle["metrics"] == rec["metrics"]
    assert (d / "cards" / "ant-00.json").is_file() and (tmp_path / "S-prime" / "phantom" / "seed2" / "record.json").is_file()
    stamp = (d / "bundle.json").stat().st_mtime_ns
    code, _ = run_cli(tmp_path, "--arms", "S,S-solo,S'", "--seeds", "1,2")
    out = capsys.readouterr().out
    assert code == 0 and out.count("skip") == 6 and "ran " not in out and (d / "bundle.json").stat().st_mtime_ns == stamp
    (d / "record.json").unlink()                                              # a killed run: no completion marker, so it is redone
    run_cli(tmp_path, "--arms", "S-solo", "--seeds", "1")
    assert "ran " in capsys.readouterr().out and (d / "record.json").is_file()


def test_cli_refuses_evaluation_and_unknown_cases(tmp_path, capsys):
    cli = load_cli()
    split = json.loads((REPO / "config" / "v4_1" / "split.json").read_text())
    for cid in (split["eval"][0], split["reserve"][0], "BRATS_nope"):
        assert cli.main(["--out", str(tmp_path), "--cases", cid, "--arms", "S"]) == 2
    assert "refused" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_cli_smoke_reports_wall_time_per_arm(tmp_path, capsys):
    cli = load_cli()
    assert cli.main(["--out", str(tmp_path), "--smoke", "--cases-root", str(tmp_path / "no-cases"), "--phantom", str(PHANTOM),
                     "--ticks", "30"]) == 0
    rows = json.loads((tmp_path / "smoke.json").read_text())
    assert {r["arm"] for r in rows} == {"S", "S-solo", "S-marks", "S-talk", "S-marks-quiet", "S-prime"}
    assert all(r["episode_wall_s"] > 0 and r["case_id"] == "phantom" for r in rows)
