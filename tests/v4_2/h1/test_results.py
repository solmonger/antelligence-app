import json
from pathlib import Path

import pytest

from antelligence.experiments.v42_harness import EvalRefused
from antelligence.experiments.v42_results import assemble, split_of, write_results
from antelligence.experiments.v41_gate import load_checked_split
from tests.v4_2.q1.helpers import harness, phantom

REPO = Path(__file__).resolve().parents[3]
FAKE_SPLIT = {"dev_a": ["phantom"], "dev_b": [], "eval": ["BRATS_EVAL"], "reserve": []}


def write_episode(out: Path, arm: str, case_id="phantom", seed=1, complete=True, split="synthetic"):
    h = harness(40, noisy=True)
    rec = h.run_episode(arm, phantom(), seed)
    d = out / arm / case_id / f"seed{seed}"
    d.mkdir(parents=True)
    from antelligence.experiments.v42_harness import bundle_of
    (d / "bundle.json").write_text(json.dumps(bundle_of(rec)))
    slim = {k: v for k, v in rec.items() if k != "colony"}
    slim.update({"bundle": "bundle.json", "complete": complete, "split": split})
    (d / "record.json").write_text(json.dumps(slim, default=str))
    return rec


def test_assemble_schema(tmp_path):
    for arm in ("S", "S-marks", "S-talk-shuffled"):
        write_episode(tmp_path, arm, split="dev")
    res = assemble(tmp_path, split_doc=FAKE_SPLIT, truth_for=lambda c: phantom().crop.truth)
    assert [r["arm"] for r in res["runs"]] == ["S", "S-marks", "S-talk-shuffled"]
    r = res["runs"][1]
    for k in ("arm", "case_id", "seed", "split", "coverage", "off_target_share", "time_to_50_coverage", "coverage_at_queen",
              "drug_on_non_tumour", "colony", "episode_s"):
        assert k in r
    assert r["split"] == "dev-A" and "colony" not in res["runs"][0]
    assert "honesty" in r["colony"]


def test_incomplete_episode_is_listed_not_used(tmp_path):
    write_episode(tmp_path, "S", split="dev")
    write_episode(tmp_path, "S-marks", complete=False, split="dev")
    res = assemble(tmp_path, split_doc=FAKE_SPLIT)
    assert len(res["runs"]) == 1 and res["meta"]["incomplete"] == ["S-marks/phantom/seed1"]


def test_refuses_non_dev_case_before_parsing(tmp_path):
    d = tmp_path / "S" / "BRATS_EVAL" / "seed1"
    d.mkdir(parents=True)
    (d / "record.json").write_text("not json")                     # must never be parsed
    with pytest.raises(EvalRefused):
        assemble(tmp_path, split_doc=FAKE_SPLIT)


def test_refuses_unknown_and_synthetic_against_real_split(tmp_path):
    doc = load_checked_split(REPO)
    with pytest.raises(EvalRefused):
        split_of(doc, "phantom")
    ev = doc["eval"][0]
    d = tmp_path / "S" / ev / "seed1"
    d.mkdir(parents=True)
    with pytest.raises(EvalRefused):
        assemble(tmp_path)


def test_write_results_roundtrip_feeds_report(tmp_path):
    for arm in ("S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S'"):
        write_episode(tmp_path, arm, split="dev")
    write_results(tmp_path, split_doc=FAKE_SPLIT)
    res = json.loads((tmp_path / "results.json").read_text())
    from antelligence.experiments.v42_report import render
    assert "Primary metrics" in render(res, None)
