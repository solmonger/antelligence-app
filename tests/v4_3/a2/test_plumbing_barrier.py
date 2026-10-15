"""Plumbing of ``scan_p`` / ``est``, determinism, the extended information barrier, claim resolution on est, the scripts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments.v42_harness import ARMS
from antelligence.experiments.v43_harness import V43Harness, V43Kit, bundle_of, code_sha_v43
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_obs import ALLOWED, expected_keys
from antelligence.worlds.tumor_mri.colony_policies import ColonyScript, ScanInfo
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from tests.v4_2.q1.helpers import SCAN, phantom, phantom_cfg
from tests.v4_3.a2.helpers import COEF, R42, R43, estimate, v43_harness

REPO = Path(__file__).resolve().parents[3]
ANT_ARMS = ("S-solo", "S-marks", "S-talk")
FORBIDDEN_OUTSIDE_LOG = ("est", "scan_p")


def looks_of(rec):
    return [e for log in rec["colony"]["reading_logs"].values() for e in log if e["kind"] == "look"]


def keys_outside(obj, banned, path=""):
    """Every (path, key) where a banned key appears in a nested structure."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in banned:
                hits.append((path, k))
            hits += keys_outside(v, banned, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits += keys_outside(v, banned, f"{path}[{i}]")
    return hits


# --- plumbing ----------------------------------------------------------------------------------------------------------

def test_est_and_scan_p_appear_in_look_entries_only_when_an_estimate_is_given():
    case, est = phantom(), estimate()
    with_est = v43_harness(R43, 60, est=est).run_episode("S-solo", case, 1)
    entries = looks_of(with_est)
    assert entries and all("est" in e and "scan_p" in e for e in entries)
    p = case.crop.p_tumour
    for e in entries:
        assert e["scan_p"] == pytest.approx(float(p[tuple(e["pos_vox"])]))          # the scan p at the ant's own voxel
        assert e["est"] == pytest.approx(est(e["scan_p"], e["value"]))               # a function of those two numbers only
    others = [e for log in with_est["colony"]["reading_logs"].values() for e in log if e["kind"] != "look"]
    assert others and all("est" not in e and "scan_p" not in e for e in others)      # dose / uptake / washout entries unchanged
    without = v43_harness(R42, 60, noisy=True).run_episode("S-solo", case, 1)
    assert all("est" not in e and "scan_p" not in e for log in without["colony"]["reading_logs"].values() for e in log)


def test_determinism_two_runs_give_identical_bundles():
    case = phantom()
    for arm in ANT_ARMS:
        a = v43_harness(R43, 60).run_episode(arm, case, 2)
        b = v43_harness(R43, 60).run_episode(arm, case, 2)
        assert json.dumps(bundle_of(a), sort_keys=True) == json.dumps(bundle_of(b), sort_keys=True)
        assert a["cards_final"] == b["cards_final"] and a["ant_rules"] == b["ant_rules"]


def test_replay_reproduces_a_v43_bundle():
    h = v43_harness(R43, 60)
    rec = h.run_episode("S-talk", phantom(), 1)
    again = h.replay(rec, phantom())
    assert again["metrics"] == rec["metrics"]


def test_v43_changes_behaviour_and_records_rules_and_estimate():
    case = phantom()
    old = v43_harness(R42, 60).run_episode("S-solo", case, 1)
    new = v43_harness(R43, 60).run_episode("S-solo", case, 1)
    assert bundle_of(old)["extra"]["colony"]["action_log"] != bundle_of(new)["extra"]["colony"]["action_log"]
    b = bundle_of(new)
    assert b["extra"]["ant_rules"] == {"rules": R43.to_dict(), "sha": R43.sha()}
    assert new["ant_rules"]["sha"] == R43.sha() and new["estimate"]["sha"] == estimate().sha()
    assert new["colony"]["params"]["answer_rule"] == "v43"
    assert all(c["rev"] == "v4.3" and c["min_resolved"] == 3 for c in new["cards_final"].values())
    assert "ant_rules" not in bundle_of(old)["extra"]["colony"]                  # ant_rules sits beside colony, not inside it


def test_v43_rules_need_an_estimate():
    with pytest.raises(ValueError):
        V43Harness(phantom_cfg(60), None, R43, None)


def test_all_ant_arms_and_the_controls_run_under_v43():
    from antelligence.experiments import v42_arms
    v42_arms.register_all()          # idempotent; tests/v4_2 can edit the arm registry after the module was first imported
    case = phantom()
    for arm in ("S-solo", "S-marks", "S-talk", "S-marks-quiet", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect"):
        rec = v43_harness(R43, 60).run_episode(arm, case, 1)
        assert rec["metrics"]["coverage"] >= 0.0 and rec["ant_rules"]["sha"] == R43.sha()
    assert v43_harness(R43, 60).run_episode("S", case, 1)["arm"] == "S"
    assert v43_harness(R43, 60).run_episode("S'", case, 1)["arm"] == "S'"


def test_harness_registers_no_new_arm():
    before = set(ARMS)
    import antelligence.experiments.v43_harness  # noqa: F401
    assert set(ARMS) == before


def test_claims_are_resolved_on_est_when_the_rules_use_it():
    kit = V43Kit("S-marks", 2, SCAN, sharing=True, rules=R43)
    assert kit.claim_look_value({"value": 0.9, "est": 0.3}) == 0.3 and kit.claim_look_value({"value": 0.9}) == 0.9
    assert all(c.min_resolved == 3 for c in kit.cards) and all(s.rules == R43 for s in kit.scripts)
    legacy = V43Kit("S-marks", 2, SCAN, sharing=True, rules=R42)
    assert legacy.claim_look_value({"value": 0.9, "est": 0.3}) == 0.9
    assert all(c.min_resolved == 1 for c in legacy.cards)

    class V:
        tick = 5
        observation = {"look": 0.9, "reading_log": [{"tick": 5, "kind": "look", "est": 0.3, "value": 0.9}]}

    assert kit.view_look_value(V) == 0.3 and legacy.view_look_value(V) == 0.9


# --- the information barrier -------------------------------------------------------------------------------------------

class Spy:
    def __init__(self, monkeypatch):
        self.obs, self.queen_in = [], []
        orig, qorig = ColonyScript.decide, ScriptedQueen.__call__
        spy = self

        def decide(this, view):
            spy.obs.append((this.ant, view.tick, dict(view.observation)))
            return orig(this, view)

        def queen(this, tick, inputs):
            spy.queen_in.append((tick, dict(inputs)))
            return qorig(this, tick, inputs)

        monkeypatch.setattr(ColonyScript, "decide", decide)
        monkeypatch.setattr(ScriptedQueen, "__call__", queen)


@pytest.mark.parametrize("arm", ANT_ARMS)
def test_barrier_est_and_scan_p_only_in_own_reading_log_entries(arm, monkeypatch):
    """The barrier test of v4.2, extended: with the estimate on, every observation still has exactly OBS_KEYS | ALLOWED[arm]
    and ``est`` / ``scan_p`` appear nowhere except inside the ant's OWN ``reading_log`` entries (at voxels it stood on)."""
    spy = Spy(monkeypatch)
    rec = v43_harness(R43, 120).run_episode(arm, phantom(), 3)
    assert spy.obs and all(set(o) == expected_keys(arm) for _, _, o in spy.obs)
    stood = {}
    for ant, _, o in spy.obs:
        stood.setdefault(ant, set()).add(tuple(o["position_vox"]))
    seen_log_est = 0
    for ant, tick, o in spy.obs:
        rest = {k: v for k, v in o.items() if k != "reading_log"}
        assert keys_outside(rest, FORBIDDEN_OUTSIDE_LOG) == [], f"est / scan_p leaked outside reading_log ({arm}, ant {ant}, tick {tick})"
        for e in o["reading_log"]:
            if e["kind"] == "look":
                assert "est" in e and "scan_p" in e and tuple(e["pos_vox"]) in stood[ant]
                seen_log_est += 1
            else:
                assert "est" not in e and "scan_p" not in e
    assert seen_log_est
    sharing = "marks_near" in ALLOWED[arm]
    for _, inputs in spy.queen_in:
        assert keys_outside(inputs, FORBIDDEN_OUTSIDE_LOG) == []
        assert set(inputs) == ({"tick", "regions", "marks", "quorum_events"} if sharing else {"tick", "regions"})
    # the bundle: est / scan_p only inside the reading logs; marks, messages, cards, actions and the queen's rows carry none
    colony = dict(rec["colony"])
    logs = colony.pop("reading_logs")
    assert keys_outside(colony, FORBIDDEN_OUTSIDE_LOG) == []
    assert keys_outside(rec["cards_final"], FORBIDDEN_OUTSIDE_LOG) == []
    assert logs and all("est" in e for log in logs.values() for e in log if e["kind"] == "look")


def test_what_ants_share_is_unchanged_by_the_estimate(monkeypatch):
    """In a sharing arm the marks other ants read are the v4.2 mark rows: the same keys, no estimate."""
    spy = Spy(monkeypatch)
    v43_harness(R43, 120).run_episode("S-marks", phantom(), 3)
    mark_keys = {"kind", "pos_vox", "strength", "distinct_authors", "strong", "disputed", "authors"}
    rows = [m for _, _, o in spy.obs for m in (o["marks_near"] or [])]
    assert rows and all(set(m) <= mark_keys | {"reading_ppm", "amount_milli", "tick", "subtype", "mark_kind", "ant", "half_life"} for m in rows)
    assert keys_outside(rows, FORBIDDEN_OUTSIDE_LOG) == []


# --- the scripts -------------------------------------------------------------------------------------------------------

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def coef_file(tmp_path):
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(COEF))
    return p


PHANTOM = REPO / "tests" / "v4_1" / "world" / "fixtures" / "phantom.npz"


def test_v43_run_smoke_on_the_phantom(tmp_path, coef_file):
    cli = load_script("v43_run")
    out = tmp_path / "out"
    assert cli.main(["--out", str(out), "--phantom", str(PHANTOM), "--ticks", "60", "--arms", "S,S-solo,S-marks,S-talk",
                     "--coefficients", str(coef_file)]) == 0
    rec = json.loads((out / "S-solo" / "phantom" / "seed1" / "record.json").read_text())
    bundle = json.loads((out / "S-solo" / "phantom" / "seed1" / "bundle.json").read_text())
    assert rec["complete"] and rec["ant_rules"]["sha"] == R43.sha() and bundle["extra"]["ant_rules"]["sha"] == R43.sha()
    assert (out / "S-marks" / "phantom" / "seed1" / "cards" / "ant-00.json").is_file()
    # resumable: a second call skips what is complete
    assert cli.main(["--out", str(out), "--phantom", str(PHANTOM), "--ticks", "60", "--arms", "S-solo",
                     "--coefficients", str(coef_file)]) == 0
    # everything stayed under --out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["coefficients.json", "out"]


def test_v43_run_exits_clearly_when_the_coefficient_file_is_missing(tmp_path, capsys):
    cli = load_script("v43_run")
    assert cli.main(["--out", str(tmp_path / "o"), "--phantom", str(PHANTOM), "--ticks", "30", "--coefficients",
                     str(tmp_path / "nope.json")]) == 2
    assert "coefficient file" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_v43_run_refuses_non_dev_cases(tmp_path, coef_file):
    cli = load_script("v43_run")
    assert cli.main(["--out", str(tmp_path / "o"), "--cases", "NOT_A_CASE", "--coefficients", str(coef_file)]) == 2


def test_gate_a_summary_on_a_stubbed_split(tmp_path, coef_file, monkeypatch):
    gate = load_script("v43_gate_a")
    cases_root = tmp_path / "cases"
    (cases_root / "phantom").mkdir(parents=True)
    (cases_root / "phantom" / "case.npz").write_bytes(b"")                       # existence only: the loader is stubbed
    monkeypatch.setattr(gate, "load_checked_split", lambda repo: {"dev_a": ["phantom"], "dev_b": ["gone"]})
    monkeypatch.setattr(gate, "load_dev_case", lambda cid, root=None: phantom())
    out = tmp_path / "gate"
    argv = ["--split", "both", "--seed", "1", "--out", str(out), "--coefficients", str(coef_file), "--cases-root", str(cases_root)]
    assert gate.main(argv) == 0
    s = json.loads((out / "gate_a_summary.json").read_text())
    assert s["n_cases"] == 1 and s["missing_cases"] == ["gone"] and s["arms"] == ["S", "S-solo"]
    assert s["pass"] == (s["mean"]["S-solo"] >= s["mean"]["S"] - 0.05)
    assert s["S_solo_minus_S"] == pytest.approx(s["mean"]["S-solo"] - s["mean"]["S"])
    assert s["ant_rules"] == R43.to_dict() and s["ant_rules_sha"] == R43.sha() and s["code_sha"] == code_sha_v43()
    assert len(s["config_sha256"]) == 64 and len(s["coefficients_sha256"]) == 64
    f2 = json.loads(json.dumps(s["coverage"]))
    # facts for gate A: F2 and F3 off, F1 on
    assert (out / "S-solo" / "phantom" / "seed1" / "record.json").is_file()
    rec = json.loads((out / "S-solo" / "phantom" / "seed1" / "record.json").read_text())
    assert rec["facts"]["f1_simlook"] is True and rec["facts"]["f2"] is False and rec["facts"]["f3"] is False
    # resumable and identical on a second call
    assert gate.main(argv) == 0
    assert json.loads((out / "gate_a_summary.json").read_text())["coverage"] == f2


def test_gate_a_missing_coefficients_exits_2(tmp_path):
    gate = load_script("v43_gate_a")
    assert gate.main(["--split", "dev_a", "--out", str(tmp_path / "g"), "--coefficients", str(tmp_path / "x.json")]) == 2
