"""B2 information barrier (v4.2 goal 2.8): the rule adds no observation key and no queen input. Uptake information comes ONLY
from the ant's own ``reading_log`` and, in sharing arms, ``marks_near`` (as the ``marks`` input)."""

import ast
import copy
import inspect

import pytest

from antelligence.kernel.colony import episode as ep
from antelligence.worlds.tumor_mri import colony_policies as cp
from antelligence.worlds.tumor_mri.colony_obs import expected_keys
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from tests.v4_2.q1.helpers import inputs, mark_row
from tests.v4_3.b2.helpers import POS, RF, run, uptake

ANT_ARMS = ("S-solo", "S-marks", "S-talk")
B2_FUNCS = ("uptake_class", "uptake_info", "known_uptake_spots", "_hold_reading", "_forage_forced", "_forage_release", "_poor_response")
ALLOWED_INP_ATTRS = {"reading_log", "marks", "pos", "voxel_mm", "tick", "payload_fraction"}


class Spy:
    def __init__(self, monkeypatch):
        self.obs, self.queen = [], []
        orig_obs, orig_q = ep._Run._observation, ScriptedQueen.__call__

        def spy_obs(run, i, tick, readings):
            o = orig_obs(run, i, tick, readings)
            self.obs.append((i, tick, copy.deepcopy(o)))
            return o

        def spy_q(q, tick, inputs):
            self.queen.append((tick, copy.deepcopy(dict(inputs))))
            return orig_q(q, tick, inputs)

        monkeypatch.setattr(ep._Run, "_observation", spy_obs)
        monkeypatch.setattr(ScriptedQueen, "__call__", spy_q)


def capture(arm, rules):
    with pytest.MonkeyPatch.context() as mp:
        spy = Spy(mp)
        rec = run(rules, arm, 2)
    return rec, spy


@pytest.mark.parametrize("arm", ANT_ARMS)
def test_observation_and_queen_keys_are_unchanged_with_the_rule_on(arm):
    from tests.v4_3.a2.helpers import R43
    rec, on = capture(arm, RF)
    _, off = capture(arm, R43)
    assert on.obs and all(set(o) == expected_keys(arm) for _, _, o in on.obs)
    assert rec["script_counters"]["probes"] > 0
    assert {frozenset(o) for _, _, o in on.obs} == {frozenset(o) for _, _, o in off.obs}
    assert on.queen and {frozenset(q) for _, q in on.queen} == {frozenset(q) for _, q in off.queen}


@pytest.mark.parametrize("arm", ANT_ARMS)
def test_only_the_sharing_arms_ever_pass_the_marks_input(arm, monkeypatch):
    seen = []
    orig = cp.uptake_info

    def spy(inp, pos, ant, rules):
        seen.append(inp.marks)
        return orig(inp, pos, ant, rules)

    monkeypatch.setattr(cp, "uptake_info", spy)
    run(RF, arm, 1)
    assert seen
    assert all((m is None) == (arm == "S-solo") for m in seen)


def test_the_information_is_a_function_of_the_log_and_the_marks_only():
    log = [uptake((10, 10, 12), 0.2, tick=1)]
    marks = [mark_row("good_uptake", (10, 10, 9), authors=(3,))]
    ref = cp.uptake_info(inputs(reading_log=log, marks=marks), POS, 0, RF)
    for kw in ({"look": 0.0}, {"scan_p": 0.0}, {"payload_fraction": 0.0}, {"dose_rel": 3.0}, {"region_status": {0: "done"}},
               {"messages": [{"type": "answer"}]}, {"uptake": {"word": "poor", "about_vox": [10, 10, 10], "value": 0.0}},
               {"zone_need": {0: "none"}}, {"visited_regions": frozenset()}):
        assert cp.uptake_info(inputs(reading_log=log, marks=marks, **kw), POS, 0, RF) == ref, kw


@pytest.mark.parametrize("name", B2_FUNCS)
def test_b2_code_reads_only_the_allowed_inputs(name):
    tree = ast.parse(inspect.getsource(getattr(cp, name)).lstrip())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "inp"}
    assert attrs <= ALLOWED_INP_ATTRS, sorted(attrs - ALLOWED_INP_ATTRS)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"world", "field", "truth", "crop", "scorer", "uptake_fn", "sensor_readings"}


def test_the_policy_module_imports_no_world_or_field():
    tree = ast.parse(inspect.getsource(cp))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.endswith(("tumor_mri.foraging", "tumor_mri.physics", "tumor_mri.colony_world")) for m in mods)
    assert "uptake_pending" not in inspect.getsource(cp)


def test_the_script_reads_only_the_observation_of_its_view():
    src = inspect.getsource(cp.ColonyScript.decide)
    assert src.count("view.") == src.count("view.observation") == 1
