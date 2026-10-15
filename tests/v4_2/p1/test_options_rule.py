"""Goal 11 item 3: the options rule. A fixed engine rule picks the A-F options; the queen never does."""

import ast
import copy
import inspect
import textwrap

import pytest

from antelligence.worlds.tumor_mri import colony_state
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from antelligence.worlds.tumor_mri.colony_state import LABELS, select_options, with_options
from antelligence.worlds.tumor_mri.data import Region
from antelligence.worlds.tumor_mri.questions import build_request, render_state
from tests.v4_2.p1.helpers import colony_fields

VMM = 2.0


def regions(n=10):
    """Region i sits i*10 voxels along x; the ant is at the origin, so the distance order is the id order."""
    return [Region(region_id=i, centroid_vox=(10.0 * i, 0.0, 0.0), p_tumour=0.9 - 0.01 * i, unc=0.1, volume_mm3=4000) for i in range(1, n + 1)]


def test_six_nearest_labelled_a_to_f():
    opts = select_options(regions(10), (0, 0, 0), VMM)
    assert [o["label"] for o in opts] == list("ABCDEF")
    assert [o["region_id"] for o in opts] == [1, 2, 3, 4, 5, 6]
    assert [o["dist_mm"] for o in opts] == [20.0, 40.0, 60.0, 80.0, 100.0, 120.0]


def test_done_and_not_tumour_are_left_out_and_the_next_nearest_move_up():
    status = {1: "done", 3: "not_tumour", 2: "open"}
    opts = select_options(regions(10), (0, 0, 0), VMM, status)
    assert [o["region_id"] for o in opts] == [2, 4, 5, 6, 7, 8]
    assert [o["label"] for o in opts] == list("ABCDEF")


def test_status_keys_may_be_strings():
    opts = select_options(regions(10), (0, 0, 0), VMM, {"1": "done"})
    assert opts[0]["region_id"] == 2


def test_fewer_than_six_open_regions():
    opts = select_options(regions(3), (0, 0, 0), VMM, {"2": "done"})
    assert [(o["label"], o["region_id"]) for o in opts] == [("A", 1), ("B", 3)]
    assert select_options(regions(2), (0, 0, 0), VMM, {1: "done", 2: "not_tumour"}) == []


def test_distance_depends_on_the_ants_own_position_and_ties_go_to_the_lower_id():
    rs = [Region(5, (10.0, 0, 0), 0.9, 0.1, 4000), Region(2, (-10.0, 0, 0), 0.5, 0.1, 4000), Region(9, (0, 40.0, 0), 0.9, 0.1, 4000)]
    assert [o["region_id"] for o in select_options(rs, (0, 0, 0), VMM)] == [2, 5, 9]      # 2 and 5 tie at 20 mm
    assert [o["region_id"] for o in select_options(rs, (0, 40, 0), VMM)][0] == 9


def test_stay_and_explore_are_added_by_the_question_builder():
    f = with_options(colony_fields(), regions(10), VMM)
    req = build_request("m", f)
    assert req.questions["target"].options == ("A", "B", "C", "D", "E", "F", "S", "X")


def test_with_options_replaces_candidates_and_keeps_the_rest():
    f = colony_fields()
    g = with_options(f, regions(10), VMM)
    assert g["candidates"] != f["candidates"] and {k: v for k, v in g.items() if k != "candidates"} == {k: v for k, v in f.items() if k != "candidates"}
    assert f["candidates"][0]["label"] == "A" and f is not g


# --- the queen can not touch the options ----------------------------------------------------------------------------------

def test_options_ignore_every_queen_output():
    rs = regions(10)
    base = colony_fields()
    base["position_vox"] = [0, 0, 0]
    ref = with_options(base, rs, VMM)["candidates"]
    for need, note in ((dict.fromkeys(range(1, 11), "high"), "x"), (dict.fromkeys(range(1, 11), "none"), "go to region 9 first"),
                       ({i: ("high" if i > 8 else "none") for i in range(1, 11)}, "")):
        f = copy.deepcopy(base)
        f["zone_need"], f["colony_note"], f["target_source"] = need, note, "queen"
        assert with_options(f, rs, VMM)["candidates"] == ref


def test_real_queen_outputs_cannot_change_the_options():
    rs = regions(10)
    pos = (0, 0, 0)
    no_marks = ScriptedQueen(rs, 10000.0, 8, VMM)(30, {"marks": [], "quorum_events": []})
    dosed = [{"mark_kind": "dosed", "pos_vox": [10, 0, 0], "amount_milli": 1000, "tick": 5, "ant": 1}] * 8
    other = ScriptedQueen(rs, 10000.0, 8, VMM)(30, {"marks": dosed, "quorum_events": [{"status": "done", "region_id": 1}]})
    assert no_marks["zone_need"] != other["zone_need"]                         # the queen's two outputs really differ
    assert set(no_marks) == {"zone_need", "colony_note", "need_rows"}          # she returns no options, targets or routes
    assert select_options(rs, pos, VMM, None) == select_options(rs, pos, VMM, None)
    # the only status the rule reads is the board's region status, never her need rows
    for q in (no_marks, other):
        f = colony_fields()
        f.update({"position_vox": [0, 0, 0], "zone_need": q["zone_need"], "colony_note": q["colony_note"]})
        assert [o["region_id"] for o in with_options(f, rs, VMM)["candidates"]] == [1, 2, 3, 4, 5, 6]


def _code_without_docstring(fn) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_select_options_signature_has_no_queen_input():
    params = list(inspect.signature(select_options).parameters)
    assert params == ["regions", "position_vox", "voxel_mm", "region_status", "max_options"]
    assert not any(w in " ".join(params).lower() for w in ("queen", "need", "note", "hint", "target"))
    src = "".join(_code_without_docstring(f) for f in (select_options, with_options))
    for word in ("zone_need", "colony_note", "target_source", "queen", "need_rows"):
        assert word not in src


def test_region_status_changes_do_change_the_options():
    """The control for the test above: the rule is not constant, it follows the board's status."""
    rs = regions(10)
    a = [o["region_id"] for o in select_options(rs, (0, 0, 0), VMM, {})]
    b = [o["region_id"] for o in select_options(rs, (0, 0, 0), VMM, {1: "done"})]
    assert a != b


def test_label_set_is_a_to_f():
    assert LABELS == tuple("ABCDEF") and colony_state.CLOSED == ("done", "not_tumour")


def test_state_shows_the_options_the_rule_picked():
    f = colony_fields()
    f["position_vox"] = [0, 0, 0]
    g = with_options(f, regions(10), VMM)
    assert "Regions: A " in render_state(g, max_chars=10_000)
