"""Goal 11 item 1: render_state with colony inputs; v4.1 output unchanged when they are absent (golden)."""

import json
from pathlib import Path

import pytest

from antelligence.kernel.decision import MAX_STATE_CHARS
from antelligence.worlds.tumor_mri.colony_obs import COLONY_OBS_KEYS
from antelligence.worlds.tumor_mri.colony_state import (NOTEBOOK_CHARS, colony_sentences, derive_voxel_mm, has_colony_fields,
                                                        notebook_sentence)
from antelligence.worlds.tumor_mri.questions import build_request, render_state
from tests.v4_2.p1.helpers import base_fields, colony_fields, mark_row

GOLDEN = json.loads((Path(__file__).parent / "golden_v41_render_state.json").read_text())


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_v41_output_is_byte_identical_without_colony_inputs(case):
    assert not has_colony_fields(case["fields"])
    assert render_state(case["fields"], case["brief"]) == case["expected"]


def test_colony_text_starts_with_the_v41_text():
    f = colony_fields()
    plain = render_state(base_fields())
    assert render_state(f, max_chars=10_000).startswith(plain)


def test_marks_near_you_has_author_and_band():
    s = render_state(colony_fields(), max_chars=10_000)
    assert "Marks near you:" in s
    assert "tumour here from ant-03 (usually right) and ant-05 (new), strong" in s
    assert "not tumour from ant-06 (mixed)" in s
    assert "drug given by 2 ants" in s


def test_own_marks_read_you_not_a_band():
    f = colony_fields()
    f["marks_near"] = [mark_row("tumour_here", (9, 10, 10), [2])]
    assert "tumour here from you" in render_state(f, max_chars=10_000)


def test_unknown_author_reads_new():
    f = colony_fields()
    f["marks_near"] = [mark_row("tumour_here", (9, 10, 10), [7])]
    f["cards_public"] = []
    assert "ant-07 (new)" in render_state(f, max_chars=10_000)


def test_disputed_is_said_plainly():
    f = colony_fields()
    f["marks_near"] = [mark_row("tumour_here", (9, 10, 10), [3], disputed=True), mark_row("not_tumour", (9, 10, 10), [4], disputed=True)]
    assert "disputed" in render_state(f, max_chars=10_000)


def test_no_marks_near():
    f = colony_fields()
    f["marks_near"] = []
    assert "No marks near you." in render_state(f, max_chars=10_000)


def test_per_region_summary_maps_marks_to_zones_with_need():
    s = colony_sentences(colony_fields())["zones"]
    # tumour_here at voxel (9,10,10) is inside zone A (centroid (10,10,10)); need_more at (40,10,10) is zone C's centroid
    assert s.startswith("Zones: A need high, tumour here from ant-03 (usually right) and ant-05 (new)")
    assert "B need low" in s and "C need none, needs more drug from ant-04 (mixed)" in s


def test_messages_in_and_note_is_quoted_not_obeyed():
    s = colony_sentences(colony_fields())["messages"]
    assert s.startswith("Messages: ant-03 asks is there tumour at region B? Its own reading: very likely.")
    assert 'Note: "look at the north edge"' in s


def test_answer_message_wording():
    f = colony_fields()
    f["messages_in"] = [{"msg_id": 8, "type": "answer", "from_ant": 4, "to_ant": 2, "tick": 14, "about": {"region_id": 1, "pos_vox": None},
                         "question": "is_tumour", "sender_reading": 0.4, "reply_to": 3, "answer": "yes", "answerer_reading": 0.6, "note": ""},
                        {"msg_id": 9, "type": "answer", "from_ant": 5, "to_ant": 2, "tick": 14, "about": {"region_id": None, "pos_vox": [3, 4, 5]},
                         "question": "is_tumour", "sender_reading": 0.4, "reply_to": 4, "answer": "unknown", "answerer_reading": None, "note": ""}]
    s = colony_sentences(f)["messages"]
    assert "ant-04 answers yes about region A (its readings: likely)." in s
    assert "ant-05 answers about voxel (3, 4, 5): no reading there." in s


def test_only_four_latest_messages_are_shown():
    f = colony_fields()
    m = f["messages_in"][0]
    f["messages_in"] = [{**m, "msg_id": i, "from_ant": 3} for i in range(7)]
    s = colony_sentences(f)["messages"]
    assert s.count("asks") == 4 and "3 older not shown" in s


def test_readings_and_sensor_words():
    s = colony_sentences(colony_fields())["readings"]
    assert "You looked 2 times near here; tumour likely." in s        # the third look is far away
    assert "Uptake after your last release: slow." in s and "Washout here: fast." in s


def test_colony_note_is_included():
    assert "Colony note: Colony note, tick 15: 12% delivered." in render_state(colony_fields(), max_chars=10_000)


def test_notebook_is_capped_at_300_characters():
    f = colony_fields()
    f["notebook_self"] = ["L" * 190, "M" * 190]
    s = notebook_sentence(f)
    inner = s[len('Your notebook: "'):-1]
    assert len(inner) <= NOTEBOOK_CHARS and inner.endswith("...")
    assert notebook_sentence({"notebook_self": []}) == ""


def test_absent_keys_leave_their_sentence_out():
    f = colony_fields()
    for k in ("marks_near", "messages_in", "notebook_self"):
        f.pop(k)
    s = colony_sentences(f)
    assert s["marks"] == "" and s["messages"] == "" and s["notebook"] == ""


def test_over_the_limit_whole_sentences_drop_lowest_priority_first():
    f = colony_fields()
    full = render_state(f, max_chars=10_000)
    for limit in range(len(render_state(base_fields())), len(full) + 1, 37):
        s = render_state(f, max_chars=limit)
        assert len(s) <= limit
    no_nb = render_state({**f, "notebook_self": []}, max_chars=10_000)
    s = render_state(f, max_chars=len(full) - 1)
    assert "notebook" not in s and s == no_nb                    # the notebook goes first


def test_drop_order_keeps_messages_longest():
    f = colony_fields()
    sents = colony_sentences(f)
    base = render_state(base_fields())
    limit = len(base) + len(sents["messages"]) + 1
    s = render_state(f, max_chars=limit)
    assert "Messages:" in s and "Marks near you" not in s


def test_voxel_mm_is_derived_from_candidates():
    assert derive_voxel_mm(colony_fields()) == pytest.approx(2.0)
    assert derive_voxel_mm({"position_vox": [0, 0, 0], "candidates": []}) is None


def test_default_limit_is_v41_and_requests_still_build():
    f = colony_fields()
    assert len(render_state(f)) <= MAX_STATE_CHARS
    req = build_request("m", f, with_mark=True)
    assert "mark" in req.questions and len(req.state) <= MAX_STATE_CHARS


def test_no_truth_words_in_the_state():
    s = render_state(colony_fields(), max_chars=10_000).lower()
    for bad in ("truth", "ground", "oracle", "scorer", "label"):
        assert bad not in s


def test_colony_keys_all_known():
    from antelligence.worlds.tumor_mri.colony_state import COLONY_FIELD_KEYS
    assert set(COLONY_FIELD_KEYS) <= set(COLONY_OBS_KEYS)
