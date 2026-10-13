import statistics

import pytest

from antelligence.kernel.board import Board
from antelligence.kernel.briefs import (
    BriefError, build_brief, build_brief_board_off, check_citations, citation_validity_rate,
    estimate_tokens, median_entry_age_in_briefs, never_read_share, uncited_decisions,
)
from tests.v4_1.vectors import generate_vectors as gv

OTHER = "OTHER-SQUAD-SECRET"


def _knowledge(region, source="perception"):
    return {
        "region_id": region, "centroid_dmm": [region, 2, 3], "cls": "ET", "p_tumour_ppm": 500000 + region,
        "volume_mm3": 10, "unc_ppm": 1000, "source": source, "prov": "unet@1",
    }


def _evidence(squad, note=0):
    return {"squad": squad, "pos_vox": [1, 2, 3], "look_p_ppm": 100 + note, "p_tumour_ppm": 200, "dose_ug": note, "released_ug": 0}


def make_board(n_knowledge=8, n_evidence=8):
    board = Board(gv.test_header(), mode="A")
    board.append("task", "queen", {"objective": "Treat the tumour core", "squads": [0, 1], "until_tick": 50}, 0)
    for r in range(n_knowledge):
        board.append("knowledge", "perception", _knowledge(r), 1)
    board.append("knowledge", "perception", _knowledge(98, "local_look"), 2)
    for i in range(n_evidence):
        board.append("evidence", "squad-1", _evidence(1, i), 3 + i)
    board.append("decision", "squad-1", {
        "squad": 1, "round": 1, "target": "B", "release": 0, "amount_level": 1, "p_target_ppm": 1, "p_release_ppm": 1,
        "p_amount_ppm": 1, "backend": OTHER, "deferred": 0, "router_pick": "slm", "brief_ids": []}, 20)
    board.append("route", "queen", {"squad": 1, "region_ids": [5], "priority": 1, "brief_ids": []}, 21)
    route0 = board.append("route", "queen", {"squad": 0, "region_ids": [3, 1], "priority": 2, "brief_ids": []}, 22)
    return board, route0


def test_brief_size_between_6_and_12():
    board, route = make_board()
    brief = build_brief(board, 0, route)
    assert 6 <= len(brief.entries) <= 12
    assert list(brief.brief_ids) == [e.id for e in brief.entries]
    assert brief.entries[0].type == "task" and route.id in brief.brief_ids
    # priority regions come first
    knowledge = [e for e in brief.entries if e.type == "knowledge"]
    assert [e.payload["region_id"] for e in knowledge[:2]] == [3, 1]
    assert all(i in board for i in brief.brief_ids)
    # fewer than 6 entries exist: all of them
    tiny = Board(gv.test_header(), mode="A")
    tiny.append("task", "queen", {"objective": "o", "squads": [0], "until_tick": 1}, 0)
    r = tiny.append("route", "queen", {"squad": 0, "region_ids": [], "priority": 1, "brief_ids": []}, 1)
    assert len(build_brief(tiny, 0, r).entries) == 2
    # a draft route (payload only) sees the whole board
    draft = {"squad": 0, "region_ids": [3], "priority": 1, "brief_ids": []}
    assert 6 <= len(build_brief(board, 0, draft).entries) <= 12
    with pytest.raises(BriefError):
        build_brief(board, 1, route)


def test_brief_under_token_cap():
    board, route = make_board()
    brief = build_brief(board, 0, route, max_tokens=1500)
    assert brief.tokens <= 1500 and brief.tokens == estimate_tokens(brief.text)
    tight = build_brief(board, 0, route, max_tokens=brief.tokens - 1)
    assert tight.tokens <= brief.tokens - 1 and len(tight.entries) < len(brief.entries)
    assert tight.brief_ids == brief.brief_ids[: len(tight.entries)]
    assert all(i in tight.text for i in tight.brief_ids)
    with pytest.raises(BriefError):
        build_brief(board, 0, route, max_tokens=50)
    assert build_brief(board, 0, route, tokenizer=lambda t: 1).tokens == 1


def test_invalid_citations_flagged():
    board, route = make_board()
    brief = build_brief(board, 0, route)
    outside = [e.id for e in board.entries() if e.id not in brief.brief_ids][:2]
    assert outside
    result = check_citations([brief.brief_ids[0], outside[0], brief.brief_ids[1], "ff" * 32], brief.brief_ids)
    assert result.valid == [brief.brief_ids[0], brief.brief_ids[1]]
    assert result.invalid == [outside[0], "ff" * 32]
    assert check_citations([], brief.brief_ids) == ([], [])
    # metrics over a board of decisions
    m = Board(gv.test_header(), mode="A")
    t = m.append("task", "queen", {"objective": "o", "squads": [0], "until_tick": 1}, 0)
    k = m.append("knowledge", "perception", _knowledge(1), 2)
    stray = m.append("knowledge", "perception", _knowledge(2), 3)
    dec = {
        "squad": 0, "round": 1, "target": "A", "release": 1, "amount_level": 1, "p_target_ppm": 1, "p_release_ppm": 1,
        "p_amount_ppm": 1, "backend": "b", "deferred": 0, "router_pick": "slm", "brief_ids": [t.id, k.id]}
    m.append("decision", "squad-0", dec, 10, cites=[k.id, stray.id])
    m.append("decision", "squad-0", dec, 12)
    assert citation_validity_rate(m) == 0.5
    assert len(uncited_decisions(m)) == 1
    assert never_read_share(m) == pytest.approx(3 / 5)   # briefable: task, 2 knowledge, 2 decisions; only task and k were read
    assert median_entry_age_in_briefs(m) == statistics.median([10, 8, 12, 10])
    assert citation_validity_rate(Board(gv.test_header(), mode="A")) is None


def test_board_off_brief_has_no_other_squad_entries():
    board, route = make_board()
    brief = build_brief_board_off(board, 0, route)
    types = {e.type for e in brief.entries}
    assert types <= {"task", "knowledge", "route"}
    for e in brief.entries:
        if e.type == "route":
            assert e.payload["squad"] == 0
        if e.type == "knowledge":
            assert e.author["agent"] == "perception" and e.payload["source"] == "perception"
    assert OTHER not in brief.text
    assert not any(e.type in ("evidence", "decision") for e in brief.entries)
    assert not any(e.author["agent"].startswith("squad-") for e in brief.entries)
    assert not any(e.type == "knowledge" and e.payload["source"] == "local_look" for e in brief.entries)
    assert brief.entries[0].type == "task" and route.id in brief.brief_ids
    assert len(brief.entries) <= 12 and brief.tokens <= 1500
    assert [e.payload["region_id"] for e in brief.entries if e.type == "knowledge"][:2] == [3, 1]
