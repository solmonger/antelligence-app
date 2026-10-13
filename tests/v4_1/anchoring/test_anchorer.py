"""Anchorer: anchor points every 50 entries and at run end, anchors.json shape, failure isolation."""

import json

import pytest

from antelligence.anchoring import mocks
from antelligence.anchoring.anchorer import Anchorer, SingleShotAnchorer, single_shot_id
from antelligence.anchoring.tamper import Lab
from antelligence.kernel.board import Board, anchor_sizes, generate_run_keys


def build(lab, tmp_path, n, modes=("OTS", "R", "C"), finalize=True):
    """Drive a board the way the harness does: anchor hook after every append."""
    full, keys, akey = mocks.synthetic_run(n)
    board = Board(full.header, mode="B", keys=keys.seeds)
    anchorer = Anchorer(board.run_id, modes, tmp_path, ots=lab.ots_client(), rekor=lab.rekor, anchor_key=akey,
                        chain=lab.chain_mode())
    for e in full.entries():
        board.append(e.type, dict(e.author), dict(e.payload), e.tick)
        anchorer.maybe_anchor(board)
    if finalize:
        anchorer.finalize(board)
    return board, anchorer, akey


@pytest.mark.parametrize("n,expected", [(120, [50, 100, 120]), (100, [50, 100]), (49, [49]), (50, [50])])
def test_anchorer_points_every_50_and_end(tmp_path, n, expected):
    lab = Lab()
    board, anchorer, _ = build(lab, tmp_path, n)
    assert anchorer.anchored_sizes == expected == anchor_sizes(n)      # run end on a multiple of 50 is not repeated
    records = json.loads(anchorer.path.read_text())
    assert [r["size"] for r in records] == expected
    for rec, want in zip(records, board.anchors()):
        assert rec["size"] == want["size"] and rec["root"] == want["root"] and rec["digest"] == want["digest"]
        assert (tmp_path / rec["ots_path"]).exists()
        assert rec["rekor_uuid"] and isinstance(rec["rekor_log_index"], int)
    assert not anchorer.errors
    # Rekor holds exactly the anchor digests; the registry holds exactly the roots
    rid = bytes.fromhex(board.run_id)
    assert {r.digest for r in lab.rekor.entries_for_key(anchorer.anchor_key.pem)} == {w["digest"] for w in board.anchors()}
    for w in board.anchors():
        assert lab.registry.root_at[(rid, w["size"])].hex() == w["root"]


def test_anchorer_tail_without_finalize_leaves_tail_unanchored(tmp_path):
    board, anchorer, _ = build(Lab(), tmp_path, 120, finalize=False)
    assert anchorer.anchored_sizes == [50, 100]


def test_anchorer_failure_never_raises_into_the_run(tmp_path):
    lab = Lab()
    lab.ots_transport.fail = {"https://cal-a.mock", "https://cal-b.mock"}
    board, anchorer, _ = build(lab, tmp_path, 60)
    assert anchorer.anchored_sizes == [50, 60]
    assert {e["mode"] for e in anchorer.errors} == {"OTS"}
    assert all("ots_path" not in r and "rekor_uuid" in r for r in anchorer.records)


def test_anchorer_rejects_wrong_board_and_bad_config(tmp_path):
    lab = Lab()
    board, anchorer, akey = build(lab, tmp_path, 10)
    other, _, _ = mocks.synthetic_run(10, seed=9)
    with pytest.raises(ValueError):
        anchorer.anchor(other, 10)
    with pytest.raises(ValueError):
        Anchorer(board.run_id, {"R"}, tmp_path, rekor=lab.rekor)         # R without a key
    with pytest.raises(ValueError):
        Anchorer(board.run_id, {"D"}, tmp_path)


def test_single_shot_anchors_once(tmp_path):
    lab = Lab()
    s = lab.single_shot(tmp_path)
    rec = s.anchor("prereg", b"\x07" * 32)
    assert rec["ots_path"] == "prereg.ots" and rec["rekor_uuid"]
    assert lab.registry.once[single_shot_id("prereg")] == b"\x07" * 32
    with pytest.raises(ValueError):
        s.anchor("prereg", b"\x08" * 32)
    s2 = lab.single_shot(tmp_path / "again")
    s2.anchor("prereg", b"\x08" * 32)              # a second anchoring attempt: chain refuses, recorded not raised
    assert any(e["mode"] == "C" for e in s2.errors)
    assert lab.registry.once[single_shot_id("prereg")] == b"\x07" * 32
