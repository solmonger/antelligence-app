"""Evidence subtypes (INTERFACES v4.2 section 5): v4.1 entries unchanged, each subtype round-trips, wrong keys refused."""

import hashlib

import pytest

from antelligence.anchoring.chain_modes import compact_entry, payload_ints
from antelligence.kernel.board import ValidationError, validate_payload  # noqa: F401
from antelligence.kernel.colony.messages import Message, message_payload
from tests.v4_1.vectors import generate_vectors as gv

# Golden values taken from the v4.1 code before the v4.2 change (tests/v4_1 fixture, entry 2).
V41_EVIDENCE_ID = "280c0f7a935bf50118cc6240279005157d43f0aa3e219c356b419e8a32c17fe1"
V41_EVIDENCE_INTS = [0, 60, 61, 62, 910000, 873000, 0, 125000]
V41_EVIDENCE_COMPACT_SHA = "c303cad96d2c4597f2c8c5a4687d5921f6de6f79cc5bdcf63db5811420d23c22"

MARK = {"subtype": "mark", "mark_kind": "tumour_here", "ant": 3, "pos_vox": [10, 11, 12], "amount_milli": 1000,
        "reading_ppm": 540000}
STATUS = {"subtype": "status", "region_id": 7, "status": "not_tumour", "k": 2, "tick_first": 35}
NEED = {"subtype": "need", "region_id": 7, "need": "high", "need_milli": 910, "delivered_milli": 120}


def _message():
    ask = Message(12, "ask", 3, 5, 40, {"region_id": 7, "pos_vox": None}, "is_tumour", 0.54, note="look here")
    return message_payload(ask, (20, 21, 22))


def _board():
    return gv.build_board("A")


def test_v41_evidence_validates_hashes_and_encodes_unchanged():
    e = _board().entries()[2]
    assert e.type == "evidence" and "subtype" not in e.payload
    assert e.id == V41_EVIDENCE_ID
    assert payload_ints(e) == V41_EVIDENCE_INTS
    assert hashlib.sha256(compact_entry(e)).hexdigest() == V41_EVIDENCE_COMPACT_SHA
    validate_payload("evidence", dict(e.payload))


def test_v41_evidence_still_refuses_extra_or_missing_keys():
    base = dict(_board().entries()[2].payload)
    with pytest.raises(ValidationError):
        validate_payload("evidence", {**base, "extra": 1})
    with pytest.raises(ValidationError):
        validate_payload("evidence", {k: v for k, v in base.items() if k != "dose_ug"})


@pytest.mark.parametrize("payload,ints", [
    (MARK, [3, 10, 11, 12, 5, 1000, 540000]),
    (STATUS, [7, 2, 2, 35]),
    (NEED, [7, 3, 910, 120]),
])
def test_subtype_round_trip_and_compact(payload, ints):
    b = _board()
    entry = b.append("evidence", "squad-3", payload, 40)
    assert dict(entry.payload) == payload
    assert payload_ints(entry) == ints
    assert compact_entry(entry)                        # abi-encodes without error
    assert b.get(entry.seq).id == entry.id


def test_message_round_trip_and_compact_has_no_note():
    p = _message()
    b = _board()
    entry = b.append("evidence", "squad-3", p, 41)
    assert payload_ints(entry) == [1, 12, 3, 5, 20, 21, 22, 7, 1, 540000, -1, -1, -1]
    assert "note" in entry.payload and entry.payload["note"] == "look here"   # the id commits to the note
    assert len(payload_ints(entry)) == 13


def test_note_changes_entry_id_but_not_compact_ints():
    a = _board().append("evidence", "squad-3", _message(), 41)
    b = _board().append("evidence", "squad-3", {**_message(), "note": "other"}, 41)
    assert a.id != b.id and payload_ints(a) == payload_ints(b)


@pytest.mark.parametrize("bad", [
    {**MARK, "extra": 1},
    {k: v for k, v in MARK.items() if k != "reading_ppm"},
    {**MARK, "mark_kind": "unknown"},
    {**MARK, "pos_vox": [1, 2]},
    {**MARK, "amount_milli": -1},
    {**MARK, "reading_ppm": 1_000_001},
    {**STATUS, "status": "open"},
    {**NEED, "need": "extreme"},
    {"subtype": "bogus"},
    {"subtype": ["mark"]},
])
def test_wrong_key_sets_and_values_refused(bad):
    with pytest.raises(ValidationError):
        validate_payload("evidence", bad)
    with pytest.raises(ValidationError):
        _board().append("evidence", "squad-3", bad, 40)


def test_message_refusals():
    p = _message()
    for bad in ({**p, "question": 4}, {**p, "answer": 3}, {**p, "note": "x" * 121}, {**p, "msg_type": "tell"}):
        with pytest.raises(ValidationError):
            validate_payload("evidence", bad)
