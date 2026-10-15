"""B1 section 4: the new ``good_uptake`` kind must not move any v4.2 entry. Golden values were computed at the base
commit (64b678d) BEFORE the change: board = ``gv.build_board("A")``, then 7 mark entries (one per old kind, in kind order,
tick 40) and one ask message (tick 41), appended in that order."""

import hashlib

import pytest

from antelligence.anchoring.chain_modes import MARK_KIND_CODES, compact_entry, payload_ints
from antelligence.kernel.board import MARK_KINDS
from antelligence.kernel.colony.marks import DEFAULT_HALF_LIFE, KIND_CODES
from antelligence.kernel.colony.messages import Message, message_payload
from tests.v4_1.vectors import generate_vectors as gv

OLD_KINDS = ("dosed", "dose_ok", "slow_uptake", "washout", "tumour_here", "not_tumour", "need_more")
GOLDEN = {   # kind -> (entry id, compact sha256)
    "dosed": ("1844f83c4103039a9129925cf85938aa7bbfcdc453e79c8e0dc6165b02673540", "a62abc6ce8b24c546325aae6396ab84049ddc8efa876236c3acc462a33d6b1bf"),
    "dose_ok": ("289e92662f152f1b6e000a60d98e076811ed8a86a36085864a1a1159ee1813f7", "dfba31b983b41bcb25b0de34a18fb8fd2cd8f8dfd662fe1cdc4e6d0c4bc01826"),
    "slow_uptake": ("2ea07afb76d8afe78193997acd57c799586241d5c92ed79cc3954f2a2498e693", "34efaeabcf815bfd0f2992c54d67a316d4c05f7e8030b702d4d93cb41fd55868"),
    "washout": ("97a01dc95e533f7d345da9dafc55e16e3a54e91d3c4b1907585f66574505ad0d", "60d0b24ba142af328e1ab98085356203c126538a87e54b75f7ff59cd6fb3ff52"),
    "tumour_here": ("ae239a61930ee0995537674b4d00b2c697d068b94996695096185d14352028eb", "6e085e6c629ffe13a7657af9903db6a5f13a7b21a7238f48433c1e816dac6973"),
    "not_tumour": ("73c0aec164779aab53fb960045a09478d4115e0892ccec1f7de479788cb5026b", "df096642f9dd100ac39d0a28fa2ded7fa43e85dcd995c3df0a2a5eb359eca36c"),
    "need_more": ("2219ae3df3cbc4fd5032e392b52d71ef0cda1e47084ad9af80389fc48fd74044", "9965295d1577b32a23f03ae39cdba366e8165fb498c49b74b915eb2428f64408"),
}
MESSAGE_ID = "6e1e638a3506953ab39e6d8e8e5cfc70342b6045156b62d851810941571e0b38"
MESSAGE_INTS = [1, 12, 3, 5, 20, 21, 22, 7, 1, 540000, -1, -1, -1]   # not abi-encodable (-1), so the id and ints are pinned


def _entries():
    b = gv.build_board("A")
    out = {}
    for k in OLD_KINDS:
        out[k] = b.append("evidence", "squad-3", {"subtype": "mark", "mark_kind": k, "ant": 3, "pos_vox": [10, 11, 12],
                                                  "amount_milli": 1000, "reading_ppm": 540000}, 40)
    ask = Message(12, "ask", 3, 5, 40, {"region_id": 7, "pos_vox": None}, "is_tumour", 0.54, note="look here")
    out["message"] = b.append("evidence", "squad-3", message_payload(ask, (20, 21, 22)), 41)
    return out


@pytest.mark.parametrize("kind", OLD_KINDS)
def test_old_mark_entries_unchanged(kind):
    e = _entries()[kind]
    assert e.id == GOLDEN[kind][0]
    assert hashlib.sha256(compact_entry(e)).hexdigest() == GOLDEN[kind][1]
    assert payload_ints(e)[4] == OLD_KINDS.index(kind) + 1


def test_message_entry_unchanged():
    e = _entries()["message"]
    assert e.id == MESSAGE_ID and payload_ints(e) == MESSAGE_INTS


def test_new_kind_appended_after_the_old_seven():
    assert MARK_KINDS[:7] == OLD_KINDS and MARK_KINDS[7:] == ("good_uptake",)
    assert KIND_CODES["good_uptake"] == 8 and MARK_KIND_CODES["good_uptake"] == 8
    assert {k: MARK_KIND_CODES[k] for k in OLD_KINDS} == {k: i + 1 for i, k in enumerate(OLD_KINDS)}
    assert DEFAULT_HALF_LIFE["good_uptake"] == 60 and set(DEFAULT_HALF_LIFE) == set(MARK_KINDS)


def test_good_uptake_entry_encodes_with_code_8():
    e = gv.build_board("A").append("evidence", "squad-3", {"subtype": "mark", "mark_kind": "good_uptake", "ant": 3,
                                                           "pos_vox": [10, 11, 12], "amount_milli": 1000, "reading_ppm": 540000}, 40)
    assert payload_ints(e) == [3, 10, 11, 12, 8, 1000, 540000] and compact_entry(e)
