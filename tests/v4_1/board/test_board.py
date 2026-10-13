import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from antelligence.kernel.board import (
    Board, BoardError, BoardIntegrityError, ValidationError, anchor_digest, anchor_sizes,
    compute_id, consistency_path, generate_run_keys, inclusion_path, load_knowledge_from, merkle_root,
    pubkey_hex, validate_payload, verify_consistency, verify_inclusion, verify_sig,
)
from antelligence.kernel.canonical import CanonicalError, jcs_bytes, jcs_json
from tests.v4_1.vectors import generate_vectors as gv

VECTORS = json.loads((Path(gv.__file__).with_name("interfaces.json")).read_text(encoding="utf-8"))


def _rewrite(path, index, mutate):
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[index])
    mutate(row)
    lines[index] = jcs_json(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_vectors_file_matches_generator():
    assert VECTORS == json.loads(json.dumps(gv.generate(), ensure_ascii=False))


def test_entry_id_is_jcs_sha256():
    for row in VECTORS["entries"]:
        body = {k: v for k, v in row["entry"].items() if k not in ("id", "sig")}
        assert row["body_jcs"] == jcs_json(body)
        assert hashlib.sha256(row["body_jcs"].encode("utf-8")).hexdigest() == row["id"] == compute_id(body)
    board = gv.build_board("B")
    assert [e.id for e in board.entries()] == [r["id"] for r in VECTORS["entries"]]
    assert [e.sig for e in board.entries()] == [r["sig_mode_b"] for r in VECTORS["entries"]]
    for case in VECTORS["jcs"]["accept"]:
        assert jcs_json(case["value"]) == case["canonical"]
    # UTF-16 key order differs from code point order for astral characters
    assert jcs_json({"\U0001F600": 1, "￿": 2}).startswith('{"\U0001F600"')


def test_float_rejected():
    board = gv.build_board("A")
    for bad in (1.5, 2 ** 53, True):
        with pytest.raises(ValidationError):
            board.append("alarm", "router", {"kind": "other", "value_ppm": bad, "note": "x"}, 1)
    with pytest.raises(CanonicalError):
        jcs_bytes({"a": 1.0})
    for case in VECTORS["payload_rejects"]:
        with pytest.raises(ValidationError):
            validate_payload(case["type"], case["payload"])
    assert board.size() == 8


def test_signature_failure_on_tamper(tmp_path):
    board = gv.build_board("B")
    path = tmp_path / "run.jsonl"
    board.export_jsonl(path)
    Board.load_jsonl(path)  # clean copy verifies

    def forge(row):
        row["payload"]["p_tumour_ppm"] = 1
        row["id"] = compute_id(row)  # attacker fixes the id but cannot re-sign

    _rewrite(path, 2, forge)
    with pytest.raises(BoardIntegrityError):
        Board.load_jsonl(path)
    pub = VECTORS["keys"]["pubkeys_hex"]["queen"]
    good = VECTORS["entries"][0]
    assert verify_sig(pub, good["id"], good["sig_mode_b"])
    assert not verify_sig(pub, "00" * 32, good["sig_mode_b"])
    assert not verify_sig(VECTORS["keys"]["pubkeys_hex"]["router"], good["id"], good["sig_mode_b"])


def test_prev_hash_break_detected(tmp_path):
    board = gv.build_board("A")
    path = tmp_path / "run.jsonl"
    board.export_jsonl(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[3]  # drop seq 2: seq and chain both break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(BoardIntegrityError):
        Board.load_jsonl(path)
    board.export_jsonl(path)
    _rewrite(path, 4, lambda row: row.update(prev_hash="11" * 32))  # id covers prev_hash, so this is caught too
    with pytest.raises(BoardIntegrityError):
        Board.load_jsonl(path)
    board.export_jsonl(path)

    def relink(row):
        row["prev_hash"] = "11" * 32
        row["id"] = compute_id(row)  # consistent id, but the chain no longer links

    _rewrite(path, 4, relink)
    with pytest.raises(BoardIntegrityError):
        Board.load_jsonl(path)


def test_export_roundtrip(tmp_path):
    for mode in ("A", "B"):
        board = gv.build_board(mode)
        path = tmp_path / f"{mode}.jsonl"
        board.export_jsonl(path)
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == jcs_json(board.header)
        assert len(text.splitlines()) == 1 + board.size()
        loaded = Board.load_jsonl(path)
        assert loaded.mode == mode and loaded.run_id == board.run_id
        assert loaded.entries() == board.entries()
        loaded.export_jsonl(tmp_path / f"{mode}2.jsonl")
        assert (tmp_path / f"{mode}2.jsonl").read_text(encoding="utf-8") == text


def test_rfc6962_roots_match_vectors():
    board = gv.build_board("B")
    for n, root in VECTORS["merkle"]["roots"].items():
        assert board.root(int(n)).hex() == root
    assert board.root(0) == hashlib.sha256(b"").digest()
    leaves = [bytes.fromhex(e.id) for e in board.entries()]
    for case in VECTORS["merkle"]["inclusion"]:
        path = [bytes.fromhex(p) for p in case["path"]]
        assert board.inclusion_proof(case["index"], case["size"]) == path
        assert verify_inclusion(bytes.fromhex(case["leaf_id"]), case["index"], case["size"], path, bytes.fromhex(case["root"]))
    assert leaves  # sanity


def _naive_root(leaves):
    """Independent implementation: perfect blocks from the left, folded right to left (RFC 6962 shape)."""
    level = [hashlib.sha256(b"\x00" + d).digest() for d in leaves]
    if not level:
        return hashlib.sha256(b"").digest()
    blocks, i, n = [], 0, len(level)
    while i < n:
        size = 1
        while size * 2 <= n - i:
            size *= 2
        chunk = level[i:i + size]
        while len(chunk) > 1:
            chunk = [hashlib.sha256(b"\x01" + chunk[j] + chunk[j + 1]).digest() for j in range(0, len(chunk), 2)]
        blocks.append(chunk[0])
        i += size
    root = blocks[-1]
    for left in reversed(blocks[:-1]):
        root = hashlib.sha256(b"\x01" + left + root).digest()
    return root


def test_merkle_matches_independent_implementation():
    leaves = [hashlib.sha256(bytes([i])).digest() for i in range(40)]
    for n in range(0, 40):
        assert merkle_root(leaves[:n]) == _naive_root(leaves[:n])


def test_anchor_digest_vector():
    for case in VECTORS["anchor"]["digests"]:
        pre = bytes.fromhex(case["run_id"]) + int(case["size"]).to_bytes(8, "big") + bytes.fromhex(case["root"])
        assert len(pre) == 72 and pre.hex() == case["preimage_hex"]
        assert hashlib.sha256(pre).hexdigest() == case["digest"]
        assert anchor_digest(case["run_id"], case["size"], bytes.fromhex(case["root"])).hex() == case["digest"]
    board = gv.build_board("B")
    assert board.anchor_digest(8).hex() == VECTORS["anchor"]["digests"][-1]["digest"]
    assert anchor_sizes(0) == [] and anchor_sizes(8) == [8] and anchor_sizes(100) == [50, 100]
    assert anchor_sizes(120) == [50, 100, 120] and anchor_sizes(51) == [50, 51]
    assert board.anchors()[-1]["digest"] == board.anchor_digest().hex()


def test_consistency_proof():
    board = gv.build_board("A")
    for case in VECTORS["merkle"]["consistency"]:
        proof = [bytes.fromhex(p) for p in case["proof"]]
        assert board.consistency_proof(case["old_size"], case["new_size"]) == proof
        assert verify_consistency(case["old_size"], case["new_size"], bytes.fromhex(case["old_root"]), bytes.fromhex(case["new_root"]), proof)
    leaves = [hashlib.sha256(bytes([i])).digest() for i in range(20)]
    for n in range(1, 21):
        for m in range(1, n + 1):
            proof = consistency_path(m, leaves[:n])
            assert verify_consistency(m, n, merkle_root(leaves[:m]), merkle_root(leaves[:n]), proof)
            if proof:
                assert not verify_consistency(m, n, merkle_root(leaves[:m]), merkle_root(leaves[:n]), [bytes(32)] + proof[1:])
        for i in range(n):
            assert verify_inclusion(leaves[i], i, n, inclusion_path(i, leaves[:n]), merkle_root(leaves[:n]))
    other = [hashlib.sha256(b"x" + bytes([i])).digest() for i in range(8)]
    assert not verify_consistency(3, 8, merkle_root(leaves[:3]), merkle_root(other), consistency_path(3, other))


def test_mode_a_unsigned_keeps_prev_hash():
    a, b = gv.build_board("A"), gv.build_board("B")
    assert all(e.sig is None for e in a.entries())
    assert all(e.sig is not None for e in b.entries())
    assert a.entries()[0].prev_hash == "0" * 64
    for prev, cur in zip(a.entries(), a.entries()[1:]):
        assert cur.prev_hash == prev.id
    assert [e.id for e in a.entries()] == [e.id for e in b.entries()]
    assert VECTORS["mode_a_ids_equal_mode_b_ids"] is True


def test_load_knowledge_from_earlier_run(tmp_path):
    board = gv.build_board("B")
    path = tmp_path / "earlier.jsonl"
    board.export_jsonl(path)
    got = load_knowledge_from(path)
    assert [e.id for e in got] == [e.id for e in board.by_type("knowledge")]
    assert len(got) == 2 and all(e.type == "knowledge" for e in got)
    path.write_text(path.read_text(encoding="utf-8").replace('"p_tumour_ppm":873000', '"p_tumour_ppm":1'), encoding="utf-8")
    with pytest.raises(BoardIntegrityError):
        load_knowledge_from(path)


def test_cites_must_exist_and_are_sorted():
    board = gv.build_board("A")
    with pytest.raises(ValidationError):
        board.append("alarm", "router", {"kind": "other", "value_ppm": 1, "note": "x"}, 11, cites=["ab" * 32])
    ids = [e.id for e in board.entries()[:3]]
    e = board.append("alarm", "router", {"kind": "other", "value_ppm": 1, "note": "x"}, 11, cites=reversed(ids + ids))
    assert list(e.cites) == sorted(ids)


def test_mode_b_requires_matching_keys():
    header = gv.test_header()
    with pytest.raises(BoardError):
        Board(header, mode="B", keys={"queen": gv.test_seed("router")})
    board = Board(header, mode="B", keys={"queen": gv.test_seed("queen")})
    with pytest.raises(BoardError):
        board.append("alarm", "router", {"kind": "other", "value_ppm": 1, "note": "x"}, 0)


def test_sqlite_is_append_only_and_reopens(tmp_path):
    db = tmp_path / "run.sqlite"
    board = Board(gv.test_header(), mode="A", db_path=db)
    board.append("alarm", "router", {"kind": "other", "value_ppm": 1, "note": "x"}, 0)
    with pytest.raises(sqlite3.DatabaseError):
        board._db.execute("UPDATE entries SET body='{}' WHERE seq=0")
    with pytest.raises(sqlite3.DatabaseError):
        board._db.execute("DELETE FROM entries")
    board.close()
    again = Board(gv.test_header(), mode="A", db_path=db)
    assert again.size() == 1 and again.get(0).type == "alarm"
    again.close()
    with pytest.raises(BoardError):
        Board(dict(gv.test_header(), seed=8), mode="A", db_path=db)


def test_generate_run_keys_writes_only_under_ledger(tmp_path):
    keys = generate_run_keys(["queen", "squad-0"])
    assert set(keys.pubkeys) == {"queen", "squad-0"}
    assert all(pubkey_hex(s) == keys.pubkeys[a] for a, s in keys.seeds.items())
    assert generate_run_keys(["queen"]).pubkeys["queen"] != keys.pubkeys["queen"]
    with pytest.raises(BoardError):
        generate_run_keys(["queen"], ledger_dir=tmp_path / "elsewhere")
    k = generate_run_keys(["queen"], ledger_dir=tmp_path / "ledger" / "r1", label="r1")
    written = tmp_path / "ledger" / "r1" / "r1.keys.json"
    assert json.loads(written.read_text())["queen"] == k.seeds["queen"].hex()
    assert written.stat().st_mode & 0o077 == 0
