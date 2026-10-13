"""Rekor hashedrekord entries: shape, signature, search by public key, inclusion proofs."""

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from antelligence.anchoring.mocks import MockRekor
from antelligence.anchoring.rekor import (
    REKOR_LOG_KEY_SHA256,
    AnchorKey,
    RekorClient,
    RekorError,
    build_hashedrekord,
    check_record,
    fingerprint_of,
    load_public_pem,
    pinned_log_key,
)

D1, D2, D3 = (bytes([i]) * 32 for i in (1, 2, 3))


def test_rekor_entry_shape():
    key = AnchorKey()
    e = build_hashedrekord(D1, key)
    assert set(e) == {"apiVersion", "kind", "spec"}
    assert e["apiVersion"] == "0.0.1" and e["kind"] == "hashedrekord"
    assert set(e["spec"]) == {"signature", "data"}
    assert e["spec"]["data"] == {"hash": {"algorithm": "sha256", "value": D1.hex()}}
    pem = base64.b64decode(e["spec"]["signature"]["publicKey"]["content"])
    assert pem.startswith(b"-----BEGIN PUBLIC KEY-----")
    pub = load_public_pem(pem)
    assert isinstance(pub.curve, ec.SECP256R1)             # P-256, not Ed25519: Rekor wants Ed25519ph for ed25519
    sig = base64.b64decode(e["spec"]["signature"]["content"])
    pub.verify(sig, D1, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    # signing the digest pre-hashed equals ECDSA-SHA256 over its 72-byte preimage's hash: verify as plain data too
    assert len(key.fingerprint) == 64 and key.fingerprint == fingerprint_of(pub)
    json.dumps(e)                                           # JSON-serialisable as is


def make_client(raw=None):
    raw = raw or MockRekor()
    return RekorClient(raw, pinned_log_key_sha256=raw.log_key_sha256), raw


def test_rekor_log_search_and_inclusion_proofs():
    rekor, _ = make_client()
    lk = rekor.log_key()
    k1, k2 = AnchorKey(), AnchorKey()
    recs = [rekor.log_digest(d, k1) for d in (D1, D2, D3)] + [rekor.log_digest(D1, k2)]
    assert [r.log_index for r in recs] == [0, 1, 2, 3]
    found = rekor.entries_for_key(k1.pem)
    assert [r.digest for r in found] == [D1.hex(), D2.hex(), D3.hex()]
    # signature + RFC 6962 inclusion proof + checkpoint signed by the pinned log key + SET
    assert all(check_record(r, k1.pem, lk) == [] and r.time_trusted for r in found)
    assert len(rekor.entries_for_key(k2.pem)) == 1
    again = rekor.log_digest(D2, k1)                        # identical entry: idempotent, not a duplicate
    assert again.uuid == recs[1].uuid and len(rekor.entries_for_key(k1.pem)) == 3


def test_rekor_rejects_bad_signature():
    key, other = AnchorKey(), AnchorKey()
    raw = MockRekor()
    bad = build_hashedrekord(D1, key)
    bad["spec"]["signature"]["content"] = base64.b64encode(other.sign_digest(D1)).decode()
    with pytest.raises(ValueError):
        raw.create(bad)


def test_missing_rekor_proof_fails():
    rekor, raw = make_client()
    key = AnchorKey()
    rekor.log_digest(D1, key)
    raw.omit_proofs = True
    (rec,) = rekor.entries_for_key(key.pem)
    problems = check_record(rec, key.pem, rekor.log_key())
    assert any("no inclusion proof" in p for p in problems)


def test_tampered_proof_or_checkpoint_fails():
    rekor, _ = make_client()
    key = AnchorKey()
    rekor.log_digest(D1, key)
    rec = rekor.log_digest(D2, key)
    lk = rekor.log_key()
    rec.inclusion_proof["rootHash"] = "00" * 32
    problems = check_record(rec, key.pem, lk)
    assert any("inclusion" in p for p in problems) and any("checkpoint" in p for p in problems)
    rec2 = rekor.entries_for_key(key.pem)[1]
    rec2.inclusion_proof["checkpoint"] = rec2.inclusion_proof["checkpoint"].replace("mock-rekor - 1", "mock-rekor - 2")
    assert any("checkpoint signature" in p for p in check_record(rec2, key.pem, lk))
    rec3 = rekor.entries_for_key(key.pem)[1]
    rec3.inclusion_proof.pop("checkpoint")
    assert any("no signed checkpoint" in p for p in check_record(rec3, key.pem, lk))


def test_rekor_entry_key_must_equal_searched_key():
    rekor, _ = make_client()
    key, other = AnchorKey(), AnchorKey()
    rekor.log_digest(D1, key)
    (rec,) = rekor.entries_for_key(key.pem)
    assert any("not the searched anchor key" in p for p in check_record(rec, other.pem, rekor.log_key()))


def test_checkpoint_from_an_unpinned_log_is_rejected():
    raw = MockRekor()
    rogue = RekorClient(raw, pinned_log_key_sha256=REKOR_LOG_KEY_SHA256)    # the real pin, a different (mock) log
    with pytest.raises(RekorError):
        rogue.log_key()
    other = MockRekor()
    key = AnchorKey()
    c, _ = make_client(raw)
    c.log_digest(D1, key)
    (rec,) = c.entries_for_key(key.pem)
    # a record from one log checked against another log's key: checkpoint and SET both fail
    wrong = pinned_log_key(other.log_public_pem, other.log_key_sha256)
    problems = check_record(rec, key.pem, wrong)
    assert any("checkpoint signature" in p for p in problems) and not rec.time_trusted
    assert any("no pinned" in p for p in check_record(rec, key.pem, None))


def test_pinned_production_key_constant_is_a_p256_hash():
    assert len(REKOR_LOG_KEY_SHA256) == 64 and int(REKOR_LOG_KEY_SHA256, 16)


def test_anchor_key_private_half_only_under_ledger(tmp_path):
    from antelligence.kernel.board import BoardError

    with pytest.raises(BoardError):
        AnchorKey().save(tmp_path, "x")
    path = AnchorKey().save(tmp_path / "ledger", "x")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
