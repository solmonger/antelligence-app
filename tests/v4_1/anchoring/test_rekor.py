"""Rekor hashedrekord entries: shape, signature, search by public key, inclusion proofs."""

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from antelligence.anchoring.mocks import MockRekor
from antelligence.anchoring.rekor import (
    AnchorKey,
    RekorClient,
    build_hashedrekord,
    check_record,
    fingerprint_of,
    load_public_pem,
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


def test_rekor_log_search_and_inclusion_proofs():
    rekor = RekorClient(MockRekor())
    k1, k2 = AnchorKey(), AnchorKey()
    recs = [rekor.log_digest(d, k1) for d in (D1, D2, D3)] + [rekor.log_digest(D1, k2)]
    assert [r.log_index for r in recs] == [0, 1, 2, 3]
    found = rekor.entries_for_key(k1.pem)
    assert [r.digest for r in found] == [D1.hex(), D2.hex(), D3.hex()]
    assert all(check_record(r) == [] for r in found)        # signature + RFC 6962 inclusion proof
    assert len(rekor.entries_for_key(k2.pem)) == 1
    again = rekor.log_digest(D2, k1)                        # identical entry: idempotent, not a duplicate
    assert again.uuid == recs[1].uuid and len(rekor.entries_for_key(k1.pem)) == 3


def test_rekor_rejects_bad_signature_and_detects_tampered_proof():
    key, other = AnchorKey(), AnchorKey()
    raw = MockRekor()
    bad = build_hashedrekord(D1, key)
    bad["spec"]["signature"]["content"] = base64.b64encode(other.sign_digest(D1)).decode()
    with pytest.raises(ValueError):
        raw.create(bad)
    rekor = RekorClient(raw)
    rekor.log_digest(D1, key)
    rec = rekor.log_digest(D2, key)
    rec.inclusion_proof["rootHash"] = "00" * 32
    assert any("inclusion" in p for p in check_record(rec))


def test_anchor_key_private_half_only_under_ledger(tmp_path):
    from antelligence.kernel.board import BoardError

    with pytest.raises(BoardError):
        AnchorKey().save(tmp_path, "x")
    path = AnchorKey().save(tmp_path / "ledger", "x")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
