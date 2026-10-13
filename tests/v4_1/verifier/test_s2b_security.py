"""S2b: security follow-ups on the outside verifier. Each test is a forgery attempt that must NOT verify.

Issue 1  unpinned trust anchors        -> pins (deployments file at a commit + preregistration)
Issue 2a authentication bypass          -> plan authenticates only through a pinned, on-chain preregistration
Issue 2b Rekor split view               -> checkpoint consistency proofs (anchor time vs audit time)
Issue 3  verdict downgrade              -> worst-of everywhere; failed dominates
Issue 4  Bitcoin header not validated   -> PoW / difficulty / hash checks in the header provider
"""

import base64
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import pytest
from bitcoin.core import CBlockHeader
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
from opentimestamps.core.op import OpSHA256
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp
from web3 import Web3

from antelligence.anchoring import mocks, ots
from antelligence.anchoring import pins as pinmod
from antelligence.anchoring.pins import PinError, TrustPins
from antelligence.anchoring.rekor import (
    AnchorKey,
    RekorClient,
    build_hashedrekord,
    RekorError,
    parse_checkpoint,
    split_view_check,
    verify_checkpoint,
)
from antelligence.anchoring.tamper import N_ENTRIES, Lab, _anchor_all, _export, _plan, rebuild
from antelligence.anchoring.verify import OutsideVerifier
from antelligence.kernel.board import anchor_sizes
from antelligence.kernel.canonical import jcs_bytes

REPO = Path(__file__).resolve().parents[3]
ALL = {"OTS", "R", "C"}


def make(h, conditions=ALL, directory=None, **kw):
    lab = h["lab"]
    args = dict(chain=lab.reader(), rekor=lab.rekor, ots_dir=directory or h["dir"], header_provider=lab.header_provider(),
                conditions=conditions, pins=lab.pins(), prereg=lab.prereg_path(), study_pem=lab.study_key.pem)
    args.update(kw)
    return OutsideVerifier(directory or h["dir"], h["plan"], **args)


# ---------------------------------------------------------------------------------------------
# Issue 1: pins
# ---------------------------------------------------------------------------------------------

def test_s2b_honest_run_verifies_with_pins(honest):
    report = make(honest).verify()
    assert report["verdict"] == "verified" and report["plan_authenticated"], json.dumps(report, indent=1)


def test_s2b_caller_supplied_registry_deployer_or_study_key_must_match_pins(honest):
    lab = honest["lab"]
    cases = {
        "registry": dict(pinned_registry="0x" + "ee" * 20),
        "deployer": dict(pinned_deployer="0x" + "11" * 20),
        "study key": dict(study_pem=AnchorKey().pem),            # the operator's own key, not the preregistered one
    }
    for name, kw in cases.items():
        report = make(honest, **kw).verify()
        assert report["verdict"] == "failed" and not report["ok"], name
        assert any("pinned" in p or "preregistration" in p for p in report["problems"]), (name, report["problems"])
    # asserting the SAME values as the pins is fine
    assert make(honest, pinned_registry=lab.registry_address, pinned_deployer=lab.deployer).verify()["verdict"] == "verified"


def test_s2b_forged_run_plan_under_attackers_study_key_is_not_authenticated(honest):
    """The operator makes its own study key, logs a forged plan under it and hands the verifier that key."""
    h = honest
    lab = h["lab"]
    rogue = AnchorKey()
    forged_plan = {**h["plan"], "study": "v4.1-forged"}
    plan_digest = hashlib.sha256(jcs_bytes(forged_plan)).digest()
    lab.rekor.log_digest(plan_digest, rogue)
    # (a) no pins at all -> the plan can never authenticate, whatever key is supplied
    unpinned = OutsideVerifier(h["dir"], forged_plan, rekor=lab.rekor, conditions={"R"}, study_pem=rogue.pem).verify()
    assert unpinned["verdict"] != "verified" and not unpinned["plan_authenticated"]
    # (b) with real pins the rogue key is rejected outright
    pinned = OutsideVerifier(h["dir"], forged_plan, rekor=lab.rekor, conditions={"R"}, pins=lab.pins(),
                             prereg=lab.prereg_path(), study_pem=rogue.pem).verify()
    assert pinned["verdict"] == "failed" and not pinned["plan_authenticated"]
    # (c) the right key, but the plan was never logged/anchored by the study
    forged_on_chain = OutsideVerifier(h["dir"], forged_plan, chain=lab.reader(), conditions={"C"}, pins=lab.pins(),
                                      prereg=lab.prereg_path()).verify()
    assert forged_on_chain["verdict"] == "failed" and not forged_on_chain["plan_authenticated"]


def test_s2b_second_run_plan_under_the_study_key_fails(honest):
    """The study key is operator-held: a second plan logged under it must be seen, not ignored."""
    h = honest
    lab = h["lab"]
    lab.rekor.log_digest(hashlib.sha256(jcs_bytes({**h["plan"], "study": "other"})).digest(), lab.study_key)
    report = make(h, conditions={"R"}, chain=None).verify()
    assert report["verdict"] == "failed" and not report["plan_authenticated"]
    assert any("other digest(s) logged under the study key" in p for p in report["problems"])


def test_s2b_plan_needs_the_prereg_and_the_chain_anchor(honest):
    h = honest
    no_prereg = make(h, prereg=None).verify()
    assert no_prereg["verdict"] == "unverified" and not no_prereg["plan_authenticated"] and not no_prereg["ok"]
    no_chain = make(h, conditions={"R"}, chain=None).verify()          # Rekor alone cannot root the study key
    assert no_chain["verdict"] == "unverified" and not no_chain["plan_authenticated"]
    swapped = h["tmp"] / "swapped.md"
    swapped.write_bytes(h["lab"].forged_prereg_doc())
    bad = make(h, prereg=swapped).verify()                              # same key line, different text: digest differs on chain
    assert bad["verdict"] == "failed" and any("preregistration document does not match" in p for p in bad["problems"])


def test_s2b_prereg_naming_another_study_key_fails(honest):
    h = honest
    other = h["tmp"] / "other.md"
    other.write_bytes(h["lab"].prereg_doc(b"x\n").replace(h["lab"].study_key.fingerprint.encode(), AnchorKey().fingerprint.encode()))
    report = make(h, prereg=other).verify()
    assert report["verdict"] == "failed" and any("different study key" in p for p in report["problems"])


def test_s2b_study_check_rejects_unpinned_inputs(honest):
    h = honest
    lab = h["lab"]
    plan_digest = hashlib.sha256(jcs_bytes(h["plan"])).digest()
    doc = h["tmp"] / "p.md"
    doc.write_bytes(lab.prereg_doc())
    lab.confirm_ots(h["tmp"] / "study")
    v = make(h)
    assert v.verify_study(doc, plan_digest, study_pem=lab.study_key.pem, prereg_ots=h["tmp"] / "study" / "prereg.ots")["ok"]
    assert make(h, conditions={"R"}).verify_study(doc, plan_digest, study_pem=AnchorKey().pem)["failed"]
    no_pins = OutsideVerifier(h["dir"], h["plan"], rekor=lab.rekor, conditions={"R"})
    assert not no_pins.verify_study(doc, plan_digest, study_pem=lab.study_key.pem)["ok"]
    wrong_registry = TrustPins(Web3.to_checksum_address("0x" + "ee" * 20), lab.deployer, lab.pins().code_hash, lab.pins().study_key_sha256)
    assert make(h, conditions={"C"}, pins=wrong_registry).verify_study(doc, plan_digest)["failed"]


DEPLOY = {"v4_1": {"complete": True, "deployer": "0x" + "d1" * 20, "contracts": [
    {"name": "AnchorRegistry", "address": "0x" + "a1" * 20, "code_hash": "0x" + "c0" * 32, "deployer": "0x" + "d1" * 20}]}}
PREREG = "text\n\nStudy key fingerprint (sha256 of SPKI DER): `" + "ab" * 32 + "`\n"


def test_s2b_load_pins_from_public_inputs_and_refuses_gaps():
    pins = pinmod.load_pins(DEPLOY, PREREG)
    assert pins.registry == Web3.to_checksum_address("0x" + "a1" * 20) and pins.code_hash == "0x" + "c0" * 32
    assert pins.study_key_sha256 == "ab" * 32
    broken = json.loads(json.dumps(DEPLOY))
    del broken["v4_1"]["contracts"][0]["code_hash"]
    for bad_deploy, bad_prereg in ((broken, PREREG), ({}, PREREG), (DEPLOY, "no key line here"), (DEPLOY, PREREG + PREREG)):
        with pytest.raises(PinError):
            pinmod.load_pins(bad_deploy, bad_prereg)
    incomplete = json.loads(json.dumps(DEPLOY))
    incomplete["v4_1"]["complete"] = False
    with pytest.raises(PinError):
        pinmod.load_pins(incomplete, PREREG)


def test_s2b_pins_come_from_a_commit_not_a_branch(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    target = tmp_path / pinmod.DEFAULT_DEPLOYMENTS
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(DEPLOY))
    git("add", "-A")
    git("commit", "-q", "-m", "deploy")
    commit = git("rev-parse", "HEAD")
    target.write_text(json.dumps({"v4_1": {**DEPLOY["v4_1"], "deployer": "0x" + "bb" * 20}}))      # later edit of the working tree
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(PREREG)
    assert pinmod.pins_from_repo(tmp_path, commit, prereg).deployer == Web3.to_checksum_address("0x" + "d1" * 20)
    with pytest.raises(PinError):
        pinmod.pins_from_repo(tmp_path, "HEAD", prereg)               # a moving ref is not a pin
    with pytest.raises(PinError):
        pinmod.pins_from_repo(tmp_path, "0" * 40, prereg)


def test_s2b_cli_refuses_condition_c_without_pins_and_bad_pins(honest, tmp_path):
    import sys

    h = honest
    cli = str(REPO / "scripts/v41_verify_board.py")
    base = [sys.executable, cli, "--export-dir", str(h["dir"]), "--plan", str(h["tmp"] / "plan.json")]
    no_pins = subprocess.run(base + ["--conditions", "C", "--rpc", "http://127.0.0.1:1", "--registry", "0x" + "a1" * 20],
                             capture_output=True, text=True)
    assert no_pins.returncode == 2 and "--deployments-commit" in no_pins.stderr
    bad = subprocess.run(base + ["--conditions", "B", "--prereg", str(h["lab"].prereg_path()), "--repo", str(tmp_path),
                                 "--deployments-commit", "HEAD"], capture_output=True, text=True)
    assert bad.returncode == 2 and "cannot build the pins" in bad.stderr


# ---------------------------------------------------------------------------------------------
# Issue 2b: Rekor split view
# ---------------------------------------------------------------------------------------------

def _rekor_only(h, rekor):
    return make(h, conditions={"R"}, chain=None, rekor=rekor, study_pem=None)


def test_s2b_forked_rekor_view_is_a_split_view(honest):
    """The log shows this verifier a different tree than the one the harness saw at anchor time."""
    h = honest
    lab = h["lab"]
    fork = lab.mock_rekor.fork(replace_index=1)
    forked_client = RekorClient(fork, pinned_log_key_sha256=lab.mock_rekor.log_key_sha256)
    # every inclusion proof and checkpoint in the fork is validly signed and internally consistent...
    for grow in (0, 3):
        for i in range(grow):
            fork.create(build_hashedrekord(bytes([i]) * 32, AnchorKey()))
        report = _rekor_only(h, forked_client).verify()
        chk = report["runs"][h["board"].run_id]["checks"]["rekor"]
        # ...but the archived anchor-time checkpoints do not belong to it
        assert report["verdict"] == "failed" and any("split view" in p for p in chk["problems"]), (grow, chk)


def test_s2b_honest_log_growth_stays_consistent(honest):
    h = honest
    for i in range(7):                               # other users keep writing to the public log
        h["lab"].mock_rekor.create(build_hashedrekord(bytes([i + 1]) * 32, AnchorKey()))
    report = make(h).verify()
    chk = report["runs"][h["board"].run_id]["checks"]["rekor"]
    assert chk["ok"] and report["verdict"] == "verified", chk


def test_s2b_missing_archived_checkpoints_cannot_verify(honest):
    h = honest
    (h["dir"] / f"{h['board'].run_id}.anchors.json").unlink()
    report = make(h).verify()
    chk = report["runs"][h["board"].run_id]["checks"]["rekor"]
    assert report["verdict"] == "unverified" and chk["unverified"] and "split view cannot be excluded" in chk["error"]


def test_s2b_forged_or_stripped_archived_checkpoint(honest):
    h = honest
    path = h["dir"] / f"{h['board'].run_id}.anchors.json"
    recs = json.loads(path.read_text())
    # a checkpoint whose root was changed after the log signed it
    cp = parse_checkpoint(recs[0]["rekor_checkpoint"])
    recs[0]["rekor_checkpoint"] = recs[0]["rekor_checkpoint"].replace(base64.b64encode(cp.root).decode(), base64.b64encode(b"x" * 32).decode())
    path.write_text(json.dumps(recs))
    report = make(h).verify()
    assert report["verdict"] == "failed"
    assert any("archived checkpoint" in p for p in report["runs"][h["board"].run_id]["checks"]["rekor"]["problems"])


def test_s2b_archived_checkpoint_must_cover_its_entry(honest):
    h = honest
    path = h["dir"] / f"{h['board'].run_id}.anchors.json"
    recs = json.loads(path.read_text())
    late = recs[-1]
    early_cp = recs[0]["rekor_checkpoint"]                # size covers only the first entry
    late["rekor_checkpoint"] = early_cp
    path.write_text(json.dumps(recs))
    report = make(h).verify()
    assert report["verdict"] == "failed"
    assert any("predates its own entry" in p for p in report["runs"][h["board"].run_id]["checks"]["rekor"]["problems"])


def test_s2b_split_view_check_unit():
    log = mocks.MockRekor()
    client = RekorClient(log, pinned_log_key_sha256=log.log_key_sha256)
    key = AnchorKey()
    for i in range(5):
        log.create(build_hashedrekord(bytes([i]) * 32, key))
    old = verify_checkpoint(log.log_info()["signedTreeHead"], client.log_key())
    for i in range(5, 9):
        log.create(build_hashedrekord(bytes([i]) * 32, key))
    new = verify_checkpoint(log.log_info()["signedTreeHead"], client.log_key())
    assert split_view_check(client, [old], [new]) == ([], [])
    fork = log.fork(2)
    other = RekorClient(fork, pinned_log_key_sha256=log.log_key_sha256)
    problems, _ = split_view_check(other, [old], [])
    assert problems and "split view" in problems[0]
    # a checkpoint for the same size with a different root is a fork on its face
    same_size_fork = verify_checkpoint(fork.log_info()["signedTreeHead"], client.log_key())
    problems, _ = split_view_check(client, [new], [same_size_fork])
    assert any("two different roots" in p for p in problems)
    # an unreachable log is "could not check", never a pass or a fabricated failure
    class Down(mocks.MockRekor):
        def log_info(self):
            raise ConnectionError("down")
    d = Down()
    d._log_key = log._log_key
    problems, unverified = split_view_check(RekorClient(d, pinned_log_key_sha256=log.log_key_sha256), [old], [])
    assert not problems and unverified


def test_s2b_http_transport_requests_the_documented_endpoints(monkeypatch):
    import httpx

    from antelligence.anchoring.rekor import HttpRekorTransport

    seen = []

    class Resp:
        status_code = 200
        text = ""
        def json(self):
            return {"hashes": ["aa" * 32], "signedTreeHead": "x"}

    monkeypatch.setattr(httpx, "get", lambda url, **kw: seen.append((url, kw.get("params"))) or Resp())
    t = HttpRekorTransport("https://rekor.example")
    assert t.consistency_proof(5, 9, "123") == ["aa" * 32]
    t.log_info()
    assert seen == [("https://rekor.example/api/v1/log/proof", {"firstSize": 5, "lastSize": 9, "treeID": "123"}),
                    ("https://rekor.example/api/v1/log", None)]


# ---------------------------------------------------------------------------------------------
# Issue 3: no verdict downgrade
# ---------------------------------------------------------------------------------------------

def _two_attestation_proof(digest, heights):
    ts = Timestamp(digest)
    for height in heights:
        ts.attestations.add(BitcoinBlockHeaderAttestation(height))
    return ots._serialise(DetachedTimestampFile(OpSHA256(), ts))


@pytest.mark.parametrize("flaky_height", [100, 101])
def test_s2b_a_failed_lookup_never_hides_an_invalid_attestation(flaky_height):
    """Whichever attestation is visited first, a proven-bad one makes the proof invalid, not unverified."""
    digest = bytes(range(32))
    proof = _two_attestation_proof(digest, [100, 101])

    def provider(height):
        if height == flaky_height:
            raise ConnectionError("esplora down")
        return CBlockHeader(hashMerkleRoot=b"\x07" * 32, nTime=1)       # wrong merkle root: the attestation fails

    res = ots.verify_ots(proof, digest, provider)
    assert not res.ok and res.status == "invalid", res
    only_flaky = ots.verify_ots(proof, digest, lambda h: (_ for _ in ()).throw(ConnectionError("down")))
    assert only_flaky.status == "unverified" and not only_flaky.ok


def test_s2b_unreadable_proof_is_invalid_whatever_the_parser_raises(monkeypatch):
    def boom(source):
        raise TypeError("parser blew up")

    monkeypatch.setattr(ots, "load_ots", boom)
    res = ots.verify_ots(b"anything", bytes(32))
    assert res.status == "invalid" and not res.ok and "TypeError" in res.error


@pytest.mark.parametrize("with_headers,conditions", list(itertools.product((True, False), (("OTS",), ("OTS", "R"), ("OTS", "R", "C")))))
def test_s2b_no_flag_combination_turns_a_failed_ots_into_unverified(honest, with_headers, conditions):
    """A proof for the wrong digest, or a missing proof, fails with or without a Bitcoin header source."""
    h = honest
    rid = h["board"].run_id
    sizes = anchor_sizes(N_ENTRIES)
    for name, damage in (("wrong digest", lambda: (h["dir"] / f"{rid}.{sizes[1]}.ots").write_bytes((h["dir"] / f"{rid}.{sizes[0]}.ots").read_bytes())),
                         ("missing proof", lambda: (h["dir"] / f"{rid}.{sizes[1]}.ots").unlink())):
        damage()
        report = make(h, conditions=set(conditions), header_provider=h["lab"].header_provider() if with_headers else None).verify()
        assert report["verdict"] == "failed" and not report["ok"], (name, with_headers, conditions, report["runs"][rid]["checks"]["ots"])
        if name == "wrong digest":                    # restore for the next case
            (h["dir"] / f"{rid}.{sizes[1]}.ots").write_bytes(b"")


def test_s2b_failed_study_ots_survives_a_missing_header_source(honest):
    h = honest
    lab = h["lab"]
    doc = h["tmp"] / "forged.md"
    doc.write_bytes(lab.forged_prereg_doc())
    plan_digest = hashlib.sha256(jcs_bytes(h["plan"])).digest()
    res = make(h, conditions={"OTS"}, header_provider=None, chain=None).verify_study(doc, plan_digest, prereg_ots=h["tmp"] / "study" / "prereg.ots")
    assert res["failed"] and not res["ok"]


def test_s2b_cli_verdict_cannot_be_improved_by_options(honest, tmp_path):
    """--no-check-bitcoin drops the header source; a tampered board still exits 1 (failed), never 3 (unverified)."""
    import sys

    h = honest
    forged = rebuild(h["board"], h["keys"], rewrite_at=5)
    bad = tmp_path / "bad"
    _export(forged, h["akey"], bad)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(forged, h["akey"])))
    for extra in ([], ["--no-check-bitcoin"]):
        r = subprocess.run([sys.executable, str(REPO / "scripts/v41_verify_board.py"), "--export-dir", str(bad), "--plan", str(plan),
                            "--conditions", "OTS", *extra], capture_output=True, text=True)
        assert r.returncode == 1 and json.loads(r.stdout)["verdict"] == "failed", (extra, r.stdout[-300:], r.stderr[-300:])


# ---------------------------------------------------------------------------------------------
# Issue 4: Bitcoin headers are validated, not trusted
# ---------------------------------------------------------------------------------------------

REGTEST_BITS = 0x207FFFFF


def _header(nonce, bits=REGTEST_BITS):
    return CBlockHeader(nVersion=1, hashPrevBlock=b"\x00" * 32, hashMerkleRoot=b"\x11" * 32, nTime=1_700_000_000, nBits=bits, nNonce=nonce)


def _mined(good: bool, bits=REGTEST_BITS):
    target = ots.nbits_target(bits)
    for nonce in range(10_000):
        h = _header(nonce, bits)
        if (int.from_bytes(h.GetHash(), "little") <= target) == good:
            return h
    raise AssertionError("no nonce found")


def _provider(header, *, named_hash=None, **kw):
    named = named_hash or header.GetHash()[::-1].hex()
    served = {"block-height/1": named, f"block/{named}/header": header.serialize().hex()}
    return ots.BlockstreamHeaderProvider(fetch=lambda path: served[path], **kw)


def test_s2b_header_provider_checks_pow_difficulty_and_hash():
    ok = _mined(True)
    lenient = dict(pow_limit=ots.nbits_target(REGTEST_BITS), min_difficulty=0)
    assert _provider(ok, **lenient)(1).hashMerkleRoot == b"\x11" * 32
    with pytest.raises(ots.HeaderError, match="proof of work"):
        _provider(_mined(False), **lenient)(1)                                   # hash above its own target
    with pytest.raises(ots.HeaderError, match="easier than the proof-of-work limit"):
        _provider(ok)(1)                                                         # regtest target is no Bitcoin target
    with pytest.raises(ots.HeaderError, match="difficulty"):
        _provider(ok, pow_limit=ots.nbits_target(REGTEST_BITS))(1)               # a real target, but far below the difficulty floor
    with pytest.raises(ots.HeaderError, match="named"):
        _provider(ok, named_hash="ab" * 32, **lenient)(1)                        # the server named another block
    with pytest.raises(ots.HeaderError):
        ots.nbits_target(0x04800000)                                             # negative compact target


def test_s2b_forged_bitcoin_attestation_needs_real_work(tmp_path):
    """A lying Esplora that returns a header with a merkle root it chose (no PoW) cannot confirm an anchor."""
    digest = bytes(range(32))
    ts = Timestamp(digest)
    ts.attestations.add(BitcoinBlockHeaderAttestation(1))
    proof = ots._serialise(DetachedTimestampFile(OpSHA256(), ts))
    liar = CBlockHeader(hashMerkleRoot=digest, nTime=1_700_000_000, nBits=REGTEST_BITS, nNonce=0)
    while int.from_bytes(liar.GetHash(), "little") <= ots.nbits_target(REGTEST_BITS):      # make sure it has no PoW even at regtest
        liar.nNonce += 1
    res = ots.verify_ots(proof, digest, _provider(liar, pow_limit=ots.nbits_target(REGTEST_BITS), min_difficulty=0))
    assert not res.ok and res.status == "unverified" and "proof of work" in res.error
    res = ots.verify_ots(proof, digest, _provider(liar))
    assert not res.ok
