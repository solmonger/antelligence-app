"""S2c: final hardening. Each test is a forgery attempt that must NOT verify (or, for 'unverified', must not pass)."""

import json
import subprocess

import pytest
from web3 import Web3

from antelligence.anchoring import ots
from antelligence.anchoring import pins as pinmod
from antelligence.anchoring.pins import PinError
from antelligence.anchoring.verify import OutsideVerifier

from .test_s2b_security import DEPLOY, PREREG, make

KEY = "ab" * 32
LINE = f"Study key fingerprint (sha256 of SPKI DER): `{KEY}`\n"
OTHER = "Study key fingerprint (sha256 of SPKI DER): `" + "cd" * 32 + "`\n"


# ---------------------------------------------------------------------------------------------
# Issues 1-2: trust-anchor bypasses in pins.py
# ---------------------------------------------------------------------------------------------

class Repo:
    def __init__(self, path):
        self.path = path
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.org")
        self.git("config", "user.name", "t")

    def git(self, *a):
        return subprocess.run(["git", "-C", str(self.path), *a], capture_output=True, text=True, check=True).stdout.strip()

    def commit(self, files, msg="c"):
        for name, text in files.items():
            f = self.path / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")


def _deploy(deployer):
    d = json.loads(json.dumps(DEPLOY))
    d["v4_1"]["deployer"] = deployer
    d["v4_1"]["contracts"][0]["deployer"] = deployer
    return json.dumps(d)


def test_s2c_replace_ref_cannot_swap_the_pinned_deployments_blob(tmp_path):
    """Real: ``git show`` honours refs/replace/*; a caller-supplied repo could make the pinned commit serve other bytes."""
    r = Repo(tmp_path)
    commit = r.commit({pinmod.DEFAULT_DEPLOYMENTS: _deploy("0x" + "d1" * 20)})
    evil_blob = subprocess.run(["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"], input=_deploy("0x" + "bb" * 20),
                               capture_output=True, text=True, check=True).stdout.strip()
    good_blob = r.git("rev-parse", f"{commit}:{pinmod.DEFAULT_DEPLOYMENTS}")
    r.git("replace", good_blob, evil_blob)
    assert "bb" * 20 in r.git("show", f"{commit}:{pinmod.DEFAULT_DEPLOYMENTS}").lower()      # plain git is fooled
    prereg = tmp_path / "P.md"
    prereg.write_text(PREREG)
    assert pinmod.pins_from_repo(tmp_path, commit, prereg).deployer == Web3.to_checksum_address("0x" + "d1" * 20)


def test_s2c_pinned_commit_must_be_a_commit_and_an_ancestor_of_the_prereg_commit(tmp_path):
    r = Repo(tmp_path)
    rel = pinmod.DEFAULT_DEPLOYMENTS
    base = r.commit({rel: _deploy("0x" + "d1" * 20)})
    prereg_commit = r.commit({"PREREG.md": PREREG})
    r.git("checkout", "-q", "-b", "side", base)
    side = r.commit({rel: _deploy("0x" + "bb" * 20)}, "attacker deployment on a side branch")
    # a tree or blob id is not a commit
    for not_a_commit in (r.git("rev-parse", f"{base}^{{tree}}"), r.git("rev-parse", f"{base}:{rel}")):
        with pytest.raises(PinError, match="not a commit"):
            pinmod.deployments_at_commit(tmp_path, not_a_commit)
    ok = pinmod.pins_from_repo(tmp_path, base, "PREREG.md", prereg_commit=prereg_commit)
    assert ok.deployer == Web3.to_checksum_address("0x" + "d1" * 20)
    with pytest.raises(PinError, match="not an ancestor"):
        pinmod.pins_from_repo(tmp_path, side, "PREREG.md", prereg_commit=prereg_commit)
    with pytest.raises(PinError, match="40-hex"):
        pinmod.pins_from_repo(tmp_path, base, "PREREG.md", prereg_commit="main")


@pytest.mark.parametrize("doc,why", [
    ("<!--\n" + LINE + "-->\n", "only in an HTML comment"),
    ("```\n" + LINE + "```\n", "only in a code fence"),
    ("~~~text\n" + LINE + "~~~\n", "only in a tilde fence"),
    ("```\n" + LINE, "unterminated fence swallows the rest"),
    ("<!-- " + LINE, "unterminated comment"),
    (OTHER + "```\n" + LINE + "```\n", "real line plus a hidden one with another key"),
    (LINE + "<!--\n" + LINE + "-->\n", "the same key twice, once hidden"),
    ("```\n" + OTHER + "```\n" + LINE + "<!-- " + OTHER + " -->\n", "hidden decoys around the real line"),
])
def test_s2c_study_key_line_in_a_forgeable_spot_is_rejected(doc, why):
    with pytest.raises(PinError):
        pinmod.study_key_in(doc)


def test_s2c_study_key_visible_line_is_accepted_beside_unrelated_blocks():
    assert pinmod.study_key_in("```\ncode\n```\n" + LINE + "tail\n") == KEY
    with pytest.raises(PinError, match="raw HTML"):          # S2d: HTML comments are raw HTML, now refused outright
        pinmod.study_key_in("<!-- note -->\n" + LINE + "tail\n")
    assert pinmod.study_key_in((LINE.replace("\n", "\r\n")).encode()) == KEY
    with pytest.raises(PinError):
        pinmod.study_key_in(LINE.replace(KEY, KEY.upper()))              # no case-insensitive match


def test_s2c_pins_have_no_defaults_and_compare_exactly():
    mixed = json.loads(json.dumps(DEPLOY))
    mixed["v4_1"]["contracts"][0]["code_hash"] = "0x" + "C0" * 32
    no_entry_deployer = json.loads(json.dumps(DEPLOY))
    del no_entry_deployer["v4_1"]["contracts"][0]["deployer"]
    no_deployer = json.loads(json.dumps(DEPLOY))
    del no_deployer["v4_1"]["deployer"]
    short = json.loads(json.dumps(DEPLOY))
    short["v4_1"]["contracts"][0]["code_hash"] = "0x" + "c0" * 31                  # a prefix is not a hash
    for bad in (mixed, no_entry_deployer, no_deployer, short, {"v4_1": []}, {"v4_1": {"complete": True, "contracts": "x"}}, []):
        with pytest.raises(PinError):
            pinmod.load_pins(bad, PREREG)


# ---------------------------------------------------------------------------------------------
# Issues 3-4: fail-open / incomplete validation in verify.py
# ---------------------------------------------------------------------------------------------

def test_s2c_duplicate_and_malformed_plan_entries_fail(honest):
    h = honest
    first = h["plan"]["runs"][0]
    dup = {**h["plan"], "runs": [first, {**first, "anchor_pubkey": "00" * 32}]}      # last-wins would hide the first
    r = OutsideVerifier(h["dir"], dup).verify()
    assert r["verdict"] == "failed" and any("more than once" in p for p in r["problems"])
    for broken in ({k: v for k, v in first.items() if k != "anchor_pubkey"}, {k: v for k, v in first.items() if k != "pubkeys"},
                   {k: v for k, v in first.items() if k != "run_id"}):
        r = OutsideVerifier(h["dir"], {**h["plan"], "runs": [broken]}).verify()
        assert r["verdict"] == "failed" and not r["ok"], broken.keys()
    assert OutsideVerifier(h["dir"], {"study": "x"}).verify()["verdict"] == "failed"


def test_s2c_claim_about_an_unknown_run_fails_instead_of_being_skipped(honest):
    h = honest
    claim = {"id": "c1", "run_id": "ff" * 32, "covers_size": 1, "made_at": 1}
    r = make(h, claims=[claim]).verify()
    assert r["verdict"] == "failed" and any("claim c1 names no verified run" in p for p in r["problems"])
    r = make(h, claims=[{"id": "c2", "covers_size": 1, "made_at": 1}]).verify()      # no run_id is not "every run"
    assert r["verdict"] == "failed"


def test_s2c_plan_confirmed_by_chain_alone_is_not_authenticated_when_rekor_cannot_be_read(honest, monkeypatch):
    h = honest
    lab = h["lab"]
    orig = lab.rekor.entries_for_key

    def flaky(pem):
        if pem == lab.study_key.pem:
            raise ConnectionError("rekor down")
        return orig(pem)

    monkeypatch.setattr(lab.rekor, "entries_for_key", flaky)
    r = make(h).verify()
    assert r["verdict"] == "unverified" and not r["plan_authenticated"] and not r["ok"]


def _anchors(h):
    path = h["dir"] / f"{h['board'].run_id}.anchors.json"
    return path, json.loads(path.read_text())


def _rekor_check(h, report):
    return report["runs"][h["board"].run_id]["checks"]["rekor"]


def test_s2c_record_with_rekor_uuid_but_no_checkpoint_is_unverified(honest):
    h = honest
    path, recs = _anchors(h)
    del recs[1]["rekor_checkpoint"]
    path.write_text(json.dumps(recs))
    r = make(h).verify()
    chk = _rekor_check(h, r)
    assert r["verdict"] == "unverified" and not r["ok"] and chk["unverified"] and "no rekor_checkpoint" in chk["error"]
    recs[1]["rekor_checkpoint"] = ""
    path.write_text(json.dumps(recs))
    assert make(h).verify()["verdict"] == "unverified"


def test_s2c_record_whose_uuid_contradicts_its_digest_fails(honest):
    h = honest
    path, recs = _anchors(h)
    recs[0]["rekor_uuid"], recs[1]["rekor_uuid"] = recs[1]["rekor_uuid"], recs[0]["rekor_uuid"]
    path.write_text(json.dumps(recs))
    r = make(h).verify()
    assert r["verdict"] == "failed" and any("logs a different digest" in p for p in _rekor_check(h, r)["problems"])
    # stripping the checkpoint cannot soften a contradiction into 'unverified'
    for rec in recs:
        rec.pop("rekor_checkpoint")
    path.write_text(json.dumps(recs))
    assert make(h).verify()["verdict"] == "failed"


def test_s2c_anchors_json_is_authenticated_against_the_recomputed_roots(honest):
    h = honest
    path, recs = _anchors(h)
    original = json.dumps(recs)
    forgeries = {
        "root": lambda r: r[0].update(root="00" * 32),
        "digest": lambda r: r[0].update(digest="11" * 32),
        "size past the export": lambda r: r[0].update(size=10_000),
        "size is a string": lambda r: r[0].update(size="50"),
        "unknown uuid": lambda r: r[0].update(rekor_uuid="ee" * 32),
        "repeated uuid": lambda r: r[1].update(rekor_uuid=r[0]["rekor_uuid"]),
        "record not an object": lambda r: r.append("x"),
    }
    for name, forge in forgeries.items():
        recs = json.loads(original)
        forge(recs)
        path.write_text(json.dumps(recs))
        r = make(h).verify()
        assert r["verdict"] == "failed" and not r["ok"], (name, _rekor_check(h, r))
    path.write_text(json.dumps({"not": "a list"}))
    assert make(h).verify()["verdict"] == "failed"


def test_s2c_an_old_unrelated_checkpoint_cannot_stand_in_for_a_missing_anchor_record(honest):
    """Drop every record but keep one tiny valid checkpoint: it is not tied to the entries, so the run cannot verify."""
    h = honest
    path, recs = _anchors(h)
    path.write_text(json.dumps([{k: v for k, v in recs[0].items()}, ]))
    r = make(h).verify()
    assert r["verdict"] == "unverified" and not r["ok"]
    assert "no authenticated anchors.json record" in _rekor_check(h, r)["error"]


# ---------------------------------------------------------------------------------------------
# Reviewer follow-ups
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nbits", [0x00000000, 0x04800000, 0xFF7FFFFF, 0x21010000, 0x2300FFFF, 1 << 32, -1])
def test_s2c_nbits_target_rejects_zero_negative_and_overflow(nbits):
    with pytest.raises(ots.HeaderError):
        ots.nbits_target(nbits)


def test_s2c_nbits_target_accepts_real_targets():
    assert ots.nbits_target(0x1D00FFFF) == 0xFFFF << (8 * (0x1D - 3))
    assert ots.nbits_target(0x207FFFFF) < 1 << 256
