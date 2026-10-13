"""Key custody: 0600, ledger-only, study key once, deletion logged, no key material in output."""
import json
import stat

import pytest

from antelligence.anchoring import keys
from antelligence.anchoring.rekor import AnchorKey
from antelligence.experiments.v41_harness import KeyVault
from scripts import v41_keys


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_run_keys_are_0600_and_ledger_only(tmp_path):
    ledger = tmp_path / "ledger"
    keys_, akey = KeyVault(ledger).get("SW-B-case-1")
    assert mode(ledger / "SW-B-case-1.keys.json") == 0o600 and mode(ledger / "SW-B-case-1.anchor.pem") == 0o600
    assert keys.audit(ledger) == {"run_key_files": 2, "study_key_present": False, "bad_mode": []}
    with pytest.raises(Exception, match="ledger"):
        KeyVault(tmp_path / "elsewhere").get("x")
    with pytest.raises(keys.KeyError_, match="ledger"):
        keys.create_study_key(tmp_path / "elsewhere")


def test_study_key_is_p256_0600_once_and_pub_matches(tmp_path):
    ledger = tmp_path / "ledger"
    fp = keys.create_study_key(ledger)
    assert mode(ledger / keys.STUDY_KEY_NAME) == 0o600
    assert keys.load_study_key(ledger).fingerprint == fp == keys.study_fingerprint(ledger)
    assert isinstance(keys.load_study_key(ledger), AnchorKey)
    with pytest.raises(keys.KeyError_, match="already exists"):
        keys.create_study_key(ledger)


def test_study_key_refused_when_group_readable(tmp_path):
    ledger = tmp_path / "ledger"
    keys.create_study_key(ledger)
    (ledger / keys.STUDY_KEY_NAME).chmod(0o644)
    with pytest.raises(keys.KeyError_, match="readable"):
        keys.load_study_key(ledger)
    assert keys.audit(ledger)["bad_mode"] == [keys.STUDY_KEY_NAME]


def test_delete_needs_confirm_removes_only_run_keys_and_logs(tmp_path, capsys):
    ledger = tmp_path / "ledger"
    keys.create_study_key(ledger)
    vault = KeyVault(ledger)
    fps = {vault.get(f"L{i}")[1].fingerprint for i in range(3)}
    (ledger / "v4.1-usage.jsonl").write_text("{}\n")
    with pytest.raises(keys.KeyError_, match="confirm"):
        keys.delete_run_keys(ledger, confirm=False)
    assert len(keys.run_key_files(ledger)) == 6
    assert v41_keys.main(["delete-run-keys"], ledger_dir=ledger) == 2          # no --confirm: argparse flag is False
    assert len(keys.run_key_files(ledger)) == 6
    assert v41_keys.main(["delete-run-keys", "--confirm"], ledger_dir=ledger) == 0
    assert keys.run_key_files(ledger) == []
    assert (ledger / keys.STUDY_KEY_NAME).exists() and (ledger / "v4.1-usage.jsonl").exists()
    log = [json.loads(l) for l in (ledger / keys.DELETION_LOG).read_text().splitlines()]
    assert len(log) == 6 and {r["anchor_fingerprint"] for r in log if r["anchor_fingerprint"]} == fps
    out = capsys.readouterr()
    assert "deleted 6" in out.out


def test_cli_prints_fingerprints_never_key_material(tmp_path, capsys):
    ledger = tmp_path / "ledger"
    assert v41_keys.main(["init-study"], ledger_dir=ledger) == 0
    assert v41_keys.main(["fingerprint"], ledger_dir=ledger) == 0
    assert v41_keys.main(["audit"], ledger_dir=ledger) == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err
    pem = (ledger / keys.STUDY_KEY_NAME).read_text()
    body = "".join(pem.splitlines()[1:-1])
    assert "PRIVATE" not in text and body[:40] not in text
