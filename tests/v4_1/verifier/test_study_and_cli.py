"""Study-level anchors (preregistration, run plan) and the two CLIs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from antelligence.anchoring.anchorer import single_shot_id
from antelligence.anchoring.tamper import Lab
from antelligence.anchoring.verify import OutsideVerifier
from antelligence.kernel.canonical import jcs_bytes

REPO = Path(__file__).resolve().parents[3]


def test_preregistration_swap_is_caught_by_every_outside_log(tmp_path):
    lab = Lab()
    plan = {"study": "v4.1", "runs": []}
    plan_digest = hashlib.sha256(jcs_bytes(plan)).digest()
    s = lab.single_shot(tmp_path / "public")
    s.anchor("prereg", hashlib.sha256(lab.prereg_doc()).digest())
    s.anchor("runplan", plan_digest)
    good, forged = tmp_path / "prereg.md", tmp_path / "forged.md"
    good.write_bytes(lab.prereg_doc())
    forged.write_bytes(lab.forged_prereg_doc())

    lab.confirm_ots(tmp_path / "public")

    def study(doc, conds):
        v = OutsideVerifier(tmp_path, plan, chain=lab.reader(), rekor=lab.rekor, header_provider=lab.header_provider(),
                            conditions=conds, pins=lab.pins())
        return v.verify_study(doc, plan_digest, prereg_ots=tmp_path / "public" / "prereg.ots", study_pem=lab.study_key.pem)

    for cond in ({"OTS"}, {"R"}, {"C"}):
        assert study(good, cond)["ok"], cond
        assert study(forged, cond)["failed"], cond
    assert not study(forged, set())["ok"] and not study(forged, set())["failed"]   # B alone: nothing compared, never ok
    assert lab.reader().once_digest(single_shot_id("runplan")) == plan_digest


def test_cli_verify_and_tamper_smoke(honest):
    """The verifier CLI refuses to run condition C without pins from a deployments commit; the tamper CLI passes its own table."""
    h = honest
    py = sys.executable
    bad = subprocess.run([py, str(REPO / "scripts/v41_verify_board.py"), "--export-dir", str(h["dir"]), "--plan",
                          str(h["tmp"] / "plan.json"), "--conditions", "C"], capture_output=True, text=True)
    assert bad.returncode == 2 and "--deployments-commit" in bad.stderr
    ok = subprocess.run([py, str(REPO / "scripts/v41_verify_board.py"), "--export-dir", str(h["dir"]), "--plan",
                         str(h["tmp"] / "plan.json"), "--conditions", "B"], capture_output=True, text=True)
    assert ok.returncode == 3 and json.loads(ok.stdout)["verdict"] == "unverified"    # B alone is never "verified"
    out = h["tmp"] / "tamper.json"
    t = subprocess.run([py, str(REPO / "scripts/v41_tamper.py"), "--runs", "1", "--out", str(out)], capture_output=True, text=True)
    assert t.returncode == 0, t.stderr
    assert json.loads(out.read_text())["table_mismatches"] == []
