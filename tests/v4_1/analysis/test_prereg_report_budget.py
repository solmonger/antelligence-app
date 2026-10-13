import json
import re

import pytest

from antelligence.experiments import v41_analysis as A
from antelligence.experiments import v41_budget as B

from .conftest import ROOT, load_script, synth


@pytest.fixture()
def inputs(tmp_path):
    for n in ("world.yaml", "harness.yaml", "conformal.json", "split.json"):
        (tmp_path / n).write_text(f"content of {n}")
    return dict(code_sha="a" * 40, world_yaml=tmp_path / "world.yaml", harness_yaml=tmp_path / "harness.yaml",
                conformal_json=tmp_path / "conformal.json", split_json=tmp_path / "split.json",
                router_model="qwen3-4b", model_revisions=["qwen3-4b@abc123", "flair@def456"],
                contracts={"AnchorRegistry": {"address": "0x" + "1" * 40, "code_hash": "0x" + "2" * 64}})


def test_prereg_refuses_missing_inputs(inputs, tmp_path):
    pre = load_script("v41_prereg")
    text = pre.render_prereg(**inputs)
    assert not re.search(r"\$[a-z_]+", text) and "a" * 40 in text and "AnchorRegistry" in text and "qwen3-4b@abc123" in text
    assert "SW-D: 12 x 1" in text and "sandbox -> 3" in text and "never cut" in text.lower()
    import hashlib
    assert hashlib.sha256(b"content of world.yaml").hexdigest() in text
    for key in ("code_sha", "router_model", "model_revisions", "contracts", "world_yaml", "split_json"):
        bad = {**inputs, key: None}
        with pytest.raises(pre.PreregError, match=key):
            pre.render_prereg(**bad)
    (tmp_path / "world.yaml").unlink()
    with pytest.raises(pre.PreregError, match="world_yaml"):
        pre.render_prereg(**inputs)
    with pytest.raises(pre.PreregError):
        pre.render_prereg(**{**inputs, "contracts": {"X": {"address": "0x1"}}})


def test_report_numbers_trace_to_results():
    rep = load_script("v41_report")
    results = json.loads(json.dumps(A.analyze(synth() + synth(split="dev", rng_seed=3), "eval")))
    md = rep.render(results)
    links = re.findall(r"\[([^\]]+)\]\(results\.json#([^)]+)\)", md)
    assert len(links) > 40
    for text, key in links:
        assert text == rep.fmt(rep.resolve(results, key)) or text == rep.fmt(rep.resolve(results, key), 2), key
    stripped = re.sub(r"\[[^\]]+\]\(results\.json#[^)]+\)", "", md)
    stripped = stripped.replace("v4.1", "")
    stripped = re.sub(r"[A-Za-z][\w-]*", "", stripped)  # labels such as P1, S12
    stripped = re.sub(r"v?\d+\.\d+(?=\))", "", stripped)
    assert not re.findall(r"\d+", stripped.replace("# ", "")), "bare numbers outside links"
    assert "n/a" in rep.render({**results, "power": {**results["power"], "mdd_p1": None}})


def test_cut_orders():
    cheap = {a: {"nous": 0.0, "runpod": 0.0} for a in B.PLAN}
    # within budget: nothing is cut
    assert B.project(cheap)["decisions"] == []
    # force both providers over: cuts follow the fixed order and stop as soon as the projection fits
    per_run = {a: {"nous": 0.02, "runpod": 0.03} for a in B.PLAN}
    out = B.project(per_run, sandbox_per_round={"nous": 0.5, "runpod": 0.0})
    nous = [d["cut"] for d in out["decisions"] if d["provider"] == "nous"]
    runpod = [d["cut"] for d in out["decisions"] if d["provider"] == "runpod"]
    def in_order(done, order):  # steps that are already at target (missingFLAIR 20 -> 20 on Nous) are skipped
        names = [f"{n} -> {v}" for n, v in order]
        idx = [names.index(d) for d in done]
        return idx == sorted(idx)
    assert in_order(nous, B.NOUS_CUT_ORDER) and nous[0] == "sandbox -> 3" and "SW-B-missingFLAIR -> 20" not in nous
    assert in_order(runpod, B.RUNPOD_CUT_ORDER) and runpod[0] == "SW-B-missingFLAIR -> 10"
    # order fidelity against the goal
    assert [n for n, _ in B.NOUS_CUT_ORDER] == ["sandbox", "FW", "SW-B-missingFLAIR", "SW-Bonsai", "SW-staticQueen"]
    assert [n for n, _ in B.RUNPOD_CUT_ORDER] == ["SW-B-missingFLAIR", "SW-staticQueen", "SW-noQueen", "SW-Byz-local",
                                                  "SW-Byz-chain", "SW-D"]
    assert dict(B.NOUS_CUT_ORDER)["FW"] == 30 and dict(B.RUNPOD_CUT_ORDER)["SW-D"] == 8
    # absurd costs: every cut applied, protected arms untouched, overage reported not hidden
    huge = {a: {"nous": 5.0, "runpod": 5.0} for a in B.PLAN}
    out = B.project(huge, sandbox_per_round={"nous": 1.0})
    p = out["plan"]
    assert p.cases["SW-B"] == 80 and p.seeds["SW-B"] == 2 and p.cases["SW-noBoard"] == 80 and p.cases["FW"] == 30
    assert p.sandbox_rounds == 3 and p.cases["SW-D"] == 8 and p.cases["SW-Byz-chain"] == 10
    assert out["over_after_cuts"] == ["nous", "runpod"]
    # floors are enforced
    with pytest.raises(B.FloorViolation):
        B.check_floors(B.Plan({**{a: c for a, (c, _) in B.PLAN.items()}, "FW": 29}, {a: s for a, (_, s) in B.PLAN.items()}))
    with pytest.raises(B.FloorViolation):
        B.check_floors(B.Plan({a: c for a, (c, _) in B.PLAN.items()}, {**{a: s for a, (_, s) in B.PLAN.items()}, "SW-B": 1}))
