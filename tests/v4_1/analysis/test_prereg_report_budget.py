import hashlib
import json
import re

import pytest

from antelligence.experiments import v41_analysis as A
from antelligence.experiments import v41_budget as B

from .conftest import load_script, synth


@pytest.fixture()
def inputs(tmp_path):
    for n in ("world.yaml", "harness.yaml", "conformal.json", "split.json", "router.pkl"):
        (tmp_path / n).write_text(f"content of {n}")
    (tmp_path / "revs.json").write_text(json.dumps({"models": [
        {"name": "qwen3-4b", "revision": "abc123"}, {"name": "flair", "revision": "def456"}]}))
    return dict(code_sha="a" * 40, world_yaml=tmp_path / "world.yaml", harness_yaml=tmp_path / "harness.yaml",
                conformal_json=tmp_path / "conformal.json", split_json=tmp_path / "split.json",
                router_model_file=tmp_path / "router.pkl", model_revisions_json=tmp_path / "revs.json",
                contracts={"AnchorRegistry": {"address": "0x" + "1" * 40, "code_hash": "0x" + "2" * 64}},
                study_key_sha256="c" * 64)


def test_prereg_refuses_missing_inputs(inputs, tmp_path):
    pre = load_script("v41_prereg")
    text = pre.render_prereg(**inputs)
    assert not re.search(r"\$[a-z_]+", text) and "a" * 40 in text and "AnchorRegistry" in text and "qwen3-4b@abc123" in text
    assert "SW-D** | as SW-B | promoted harness | D (board on-chain) | 12 x 1" in text
    assert "sandbox rounds down to 3" in text and "Never cut SW-B or SW-noBoard below 80 x 2" in text
    assert "lower end of a two-sided 90% BCa" in text and "(seed 1)" in text and "$8, with a hard stop at $7.50" in text
    assert "Changes after the lock" in text and "None, except cuts made for a budget hard stop" in text
    assert "outside its band (5% to 40% per run) in more than 25% of runs" in text
    for arm, (cases, seeds) in B.PLAN.items():  # template sizes agree with the budget module
        name = arm.replace("-prime", "'")
        row = next(ln for ln in text.splitlines() if ln.startswith((f"| {name} |", f"| **{name}** |")))
        assert row.rstrip(" |").endswith(f"{cases} x {seeds}"), arm
    assert hashlib.sha256(b"content of world.yaml").hexdigest() in text
    assert hashlib.sha256(b"content of router.pkl").hexdigest() in text
    assert hashlib.sha256((tmp_path / "revs.json").read_bytes()).hexdigest() in text
    from antelligence.anchoring.pins import study_key_in

    assert study_key_in(text) == "c" * 64                  # the verifier reads the pin back out of the rendered document
    for key in ("code_sha", "router_model_file", "model_revisions_json", "contracts", "world_yaml", "split_json",
                "study_key_sha256"):
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


def test_committed_model_revisions_file():
    from .conftest import ROOT
    models = json.loads((ROOT / "config" / "v4_1" / "model_revisions.json").read_text())["models"]
    revs = {m["revision"] for m in models}
    assert {"860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae",
            "370f7f9d062745fbac445e7fe6d6616d35df04ec", "051bcc464b01b9f92942b364d9586b0ef5912432",
            "790f0576d2d77dd5322aa0603a470bd9e3a3d1f6", "397b7ba47a99b3221ebfc0cfc5a279118cb733ad",
            "c2aefbeb4b24469cd11579c3384b990404c17a30", "ac20f03fc62e872399218b659c8e949dfca05769"} <= revs
    assert not any("TODO" in m["revision"] for m in models)


def test_cut_orders():
    names = lambda order: [B.step_name(s) for s in order]  # noqa: E731
    assert names(B.NOUS_CUT_ORDER) == ["sandbox -> 3", "FW -> 30",
                                       "SW-B-missingFLAIR -> 20; SW-Bonsai -> 20; SW-staticQueen -> 20"]
    assert names(B.RUNPOD_CUT_ORDER) == ["SW-B-missingFLAIR -> 0", "SW-staticQueen -> 0", "SW-noQueen -> 0",
                                         "SW-Byz-local -> 10; SW-Byz-chain -> 10", "SW-D -> 8"]
    cheap = {a: {"nous": 0.0, "runpod": 0.0} for a in B.PLAN}
    assert B.project(cheap)["decisions"] == []
    # Nous only, just over the stop: step 1 alone is applied, nothing later
    per_run = {a: {"nous": 0.0, "runpod": 0.0} for a in B.PLAN}
    out = B.project(per_run, sandbox_per_round={"nous": 2.0})  # 4 rounds = 8 > 7.5
    assert [d["cut"] for d in out["decisions"]] == ["sandbox -> 3"] and out["plan"].sandbox_rounds == 3
    # RunPod: steps are atomic and drop whole arms
    per_run["SW-B-missingFLAIR"]["runpod"] = 0.3  # 20 runs = 6.0
    per_run["SW-staticQueen"]["runpod"] = 0.1  # 40 runs = 4.0 -> total 10 > 9.5
    out = B.project(per_run)
    assert [d["cut"] for d in out["decisions"]] == ["SW-B-missingFLAIR -> 0"] and out["plan"].cases["SW-B-missingFLAIR"] == 0
    # absurd costs: every cut applied, protected arms untouched, overage reported not hidden
    huge = {a: {"nous": 5.0, "runpod": 5.0} for a in B.PLAN}
    out = B.project(huge, sandbox_per_round={"nous": 1.0})
    p = out["plan"]
    assert p.cases["SW-B"] == 80 and p.seeds["SW-B"] == 2 and p.cases["SW-noBoard"] == 80 and p.cases["FW"] == 30
    assert p.sandbox_rounds == 3 and p.cases["SW-D"] == 8 and p.cases["SW-Byz-chain"] == 10 and p.cases["SW-Byz-local"] == 10
    assert p.cases["SW-noQueen"] == p.cases["SW-staticQueen"] == p.cases["SW-B-missingFLAIR"] == 0
    assert p.cases["SW-Bonsai"] == 20
    assert out["over_after_cuts"] == ["nous", "runpod"]
    with pytest.raises(B.FloorViolation):
        B.check_floors(B.Plan({**{a: c for a, (c, _) in B.PLAN.items()}, "FW": 29}, {a: s for a, (_, s) in B.PLAN.items()}))
    with pytest.raises(B.FloorViolation):
        B.check_floors(B.Plan({a: c for a, (c, _) in B.PLAN.items()}, {**{a: s for a, (_, s) in B.PLAN.items()}, "SW-B": 1}))
