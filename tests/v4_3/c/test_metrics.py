"""C: the v4.3 measurement additions on fabricated records and bundles (no run, no case)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from antelligence.experiments import v43_metrics as M
from antelligence.experiments import v43_report as RP
from antelligence.experiments import v43_results as RES
from tests.v4_3.c.helpers import REPO

TOTAL = 800.0
SHARE = TOTAL / 8                      # 100 per squad


def mark(kind: str, pos, fraction: float = 0.0, tick: int = 0, ant: int = 0):
    return {"ant": ant, "mark_kind": kind, "pos_vox": list(pos), "tick": tick, "amount_milli": int(round(fraction * 1000)), "reading_ppm": 0}


# a hand-made field: tumour block x 0..9, tissue u = 0.2 for x < 3 (poor), 0.5 for x 3..5, 0.9 for x 6..9; outside x >= 10 u = 1
def dist_mm(pos):
    x = pos[0]
    return 0.0 if x < 10 else 2.0 * (x - 9)           # 2 mm voxels: x = 12 is 6 mm from the tumour, x = 11 is 4 mm


def u_at(pos):
    x = pos[0]
    return 0.2 if x < 3 else 0.5 if x < 6 else 0.9 if x < 10 else 1.0


def bundle():
    return {"marks": [mark("dosed", (1, 0, 0), 0.50), mark("dosed", (2, 5, 5), 0.25),          # poor tissue: 50 + 25 = 75
                      mark("dosed", (7, 0, 0), 0.10),                                          # good tissue: 10
                      mark("dosed", (4, 0, 0), 0.20),                                          # middling tissue: 20
                      mark("dosed", (11, 0, 0), 0.30),                                         # 4 mm out: not a false alarm (needs more than 4): 30
                      mark("dosed", (12, 0, 0), 0.40), mark("dosed", (14, 0, 0), 0.10),       # 6 mm and 10 mm out: false alarms, 40 + 10 = 50
                      mark("dose_ok", (1, 0, 0), 1.0), mark("good_uptake", (1, 0, 0), 1.0)]}   # not releases


def test_release_pools_split_the_dosed_marks_into_disjoint_classes():
    p = M.release_pools(bundle(), TOTAL, dist_mm_at=dist_mm, u_at=u_at)
    assert p["poor_patch"] == pytest.approx(75.0) and p["false_alarm"] == pytest.approx(50.0)
    assert p["other"] == pytest.approx(10 + 20 + 30)
    assert p["released_from_marks"] == pytest.approx(p["poor_patch"] + p["false_alarm"] + p["other"]) == pytest.approx(185.0)
    assert (p["n_releases"], p["n_false_alarm"], p["n_poor_patch"]) == (7, 2, 2)


def test_release_pools_use_the_squad_share_total_over_n_squads():
    p = M.release_pools({"marks": [mark("dosed", (1, 0, 0), 1.0)]}, 160.0, dist_mm_at=dist_mm, u_at=u_at, n_squads=4)
    assert p["poor_patch"] == pytest.approx(40.0)


def test_gap_by_cause_pools_and_remainder_sum_to_the_gap():
    p = M.release_pools(bundle(), TOTAL, dist_mm_at=dist_mm, u_at=u_at)
    prime_cov, prime_rel = 0.60, 640.0                      # S' released 80% of the budget: 0.75 coverage per unit of budget released
    released = p["released_from_marks"]
    g = M.gap_by_cause(cov_arm=0.35, cov_prime=prime_cov, prime_released=prime_rel, total_drug=TOTAL, false_alarm=p["false_alarm"],
                       poor_patch=p["poor_patch"], unreleased=TOTAL - released)
    c = 0.60 / 0.80
    assert g["per_released_drug"] == pytest.approx(c)
    assert g["gap"] == pytest.approx(0.25)
    assert g["false_alarm"] == pytest.approx(50.0 / TOTAL * c)
    assert g["poor_patch"] == pytest.approx(75.0 / TOTAL * c)
    assert g["unreleased"] == pytest.approx((TOTAL - released) / TOTAL * c)
    assert g["false_alarm"] + g["poor_patch"] + g["unreleased"] + g["remainder"] == pytest.approx(g["gap"])


def test_gap_by_cause_for_s_enters_through_its_totals_only():
    g = M.gap_by_cause(cov_arm=0.2, cov_prime=0.6, prime_released=600.0, total_drug=TOTAL, false_alarm=None, poor_patch=None, unreleased=2.0)
    assert g["false_alarm"] is None and g["poor_patch"] is None
    assert g["unreleased"] == pytest.approx(2.0 / TOTAL * 0.6 / 0.75)
    assert g["unreleased"] + g["remainder"] == pytest.approx(g["gap"])


def test_gap_by_cause_negative_gap_and_a_zero_release_prime():
    g = M.gap_by_cause(cov_arm=0.7, cov_prime=0.6, prime_released=600.0, total_drug=TOTAL, false_alarm=0.0, poor_patch=0.0, unreleased=0.0)
    assert g["gap"] == pytest.approx(-0.1) and g["remainder"] == pytest.approx(-0.1)
    z = M.gap_by_cause(cov_arm=0.1, cov_prime=0.0, prime_released=0.0, total_drug=TOTAL, false_alarm=1.0, poor_patch=1.0, unreleased=1.0)
    assert z["per_released_drug"] is None and z["remainder"] == pytest.approx(z["gap"])


def test_summarise_gaps_averages_and_states_the_assumption():
    rows = [M.gap_by_cause(cov_arm=0.3, cov_prime=0.6, prime_released=600.0, total_drug=TOTAL, false_alarm=f, poor_patch=q, unreleased=u)
            for f, q, u in ((40.0, 20.0, 10.0), (0.0, 0.0, 0.0), (None, None, 5.0))]
    s = M.summarise_gaps(rows)
    assert s["n_runs"] == 3 and s["n_false_alarm"] == 2 and s["n_unreleased"] == 3 and s["gap"] == pytest.approx(0.3)
    assert "estimate and an association" in s["assumption"] and "tumour-side yield" in s["assumption"] and "S''s" in s["assumption"]


def test_probe_counts_read_the_script_counters_and_treat_missing_keys_as_zero():
    assert M.probe_counts({"probes": 7, "probes_saved_by_marks": 3, "asks": 9}) == {"probes": 7, "probes_saved_by_marks": 3}
    assert M.probe_counts({"asks": 9}) == {"probes": 0, "probes_saved_by_marks": 0}
    assert M.probe_counts(None) == {"probes": 0, "probes_saved_by_marks": 0}
    assert M.probe_counts({"probes": None}) == {"probes": 0, "probes_saved_by_marks": 0}


def test_run_metrics_v43_adds_probes_and_the_release_totals():
    metrics = {"coverage": 0.4, "off_target_share": 0.5, "released_total": 600.0, "time_to_50_coverage": None, "coverage_at_queen": {}}
    r = M.run_metrics_v43(metrics, None, script_counters={"probes": 4, "probes_saved_by_marks": 1}, total_drug=TOTAL)
    assert r["coverage"] == 0.4 and r["probes"] == 4 and r["probes_saved_by_marks"] == 1
    assert r["released_total"] == 600.0 and r["total_drug"] == TOTAL and r["unreleased"] == pytest.approx(200.0)
    assert M.run_metrics_v43(metrics)["probes"] == 0 and M.run_metrics_v43(metrics)["unreleased"] is None


# --- message effect, uptake marks apart ------------------------------------------------------------------------------------------------

def test_effect_rows_are_tagged_with_the_mark_kind_and_summarised_by_group():
    marks = {("c", 5): [mark("good_uptake", (0, 0, 0)), mark("dosed", (0, 0, 0)), mark("slow_uptake", (0, 0, 0)), mark("tumour_here", (0, 0, 0))]}
    rows = [{"case_id": "c", "seed": 5, "item": {"kind": "mark", "id": i}, "changed": ch, "no_ant_changed": not ch, "coverage_delta": d}
            for i, ch, d in ((0, True, 0.01), (1, False, 0.0), (2, False, 0.0), (3, True, -0.02))]
    rows.append({"case_id": "c", "seed": 5, "item": {"kind": "message", "id": 4}, "changed": True, "no_ant_changed": False, "coverage_delta": 0.0})
    tagged = M.tag_mark_kinds(rows, marks)
    assert [r.get("mark_kind") for r in tagged] == ["good_uptake", "dosed", "slow_uptake", "tumour_here", None]
    g = M.effect_by_group(tagged)
    assert g["all"]["n_items"] == 5 and g["all"]["effect_rate"] == pytest.approx(3 / 5)
    assert g["uptake_marks"]["n_items"] == 2 and g["uptake_marks"]["effect_rate"] == pytest.approx(0.5)
    assert g["other_marks"]["n_items"] == 2 and g["other_marks"]["effect_rate"] == pytest.approx(0.5)
    assert g["messages"]["n_items"] == 1 and g["messages"]["effect_rate"] == 1.0
    assert M.effect_by_group([])["uptake_marks"]["effect_rate"] is None


# --- results assembly on a fabricated run directory ---------------------------------------------------------------------------------------

def write_episode(root: Path, arm: str, case: str, seed: int, *, coverage: float, counters=None, complete=True, split="dev", total=TOTAL, released=600.0):
    d = root / arm / case / f"seed{seed}"
    d.mkdir(parents=True)
    rec = {"arm": arm, "case_id": case, "seed": seed, "split": split, "complete": complete, "bundle": "bundle.json", "total_drug": total,
           "metrics": {"coverage": coverage, "off_target_share": 0.4, "released_total": released, "time_to_50_coverage": 30,
                       "coverage_at_queen": {"0": 0.0}}, "timing_s": {"wall": 1.0}, "n_regions": 3}
    if counters is not None:
        rec["script_counters"] = counters
    (d / "record.json").write_text(json.dumps(rec))
    colony = None if arm in ("S", "S'") else {"params": {"half_life": None, "read_radius_mm": 6.0, "confirm_radius_mm": 4.0}, "marks": [], "messages": [],
                                               "counters": {}, "cards_after": {}, "reading_logs": {}}
    (d / "bundle.json").write_text(json.dumps({"extra": {"colony": colony}}))


def doc():
    from antelligence.experiments.v41_gate import load_checked_split
    return load_checked_split(REPO)


def test_assemble_carries_probes_and_totals_and_refuses_non_dev_cases(tmp_path):
    d = doc()
    case = d["dev_a"][0]
    write_episode(tmp_path, "S-marks", case, 5, coverage=0.42, counters={"probes": 6, "probes_saved_by_marks": 2})
    write_episode(tmp_path, "S-solo", case, 5, coverage=0.35, counters={"asks": 1})
    write_episode(tmp_path, "S", case, 5, coverage=0.2)
    write_episode(tmp_path, "S'", case, 5, coverage=0.6, released=700.0)
    write_episode(tmp_path, "S-talk", case, 5, coverage=0.5, complete=False)
    res = RES.assemble(tmp_path, meta={"level": "mid"})
    rows = {r["arm"]: r for r in res["runs"]}
    assert set(rows) == {"S-marks", "S-solo", "S", "S'"} and res["meta"]["incomplete"] == [f"S-talk/{case}/seed5"] and res["meta"]["level"] == "mid"
    assert rows["S-marks"]["probes"] == 6 and rows["S-marks"]["probes_saved_by_marks"] == 2 and rows["S-solo"]["probes"] == 0
    assert rows["S-marks"]["split"] == "dev-A" and rows["S'"]["released_total"] == 700.0 and rows["S"]["unreleased"] == pytest.approx(200.0)
    assert rows["S-marks"]["colony"]["marks_total"] == 0
    # a case that is not a development case is refused before any file of it is parsed
    write_episode(tmp_path, "S-solo", d["eval"][0], 5, coverage=0.3)
    from antelligence.experiments.v42_harness import EvalRefused
    with pytest.raises(EvalRefused):
        RES.assemble(tmp_path)


def test_assemble_refuses_a_synthetic_record(tmp_path):
    from antelligence.experiments.v42_harness import EvalRefused
    write_episode(tmp_path, "S-solo", doc()["dev_a"][0], 5, coverage=0.3, split="synthetic")
    with pytest.raises(EvalRefused):
        RES.assemble(tmp_path)


# --- report sections ------------------------------------------------------------------------------------------------------------------------

def fab_results():
    runs = []
    for arm, cov, pr, sv in (("S-solo", 0.3, 0, 0), ("S-marks", 0.34, 5, 2), ("S-talk", 0.35, 6, 3)):
        for c, sp in (("c1", "dev-A"), ("c2", "dev-B")):
            runs.append({"arm": arm, "case_id": c, "seed": 5, "split": sp, "coverage": cov, "off_target_share": 0.4, "time_to_50_coverage": 30,
                         "drug_on_non_tumour": 1.0, "probes": pr, "probes_saved_by_marks": sv,
                         "colony": {"marks_total": 10, "share_strong": 0.1, "share_disputed": 0.0, "asks_sent": 0, "asks_answered": 0, "asks_lapsed": 0,
                                    "asks_refused": 0, "quorum_events": 0, "turns_skipped": 0, "marks_by_kind": {"dosed": 5},
                                    "honesty": {"correlation": None, "n_ants": 0}}})
    return {"meta": {"source_dir": "x/y", "level": "mid"}, "runs": runs}


def test_report_has_probe_poor_tissue_gap_effect_and_realised_u_sections_each_with_a_source():
    diag = {"poor_tissue_release": {"S-marks": {"n_runs": 2, "total_drug": 800.0, "released_from_marks": 600.0, "poor_patch": 60.0,
                                                  "poor_share_of_released": 0.1, "false_alarm": 90.0}},
            "gap_to_s_prime_by_cause": {"S-marks": {"n_runs": 2, "gap": 0.2, "false_alarm": 0.05, "poor_patch": 0.02, "unreleased": 0.03, "remainder": 0.1,
                                                      "assumption": M.ASSUMPTION},
                                         "S": {"n_runs": 2, "gap": 0.3, "false_alarm": None, "poor_patch": None, "unreleased": 0.01, "remainder": 0.29,
                                               "assumption": M.ASSUMPTION}}}
    eff = {"S-marks": {"by_group": {"all": {"n_items": 10, "effect_rate": 0.4, "mean_delta": 0.001, "spam_rate": 0.6},
                                     "uptake_marks": {"n_items": 3, "effect_rate": 0.67, "mean_delta": 0.0, "spam_rate": 0.33}}}}
    text = RP.render(fab_results(), None, diagnostics=diag, effect=eff, cite="docs/research/v4.3-20261015/verdict-mid/results.json",
                     cite_diagnostics="docs/research/v4.3-20261015/verdict-mid/diagnostics_numbers.json",
                     cite_effect="docs/research/v4.3-20261015/verdict-mid/effect.json")
    assert text.startswith("# v4.3 Phase 0 report (generated)") and "v4.2" not in text.split("\n")[0]
    for h in ("## Probes (mean per run)", "## Drug released on poor-uptake tissue", "## Gap to S' by cause", "## Message effect by kind",
              "## Realised mean u per sub-region and level"):
        assert h in text, h
    for p in ("docs/research/v4.3-20261015/verdict-mid/results.json", "docs/research/v4.3-20261015/verdict-mid/diagnostics_numbers.json",
              "docs/research/v4.3-20261015/verdict-mid/effect.json", "docs/research/v4.3-20261015/uptake/realised_u.json"):
        assert f"`{p}`" in text, p
    assert "estimate and an association" in text and "| S-marks | 2 | 5.00 | 2.00 |" in text
    assert "| S | 2 | 0.3000 | n/a | n/a | 0.0100 | 0.2900 |" in text                    # S: only the unreleased pool is available
    assert "| mid | 40 |" in text                                                          # realised u: B1's committed file, all 40 cases


def test_realised_u_section_reads_the_committed_b1_file_and_degrades_without_it():
    doc = RP.load_realised_u()
    assert doc is not None and set(doc["levels"]) == {"low", "mid", "high"}
    lines = "\n".join(RP.realised_u_section(doc))
    assert "| low | 40 |" in lines and "| high | 40 |" in lines
    assert "Not available" in "\n".join(RP.realised_u_section(None))


def test_every_table_the_report_adds_names_a_source():
    text = RP.render(fab_results(), None, diagnostics={"poor_tissue_release": {"S-marks": {"n_runs": 1}}}, cite="r.json")
    sections = text.split("\n## ")[1:]
    added = [s for s in sections if s.startswith(("Probes", "Drug released", "Realised mean u"))]
    assert len(added) == 3 and all("Source:" in s for s in added)
