"""Goal 11 item 2: the `mark` question, single-token labels on all three tokenizers, the ask trigger, the answer spec."""

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

from antelligence.kernel.decision import Answer, DecisionRequest
from antelligence.worlds.tumor_mri.colony_state import (ANSWER_QUESTION, MARK_KIND_OF, MARK_LABELS, PLACEHOLDER_THRESHOLDS, AskThresholds,
                                                        answer_from_p_yes, build_answer_request, should_ask)
from antelligence.worlds.tumor_mri.questions import (ALL_LABELS, CANDIDATE_LABELS, STAY, EXPLORE, build_questions, build_request,
                                                     decide_from_answers)
from tests.v4_2.p1.helpers import base_fields, colony_fields

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import v42_phase1_prep as prep  # noqa: E402


def test_mark_labels_are_four_distinct_single_characters_outside_the_other_labels():
    assert len(MARK_LABELS) == 4 and len(set(MARK_LABELS)) == 4
    assert all(len(l) == 1 for l in MARK_LABELS)
    assert not set(MARK_LABELS) & set(ALL_LABELS)
    assert [MARK_KIND_OF[l] for l in MARK_LABELS] == [None, "tumour_here", "not_tumour", "need_more"]


def test_v41_questions_unchanged_without_mark():
    qs = build_questions(["A", "B"])
    assert list(qs) == ["target", "release", "amount"]
    assert list(build_questions(["A", "B"], with_mark=True)) == ["target", "release", "amount", "mark"]
    r1 = build_request("m", base_fields())
    r2 = build_request("m", base_fields(), with_mark=False)
    assert r1.request_hash == r2.request_hash
    assert "mark" not in r1.payload()["questions"]


def test_target_release_amount_are_kept_with_mark():
    qs = build_questions(["A", "B", "C"], with_mark=True)
    assert qs["target"].options == ("A", "B", "C", STAY, EXPLORE)
    assert qs["release"].type == "noul" and qs["amount"].options == ("1", "2", "3", "4", "5")
    assert qs["mark"].type == "choice" and qs["mark"].options == MARK_LABELS


def test_decide_from_answers_reads_the_mark():
    req = build_request("m", colony_fields(), with_mark=True)
    opts = req.questions["target"].options
    tgt = {o: 1.0 / len(opts) for o in opts}
    ans = {"target": Answer("choice", opts[0], tgt), "release": Answer("noul", 0.2, {"yes": 0.2, "no": 0.8}),
           "amount": Answer("score", 2.0, {str(i): 0.2 for i in range(5)}),
           "mark": Answer("choice", "T", {"N": 0.1, "T": 0.6, "R": 0.2, "M": 0.1})}
    d = decide_from_answers(ans, req)
    assert d["mark"] == "T" and d["p_mark"] == pytest.approx(0.6) and MARK_KIND_OF[d["mark"]] == "tumour_here"
    tied = {**ans, "mark": Answer("choice", "N", {l: 0.25 for l in MARK_LABELS})}
    assert decide_from_answers(tied, req)["mark"] == "N"                       # ties go to the earlier label
    plain = build_request("m", base_fields())
    assert "mark" not in decide_from_answers({k: v for k, v in ans.items() if k != "mark"}, plain)


@pytest.mark.parametrize("name", sorted(prep.TOKENIZERS))
def test_mark_labels_are_single_tokens(name):
    """Bare, after the JSON quote the backends force, and with a leading space; every label is one token on this tokenizer.
    Skips when the public tokenizer is not in /Volumes/WD_BLACK/cache/hf-tokenizers (run `download-tokenizers`)."""
    tok = prep.load_tokenizers([name]).get(name)
    if tok is None:
        pytest.skip(f"{name}: tokenizer not in the offline cache")
    prefix = prep.n_tokens(tok, '{"mark": "')
    for lab in MARK_LABELS:
        assert prep.n_tokens(tok, lab) == 1
        assert prep.n_tokens(tok, '{"mark": "' + lab) - prefix == 1
        assert prep.n_tokens(tok, " " + lab) == 1


# --- the ask trigger ------------------------------------------------------------------------------------------------------

TH = AskThresholds(top_below=0.5, floor=0.3)


def test_confident_answer_does_not_ask():
    d = should_ask({"A": 0.8, "B": 0.1, "S": 0.05, "X": 0.05}, TH)
    assert not d.ask and d.reason is None and d.n_above_floor == 1 and d.top == pytest.approx(0.8)


def test_low_top_probability_asks():
    d = should_ask({l: 0.125 for l in "ABCDEFSX"}, TH)
    assert d.ask and d.reason == "low_top"


def test_two_options_above_the_floor_ask_even_when_top_is_high():
    d = should_ask({"A": 0.55, "B": 0.35, "S": 0.1}, TH)
    assert d.ask and d.reason == "split" and d.top == pytest.approx(0.55)


def test_both_reasons_are_reported():
    assert should_ask({"A": 0.4, "B": 0.4, "S": 0.2}, TH).reason == "low_top+split"


def test_thresholds_are_parameters_not_constants():
    probs = {"A": 0.6, "B": 0.4}
    assert not should_ask(probs, AskThresholds(0.5, 0.45)).ask
    assert should_ask(probs, AskThresholds(0.7, 0.45)).ask
    assert should_ask(probs, AskThresholds(0.5, 0.35)).ask
    assert "thresholds" in inspect.signature(should_ask).parameters
    with pytest.raises(ValueError):
        AskThresholds(1.5, 0.2)
    with pytest.raises(ValueError):
        should_ask({}, TH)


def test_trigger_is_pure_and_uses_only_the_ants_own_answer():
    probs = {"A": 0.45, "B": 0.4, "S": 0.15}
    before = dict(probs)
    a, b = should_ask(probs, TH), should_ask(probs, TH)
    assert a == b and probs == before
    assert list(inspect.signature(should_ask).parameters) == ["target_probs", "thresholds"]
    assert PLACEHOLDER_THRESHOLDS == AskThresholds(0.5, 0.3)


# --- the answer: one noul question about the receiver's own reading log ----------------------------------------------------

LOG = [{"tick": 1, "pos_vox": [10, 10, 10], "kind": "look", "value": 0.8, "band": 2, "word": None},
       {"tick": 2, "pos_vox": [11, 10, 10], "kind": "look", "value": 0.6, "band": 2, "word": None},
       {"tick": 3, "pos_vox": [60, 60, 60], "kind": "look", "value": 0.0, "band": 0, "word": None},
       {"tick": 3, "pos_vox": [10, 10, 10], "kind": "dose", "value": 1.2, "band": None, "word": None}]
ASK = {"from_ant": 3, "question": "is_tumour"}


def test_answer_request_is_exactly_one_noul_question():
    req = build_answer_request("m", ASK, LOG, (10, 10, 10), 2.0)
    assert isinstance(req, DecisionRequest) and list(req.questions) == ["answer"]
    assert req.questions["answer"].type == "noul"
    assert req.questions["answer"].instructions == ANSWER_QUESTION["is_tumour"]


def test_answer_state_uses_only_the_receivers_own_log_near_the_spot():
    req = build_answer_request("m", ASK, LOG, (10, 10, 10), 2.0)
    assert "2, mean tumour likely" in req.state            # the two near looks (0.8, 0.6), not the far one
    assert "ant-03" in req.state and len(req.state) < 300
    far = build_answer_request("m", ASK, LOG, (30, 30, 30), 2.0)
    assert far is None                                       # no reading near: unknown, no call at all


def test_dose_question_uses_dose_readings():
    req = build_answer_request("m", {"from_ant": 1, "question": "dosed_enough"}, LOG, (10, 10, 10), 2.0)
    assert "enough" in req.state and "looks" not in req.state


def test_p_yes_to_word():
    assert answer_from_p_yes(0.5) == "yes" and answer_from_p_yes(0.49) == "no" and answer_from_p_yes(None) == "unknown"
