"""Typed questions per squad-round (INTERFACES section 5), state rendering and the harness decision rule.

Labels are single characters so a model answers with one token: targets ``A``..``F`` (candidates present) plus
``S`` (stay) and ``X`` (explore); release ``yes``/``no``; amount ``1``..``5``. Choice options are always in fixed
label order and never shuffled (ties go to the earlier label), so requests are built directly, not with
``DecisionRequest.build`` (which shuffles).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from antelligence.kernel.decision import MAX_STATE_CHARS, Answer, DecisionRequest, Question
from antelligence.worlds.tumor_mri.colony_state import (Q_MARK, colony_sentences, fit_colony_text, has_colony_fields,
                                                         mark_question)
from antelligence.worlds.tumor_mri.config import AMOUNT_FRACTIONS

CANDIDATE_LABELS = ("A", "B", "C", "D", "E", "F")
STAY, EXPLORE = "S", "X"
TARGET_ORDER = CANDIDATE_LABELS + (STAY, EXPLORE)
AMOUNT_LABELS = ("1", "2", "3", "4", "5")
RELEASE_LABELS = ("yes", "no")
ALL_LABELS = TARGET_ORDER + AMOUNT_LABELS + RELEASE_LABELS
Q_TARGET, Q_RELEASE, Q_AMOUNT = "target", "release", "amount"


def _pct(f: float) -> str:
    return f"{int(round(f * 100))}%"


def build_questions(candidate_labels: Sequence[str], with_mark: bool = False) -> Dict[str, Question]:
    """The three questions for one squad-round; only candidates present are offered (in A..F order).
    ``with_mark`` (v4.2, model ants) adds the fourth question ``mark``; off, the dict and its request hash are exactly v4.1's."""
    present = [l for l in CANDIDATE_LABELS if l in set(candidate_labels)]
    options = tuple(present) + (STAY, EXPLORE)
    criteria = {l: f"head to region {l}" for l in present}
    # Goal 4.2 wording, verbatim. The prompt text is hashed into config/cache keys, so changing it invalidates reply caches.
    # Goal 4.2: "Stay: keep the current target."
    criteria[STAY] = "keep the current target"
    # Goal 4.2: "Explore: go to the nearest perceived region no squad has visited. If there is none, go to the nearest
    # highest-uncertainty voxel cluster."
    criteria[EXPLORE] = ("go to the nearest perceived region no squad has visited. If there is none, "
                         "go to the nearest highest-uncertainty voxel cluster")
    qs = {
        Q_TARGET: Question("choice", "Where should this squad go next?", options, criteria),
        Q_RELEASE: Question("noul", "Should this squad release drug here and now?",
                            criteria={"true": "release drug", "false": "keep the drug"}),
        Q_AMOUNT: Question("score", "How much of its remaining drug should it release? "
                           + ", ".join(f"{l} = {_pct(f)}" for l, f in zip(AMOUNT_LABELS, AMOUNT_FRACTIONS)) + ".",
                           AMOUNT_LABELS),
    }
    if with_mark:
        qs[Q_MARK] = mark_question()
    return qs


def _band(x: float, cuts: Tuple[float, float, float], words: Tuple[str, str, str, str]) -> str:
    for c, w in zip(cuts, words):
        if x < c:
            return w
    return words[-1]


def _where(fields: Mapping[str, Any]) -> str:
    """Position and current target, as the world holds them (the spec's state: position, payload, last release)."""
    pos = fields.get("position_vox")
    at = f"Position: voxel ({pos[0]}, {pos[1]}, {pos[2]})." if pos else ""
    if "goal_dist_mm" not in fields:
        return at
    d = fields["goal_dist_mm"]
    now = ("Current target: none, standing still." if d is None else
           "Current target: reached." if d <= 0 else f"Current target: moving, {int(round(d))} mm to go.")
    if d is not None and fields.get("target_source") == "queen":
        now = now[:-1] + " (assigned by the queen)."
    return f"{at} {now}".strip()


def render_state(fields: Mapping[str, Any], brief: str = "", max_chars: int = MAX_STATE_CHARS) -> str:
    """Words, not raw numbers, from the observation fields (never truth). Truncated to ``max_chars`` (v4.1: 1,200).

    With no colony key in ``fields`` the output is byte-for-byte v4.1 (golden test). With colony keys the colony sentences
    (marks near you, zones, messages, readings, colony note, notebook) follow the v4.1 text; over the limit, whole colony
    sentences are dropped, lowest priority first (``colony_state.DROP_ORDER``)."""
    p, unc, dose = fields["local_p_tumour"], fields["local_unc"], fields["local_dose"]
    look = fields.get("local_look")
    parts = [
        "Task: deliver drug to the tumour; the score is the share of tumour tissue that gets enough drug.",
        f"Round {fields['round'] + 1} of {fields['rounds']}, tick {fields['tick']}.",
        _where(fields),
        f"Payload left: {_pct(fields['payload_fraction'])}.",
        f"Here the scan says tumour is {_band(p, (0.2, 0.5, 0.8), ('unlikely', 'possible', 'likely', 'very likely'))}"
        f" and the scan is {_band(unc, (0.15, 0.35, 0.6), ('sure', 'fairly sure', 'unsure', 'very unsure'))}.",
        f"Drug already here: {'none' if dose <= 0 else _band(fields['local_dose_rel'], (0.25, 0.75, 1.0), ('little', 'some', 'nearly enough', 'enough'))}.",
    ]
    if look is not None:
        parts.append(f"Local look: tumour {_band(look, (0.2, 0.5, 0.8), ('unlikely', 'possible', 'likely', 'very likely'))}.")
    lr = fields.get("last_release")
    parts.append(f"Last release: {_pct(lr['fraction'])} at tick {lr['tick']}." if lr else "No release yet.")
    cands = fields.get("candidates", [])
    if cands:
        parts.append("Regions: " + "; ".join(
            f"{c['label']} {_band(c['p_tumour'], (0.5, 0.7, 0.9), ('weak', 'probable', 'strong', 'certain'))}, {int(round(c['dist_mm']))} mm away"
            for c in cands) + ".")
    else:
        parts.append("No regions known.")
    text = " ".join(x for x in parts if x)
    if has_colony_fields(fields):
        text = fit_colony_text(text, colony_sentences(fields), max_chars)
    if brief:
        room = max_chars - len(text) - len(" Brief: ")
        if room > 0:
            text += " Brief: " + brief[:room]
    return text[:max_chars]


def build_request(model: str, fields: Mapping[str, Any], brief: str = "", seed: int = 0, with_mark: bool = False) -> DecisionRequest:
    labels = [c["label"] for c in fields.get("candidates", [])]
    return DecisionRequest(model=model, state=render_state(fields, brief), questions=build_questions(labels, with_mark), seed=seed)


def decide_from_answers(answers: Mapping[str, Answer], request: DecisionRequest) -> Dict[str, Any]:
    """Harness decision rule: target = argmax (ties to the earlier label), release iff P(yes) >= 0.5, amount = argmax."""
    target_q = request.questions[Q_TARGET]
    tp = answers[Q_TARGET].probabilities
    target, best = target_q.options[0], -1.0
    for opt in target_q.options:  # fixed label order -> strict '>' keeps the earlier label on ties
        if tp[opt] > best:
            target, best = opt, tp[opt]
    p_yes = answers[Q_RELEASE].p_yes
    ap = answers[Q_AMOUNT].probabilities
    level, abest = 1, -1.0
    for i in range(len(request.questions[Q_AMOUNT].options)):
        if ap[str(i)] > abest:
            level, abest = i + 1, ap[str(i)]
    out = {"target": target, "release": int(p_yes >= 0.5), "amount": level,
           "p_target": best, "p_release": p_yes, "p_amount": abest}
    if Q_MARK in request.questions and Q_MARK in answers:  # v4.2: argmax label (ties to the earlier label); keys absent in v4.1
        mp = answers[Q_MARK].probabilities
        label, mbest = request.questions[Q_MARK].options[0], -1.0
        for opt in request.questions[Q_MARK].options:
            if mp[opt] > mbest:
                label, mbest = opt, mp[opt]
        out[Q_MARK] = label
        out["p_mark"] = mbest
    return out
