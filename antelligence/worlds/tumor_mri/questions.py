"""Typed questions per squad-round (INTERFACES section 5), state rendering and the harness decision rule.

Labels are single characters so a model answers with one token: targets ``A``..``F`` (candidates present) plus
``S`` (stay) and ``X`` (explore); release ``yes``/``no``; amount ``1``..``5``. Choice options are always in fixed
label order and never shuffled (ties go to the earlier label), so requests are built directly, not with
``DecisionRequest.build`` (which shuffles).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from antelligence.kernel.decision import MAX_STATE_CHARS, Answer, DecisionRequest, Question
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


def build_questions(candidate_labels: Sequence[str]) -> Dict[str, Question]:
    """The three questions for one squad-round; only candidates present are offered (in A..F order)."""
    present = [l for l in CANDIDATE_LABELS if l in set(candidate_labels)]
    options = tuple(present) + (STAY, EXPLORE)
    criteria = {l: f"head to region {l}" for l in present}
    criteria[STAY] = "keep the current target (a squad with no target stays where it is)"
    criteria[EXPLORE] = "go to the nearest perceived region no squad has visited, else the nearest most uncertain area"
    return {
        Q_TARGET: Question("choice", "Where should this squad go next?", options, criteria),
        Q_RELEASE: Question("noul", "Should this squad release drug here and now?",
                            criteria={"true": "release drug", "false": "keep the drug"}),
        Q_AMOUNT: Question("score", "How much of its remaining drug should it release? "
                           + ", ".join(f"{l} = {_pct(f)}" for l, f in zip(AMOUNT_LABELS, AMOUNT_FRACTIONS)) + ".",
                           AMOUNT_LABELS),
    }


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
    return f"{at} {now}".strip()


def render_state(fields: Mapping[str, Any], brief: str = "") -> str:
    """Words, not raw numbers, from the observation fields (never truth). Truncated to MAX_STATE_CHARS."""
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
    if brief:
        room = MAX_STATE_CHARS - len(text) - len(" Brief: ")
        if room > 0:
            text += " Brief: " + brief[:room]
    return text[:MAX_STATE_CHARS]


def build_request(model: str, fields: Mapping[str, Any], brief: str = "", seed: int = 0) -> DecisionRequest:
    labels = [c["label"] for c in fields.get("candidates", [])]
    return DecisionRequest(model=model, state=render_state(fields, brief), questions=build_questions(labels), seed=seed)


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
    return {"target": target, "release": int(p_yes >= 0.5), "amount": level,
            "p_target": best, "p_release": p_yes, "p_amount": abest}
