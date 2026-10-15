"""Model-facing colony state, mark question, ask trigger and options rule (v4.2 goal section 11, items 1-3; no model calls).

Everything here is a pure function of an ant's own observation (``fields``) or of the perceived regions. Nothing reads the
scorer, the F1-F3 facts or a queen output other than what already arrives as the observation keys ``zone_need`` and
``colony_note`` (information barrier, goal 2.8).

* :func:`colony_sentences`     the colony part of ``render_state`` (marks near you, zones, messages, readings, note, notebook).
* :func:`should_ask`           the ask trigger for model ants, computed from the ant's own ``target`` answer. Thresholds are
                               parameters; they are fitted in Phase 1 on development logs (``PLACEHOLDER_THRESHOLDS`` is NOT a fit).
* :func:`build_answer_request` spec of the one ``noul`` question a receiving ant is asked about its own reading log.
* :func:`select_options`       the fixed engine rule for the A-F options. The queen never touches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.kernel.decision import DecisionRequest, Question

COLONY_FIELD_KEYS = ("marks_near", "messages_in", "zone_need", "colony_note", "reading_log", "notebook_self",
                     "cards_public", "region_status", "uptake_reading", "washout_reading")
NOTEBOOK_CHARS = 300
READ_RADIUS_MM = 6.0
ZONE_MARGIN_MM = 6.0
MAX_AUTHORS_SHOWN = 3
MAX_MESSAGES_SHOWN = 4
NOTE_SHOWN_CHARS = 80

# plain words for the mark kinds (v4.3 appends good_uptake), judgement kinds first (they carry an author's claim)
KIND_ORDER = ("tumour_here", "not_tumour", "need_more", "dosed", "dose_ok", "slow_uptake", "washout", "good_uptake")
KIND_WORDS = {"tumour_here": "tumour here", "not_tumour": "not tumour", "need_more": "needs more drug",
              "dosed": "drug given", "dose_ok": "enough drug", "slow_uptake": "slow uptake", "washout": "fast washout",
              "good_uptake": "good uptake"}
JUDGEMENT = ("tumour_here", "not_tumour", "need_more")
ASK_WORDS = {"is_tumour": "is there tumour at {where}?", "dosed_enough": "has {where} had enough drug?",
             "should_i_come": "should it come to {where}?"}
ANSWER_QUESTION = {"is_tumour": "Is there tumour at the spot the other ant asked about?",
                   "dosed_enough": "Has the spot the other ant asked about had enough drug?",
                   "should_i_come": "Does the spot the other ant asked about still need this ant?"}
LOOK_WORDS = ("unlikely", "possible", "likely", "very likely")


def has_colony_fields(fields: Mapping[str, Any]) -> bool:
    """True when any colony observation key is present. Absent keys leave the v4.1 rendering untouched."""
    return any(k in fields for k in COLONY_FIELD_KEYS)


def ant_name(i: int) -> str:
    return f"ant-{int(i):02d}"


def look_word(x: float) -> str:
    for cut, w in zip((0.2, 0.5, 0.8), LOOK_WORDS):
        if x < cut:
            return w
    return LOOK_WORDS[-1]


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: max(0, n - 3)].rstrip() + "..."


def derive_voxel_mm(fields: Mapping[str, Any]) -> Optional[float]:
    """Voxel size from a candidate's own numbers (mm distance over voxel distance). None when it cannot be told."""
    pos = fields.get("position_vox")
    if not pos:
        return None
    for c in fields.get("candidates", ()):
        d = math.dist(pos, c["centroid_vox"])
        if d > 0 and c.get("dist_mm", 0) > 0:
            return float(c["dist_mm"]) / d
    return None


def _bands(fields: Mapping[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for c in fields.get("cards_public", ()) or ():
        try:
            out[int(str(c["ant_id"]).rsplit("-", 1)[1])] = str(c["band"])
        except (KeyError, ValueError, IndexError):
            continue
    return out


def _self_index(fields: Mapping[str, Any]) -> Optional[int]:
    card = fields.get("card_self")
    try:
        return int(str(card["ant_id"]).rsplit("-", 1)[1])
    except (TypeError, KeyError, ValueError, IndexError):
        return int(fields["squad"]) if "squad" in fields else None


def _author_phrase(authors: Sequence[int], bands: Mapping[int, str], me: Optional[int], with_band: bool) -> str:
    names = []
    for a in sorted(set(authors)):
        if a == me:
            names.append("you")
        elif with_band:
            names.append(f"{ant_name(a)} ({bands.get(a, 'new')})")
        else:
            names.append(ant_name(a))
    if len(names) > MAX_AUTHORS_SHOWN:
        names = names[:MAX_AUTHORS_SHOWN] + [f"{len(names) - MAX_AUTHORS_SHOWN} more"]
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] if names else ""


# --- the sentences -----------------------------------------------------------------------------------------------------

def marks_sentence(fields: Mapping[str, Any]) -> str:
    """"Marks near you": one clause per kind. Judgement marks name their authors with the track-record band
    (usually right / mixed / new); automatic marks only count the ants."""
    if "marks_near" not in fields:
        return ""
    rows = fields["marks_near"] or []
    if not rows:
        return "No marks near you."
    bands, me = _bands(fields), _self_index(fields)
    by_kind: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    clauses = []
    for kind in KIND_ORDER:
        rs = by_kind.get(kind)
        if not rs:
            continue
        authors = sorted({a for r in rs for a in r.get("authors", ())})
        strong = any(r.get("strong") for r in rs)
        if kind in JUDGEMENT:
            who = _author_phrase(authors, bands, me, True)
            clauses.append(f"{KIND_WORDS[kind]} from {who}" + (", strong" if strong else ""))
        else:
            clauses.append(f"{KIND_WORDS[kind]} by {len(authors)} ant{'s' if len(authors) != 1 else ''}")
    text = "Marks near you: " + "; ".join(clauses) + "."
    if any(r.get("disputed") for r in rows):
        text += " Tumour and not tumour are both marked here: disputed."
    return text


def _region_of(pos: Sequence[float], cands: Sequence[Mapping[str, Any]], vmm: float) -> Optional[str]:
    """Label of the nearest candidate whose zone (equal-volume sphere + 6 mm, INTERFACES 0) holds ``pos``."""
    best: Optional[Tuple[float, str]] = None
    for c in cands:
        radius = (3.0 * float(c.get("volume_mm3", 0)) / (4.0 * math.pi)) ** (1.0 / 3.0) + ZONE_MARGIN_MM
        d = vmm * math.dist(pos, c["centroid_vox"])
        if d <= radius * (1 + 1e-9) and (best is None or (d, c["label"]) < best):
            best = (d, c["label"])
    return None if best is None else best[1]


def zones_sentence(fields: Mapping[str, Any], voxel_mm: Optional[float] = None) -> str:
    """Per-region summary: zone need, plus the judgement marks near you that fall inside the zone, with author and band."""
    need = fields.get("zone_need")
    cands = fields.get("candidates", ())
    if not need or not cands:
        return ""
    vmm = voxel_mm or derive_voxel_mm(fields)
    bands, me = _bands(fields), _self_index(fields)
    per: Dict[str, Dict[str, set]] = {}
    if vmm and fields.get("marks_near"):
        for r in fields["marks_near"]:
            if r["kind"] in JUDGEMENT:
                lab = _region_of(r["pos_vox"], cands, vmm)
                if lab:
                    per.setdefault(lab, {}).setdefault(r["kind"], set()).update(r.get("authors", ()))
    parts = []
    for c in cands:
        n = need.get(str(c["region_id"]), need.get(c["region_id"]))
        if n is None:
            continue
        s = f"{c['label']} need {n}"
        for kind in JUDGEMENT:
            au = per.get(c["label"], {}).get(kind)
            if au:
                s += f", {KIND_WORDS[kind]} from {_author_phrase(sorted(au), bands, me, True)}"
        parts.append(s)
    return "Zones: " + "; ".join(parts) + "." if parts else ""


def _where(about: Mapping[str, Any], cands: Sequence[Mapping[str, Any]]) -> str:
    rid = about.get("region_id")
    if rid is not None:
        for c in cands:
            if c["region_id"] == rid:
                return f"region {c['label']}"
        return f"region {rid}"
    p = about.get("pos_vox")
    return f"voxel ({p[0]}, {p[1]}, {p[2]})" if p else "that spot"


def messages_sentence(fields: Mapping[str, Any]) -> str:
    """Messages in, newest last. The note is already quoted by the engine and is repeated only as a quote."""
    if "messages_in" not in fields:
        return ""
    msgs = fields["messages_in"] or []
    if not msgs:
        return "No messages."
    cands = fields.get("candidates", ())
    out = []
    for m in msgs[-MAX_MESSAGES_SHOWN:]:
        who = ant_name(m["from_ant"])
        where = _where(m.get("about", {}), cands)
        if m["type"] == "ask":
            s = f"{who} asks {ASK_WORDS.get(m['question'], 'about {where}').format(where=where)}"
            if m.get("sender_reading") is not None:
                s += f" Its own reading: {look_word(float(m['sender_reading']))}."
        else:
            a = m.get("answer")
            if a == "unknown":
                s = f"{who} answers about {where}: no reading there."
            else:
                s = f"{who} answers {a} about {where}"
                if m.get("answerer_reading") is not None:
                    s += f" (its readings: {look_word(float(m['answerer_reading']))})"
                s += "."
        if m.get("note"):
            s += " Note: " + _clip(str(m["note"]), NOTE_SHOWN_CHARS)
        out.append(s)
    extra = len(msgs) - MAX_MESSAGES_SHOWN
    return "Messages: " + " ".join(out) + (f" ({extra} older not shown.)" if extra > 0 else "")


def readings_sentence(fields: Mapping[str, Any], voxel_mm: Optional[float] = None) -> str:
    """Own readings: close looks within 6 mm (count and mean), the last uptake word and the washout word."""
    parts = []
    log = fields.get("reading_log") or []
    pos, vmm = fields.get("position_vox"), voxel_mm or derive_voxel_mm(fields)
    if log and pos and vmm:
        near = [e["value"] for e in log if e["kind"] == "look" and vmm * math.dist(pos, e["pos_vox"]) <= READ_RADIUS_MM * (1 + 1e-9)]
        if near:
            parts.append(f"You looked {len(near)} time{'s' if len(near) != 1 else ''} near here; tumour {look_word(sum(near) / len(near))}.")
    up, wo = fields.get("uptake_reading"), fields.get("washout_reading")
    if up:
        parts.append(f"Uptake after your last release: {up['word']}.")
    if wo:
        parts.append(f"Washout here: {wo['word']}.")
    return " ".join(parts)


def note_sentence(fields: Mapping[str, Any]) -> str:
    note = fields.get("colony_note")
    return f"Colony note: {note}" if note else ""


def notebook_sentence(fields: Mapping[str, Any], cap: int = NOTEBOOK_CHARS) -> str:
    """Up to ``cap`` characters of the frozen notebook (lesson texts joined), in quotes: advice, not an order."""
    lessons = [str(t).strip() for t in (fields.get("notebook_self") or []) if str(t).strip()]
    if not lessons:
        return ""
    return 'Your notebook: "' + _clip(" | ".join(lessons), cap) + '"'


# priority = order of dropping when the state is over its limit (first listed is dropped first)
DROP_ORDER = ("notebook", "note", "readings", "zones", "marks", "messages")


def colony_sentences(fields: Mapping[str, Any], voxel_mm: Optional[float] = None) -> Dict[str, str]:
    """name -> sentence(s), '' when the key is absent or empty."""
    vmm = voxel_mm or derive_voxel_mm(fields)
    return {"marks": marks_sentence(fields), "zones": zones_sentence(fields, vmm), "messages": messages_sentence(fields),
            "readings": readings_sentence(fields, vmm), "note": note_sentence(fields), "notebook": notebook_sentence(fields)}


DISPLAY_ORDER = ("marks", "zones", "messages", "readings", "note", "notebook")


def fit_colony_text(base: str, sentences: Mapping[str, str], max_chars: int) -> str:
    """``base`` + the colony sentences in display order. Over the limit, whole sentences are dropped in ``DROP_ORDER``
    (never cut mid-sentence); the base is never touched here."""
    keep = {k for k, v in sentences.items() if v}

    def build() -> str:
        return " ".join([base] + [sentences[k] for k in DISPLAY_ORDER if k in keep])

    for k in DROP_ORDER:
        if len(build()) <= max_chars:
            break
        keep.discard(k)
    return build()


# --- the mark question -------------------------------------------------------------------------------------------------

Q_MARK = "mark"
# Single characters, none of them a target label (A-F, S, X), so one token on every tokenizer (tested).
MARK_LABELS = ("N", "T", "R", "M")
MARK_KIND_OF = {"N": None, "T": "tumour_here", "R": "not_tumour", "M": "need_more"}
MARK_CRITERIA = {"N": "leave no mark", "T": "mark tumour here", "R": "mark not tumour (ruled out)",
                 "M": "mark that this place needs more drug"}


def mark_question() -> Question:
    return Question("choice", "Should this squad leave a mark here?", MARK_LABELS, dict(MARK_CRITERIA))


# --- the ask trigger ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AskThresholds:
    """``top_below``: ask when the top target probability is under it. ``floor``: ask when more than one option has at least
    this probability. Both are fitted in Phase 1 on development logs; neither is fitted here."""

    top_below: float
    floor: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.top_below <= 1.0 and 0.0 <= self.floor <= 1.0):
            raise ValueError("thresholds are probabilities in [0, 1]")


PLACEHOLDER_THRESHOLDS = AskThresholds(top_below=0.5, floor=0.3)  # a starting point for tests and the projection, NOT a fit


@dataclass(frozen=True)
class AskDecision:
    ask: bool
    reason: Optional[str]          # "low_top" | "split" | "low_top+split" | None
    top: float
    n_above_floor: int


def should_ask(target_probs: Mapping[str, float], thresholds: AskThresholds) -> AskDecision:
    """Pure: ask iff the ant's own top target probability is below ``top_below`` or more than one option is at or above
    ``floor``. Uses only the ant's own answer."""
    if not target_probs:
        raise ValueError("target_probs is empty")
    top = max(float(p) for p in target_probs.values())
    n_above = sum(1 for p in target_probs.values() if float(p) >= thresholds.floor)
    low, split = top < thresholds.top_below, n_above > 1
    reason = "+".join(r for r, f in (("low_top", low), ("split", split)) if f) or None
    return AskDecision(low or split, reason, top, n_above)


# --- the answer: one noul question to the receiver about its own reading log -----------------------------------------------

def answer_state(ask: Mapping[str, Any], reading_log: Sequence[Mapping[str, Any]], about_vox: Sequence[float],
                 voxel_mm: float, radius_mm: float = READ_RADIUS_MM) -> Tuple[str, int]:
    """(state text, number of readings used). Only this ant's own log entries within ``radius_mm`` of the asked spot."""
    near = [e for e in reading_log if voxel_mm * math.dist(e["pos_vox"], about_vox) <= radius_mm * (1 + 1e-9)]
    q = ask["question"]
    if q == "dosed_enough":
        vals = [e["value"] for e in near if e["kind"] == "dose"]
        line = ("Your drug readings there: " + ", ".join("enough" if v >= 1 else "not enough" for v in vals[-3:]) + ".") if vals else ""
    else:
        vals = [e["value"] for e in near if e["kind"] == "look"]
        line = (f"Your close looks there: {len(vals)}, mean tumour {look_word(sum(vals) / len(vals))}.") if vals else ""
    head = f"Another ant, {ant_name(ask['from_ant'])}, asks about a spot near voxel ({int(about_vox[0])}, {int(about_vox[1])}, {int(about_vox[2])})."
    return " ".join(x for x in (head, line or "You have no readings within 6 mm of it.") if x), len(vals)


def build_answer_request(model: str, ask: Mapping[str, Any], reading_log: Sequence[Mapping[str, Any]],
                         about_vox: Sequence[float], voxel_mm: float) -> Optional[DecisionRequest]:
    """The spec of the answer call: ONE ``noul`` question, state built only from the receiver's own reading log.
    Returns None when the receiver has no reading there: the answer is then ``unknown`` and no call is made. This function
    builds a request object; it calls nothing."""
    state, n = answer_state(ask, reading_log, about_vox, voxel_mm)
    if n == 0:
        return None
    return DecisionRequest(model=model, state=state, questions={"answer": Question("noul", ANSWER_QUESTION[ask["question"]])})


def answer_from_p_yes(p_yes: Optional[float]) -> str:
    """yes / no from the noul probability; ``None`` (no reading, no call) is unknown. Same 0.5 line as the scripted rule."""
    return "unknown" if p_yes is None else ("yes" if p_yes >= 0.5 else "no")


# --- the options rule --------------------------------------------------------------------------------------------------

LABELS = ("A", "B", "C", "D", "E", "F")
CLOSED = ("done", "not_tumour")


def select_options(regions: Sequence[Any], position_vox: Sequence[float], voxel_mm: float,
                   region_status: Optional[Mapping[Any, str]] = None, max_options: int = 6) -> List[Dict[str, Any]]:
    """The A-F options for one model ant: the ``max_options`` perceived regions nearest to the ant (centroid distance, ties to
    the lower region id) whose status is not ``done`` or ``not_tumour``. Labelled A-F in order of distance. A fixed engine
    rule: the inputs are the scan's regions, the ant's position and the board's region status. No queen output enters.

    Rows have the shape of v4.1 candidates (label, region_id, centroid_vox, p_tumour, unc, volume_mm3, dist_mm, visited)."""
    status = {int(k): v for k, v in (region_status or {}).items()}
    open_regions = [r for r in regions if status.get(int(r.region_id), "open") not in CLOSED]
    ranked = sorted(open_regions, key=lambda r: (voxel_mm * math.dist(position_vox, r.centroid_vox), r.region_id))[: min(max_options, len(LABELS))]
    return [{"label": LABELS[i], "region_id": r.region_id, "centroid_vox": list(r.centroid_vox), "p_tumour": r.p_tumour,
             "unc": r.unc, "volume_mm3": r.volume_mm3, "dist_mm": round(voxel_mm * math.dist(position_vox, r.centroid_vox), 3),
             "visited": False} for i, r in enumerate(ranked)]


def with_options(fields: Mapping[str, Any], regions: Sequence[Any], voxel_mm: float) -> Dict[str, Any]:
    """A copy of an observation whose ``candidates`` are replaced by this ant's options (from its own position and the
    observed ``region_status``). ``visited`` is carried over from the original candidate list when the region is in it."""
    seen = {c["region_id"]: c.get("visited", False) for c in fields.get("candidates", ())}
    opts = select_options(regions, fields["position_vox"], voxel_mm, fields.get("region_status"))
    for o in opts:
        o["visited"] = bool(seen.get(o["region_id"], False))
    return {**fields, "candidates": opts}
