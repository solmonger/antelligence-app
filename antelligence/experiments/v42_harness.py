"""v4.2 Phase 0 colony harness (Q1, goal section 7): builds one episode per (arm, case, seed) and runs it.

Per episode it builds the tumour world with scorer-side truth (``HarnessWorld``), injects SimLook as ``look_fn`` (F1), turns
on the W1 sensors (F2 / F3, when configured), wires the K1 episode driver and the K2 cards / notebooks / thresholds, and runs
one of the registered arms:

    S           the v4.1 script, unchanged (colony off)
    S-solo      colony script, own readings only
    S-marks     colony script + marks, region status, public cards
    S-talk      S-marks + asking when unsure
    S-marks-quiet   S-marks with the quiet rule ON (the only arm that has it on)
    S'          the oracle S' (knows F1-F3); "S-prime" is accepted as a command-line alias only

Rules this module enforces:

* Development cases only. A case whose split is ``eval`` or ``reserve`` (or unknown) is refused before anything is loaded.
  A synthetic phantom is labelled ``synthetic``.
* ``world.queen_hook`` stays UNSET in every colony arm; the scripted queen goes through the driver's ``queen=`` argument.
* The quiet rule is OFF in every verdict arm.
* Shared state (marks, messages, status) reaches an ant only through the driver's observation, which the runtime barrier
  check (``base_keys=OBS_KEYS``) pins per arm. The claim ledger below feeds name cards only; the colony scripts never read
  cards, and in S-solo ``card_self`` carries no count that depends on another ant's looks.

Control hooks (M1 supplies the implementations): ``mark_pos_fn`` (displaced marks), ``obs_providers`` (per-key overrides for
the keys the driver does not own), and :func:`register_arm` for arms such as ``S-marks-displaced`` / ``S-talk-shuffled``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from antelligence.experiments.v41_gate import GateError, load_checked_split
from antelligence.experiments.v41_split import which_split
from antelligence.kernel.board import Board
from antelligence.kernel.colony.cards import Card
from antelligence.kernel.colony.episode import (ColonyConfig, ColonyResult, plain, run_colony_episode)
from antelligence.kernel.colony.marks import JUDGEMENT_KINDS
from antelligence.kernel.colony.notebook import Notebook
from antelligence.worlds.tumor_mri import OBS_KEYS, ScriptS, WorldConfig, oracle_for
from antelligence.worlds.tumor_mri.colony_obs import arm_spec
from antelligence.worlds.tumor_mri.colony_policies import ColonyScript, ScanInfo, automatic_marks
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from antelligence.worlds.tumor_mri.colony_world import ColonyTumorWorld
from antelligence.worlds.tumor_mri.data import Crop, Region
from antelligence.worlds.tumor_mri.simlook import SimLook, load_table
from antelligence.worlds.tumor_mri.store_adapter import load_case

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL_TABLE = REPO_ROOT / "docs" / "research" / "v4.2-20261014" / "simlook" / "cells_v4_2.json"
ALLOWED_SPLITS = ("dev", "synthetic")
CLAIM_RADIUS_MM = 4.0                      # INTERFACES 8: a claim is confirmed / contradicted within 4 mm
TUMOUR_CLAIMS = ("tumour_here", "need_more")


class EvalRefused(PermissionError):
    """The harness runs development cases only (and synthetic phantoms); an evaluation case is refused."""


# --- arms ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmDef:
    """``kind``: ``script_s`` (v4.1 S, colony off), ``ant`` (the colony script) or ``oracle`` (S')."""

    name: str
    kind: str
    driver_arm: str                        # key of colony_obs.ALLOWED (what the ant may see)
    quiet_rule: bool = False
    verdict: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("script_s", "ant", "oracle"):
            raise ValueError(f"bad arm kind {self.kind!r}")


ARMS: Dict[str, ArmDef] = {a.name: a for a in (
    ArmDef("S", "script_s", "S"),
    ArmDef("S-solo", "ant", "S-solo"),
    ArmDef("S-marks", "ant", "S-marks"),
    ArmDef("S-talk", "ant", "S-talk"),
    ArmDef("S-marks-quiet", "ant", "S-marks-quiet", quiet_rule=True, verdict=False),
    ArmDef("S'", "oracle", "S"),
)}
VERDICT_ARMS = ("S", "S-solo", "S-marks", "S-talk", "S'")


def register_arm(arm: ArmDef) -> None:
    """Add an arm (M1's controls). The script of an ``ant`` arm is the shared decision function; only the inputs differ."""
    ARMS[arm.name] = arm


def resolve_arm(name: str) -> ArmDef:
    """The arm named exactly ``name``. The oracle is ``S'`` everywhere; ``S-prime`` is a command-line alias only
    (``scripts/v42_run.py`` rewrites it before it gets here)."""
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; registered: {sorted(ARMS)}")
    return ARMS[name]


# --- cases --------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseData:
    case_id: str
    crop: Crop
    regions: Optional[Sequence[Region]]
    scorer_channels: Optional[Mapping[str, np.ndarray]]
    split: str                              # "dev" | "synthetic"


def assert_dev_case(case_id: str, repo_root: Path = REPO_ROOT) -> None:
    """Refuse any case that is not in the development split. Loads nothing but the split file."""
    try:
        kind = which_split(load_checked_split(repo_root), case_id)
    except KeyError as exc:
        raise EvalRefused(f"case {case_id!r} is not in the split; refused") from exc
    if kind != "dev":
        raise EvalRefused(f"case {case_id!r} is in the {kind} set; the Phase 0 harness runs development cases only")


def load_dev_case(case_id: str, *, root: Optional[Path] = None, repo_root: Path = REPO_ROOT) -> CaseData:
    assert_dev_case(case_id, repo_root)
    try:
        loaded = load_case(case_id, root=root, repo_root=repo_root)
    except GateError as exc:                                   # belt and braces: the adapter gates too
        raise EvalRefused(str(exc)) from exc
    return CaseData(case_id, loaded.crop, loaded.regions, loaded.scorer_channels, "dev")


def phantom_case(path: Path, case_id: str = "phantom") -> CaseData:
    return CaseData(case_id, Crop.from_npz(path, case_id=case_id), None, None, "synthetic")


# --- the world ----------------------------------------------------------------------------------------------------------

class HarnessWorld(ColonyTumorWorld):
    """``ColonyTumorWorld`` plus the W1 sensor readings behind the colony sensors (scorer-side TC / ET channels, F2 Source A,
    are forwarded by ``ColonyTumorWorld`` itself)."""

    def __init__(self, crop: Crop, config: WorldConfig, seed: int, *, arm: str, look_fn: Optional[Callable] = None,
                 regions: Optional[Sequence[Region]] = None, scorer_channels: Optional[Mapping[str, np.ndarray]] = None) -> None:
        super().__init__(crop, config, seed, arm=arm, look_fn=look_fn, queen_hook=None, regions=regions,
                         scorer_channels=scorer_channels)
        self.uptake_fn = self._uptake
        self.washout_fn = self._washout

    def _uptake(self, i: int, tick: int, release: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        r = self.sensor_readings(f"squad-{i}", tick)["uptake"]
        return None if r is None else {"value": r["reading"], "word": r["word"], "pos_vox": list(r["pos_vox"])}

    def _washout(self, i: int, tick: int, pos: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        r = self.sensor_readings(f"squad-{i}", tick)["washout"]
        return None if r is None else {"value": r["reading"], "word": r["word"], "pos_vox": list(r["pos_vox"])}


# --- claims, cards, notebooks ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    ant: int
    kind: str                               # judgement mark kind, or "answer_yes" / "answer_no"
    pos: Tuple[int, int, int]
    tick: int

    @property
    def says_tumour(self) -> bool:
        return self.kind in TUMOUR_CLAIMS or self.kind == "answer_yes"


def resolve_claims(claims: Sequence[Claim], looks: Sequence[Tuple[int, Tuple[int, int, int], int, float]], voxel_mm: float,
                   before_tick: Optional[int] = None) -> List[Optional[bool]]:
    """For each claim: True (confirmed), False (contradicted) or None (unresolved). The first look by ANOTHER ant after the
    claim (strictly later tick, and strictly before ``before_tick`` when given) within 4 mm decides: look >= 0.5 confirms a
    tumour claim and contradicts a not-tumour claim."""
    out: List[Optional[bool]] = []
    for c in claims:
        best: Optional[Tuple[int, int, float]] = None
        for ant, pos, tick, value in looks:
            if ant == c.ant or tick <= c.tick or (before_tick is not None and tick >= before_tick):
                continue
            if voxel_mm * float(np.linalg.norm(np.subtract(pos, c.pos))) > CLAIM_RADIUS_MM * (1 + 1e-9):
                continue
            if best is None or (tick, ant) < best[:2]:
                best = (tick, ant, value)
        out.append(None if best is None else (best[2] >= 0.5) == c.says_tumour)
    return out


class ColonyKit:
    """Per-episode K2 wiring: one card, one frozen notebook and one script per ant, plus the observation providers.

    ``sharing`` mirrors the arm: only a sharing arm sees other ants' confirmations / contradictions on the cards (S-solo's
    ``card_self`` carries own claims and thresholds only, so it never depends on another ant's reading)."""

    def __init__(self, arm: str, n: int, scan: ScanInfo, *, sharing: bool, notebook_cap: int = 1500) -> None:
        self.arm, self.n, self.scan, self.sharing = arm, int(n), scan, bool(sharing)
        self.cards = [Card(f"ant-{i:02d}", arm) for i in range(n)]
        for c in self.cards:
            c.begin_run()                                        # cards reset at the start of every run (goal 6.4)
        self.notebooks = [Notebook(f"ant-{i:02d}", notebook_cap) for i in range(n)]
        for nb in self.notebooks:
            nb.freeze()                                          # frozen: required for any evaluation (goal 6.7)
        self.scripts = [ColonyScript(scan, i) for i in range(n)]
        self._claims: List[Claim] = []
        self._resolved: Dict[int, bool] = {}
        self._looks: List[Tuple[int, Tuple[int, int, int], int, float]] = []
        self.providers: Dict[str, Callable[[int, int], Any]] = {
            "card_self": self._card_self, "cards_public": self._cards_public, "notebook_self": self._notebook_self}

    # observation providers (called by the driver per ant and tick)
    def _sync(self, tick: int) -> None:
        if not self.sharing:
            return
        idx = [k for k in range(len(self._claims)) if k not in self._resolved]
        for k, r in zip(idx, resolve_claims([self._claims[k] for k in idx], self._looks, self.scan.voxel_mm, before_tick=tick)):
            if r is not None:
                self._resolved[k] = r
                (self.cards[self._claims[k].ant].confirm if r else self.cards[self._claims[k].ant].contradict)()

    def _card_self(self, i: int, tick: int) -> Dict[str, Any]:
        self._sync(tick)
        return self.cards[i].to_dict()

    def _cards_public(self, i: int, tick: int) -> List[Dict[str, Any]]:
        self._sync(tick)
        return [c.public() for j, c in enumerate(self.cards) if j != i]

    def _notebook_self(self, i: int, tick: int) -> List[str]:
        return self.notebooks[i].snapshot()

    def policy_for(self, agent_id: str) -> "_AntPolicy":
        return _AntPolicy(self, int(agent_id.rsplit("-", 1)[1]))

    def cards_fn(self) -> Dict[str, Any]:
        return {c.ant_id: c.to_dict() for c in self.cards}

    def counters(self) -> Dict[str, int]:
        total: Dict[str, int] = {}
        for s in self.scripts:
            for k, v in s.state.counters.items():
                total[k] = total.get(k, 0) + v
        return dict(sorted(total.items()))

    def finalize(self, bundle: Mapping[str, Any], truth: Optional[np.ndarray], regions: Sequence[Region]) -> Dict[str, Any]:
        """Final in-run counts (judgement claims AND answers, resolved against every ant's later looks) and, on a dev or
        synthetic case, the cross-run check against the scorer. Returns the final card dicts; the live ``cards_after`` in the
        bundle is left exactly as the driver saw it, so a replay still reproduces the bundle."""
        vmm = self.scan.voxel_mm
        centroid = {r.region_id: tuple(int(round(c)) for c in r.centroid_vox) for r in regions}
        claims = [Claim(m["ant"], m["mark_kind"], tuple(m["pos_vox"]), m["tick"]) for m in bundle["marks"]
                  if m["mark_kind"] in JUDGEMENT_KINDS]
        for m in bundle["messages"]:
            if m["type"] == "answer" and m["answer"] in ("yes", "no"):
                a = m["about"]
                pos = tuple(a["pos_vox"]) if a["pos_vox"] is not None else centroid.get(a["region_id"])
                if pos is not None:
                    claims.append(Claim(m["from_ant"], f"answer_{m['answer']}", pos, m["tick"]))
        looks = [(int(ant), tuple(e["pos_vox"]), int(e["tick"]), float(e["value"]))
                 for ant, log in bundle["reading_logs"].items() for e in log if e["kind"] == "look"]
        res = resolve_claims(claims, looks, vmm)
        for c in self.cards:
            c.in_run = {"claims": 0, "confirmed": 0, "contradicted": 0}
        correct = {i: 0 for i in range(self.n)}
        checked = {i: 0 for i in range(self.n)}
        for c, r in zip(claims, res):
            card = self.cards[c.ant]
            card.add_claim()
            if r is True:
                card.confirm()
            elif r is False:
                card.contradict()
            if truth is not None and all(0 <= p < s for p, s in zip(c.pos, truth.shape)):
                checked[c.ant] += 1
                correct[c.ant] += bool(truth[c.pos]) == c.says_tumour
        for i, c in enumerate(self.cards):
            c.thresholds = dict(self.scripts[i].state.thresholds.theta)
            c.end_run(correct[i], checked[i])
        return {c.ant_id: c.to_dict() for c in self.cards}


class _AntPolicy:
    """What the driver sees: ``decide(LocalView) -> Intent``. Wraps the shared script and feeds the cards."""

    def __init__(self, kit: ColonyKit, i: int) -> None:
        self.kit, self.i = kit, i

    def decide(self, view: Any) -> Any:
        kit, i = self.kit, self.i
        o = view.observation
        look = o.get("look")
        if look is not None:
            kit._looks.append((i, tuple(int(c) for c in o["position_vox"]), int(view.tick), float(look)))
        intent = kit.scripts[i].decide(view)
        card = kit.cards[i]
        for d in intent.emit:
            if d.kind in JUDGEMENT_KINDS:
                card.add_claim()
                kit._claims.append(Claim(i, d.kind, tuple(int(round(c)) for c in d.pos), int(view.tick)))
        card.thresholds = dict(kit.scripts[i].state.thresholds.theta)
        return intent

    def describe(self) -> Dict[str, Any]:
        return self.kit.scripts[self.i].describe()


# --- the harness ----------------------------------------------------------------------------------------------------------

def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_sha() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parents[1]
    for rel in ("worlds/tumor_mri/colony_policies.py", "worlds/tumor_mri/colony_queen.py", "experiments/v42_harness.py"):
        h.update((here / rel).read_bytes())
    return h.hexdigest()


def make_board(arm: str, case_id: str, seed: int, split: str, world_cfg: Mapping[str, Any], harness_cfg: Mapping[str, Any]) -> Board:
    """An unsigned (mode A) board. The header's keys are placeholders derived from author names: nothing here is a secret."""
    authors = ["perception", "queen", "harness", "router"] + [f"squad-{i}" for i in range(8)]
    header = {"study": "v4.2", "arm": arm, "case": case_id, "seed": int(seed), "split": "dev" if split == "synthetic" else split,
              "world_cfg_sha256": _sha(world_cfg), "harness_cfg_sha256": _sha(harness_cfg),
              "pubkeys": {a: hashlib.sha256(f"v4.2-q1-placeholder-key|{a}".encode()).hexdigest() for a in authors},
              "anchor_pubkey": hashlib.sha256(b"v4.2-q1-placeholder-key|anchor").hexdigest(), "code_sha": code_sha()}
    return Board(header, "A")


class V42Harness:
    """Builds and runs episodes. One harness holds the world config, the colony config and the SimLook cell table."""

    def __init__(self, world_cfg: WorldConfig, colony_cfg: Optional[ColonyConfig] = None, *,
                 cell_table: Optional[Mapping[str, Any]] = None, cell_table_path: Optional[Path] = None,
                 look_fn_factory: Optional[Callable[[CaseData], Callable]] = None, with_board: bool = True) -> None:
        self.world_cfg = world_cfg
        self.colony_cfg = colony_cfg or ColonyConfig()
        self.with_board = with_board
        self.look_fn_factory = look_fn_factory
        self._table = cell_table
        self._table_path = Path(cell_table_path) if cell_table_path else DEFAULT_CELL_TABLE
        self._simlook_cache: Tuple[Optional[str], Optional[SimLook]] = (None, None)

    # SimLook ----------------------------------------------------------------------------------------------------------
    def cell_table(self) -> Mapping[str, Any]:
        if self._table is None:
            self._table = load_table(self._table_path)
        return self._table

    def _look_fn(self, case: CaseData) -> Tuple[Optional[Callable], Dict[str, Any], float]:
        """(look_fn, description, build seconds). The SimLook field needs scorer-side truth, so it is built here, never in a policy."""
        if self.look_fn_factory is not None:
            return self.look_fn_factory(case), {"sensor": "injected look_fn"}, 0.0
        if not self.world_cfg.facts.f1_simlook:
            return None, {"sensor": None}, 0.0
        t0 = time.perf_counter()
        key, cached = self._simlook_cache
        if key != case.case_id:
            table = self.cell_table()
            if table.get("source") != self.world_cfg.facts.f1_source:
                raise ValueError(f"facts.f1_source is {self.world_cfg.facts.f1_source!r} but the cell table was built from "
                                 f"{table.get('source')!r}; the source is recorded, never inferred")
            cached = SimLook(case.case_id, case.crop.truth, case.crop.p_tumour, case.crop.voxel_mm, table)
            self._simlook_cache = (case.case_id, cached)
        return cached, cached.describe(), time.perf_counter() - t0

    # world ------------------------------------------------------------------------------------------------------------
    def build_world(self, arm: ArmDef, case: CaseData, seed: int, look_fn: Optional[Callable]) -> HarnessWorld:
        world_arm = "S" if arm.kind != "ant" else arm.driver_arm
        w = HarnessWorld(case.crop, self.world_cfg, seed, arm=world_arm, look_fn=look_fn, regions=case.regions,
                         scorer_channels=case.scorer_channels)
        if w.queen_hook is not None:
            raise RuntimeError("world.queen_hook must stay unset in every colony arm")
        return w

    # episode ----------------------------------------------------------------------------------------------------------
    def run_episode(self, arm_name: str, case: CaseData, seed: int, *,
                    mark_pos_fn: Optional[Callable[[int, int, str, Sequence[int]], Sequence[int]]] = None,
                    obs_providers: Optional[Mapping[str, Callable[..., Any]]] = None,
                    replay: Optional[Mapping[str, Any]] = None, drop: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        arm = resolve_arm(arm_name)
        if case.split not in ALLOWED_SPLITS:
            raise EvalRefused(f"split {case.split!r} is refused; development cases and synthetic phantoms only")
        if case.split == "dev":
            assert_dev_case(case.case_id)
        if arm.quiet_rule is False and self.colony_cfg.quiet_rule and arm.kind == "ant":
            raise ValueError("the quiet rule must be OFF in every verdict arm; use S-marks-quiet for the pair")
        ccfg = replace(self.colony_cfg, quiet_rule=True) if arm.quiet_rule else self.colony_cfg
        t_start = time.perf_counter()
        look_fn, look_desc, look_build_s = self._look_fn(case)
        if arm.kind == "ant" and look_fn is None:
            raise ValueError("colony arms take a close look each turn: turn on facts.f1_simlook (or inject a look_fn)")
        world = self.build_world(arm, case, seed, look_fn)
        t_built = time.perf_counter()

        board = (make_board(arm.name, case.case_id, seed, case.split, self.world_cfg.to_dict(),
                            {"arm": arm.name, "colony": ccfg.to_dict()}) if self.with_board and arm.kind == "ant" else None)
        kit: Optional[ColonyKit] = None
        res: ColonyResult
        if arm.kind == "script_s":
            res = run_colony_episode(world, lambda a: ScriptS(), arm=arm_spec("S"), base_keys=OBS_KEYS)
        elif arm.kind == "oracle":
            oracle = oracle_for(world)
            res = run_colony_episode(world, lambda a: oracle, arm=arm_spec("S"), base_keys=OBS_KEYS)
        else:
            spec = arm_spec(arm.driver_arm)
            scan = ScanInfo(tuple(world.regions), world.voxel_mm)
            kit = ColonyKit(arm.name, self.world_cfg.n_squads, scan, sharing=spec.reads_marks)
            queen = ScriptedQueen(world.regions, world.total_drug, self.world_cfg.n_squads, world.voxel_mm, board=board,
                                  half_life=ccfg.half_life)
            providers = {**kit.providers, **(obs_providers or {})}
            policy_for = None if (replay is not None and not drop) else kit.policy_for
            res = run_colony_episode(world, policy_for, arm=spec, config=ccfg, queen=queen, obs_providers=providers,
                                     auto_mark_rule=automatic_marks, board=board, base_keys=OBS_KEYS, cards_fn=kit.cards_fn,
                                     mark_pos_fn=mark_pos_fn, replay=replay, drop=drop)
            if world.queen_hook is not None:
                raise RuntimeError("world.queen_hook was set during a colony episode")
        t_run = time.perf_counter()

        record: Dict[str, Any] = {
            "arm": arm.name, "case_id": case.case_id, "seed": int(seed), "split": case.split, "metrics": res.metrics,
            "timing_s": {"build": round(t_built - t_start, 4), "simlook": round(look_build_s, 4),
                         "run": round(t_run - t_built, 4), "wall": round(t_run - t_start, 4)},
            "sensor": look_desc, "facts": self.world_cfg.facts.__dict__ if self.world_cfg.facts.any_on else None,
            "total_drug": world.total_drug, "n_regions": len(world.regions), "quiet_rule": bool(ccfg.quiet_rule and arm.kind == "ant"),
            "queen_hook_unset": world.queen_hook is None,
        }
        if arm.kind == "ant" and kit is not None:
            record["colony"] = res.colony
            record["script_counters"] = kit.counters()
            record["cards_final"] = kit.finalize(res.colony, world.scorer_truth().truth, world.regions)
            record["queen_alarms"] = list(queen.alarms)
            record["board_entries"] = 0 if board is None else len(board)
        else:
            record["colony"] = res.colony if arm.kind != "ant" else None
        if res.effect is not None:
            record["effect"] = res.effect
        return record

    def replay(self, record: Mapping[str, Any], case: CaseData) -> Dict[str, Any]:
        """Re-run an ant-arm episode from its bundle with no policy. Metrics must equal the original's exactly."""
        return self.run_episode(record["arm"], case, record["seed"], replay=record["colony"])


def bundle_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``bundle.json`` body of a record (INTERFACES 12): ``extra.colony`` plus the metrics and identity."""
    return plain({"arm": record["arm"], "case_id": record["case_id"], "seed": record["seed"], "split": record["split"],
                  "metrics": record["metrics"], "extra": {"colony": record.get("colony")}})
