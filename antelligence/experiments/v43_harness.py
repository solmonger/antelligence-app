"""v4.3 Step A harness: the v4.2 harness with the fixed ant (``AntRules``) and the combined estimate.

``V43Harness`` reuses :class:`V42Harness` (it overrides only the construction hooks): every ``ColonyScript`` gets the rules,
every card gets ``min_resolved``, the world gets the estimate (so each look reading carries ``scan_p`` and ``est``), claims are
resolved on ``est`` when the rules use it, and each record carries the rules and their sha. With ``AntRules.v42()`` and no
estimate the episodes are byte-identical to ``V42Harness`` (the legacy golden test pins it).

It registers no arm. The default SimLook cell table stays the committed, read-only v4.2 one.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from antelligence.experiments.v42_harness import (CaseData, ColonyKit, HarnessWorld, V42Harness, bundle_of as v42_bundle_of)
from antelligence.kernel.colony.cards import Card
from antelligence.kernel.colony.episode import ColonyConfig
from antelligence.worlds.tumor_mri import WorldConfig
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_policies import ColonyScript, ScanInfo, _est_at_tick
from antelligence.worlds.tumor_mri.combined import CombinedEstimate
from antelligence.worlds.tumor_mri.data import Region
from antelligence.worlds.tumor_mri.foraging import queen_uptake_scaling, select_auto_mark_rule

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL_TABLE = REPO_ROOT / "docs" / "research" / "v4.2-20261014" / "simlook" / "cells_v4_2.json"   # read-only
CODE_FILES = ("worlds/tumor_mri/colony_policies.py", "worlds/tumor_mri/move_on.py", "worlds/tumor_mri/ant_rules.py", "worlds/tumor_mri/combined.py",
              "worlds/tumor_mri/colony_world.py", "worlds/tumor_mri/colony_queen.py", "worlds/tumor_mri/foraging.py", "kernel/colony/episode.py",
              "kernel/colony/messages.py", "kernel/colony/thresholds.py", "kernel/colony/cards.py",
              "experiments/v42_harness.py", "experiments/v43_harness.py")


def code_sha_v43() -> str:
    """sha256 over the files whose behaviour the v4.3 ant depends on (a superset of the v4.2 ``code_sha``)."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parents[1]
    for rel in CODE_FILES:
        h.update(rel.encode())
        h.update((here / rel).read_bytes())
    return h.hexdigest()


class V43World(HarnessWorld):
    """``HarnessWorld`` with the combined estimate: each look reading also carries ``scan_p`` and ``est``."""

    def __init__(self, crop: Any, config: WorldConfig, seed: int, *, arm: str, look_fn: Optional[Callable] = None,
                 regions: Optional[Sequence[Region]] = None, scorer_channels: Optional[Mapping[str, np.ndarray]] = None,
                 estimate: Optional[CombinedEstimate] = None) -> None:
        super().__init__(crop, config, seed, arm=arm, look_fn=look_fn, regions=regions, scorer_channels=scorer_channels)
        self.estimate = estimate


class V43Kit(ColonyKit):
    """K2 wiring with the rules: scripts get ``rules``, cards get ``min_resolved``, claims resolve on ``est`` when used."""

    def __init__(self, arm: str, n: int, scan: ScanInfo, *, sharing: bool, rules: AntRules, notebook_cap: int = 1500) -> None:
        self.rules = rules                                  # set before the base class builds cards and scripts
        super().__init__(arm, n, scan, sharing=sharing, notebook_cap=notebook_cap)

    def _new_card(self, i: int) -> Card:
        legacy = self.rules.version == "v42"
        return Card(f"ant-{i:02d}", self.arm, rev="v4.2" if legacy else "v4.3", min_resolved=self.rules.card_min_resolved)

    def _new_script(self, i: int) -> ColonyScript:
        return ColonyScript(self.scan, i, self.rules)

    def claim_look_value(self, entry: Mapping[str, Any]) -> float:
        if self.rules.use_combined and entry.get("est") is not None:
            return float(entry["est"])
        return float(entry["value"])

    def view_look_value(self, view: Any) -> float:
        o = view.observation
        if self.rules.use_combined:
            est = _est_at_tick(o.get("reading_log") or (), int(view.tick))
            if est is not None:
                return est
        return float(o["look"])


class V43Harness(V42Harness):
    def __init__(self, world_cfg: WorldConfig, colony_cfg: Optional[ColonyConfig] = None, rules: Optional[AntRules] = None,
                 estimate: Optional[CombinedEstimate] = None, *, cell_table: Optional[Mapping[str, Any]] = None,
                 cell_table_path: Optional[Path] = None, look_fn_factory: Optional[Callable[[CaseData], Callable]] = None,
                 with_board: bool = True) -> None:
        self.rules = rules if rules is not None else AntRules.v42()
        self.estimate = estimate
        if self.rules.use_combined and estimate is None:
            raise ValueError("AntRules.use_combined needs a CombinedEstimate (pass the coefficient file)")
        base = colony_cfg or ColonyConfig()
        base = replace(base, answer_rule=self.rules.answer_rule)       # the kernel's answer rule follows the ant rules
        super().__init__(world_cfg, base, cell_table=cell_table, cell_table_path=cell_table_path or DEFAULT_CELL_TABLE,
                         look_fn_factory=look_fn_factory, with_board=with_board)

    def build_world(self, arm: Any, case: CaseData, seed: int, look_fn: Optional[Callable]) -> V43World:
        world_arm = "S" if arm.kind != "ant" else arm.driver_arm
        w = V43World(case.crop, self.world_cfg, seed, arm=world_arm, look_fn=look_fn, regions=case.regions,
                     scorer_channels=case.scorer_channels, estimate=self.estimate)
        if w.queen_hook is not None:
            raise RuntimeError("world.queen_hook must stay unset in every colony arm")
        return w

    def build_scan(self, world: Any, case: CaseData) -> ScanInfo:
        """The scan every ant sees. With ``rules.dose_move_on`` or ``rules.foraging`` it also carries the scan's own p map
        (``case.crop.p_tumour``, perception, never truth), which the move-on and the probe-and-hold rule need; with both switches
        off it is the v4.2 scan, byte for byte."""
        scan = super().build_scan(world, case)
        if self.rules.dose_move_on or self.rules.foraging:
            scan = replace(scan, p_tumour=np.asarray(case.crop.p_tumour))
        return scan

    def make_kit(self, arm: str, n: int, scan: ScanInfo, sharing: bool) -> V43Kit:
        return V43Kit(arm, n, scan, sharing=sharing, rules=self.rules)

    def make_queen(self, world: Any, board: Any, ccfg: ColonyConfig, sharing: bool) -> Any:
        """B3: the queen scales the zone need by the uptake marks only in the foraging model and only in sharing arms."""
        queen = super().make_queen(world, board, ccfg, sharing)
        queen.uptake_scaling = queen_uptake_scaling(self.world_cfg, sharing)
        return queen

    def auto_mark_rule(self) -> Any:
        """B1: the foraging model adds ``good_uptake`` and fires ``slow_uptake`` on the word ``poor``; otherwise the v4.2 rule."""
        return select_auto_mark_rule(self.world_cfg)

    def run_episode(self, arm_name: str, case: CaseData, seed: int, **kw: Any) -> Dict[str, Any]:
        rec = super().run_episode(arm_name, case, seed, **kw)
        rec["ant_rules"] = {"rules": self.rules.to_dict(), "sha": self.rules.sha()}
        rec["estimate"] = None if self.estimate is None else {"coef": list(self.estimate.coef), "sha": self.estimate.sha()}
        return rec


def bundle_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    """The v4.2 ``bundle.json`` body plus ``extra.ant_rules`` (the rules and their sha) when the record has them."""
    b = v42_bundle_of(record)
    if record.get("ant_rules") is not None:
        b["extra"]["ant_rules"] = record["ant_rules"]
    return b
