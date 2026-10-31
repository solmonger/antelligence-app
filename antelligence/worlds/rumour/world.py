"""Rumour world: one wrong mark, many ants, and how far it spreads. Scripted, seeded, deterministic, no model calls.

The question every ant must answer is one hidden yes/no fact (the truth is drawn per seed). Ants walk a site graph
(Watts-Strogatz, reused from the colouring world). Each tick:

1. **Private evidence.** With probability ``q`` an ant gets a private reading of the fact, right with probability
   ``accuracy``. Its private log-odds ``P`` adds +/- ``LLR`` per reading (``LLR = log(accuracy / (1 - accuracy))``).
   The readings are drawn from a per-ant stream that does not depend on the policy, so every arm sees the same
   evidence (paired comparisons).
2. **Read.** The ant reads every live mark at its current site (marks are visible the tick after they are left) and
   forms its answer ``B = P + social_weight * LLR * (sum of strengths, + for yes, - for no)``. Its own marks are left
   out, so a lone ant cannot convince itself. Before updating, the ant gives each mark it read a feedback signal from
   its **own view** (see the two policies). That feedback moves the mark's half-life only under the smart fade rule.
3. **Leave a mark** at its site (replacing its own older mark there), then move to a random neighbour with
   probability ``move_p``.

Policies (each crossed with the fade rule):

* ``herd`` ("seen each other"): the ant's view is its answer ``B``; it leaves a mark saying ``sign(B)``. What it heard
  becomes what it says, so a wrong mark can be relayed.
* ``alone`` ("think alone, then compare"): the ant's view is its private ``P``, committed before it reads anything; it
  leaves a mark saying ``sign(P)``, and only when it has private evidence. It still *answers* with ``B``. What it heard
  never becomes what it says, so a wrong mark cannot be relayed. No checker is involved.
* ``graded`` ("waggle-dance marks"): think alone, but the mark's strength grows with the ant's own evidence. Its
  amount is the ant's net number of private readings (``|P| / LLR``), at least 1 and capped at ``grade_cap``, like a
  honeybee scout that dances longer for a better site. Liars shout at the cap. With ``grade_cap = 1`` it is exactly
  ``alone``.

Option ``notebook`` ("field notebook", any policy but ``solo``): each honest ant keeps, for every other author whose mark
it has read anywhere, only that author's latest reading (value, strength as read, tick read). Marks at its current site
update the entries (latest overwrites), and it answers ``B = P + social_weight * LLR * sum(value * strength_read *
0.5 ** ((t - t_read) / notebook_half_life))`` over the notebook, so what it heard at earlier sites is not forgotten and
each source counts once, wherever it was met. What it says is unchanged (private evidence only), so nothing heard is
relayed. Feedback to memory is unchanged. ``notebook_half_life = 0`` means "use ``half_life``". Off by default (bit-identical).

Seeds (what is planted at tick 0):

* ``none``: the control.
* ``planted``: one wrong mark of ``planted_amount`` at a random site (as strong as that many ants), default half-life.
* ``burst``: the same wrong mark at a share ``burst_frac`` of the sites at once (a burst of misinformation).
* ``liars``: a share ``liar_frac`` of the ants always leave a wrong mark (and are not scored).

Scores are over honest ants: the share whose answer ``B`` has the true sign at the last tick (``B == 0`` is
undecided, not right). ``spread`` compares a seeded run with its own no-seed control on the same streams.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from antelligence.kernel.fading import FadeRule, FadingMemory
from antelligence.worlds.graph_coloring.graphs import make_graph

POLICIES = ("solo", "herd", "alone", "graded")
SEEDS = ("none", "planted", "burst", "liars")
PLANTED = "planted"


@dataclass(frozen=True)
class RumourConfig:
    n_ants: int = 256
    ticks: int = 60
    q: float = 0.1              # chance of a private reading per ant per tick
    accuracy: float = 0.6       # chance a private reading is right
    move_p: float = 0.5
    social_weight: float = 1.0  # one unit of mark strength counts as much as one private reading
    half_life: float = 10.0
    smart: bool = False
    confirm: float = 1.25
    contradict: float = 0.5
    planted_amount: float = 8.0
    burst_frac: float = 0.05
    liar_frac: float = 0.02
    grade_cap: float = 3.0      # graded policy only: the strongest mark, in private readings
    notebook: bool = False      # field notebook: remember each author's latest mark read anywhere (see module docstring)
    notebook_half_life: float = 0.0  # 0.0 means use half_life
    family: str = "ws"

    def __post_init__(self) -> None:
        if self.n_ants < 8:
            raise ValueError("n_ants must be at least 8")
        if not 0.5 < self.accuracy < 1:
            raise ValueError("accuracy must be in (0.5, 1)")
        if not 0 <= self.q <= 1 or not 0 <= self.move_p <= 1:
            raise ValueError("q and move_p are probabilities")
        if not self.grade_cap >= 1:
            raise ValueError("grade_cap must be at least 1 (one reading)")
        if self.notebook_half_life < 0:
            raise ValueError("notebook_half_life must be 0 (use half_life) or positive")

    @property
    def llr(self) -> float:
        return math.log(self.accuracy / (1 - self.accuracy))

    def fade_rule(self) -> FadeRule:
        if self.smart:
            return FadeRule.smart_rule(self.half_life, self.confirm, self.contradict)
        return FadeRule.fixed(self.half_life)


@dataclass
class RumourResult:
    policy: str
    seed_kind: str
    seed: int
    config: Dict[str, Any]
    truth: int
    right: float            # share of honest ants answering right at the last tick
    wrong: float
    undecided: float
    wrong_by_tick: List[float] = field(default_factory=list)
    planted_gone_tick: Optional[int] = None  # first tick the (last) planted mark is weaker than half a reading
    mean_half_life: float = 0.0
    feedback_events: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class RumourWorld:
    def __init__(self, config: RumourConfig, seed: int) -> None:
        self.cfg, self.seed = config, seed
        graph = make_graph(config.family, config.n_ants, seed)
        self.adj: Dict[int, List[int]] = {i: list(nb) for i, nb in graph.neighbours().items()}
        world_rng = random.Random(f"rumour-world:{seed}:{config.n_ants}")
        self.truth = world_rng.choice((-1, 1))
        self.start = [world_rng.randrange(config.n_ants) for _ in range(config.n_ants)]
        self.planted_site = world_rng.randrange(config.n_ants)
        b = max(1, round(config.burst_frac * config.n_ants))
        self.burst_sites = sorted(world_rng.sample(range(config.n_ants), b))
        k = max(1, round(config.liar_frac * config.n_ants))
        self.liar_pool = sorted(world_rng.sample(range(config.n_ants), k))
        # Private readings and moves: one stream per ant, independent of the policy (common random numbers).
        self.readings: List[List[int]] = []
        self.moves: List[List[float]] = []
        for a in range(config.n_ants):
            r = random.Random(f"rumour-ant:{seed}:{config.n_ants}:{a}")
            row, mv = [], []
            for _ in range(config.ticks):
                got = r.random() < config.q
                right = r.random() < config.accuracy
                row.append(0 if not got else (self.truth if right else -self.truth))
                mv.append(r.random())
            self.readings.append(row)
            self.moves.append(mv)

    def _amount(self, policy: str, p: float, llr: float) -> float:
        """How strong a mark is. 1 for every policy but ``graded``, where it is the ant's net count of private readings
        (rounded: ``|P| / LLR`` is a whole number up to float residue), at least 1 and at most ``grade_cap``."""
        if policy != "graded":
            return 1.0
        return float(min(self.cfg.grade_cap, max(1, round(abs(p) / llr))))

    def run(self, policy: str = "alone", seed_kind: str = "none") -> RumourResult:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if seed_kind not in SEEDS:
            raise ValueError(f"seed_kind must be one of {SEEDS}")
        cfg = self.cfg
        n, llr = cfg.n_ants, cfg.llr
        memory = FadingMemory(cfg.fade_rule())
        liars = set(self.liar_pool) if seed_kind == "liars" else set()
        honest = [a for a in range(n) if a not in liars]
        pos = list(self.start)
        private = [0.0] * n
        answer = [0.0] * n
        nb_half = cfg.notebook_half_life or cfg.half_life
        # Per ant: author -> (value, strength as read, tick read). Kept on the world so tests can look at it.
        self.notebooks: List[Dict[str, tuple]] = [dict() for _ in range(n)]
        planted = None
        if seed_kind == "planted":
            planted = memory.deposit(self.planted_site, PLANTED, -self.truth, tick=0, amount=cfg.planted_amount)
        elif seed_kind == "burst":
            for site in self.burst_sites:
                planted = memory.deposit(site, PLANTED, -self.truth, tick=0, amount=cfg.planted_amount)
        wrong_by_tick: List[float] = []
        gone: Optional[int] = None
        for t in range(1, cfg.ticks + 1):
            # Views are committed before anyone reads (synchronous update).
            view = [(_sign(private[a]) if policy in ("alone", "graded") else _sign(answer[a])) for a in range(n)]
            new_answer = list(answer)
            for a in range(n):
                if a in liars:
                    continue
                private[a] += llr * self.readings[a][t - 1]
                social = 0.0
                if policy != "solo":
                    for m in memory.at(pos[a], t):
                        if m.author == str(a):
                            continue
                        s = m.strength(t)
                        social += s if m.value > 0 else -s
                        if cfg.notebook:
                            self.notebooks[a][m.author] = (m.value, s, t)
                        if view[a] != 0:
                            memory.feedback(m, agrees=(m.value == view[a]), tick=t)
                if cfg.notebook and policy != "solo":
                    social = sum(v * s * 0.5 ** ((t - tr) / nb_half) for v, s, tr in self.notebooks[a].values())
                new_answer[a] = private[a] + cfg.social_weight * llr * social
            answer = new_answer
            if policy != "solo":
                for a in range(n):
                    if a in liars:
                        memory.deposit(pos[a], str(a), -self.truth, tick=t,
                                       amount=cfg.grade_cap if policy == "graded" else 1.0)
                        continue
                    said = _sign(answer[a]) if policy == "herd" else _sign(private[a])
                    if said != 0:
                        memory.deposit(pos[a], str(a), said, tick=t, amount=self._amount(policy, private[a], llr))
            for a in range(n):
                if self.moves[a][t - 1] < cfg.move_p and self.adj[pos[a]]:
                    nb = self.adj[pos[a]]
                    pos[a] = nb[int(self.moves[a][t - 1] / cfg.move_p * len(nb)) % len(nb)]
            if t % 10 == 0:
                memory.prune(t)
            wrong_by_tick.append(sum(_sign(answer[a]) == -self.truth for a in honest) / len(honest))
            if planted is not None and gone is None and planted.strength(t) < 0.5:
                gone = t
        right = sum(_sign(answer[a]) == self.truth for a in honest) / len(honest)
        wrong = wrong_by_tick[-1]
        summary = memory.summary(cfg.ticks)
        return RumourResult(policy, seed_kind, self.seed, asdict(cfg), self.truth, round(right, 6), round(wrong, 6),
                            round(1 - right - wrong, 6), [round(w, 6) for w in wrong_by_tick], gone,
                            round(float(summary["mean_half_life"]), 4), int(summary["feedback_events"]))
