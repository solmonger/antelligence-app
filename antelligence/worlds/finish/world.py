"""Finish world: some ants skip the last bit of a job and call it done. Does the swarm notice? Scripted, seeded, $0.

A job is split into pieces, one per site of a site graph (``sites_per_ant`` pieces per ant) (Watts-Strogatz, reused from the colouring world). Piece
``s`` needs ``work[s]`` units; an ant standing on an unfinished piece does one unit per tick. Ants leave marks at the
site they stand on with ``FadingMemory``: ``+1`` means "work left here", ``-1`` means "done here". Each tick, every
ant:

1. **Decides whether the piece is open.** How depends on the arm (below).
2. **Works** one unit if it believes the piece is open and there is work left.
3. **Gives feedback** to the marks it read (agree or disagree with its own view). Under the smart fade rule that
   feedback moves each mark's half-life: contradicted marks fade fast, confirmed ones last.
4. **Leaves a mark** saying what it believes about the piece, replacing its own older mark there.
5. **Moves** to a neighbour (following "work left" marks next door, otherwise at random) unless it is in the middle
   of an open piece, in which case it stays and keeps working.

**Shortcut-takers** (a share of the ants) work like everyone else until a piece is down to its last ``tail`` share
(default the last 20%), then mark it "done", stop working on it and walk away. That is the reward hack: most of the
job is done, the last bit is skipped, and the record says done.

**When does the swarm say it is finished?** Nobody counts. Each working ant keeps a count of ticks in a row where it
found nothing to do; at ``patience`` ticks it is idle. The swarm calls the job finished when ``stop_share`` of its
working ants are idle (the encounter-rate rule harvester ants use, Gordon 2010). The score is the share of the job's
work still left at that moment, which is work the swarm called done but never did.

**Stop rules** (``idle_rule``). ``ticks`` (the rule above, used by every result recorded before 2026-10-30): an ant
is idle after ``patience`` ticks in a row with nothing to do, wherever it spent them. ``places`` ("look around before
you rest", the default since 2026-10-30, see ``STOP_RULE.md``): an ant is idle only after it
has found nothing to do at ``patience`` *different* sites since it last worked, and while it is looking it walks to
next-door sites it has not yet found empty, when there are any. Ticks spent re-checking or standing on a site it already
found empty do not count. Still no counter and no boss: each ant only knows where it has been. ``walk`` is the
ablation: the same walk to sites not yet found empty, but idle after ``patience`` ticks as in ``ticks``.

Arms:

* ``trust`` (the tick-box): an ant reads the marks where it stands; if "done" outweighs "work left" it believes the
  piece is done, does not look, and repeats "done" in its own mark. A claim travels without anyone looking.
* ``look`` (think alone): an ant always looks at the piece under its feet. Looking at a piece someone marked done costs
  ``look_cost`` ticks (the price of not taking a colleague's word for it). With ``look_p < 1`` it re-checks only
  that share of the time (spot looking) and otherwise walks on without leaving a mark. It marks only what it saw: "work left"
  if the piece is unfinished, "done" only if it really is. Claims are never relayed.
* ``record`` (earned trust, plan idea 1): ``look`` with spot looking, plus a memory of who was caught. When an ant
  re-checks a piece marked done and finds work left, it remembers every ant whose "done" mark was on that piece. From
  then on it always re-checks a claim from any of them, and spot-checks everyone else at ``look_p``. It only trusts
  what it saw itself: nobody tells it who to suspect. With ``look_p = 1`` it is exactly ``look``.
* ``gossip`` (earned trust, shared): ``record``, plus ants standing on the same site at the end of a tick pool the
  names they have caught (word of mouth among peers, as in Ostrom's commons). A name passed on is a claim too, so this
  arm only helps when nobody lies about others; the shortcut-takers here do not.
* ``inspect`` (the checker, for comparison): ``trust`` ants plus a share ``inspector_frac`` of the ants turned into
  inspectors. Inspectors do no work; they walk towards pieces marked done, look, and if work is left leave a strong
  "work left" mark (``inspector_amount``). Inspectors are not counted in the stop rule.

Every arm runs with fixed or smart fading. With the same world seed, the work, the graph, the start sites and which
ants take shortcuts are the same in every arm, so arms compare pair by pair.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from antelligence.kernel.fading import FadeRule, FadingMemory
from antelligence.worlds.graph_coloring.graphs import make_graph

ARMS = ("trust", "look", "inspect", "record", "gossip")
OPEN, DONE = 1, -1
IDLE_RULES = ("ticks", "places", "walk")


@dataclass(frozen=True)
class FinishConfig:
    n_ants: int = 256
    sites_per_ant: float = 4.0     # pieces of the job per ant (one piece per site)
    work_min: int = 2
    work_max: int = 10
    tail: float = 0.2              # shortcut-takers walk away once a piece is down to this share of its work
    look_cost: int = 1             # ticks spent re-checking a piece someone marked done
    look_p: float = 1.0            # look arm: chance an ant re-checks a piece marked done (else it walks on, marking nothing)
    patience: int = 10             # ticks in a row with nothing to do before an ant counts itself idle
    stop_share: float = 0.95       # share of idle working ants at which the swarm calls the job finished
    follow_p: float = 0.8          # chance an ant follows the strongest "work left" mark next door
    half_life: float = 20.0
    smart: bool = False
    confirm: float = 1.25
    contradict: float = 0.5
    inspector_frac: float = 0.1
    inspector_amount: float = 3.0
    max_ticks: int = 600
    family: str = "ws"
    idle_rule: str = "places"      # "places": patience different empty sites (default); "ticks": the old rule; "walk": ablation

    def __post_init__(self) -> None:
        if self.n_ants < 8:
            raise ValueError("n_ants must be at least 8")
        if self.n_sites < 8:
            raise ValueError("need at least 8 sites")
        if not 1 <= self.work_min <= self.work_max:
            raise ValueError("need 1 <= work_min <= work_max")
        if not 0 < self.tail < 1:
            raise ValueError("tail must be in (0, 1)")
        if not 0 < self.stop_share <= 1 or not 0 <= self.follow_p <= 1 or not 0 <= self.look_p <= 1:
            raise ValueError("stop_share and follow_p are shares")
        if self.patience < 1 or self.look_cost < 0:
            raise ValueError("patience must be >= 1 and look_cost >= 0")
        if self.idle_rule not in IDLE_RULES:
            raise ValueError(f"idle_rule must be one of {IDLE_RULES}")

    @property
    def n_sites(self) -> int:
        return max(1, round(self.sites_per_ant * self.n_ants))

    def fade_rule(self) -> FadeRule:
        if self.smart:
            return FadeRule.smart_rule(self.half_life, self.confirm, self.contradict)
        return FadeRule.fixed(self.half_life)


@dataclass
class FinishResult:
    arm: str
    shortcut_share: float
    seed: int
    config: Dict[str, Any]
    total_work: int
    stop_tick: Optional[int]          # tick the swarm called the job finished (None: never, ran to max_ticks)
    left_at_stop: int                 # work units still left when it called the job finished
    skipped_share: float              # left_at_stop / total_work: work called done but never done
    open_pieces_at_stop: int
    finished_tick: Optional[int]      # tick the last unit of work was really done (None: never)
    work_ticks: int                   # ant-ticks spent working
    look_ticks: int                   # ant-ticks spent re-checking pieces someone marked done
    lies: int                         # times a shortcut-taker marked an unfinished piece "done"
    left_by_tick: List[int] = field(default_factory=list)
    caught: int = 0                   # record and gossip: (ant, name) pairs held at the end, i.e. who suspects whom
    caught_wrongly: int = 0           # ... of which the name is not a shortcut-taker (should stay 0)

    @property
    def said_done_early(self) -> bool:
        return self.stop_tick is not None and self.left_at_stop > 0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["said_done_early"] = self.said_done_early
        return d


class FinishWorld:
    def __init__(self, config: FinishConfig, seed: int) -> None:
        self.cfg, self.seed = config, seed
        n, k = config.n_ants, config.n_sites
        graph = make_graph(config.family, k, seed)
        self.adj: Dict[int, List[int]] = {i: sorted(nb) for i, nb in graph.neighbours().items()}
        rng = random.Random(f"finish-world:{seed}:{n}")
        self.work0 = [rng.randint(config.work_min, config.work_max) for _ in range(k)]
        self.start = [rng.randrange(k) for _ in range(n)]
        # One fixed order of ants: shortcut-takers are taken from the front, inspectors from the back, so a larger
        # shortcut share contains the smaller one and the two roles never overlap below half the swarm.
        self.order = list(range(n))
        rng.shuffle(self.order)

    def shortcut_ants(self, share: float) -> Set[int]:
        return set(self.order[: round(share * self.cfg.n_ants)])

    def inspector_ants(self) -> Set[int]:
        k = max(1, round(self.cfg.inspector_frac * self.cfg.n_ants))
        return set(self.order[-k:])

    def run(self, arm: str = "look", shortcut_share: float = 0.0) -> FinishResult:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if not 0 <= shortcut_share < 0.5:
            raise ValueError("shortcut_share must be in [0, 0.5)")
        cfg, n = self.cfg, self.cfg.n_ants
        memory = FadingMemory(cfg.fade_rule())
        work = list(self.work0)
        total = sum(work)
        left = total
        cheats = self.shortcut_ants(shortcut_share)
        inspectors = self.inspector_ants() if arm == "inspect" else set()
        workers = [a for a in range(n) if a not in inspectors]
        look = arm in ("look", "record", "gossip")
        caught: List[Set[str]] = [set() for _ in range(n)]  # record and gossip: authors this ant caught claiming done
        places = cfg.idle_rule in ("places", "walk")  # track the sites found empty since last work, and walk to new ones
        empty: List[Set[int]] = [set() for _ in range(n)]

        def rest(a: int, site: int) -> None:
            if cfg.idle_rule != "places":
                if places:
                    empty[a].add(site)
                idle[a] += 1
            elif site not in empty[a]:
                empty[a].add(site)
                idle[a] += 1

        def wake(a: int) -> None:
            idle[a] = 0
            empty[a].clear()
        pos = list(self.start)
        busy = [0] * n          # ticks still to spend re-checking (look arm)
        idle = [0] * n
        checked: List[Optional[int]] = [None] * n  # the site a look ant already re-checked, until it moves on
        rngs = [random.Random(f"finish-ant:{self.seed}:{n}:{a}") for a in range(n)]
        work_ticks = look_ticks = lies = 0
        stop_tick: Optional[int] = None
        left_at_stop = left
        open_at_stop = sum(w > 0 for w in work)
        finished: Optional[int] = None
        left_by_tick: List[int] = []

        def tally(site: int, t: int, who: int) -> Dict[int, float]:
            out = {OPEN: 0.0, DONE: 0.0}
            for m in memory.at(site, t):
                if m.author != str(who):
                    out[m.value] += m.strength(t)
            return out

        for t in range(1, cfg.max_ticks + 1):
            stay = [False] * n
            for a in range(n):
                s = pos[a]
                marks = [m for m in memory.at(s, t) if m.author != str(a)]
                said_open = sum(m.strength(t) for m in marks if m.value == OPEN)
                said_done = sum(m.strength(t) for m in marks if m.value == DONE)
                claimed = said_done > said_open

                if a in inspectors:
                    view = OPEN if work[s] > 0 else DONE
                    for m in marks:
                        memory.feedback(m, agrees=(m.value == view), tick=t)
                    memory.deposit(s, str(a), view, tick=t, amount=cfg.inspector_amount)
                    continue

                if busy[a] > 0:  # still re-checking this piece
                    busy[a] -= 1
                    look_ticks += 1
                    stay[a] = True
                    continue

                if look:
                    # Earned trust: a claim from an ant this one has caught before is always re-checked.
                    look_p = 1.0 if caught[a] and any(m.author in caught[a] for m in marks if m.value == DONE) \
                        else cfg.look_p
                    if claimed and cfg.look_cost > 0 and checked[a] != s and cfg.look_p < 1 \
                            and rngs[a].random() >= look_p:
                        # Spot looking: this time the ant takes the claim at face value, but it saw nothing, so it
                        # leaves no mark and repeats nothing.
                        checked[a] = s
                        rest(a, s)
                        continue
                    if claimed and cfg.look_cost > 0 and checked[a] != s:
                        checked[a] = s
                        busy[a] = cfg.look_cost - 1
                        look_ticks += 1
                        view = OPEN if work[s] > 0 else DONE
                        for m in marks:
                            memory.feedback(m, agrees=(m.value == view), tick=t)
                        if view == OPEN and arm in ("record", "gossip"):
                            caught[a].update(m.author for m in marks if m.value == DONE)
                        memory.deposit(s, str(a), view, tick=t)
                        stay[a] = view == OPEN or busy[a] > 0
                        if view == OPEN:
                            wake(a)
                        else:
                            rest(a, s)
                        continue
                    believes_open = work[s] > 0
                else:
                    believes_open = (not claimed) and work[s] > 0

                if a in cheats and believes_open and work[s] <= cfg.tail * self.work0[s]:
                    # The reward hack: the last bit is left, the record says done.
                    for m in marks:
                        memory.feedback(m, agrees=(m.value == DONE), tick=t)
                    memory.deposit(s, str(a), DONE, tick=t)
                    lies += 1
                    rest(a, s)
                    continue

                if believes_open:
                    work[s] -= 1
                    left -= 1
                    work_ticks += 1
                    wake(a)
                else:
                    rest(a, s)
                view = OPEN if believes_open and work[s] > 0 else DONE
                if look or not claimed:
                    # What the ant itself saw (look arm, or a trust ant on an unclaimed piece).
                    seen = OPEN if work[s] > 0 else DONE
                    for m in marks:
                        memory.feedback(m, agrees=(m.value == seen), tick=t)
                    memory.deposit(s, str(a), seen, tick=t)
                else:
                    # Trust: a claim is believed and repeated without looking.
                    for m in marks:
                        memory.feedback(m, agrees=(m.value == DONE), tick=t)
                    memory.deposit(s, str(a), DONE, tick=t)
                stay[a] = view == OPEN

            for a in range(n):
                if stay[a] or not self.adj[pos[a]]:
                    continue
                nbs = self.adj[pos[a]]
                r = rngs[a]
                if a in inspectors:
                    score = {v: tally(v, t + 1, a)[DONE] for v in nbs}
                else:
                    score = {v: (lambda c: c[OPEN] - c[DONE])(tally(v, t + 1, a)) for v in nbs}
                best = max(score.values())
                if best > 0 and r.random() < cfg.follow_p:
                    top = [v for v in nbs if score[v] == best]
                    pos[a] = top[r.randrange(len(top))]
                else:
                    fresh = [v for v in nbs if v not in empty[a]] if places else nbs
                    pick = fresh or nbs
                    pos[a] = pick[r.randrange(len(pick))]
                checked[a] = None
            if arm == "gossip":
                here: Dict[int, List[int]] = {}
                for a in workers:
                    here.setdefault(pos[a], []).append(a)
                for group in here.values():
                    if len(group) > 1:
                        pooled = set().union(*(caught[a] for a in group))
                        for a in group:
                            caught[a] = set(pooled)
            if t % 20 == 0:
                memory.prune(t)
            left_by_tick.append(left)
            if finished is None and left == 0:
                finished = t
            if stop_tick is None and t >= cfg.patience:
                idle_share = sum(idle[a] >= cfg.patience for a in workers) / len(workers)
                if idle_share >= cfg.stop_share:
                    stop_tick, left_at_stop = t, left
                    open_at_stop = sum(w > 0 for w in work)
                    break
        if stop_tick is None:
            left_at_stop = left
            open_at_stop = sum(w > 0 for w in work)
        cheat_names = {str(c) for c in cheats}
        held = sum(len(c) for c in caught)
        wrongly = sum(len(c - cheat_names) for c in caught)
        return FinishResult(arm, shortcut_share, self.seed, asdict(cfg), total, stop_tick, left_at_stop,
                            round(left_at_stop / total, 6), open_at_stop, finished, work_ticks, look_ticks, lies,
                            left_by_tick, held, wrongly)
