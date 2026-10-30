# Look around before you rest: a better stop rule (2026-10-30)

**The leak.** In the finish world most of the work a swarm leaves when it says "done" is left by its stop rule, not
by cheats (`EARNED_TRUST.md`). Today an ant counts itself idle after 10 ticks with nothing to do, wherever it spent
them, so it often paces over the same empty sites while work waits a few steps away. When 95% of ants are idle, the
swarm calls it done.

**The fix.** While an ant has nothing to do, it walks to next-door sites it has not yet found empty, and it counts
itself idle only after finding nothing at several *different* sites since it last worked (`places`). Still no counter,
no checker and no boss: each ant only knows where it has been. `walk` is the ablation: the same walk, but counted in
ticks as before.

Code: `idle_rule` in `antelligence/worlds/finish/world.py` (`ticks`, the default and unchanged; `places`; `walk`).

## The fair test, at equal time

Waiting longer always leaves less work, so rules are compared at the same time to "done". On development seeds 0 to
9, each new rule got the largest patience whose mean time to "done" was no longer than the current rule's at 10 and
at 20 ticks. Then all ran on fresh seeds 100 to 129, at 256 and 1,024 ants, with no shortcut-takers and with 20%,
paired by seed. Two looking regimes: slow (re-checking a claim takes 3 ticks, spot looking one claim in four) and
cheap (1 tick, look at every claim). `stop_rule.json`.

Work left when the swarm said "done" (lower is better), current rule at patience 10 against "look around" matched to
it:

| looking | ants | shortcut-takers | now | look around | difference (95% CI) | seeds lower | time to "done" |
|---|---|---|---|---|---|---|---|
| slow | 256 | none | 0.368% | **0.079%** | -0.29 [-0.38, -0.20] | 26 / 30 | 92 vs 91 ticks |
| slow | 256 | 20% | 0.443% | **0.186%** | -0.26 [-0.35, -0.17] | 26 / 30 | 106 vs 108 |
| slow | 1,024 | none | 0.437% | **0.100%** | -0.34 [-0.39, -0.29] | 30 / 30 | 92 vs 91 |
| slow | 1,024 | 20% | 0.414% | **0.177%** | -0.24 [-0.29, -0.19] | 30 / 30 | 109 vs 111 |
| cheap | 256 | none | 0.250% | **0.039%** | -0.21 [-0.28, -0.15] | 25 / 30 | 71 vs 70 |
| cheap | 256 | 20% | 0.257% | **0.041%** | -0.22 [-0.29, -0.16] | 28 / 30 | 72 vs 71 |
| cheap | 1,024 | none | 0.263% | **0.043%** | -0.22 [-0.25, -0.19] | 30 / 30 | 71 vs 70 |
| cheap | 1,024 | 20% | 0.233% | **0.059%** | -0.17 [-0.21, -0.14] | 29 / 30 | 72 vs 71 |

- **Less work left at the same time, in every case.** Against patience 10 the work left falls by 57% to 85%, in the
  same time. Against patience 20, "look around" leaves 22% to 81% less work *and* says "done" 7 to 16 ticks sooner;
  every interval is below 0.
- **It holds with cheats and without**, at both sizes and both looking costs. The one time cost: with 20% cheats and
  slow looking it says "done" about 2 ticks later than the current rule at patience 10 (intervals at or just over 0).
- **The walk does most of the work.** `walk` (same walk, old count) gets nearly all the gain. Counting places instead
  of ticks adds a little more when looking is slow, because ticks spent re-checking no longer count as idle
  (curve below), and is the same as `walk` when looking is cheap.

Curve at 256 ants, slow looking, 20% shortcut-takers (work left @ ticks to "done", by patience):

| rule | | | | | | |
|---|---|---|---|---|---|---|
| now (ticks) | 10: 0.443% @ 106 | 15: 0.266% @ 120 | 20: 0.171% @ 133 | 30: 0.093% @ 156 | | |
| walk | 5: 0.659% @ 79 | 8: 0.376% @ 88 | 10: 0.315% @ 94 | 12: 0.253% @ 99 | 15: 0.179% @ 108 | 20: 0.106% @ 122 |
| look around (places) | 5: 0.464% @ 84 | 8: 0.301% @ 95 | 10: 0.231% @ 101 | 12: 0.186% @ 108 | 15: 0.110% @ 119 | 20: 0.093% @ 130 |

Look around at patience 20 matches the current rule at patience 30 (0.093%) 26 ticks sooner.

**Reading.** This is the harvester-ant rule done properly: a colony decides from how often ants *meet* work, so an ant
should be meeting new places, not the same empty one. It fixes the leak that earned trust could not. Not yet the
default (`ticks` stays the default so every recorded result regenerates); switching it is one line once agreed.

## Reproduce

```
uv run python scripts/stop_rule_check.py --out docs/research/swarm-scale-20261029/stop_rule.json
uv run --extra test pytest -q tests/smarter_swarm/test_stop_rule.py
```
