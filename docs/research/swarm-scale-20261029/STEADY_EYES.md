# Steady eyes: an ant that checks its own eyes (2026-10-30)

**The gap.** Think alone fixed stale news, but not broken eyes. In the SwarmBench foraging world's ``v2`` case, every
ant's view is mirrored west to east from some tick on, and think alone drops from about 16 deliveries to about 4
(`foraging.json`, plan section 0.4: "an ant can't out-look a broken eye"). That was item 3 on the plan's "What next".

**The idea.** An ant already knows where some walls are, because it saw them before. Each tick it asks: does what I
see now fit my own map better as it is, or mirrored? It keeps a small running score of that answer, and once the
mirrored reading has clearly fitted better for a few ticks, it reads its eyes mirrored from then on. It only adds a
view to its map when that view agrees with how it is reading its eyes, and while it reads mirrored it ignores what
others say (their eyes are flipped too). It is never told whether or when the flip comes. No checker, no teacher,
no talk: one ant checking itself against its own past.

Code: `SelfCheckingSensor` and `SteadyEyesForager` in `antelligence/worlds/foraging_sb/think_alone.py`. While the eyes
are right it reports exactly what the plain sensor reports (tested), so nothing changes in ``base`` or ``v1``.

**Now the default.** `default_ant()` in the foraging world returns steady eyes. With fine eyes it is think alone,
move for move (tested), so think alone stays the fallback; the eye check only acts once the ant's own map says the
view is mirrored.

## The fair test

Fresh seeds 200 to 279 (80 layouts, 10 ants each; earlier checks used 100 to 139), paired by seed, bootstrap 95%
intervals. Four ants on the same layouts: the pilot's script for v1 (`S_v1`), think alone as it stands (`look`), steady
eyes (`steady`), and the pilot's un-mirroring script, which is **told** the flip tick (`unmirror`, a reference that
knows more than any ant should). The flip came at tick 30 (the benchmark's), and also at 10 and 60, so a rule that
only works for one tick would show. Score: deliveries.

| case | think alone | steady eyes | told the flip tick | steady minus think alone (95% CI) | seeds steady higher / lower |
|---|---|---|---|---|---|
| eyes fine (base) | 16.40 | 16.40 | 15.80 | +0.00 (identical runs) | 0 / 0 |
| food runs out (v1) | 16.89 | 16.89 | 16.50 | +0.00 (identical runs) | 0 / 0 |
| eyes flip at tick 30 (v2) | 3.94 | **15.36** | 15.78 | +11.43 [+10.95, +11.89] | 80 / 0 |
| eyes flip at tick 10 | 0.31 | **14.84** | 15.43 | +14.53 [+14.00, +15.03] | 80 / 0 |
| eyes flip at tick 60 | 9.81 | **16.16** | 15.79 | +6.35 [+6.04, +6.66] | 80 / 0 |

First-hand talk shown. With relayed memories too, every steady-minus-think-alone gap is the same (flip at tick 10:
+14.49 [+13.96, +14.99]); only v1 moves, to 17.07 for both, because think alone and steady eyes play identically
there.

- **It fixes the case.** Steady eyes gets back almost everything the flip took, on every one of 80 layouts, whenever
  the flip comes.
- **It nearly matches the ant that was told.** Against the script told the exact flip tick: -0.41 [-1.14, +0.31] at
  tick 30, -0.59 [-1.23, +0.03] at tick 10, +0.38 [-0.16, +0.94] at tick 60. None of these is clear either way.
- **It costs nothing when eyes are fine.** In base and v1 not one ant doubted its eyes (0 false switches in 320 runs),
  so the runs are identical to think alone.
- **It notices fast.** Each ant switched once and never flapped. About 77% of ants switched within 2 ticks of the
  flip and about 91 to 93% within 5. A few took longer (at most 70 to 84 ticks): ants that knew little of the map yet
  had little to check against. 5 to 13 ants of 800 per case never switched.

## Does it hinge on the settings?

The two settings (how much evidence before switching, 2; how fast old evidence fades, 0.8) were picked on quick runs
that touched seeds 200 to 219. So a grid of 12 settings ran on the older seeds 100 to 139, which played no part in
picking them (`steady_eyes_grid.json`). Think alone on those seeds: 16.18 base, 16.62 v1, 3.92 v2.

| needs | fades | v2 (flip 30) | flip 10 | flip 60 | false switches (all cases) |
|---|---|---|---|---|---|
| 1 | 0.6 / 0.8 / 0.9 | 16.00 / 15.97 / 15.68 | 15.15 / 15.18 / 15.05 | 15.72 / 15.95 / 16.00 | 0 |
| 2 | 0.6 / 0.8 / 0.9 | 15.93 / 15.65 / 15.68 | 15.28 / 15.25 / 15.18 | 16.05 / 16.02 / 16.07 | 0 |
| 3 | 0.6 / 0.8 / 0.9 | 15.65 / 15.57 / 15.53 | 15.35 / 14.75 / 14.43 | 15.95 / 16.10 / 16.12 | 0 |
| 4 | 0.6 / 0.8 / 0.9 | 15.47 / 15.55 / 15.43 | 14.40 / 14.50 / 14.62 | 16.00 / 16.10 / 16.07 | 0 |

Every setting lands between 14.4 and 16.1 deliveries after a flip, against about 4 for think alone, and base and v1
stay identical to think alone (16.18 and 16.62) at every setting. Needing less evidence is a little better here; the
default stays at 2 because it was set before this check.

## What this does not show

- The world only flips once and never back. The sensor is built to switch back the same way, but no run here tests
  that.
- It needs walls it has already seen. An ant dropped somewhere new, with eyes broken from the start, has nothing to
  check against. The flip at tick 10 is the closest case here, and it still worked.
- It is a scripted ant in one world. The lesson for model ants is the rule, not the code: check what you see against
  what you already know, and trust the reading that fits.

## Reproduce

```
uv run python scripts/steady_eyes_check.py --out docs/research/swarm-scale-20261029/steady_eyes.json
uv run python scripts/steady_eyes_check.py --grid --seeds 100-139 --out docs/research/swarm-scale-20261029/steady_eyes_grid.json
uv run --extra test pytest -q tests/smarter_swarm/test_steady_eyes.py
```
