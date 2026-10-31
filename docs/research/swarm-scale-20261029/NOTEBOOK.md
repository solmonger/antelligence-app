# Field notebook (dream run idea, 2026-10-31)

The nightly dream run built and tested this on 2026-10-31 but could not push, so its code never reached the fork. This
is a rebuild from that run's own record, on the same base commit. The fair check was run again from scratch and every
number below came out the same to the last digit.

**Idea.** An ant forgets everything it heard the moment it walks off a site. Give each honest ant a notebook: for every
*other* author whose mark it has read, anywhere, it keeps only that author's latest reading (value, strength as read,
tick read). Each tick it answers `B = P + weight * LLR * sum(value * strength_read * 0.5 ** ((t - t_read) / notebook
half-life))` over the notebook. Each source counts once, wherever it was met. What it *says* is unchanged (only the sign
and strength of its own private evidence), so nothing heard is relayed. No checker. Option on `RumourConfig`
(`notebook`, `notebook_half_life`; 0 means use `half_life`); off by default and bit-identical to before when off (tested).

## The fair test

Rumour world, scripted, $0, no model calls, fixed fade, graded cap 3. Four arms: alone, graded, alone + notebook,
graded + notebook. Each arm's listening weight was tuned on development seeds 0 to 9 at 256 ants (mean right over
none, burst, liars); notebook arms also tuned the notebook half-life (jointly). Notebook picks sat on a grid edge twice,
so the grids were extended twice (weights to 2, then 4, then 8; half-lives 5-20, then to 80, then to 160). Final picks,
none on an edge: alone weight 4; graded weight 2; alone + notebook weight 6, half-life 40; graded + notebook weight 3,
half-life 80. The plateau is flat: alone + notebook scored 0.9842 at weight 4 and 0.9845 at weight 6. Then every tuned
arm ran on fresh seeds 300 to 319 (256 ants) and 300 to 309 (1,024 ants), paired by seed.

Share of honest ants right at the end:

| ants | condition | alone | graded (current best) | alone + notebook | graded + notebook |
|---|---|---|---|---|---|
| 256 | none | 0.889 | 0.909 | 0.986 | **0.994** |
| 256 | planted | 0.889 | 0.909 | 0.986 | **0.994** |
| 256 | burst | 0.888 | 0.909 | 0.981 | **0.992** |
| 256 | liars | 0.870 | 0.885 | 0.979 | **0.985** |
| 1,024 | none | 0.894 | 0.912 | 0.994 | **0.997** |
| 1,024 | planted | 0.894 | 0.912 | 0.994 | **0.997** |
| 1,024 | burst | 0.894 | 0.911 | 0.989 | **0.995** |
| 1,024 | liars | 0.878 | 0.886 | 0.989 | **0.990** |

Paired differences in right (95% bootstrap CI over seeds):

| ants | condition | graded + notebook minus graded | alone + notebook minus alone |
|---|---|---|---|
| 256 | none | +0.0848 [+0.0723, +0.0982], 20/0 | +0.0977 [+0.0850, +0.1105], 20/0 |
| 256 | burst | +0.0824 [+0.0709, +0.0945], 20/0 | +0.0926 [+0.0805, +0.1049], 20/0 |
| 256 | liars | +0.1000 [+0.0861, +0.1149], 20/0 | +0.1084 [+0.0960, +0.1211], 20/0 |
| 1,024 | none | +0.0851 [+0.0758, +0.0950], 10/0 | +0.0992 [+0.0916, +0.1064], 10/0 |
| 1,024 | burst | +0.0842 [+0.0750, +0.0938], 10/0 | +0.0951 [+0.0871, +0.1028], 10/0 |
| 1,024 | liars | +0.1048 [+0.0956, +0.1140], 10/0 | +0.1112 [+0.1025, +0.1195], 10/0 |

(Planted rows are within 0.001 of none; all in the JSON.)

## The cost: damage from the seed

Extra honest ants wrong, averaged over the run, versus the same world with nothing planted:

| ants | condition | graded | graded + notebook | alone | alone + notebook |
|---|---|---|---|---|---|
| 256 | burst | 0.0135 | 0.0582 | 0.0165 | 0.0758 |
| 256 | liars | 0.0368 | 0.0492 | 0.0227 | 0.0259 |
| 1,024 | burst | 0.0139 | 0.0586 | 0.0167 | 0.0756 |
| 1,024 | liars | 0.0368 | 0.0490 | 0.0226 | 0.0258 |

Graded + notebook minus graded, burst: +0.0447 [+0.0410, +0.0488] (256) and +0.0447 [+0.0420, +0.0477] (1,024); liars:
+0.0124 [+0.0079, +0.0172] (256) and +0.0122 [+0.0094, +0.0153] (1,024). Alone + notebook minus alone, liars: +0.0032
[+0.0004, +0.0063] (256), +0.0032 [+0.0008, +0.0056] (1,024).

## What this means

- Remembering what was heard is worth about 8 to 11 points of ants right, on fresh seeds, with every seed in the same
  direction. Far bigger than waggle-dance marks (+2). Graded + notebook is the best arm everywhere (0.985 to 0.997).
- It is the memory, not the louder voice: even plain alone gains about 10 points with a notebook.
- It is a longer memory of the *past*, not more voices: tuned half-lives are long (40 to 80 ticks, run is 60), so
  the notebook is close to "count every source I ever met once".
- The damage is larger in absolute terms. A wrong planted mark read once is remembered for a long time, so a burst does
  about 4 times the harm (0.058 against 0.014) even though the ants end up far more right. Liars also hurt a little more.
- Burst and planted marks all share one author name, so a notebook counts a whole burst once. The probe below gives each
  copy its own author: the gain mostly holds, and the damage rises a little more.
- Scripted world only. In a real swarm a notebook needs stable author names (the dedup is only as good as the identity).

## A burst from many authors (probe)

`scripts/notebook_burst_probe.py` re-runs the tuned arms on the same fresh seeds with each planted copy carrying its own
author (one per site, 13 copies at 256 ants, 51 at 1,024), so every copy counts in a notebook. With one author it
reproduces the burst rows above exactly.

| ants | burst authors | graded | graded + notebook | gain in right (95% CI) | damage, graded then + notebook |
|---|---|---|---|---|---|
| 256 | one | 0.909 | 0.992 | +0.082 [+0.071, +0.095] | 0.014 then 0.058 |
| 256 | many | 0.909 | **0.988** | +0.079 [+0.068, +0.091] | 0.014 then 0.068 |
| 1,024 | one | 0.911 | 0.995 | +0.084 [+0.075, +0.094] | 0.014 then 0.059 |
| 1,024 | many | 0.911 | **0.993** | +0.082 [+0.073, +0.092] | 0.014 then 0.070 |

Every seed is higher with the notebook in both cases. Without a notebook it makes no difference who signed the marks.
With one, many authors cost about 0.2 to 0.3 points of the gain and add about 1 point of damage
(graded + notebook minus graded, damage: +0.055 [+0.050, +0.060] at 256, +0.056 [+0.053, +0.060] at 1,024). So the
lesson stands, and the cost to cut next is wrong marks lingering in notebooks.

## Reproduce

    uv run python scripts/notebook_check.py --workers 4 --out docs/research/swarm-scale-20261029/notebook.json

    uv run python scripts/notebook_burst_probe.py --out docs/research/swarm-scale-20261029/notebook_burst_probe.json

About 7 to 14 minutes for the check and 3 for the probe on 4 CPUs. Tests: `uv run --extra test pytest -q tests/smarter_swarm/test_notebook.py`.
