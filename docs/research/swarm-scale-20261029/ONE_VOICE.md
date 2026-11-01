# One source, one voice (dream run idea, 2026-11-01)

The nightly dream run built and tested this on 2026-11-01 but could not push. This is a rebuild from that run's own
record, on the same base (draft PR 24). The fair check was run again from scratch and every number below came out the
same. A second probe, on other seeds and with a burst from many authors, was added after.

**Problem.** The field notebook (draft PR 24, the current best) remembers each source once, but at the strength it was
read. A wrong mark planted as strong as eight ants is then remembered as eight ants' word for most of the run, so a
burst of misinformation does about four times the harm it did without a notebook.

**Idea.** One source, one voice. A remembered source counts for at most what one honest ant can ever say: the waggle
cap (3) under graded marks, 1 otherwise. Like one person, one vote, or clipping outliers in robust statistics. Marks
read at the current site are unchanged; only the notebook clips. Honest ants and liars never go over the cap, so with
no planted mark nothing changes at all. Option `notebook_one_voice` on `RumourConfig`, off by default and
bit-identical when off (tested).

Also tried, same check: **seen twice to stick** (`notebook_consolidate`). A source seen writing only once fades fast;
seen writing at two different ticks, it fades slowly, like short-term memory becoming long-term. Planted marks are
written once, so they never stick.

## The fair test

Rumour world, scripted, $0, no model calls, fixed fade, graded cap 3, notebook on. Every arm, the current best
included, had its listening weight, notebook half-life and (where used) short half-life tuned on development seeds 0
to 9 at 256 ants (mean right over none, burst, liars). Then each ran on fresh seeds 400 to 419 (256 ants) and 400 to
409 (1,024), paired by seed. The current best re-tuned to the same picks as PR 24 (weight 3, half-life 80).

Picks: best and one voice both weight 3, half-life 80 (no edge). Seen twice: weight 2, half-life 80, short half-life
20, which is the top of its grid, so the tuning wanted it closer to "off".

Share of honest ants right / extra honest ants wrong over the run (damage):

| ants | condition | best | one voice | seen twice | both |
|---|---|---|---|---|---|
| 256 | none | 0.998 / 0 | 0.998 / 0 | 0.998 / 0 | 0.998 / 0 |
| 256 | burst | 0.996 / 0.056 | 0.997 / **0.044** | 0.997 / 0.047 | 0.997 / **0.038** |
| 256 | liars | 0.992 / 0.043 | 0.992 / 0.043 | 0.991 / 0.042 | 0.991 / 0.042 |
| 1,024 | none | 0.996 / 0 | 0.996 / 0 | 0.995 / 0 | 0.995 / 0 |
| 1,024 | burst | 0.994 / 0.056 | 0.995 / **0.044** | 0.995 / 0.046 | 0.995 / **0.038** |
| 1,024 | liars | 0.988 / 0.050 | 0.988 / 0.050 | 0.987 / 0.048 | 0.987 / 0.048 |

One voice minus best (95% bootstrap CI over seeds):

- Burst damage: -0.0118 [-0.0132, -0.0106] at 256 (lower on 20 of 20 seeds), -0.0118 [-0.0128, -0.0108] at 1,024
  (10 of 10). About a fifth of the burst harm gone.
- Burst right: +0.0004 [+0.0000, +0.0010] and +0.0004 [+0.0001, +0.0008]. Never lower on any seed.
- None and liars: exactly equal (as designed). Planted: damage -0.0013 and -0.0004, right +0.0002.

Both (one voice + seen twice) minus best: burst damage -0.018 (256 and 1,024, every seed lower), but right drops a
little with no seed at 1,024 (-0.0008 [-0.0014, -0.0002]) and under liars (-0.001, CI crosses 0).

## Second probe: other seeds, and a burst from many authors

`scripts/notebook_burst_probe.py --one-voice --baseline notebook_burst_probe.json` re-runs PR 24's tuned notebook arms
(same picks: weight 3, half-life 80 for graded) with one voice on, on PR 24's test seeds (300 to 319 and 300 to 309,
which this check never touched), under a burst whose copies share one author and one whose copies each have their own,
paired by seed against the same arms without one voice.

| ants | burst authors | burst harm, notebook | with one voice | difference (95% CI) | seeds lower |
|---|---|---|---|---|---|
| 256 | one | 0.058 | **0.046** | -0.012 [-0.013, -0.011] | 20 / 20 |
| 256 | many | 0.068 | **0.053** | -0.015 [-0.017, -0.014] | 20 / 20 |
| 1,024 | one | 0.059 | **0.047** | -0.012 [-0.013, -0.011] | 10 / 10 |
| 1,024 | many | 0.070 | **0.054** | -0.016 [-0.017, -0.015] | 10 / 10 |

Share right is never lower on any seed, and a hair higher (+0.001 to +0.002). With plain think alone and a notebook,
where one voice clips to a single reading, the burst harm drops much more: by 0.047 with one author and 0.060 with many.



- One voice is a free win: same answers everywhere, a fifth less harm from a burst, no new knob to tune. It holds on
  other seeds and does a little more when a burst comes from many authors.
- Seen twice cuts more harm but costs a little accuracy, and the tuning pushed it toward off. Not worth it yet.
- What is left of the burst harm comes from the early ticks, before ants have much evidence of their own.
- Scripted world only. In a real swarm, "one ant's strongest mark" needs a cap every honest agent shares.

## Reproduce

    uv run python scripts/one_voice_check.py --workers 4 --out docs/research/swarm-scale-20261029/one_voice.json

    uv run python scripts/notebook_burst_probe.py --one-voice \
        --baseline docs/research/swarm-scale-20261029/notebook_burst_probe.json \
        --out docs/research/swarm-scale-20261029/one_voice_burst_probe.json

About 15 to 27 minutes for the check and 3 for the probe on 4 CPUs. Tests: `uv run --extra test pytest -q tests/smarter_swarm/test_notebook_fixes.py`.
