# Waggle-dance marks (dream run idea, 2026-10-30)

**Idea.** Honeybee scouts dance longer for a better site. Here a think-alone ant leaves a stronger mark when it has
more of its own evidence: the mark's strength is its net count of private readings, at least 1 and at most a cap.
Everything else is think alone. Liars shout at the cap. No checker. With a cap of 1 it is exactly think alone (tested).

The nightly dream run built and tested this on 2026-10-30 but could not push, so the code was lost. This is a clean
rebuild, re-tested from scratch on new fresh seeds. Rumour world, scripted, $0, no model calls.

## The fair test

A stronger mark is also a louder mark, so think alone gets to be loud too. Each of the four arms (fixed or smart fade,
think alone or graded) had its listening weight tuned on development seeds 0 to 9 at 256 ants. Then every tuned arm
ran on fresh seeds 100 to 119 (256 ants) and 100 to 109 (1,024 ants). Same worlds, same evidence, paired by seed.

Share of honest ants right at the end (cap 3; the best think-alone arm is fixed fade at weight 4, graded at weight 2):

| ants | condition | think alone, as in PR 21 (weight 1) | think alone, tuned | graded, tuned | graded minus tuned alone (95% CI) | seeds graded higher / lower |
|---|---|---|---|---|---|---|
| 256 | none | 0.841 | 0.882 | **0.900** | +0.018 [+0.012, +0.024] | 18 / 0 |
| 256 | burst | 0.840 | 0.882 | **0.900** | +0.018 [+0.013, +0.024] | 19 / 0 |
| 256 | liars | 0.828 | 0.867 | **0.877** | +0.010 [+0.003, +0.019] | 13 / 4 |
| 1,024 | none | 0.845 | 0.887 | **0.909** | +0.022 [+0.015, +0.030] | 10 / 0 |
| 1,024 | burst | 0.844 | 0.887 | **0.909** | +0.022 [+0.015, +0.030] | 10 / 0 |
| 1,024 | liars | 0.833 | 0.868 | **0.883** | +0.015 [+0.009, +0.022] | 10 / 0 |

Under smart fading the picture is the same: graded beats tuned smart think alone by +0.015 to +0.019 (no seed) and
+0.011 to +0.014 (liars), every interval above 0. Fixed fade stays the better of the two for "right".

## The cost: liars

Damage = how many more honest ants are wrong, averaged over the run, than with nothing planted.

| ants | cap | liar damage, tuned alone | liar damage, graded | right under liars, graded minus tuned alone |
|---|---|---|---|---|
| 256 | 3 | 0.024 | 0.038 (+0.014) | +0.010 |
| 256 | 2 | 0.024 | 0.030 (+0.006) | +0.011 |
| 1,024 | 3 | 0.022 | 0.036 (+0.014) | +0.015 |

Liars who shout at the cap do more harm, and the cap sets how much. Cap 2 keeps most of the gain (+0.015 with no seed,
+0.011 under liars) at under half the extra liar harm. Under a burst of planted wrong marks, graded does slightly *less*
harm than tuned alone (about -0.003).

## What this means

- It beats the current best, fairly, at both sizes: about 2 points more ants right, on seeds the tuning never saw.
- Turning think alone up from weight 1 to its best is itself worth about 4 points. That is free and worth taking
  whatever happens to graded marks.
- Cap 2 looks like the safer default if liars are a real worry.
- It does not carry over to colouring or foraging (below). Next: the fair test with models, where "how sure am I"
  would come from the model's own stated confidence.

## Ported to colouring and foraging (cap 2): no gain

The same idea in the two worlds we did not design, against think alone in the same world, on fresh seeds 200 and up
(`graded_ports.json`, `scripts/graded_ports_check.py`).

- **Colouring** (AgentsNet): each colour announcement says how sure the sender is (1 plus its clean looks in a row,
  cap 2); on a clash the less sure ant moves. 20 seeds x 2 graph families, 256 and 1,024 agents, 3 and 4 rounds
  (at 8 rounds think alone is already at the ceiling). No seed condition improves: base is identical, lost messages
  is within -0.0010 to +0.0003, and **liars make it slightly worse** (-0.0003 to -0.0018 of edges coloured, every
  interval at or below 0; liars plus lost messages at 1,024 agents and 4 rounds too, -0.0014), because liars claim
  the cap and unsure honest ants give way to them.
- **Foraging** (SwarmBench): first-hand "food is north of me" is strength 2, a relayed memory strength 1, and the
  graded forager follows the surest lead. 80 seeds. Deliveries are unchanged everywhere (relayed talk under v1:
  17.00 vs 17.07, interval -0.375 to +0.212). It does halve the dead-end leads it chases (9.3 to 4.9 per run), but that
  does not turn into more food; and when relayed memories shout at the cap, it is exactly think alone again.

**Reading.** Think alone leaves almost no room in these two worlds (colouring is at 98.3 to 99.98% of edges after
only 3 or 4 rounds; foraging look ties the pilot's best script). Waggle-dance marks help where many weak,
noisy readings are pooled (the rumour world), not where one first-hand look settles the question. They should stay a
rumour-world (and model-confidence) idea, not a default for every world.

## Reproduce

    uv run python scripts/graded_marks_check.py --out docs/research/swarm-scale-20261029/graded_marks.json
    uv run python scripts/graded_marks_check.py --cap 2 --test 256:100-119 --out docs/research/swarm-scale-20261029/graded_marks_cap2.json
    uv run python scripts/graded_ports_check.py --out docs/research/swarm-scale-20261029/graded_ports.json
