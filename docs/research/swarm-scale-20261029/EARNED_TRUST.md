# Earned trust in the finish world: tested, no gain (2026-10-30)

**The idea (plan idea 1, and the next step from section 0.2).** When re-checking a "done" claim is slow (3 ticks),
looking at every claim doubles the time and still leaves work behind, and spot looking (one claim in four) does
better. The plan's next step was to look more often at claims from ants with a poor record. Two versions, both with
no checker and no teacher:

- **Private record** (`record`): when an ant re-checks a piece marked done and finds work left, it remembers every ant
  whose "done" was on it, and from then on always re-checks their claims. Everyone else is spot-checked one time in
  four.
- **Word of mouth** (`gossip`): the same, plus ants standing on the same site pool the names they caught (peers
  monitoring each other, as in Ostrom's commons). A passed-on name is a claim too, so this only stays safe while nobody
  lies about others; the shortcut-takers here do not.

Code: `record` and `gossip` arms in `antelligence/worlds/finish/world.py`. With full looking, or with no
shortcut-takers, both play exactly like `look` (tested); the `look` arm's recorded runs regenerate exactly (tested).

## The fair test

Fresh seeds 100 to 129 (the earlier check used 0 to 9), 256 and 1,024 ants, 10% and 20% shortcut-takers, patience 10
and 20 idle ticks, re-check cost 3, spot rate 1 in 4. Paired by seed, bootstrap 95% intervals. Score: share of the
job's work still left when the swarm said it was done (lower is better).

Work left at "done", 20% shortcut-takers:

| ants | patience | look at every claim | spot looking (now) | private record | word of mouth |
|---|---|---|---|---|---|
| 256 | 10 | 1.08% (144 ticks) | 0.44% (106) | 0.44% (107) | 0.53% (108) |
| 256 | 20 | 0.40% (181) | 0.17% (133) | 0.18% (135) | 0.18% (139) |
| 1,024 | 10 | 0.89% (148) | 0.41% (109) | 0.42% (109) | 0.43% (115) |
| 1,024 | 20 | 0.36% (183) | 0.18% (135) | **0.16%** (136) | 0.17% (143) |

With 10% shortcut-takers the picture is the same (`earned_trust.json`).

- **Neither version helps.** Private record against spot looking: no clear difference in 7 of 8 cells (every interval
  crosses 0). The one exception is 1,024 ants, 20% shortcut-takers, patience 20: -0.020 points [-0.036, -0.005], too
  small to matter. Word of mouth: no gain anywhere, and 1 to 8 ticks slower in most cells.
- **Why the private record is too thin:** an ant rarely meets the same shortcut-taker twice. At the end each ant has
  caught 0.15 to 0.38 names, out of 26 to 205 shortcut-takers.
- **Why word of mouth does not help either, though names spread:** by the end each ant knows 15 to 124 names (about
  60% of the shortcut-takers), and nobody was ever wrongly named (0 in 960 runs). But most of the work left at "done"
  is not hidden behind lies. It is left by the stop rule: in a quick run on seeds 100 to 109 (256 ants, patience 10),
  an honest swarm with no shortcut-takers and spot looking already left 0.33%, against 0.43% with 20%
  shortcut-takers. Catching liars faster cannot fix what the liars did not cause.
- **What does matter:** looking less, and waiting longer. Spot looking beats looking at every claim in every cell:
  0.18 to 0.63 points less work left, 38 to 57 ticks sooner, on 23 to 30 of 30 seeds. Patience 20 instead of 10 more
  than halves the work left with spot looking, for 21 to 27 more ticks.

**Reading.** Earned trust pays when the same member is met again and again, like the colony runs where every question
went to the same few models (section 4.1). In a large swarm of ants that meet at random, a record of who cheated is
too thin to use, and passing names around spreads them but does not stop the leak. Keep spot looking as the rule; the
lever left in this world is the stop rule, not trust. Not added to the default.

## Reproduce

```
uv run python scripts/earned_trust_check.py --out docs/research/swarm-scale-20261029/earned_trust.json
uv run --extra test pytest -q tests/smarter_swarm/test_earned_trust.py
```
