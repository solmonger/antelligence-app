# Dreams

A log of the nightly dream run: five ideas a night, one built and tested for free against the current best.
Verdicts: beats best, no better, broken.

## 2026-10-30

Ideas:
1. Waggle-dance marks: ants with more of their own evidence leave stronger marks, capped (honeybee dance length).
2. Surprisingly popular: each ant also guesses what others will say; trust the answer more popular than predicted.
3. Lateral inhibition: a mark weakens nearby marks that say the same thing, so one crowd can't stack up.
4. Annealed listening: ants listen little early and more later, like cooling metal.
5. Reputation staking: ants stake credibility on marks and earn it back when later readers agree.

Picked: 1, the simplest to test and closest to think alone.

Built: rumour world policy `graded` (mark strength = net private readings, at least 1, cap 3; liars shout at the cap)
and a fair check that tunes every arm's listening weight on seeds 0 to 9 and scores it on fresh seeds.
Branch claude/dream-2026-10-30, commit 2e437b8. Write-up: docs/research/swarm-scale-20261029/GRADED_MARKS.md.

Note: the night's own run could not push (the fork was not in its session's repos), so its code was lost. This entry
is from the rebuild the same morning, re-tested from scratch on new fresh seeds (100 to 119 at 256 ants, 100 to 109
at 1,024). The rebuild picked the same weights and its gains are within about half a point of the night's.

Numbers (share of honest ants right, fixed fade, graded vs think alone tuned to its best loudness):
- No seed: +0.018 (256 ants), +0.022 (1,024). Burst: same. Liars: +0.010 (256), +0.015 (1,024). Every 95% CI above 0.
- Against think alone as it stands in PR 21 (weight 1): about +0.06.
- Cost: liars at cap 3 add +0.014 damage. Cap 2 keeps +0.015 of the gain at +0.006 damage.

Verdict: beats best (about 2 points over a fairly tuned think alone).

Next: port to colouring and foraging; decide cap 2 or 3; then the fair test with models.
