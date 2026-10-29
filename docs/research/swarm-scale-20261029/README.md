# Swarm-at-scale probe (2026-10-29)

Scripted, $0, development seeds 100 to 102 only. No model call, no network.
Plan and reading of the numbers: `docs/plans/2026-10-29-swarm-at-scale.md`, section 5.

| file | command |
|---|---|
| `probe_r8.json` | `uv run python scripts/swarm_scale_probe.py --ns 8,64,512,2048 --seeds 100,101,102 --rounds 8 --out docs/research/swarm-scale-20261029/probe_r8.json` |
| `probe_r32.json` | `uv run python scripts/swarm_scale_probe.py --ns 64,512 --seeds 100,101,102 --rounds 32 --out docs/research/swarm-scale-20261029/probe_r32.json` |

Each file holds a `summary` (one entry per N and liar share) and every `row` (one per N, liar share, graph family,
seed and policy). Timings (`s_per_agent_decision`) were measured on a shared cloud container and are indicative only.
Tests: `uv run --extra test pytest -q tests/swarm_scale`.
