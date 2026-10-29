# Free checks for the smarter-swarm plan (2026-10-29)

All $0, no model call, no network. The probe files are scripted ants on development seeds 100 to 102; the cached
colony file replays model answers saved by the Oct 10 colony run.
Plan and reading of the numbers: `docs/plans/2026-10-29-swarm-at-scale.md`, section 4.

| file | command |
|---|---|
| `probe_r8.json` | `uv run python scripts/swarm_scale_probe.py --ns 8,64,512,2048 --seeds 100,101,102 --rounds 8 --out docs/research/swarm-scale-20261029/probe_r8.json` |
| `probe_r32.json` | `uv run python scripts/swarm_scale_probe.py --ns 64,512 --seeds 100,101,102 --rounds 32 --out docs/research/swarm-scale-20261029/probe_r32.json` |
| `cached_colony.json` | `uv run python scripts/cached_colony_check.py <cells dir> --out docs/research/swarm-scale-20261029/cached_colony.json` (how to fill the cells dir: the script's docstring) |

The probe files hold a `summary` (one entry per N and liar share) and every `row` (one per N, liar share, graph family,
seed and policy). Timings (`s_per_agent_decision`) were measured on a shared cloud container and are indicative only.
Tests: `uv run --extra test pytest -q tests/swarm_scale`.
