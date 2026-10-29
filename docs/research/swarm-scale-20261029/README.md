# Free checks for the smarter-swarm plan (2026-10-29)

All $0, no model call, no network. The probe files are scripted ants on development seeds 100 to 102; the cached
colony file replays model answers saved by the Oct 10 colony run.
Plan and reading of the numbers: `docs/plans/2026-10-29-swarm-at-scale.md` (section 0 for think alone and smart
fading, 0.2 for skipped work, 0.3 for deep thinkers, section 4 for the probe and the cached colony).

| file | command |
|---|---|
| `probe_r8.json` | `uv run python scripts/swarm_scale_probe.py --ns 8,64,512,2048 --seeds 100,101,102 --rounds 8 --out docs/research/swarm-scale-20261029/probe_r8.json` |
| `probe_r32.json` | `uv run python scripts/swarm_scale_probe.py --ns 64,512 --seeds 100,101,102 --rounds 32 --out docs/research/swarm-scale-20261029/probe_r32.json` |
| `cached_colony.json` | `uv run python scripts/cached_colony_check.py <cells dir> --out docs/research/swarm-scale-20261029/cached_colony.json` (how to fill the cells dir: the script's docstring) |
| `fading_alone_n{64,256,1024,2048}.json` | `uv run python scripts/fading_alone_check.py --ns 256 --out docs/research/swarm-scale-20261029/fading_alone_n256.json` (one N per file; seeds 0 to 9, half-life 10) |
| `fading_alone_n256_hl5.json`, `_hl20.json` | same, with `--half-life 5` or `--half-life 20` |
| `fading_alone_n256_acc55.json` | same, with `--accuracy 0.55` |
| `alone_vs_seen.json` | `uv run python scripts/alone_vs_seen_check.py <cells dir> --out docs/research/swarm-scale-20261029/alone_vs_seen.json` (cells: the saved W2 `cells-*.jsonl`, see the script's docstring) |
| `finish_n{64,256,1024}.json` | `uv run python scripts/finish_check.py --ns 256 --out docs/research/swarm-scale-20261029/finish_n256.json` (one N per file; seeds 0 to 9; 0, 5, 10 and 20% shortcut-takers) |
| `finish_n256_p20.json` | same, with `--patience 20` |
| `finish_n256_look3.json`, `finish_n256_look3_spot25.json` | same, with `--look-cost 3`, and with `--look-cost 3 --look-p 0.25` |
| `deep_on_split.json` | `uv run python scripts/deep_on_split_check.py <cells dir> --out docs/research/swarm-scale-20261029/deep_on_split.json` (same cells as `cached_colony.json`) |

The probe files hold a `summary` (one entry per N and liar share) and every `row` (one per N, liar share, graph family,
seed and policy). Timings (`s_per_agent_decision`) were measured on a shared cloud container and are indicative only.
The `fading_alone_*` files hold a `summary` (cells and paired comparisons with bootstrap intervals) and every `row`
(one per N, seed, fade rule, policy and what was planted).
Tests: `uv run --extra test pytest -q tests/swarm_scale tests/smarter_swarm`.
