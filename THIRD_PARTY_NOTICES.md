# Third-party notices

Antelligence v4.2. Last checked: 2026-10-14.

Scope: `docs/v4.2/`, `docs/research/v4.2-20261014/`, `antelligence/kernel/colony/notebook.py` (docstring), and `docs/v4.2/INTERFACES.md`.

Rule: ideas can be borrowed with credit. Code is not copied unless this file says so.

## Borrowed work

| # | Source | Licence | What was taken | Where it shows up | Status |
|---|---|---|---|---|---|
| 1 | Hermes Agent (NousResearch), `agent/curator.py` and `tools/memory_tool_store.py` | MIT (as stated in the notebook docstring and the brief) | Design ideas only for the per-ant notebook: a character cap, archive-never-delete curation, and frozen reads. No code copied. | `antelligence/kernel/colony/notebook.py`; `docs/v4.2/INTERFACES.md` section 9 | Licence unverified. The upstream LICENSE file was not read in this session (access to the local Hermes checkout was denied). |
| 2 | agentskills.io lesson format | Unverified | The export shape only: `name` up to 64 characters, `description` up to 1,024 characters, and a body. No code copied. | `docs/v4.2/INTERFACES.md` section 9 (export format) | Licence unverified. |
| 3 | Medical Segmentation Decathlon Task01 brain tumour data (case ids `BRATS_xxx`) | CC BY-SA 4.0 (per the goal's references: registry.opendata.aws/msd; not independently re-read in this pass) | Case images are the input data for the runs. The SimLook cell table `cells_v4_2.json` holds counts and quantiles per cell, not MRI voxels. Run bundles carry no MRI arrays. No MRI data is committed. | Run outputs under `runs/` (not in git); `docs/research/v4.2-20261014/simlook/` | Licence taken from the goal file; lead to confirm before release. |

## Not used

- CoELA, `cwah/` directory: never copied. No `cwah/` directory and no CoELA text appear in this worktree (grep found no matches). No design ideas are recorded from it in the v4.2 docs.

## Not checked in this pass

- Other `docs/research/` folders (`desci-paper-20260915`, `fugu-20260910`, `hive-20260910`, `sol-luna-features-20260910`) were not searched for borrowed sources. Some of them have source manifests. Those need their own pass.
