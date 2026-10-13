import hashlib
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_cell(arm, case, seed, score, split="eval", **kw):
    run_id = hashlib.sha256(f"{arm}|{case}|{seed}".encode()).hexdigest()
    cell = dict(run_id=run_id, arm=arm, case=case, seed=seed, split=split, wall_s=100.0,
                ticks_to_50=20, seconds_to_50=40.0, queen_calls=5, board_size=60, anchors=2,
                visibility_latency_ms=None, coverage=score, off_target=0.1, defer_rate=0.2, citations=7, tokens_in=1000, tokens_out=100, pod_minutes=1.0, gas=None, alarms=0,
                audit_conformal_coverage=0.9, citation_validity=1.0, uncited_decisions=0, perception_dice=0.8,
                locallook_ece=0.05, truth_outside_crop=0.02,
                unread_frac=0.2, brief_age_median=10.0, router_top_share=0.5, perception_ece=0.05, cost_usd=0.0,
                backends_kind="live", router_kind="frozen", router_sha256="a" * 64, identities="d1,qwen3.5-9b")
    cell.update(kw)
    return cell


def synth(n_cases=30, seeds=2, effect=0.05, noise=0.03, split="eval", rng_seed=1, arms=None, base=0.5):
    """SW-B = SW-noBoard + effect (+noise); FW = SW-B + 0.01; S low, S-prime high."""
    rng = random.Random(rng_seed)
    cells = []
    for i in range(n_cases):
        case = f"BRATS_{split}_{i:03d}"
        b = base + rng.uniform(-0.1, 0.1)
        for seed in range(seeds):
            vals = {"SW-noBoard": b, "SW-B": b + effect + rng.gauss(0, noise), "FW": b + effect + 0.01 + rng.gauss(0, noise),
                    "S": b - 0.2, "S-prime": b + 0.2, "SW-D": b + effect + rng.gauss(0, noise)}
            for arm, v in vals.items():
                if arms is None or arm in arms:
                    cells.append(make_cell(arm, case, seed, v, split))
    return cells
