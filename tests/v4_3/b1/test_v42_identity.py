"""B1: with the foraging model OFF, v4.2 behaviour and the gate-A config are byte-identical to the base commit (64b678d).

Goldens were computed at the base commit, before any B1 change: phantom with synthetic TC / ET channels (``helpers.channels``),
120 ticks, the SimLook stand-in ``noisy_look_factory``, seeds as named. A digest is sha256 of the bundle body, the cards, the
script counters, the board entry count and the queen's alarms (``helpers.record_digest``)."""

import pytest

from antelligence.experiments.v42_harness import V42Harness
from antelligence.experiments.v43_harness import V43Harness, bundle_of as bundle43
from tests.v4_2.q1.helpers import noisy_look_factory
from tests.v4_3.a2.helpers import R43, estimate
from tests.v4_3.b1.helpers import digest, load_yaml_configs, phantom_with_channels, record_digest

V42 = {
    "cfg": ["d8bf8b6f87443087fd1f7faee46c4e541710047a8fcd7b9097f969c28b01c272", "7e934fa67bf977cc84a03359f5c93356717bef5daf94433b96839741714fe44a"],
    ("S", 1): "6e52e11e4590b7b076628d698eeb0020c6b16b9aaa661e208915e004477384a3",
    ("S", 2): "a8ec25a1e14012d659b596030e3cf061656c4af7626aa623851ea8446dccc012",
    ("S-solo", 1): "fe7c521e53a68e5fc24368d87b3a2854025ab6b0f420a96d7cdff458f494955b",
    ("S-solo", 2): "5970be88b748f1b0e96de3c1605d901e2acfd45aa89204ae75f6a588a17cff9c",
    ("S-marks", 1): "f0f1c99fad798220eb0d47c1a582cabcfafbcd0f4781633a27664be1f864d58c",
    ("S-marks", 2): "8db469ac599b4a7c224ecf05099f223e9a3ae1b5168a777c288f672692455084",
    ("S-talk", 1): "87ba59d7a77a9c509560302f6a429556dd7c28abe848de68339eefa47e45b1c9",
    ("S-talk", 2): "3ced78bc06ecb058fd4f2165d67c8d647ce70170873344b5857eb653d568c097",
}
GATE_A = {
    "cfg": ["5d0a12d87a2772595e5448583a8315a5159b68f03fddf0cfca597ab069648c25", "7e934fa67bf977cc84a03359f5c93356717bef5daf94433b96839741714fe44a"],
    "v42-harness": {"S": "7fdc8cbda3e3b8b84b682d5413fd8f0a3fb3930b94c7cad6078c928fd3eac893",
                    "S-solo": "c4ecf2c198d7981c4be0f8bb3518edb4cc963456b2c00294b8b2d7b85b0dc5b2"},
    "v43-harness": {"S": "625dbcfe8fc86511b57bc2db4dae736c7ffddb1d961a4adbf16d8306aa6b420e",
                    "S-solo": "a9eef694cc17a5097341b074b5b20252d8300f0153cf50e73bc05cb8c814c0cd"},
}


def test_v42_config_dicts_unchanged():
    w, c, _ = load_yaml_configs("config/v4_2/world.yaml", 120)
    assert [digest(w.to_dict()), digest(c.to_dict())] == V42["cfg"]       # board header hashes follow from these dicts
    assert "f2_model" not in w.to_dict()["facts"] and "uptake_level" not in w.to_dict()["facts"]
    assert "good_uptake" not in c.to_dict()["half_life"]


@pytest.mark.parametrize("arm", ("S", "S-solo", "S-marks", "S-talk"))
@pytest.mark.parametrize("seed", (1, 2))
def test_v42_config_records_identical_to_base(arm, seed):
    w, c, _ = load_yaml_configs("config/v4_2/world.yaml", 120)
    rec = V42Harness(w, c, look_fn_factory=noisy_look_factory).run_episode(arm, phantom_with_channels(), seed)
    assert record_digest(rec) == V42[(arm, seed)]


def test_gate_a_config_dicts_unchanged():
    w, c, _ = load_yaml_configs("config/v4_3/world_gateA.yaml", 120)
    assert [digest(w.to_dict()), digest(c.to_dict())] == GATE_A["cfg"]


@pytest.mark.parametrize("arm", ("S", "S-solo"))
def test_gate_a_s_and_s_solo_identical_to_base(arm):
    w, c, _ = load_yaml_configs("config/v4_3/world_gateA.yaml", 120)
    case = phantom_with_channels()
    rec = V42Harness(w, c, look_fn_factory=noisy_look_factory).run_episode(arm, case, 1)
    assert record_digest(rec) == GATE_A["v42-harness"][arm]
    rec = V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory).run_episode(arm, case, 1)
    assert digest({"b": bundle43(rec), "cards": rec.get("cards_final"), "ctr": rec.get("script_counters")}) == GATE_A["v43-harness"][arm]
