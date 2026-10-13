import pytest

from antelligence.kernel.policies.v41_harness import (
    MAX_YAML_BYTES, HarnessError, _template_ok, load_harness_doc, load_yaml_text,
)

F = {"n_regions", "n_squads"}


@pytest.mark.parametrize("text", [
    "{n_regions:>{n_squads}}", "{n_regions:5}", "{n_regions!r}", "{n_regions!s:>4}",
    "{n_regions.real}", "{n_regions[0]}", "{n_regions.__class__}", "{0}", "{}", "{other}", "{n_regions", "n_regions}", "{{}", "",
    "x" * 300,
])
def test_template_rejects(text):
    assert not _template_ok(text, F)


def test_template_accepts_plain_fields_and_escaped_braces():
    assert _template_ok("Cover {n_regions} regions with {n_squads} squads. {{literal}}", F)
    assert _template_ok("no fields at all", F)


def test_yaml_size_cap_and_aliases(tmp_path):
    with pytest.raises(HarnessError, match="over"):
        load_yaml_text("a: " + "x" * MAX_YAML_BYTES)
    big = tmp_path / "h.yaml"
    big.write_text("# " + "x" * MAX_YAML_BYTES + "\nharness: {}\nallowed_keys: []\n")
    with pytest.raises(HarnessError, match="over"):
        load_harness_doc(big)
    with pytest.raises(HarnessError, match="anchor|alias"):
        load_yaml_text("a: &x [1, 2]\nb: *x\n")
    with pytest.raises(HarnessError, match="anchor"):
        load_yaml_text("a: &x 1\n")
    assert load_yaml_text("a: [1, 2]\n") == {"a": [1, 2]}
