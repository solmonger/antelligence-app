import importlib.util
from pathlib import Path
import pytest

def load():
    spec = importlib.util.spec_from_file_location('fit_pilot', Path(__file__).with_name('fit_pilot.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_legacy_fixture_roundtrip():
    m = load()
    from backend.research_coldroom import make_task, worker_views, plan_from_rules
    task = make_task(100, 1)
    extended = m.extend(task)
    from backend.research_hive_tasks import project_task, compose_canonical_bytes
    wire = project_task(extended, adapter_key=m.KEY)['wire']
    decoded = m.decode(task, compose_canonical_bytes(wire))
    # Legacy orders insertion IDs; the new composer sorts opaque IDs. With no
    # prerequisites, order may differ but placements and safe completion may not.
    from backend.research_coldroom import replay
    baseline = replay(task, plan_from_rules(worker_views(task), task['rules']))
    actual = replay(task, decoded)
    assert actual['safe_success'] is True
    assert actual['placed'] == baseline['placed']
    assert len(decoded) == baseline['submitted_actions']

@pytest.mark.parametrize('condition', ['warm','cold','changed','changed_no_reply'])
def test_all_arms_replay_same_task(tmp_path, condition):
    m = load()
    rows = m.run_case(100, 1, condition, tmp_path)
    assert set(r['arm'] for r in rows) == set(m.ARMS)
    assert len({r['task_digest'] for r in rows}) == 1
    assert all(r['error'] is None for r in rows)
    assert all(r['new_verifier_agrees'] is not False for r in rows)
    assert all(not r['actual_state_violation'] for r in rows)
    assert next(r for r in rows if r['arm']=='legacy_full')['safe_success']
    assert next(r for r in rows if r['arm']=='new_full')['safe_success']
    combined = next(r for r in rows if r['arm']=='memory_plus_exchange')
    assert combined['safe_success'] is (condition != 'changed_no_reply')
    assert combined['submitted_actions'] == (0 if condition == 'changed_no_reply' else 6)
    assert combined['transport_events'] == {'warm':0,'cold':2,'changed':2,'changed_no_reply':1}[condition]
