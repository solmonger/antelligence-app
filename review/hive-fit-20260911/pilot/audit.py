"""Independent readback of every raw outcome; explicit unreported-drift control."""
import sys
import json
import copy
import hashlib
import collections
from pathlib import Path
ROOT=Path(sys.argv[1]).resolve()
LANE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LANE))
from backend import research_coldroom as old
from backend.research_hive_memory import HiveMemoryStore,adapt_memory_query_bytes,consume_memory_result_bytes
from backend.research_hive_contracts import serialize_envelope,_event_id,deserialize_envelope
from backend.research_hive_tasks import project_task,compose_canonical_bytes
from backend.research_hive_verifier import verify_task
from fit_pilot import extend,decode,KEY
from scripts.probe_hive_communication import _canonical_stream
p=ROOT
rows=[json.loads(line) for line in (p/'rows.jsonl').read_text().splitlines()]
m=json.loads((p/'manifest.json').read_text())
expected={(s,r,c,a) for s in m['seeds'] for r in m['revisions'] for c in m['conditions'] for a in m['arms']}
assert {(r['seed'],r['revision'],r['condition'],r['arm']) for r in rows}==expected
assert len(rows)==len(expected)
for row in rows:
    assert row['error'] is None, row
    t=json.loads((p / row['trace_path']).read_text())
    replay=old.replay(t['task'],t['actions'])
    assert replay==t['old_replay']
    assert row['safe_success']==replay['safe_success']
    assert row['submitted_actions']==len(t['actions'])
    assert row['new_verifier_agrees'] is not False
    assert hashlib.sha256(json.dumps(t['task'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==row['task_digest']
# Additional development diagnostic, NOT a new held-out test or part of the
# balanced matrix: change the task without sending any source invalidation.
controls=[]
for seed in m['seeds']:
    for revision in m['revisions']:
        directory=p/f'{seed}-r{revision}-warm'/'new_memory'
        trace=json.loads((directory/'trace.json').read_text())
        task=copy.deepcopy(trace['task'])
        task['rules']={k:(v+1)%3 for k,v in task['rules'].items()}
        projection=project_task(extend(task),adapter_key=KEY)
        incomplete=[w for w in projection['wire'] if deserialize_envelope(w).get('role')!='protocol']
        query={'version':'hive-contract-v1','kind':'memory-query','event_id':'pending',
               'ordinal':0,'scope':projection['public']['scope'],'query':'recall-protocol-role'}
        query['event_id']=_event_id(query)
        recalled=consume_memory_result_bytes(adapt_memory_query_bytes(HiveMemoryStore(directory/'memory.sqlite3'),serialize_envelope('memory-query',query),applicability=task['protocol']))
        assert recalled['found'] is True
        wire=recalled['result']['evidence'][0]['content']['wire'].encode()
        actions=compose_canonical_bytes(_canonical_stream(incomplete+[wire]))
        verified=verify_task(extend(task),actions,adapter_key=KEY)
        legacy=old.replay(task,decode(task,actions))
        controls.append({'seed':seed,'revision':revision,'memory_found':recalled['found'],
                         'safe_success':legacy['safe_success'],'submitted_actions':len(actions),
                         'rejected_attempt':legacy['stop_reason'] is not None,
                         'new_classification':verified['classification'],
                         'state_violation':verified['classification']=='actual state violation',
                         'legacy_replay':legacy})
summary=[]
for condition in m['conditions']:
    for arm in m['arms']:
        g=[r for r in rows if r['condition']==condition and r['arm']==arm]
        summary.append({'condition':condition,'arm':arm,'assigned':len(g),
                        'success':sum(r['safe_success'] for r in g),
                        'rejected_attempts_cases':sum(r['rejected_attempt'] for r in g),
                        'abstained':sum(r['abstained'] is True for r in g),
                        'transport_events':sum(r['transport_events'] for r in g),
                        'transport_bytes':sum(r['transport_bytes'] for r in g)})
result={'rows_verified':len(rows),'conditions':summary,'unreported_drift':{
        'assigned':len(controls),'success':sum(r['safe_success'] for r in controls),
        'rejected_attempts_cases':sum(r['rejected_attempt'] for r in controls),
        'memory_found':sum(r['memory_found'] for r in controls)},
        'limits':['No Rscript found in PATH or standard installation paths; descriptive counts only, calculated in Python. No inferential statistics.',
                  'Not a model study: deterministic planners and trusted evidence sources.',
                  '24 task variants share one small simulator family, not 24 independent real-world problems.',
                  'Sources/caches are seeded by the trusted harness; no learned memory claim.',
                  'Changed-source successes require explicit invalidation plus a correct fresh reply.',
                  'No message/token/latency superiority claim. Memory-plus-exchange adds request/reply overhead.',
                  'No live workbench/tumor integration, no F4/F5/F6 completion or promotion.']}
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'unreported-drift.json').write_text(json.dumps(controls,indent=2)+'\n')
print(json.dumps(result,indent=2))
