"""Development-only bridge/ablation, not model accuracy or F6 completion.

Run from the frozen F3 lane with `python -m pytest ...` or the supplied runner.
Legacy task fields/rules are preserved. New scheduler fields are neutral defaults.
Freshness invalidation is externally supplied, NOT discovered by the memory.
"""
import copy
import hashlib
import hmac
import json
import time
from pathlib import Path
from backend import research_coldroom as old
from backend.research_hive_contracts import serialize_envelope, deserialize_envelope, _event_id
from backend.research_hive_tasks import project_task, compose_canonical_bytes
from backend.research_hive_verifier import verify_task
from backend.research_hive_memory import HiveMemoryStore, adapt_memory_query_bytes, consume_memory_result_bytes
from backend.research_hive_communication import LocalEvidenceTransport, build_request, build_reply
from scripts.probe_hive_communication import _canonical_stream, _evidence

KEY = b'fit-pilot-development-only-key'
ARMS = ['legacy_full','legacy_memory','new_full','new_memory','memory_plus_exchange']
CONDITIONS = ['warm','cold','changed','changed_no_reply']

def extend(task):
    return {**copy.deepcopy(task),
            'prerequisites': {s: [] for s in task['sample_kinds']},
            'resources': {'bench': len(task['sample_kinds'])},
            'requirements': {s: 'bench' for s in task['sample_kinds']}}

def alias(prefix, name):
    return prefix + '-' + hmac.new(KEY, name.encode(), hashlib.sha256).hexdigest()[:16]

def decode(task, actions):
    samples = {alias('sample', s): s for s in task['sample_kinds']}
    slots = {alias('slot', s): s for s in task['slot_zones']}
    return [{**a, 'sample': samples[a['sample']], 'slot': slots[a['slot']]} for a in actions]

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))

def seed_memory(path, prior, protocol_wire):
    store = HiveMemoryStore(path)
    metadata = {'source_id':'protocol-source','source_revision':'v1',
                'applicability':prior['protocol'],'protocol':prior['protocol'],
                'revision':prior['revision'],'dependencies':[]}
    fact = {'kind':'evidence_claim','content':{'wire':protocol_wire.decode()},**metadata}
    fact_id = store.insert_candidate(fact)
    proc = {**metadata,'source_id':'recall-procedure','kind':'conditional_procedure',
            'content':{'purpose':'recall-protocol-role','wire':protocol_wire.decode()},
            'dependencies':[fact_id],'evidence_refs':[fact_id]}
    proc_id = store.insert_candidate(proc)
    store.admit_conditional_procedure(proc_id,evaluator_id='fit-pilot-trusted-seeder',evaluator_revision='v1')
    return fact, proc_id

def run_case(seed, revision, condition, root):
    root = Path(root)
    prior = old.make_task(seed, revision)
    task = copy.deepcopy(prior)
    changed = condition in ('changed','changed_no_reply')
    if changed:
        # Explicit development fault injection: source changes but coarse protocol
        # revision does not. Both old and new arms see the SAME changed task.
        task['rules'] = {k:(v+1)%3 for k,v in task['rules'].items()}
    extended = extend(task)
    projection = project_task(extended, adapter_key=KEY)
    protocol = next(w for w in projection['wire'] if deserialize_envelope(w).get('role')=='protocol')
    incomplete = [w for w in projection['wire'] if w != protocol]
    prior_projection = project_task(extend(prior), adapter_key=KEY)
    prior_protocol = next(w for w in prior_projection['wire'] if deserialize_envelope(w).get('role')=='protocol')
    scope = projection['public']['scope']
    rows=[]
    for arm in ARMS:
        out=root/arm
        out.mkdir(parents=True,exist_ok=False)
        started=time.perf_counter_ns()
        row={'seed':seed,'revision':revision,'condition':condition,'arm':arm,
             'task_digest':hashlib.sha256(canonical_json(task).encode()).hexdigest(),
             'error':None,'planner_calls':0,'memory_found':None,'transport_events':0,
             'transport_bytes':0,'new_verifier_agrees':None,'source_invalidation_supplied':False,
             'model_calls':0,'model_tokens':0,'api_cost_usd':0}
        actions=[]; new_actions=None; trace={}
        try:
            if arm=='legacy_full':
                row['planner_calls']=1
                actions=old.plan_from_rules(old.worker_views(task),task['rules'])
            elif arm=='legacy_memory':
                path=out/'legacy.sqlite3'
                if condition!='cold':
                    old.remember(path,prior,old.plan_from_rules(old.worker_views(prior),prior['rules']))
                result=old.replay_with_memory(task,path)
                row['memory_found']=result['memory'] is not None
                row['planner_calls']=int(row['memory_found'])
                if row['memory_found']:
                    actions=old.plan_from_rules(old.worker_views(task),result['memory']['rules'])
                trace['legacy_result']=result
            else:
                def planner(wire):
                    row['planner_calls']+=1
                    return compose_canonical_bytes(_canonical_stream(wire))
                if arm=='new_full':
                    new_actions=planner(projection['wire'])
                else:
                    path=out/'memory.sqlite3'
                    if condition!='cold':
                        fact,proc=seed_memory(path,prior,prior_protocol)
                        if changed:
                            HiveMemoryStore(path).insert_candidate({**fact,'source_revision':'v2','content':{'claim':'source superseded'}})
                            row['source_invalidation_supplied']=True
                    else:
                        HiveMemoryStore(path)
                    query={'version':'hive-contract-v1','kind':'memory-query','event_id':'pending',
                           'ordinal':0,'scope':scope,'query':'recall-protocol-role'}
                    query['event_id']=_event_id(query)
                    recalled=consume_memory_result_bytes(adapt_memory_query_bytes(HiveMemoryStore(path),serialize_envelope('memory-query',query),applicability=task['protocol']))
                    row['memory_found']=recalled['found']
                    trace['recall']=recalled
                    new_actions=[]
                    if recalled['found']:
                        recovered=recalled['result']['evidence'][0]['content']['wire'].encode()
                        new_actions=planner(incomplete+[recovered])
                    elif arm=='memory_plus_exchange':
                        source={'source_id':'protocol-source','revision':'v2' if changed else 'v1',
                                'applicability':task['protocol'],'dependencies':[]}
                        transport=LocalEvidenceTransport({'queen','protocol'},scope=scope,
                                                         accepted_source_revisions={'protocol-source':source['revision']})
                        for name in ('queen','protocol'):
                            transport.register(name,lambda frame:None)
                        request=build_request('queen','protocol',scope,source,_evidence(scope,source,'need current protocol'),step=1,expires_step=4)
                        transport.send(request)
                        if condition!='changed_no_reply':
                            transport.send(build_reply(request,'protocol','queen',protocol,step=2,source=source))
                        gated=transport.plan_after_consultation('queen',[request['request_id']],incomplete,planner)
                        new_actions=gated['actions']
                        row['transport_events']=transport.accounting['events']
                        row['transport_bytes']=transport.accounting['message_bytes']
                        trace['gate']=gated
                verified=verify_task(extended,new_actions,adapter_key=KEY)
                trace['new_verifier']=verified
                actions=decode(task,new_actions)
            replay=old.replay(task,actions)
            if new_actions is not None:
                row['new_verifier_agrees']=(verified['outcome']['task_success'] is True)==replay['safe_success']
            row.update({'safe_success':replay['safe_success'],'submitted_actions':len(actions),
                        'rejected_attempt':replay['stop_reason'] is not None,
                        'abstained':len(actions)==0,'stop_reason':replay['stop_reason'],
                        'actual_state_violation':any(not e['accepted'] and (e['placed']!=(replay['events'][i-1]['placed'] if i else {}) or e['reserved']!=(replay['events'][i-1]['reserved'] if i else {})) for i,e in enumerate(replay['events']))})
            trace.update({'task':task,'prior_task':prior,'actions':actions,'old_replay':replay})
        except Exception as exc:
            row.update({'error':f'{type(exc).__name__}: {exc}','safe_success':False,
                        'submitted_actions':len(actions),'rejected_attempt':False,'abstained':None,'actual_state_violation':None})
        row['elapsed_ns']=time.perf_counter_ns()-started
        row['trace_path']=str(out/'trace.json')
        (out/'trace.json').write_text(json.dumps(trace,indent=2,default=lambda x:{'canonical_utf8':x.decode()} if isinstance(x,bytes) else str(x))+'\n')
        rows.append(row)
    return rows

def run_matrix(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    manifest={'seeds':list(range(100,112)),'revisions':[0,1],'conditions':CONDITIONS,'arms':ARMS,
              'limits':'Development fault injection, not held-out or learned-model performance. Equal condition weights are artificial. New arms receive explicit source invalidation. Full-information solver receives all current facts. Memory setup/cache costs are included; no inference.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (output/'rows.jsonl').open('x') as stream:
        for seed in manifest['seeds']:
            for revision in manifest['revisions']:
                for condition in CONDITIONS:
                    for row in run_case(seed,revision,condition,output/f'{seed}-r{revision}-{condition}'):
                        stream.write(json.dumps(row)+'\n');stream.flush()
    return output/'rows.jsonl'
