"""Launch the 16 rank processes and wait for every started child naturally.

No signals, no retries, no cleanup: a failed rank is reported and the
survivors are preserved exactly as required by the fail-closed discipline.
"""
import json
import os
import subprocess
import sys
import time

import plan


def supervise(environment, popen=subprocess.Popen, sleep=time.sleep, emit=None):
    world = int(environment['WORLD_SIZE'])
    variant = environment.get('REVSD_VARIANT')
    group_key = environment.get('REVSD_GROUP')
    try:
        drops = int(environment.get('REVSD_DROPS', '0'))
    except ValueError:
        drops = -1
    if world != plan.WORLD or environment.get('ASCEND_RT_VISIBLE_DEVICES') != \
            ','.join(map(str, range(world))):
        raise ValueError('this lane supports only the fixed P16 visibility')
    if variant not in plan.VARIANTS or group_key not in plan.GROUPS[variant] \
            or not 0 <= drops <= plan.MAX_DROPS:
        raise ValueError('variant/group/drops outside the reviewed plan')
    if environment.get('REVSD_CASES_SHA256') != plan.CASES_SHA256:
        raise ValueError('cases identity environment differs')
    emit = emit or (lambda value: print(value, flush=True))
    processes, errors = {}, []
    for rank in range(world):
        env = environment.copy()
        env.update({'RANK': str(rank), 'LOCAL_RANK': str(rank), 'PYTHONUNBUFFERED': '1'})
        try:
            processes[rank] = popen([sys.executable, '-B', '-u', '/workspace/lane/worker.py'], env=env)
        except Exception as exc:
            errors.append({'rank': rank, 'error': type(exc).__name__ + ': ' + str(exc)})
            emit(json.dumps({'event': 'start_failed', **errors[-1]}))
            continue
        emit(json.dumps({'event': 'rank_started', 'rank': rank, 'pid': processes[rank].pid}))
    observed, warned = {}, False
    while len(observed) < len(processes):
        for rank, process in processes.items():
            code = process.poll()
            if code is not None and rank not in observed:
                observed[rank] = code
                emit(json.dumps({'event': 'natural_exit', 'rank': rank, 'code': code}))
        if not warned and (errors or any(code != 0 for code in observed.values())):
            emit('PEER_FAILED: preserve surviving processes; no signaling or retries')
            warned = True
        if len(observed) < len(processes):
            sleep(1)
    return observed, errors


if __name__ == '__main__':
    world = int(os.environ['WORLD_SIZE'])
    observed, errors = supervise(dict(os.environ))
    if errors or len(observed) != world or any(value != 0 for value in observed.values()):
        raise SystemExit(1)
    print(json.dumps({'event': 'all_natural_exit', 'world': world,
                      'variant': os.environ['REVSD_VARIANT'],
                      'group': os.environ['REVSD_GROUP'],
                      'drops': int(os.environ.get('REVSD_DROPS', '0'))}), flush=True)
