"""Launch ranks and wait for every started child naturally; never signal or retry."""
import json
import os
import subprocess
import sys
import time

import plan


def supervise(environment, popen=subprocess.Popen, sleep=time.sleep, emit=None):
    world = int(environment['WORLD_SIZE'])
    arm = environment.get('REV3_ARM')
    try:
        group = int(environment.get('REV3_GROUP_INDEX', '0'))
    except ValueError:
        group = -1
    if world != plan.WORLD or environment.get('ASCEND_RT_VISIBLE_DEVICES') != ','.join(map(str, range(world))):
        raise ValueError('v3 supports only the fixed P16 visibility')
    if arm not in plan.ARMS or (arm, group) not in plan.GROUP_ORDER:
        raise ValueError('arm/group outside the reviewed plan')
    emit = emit or (lambda value: print(value, flush=True))
    processes, errors = {}, []
    for rank in range(world):
        env = environment.copy()
        env.update({'RANK': str(rank), 'LOCAL_RANK': str(rank), 'PYTHONUNBUFFERED': '1'})
        try:
            processes[rank] = popen([sys.executable, '-B', '-u', '/workspace/barrier/worker.py'], env=env)
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
                      'arm': os.environ['REV3_ARM'], 'group': int(os.environ['REV3_GROUP_INDEX'])}), flush=True)
