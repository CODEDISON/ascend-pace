"""Launch the job driver and wait for natural exit; no signals, no retries."""
import json
import os
import subprocess
import sys
import time

import plan


def supervise(environment, popen=subprocess.Popen, sleep=time.sleep, emit=None):
    job_id = environment.get('MEMCAMP_JOB')
    trim = environment.get('MEMCAMP_TRIM', '0')
    if job_id not in plan.CAMPAIGN_ORDER:
        raise ValueError('job outside the registered campaign')
    try:
        trim_steps = int(trim)
    except ValueError:
        raise ValueError('trim steps not an integer')
    spec = plan.job_spec(job_id, trim_steps)
    if environment.get('MEMCAMP_ARMS_SHA256') != plan.ARMS_SHA256:
        raise ValueError('arms identity environment differs')
    if environment.get('MEMCAMP_L3') not in ('0', '1'):
        raise ValueError('L3 flag must be explicit (A0 pilot rule decides)')
    emit = emit or (lambda value: print(value, flush=True))
    env = environment.copy()
    env.update({'PYTHONUNBUFFERED': '1'})
    try:
        process = popen([sys.executable, '-B', '-u', '/workspace/lane/worker.py'], env=env)
    except Exception as exc:
        emit(json.dumps({'event': 'start_failed', 'error': type(exc).__name__ + ': ' + str(exc)}))
        return {0: -1}, [{'error': str(exc)}]
    emit(json.dumps({'event': 'driver_started', 'job': job_id, 'pid': process.pid}))
    observed = None
    warned = False
    while observed is None:
        code = process.poll()
        if code is not None:
            observed = code
            emit(json.dumps({'event': 'natural_exit', 'job': job_id, 'code': code}))
        if not warned and observed not in (None, 0):
            emit('DRIVER_FAILED: preserve container and evidence; no signaling or retries')
            warned = True
        if observed is None:
            sleep(1)
    return {0: observed}, []


if __name__ == '__main__':
    observed, errors = supervise(dict(os.environ))
    if errors or observed.get(0) != 0:
        raise SystemExit(1)
    print(json.dumps({'event': 'job_natural_exit', 'job': os.environ['MEMCAMP_JOB'],
                      'trim': int(os.environ.get('MEMCAMP_TRIM', '0'))}), flush=True)
