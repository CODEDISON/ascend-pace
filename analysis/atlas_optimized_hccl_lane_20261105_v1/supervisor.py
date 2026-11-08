"""Keep the container alive while the campaign driver (and any vLLM child it
spawned) runs; natural exits only, no signals, no retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def live_children():
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            status = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if status['State'].strip().split()[0] not in ('Z', 'X'):
            result.append({'pid': int(path.name), 'name': status['Name'].strip(),
                           'state': status['State'].strip()})
    return result


def supervise(entry='/workspace/lane/worker.py', popen=subprocess.Popen,
              live=live_children, sleep=time.sleep, emit=None, reap=True):
    emit = emit or (lambda row: print(json.dumps(row), flush=True))
    child = popen([sys.executable, '-B', '-u', entry])
    emit({'event': 'driver_started', 'pid': child.pid})
    while child.poll() is None:
        sleep(1)
    code = child.returncode
    emit({'event': 'driver_exit', 'pid': child.pid, 'exit_code': code})
    warned = False
    while True:
        if reap:
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
        peers = live()
        if not peers:
            break
        if not warned:
            emit({'event': 'peers_still_live', 'driver_exit_code': code, 'peers': peers,
                  'action': 'preserve until natural exit; no signals or retries'})
            warned = True
        sleep(1)
    emit({'event': 'supervisor_complete', 'driver_exit_code': code,
          'remaining_live_processes': 0})
    return code


if __name__ == '__main__':
    raise SystemExit(supervise())
