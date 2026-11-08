"""Source-frozen launcher for the concern-#4 four-arm optimized-HCCL campaign.

One no-device precheck, then the optional O2 honorability probe and the F-arm
restoration-proof gate (both BEFORE g01), then strictly sequential fresh
containers for the 16 Williams-schedule groups under the pre-registered
trimming ladder, each gated by two complete 16-die admission preflights
(which double as the exclusivity-window re-check between groups), then a
full-rehash aggregate. The independent-review expectation is defined ONCE in
plan.py (plan.validate_review, aliased below) so the v3-class BLOCKER cannot
recur. No automatic retries, signals, container removal, resets or
destructive writes; probe/pilot/proof receipts are optional only where the
plan makes them optional (probe: P0 default; f_proof: hard gate for F).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import plan

WORK = Path('/home/workspace/chenyun/hpa_cgo2027_workspace')
NAME = 'atlas_optimized_hccl_lane_20261105_v1'
SOURCE = WORK / 'experiments' / NAME
PLAN_SOURCE = WORK / 'experiments' / 'atlas_optimized_hccl_plan_20261105_v1'
RESULTS = WORK / 'results/diagnostic' / NAME
GUARD = WORK / 'experiments/revision2_group_admission_device_20261028_v3'
BINARY = WORK / 'src/revision2_sdma_revalidation_20261028_v1'
SHMEM = WORK / 'src/shmem'
HELPER = WORK / 'experiments/revision2_sdma_queue_metadata_20261028_v1/metadata.py'
LARGE = WORK / 'experiments/revision2_sdma_large_correctness_20261028_v1'
CANDIDATE = WORK / 'experiments/revision2_native_hccl_no_resize_20261028_v1'
RUNNER = WORK / 'experiments/revision2_native_hccl_no_resize_o1_20261028_v2'
TIMING = WORK / 'experiments/revision2_native_hccl_no_resize_timing_20261028_v1'
SYNC = WORK / 'experiments/revision2_olmoe_o1_sync_model_20261028_v1'
# Home of the sync-model runtime module FAMILY: the qualification HARNESS
# v3 directory — model_contract.QUALIFICATION_SHA binds this manifest and
# sources() verifies per-module digests against it (the v2 dir satisfied
# imports only; see plan.py for the lineage).
QUALIFICATION = WORK / 'experiments/revision2_olmoe_o1_sync_qualification_20261028_v3'
# Delegated-chain data-file homes (AST audit, plan.DELEGATED_DATA_FILES):
LEGACY = WORK / plan.LEGACY_HOST_DIR                 # prompt helper (PROMPT_SOURCE_SHA)
MODEL = WORK / plan.MODEL_HOST_DIR                   # check_model + prompt_batch
MIXED_RESULTS = WORK / plan.MIXED_RESULTS_HOST_DIR   # prior mixed qualification receipt
# Home of capture_contract/capture_core/run_capture/worker_extension, imported
# by bare name from run_timing.py / run_o1.py / no_resize_worker.py
# (device-observed f_proof failure 216a2400: ModuleNotFoundError
# 'capture_contract'); its manifest was already pinned as CAPTURE_MANIFEST.
CAPTURE = WORK / plan.CAPTURE_HOST_DIR
REFERENCE_DIR = WORK / 'results/diagnostic/revision2_olmoe_split_capture_20261028_v1/hccl-o1'
PARITY_RECEIPTS = {'native': WORK / 'results/diagnostic/revision2_native_no_resize_o1_lane_20261028_v2/native/completion.json',
                   'no-resize': WORK / 'results/diagnostic/revision2_native_no_resize_o1_lane_20261028_v2/no-resize/completion.json'}
SYNC_RECEIPT = WORK / 'results/diagnostic/revision2_olmoe_o1_sync_model_20261028_v2/p16/completion.json'

FILES = plan.FILES
verify_review_binding = plan.validate_review  # the ONE shared contract consumer


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('expected ordinary file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = path.stat()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('file changed while hashing: ' + str(path))
    return digest.hexdigest()


def publish(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read(path):
    path = Path(path)
    sha(path)
    return json.loads(path.read_text())


def command(args, timeout=30):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout


def check_source_tree(source):
    """Recursive tree coverage: only the frozen lane files, the manifest, the
    review and __pycache__ subtrees are permitted. Symlinks are rejected
    UNCONDITIONALLY — including inside __pycache__ (only regular files are
    permitted there) — so no intermediate version or alias ships with the
    lane."""
    observed_files, observed_dirs = set(), set()
    for path in Path(source).rglob('*'):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError('source symlink forbidden: ' + relative)
        if '__pycache__' in path.parts:
            continue
        if path.is_dir():
            observed_dirs.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise ValueError('unexpected source tree entry: ' + relative)
    if observed_dirs or observed_files != FILES | {'source_manifest.json', 'review.json'}:
        raise ValueError('source tree coverage differs; remove stray intermediate files')


def load_guard():
    if sha(GUARD / 'launch.py') != plan.GUARD_LAUNCH_SHA:
        raise ValueError('admission guard source mismatch')
    spec = importlib.util.spec_from_file_location('opt_hccl_admission_guard', GUARD / 'launch.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.manifest(GUARD, plan.GUARD_MANIFEST_SHA, guard.FILES)
    guard.RESULTS = RESULTS
    return guard


def verify(manifest_sha, review_sha):
    if sha(SOURCE / 'source_manifest.json') != manifest_sha:
        raise ValueError('source identity mismatch')
    manifest = json.loads((SOURCE / 'source_manifest.json').read_text())
    if set(manifest) != FILES or any(sha(SOURCE / name) != digest for name, digest in manifest.items()):
        raise ValueError('source graph changed')
    if sha(Path(__file__)) != manifest['launch.py']:
        raise ValueError('launcher does not run from the frozen source graph')
    check_source_tree(SOURCE)
    if sha(SOURCE / 'review.json') != review_sha:
        raise ValueError('review mismatch')
    verify_review_binding(read(SOURCE / 'review.json'), manifest_sha)
    plan.validate_schedule()
    if sha(PLAN_SOURCE / 'arms.json') != plan.ARMS_JSON_SHA \
            or sha(PLAN_SOURCE / 'PLAN.md') != plan.PLAN_MD_SHA:
        raise ValueError('bound plan documents differ')
    if sha(BINARY / 'cashflex/lib/libcashflex.so') != plan.LIB_SHA256 \
            or sha(SHMEM / 'build_cann9_vllm/lib/libshmem.so') != plan.SHMEM_SHA256:
        raise ValueError('runtime provider mismatch')
    if sha(HELPER) != plan.HELPER_SHA256 or sha(LARGE / 'layout.json') != plan.LAYOUT_SHA256:
        raise ValueError('frozen queue-metadata helper/layout mismatch')
    for name, root, pin in (('candidate', CANDIDATE, plan.CANDIDATE_MANIFEST),
                            ('runner', RUNNER, plan.RUNNER_MANIFEST),
                            ('timing', TIMING, plan.TIMING_MANIFEST),
                            ('sync_model', SYNC, plan.SYNC_MODEL_MANIFEST),
                            ('qualification', QUALIFICATION, plan.QUALIFICATION_MANIFEST),
                            ('capture', CAPTURE, plan.CAPTURE_MANIFEST)):
        if sha(root / 'source_manifest.json') != pin:
            raise ValueError(f'frozen lane manifest differs: {name}')
    for member in plan.F_RUNTIME_FAMILY:
        if not (QUALIFICATION / member).is_file():
            raise ValueError(f'sync-model runtime family member missing: {member}')
    for member in ('capture_contract.py', 'capture_core.py', 'run_capture.py',
                   'worker_extension.py'):
        if not (CAPTURE / member).is_file():
            raise ValueError(f'capture-lane module missing: {member}')
    # Delegated-chain data files, HOST side (existence hard gates + digest
    # pins; the container-side existence proof over plan.DELEGATED_DATA_FILES
    # runs in the no-device precheck). Device failure d4e26ece was the
    # legacy prompt helper — a path+SHA read the import closure cannot see.
    if sha(LEGACY / 'benchmark_olmoe_fixed_batch.py') != plan.PROMPT_HELPER_SHA:
        raise ValueError('legacy prompt helper missing or digest differs')
    if not MODEL.is_dir():
        raise ValueError('fixed model directory missing')
    if sha(MIXED_RESULTS / 'completion.json') != plan.MIXED_COMPLETION_SHA:
        raise ValueError('prior mixed qualification receipt missing or differs')
    require_qualification_receipts()
    return load_guard()


def require_qualification_receipts():
    """Prior PASS receipts revalidated by exact hash (PLAN section 8.6)."""
    for arm, (path, pin) in {'native': (PARITY_RECEIPTS['native'], plan.PARITY_NATIVE_COMPLETION),
                             'no-resize': (PARITY_RECEIPTS['no-resize'], plan.PARITY_NORESIZE_COMPLETION),
                             'sync-model': (SYNC_RECEIPT, plan.SYNC_MODEL_COMPLETION)}.items():
        if sha(path) != pin:
            raise ValueError(f'prior qualification receipt differs: {arm}')
        receipt = read(path)
        if receipt.get('status') != 'PASS':
            raise ValueError(f'prior qualification did not pass: {arm}')


def fixed_mounts():
    return [(Path('/usr/local/Ascend/driver'), '/usr/local/Ascend/driver', 'ro'),
            (Path('/etc/ascend_install.info'), '/etc/ascend_install.info', 'ro'),
            (BINARY, '/workspace/cashflex', 'ro'), (SHMEM, '/workspace/shmem', 'ro'),
            (HELPER.parent, '/workspace/queue_metadata', 'ro'),
            (LARGE, '/workspace/sdma_large', 'ro'),
            (CANDIDATE, '/workspace/no_resize_candidate', 'ro'),
            (RUNNER, '/workspace/no_resize_lane', 'ro'),
            (TIMING, '/workspace/native_timing_lane', 'ro'),
            (SYNC, '/workspace/model_hook', 'ro'),
            (QUALIFICATION, '/workspace/qualification', 'ro'),
            (CAPTURE, plan.CAPTURE_CONTAINER_DIR, 'ro'),
            (LEGACY, '/workspace/legacy', 'ro'),
            (MODEL, '/workspace/model', 'ro'),
            (MIXED_RESULTS, '/workspace/mixed_qualification', 'ro'),
            (PLAN_SOURCE, '/workspace/plan', 'ro'),
            (SOURCE, '/workspace/lane', 'ro'),
            (REFERENCE_DIR, '/workspace/reference', 'ro')]


def expected_mounts(payload=None):
    mounts = {target: (str(host), mode == 'rw') for host, target, mode in fixed_mounts()}
    if payload is not None:
        mounts['/workspace/results'] = (str(payload), True)
    return mounts


def entry_script(entry):
    # The seven module-home directories appear in plan.DELEGATED_MODULE_ROOTS
    # order — resolution semantics equal sys.path order. /workspace/
    # qualification precedes model_hook and queue_metadata so the sync-model
    # family's internal queue_metadata sibling wins same-name resolution
    # (the lane's own helper use is by absolute SHA-checked path). The
    # no-device precheck proves the delegated closure resolves fully under
    # these roots (f_proof failures 54167a66 / 216a2400 were closure members
    # whose home was missing).
    return ('set -euo pipefail; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
            'export PYTHONPATH=' + ':'.join(DELEGATED_PYTHONPATH) + ':${PYTHONPATH:-}; '
            'export LD_LIBRARY_PATH=/workspace/cashflex/cashflex/lib:'
            '/workspace/shmem/build_cann9_vllm/lib:${LD_LIBRARY_PATH:-}; '
            'exec python3 -B -u ' + entry)


# The container PYTHONPATH, single source: the seven module-home roots in
# plan.DELEGATED_MODULE_ROOTS order, with the non-module roots (plan dir,
# cashflex package parent, queue-metadata helper dir, sdma_large) appended
# where they carry no bare-name module used by the delegated closure.
DELEGATED_PYTHONPATH = [
    '/workspace/lane', '/workspace/plan', '/workspace/cashflex',
    '/workspace/qualification',
    '/workspace/no_resize_candidate',
    '/workspace/capture',
    '/workspace/no_resize_lane',
    '/workspace/native_timing_lane',
    '/workspace/model_hook',
    '/workspace/queue_metadata',
    '/workspace/sdma_large',
]


def runtime_args(name, network, ipc, env, entry, guard=None, world=0, payload=None):
    args = ['docker', 'run', '-d', '--name', name, '--network', network, '--ipc', ipc]
    if guard is not None:
        args.append('--shm-size=64g')
        for dev in guard.nodes(world):
            args += ['--device', f'{dev}:{dev}:rwm']
    mounts = fixed_mounts() + ([(payload, '/workspace/results', 'rw')] if payload is not None else [])
    for host, target, mode in mounts:
        if not host.exists():
            raise ValueError('missing mount: ' + str(host))
        args += ['-v', f'{host}:{target}:{mode}']
    for key, value in env.items():
        args += ['-e', f'{key}={value}']
    args += [plan.IMAGE, 'bash', '-lc', entry_script(entry)]
    return args


def inspect_contract(info, argv, env, guard=None, world=0, payload=None):
    host = info['HostConfig']
    expected_network = 'host' if guard else 'none'
    expected_ipc = 'host' if guard else 'private'
    if info['Image'] != plan.IMAGE or host.get('NetworkMode') != expected_network \
            or host.get('IpcMode') != expected_ipc:
        raise RuntimeError('image/network/ipc differs')
    state = info['State']
    if state.get('Running') or state.get('ExitCode') != 0 or state.get('OOMKilled') or state.get('Error'):
        raise RuntimeError('container did not naturally complete successfully')
    got_env = dict(x.split('=', 1) for x in info['Config']['Env'] if '=' in x)
    if any(got_env.get(key) != value for key, value in env.items()):
        raise RuntimeError('environment differs')
    devices = host.get('Devices') or []
    actual = {(x['PathOnHost'], x['PathInContainer'], x['CgroupPermissions']) for x in devices}
    expected = {(x, x, 'rwm') for x in guard.nodes(world)} if guard else set()
    if actual != expected or host.get('DeviceRequests') or host.get('DeviceCgroupRules'):
        raise RuntimeError('device map differs')
    actual_mounts = {x['Destination']: (x['Source'], x['RW']) for x in info['Mounts']}
    if actual_mounts != expected_mounts(payload):
        raise RuntimeError('mount map differs')
    if info['Config']['Cmd'] != argv[-3:]:
        raise RuntimeError('command differs')


# ------------------------------------------------------------ precheck ----
def precheck_arm(manifest_sha, review_sha):
    verify(manifest_sha, review_sha)
    directory = RESULTS / 'precheck'
    if directory.exists() or directory.is_symlink():
        raise ValueError('refuse to replace the no-device precheck')
    directory.mkdir(parents=True, exist_ok=False)
    env = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
    args = runtime_args(NAME + '-precheck', 'none', 'private', env, '/workspace/lane/precheck.py')
    publish(directory / 'inputs.json', {'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
                                        'docker_argv': args, 'env': env})
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown precheck container id')
    publish(directory / 'container.json', {'id': container})
    print(json.dumps({'event': 'precheck_started', 'id': container}), flush=True)


def precheck_record(logs):
    found = []
    for line in logs.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('event') == 'NO_DEVICE_PRECHECK_PASS':
            found.append(value)
    if len(found) != 1:
        raise RuntimeError('precheck completion marker missing or duplicated')
    return found[0]


def precheck_audit(manifest_sha, review_sha):
    verify(manifest_sha, review_sha)
    directory = RESULTS / 'precheck'
    inputs = read(directory / 'inputs.json')
    info = read(directory / 'container.json')
    inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
    if inspected['State'].get('Running'):
        print(json.dumps({'event': 'precheck_running'}), flush=True)
        return
    logs = subprocess.run(['docker', 'logs', info['id']], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=30).stdout
    publish(directory / 'inspect.json', inspected)
    with (directory / 'container.log').open('x') as out:
        out.write(logs)
    inspect_contract(inspected, inputs['docker_argv'], inputs['env'])
    record = precheck_record(logs)
    if (record.get('provider_sha256') != plan.SHMEM_SHA256
            or record.get('libhccl_sha256') != plan.LIBHCCL_SHA
            or record.get('arms_json_sha256') != plan.ARMS_JSON_SHA
            or record.get('f_fake_import_proof', {}).get('proof') != 'cpu_fake_import_restoration'
            or record.get('f_fake_import_proof', {}).get('hook_layer_retained') is not True
            or record.get('frozen_lane_manifests') != {'candidate': plan.CANDIDATE_MANIFEST,
                                                       'runner': plan.RUNNER_MANIFEST,
                                                       'timing': plan.TIMING_MANIFEST,
                                                       'sync_model': plan.SYNC_MODEL_MANIFEST,
                                                       'qualification': plan.QUALIFICATION_MANIFEST}
            or record.get('f_runtime_family_resolved') != sorted(plan.F_RUNTIME_FAMILY)
            or record.get('delegated_closure', {}).get('entries') != list(plan.DELEGATED_ENTRY_FILES)
            or record.get('delegated_closure', {}).get('module_roots') != list(plan.DELEGATED_MODULE_ROOTS)
            or record.get('delegated_closure', {}).get('unresolved') != []
            or not record.get('delegated_closure', {}).get('resolved_modules')
            or any(not home.startswith('/workspace/') for home
                   in record['delegated_closure']['resolved_modules'].values())
            or sorted(record.get('delegated_data_files', {})) != sorted(
                path for path, _kind, _pin, _c in plan.DELEGATED_DATA_FILES)
            or any(row.get('pinned') is True
                   and row.get('sha256') != dict((p, pin) for p, _k, pin, _c
                                                 in plan.DELEGATED_DATA_FILES)[path]
                   for path, row in record.get('delegated_data_files', {}).items())
            or sorted(record.get('delegated_env_ready', {})) != sorted(plan.ARMS)
            or any(record.get('delegated_env_ready', {}).get(arm, {}).get(name, '') == ''
                   for arm in plan.ARMS for name in plan.required_env_names(arm))
            or any(record['delegated_env_ready'][arm]['REV2_NATIVE_TIMING_MANIFEST'] != plan.TIMING_MANIFEST
                   or record['delegated_env_ready'][arm]['REV2_NO_RESIZE_SOURCE_MANIFEST'] != plan.RUNNER_MANIFEST
                   for arm in plan.HCCL_ARMS)
            or record.get('delegated_env_ready', {}).get('F', {}).get('REV2_MODEL_MANIFEST_SHA256') != plan.SYNC_MODEL_MANIFEST
            or record.get('schedule') != {'groups': list(plan.GROUPS), 'arm_map': dict(plan.GROUP_ARM)}
            or record.get('review_contract') != {'schema': plan.REVIEW_SCHEMA,
                                                 'scope': plan.REVIEW_SCOPE,
                                                 'groups': plan.REVIEW_GROUPS,
                                                 'reviewed_files': sorted(plan.REVIEWED_FILES)}):
        raise RuntimeError('precheck evidence missing or inconsistent')
    plan.check_pace_schema(record['pace_schema'])
    if not re.fullmatch('[0-9a-f]{64}', record.get('acl_library_sha256', '')):
        raise RuntimeError('ACL provider identity missing')
    publish(directory / 'completion.json', {'status': 'PASS', 'manifest_sha256': manifest_sha,
            'review_sha256': review_sha, 'image_id': plan.IMAGE, 'record': record,
            'acl': {'path': record['acl_library_path'], 'sha256': record['acl_library_sha256']},
            'log_sha256': sha(directory / 'container.log'),
            'inspect_sha256': sha(directory / 'inspect.json')})
    print(json.dumps({'event': 'precheck_pass'}), flush=True)


def require_precheck(manifest_sha, review_sha):
    receipt = read(RESULTS / 'precheck' / 'completion.json')
    if (receipt.get('status') != 'PASS' or receipt.get('manifest_sha256') != manifest_sha
            or receipt.get('review_sha256') != review_sha or receipt.get('image_id') != plan.IMAGE
            or receipt.get('log_sha256') != sha(RESULTS / 'precheck' / 'container.log')
            or receipt.get('inspect_sha256') != sha(RESULTS / 'precheck' / 'inspect.json')):
        raise ValueError('matched no-device precheck required')
    return receipt


def review_view(manifest_sha, review_sha):
    return plan.validate_review(read(SOURCE / 'review.json'), manifest_sha)


# ------------------------------------------------ probe / f-proof gate ----
PROBE_WORKER = r'''
import hashlib, json, os
import torch
import torch.distributed as dist
import torch_npu
rank = 0
world = int(os.environ['WORLD_SIZE'])
torch.npu.set_device(rank)
dist.init_process_group('hccl')
marker = torch.ones(1, dtype=torch.int32, device='npu:0')
dist.all_reduce(marker)
torch.npu.synchronize()
assert int(marker.cpu().item()) == world
dist.destroy_process_group()
import glob
roots = ['/root/ascend/log', os.path.expanduser('~/ascend/log'), '/var/log/ascend',
         os.environ.get('ASCEND_PROCESS_LOG_PATH', '/nonexistent')]
files = sorted(path for root in roots for path in glob.glob(root + '/**/*.log', recursive=True))
payload = os.environ['PROBE_PAYLOAD_DIR']
os.makedirs(payload, exist_ok=True)
matched_files, combined = [], []
for path in files:
    try:
        raw = open(path, 'rb').read()
    except OSError:
        continue
    text = raw.decode('utf-8', 'replace')
    if 'HCCL_BUFFSIZE' in text:
        matched_files.append({'path': path, 'sha256': hashlib.sha256(raw).hexdigest()})
        combined.append('#### ' + path + '\n' + text)
blob = '\n'.join(combined).encode()
open(payload + '/probe_logs.txt', 'wb').write(blob)
matched = [line for line in blob.decode('utf-8', 'replace').splitlines()
           if 'HCCL_BUFFSIZE' in line]
receipt = {'event': 'PROBE_RAW', 'matched_files': matched_files,
           'combined_sha256': hashlib.sha256(blob).hexdigest(),
           'matched_lines': matched, 'probe_value': os.environ.get('HCCL_BUFFSIZE')}
open(payload + '/probe_receipt.json', 'w').write(json.dumps(receipt, indent=2, sort_keys=True))
print(json.dumps({'event': 'PROBE_RAW', 'matched': len(matched),
                  'matched_files': len(matched_files), 'scanned': len(files)}), flush=True)
'''


def probe_entry_script():
    return ('set -euo pipefail; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
            'export PYTHONPATH=/workspace/cashflex:${PYTHONPATH:-}; '
            'python3 -B - <<\'REV4PROBE\'\n' + PROBE_WORKER + 'REV4PROBE\n')


def probe_runtime_args(name, env, guard, payload):
    args = ['docker', 'run', '-d', '--name', name, '--network', 'host', '--ipc', 'host',
            '--shm-size=64g']
    for dev in guard.nodes(plan.WORLD):
        args += ['--device', f'{dev}:{dev}:rwm']
    for host, target, mode in fixed_mounts() + [(payload, '/workspace/results', 'rw')]:
        if not host.exists():
            raise ValueError('missing mount: ' + str(host))
        args += ['-v', f'{host}:{target}:{mode}']
    for key, value in env.items():
        args += ['-e', f'{key}={value}']
    args += [plan.IMAGE, 'bash', '-lc', probe_entry_script()]
    return args


def probe_start(manifest_sha, review_sha):
    """Optional O2 honorability probe (<= 3 min), before g01, exactly once."""
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    if group_receipts():
        raise ValueError('probe must run before the first campaign group')
    directory = RESULTS / 'probe'
    if directory.exists() or directory.is_symlink():
        raise ValueError('probe already attempted; no retries')
    _, master_port = plan.ports('probe')
    first = snapshot(guard, 'probe')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'payload').mkdir()
    env = {'WORLD_SIZE': str(plan.WORLD), 'MASTER_ADDR': '127.0.0.1',
           'MASTER_PORT': str(master_port), 'RANK': '0',
           'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(plan.WORLD))),
           'HCCL_BUFFSIZE': plan.PROBE_VALUE, 'PROBE_PAYLOAD_DIR': '/workspace/results',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
    args = probe_runtime_args(NAME + '-probe', env, guard, directory / 'payload')
    publish(directory / 'preflight.json', {'first': first, 'docker_argv': args, 'env': env,
            'scope': 'one HCCL init + one all-reduce with HCCL logs enabled'})
    publish(directory / 'prelaunch_recheck.json', snapshot(guard, 'probe'))
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown probe container id')
    publish(directory / 'container.json', {'id': container})
    print(json.dumps({'event': 'probe_started', 'id': container}), flush=True)


def probe_audit(manifest_sha, review_sha):
    verify(manifest_sha, review_sha)
    directory = RESULTS / 'probe'
    info = read(directory / 'container.json')
    inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
    if inspected['State'].get('Running'):
        print(json.dumps({'event': 'probe_running'}), flush=True)
        return
    logs = subprocess.run(['docker', 'logs', info['id']], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=60).stdout
    publish(directory / 'inspect.json', inspected)
    with (directory / 'container.log').open('x') as out:
        out.write(logs)
    before = read(directory / 'preflight.json')
    inspect_contract(inspected, before['docker_argv'], before['env'], load_guard(),
                     plan.WORLD, directory / 'payload')
    receipt_path = directory / 'payload' / 'probe_receipt.json'
    if not receipt_path.is_file() or not (directory / 'payload' / 'probe_logs.txt').is_file():
        judgment = plan.judge_probe_receipt('', [])
    else:
        receipt = read(receipt_path)
        blob = (directory / 'payload' / 'probe_logs.txt').read_bytes()
        if hashlib.sha256(blob).hexdigest() != receipt.get('combined_sha256'):
            raise ValueError('probe log evidence changed')
        judgment = plan.judge_probe_receipt(blob.decode('utf-8', 'replace'),
                                            receipt.get('matched_lines', []))
    publish(directory / 'completion.json', {'status': 'DONE', 'judgment': judgment,
            'container_log_sha256': sha(directory / 'container.log'),
            'inspect_sha256': sha(directory / 'inspect.json'),
            'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
            'scope': 'no honored knob => P0 (HOPT == HmR configuration)'})
    print(json.dumps({'event': 'probe_audited', 'path': judgment['path']}), flush=True)


def probe_path(manifest_sha, review_sha):
    """Campaign probe path: review-declared path governs; an audited probe
    receipt must agree with it when present."""
    review = review_view(manifest_sha, review_sha)
    directory = RESULTS / 'probe'
    if not (directory / 'completion.json').is_file():
        if review['probe_path'] != 'P0':
            raise ValueError('review declares P1 without an audited probe receipt')
        return 'P0', None
    judgment = read(directory / 'completion.json')['judgment']
    if judgment['path'] != review['probe_path']:
        raise ValueError('probe receipt and review disagree on the path')
    return review['probe_path'], review['hopt_env']


def group_receipts():
    receipts = {}
    for group_key in plan.GROUPS:
        path = RESULTS / group_key / 'completion.json'
        if path.is_file():
            receipts[group_key] = read(path)
    return receipts


def require_f_proof(manifest_sha, review_sha, arm):
    """Hard gate for F groups (PLAN section 13.2): the review must say PROVEN
    AND an audited one-request restoration proof must exist and pass."""
    review = review_view(manifest_sha, review_sha)
    if arm != 'F':
        return None
    if review['f_restoration_gate'] != 'PROVEN':
        raise ValueError('review did not certify the F-arm restoration gate')
    directory = RESULTS / 'f_proof'
    receipt = read(directory / 'completion.json')
    if (receipt.get('status') != 'PASS' or receipt.get('manifest_sha256') != manifest_sha
            or receipt.get('review_sha256') != review_sha
            or receipt.get('recorder_removed') is not True
            or receipt.get('hook_retained') is not True
            or receipt.get('tokens_equal_reference') is not True):
        raise ValueError('F-arm one-request restoration proof missing or failed')
    return receipt


def f_proof_start(manifest_sha, review_sha):
    """One-request F restoration proof container (device, before g01)."""
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    if review_view(manifest_sha, review_sha)['f_restoration_gate'] != 'PROVEN':
        raise ValueError('review did not certify the F-arm restoration gate')
    directory = RESULTS / 'f_proof'
    if directory.exists() or directory.is_symlink():
        raise ValueError('F proof already attempted; no retries')
    first = snapshot(guard, 'f_proof')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'payload').mkdir()
    acl = require_precheck(manifest_sha, review_sha)['acl']
    env = group_env('f_proof', 'F', 'L0', acl, {}, manifest_sha, review_sha)
    env['REV4_MODE'] = 'f_proof'
    name = NAME + '-f-proof'
    args = runtime_args(name, 'host', 'host', env, '/workspace/lane/supervisor.py',
                        guard, plan.WORLD, directory / 'payload')
    publish(directory / 'preflight.json', {'first': first, 'docker_argv': args, 'env': env,
            'scope': 'F one-request restoration proof (worker REV4_MODE=f_proof)'})
    publish(directory / 'prelaunch_recheck.json', snapshot(guard, 'f_proof'))
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown f_proof container id')
    publish(directory / 'container.json', {'id': container})
    print(json.dumps({'event': 'f_proof_started', 'id': container}), flush=True)


def f_proof_audit(manifest_sha, review_sha):
    verify(manifest_sha, review_sha)
    directory = RESULTS / 'f_proof'
    info = read(directory / 'container.json')
    inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
    if inspected['State'].get('Running'):
        print(json.dumps({'event': 'f_proof_running'}), flush=True)
        return
    logs = subprocess.run(['docker', 'logs', info['id']], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=60).stdout
    publish(directory / 'inspect.json', inspected)
    with (directory / 'container.log').open('x') as out:
        out.write(logs)
    before = read(directory / 'preflight.json')
    inspect_contract(inspected, before['docker_argv'], before['env'], load_guard(),
                     plan.WORLD, directory / 'payload')
    marker = [row for row in json_rows(logs) if row.get('event') == 'F_PROOF_PASS']
    state = inspect_exit(inspected)
    if len(marker) != 1 or state:
        publish(directory / 'completion.json', {'status': 'FAIL', 'natural_exit': not state,
                'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
                'scope': 'f_proof failed; campaign degrades to three arms (explicit, '
                         'never silent; aggregate --degraded-3arm required)'})
        print(json.dumps({'event': 'f_proof_failed'}), flush=True)
        return
    publish(directory / 'completion.json', {'status': 'PASS', 'manifest_sha256': manifest_sha,
            'review_sha256': review_sha, 'recorder_removed': True, 'hook_retained': True,
            'tokens_equal_reference': True, 'record': marker[0],
            'container_log_sha256': sha(directory / 'container.log'),
            'inspect_sha256': sha(directory / 'inspect.json')})
    print(json.dumps({'event': 'f_proof_pass'}), flush=True)


def inspect_exit(inspected):
    state = inspected['State']
    return state.get('Running') or state.get('ExitCode') != 0 or state.get('OOMKilled') or state.get('Error')


def json_rows(logs):
    rows = []
    for line in logs.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def directory_of(group_key):
    return RESULTS / group_key


# ------------------------------------------------------------- groups ----
def snapshot(guard, job):
    """One complete 16-die admission preflight bound to this job's ports;
    doubles as the exclusivity-window re-check between groups."""
    shmem_port, master_port = plan.ports(job)
    guard.stage_info = lambda stage: ('native', plan.WORLD, (shmem_port, master_port))
    return guard.preflight('native-p16')


def check_snapshot(guard, snap, job):
    if snap.get('processes') != guard.parse_npu_info(snap['npu_smi_info']):
        raise ValueError('process table interpretation differs')
    shmem_port, master_port = plan.ports(job)
    if snap.get('ports') != [shmem_port, master_port] or \
            snap.get('devices') != list(guard.nodes(plan.WORLD)):
        raise ValueError('preflight port or device coverage differs')
    if set(snap['health']) != set(map(str, range(8))) or \
            set(snap['usage']) != set(map(str, range(8))):
        raise ValueError('health/usage card coverage differs')
    for card in range(8):
        guard.check_health(snap['health'][str(card)])
        guard.check_usage(snap['usage'][str(card)])


def group_env(job, arm, level, acl, hopt_env, manifest_sha, review_sha):
    shmem_port, master_port = plan.ports(job)
    env = {'REV4_ARM': arm, 'REV4_GROUP': job if job in plan.GROUPS else 'f_proof',
           'REV4_LADDER_LEVEL': level, 'REV4_RESULT_DIR': '/workspace/results',
           'REV4_HOPT_ENV': json.dumps(hopt_env, sort_keys=True),
           'REV4_ACL_LIBRARY_PATH': acl['path'], 'REV4_ACL_LIBRARY_SHA256': acl['sha256'],
           'REV2_RESULT_DIR': '/workspace/results', 'REV2_BACKEND': 'hccl', 'REV2_OUTPUT_TOKENS': '1',
           'VLLM_USE_V1': '1', 'VLLM_ASCEND_ENABLE_FUSED_MC2': '0',
           'WORLD_SIZE': str(plan.WORLD), 'MASTER_ADDR': '127.0.0.1',
           'MASTER_PORT': str(master_port),
           'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(plan.WORLD))),
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    if arm == 'F':
        env.update({'HPA_SHMEM_IP_PORT': f'tcp://127.0.0.1:{shmem_port}',
                    'HPA_SHMEM_WORKSPACE_PLANES': '2', 'REV2_RUN_NONCE': nonce(job),
                    # model_contract.sources() reads the prior mixed
                    # qualification receipt by this env-bound digest
                    'REV2_QUALIFICATION_COMPLETION_SHA256': plan.MIXED_COMPLETION_SHA})
    else:
        if any(key.startswith('HPA_') or key.startswith('REV4_F_') for key in env):
            raise ValueError('HCCL arms must not carry SHMEM env')
    for key, value in hopt_env.items():
        env[key] = value
    return env


def nonce(job):
    digest = hashlib.sha256((NAME + ':' + job).encode()).hexdigest()[:32]
    if not re.fullmatch('[0-9a-f]{32}', digest):
        raise ValueError('nonce derivation failed')
    return digest


def require_prior_groups(group_key, manifest_sha, review_sha, level):
    prior_keys = [key for key in plan.ladder_groups(level) if key != group_key
                  and plan.GROUPS.index(key) < plan.GROUPS.index(group_key)]
    receipts = []
    for prior in prior_keys:
        receipt = read(RESULTS / prior / 'completion.json')
        plan.validate_completion(receipt, prior, manifest_sha, review_sha)
        receipts.append(receipt)
    return receipts


def group_start(group_key, manifest_sha, review_sha, level):
    if group_key not in plan.GROUPS:
        raise ValueError('group outside the reviewed campaign')
    arm = plan.GROUP_ARM[group_key]
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    path, hopt = probe_path(manifest_sha, review_sha)
    if arm == 'HOPT' and plan.hopt_env(path, hopt) != (hopt or {}):
        raise ValueError('HOPT env delta outside the probe path')
    hopt = hopt or {}
    require_f_proof(manifest_sha, review_sha, arm)
    trigger = plan.ladder_step(require_prior_groups(group_key, manifest_sha, review_sha, level), level)
    directory = RESULTS / group_key
    if directory.exists() or directory.is_symlink():
        raise ValueError('group already exists; no retries')
    acl = require_precheck(manifest_sha, review_sha)['acl']
    first = snapshot(guard, group_key)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'payload').mkdir()
    env = group_env(group_key, arm, level, acl, hopt, manifest_sha, review_sha)
    name = f'{NAME}-{group_key}'
    args = runtime_args(name, 'host', 'host', env, '/workspace/lane/supervisor.py',
                        guard, plan.WORLD, directory / 'payload')
    publish(directory / 'preflight.json',
            {'source_manifest_sha256': manifest_sha, 'review_sha256': review_sha,
             'group': group_key, 'arm': arm, 'ladder_level': level, 'probe_path': path,
             'hopt_env': hopt, 'drop_trigger': trigger,
             'exclusivity_recheck': 'both preflights re-verify 16-die emptiness between groups',
             'first': first, 'docker_argv': args, 'env': env})
    publish(directory / 'prelaunch_recheck.json', snapshot(guard, group_key))
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown container id: inspect, no automatic retry')
    publish(directory / 'container.json', {'id': container, 'name': name})
    print(json.dumps({'event': 'started', 'group': group_key, 'arm': arm,
                      'ladder_level': level, 'probe_path': path, 'id': container}), flush=True)


def group_audit(group_key, manifest_sha, review_sha):
    if group_key not in plan.GROUPS:
        raise ValueError('group outside the reviewed campaign')
    arm = plan.GROUP_ARM[group_key]
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    directory = RESULTS / group_key
    info = read(directory / 'container.json')
    inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
    if inspected['State'].get('Running'):
        print(json.dumps({'event': 'still_running', 'group': group_key}), flush=True)
        return
    logs = subprocess.run(['docker', 'logs', info['id']], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=120).stdout
    publish(directory / 'inspect.json', inspected)
    with (directory / 'container.log').open('x') as out:
        out.write(logs)
    before = read(directory / 'preflight.json')
    level = before['ladder_level']
    inspect_contract(inspected, before['docker_argv'], before['env'], guard,
                     plan.WORLD, directory / 'payload')
    rows = json_rows(logs)
    summary = [row for row in rows if row.get('event') == 'group_summary']
    if len(summary) != 1 or summary[0].get('group') != group_key or summary[0].get('arm') != arm:
        raise RuntimeError('group summary marker missing')
    record_path = directory / 'payload' / 'group_record.json'
    if not record_path.is_file() or sha(record_path) != summary[0].get('record_sha256'):
        raise RuntimeError('persisted group record differs from the summary')
    record = read(record_path)
    plan.validate_group_record(record, group_key, arm, level)
    review = review_view(manifest_sha, review_sha)
    if arm == 'HOPT' and record.get('hopt_env') != (review['hopt_env'] or {}):
        raise ValueError('HOPT env delta differs from the frozen review binding')
    if arm != 'HOPT' and record.get('hopt_env') not in (None, {}):
        raise ValueError('only HOPT may carry an HCCL env delta')
    if not any(row.get('event') == 'supervisor_complete' and row.get('driver_exit_code') == 0
               for row in rows):
        raise RuntimeError('supervisor did not report a natural zero driver exit')
    payload_hashes = {name: sha(path) for name in sorted(
        path.name for path in (directory / 'payload').iterdir() if path.is_file())
        for path in [(directory / 'payload') / name]}
    # Both 16-die preflight snapshots are re-validated, not merely stored.
    check_snapshot(guard, before['first'], group_key)
    after = read(directory / 'prelaunch_recheck.json')
    check_snapshot(guard, after, group_key)
    plan.check_time_order(before['first']['utc'], after['utc'],
                          inspected['State']['StartedAt'], inspected['State']['FinishedAt'])
    duration = (plan.exact_ns(inspected['State']['FinishedAt'])
                - plan.exact_ns(inspected['State']['StartedAt'])) / 1e9
    postflight = {'scope': 'read-only snapshot; not an exclusive reservation'}
    try:
        postflight['npu_smi_info'] = guard.command(['npu-smi', 'info'], timeout=15)
        postflight['parsed'] = guard.parse_npu_info(postflight['npu_smi_info'])
        postflight['status'] = 'OBSERVED_EMPTY_TARGET'
    except Exception as exc:
        postflight['status'] = 'UNCONFIRMED'
        postflight['error'] = type(exc).__name__ + ': ' + str(exc)
    publish(directory / 'postflight.json', postflight)
    publish(directory / 'completion.json',
            {'schema': 'revision2.optimized-hccl-completion.v1', 'status': 'PASS',
             'group': group_key, 'arm': arm, 'world': plan.WORLD,
             'ladder_level': level, 'manifest_sha256': manifest_sha,
             'review_sha256': review_sha, 'image_id': plan.IMAGE,
             'duration_seconds': duration, 'group_median_ms': summary[0]['group_median_ms'],
             'scope': 'fresh process group; W3/I20; validation outside the timer',
             'precheck_sha256': sha(RESULTS / 'precheck' / 'completion.json'),
             'container_log_sha256': sha(directory / 'container.log'),
             'inspect_sha256': sha(directory / 'inspect.json'),
             'preflight_sha256': sha(directory / 'preflight.json'),
             'prelaunch_recheck_sha256': sha(directory / 'prelaunch_recheck.json'),
             'postflight_status': postflight['status'],
             'postflight_sha256': sha(directory / 'postflight.json'),
             'payload_files_sha256': payload_hashes})
    print(json.dumps({'event': 'audited', 'group': group_key, 'arm': arm, 'status': 'PASS',
                      'duration_seconds': duration}), flush=True)


# ---------------------------------------------------------- aggregate ----
def aggregate(manifest_sha, review_sha, degraded_3arm=False, level=None):
    guard = verify(manifest_sha, review_sha)
    review = review_view(manifest_sha, review_sha)
    if degraded_3arm and not review['degraded_three_arm']:
        raise ValueError('three-arm degradation was not declared by the review')
    require_precheck(manifest_sha, review_sha)
    on_disk = {}
    for group_key in plan.GROUPS:
        receipt_path = RESULTS / group_key / 'completion.json'
        if not receipt_path.is_file():
            if degraded_3arm and plan.GROUP_ARM[group_key] == 'F':
                continue
            raise ValueError(f'missing group receipt: {group_key}')
        receipt = read(receipt_path)
        plan.validate_completion(receipt, group_key, manifest_sha, review_sha)
        if degraded_3arm and plan.GROUP_ARM[group_key] == 'F':
            raise ValueError('F groups exist; degraded aggregation is not applicable')
        if (receipt.get('precheck_sha256') != sha(RESULTS / 'precheck' / 'completion.json')
                or receipt.get('container_log_sha256') != sha(directory_of(group_key) / 'container.log')
                or receipt.get('inspect_sha256') != sha(directory_of(group_key) / 'inspect.json')
                or receipt.get('preflight_sha256') != sha(directory_of(group_key) / 'preflight.json')
                or receipt.get('prelaunch_recheck_sha256') != sha(directory_of(group_key) / 'prelaunch_recheck.json')
                or receipt.get('postflight_sha256') != sha(directory_of(group_key) / 'postflight.json')):
            raise ValueError(f'group receipt does not bind identical evidence: {group_key}')
        for name, digest in receipt['payload_files_sha256'].items():
            if sha(directory_of(group_key) / 'payload' / name) != digest:
                raise ValueError(f'payload evidence changed: {group_key} {name}')
        check_snapshot(guard, read(directory_of(group_key) / 'preflight.json')['first'], group_key)
        check_snapshot(guard, read(directory_of(group_key) / 'prelaunch_recheck.json'), group_key)
        on_disk[group_key] = receipt
    if level is None:
        # whole-campaign mode: every receipt must sit at one ladder level
        levels = {receipt['ladder_level'] for receipt in on_disk.values()}
        if len(levels) != 1:
            raise ValueError('receipts span several ladder levels; re-run with '
                             '--level (per-level audited rollup) for each level')
        level = levels.pop()
        receipts = on_disk
    else:
        # per-level mode (sparse-lane per-subgroup precedent): aggregate the
        # subset of groups whose receipts carry the requested level
        receipts = plan.select_level_receipts(on_disk, level, degraded_3arm)
    summary = plan.campaign_rollup(receipts, level, degraded_3arm)
    summary['probe_path'], summary['hopt_env'] = probe_path(manifest_sha, review_sha)
    if degraded_3arm:
        summary['degradation'] = 'F-arm restoration gate failed; three-arm campaign; ' \
                                 'explicit flag, never silent'
    summary['precheck_completion_sha256'] = sha(RESULTS / 'precheck' / 'completion.json')
    suffix = '' if level == 'L0' else f'_{level}'
    out = RESULTS / (('aggregate_3arm' if degraded_3arm else 'aggregate') + suffix + '.json')
    if out.exists():
        raise ValueError('refuse to overwrite an existing aggregate')
    publish(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['precheck_arm', 'precheck_audit', 'probe_start',
                                           'probe_audit', 'f_proof_start', 'f_proof_audit',
                                           'group_start', 'group_audit', 'aggregate'])
    parser.add_argument('--group', choices=list(plan.GROUPS))
    parser.add_argument('--level', choices=list(plan.LADDER), default=None)
    parser.add_argument('--degraded-3arm', action='store_true')
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--review-sha', required=True)
    args = parser.parse_args()
    if args.action in ('precheck_arm', 'precheck_audit'):
        (precheck_arm if args.action == 'precheck_arm' else precheck_audit)(
            args.manifest_sha, args.review_sha)
    elif args.action in ('probe_start', 'probe_audit'):
        (probe_start if args.action == 'probe_start' else probe_audit)(
            args.manifest_sha, args.review_sha)
    elif args.action in ('f_proof_start', 'f_proof_audit'):
        (f_proof_start if args.action == 'f_proof_start' else f_proof_audit)(
            args.manifest_sha, args.review_sha)
    elif args.action == 'aggregate':
        aggregate(args.manifest_sha, args.review_sha, args.degraded_3arm, args.level)
    else:
        if args.group is None:
            raise SystemExit('group actions require --group')
        (group_start if args.action == 'group_start' else group_audit)(
            args.group, args.manifest_sha, args.review_sha, args.level or 'L0')
