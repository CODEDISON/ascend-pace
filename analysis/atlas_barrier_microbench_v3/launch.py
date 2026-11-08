"""Source-frozen v3 barrier/control launcher.

One no-device precheck, then strictly sequential fresh containers for the
reviewed groups of arms A/B/C on the fixed P16 stage, each gated by two
complete 16-die admission preflights. No automatic retries, signals,
container removal, resets or destructive writes.
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
NAME = 'revision2_barrier_microbench_20261105_v3'
SOURCE = WORK / 'experiments' / NAME
RESULTS = WORK / 'results/diagnostic' / NAME
GUARD = WORK / 'experiments/revision2_group_admission_device_20261028_v3'
GUARD_LAUNCH_SHA = 'b4d96c96b5e304d8b0cf74c264d7b6006c158b018c7243bf055a74b79d91ffc2'
GUARD_MANIFEST_SHA = '288fb6913cab6181af5021505be0534b3105704c75358cb3252bf1cd31246d94'
BINARY = WORK / 'src/revision2_sdma_revalidation_20261028_v1'
SHMEM = WORK / 'src/shmem'
HELPER = WORK / 'experiments/revision2_sdma_queue_metadata_20261028_v1/metadata.py'
LARGE = WORK / 'experiments/revision2_sdma_large_correctness_20261028_v1'
IMAGE = 'sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f'
FILES = {'DESIGN.md', 'plan.py', 'worker.py', 'supervisor.py', 'precheck.py', 'launch.py',
         'test_cpu.py', 'make_manifest.py', 'observed_npu_info_fixture.json'}
ARM_EVENT = {'A': 'arm_a_samples', 'B': 'arm_b_samples', 'C': 'arm_c_samples'}


def sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('expected ordinary file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = path.stat()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('file changed while hashing: ' + str(path))
    return digest.hexdigest()


def publish(path, value):
    with path.open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def command(args, timeout=30):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout


class LaneEnv:
    """Frozen bundle of device-host paths and pinned digests.

    The default is the real device lane; CPU tests inject a synthetic tree
    through the same attribute set so verify() runs its full logic locally.
    """

    def __init__(self, source, cashflex_lib, shmem_lib, helper, layout, guard_dir,
                 guard_launch_sha, guard_manifest_sha, lib_sha, shmem_sha,
                 helper_sha, layout_sha):
        self.source = source
        self.cashflex_lib = cashflex_lib
        self.shmem_lib = shmem_lib
        self.helper = helper
        self.layout = layout
        self.guard_dir = guard_dir
        self.guard_launch_sha = guard_launch_sha
        self.guard_manifest_sha = guard_manifest_sha
        self.lib_sha = lib_sha
        self.shmem_sha = shmem_sha
        self.helper_sha = helper_sha
        self.layout_sha = layout_sha


def device_env():
    return LaneEnv(SOURCE, BINARY / 'cashflex/lib/libcashflex.so',
                   SHMEM / 'build_cann9_vllm/lib/libshmem.so', HELPER,
                   LARGE / 'layout.json', GUARD, GUARD_LAUNCH_SHA, GUARD_MANIFEST_SHA,
                   plan.LIB_SHA256, plan.SHMEM_SHA256, plan.HELPER_SHA256, plan.LAYOUT_SHA256)


def verify(manifest_sha, review_sha, env=None):
    env = device_env() if env is None else env
    manifest = json.loads((env.source / 'source_manifest.json').read_text())
    if sha(env.source / 'source_manifest.json') != manifest_sha:
        raise ValueError('source identity mismatch')
    if set(manifest) != FILES or any(sha(env.source / name) != digest for name, digest in manifest.items()):
        raise ValueError('source graph changed')
    if sha(Path(__file__)) != manifest['launch.py']:
        raise ValueError('launcher does not run from the frozen source graph')
    # Recursive tree coverage: only the frozen files, the manifest, the review
    # and __pycache__ subtrees are permitted; stray files or subdirectories
    # fail closed so no intermediate version ships with the lane.
    observed_files, observed_dirs = set(), set()
    for path in env.source.rglob('*'):
        if '__pycache__' in path.parts:
            continue
        relative = path.relative_to(env.source).as_posix()
        if path.is_symlink():
            raise ValueError('source symlink forbidden: ' + relative)
        if path.is_dir():
            observed_dirs.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise ValueError('unexpected source tree entry: ' + relative)
    if observed_dirs or observed_files != FILES | {'source_manifest.json', 'review.json'}:
        raise ValueError('source tree coverage differs; remove stray intermediate files')
    if sha(env.source / 'review.json') != review_sha:
        raise ValueError('review mismatch')
    # The review binding is the frozen contract in plan.REVIEW_CONTRACT; the
    # launcher and the CPU tests consume the same single source of truth.
    plan.check_review_binding(json.loads((env.source / 'review.json').read_text()),
                              manifest_sha, FILES)
    if sha(env.cashflex_lib) != env.lib_sha or sha(env.shmem_lib) != env.shmem_sha:
        raise ValueError('runtime provider mismatch')
    if sha(env.helper) != env.helper_sha or sha(env.layout) != env.layout_sha:
        raise ValueError('frozen queue-metadata helper/layout mismatch')
    if sha(env.guard_dir / 'launch.py') != env.guard_launch_sha:
        raise ValueError('admission guard source mismatch')
    spec = importlib.util.spec_from_file_location('v3_admission_guard', env.guard_dir / 'launch.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.manifest(env.guard_dir, env.guard_manifest_sha, guard.FILES)
    # Preserve rejected process observations under this independent lane.
    guard.RESULTS = RESULTS
    return guard


def load_helper():
    if sha(HELPER) != plan.HELPER_SHA256:
        raise ValueError('queue-metadata helper differs')
    spec = importlib.util.spec_from_file_location('v3_audit_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_dir(arm, index):
    return RESULTS / f'{arm}{index}'


def fixed_mounts():
    return [(Path('/usr/local/Ascend/driver'), '/usr/local/Ascend/driver', 'ro'),
            (Path('/etc/ascend_install.info'), '/etc/ascend_install.info', 'ro'),
            (BINARY, '/workspace/cashflex', 'ro'), (SHMEM, '/workspace/shmem', 'ro'),
            (HELPER.parent, '/workspace/queue_metadata', 'ro'), (LARGE, '/workspace/sdma_large', 'ro'),
            (SOURCE, '/workspace/barrier', 'ro')]


def expected_mounts(payload=None):
    mounts = {target: (str(host), mode == 'rw') for host, target, mode in fixed_mounts()}
    if payload is not None:
        mounts['/results'] = (str(payload), True)
    return mounts


def runtime_args(name, network, ipc, env, entry, guard=None, world=0, payload=None):
    args = ['docker', 'run', '-d', '--name', name, '--network', network, '--ipc', ipc]
    if guard is not None:
        for dev in guard.nodes(world):
            args += ['--device', f'{dev}:{dev}:rwm']
    mounts = fixed_mounts() + ([(payload, '/results', 'rw')] if payload is not None else [])
    for host, target, mode in mounts:
        if not host.exists():
            raise ValueError('missing mount: ' + str(host))
        args += ['-v', f'{host}:{target}:{mode}']
    for key, value in env.items():
        args += ['-e', f'{key}={value}']
    args += [IMAGE, 'bash', '-lc', 'set -euo pipefail; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
             'export PYTHONPATH=/workspace/cashflex:/workspace/barrier:${PYTHONPATH:-}; '
             'export LD_LIBRARY_PATH=/workspace/cashflex/cashflex/lib:/workspace/shmem/build_cann9_vllm/lib:${LD_LIBRARY_PATH:-}; '
             'exec python3 -B -u /workspace/barrier/' + entry]
    return args


def inspect_contract(info, argv, env, guard=None, world=0, payload=None):
    host = info['HostConfig']
    expected_network = 'host' if guard else 'none'
    expected_ipc = 'host' if guard else 'private'
    if info['Image'] != IMAGE or host.get('NetworkMode') != expected_network or host.get('IpcMode') != expected_ipc:
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
        raise RuntimeError('read-only mounts differ')
    if info['Config']['Cmd'] != argv[-3:]:
        raise RuntimeError('command differs')


def precheck_arm(manifest_sha, review_sha):
    verify(manifest_sha, review_sha)
    directory = RESULTS / 'precheck'
    directory.mkdir(parents=True, exist_ok=False)
    env = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
    args = runtime_args(NAME + '-precheck', 'none', 'private', env, 'precheck.py')
    publish(directory / 'inputs.json', {'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
                                        'docker_argv': args, 'env': env})
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown precheck container id')
    publish(directory / 'container.json', {'id': container})
    print(json.dumps({'event': 'precheck_started', 'id': container}), flush=True)


def precheck_audit(manifest_sha, review_sha):
    verify(manifest_sha, review_sha)
    directory = RESULTS / 'precheck'
    inputs = json.loads((directory / 'inputs.json').read_text())
    info = json.loads((directory / 'container.json').read_text())
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
    found = []
    for line in logs.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('event') == 'NO_DEVICE_PRECHECK_PASS':
            found.append(value)
    if len(found) != 1 or found[0].get('provider_sha256') != plan.SHMEM_SHA256 \
            or 'ctypes.c_void_p(self.npu_stream)' not in found[0].get('stream_parameter_source', '') \
            or found[0].get('metadata_helper_sha256') != plan.HELPER_SHA256 \
            or found[0].get('layout_sha256') != plan.LAYOUT_SHA256 \
            or found[0].get('arm_b_rounds') != [plan.arm_pace_calls('B', i) for i in range(1, plan.ARM_GROUPS['B'] + 1)] \
            or found[0].get('arm_c_iterations') != [plan.arm_pace_calls('C', i) for i in range(1, plan.ARM_GROUPS['C'] + 1)]:
        raise RuntimeError('precheck evidence missing or inconsistent')
    plan.check_pace_schema(found[0]['pace_schema'])
    if not re.fullmatch('[0-9a-f]{64}', found[0].get('acl_library_sha256', '')):
        raise RuntimeError('ACL provider identity missing')
    publish(directory / 'completion.json', {'status': 'PASS', 'manifest_sha256': manifest_sha,
            'review_sha256': review_sha, 'image_id': IMAGE, 'record': found[0],
            'acl': {'path': found[0]['acl_library_path'], 'sha256': found[0]['acl_library_sha256']},
            'log_sha256': sha(directory / 'container.log'), 'inspect_sha256': sha(directory / 'inspect.json')})
    print(json.dumps({'event': 'precheck_pass'}), flush=True)


def require_precheck(manifest_sha, review_sha):
    receipt = json.loads((RESULTS / 'precheck' / 'completion.json').read_text())
    if (receipt.get('status') != 'PASS' or receipt.get('manifest_sha256') != manifest_sha
            or receipt.get('review_sha256') != review_sha or receipt.get('image_id') != IMAGE
            or receipt.get('log_sha256') != sha(RESULTS / 'precheck' / 'container.log')
            or receipt.get('inspect_sha256') != sha(RESULTS / 'precheck' / 'inspect.json')):
        raise ValueError('matched no-device precheck required')
    return receipt


def require_prior_groups(arm, index, manifest_sha, review_sha):
    for prior_arm, prior_index in plan.GROUP_ORDER:
        if (prior_arm, prior_index) == (arm, index):
            break
        directory = group_dir(prior_arm, prior_index)
        receipt = json.loads((directory / 'completion.json').read_text())
        if (receipt.get('status') != 'PASS' or receipt.get('manifest_sha256') != manifest_sha
                or receipt.get('review_sha256') != review_sha or receipt.get('image_id') != IMAGE
                or receipt.get('arm') != prior_arm or receipt.get('group') != prior_index):
            raise ValueError('earlier group did not pass with identical sources/reviews: '
                             f'{prior_arm}{prior_index}')


def snapshot(guard, arm, index):
    """Two complete 16-die admission checks bound to this group's own ports."""
    shmem_port, master_port = plan.ports(arm, index)
    guard.stage_info = lambda stage: ('native', plan.WORLD, (shmem_port, master_port))
    return guard.preflight('native-p16')


def group_start(arm, index, manifest_sha, review_sha):
    if arm not in plan.ARMS or (arm, index) not in plan.GROUP_ORDER:
        raise ValueError('group outside the reviewed plan')
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    require_prior_groups(arm, index, manifest_sha, review_sha)
    directory = group_dir(arm, index)
    if directory.exists() or directory.is_symlink():
        raise ValueError('group already exists; no retries')
    acl = require_precheck(manifest_sha, review_sha)['acl']
    first = snapshot(guard, arm, index)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'payload').mkdir()
    shmem_port, master_port = plan.ports(arm, index)
    env = {'WORLD_SIZE': str(plan.WORLD), 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(master_port),
           'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(plan.WORLD))),
           'HPA_SHMEM_IP_PORT': f'tcp://127.0.0.1:{shmem_port}', 'HPA_SHMEM_WORKSPACE_PLANES': '2',
           'REV3_ARM': arm, 'REV3_GROUP_INDEX': str(index),
           'REV3_ACL_LIBRARY_PATH': acl['path'], 'REV3_ACL_LIBRARY_SHA256': acl['sha256'],
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1'}
    name = f'{NAME}-{arm}-g{index}'
    args = runtime_args(name, 'host', 'host', env, 'supervisor.py', guard, plan.WORLD, directory / 'payload')
    publish(directory / 'preflight.json', {'source_manifest_sha256': manifest_sha, 'review_sha256': review_sha,
            'arm': arm, 'group': index, 'first': first, 'docker_argv': args, 'env': env,
            'scope': 'barrier_v3_' + arm})
    publish(directory / 'prelaunch_recheck.json', snapshot(guard, arm, index))
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown container id: inspect, no automatic retry')
    publish(directory / 'container.json', {'id': container, 'name': name})
    print(json.dumps({'event': 'started', 'arm': arm, 'group': index, 'id': container}), flush=True)


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


def require_natural_exits(rows):
    exits = [x for x in rows if x.get('event') == 'natural_exit']
    if len(exits) != plan.WORLD or {x.get('rank') for x in exits} != set(range(plan.WORLD)) \
            or any(x.get('code') != 0 for x in exits):
        raise RuntimeError('missing or failed natural rank exits')
    if any(x.get('event') in ('start_failed',) for x in rows) \
            or not any(x.get('event') == 'all_natural_exit' for x in rows):
        raise RuntimeError('supervisor did not report a complete natural group')


def verify_payload_evidence(directory, arm, index):
    helper = load_helper()
    layout = helper.load_layout(LARGE / 'layout.json', plan.LAYOUT_SHA256)
    payload = directory / 'payload'
    names = set()
    for rank in range(plan.WORLD):
        root = payload / f'rank-{rank:02d}'
        for name in ('admission.json', 'queue_metadata.json', 'budget.json'):
            if not (root / name).is_file():
                raise RuntimeError(f'missing payload evidence: rank {rank} {name}')
            names.add(f'rank-{rank:02d}/{name}')
        queues = json.loads((root / 'queue_metadata.json').read_text())
        if (queues.get('schema') != 'revision2.sdma-queue-metadata.v1'
                or queues.get('layout_sha256') != plan.LAYOUT_SHA256
                or queues.get('libshmem_sha256') != helper.LIBSHMEM_SHA256
                or queues.get('quiescent_asserted_by_caller') is not True
                or queues.get('state', {}).get('mype') != rank
                or queues.get('state', {}).get('npes') != plan.WORLD):
            raise RuntimeError(f'queue metadata identity differs: rank {rank}')
        raw = bytes.fromhex(queues['workspace_hex'])
        channels = helper.parse_workspace(raw, layout, plan.WORLD, rank)
        if channels != queues.get('channels') \
                or hashlib.sha256(raw).hexdigest() != queues.get('workspace_sha256'):
            raise RuntimeError(f'raw queue metadata differs: rank {rank}')
        budget = plan.plan_call_budgets(arm, index, rank, plan.arm_peer_bytes(arm, rank),
                                        helper, channels)
        if json.loads((root / 'budget.json').read_text()) != budget:
            raise RuntimeError(f'planned budget differs: rank {rank}')
    if {str(path.relative_to(payload)) for path in payload.rglob('*') if path.is_file()} != names:
        raise RuntimeError('payload file coverage differs')
    return {name: sha(payload / name) for name in sorted(names)}


def group_audit(arm, index, manifest_sha, review_sha):
    if arm not in plan.ARMS or (arm, index) not in plan.GROUP_ORDER:
        raise ValueError('group outside the reviewed plan')
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    directory = group_dir(arm, index)
    info = json.loads((directory / 'container.json').read_text())
    inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
    if inspected['State'].get('Running'):
        print(json.dumps({'event': 'still_running', 'arm': arm, 'group': index}), flush=True)
        return
    logs = subprocess.run(['docker', 'logs', info['id']], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=30).stdout
    publish(directory / 'inspect.json', inspected)
    with (directory / 'container.log').open('x') as out:
        out.write(logs)
    before = json.loads((directory / 'preflight.json').read_text())
    recheck = json.loads((directory / 'prelaunch_recheck.json').read_text())
    # Re-parse both persisted 16-die admission snapshots post-hoc so the two
    # preflights receive the same fail-closed treatment as the other evidence.
    guard.parse_npu_info(before['first']['npu_smi_info'])
    guard.parse_npu_info(recheck['npu_smi_info'])
    if (before.get('source_manifest_sha256'), before.get('review_sha256'),
            before.get('arm'), before.get('group')) != (manifest_sha, review_sha, arm, index):
        raise RuntimeError('group preflight provenance differs')
    inspect_contract(inspected, before['docker_argv'], before['env'], guard, plan.WORLD, directory / 'payload')
    rows = json_rows(logs)
    require_natural_exits(rows)
    records = [x for x in rows if x.get('event') == ARM_EVENT[arm]]
    if len(records) != plan.WORLD or {x.get('rank') for x in records} != set(range(plan.WORLD)):
        raise RuntimeError('missing rank sample records')
    if arm == 'A':
        by_rank = plan.validate_arm_a_records(records, index)
        metrics = plan.arm_a_metrics(by_rank)
    elif arm == 'B':
        by_rank = plan.validate_arm_b_records(records, index)
        metrics = plan.arm_b_metrics(by_rank, index)
    else:
        by_rank = plan.validate_arm_c_records(records, index)
        metrics = plan.arm_c_metrics(by_rank)
    payload_hashes = verify_payload_evidence(directory, arm, index)
    # Observe but do not interfere with other users after the experiment.
    postflight = {'scope': 'read-only snapshot; not an exclusive reservation'}
    try:
        postflight['npu_smi_info'] = guard.command(['npu-smi', 'info'], timeout=15)
        postflight['parsed'] = guard.parse_npu_info(postflight['npu_smi_info'])
        postflight['status'] = 'OBSERVED_EMPTY_TARGET'
    except Exception as exc:
        postflight['status'] = 'UNCONFIRMED'
        postflight['error'] = type(exc).__name__ + ': ' + str(exc)
    publish(directory / 'postflight.json', postflight)
    publish(directory / 'completion.json', {'status': 'PASS', 'arm': arm, 'group': index,
            'world': plan.WORLD, 'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
            'scope': 'barrier_v3_arm_' + arm + '_fresh_group', 'metrics': metrics, 'image_id': IMAGE,
            'precheck_sha256': sha(RESULTS / 'precheck' / 'completion.json'),
            'container_log_sha256': sha(directory / 'container.log'),
            'inspect_sha256': sha(directory / 'inspect.json'),
            'preflight_sha256': sha(directory / 'preflight.json'),
            'prelaunch_recheck_sha256': sha(directory / 'prelaunch_recheck.json'),
            'payload_files_sha256': payload_hashes,
            'postflight_status': postflight['status'], 'postflight_sha256': sha(directory / 'postflight.json')})
    print(json.dumps({'event': 'audited', 'arm': arm, 'group': index, 'status': 'PASS'}), flush=True)


def aggregate(manifest_sha, review_sha):
    guard = verify(manifest_sha, review_sha)
    precheck_sha = sha(RESULTS / 'precheck' / 'completion.json')
    groups = {}
    for arm, index in plan.GROUP_ORDER:
        directory = group_dir(arm, index)
        receipt = json.loads((directory / 'completion.json').read_text())
        if (receipt.get('status') != 'PASS' or receipt.get('manifest_sha256') != manifest_sha
                or receipt.get('review_sha256') != review_sha or receipt.get('arm') != arm
                or receipt.get('group') != index or receipt.get('image_id') != IMAGE
                or receipt.get('container_log_sha256') != sha(directory / 'container.log')
                or receipt.get('inspect_sha256') != sha(directory / 'inspect.json')
                or receipt.get('postflight_sha256') != sha(directory / 'postflight.json')
                or receipt.get('precheck_sha256') != precheck_sha
                or receipt.get('preflight_sha256') != sha(directory / 'preflight.json')
                or receipt.get('prelaunch_recheck_sha256') != sha(directory / 'prelaunch_recheck.json')):
            raise ValueError(f'group receipt does not bind identical evidence: {arm}{index}')
        preflight = json.loads((directory / 'preflight.json').read_text())
        recheck = json.loads((directory / 'prelaunch_recheck.json').read_text())
        guard.parse_npu_info(preflight['first']['npu_smi_info'])
        guard.parse_npu_info(recheck['npu_smi_info'])
        for name, digest in receipt['payload_files_sha256'].items():
            if sha(directory / 'payload' / name) != digest:
                raise ValueError(f'payload evidence changed: {arm}{index} {name}')
        groups[(arm, index)] = receipt['metrics']
    arm_a = {'per_group_median_max_rank_us': [groups[('A', i)]['median_max_rank_us']
                                              for i in range(1, plan.ARM_GROUPS['A'] + 1)],
             'per_group_median_envelope_us': [groups[('A', i)]['median_envelope_us']
                                              for i in range(1, plan.ARM_GROUPS['A'] + 1)],
             'pooled_max_rank_us': []}
    for i in range(1, plan.ARM_GROUPS['A'] + 1):
        arm_a['pooled_max_rank_us'] += groups[('A', i)]['max_rank_us']
    arm_a['median_max_rank_summary_us'] = plan.summary(arm_a['per_group_median_max_rank_us'])
    arm_a['median_envelope_summary_us'] = plan.summary(arm_a['per_group_median_envelope_us'])
    arm_a['pooled_max_rank_summary_us'] = plan.summary(arm_a['pooled_max_rank_us'])
    arm_b = {'per_group_increment_us': [groups[('B', i)]['increment_median_us']
                                        for i in range(1, plan.ARM_GROUPS['B'] + 1)],
             'per_group_b1_median_us': [], 'per_group_b2_median_us': []}
    for i in range(1, plan.ARM_GROUPS['B'] + 1):
        arm_b['per_group_b1_median_us'].append(groups[('B', i)]['b1_median_us'])
        arm_b['per_group_b2_median_us'].append(groups[('B', i)]['b2_median_us'])
    arm_b['increment_summary_us'] = plan.summary(arm_b['per_group_increment_us'])
    arm_b['b1_median_summary_us'] = plan.summary(arm_b['per_group_b1_median_us'])
    arm_b['b2_median_summary_us'] = plan.summary(arm_b['per_group_b2_median_us'])
    arm_c = {'per_group_slope_k1_us': [], 'per_group_slope_k2_us': []}
    for k in plan.C_K_VALUES:
        arm_c[f'per_group_k{k}_median_us'] = []
    for i in range(1, plan.ARM_GROUPS['C'] + 1):
        metrics = groups[('C', i)]
        arm_c['per_group_slope_k1_us'].append(metrics['slope_k1_minus_k0_us'])
        arm_c['per_group_slope_k2_us'].append(metrics['slope_k2_minus_k0_over_2_us'])
        for k in plan.C_K_VALUES:
            arm_c[f'per_group_k{k}_median_us'].append(metrics['k_median_us'][str(k)])
    arm_c['slope_k1_summary_us'] = plan.summary(arm_c['per_group_slope_k1_us'])
    arm_c['slope_k2_summary_us'] = plan.summary(arm_c['per_group_slope_k2_us'])
    for k in plan.C_K_VALUES:
        arm_c[f'k{k}_median_summary_us'] = plan.summary(arm_c[f'per_group_k{k}_median_us'])
    value = {'schema': 'revision2.barrier-v3-aggregate.v1', 'status': 'COMPLETE',
             'manifest_sha256': manifest_sha, 'review_sha256': review_sha, 'image_id': IMAGE,
             'groups': {f'{arm}{index}': {'arm': arm, 'group': index} for arm, index in plan.GROUP_ORDER},
             'arms': {'A': arm_a, 'B': arm_b, 'C': arm_c},
             'historical_reference': {
                 'v2_single_group_p16_median_max_rank_us': 89.860,
                 'reverse_order_lane_pace_case038': 'same frozen shape as arm C k=0 baseline'},
             'scope': 'descriptive multi-group summaries; no confidence intervals or causal claims'}
    publish(RESULTS / 'aggregate.json', value)
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['precheck_arm', 'precheck_audit', 'group_start',
                                           'group_audit', 'aggregate'])
    parser.add_argument('--arm', choices=list(plan.ARMS))
    parser.add_argument('--index', type=int)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--review-sha', required=True)
    args = parser.parse_args()
    if args.action in ('precheck_arm', 'precheck_audit'):
        (precheck_arm if args.action == 'precheck_arm' else precheck_audit)(args.manifest_sha, args.review_sha)
    elif args.action == 'aggregate':
        aggregate(args.manifest_sha, args.review_sha)
    else:
        if args.arm is None or args.index is None:
            raise SystemExit('group actions require --arm and --index')
        (group_start if args.action == 'group_start' else group_audit)(
            args.arm, args.index, args.manifest_sha, args.review_sha)
