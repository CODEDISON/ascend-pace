"""Source-frozen launcher for the concern-#8 sparse-distribution lane.

One no-device precheck, then strictly sequential fresh containers for the
reviewed groups (plan_a: 6 groups; plan_b: 2x6 groups, only after an observed
SQ fail-closed rejection), each gated by two complete 16-die admission
preflights, with the pre-registered trimming ladder as an explicit, recorded
downgrade. No automatic retries, signals, container removal, resets or
destructive writes.

The independent-review expectation is defined ONCE in plan.py
(REVIEW_SCHEMA/REVIEW_STATUS_APPROVED/REVIEW_SCOPE/REVIEW_GROUPS/REVIEWED_FILES
via plan.validate_review); this launcher consumes exactly that shared constant
through the alias below, so the v3-class BLOCKER (launcher and reviewer
disagreeing on the review schema) cannot recur.
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
NAME = 'atlas_sparse_distribution_lane_20261105_v1'
SOURCE = WORK / 'experiments' / NAME
PLAN_SOURCE = WORK / 'experiments' / 'atlas_sparse_distribution_plan_20261105_v1'
RESULTS = WORK / 'results/diagnostic' / NAME
GUARD = WORK / 'experiments/revision2_group_admission_device_20261028_v3'
GUARD_LAUNCH_SHA = 'b4d96c96b5e304d8b0cf74c264d7b6006c158b018c7243bf055a74b79d91ffc2'
GUARD_MANIFEST_SHA = '288fb6913cab6181af5021505be0534b3105704c75358cb3252bf1cd31246d94'
BINARY = WORK / 'src/revision2_sdma_revalidation_20261028_v1'
SHMEM = WORK / 'src/shmem'
HELPER = WORK / 'experiments/revision2_sdma_queue_metadata_20261028_v1/metadata.py'
LARGE = WORK / 'experiments/revision2_sdma_large_correctness_20261028_v1'

FILES = plan.FILES
# The single review-contract consumer: identical function object as plan's.
verify_review_binding = plan.validate_review


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


def verify(manifest_sha, review_sha):
    if Path(__file__).resolve().parent != SOURCE or sha(SOURCE / 'source_manifest.json') != manifest_sha:
        raise ValueError('source identity mismatch')
    manifest = json.loads((SOURCE / 'source_manifest.json').read_text())
    if set(manifest) != FILES or any(sha(SOURCE / name) != digest for name, digest in manifest.items()):
        raise ValueError('source graph changed')
    if sha(Path(__file__)) != manifest['launch.py']:
        raise ValueError('launcher does not run from the frozen source graph')
    check_source_tree(SOURCE)
    if sha(SOURCE / 'review.json') != review_sha:
        raise ValueError('review mismatch')
    # The review is validated against the ONE shared contract constant.
    verify_review_binding(read(SOURCE / 'review.json'), manifest_sha)
    if sha(PLAN_SOURCE / 'cases.json') != plan.CASES_SHA256:
        raise ValueError('bound cases.json identity differs from the frozen plan')
    if sha(PLAN_SOURCE / 'PLAN.md') != plan.PLAN_MD_SHA256 \
            or sha(PLAN_SOURCE / 'gen_cases.py') != plan.GEN_CASES_SHA256:
        raise ValueError('frozen plan sources differ')
    plan.load_cases()  # full guard recomputation from the embedded matrices
    if sha(BINARY / 'cashflex/lib/libcashflex.so') != plan.LIB_SHA256 \
            or sha(SHMEM / 'build_cann9_vllm/lib/libshmem.so') != plan.SHMEM_SHA256:
        raise ValueError('runtime provider mismatch')
    if sha(HELPER) != plan.HELPER_SHA256 or sha(LARGE / 'layout.json') != plan.LAYOUT_SHA256:
        raise ValueError('frozen queue-metadata helper/layout mismatch')
    if sha(GUARD / 'launch.py') != GUARD_LAUNCH_SHA:
        raise ValueError('admission guard source mismatch')
    spec = importlib.util.spec_from_file_location('sparse_dist_admission_guard', GUARD / 'launch.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.manifest(GUARD, GUARD_MANIFEST_SHA, guard.FILES)
    # Preserve rejected process observations under this independent lane.
    guard.RESULTS = RESULTS
    return guard


def load_helper():
    if sha(HELPER) != plan.HELPER_SHA256:
        raise ValueError('queue-metadata helper differs')
    spec = importlib.util.spec_from_file_location('sparse_dist_audit_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source_tree(source):
    """Recursive source-tree coverage (barrier-v3 round-2 form).

    Only the frozen lane files, the manifest, the review and __pycache__
    subtrees are permitted; stray files, stray directories, symlinks and
    unexpected entry types fail closed so no intermediate version ships with
    the lane.
    """
    observed_files, observed_dirs = set(), set()
    for path in Path(source).rglob('*'):
        if '__pycache__' in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
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


def group_dir(group_key):
    variant = 'plan_a' if group_key in plan.GROUPS_PLAN_A else 'plan_b'
    return RESULTS / variant / group_key


def fixed_mounts():
    return [(Path('/usr/local/Ascend/driver'), '/usr/local/Ascend/driver', 'ro'),
            (Path('/etc/ascend_install.info'), '/etc/ascend_install.info', 'ro'),
            (BINARY, '/workspace/cashflex', 'ro'), (SHMEM, '/workspace/shmem', 'ro'),
            (HELPER.parent, '/workspace/queue_metadata', 'ro'),
            (LARGE, '/workspace/sdma_large', 'ro'),
            (SOURCE, '/workspace/lane', 'ro'), (PLAN_SOURCE, '/workspace/plan', 'ro')]


def expected_mounts(payload=None):
    mounts = {target: (str(host), mode == 'rw') for host, target, mode in fixed_mounts()}
    if payload is not None:
        mounts['/results'] = (str(payload), True)
    return mounts


def entry_script(entry):
    return ('set -euo pipefail; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
            'export PYTHONPATH=/workspace/lane:/workspace/plan:/workspace/cashflex:'
            '/workspace/queue_metadata:/workspace/sdma_large:${PYTHONPATH:-}; '
            'export LD_LIBRARY_PATH=/workspace/cashflex/cashflex/lib:'
            '/workspace/shmem/build_cann9_vllm/lib:${LD_LIBRARY_PATH:-}; '
            'exec python3 -B -u /workspace/lane/' + entry)


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
        raise RuntimeError('read-only mounts differ')
    if info['Config']['Cmd'] != argv[-3:]:
        raise RuntimeError('command differs')


def precheck_env():
    return {'REVSD_CASES_PATH': '/workspace/plan/cases.json',
            'REVSD_CASES_SHA256': plan.CASES_SHA256,
            'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}


def precheck_arm(manifest_sha, review_sha):
    verify(manifest_sha, review_sha)
    directory = RESULTS / 'precheck'
    if directory.exists() or directory.is_symlink():
        raise ValueError('refuse to replace the no-device precheck')
    directory.mkdir(parents=True, exist_ok=False)
    env = precheck_env()
    args = runtime_args(NAME + '-precheck', 'none', 'private', env, 'precheck.py')
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
    guard = verify(manifest_sha, review_sha)
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
            or record.get('cases_sha256') != plan.CASES_SHA256
            or record.get('cases_count') != 17
            or record.get('guards_recomputed_from_matrices') is not True
            or record.get('workload_sha256') != plan.WORKLOADS_SHA256
            or record.get('metadata_helper_sha256') != plan.HELPER_SHA256
            or record.get('layout_sha256') != plan.LAYOUT_SHA256
            or record.get('trim_ladder') != list(plan.LADDER)
            or record.get('groups') != {variant: list(plan.GROUPS[variant])
                                        for variant in plan.VARIANTS}
            or record.get('plan_b_subgroup_sizes') != {'a': 10, 'b': 7}):
        raise RuntimeError('precheck evidence missing or inconsistent')
    plan.check_pace_schema(record['pace_schema'])
    if record['review_contract'] != {'schema': plan.REVIEW_SCHEMA, 'scope': plan.REVIEW_SCOPE,
                                     'groups': plan.REVIEW_GROUPS,
                                     'reviewed_files': sorted(plan.REVIEWED_FILES)}:
        raise RuntimeError('precheck review contract drifts from the shared constant')
    estimate = plan.load_cases()['planning_estimate']
    if record.get('planning_estimate_crosscheck') != estimate:
        raise RuntimeError('precheck planning-estimate cross check differs')
    for group_key in plan.CAMPAIGN_SLOTS:
        if record['ports'][group_key] != list(plan.ports(group_key)):
            raise RuntimeError('precheck port plan differs')
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


def prior_receipts(variant, group_key):
    """Read the prior groups' PASS receipts in campaign order."""
    receipts = []
    for prior in plan.GROUPS[variant][:plan.group_index(variant, group_key)]:
        receipts.append(read(group_dir(prior) / 'completion.json'))
    return receipts


def require_prior_groups(variant, group_key, manifest_sha, review_sha, drops):
    for prior in plan.GROUPS[variant][:plan.group_index(variant, group_key)]:
        receipt = read(group_dir(prior) / 'completion.json')
        plan.validate_completion(receipt, variant, prior, manifest_sha, review_sha)
        # Round-14: a preregistered device-limitation not-run receipt
        # satisfies the ordering gate (validate_completion already refused
        # it for non-registered groups or mismatched reasons).
        if receipt.get('status') == 'DEVICE_LIMITATION':
            continue
        if receipt.get('duration_seconds') <= 0:
            raise ValueError(f'earlier group did not pass: {prior}')
    receipts = prior_receipts(variant, group_key)
    return plan.trim_gate(receipts, drops)  # also re-validates the ladder request


def device_limitation_receipt(group_key, manifest_sha, review_sha):
    """Round-14: record a preregistered device-limitation not-run receipt for
    a registered group, binding the observed failure evidence. The receipt
    satisfies the strict ordering gate and the aggregate accepts the campaign
    with the limitation declared (5/6 groups analysed)."""
    verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    if group_key not in plan.DEVICE_LIMITATION_GROUPS:
        raise ValueError(f'{group_key} is not a registered device-limitation group '
                         f'({sorted(plan.DEVICE_LIMITATION_GROUPS)})')
    require_prior_groups('plan_a', group_key, manifest_sha, review_sha, 0)
    directory = group_dir(group_key)
    if directory.exists() or directory.is_symlink():
        raise ValueError('group already exists; no retries')
    directory.mkdir(parents=True, exist_ok=False)
    publish(directory / 'completion.json',
            {'schema': 'revision2.sparse-distribution-completion.v1',
             'status': 'DEVICE_LIMITATION', 'variant': 'plan_a', 'group': group_key,
             'drops': 0, 'world': plan.WORLD,
             'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
             'image_id': plan.IMAGE,
             'device_limitation_reason': plan.DEVICE_LIMITATION_REASONS[group_key],
             'device_limitation_statement': plan.device_limitation_statement(),
             'evidence': 'four reproducible exits (aicpu 507018) with the group '
                         'first case syn-diag-512; priming 1 and 5 real-anchor '
                         'calls both ineffective (rounds 9-13); barrier A1 '
                         'concurrent PASS; same case passes in job0-2 at later '
                         'positions',
             'scope': 'known device-level limitation, not run; successors '
                      'proceed; aggregate declares 5/6'})
    print(json.dumps({'event': 'device_limitation_receipt', 'group': group_key,
                      'reason': plan.DEVICE_LIMITATION_REASONS[group_key]}), flush=True)


def require_sq_fail_marker():
    marker = RESULTS / 'sq_gate_fail_closed.json'
    if not marker.is_file():
        raise ValueError('plan_b requires an observed plan_a SQ fail-closed rejection')
    evidence = read(marker)
    if evidence.get('status') != 'SQ_GATE_FAIL_CLOSED' or \
            not isinstance(evidence.get('rejecting_groups'), list) or \
            not evidence.get('rejecting_groups'):
        raise ValueError('SQ fail-closed marker does not bind evidence')
    for name, digest in evidence.get('evidence_sha256', {}).items():
        if sha(RESULTS / name) != digest:
            raise ValueError('SQ fail-closed marker evidence changed')


def snapshot(guard, group_key):
    """Two complete 16-die admission checks bound to this group's own ports."""
    shmem_port, master_port = plan.ports(group_key)
    guard.stage_info = lambda stage: ('native', plan.WORLD, (shmem_port, master_port))
    return guard.preflight('native-p16')


def check_snapshot(guard, snap, group_key):
    if snap.get('processes') != guard.parse_npu_info(snap['npu_smi_info']):
        raise ValueError('process table interpretation differs')
    shmem_port, master_port = plan.ports(group_key)
    if snap.get('ports') != [shmem_port, master_port] or \
            snap.get('devices') != list(guard.nodes(plan.WORLD)):
        raise ValueError('preflight port or device coverage differs')
    if set(snap['health']) != set(map(str, range(8))) or \
            set(snap['usage']) != set(map(str, range(8))):
        raise ValueError('health/usage card coverage differs')
    for card in range(8):
        guard.check_health(snap['health'][str(card)])
        guard.check_usage(snap['usage'][str(card)])


def group_env(group_key, drops, acl):
    shmem_port, master_port = plan.ports(group_key)
    variant = 'plan_a' if group_key in plan.GROUPS_PLAN_A else 'plan_b'
    return {'WORLD_SIZE': str(plan.WORLD), 'MASTER_ADDR': '127.0.0.1',
            'MASTER_PORT': str(master_port),
            'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(plan.WORLD))),
            'HPA_SHMEM_IP_PORT': f'tcp://127.0.0.1:{shmem_port}',
            'HPA_SHMEM_WORKSPACE_PLANES': '2',
            'REVSD_VARIANT': variant, 'REVSD_GROUP': group_key, 'REVSD_DROPS': str(drops),
            'REVSD_ACL_LIBRARY_PATH': acl['path'], 'REVSD_ACL_LIBRARY_SHA256': acl['sha256'],
            'REVSD_CASES_PATH': '/workspace/plan/cases.json',
            'REVSD_CASES_SHA256': plan.CASES_SHA256,
            'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'}


def group_start(variant, group_key, drops, manifest_sha, review_sha):
    if variant not in plan.VARIANTS or group_key not in plan.GROUPS[variant]:
        raise ValueError('group outside the reviewed campaign')
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    if variant == 'plan_b':
        require_sq_fail_marker()
    trigger = require_prior_groups(variant, group_key, manifest_sha, review_sha, drops)
    directory = group_dir(group_key)
    if directory.exists() or directory.is_symlink():
        raise ValueError('group already exists; no retries')
    acl = require_precheck(manifest_sha, review_sha)['acl']
    first = snapshot(guard, group_key)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'payload').mkdir()
    env = group_env(group_key, drops, acl)
    name = f'{NAME}-{group_key}'
    args = runtime_args(name, 'host', 'host', env, 'supervisor.py', guard,
                        plan.WORLD, directory / 'payload')
    publish(directory / 'preflight.json',
            {'source_manifest_sha256': manifest_sha, 'review_sha256': review_sha,
             'variant': variant, 'group': group_key, 'drops': drops,
             'drop_trigger': trigger, 'first': first, 'docker_argv': args, 'env': env,
             'scope': 'sparse_distribution_lane_' + variant})
    publish(directory / 'prelaunch_recheck.json', snapshot(guard, group_key))
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown container id: inspect, no automatic retry')
    publish(directory / 'container.json', {'id': container, 'name': name})
    print(json.dumps({'event': 'started', 'variant': variant, 'group': group_key,
                      'drops': drops, 'id': container}), flush=True)


def json_rows(logs):
    """Robust stdout record scan (round-5 review defect).

    Unbuffered writers emit print() as separate text+newline writes; docker
    stream reassembly concatenates them into STRUCTURALLY MERGED lines, so
    line-split + json.loads loses whole records (real job0 evidence: 82 of
    272 case_samples lost). raw_decode streams objects wherever they start,
    immune to merging; only genuinely corrupt bytes are skipped.
    """
    decoder = json.JSONDecoder()
    rows = []
    index, size = 0, len(logs)
    while index < size:
        if logs[index] != '{':
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(logs, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        index = end
    return rows


def load_payload_case_records(directory, variant, group_key, drops):
    """The authoritative case-record set: every exclusively-created per-rank
    payload file for this group's rotated plan, in canonical order."""
    records = []
    for rank in range(plan.WORLD):
        for canonical, _case_id in plan.rotated_case_order(variant, group_key, drops):
            path = directory / 'payload' / f'rank-{rank:02d}' / f'case-{canonical:03d}.json'
            if not path.is_file():
                raise RuntimeError(f'payload case evidence absent: rank {rank} '
                                   f'case {canonical}')
            records.append(read(path))
    expected = plan.WORLD * len(plan.rotated_case_order(variant, group_key, drops))
    if len(records) != expected:
        raise RuntimeError('payload case coverage differs')
    return records


def cross_check_stdout_records(stdout_records, payload_records):
    """Inclusive stdout cross-check: every record recoverable from the
    merge-corrupted stdout must equal its payload twin; stdout absence is
    tolerated (payload owns coverage), any difference fails closed."""
    payload_by_key = {(record['rank'], record['case_index']): record
                      for record in payload_records}
    for record in stdout_records:
        stored = payload_by_key.get((record.get('rank'), record.get('case_index')))
        if stored is None or stored != record:
            raise ValueError('stdout case record differs from the payload twin: '
                             f"rank {record.get('rank')} case {record.get('case_index')}")


def require_natural_exits(rows):
    exits = [x for x in rows if x.get('event') == 'natural_exit']
    if len(exits) != plan.WORLD or {x.get('rank') for x in exits} != set(range(plan.WORLD)) \
            or any(x.get('code') != 0 for x in exits):
        raise RuntimeError('missing or failed natural rank exits')
    if any(x.get('event') == 'start_failed' for x in rows) \
            or not any(x.get('event') == 'all_natural_exit' for x in rows):
        raise RuntimeError('supervisor did not report a complete natural group')


def detect_sq_gate_rejection(directory, rows, exits):
    """If every rank rejected the SQ budget before payload, preserve the
    fail-closed evidence and arm the plan_b hand-off marker."""
    payload = directory / 'payload'
    rejects = sorted(path.parent.name for path in payload.rglob('sq_gate_reject.json'))
    cases = list(payload.rglob('case-*.json'))
    corroborated = any(x.get('event') == 'natural_exit' for x in rows)
    if rejects and not cases and exits and corroborated \
            and all(x.get('code') != 0 for x in exits):
        marker = RESULTS / 'sq_gate_fail_closed.json'
        if marker.exists():
            raise RuntimeError('SQ fail-closed marker already exists; no overwrite')
        evidence = {}
        for group_key in plan.GROUPS_PLAN_A:
            root = group_dir(group_key)
            for name in ('container.log', 'inspect.json', 'preflight.json',
                         'prelaunch_recheck.json'):
                target = root / name
                if target.is_file():
                    evidence[f'{group_key}/{name}'] = sha(target)
        publish(marker, {'status': 'SQ_GATE_FAIL_CLOSED',
                         'rejecting_ranks': rejects,
                         'rejecting_groups': [directory.name],
                         'evidence_sha256': evidence,
                         'scope': 'plan_b may only start after this observed rejection'})
        return True
    return False


def verify_payload_evidence(directory, variant, group_key, drops):
    helper = load_helper()
    layout = helper.load_layout(LARGE / 'layout.json', plan.LAYOUT_SHA256)
    payload = directory / 'payload'
    case_order = plan.rotated_case_order(variant, group_key, drops)
    names = set()
    for rank in range(plan.WORLD):
        root = payload / f'rank-{rank:02d}'
        for name in ('admission.json', 'queue_metadata.json', 'budget.json',
                     'cold_start_priming.json'):
            if not (root / name).is_file():
                raise RuntimeError(f'missing payload evidence: rank {rank} {name}')
            names.add(f'rank-{rank:02d}/{name}')
        for canonical, _ in case_order:
            name = f'rank-{rank:02d}/case-{canonical:03d}.json'
            if not (payload / name).is_file():
                raise RuntimeError(f'missing case evidence: {name}')
            names.add(name)
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
        case_ids = [case_id for _, case_id in case_order]
        budget = plan.plan_call_budgets(case_ids, rank, helper, channels, priming=True)
        if json.loads((root / 'budget.json').read_text()) != budget:
            raise RuntimeError(f'planned budget differs: rank {rank}')
        admission = json.loads((root / 'admission.json').read_text())
        if (admission.get('variant'), admission.get('group'), admission.get('rank'),
                admission.get('world'), admission.get('drops'),
                admission.get('heap_bytes'), admission.get('slot_bytes'),
                admission.get('cases'), admission.get('cases_sha256')) != (
                variant, group_key, rank, plan.WORLD, drops,
                plan.HEAP_BYTES, plan.SLOT_BYTES, len(case_ids), plan.CASES_SHA256):
            raise RuntimeError(f'admission identity differs: rank {rank}')
        # Round-12: the priming evidence file is required per rank, hashed
        # with the same treatment as every payload file, and identity-bound
        # (event/rank/case/calls must match the lane contract).
        priming = json.loads((root / 'cold_start_priming.json').read_text())
        if (priming.get('event'), priming.get('rank'), priming.get('world'),
                priming.get('priming_case'), priming.get('priming_calls'),
                priming.get('timed'), priming.get('validated')) != (
                'cold_start_priming', rank, plan.WORLD, plan.PRIMING_CASE_ID,
                plan.PRIMING_CALLS, False, False):
            raise RuntimeError(f'cold-start priming evidence differs: rank {rank}')
    if {str(path.relative_to(payload)) for path in payload.rglob('*') if path.is_file()} != names:
        raise RuntimeError('payload file coverage differs')
    return {name: sha(payload / name) for name in sorted(names)}


def group_audit(variant, group_key, manifest_sha, review_sha):
    if variant not in plan.VARIANTS or group_key not in plan.GROUPS[variant]:
        raise ValueError('group outside the reviewed campaign')
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    directory = group_dir(group_key)
    info = read(directory / 'container.json')
    inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
    if inspected['State'].get('Running'):
        print(json.dumps({'event': 'still_running', 'variant': variant, 'group': group_key}), flush=True)
        return
    logs = subprocess.run(['docker', 'logs', info['id']], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=30).stdout
    publish(directory / 'inspect.json', inspected)
    with (directory / 'container.log').open('x') as out:
        out.write(logs)
    before = read(directory / 'preflight.json')
    drops = before.get('drops')
    inspect_contract(inspected, before['docker_argv'], before['env'], guard,
                     plan.WORLD, directory / 'payload')
    rows = json_rows(logs)
    exits = [x for x in rows if x.get('event') == 'natural_exit']
    if detect_sq_gate_rejection(directory, rows, exits):
        raise RuntimeError('SQ budget gate rejected the planned sequence; plan_b '
                           'hand-off marker written; this group produced no completion')
    require_natural_exits(rows)
    priming_events = [x for x in rows if x.get('event') == 'cold_start_priming']
    if not priming_events:
        raise RuntimeError('cold-start priming event absent from stdout '
                           '(round-9 job3 mitigation must run before any case)')
    # Round-5 fix: the AUDIT IS PAYLOAD-DRIVEN. The per-rank case files are
    # exclusively created and hash-bound, so coverage completeness is
    # guaranteed by the payload; stdout is an inclusive cross-check — every
    # record recoverable from the (merge-corrupted) stdout must equal its
    # payload twin byte-for-byte, but stdout can never again veto a complete
    # payload set ('stdout case coverage differs' on merged lines).
    records = load_payload_case_records(directory, variant, group_key, drops)
    view = plan.validate_group_records(records, variant, group_key, drops)
    stdout_records = [x for x in rows if x.get('event') == 'case_samples']
    cross_check_stdout_records(stdout_records, records)
    passes = [x for x in rows if x.get('event') == 'rank_pass']
    if len(passes) != plan.WORLD or {x.get('rank') for x in passes} != set(range(plan.WORLD)):
        raise RuntimeError('rank pass coverage differs')
    metrics = plan.assemble_group_metrics(view, group_key)
    payload_hashes = verify_payload_evidence(directory, variant, group_key, drops)
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
            {'schema': 'revision2.sparse-distribution-completion.v1', 'status': 'PASS',
             'variant': variant, 'group': group_key, 'drops': drops,
             'drop_trigger': before.get('drop_trigger'), 'world': plan.WORLD,
             'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
             'image_id': plan.IMAGE, 'duration_seconds': duration,
             'scope': 'sparse distribution lane, fresh group, dual-clock timing; '
                      'descriptive only',
             'metrics': metrics,
             'precheck_sha256': sha(RESULTS / 'precheck' / 'completion.json'),
             'container_log_sha256': sha(directory / 'container.log'),
             'inspect_sha256': sha(directory / 'inspect.json'),
             'preflight_sha256': sha(directory / 'preflight.json'),
             'prelaunch_recheck_sha256': sha(directory / 'prelaunch_recheck.json'),
             'postflight_status': postflight['status'],
             'postflight_sha256': sha(directory / 'postflight.json'),
             'payload_files_sha256': payload_hashes})
    print(json.dumps({'event': 'audited', 'variant': variant, 'group': group_key,
                      'status': 'PASS', 'duration_seconds': duration}), flush=True)


def aggregate(variant, subgroup, manifest_sha, review_sha):
    guard = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    groups = [key for key in plan.GROUPS[variant]
              if subgroup is None or plan.subgroup_of(key) == subgroup]
    if subgroup is not None and not groups:
        raise ValueError('subgroup outside this variant')
    receipts = {}
    for group_key in groups:
        directory = group_dir(group_key)
        receipt = read(directory / 'completion.json')
        plan.validate_completion(receipt, variant, group_key, manifest_sha, review_sha)
        # Round-14: device-limitation receipts skip evidence rehash (no
        # container ran); the aggregate carries them as declared limitations.
        if receipt.get('status') == 'DEVICE_LIMITATION':
            receipts[group_key] = receipt
            continue
        # Rehash EVERY evidence artifact, including the precheck receipt and
        # both preflight snapshots (v3 review advisory lesson).
        if (receipt.get('precheck_sha256') != sha(RESULTS / 'precheck' / 'completion.json')
                or receipt.get('container_log_sha256') != sha(directory / 'container.log')
                or receipt.get('inspect_sha256') != sha(directory / 'inspect.json')
                or receipt.get('preflight_sha256') != sha(directory / 'preflight.json')
                or receipt.get('prelaunch_recheck_sha256') != sha(directory / 'prelaunch_recheck.json')
                or receipt.get('postflight_sha256') != sha(directory / 'postflight.json')):
            raise ValueError(f'group receipt does not bind identical evidence: {group_key}')
        for name, digest in receipt['payload_files_sha256'].items():
            if sha(directory / 'payload' / name) != digest:
                raise ValueError(f'payload evidence changed: {group_key} {name}')
        check_snapshot(guard, read(directory / 'preflight.json')['first'], group_key)
        check_snapshot(guard, read(directory / 'prelaunch_recheck.json'), group_key)
        receipts[group_key] = receipt
    summary = plan.summary_from_receipts(receipts, variant, subgroup, manifest_sha, review_sha)
    summary['precheck_completion_sha256'] = sha(RESULTS / 'precheck' / 'completion.json')
    summary['historical_anchor_binding'] = {
        'forward_campaign': 'analysis_results/atlas_qwen_sparse_timing_device_20261029_v1',
        'reverse_campaign': 'analysis_results/atlas_qwen_sparse_reverse_timing_independent_20261030_v1',
        'note': plan.HISTORICAL_SCOPE}
    out = RESULTS / ('aggregate.json' if subgroup is None else f'aggregate_{subgroup}.json')
    if variant == 'plan_b':
        out = RESULTS / 'plan_b' / ('aggregate_a.json' if subgroup == 'a' else 'aggregate_b.json')
        out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('refuse to overwrite an existing aggregate')
    publish(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)


def dispatch(args):
    """Single CLI dispatch point; each action receives exactly the arguments
    its implementation signature declares (round-6 fix: main passed --drops
    unconditionally and crashed group_audit with a 5-vs-4 TypeError after a
    successful device run)."""
    if args.action in ('precheck_arm', 'precheck_audit'):
        (precheck_arm if args.action == 'precheck_arm' else precheck_audit)(
            args.manifest_sha, args.review_sha)
    elif args.action == 'device_limitation_receipt':
        if args.group is None:
            raise SystemExit('device_limitation_receipt requires --group')
        if args.variant != 'plan_a':
            raise SystemExit('device-limitation receipts exist only in plan_a')
        device_limitation_receipt(args.group, args.manifest_sha, args.review_sha)
    elif args.action == 'aggregate':
        if args.variant == 'plan_a' and args.subgroup is not None:
            raise SystemExit('plan_a has no subgroups')
        if args.variant == 'plan_b' and args.subgroup is None:
            raise SystemExit('plan_b aggregates per subgroup (results never pooled with plan_a)')
        aggregate(args.variant, args.subgroup, args.manifest_sha, args.review_sha)
    else:
        if args.group is None:
            raise SystemExit('group actions require --group')
        if args.group not in plan.GROUPS[args.variant]:
            raise SystemExit('group outside this variant')
        if args.action == 'group_start':
            group_start(args.variant, args.group, args.drops, args.manifest_sha, args.review_sha)
        else:
            group_audit(args.variant, args.group, args.manifest_sha, args.review_sha)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['precheck_arm', 'precheck_audit', 'group_start',
                                           'group_audit', 'device_limitation_receipt',
                                           'aggregate'])
    parser.add_argument('--variant', choices=list(plan.VARIANTS), default='plan_a')
    parser.add_argument('--group')
    parser.add_argument('--subgroup', choices=['a', 'b'])
    parser.add_argument('--drops', type=int, default=0)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--review-sha', required=True)
    return parser


if __name__ == '__main__':
    dispatch(build_parser().parse_args())
