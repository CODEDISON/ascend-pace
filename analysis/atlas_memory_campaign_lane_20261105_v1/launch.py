"""Source-frozen launcher for the #11/#12 memory campaign (round-2 rewiring).

One no-device precheck, then strictly sequential fresh containers for the
reviewed jobs (45 slots; pilots quarantined; C3BD gated on per-peer guard
semantics from the review with an explicit recorded projection-only path so a
legitimately-not-run conditional job neither blocks successors nor poisons
the aggregate; C7 gated on an A6-feasible receipt), each gated by two
complete 16-die admission preflights.

Round-2 wiring (review B1..B11): audit CLI signature fixed; conditional
jobs route run/projection-only instead of refusing everything; the A6 KV
floor is lifted into receipts and consumed from A5's measured T1 free;
engine jobs mount model/legacy/sync-model/reference; PACE heap/slot are
plumbed through HPA_SHMEM_HEAP_BYTES/HPA_SHMEM_SLOT_BYTES and recorded as
actually-configured evidence; the L1 host poller runs as a blocking
`job_l1` action whose stream is required and validated at audit; capacity
cells are DERIVED from prior receipts by plan.next_capacity_cell (free CLI
cells only for confirm slots); the audit validates fencepost sequences,
computes L1/L2 reconciliation, lifts the KV streak from the container log
and the engine-reported block count is required evidence; aggregate wires
c_star/peak_alignment/summary and rehashes everything.

The independent-review expectation is defined ONCE in plan.py
(plan.validate_review); this launcher consumes exactly that shared constant
through the alias below.
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
NAME = 'atlas_memory_campaign_lane_20261105_v1'
SOURCE = WORK / 'experiments' / NAME
PLAN_SOURCE = WORK / 'experiments' / 'atlas_memory_campaign_plan_20261105_v1'
RESULTS = WORK / 'results/diagnostic' / NAME
GUARD = WORK / 'experiments/revision2_group_admission_device_20261028_v3'
GUARD_LAUNCH_SHA = plan.GUARD_LAUNCH_SHA
GUARD_MANIFEST_SHA = plan.GUARD_MANIFEST_SHA
BINARY = WORK / 'src/revision2_sdma_revalidation_20261028_v1'
SHMEM = WORK / 'src/shmem'
HELPER = WORK / 'experiments/revision2_sdma_queue_metadata_20261028_v1/metadata.py'
LARGE = WORK / 'experiments/revision2_sdma_large_correctness_20261028_v1'
# The frozen first-memory lane (first_memory_worker.py / first_memory_contract.py /
# first_memory_validation.py ...) — the import source for PYTHONPATH /workspace/first_memory.
# Remote existence audit 2026-11-05 (main agent, all 12 host path constants):
# this line was the ONLY defect (missing the 'olmoe_o1_' path segment, so
# runtime_args raised 'missing mount'); PLAN_SOURCE / SNAPSHOT_DIR / MODEL /
# LEGACY / SYNC_MODEL / REFERENCE / GUARD / BINARY / SHMEM / HELPER / LARGE /
# BUDGET_MATCHED all verified present on the remote host.
FIRST_MEMORY = WORK / 'experiments/revision2_olmoe_o1_first_request_memory_20261028_v3'
BUDGET_MATCHED = WORK / 'experiments/revision2_budget_matched_first_request_20261029_v1'
# L2 memory_snapshot module source: the remote lane that first used this
# script. (Remote-measured 2026-11-06: the first-memory lane directory has
# NO memory_snapshot.py; the real file lives here with SHA256 ==
# plan.MEMORY_SNAPSHOT_SHA256 exactly. The byte-identical copy in
# revision2_sdma_large_correctness_20261028_v1 is the alternative.)
SNAPSHOT_DIR = WORK / 'experiments/revision2_model_memory_init_20261028_v1'
# B4: engine-job mounts copied from the model-memory-pilot / first-memory
# host lanes (identical targets); the sync-model lane provides
# model_validation on the container PYTHONPATH.
MODEL = WORK / 'models/OLMoE-1B-7B-0125-Instruct'
LEGACY = WORK / 'experiments/postreview_olmoe_o1_interaction_o32_20260821_v2'
SYNC_MODEL = WORK / 'experiments/revision2_olmoe_o1_sync_model_20261028_v1'
REFERENCE = WORK / 'results/diagnostic/revision2_olmoe_o1_sync_model_20261028_v2/p16'

FILES = plan.FILES
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


def check_source_tree(source):
    """Recursive source-tree coverage; symlink rejection precedes the
    __pycache__ exemption so a symlinked cache entry is refused (A9)."""
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
    review = read(SOURCE / 'review.json')
    verify_review_binding(review, manifest_sha)
    if sha(PLAN_SOURCE / 'arms.json') != plan.ARMS_SHA256 \
            or sha(PLAN_SOURCE / 'PLAN.md') != plan.PLAN_MD_SHA256:
        raise ValueError('bound plan sources differ from the frozen campaign')
    plan.load_arms()
    if sha(BINARY / 'cashflex/lib/libcashflex.so') != plan.LIB_SHA256 \
            or sha(SHMEM / 'build_cann9_vllm/lib/libshmem.so') != plan.SHMEM_SHA256:
        raise ValueError('runtime provider mismatch')
    if sha(HELPER) != plan.HELPER_SHA256 or sha(LARGE / 'layout.json') != plan.LAYOUT_SHA256:
        raise ValueError('frozen queue-metadata helper/layout mismatch')
    snapshot = SNAPSHOT_DIR / 'memory_snapshot.py'
    if sha(snapshot) != plan.MEMORY_SNAPSHOT_SHA256:
        raise ValueError('L2 memory_snapshot module mismatch')
    if sha(GUARD / 'launch.py') != GUARD_LAUNCH_SHA:
        raise ValueError('admission guard source mismatch')
    spec = importlib.util.spec_from_file_location('memcamp_admission_guard', GUARD / 'launch.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.manifest(GUARD, GUARD_MANIFEST_SHA, guard.FILES)
    guard.RESULTS = RESULTS
    return guard, review


def job_dir(job_id):
    return RESULTS / 'jobs' / job_id


def fixed_mounts():
    return [(Path('/usr/local/Ascend/driver'), '/usr/local/Ascend/driver', 'ro'),
            (Path('/etc/ascend_install.info'), '/etc/ascend_install.info', 'ro'),
            (BINARY, '/workspace/cashflex', 'ro'), (SHMEM, '/workspace/shmem', 'ro'),
            (HELPER.parent, '/workspace/queue_metadata', 'ro'),
            (LARGE, '/workspace/sdma_large', 'ro'), (SOURCE, '/workspace/lane', 'ro'),
            (PLAN_SOURCE, '/workspace/plan', 'ro'), (FIRST_MEMORY, '/workspace/first_memory', 'ro'),
            (BUDGET_MATCHED, '/workspace/budget_matched', 'ro'),
            (SNAPSHOT_DIR, '/workspace/memory_snapshot', 'ro'),
            (MODEL, '/workspace/model', 'ro'), (LEGACY, '/workspace/legacy', 'ro'),
            (SYNC_MODEL, '/workspace/model_hook', 'ro'),
            (REFERENCE, '/workspace/reference', 'ro')]


def expected_mounts(payload=None):
    mounts = {target: (str(host), mode == 'rw') for host, target, mode in fixed_mounts()}
    if payload is not None:
        mounts['/workspace/memory_results'] = (str(payload), True)
    return mounts


def entry_script(entry):
    return ('set -euo pipefail; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
            'export PYTHONPATH=/workspace/lane:/workspace/plan:/workspace/first_memory:'
            '/workspace/budget_matched:/workspace/model_hook:/workspace/cashflex:'
            '/workspace/queue_metadata:/workspace/sdma_large:/workspace/legacy:'
            '${PYTHONPATH:-}; '
            'export LD_LIBRARY_PATH=/workspace/cashflex/cashflex/lib:'
            '/workspace/shmem/build_cann9_vllm/lib:${LD_LIBRARY_PATH:-}; '
            'exec python3 -B -u /workspace/lane/' + entry)


def runtime_args(name, network, ipc, env, entry, guard=None, world=0, payload=None,
                 memory_limit=None):
    args = ['docker', 'run', '-d', '--name', name, '--network', network, '--ipc', ipc]
    if memory_limit is not None:
        args += ['--memory', memory_limit]
    if guard is not None:
        for dev in guard.nodes(world):
            args += ['--device', f'{dev}:{dev}:rwm']
    mounts = fixed_mounts() + ([(payload, '/workspace/memory_results', 'rw')]
                               if payload is not None else [])
    for host, target, mode in mounts:
        if not host.exists():
            raise ValueError('missing mount: ' + str(host))
        args += ['-v', f'{host}:{target}:{mode}']
    for key, value in env.items():
        args += ['-e', f'{key}={value}']
    args += [plan.IMAGE, 'bash', '-lc', entry_script(entry)]
    return args


def inspect_contract(info, argv, env, guard=None, world=0, payload=None,
                     memory_limit=None):
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
    if memory_limit is not None and host.get('Memory') != 2 * 1024 ** 3:
        raise RuntimeError('c0-1 container memory limit differs')
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


# ------------------------------------------------------- pure job helpers ---
def job_env(job_id, kv_heap, cell=None, l3=False, a5_free_t1_gib=None):
    """The full container env for one job, heap/slot plumbed per the arm
    contract (B5) and the A6 floor input produced by A5 (B11)."""
    spec = plan.job_spec(job_id, 0)
    shmem_port, master_port = plan.ports(job_id)
    env = {'MEMCAMP_JOB': job_id, 'MEMCAMP_TRIM': '0',
           'MEMCAMP_ARMS_SHA256': plan.ARMS_SHA256,
           'MEMCAMP_ARMS_PATH': '/workspace/plan/arms.json',
           'MEMCAMP_L3': '1' if l3 else '0',
           'WORLD_SIZE': str(plan.WORLD), 'MASTER_ADDR': '127.0.0.1',
           'MASTER_PORT': str(master_port),
           'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(plan.WORLD))),
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    env.update(plan.ENGINE_ENV_BASE)
    if spec['arm'] in ('P', 'B-B', 'B-C', 'B-D', 'pace'):
        env.update({'REV2_BACKEND': 'sync-diagnostic',
                    'HPA_SHMEM_IP_PORT': f'tcp://127.0.0.1:{shmem_port}',
                    'HPA_SHMEM_WORKSPACE_PLANES': '2'})
        if kv_heap and kv_heap.get('heap_gib'):
            # B5: the only mechanism that sizes the PACE heap and slot.
            env['HPA_SHMEM_HEAP_BYTES'] = str(int(kv_heap['heap_gib'] * plan.GIB))
            env['HPA_SHMEM_SLOT_BYTES'] = str(int(kv_heap['slot_L_mib'] * plan.MIB))
    else:
        env['REV2_BACKEND'] = 'hccl'
    if job_id == 'A6' and a5_free_t1_gib is not None:
        env['MEMCAMP_A6_FREE_T1_GIB'] = repr(float(a5_free_t1_gib))
    if cell:
        env.update({'MEMCAMP_MAX_MODEL_LEN': str(cell.get('L', 2048)),
                    'MEMCAMP_MAX_NUM_SEQS': str(cell.get('C', 16)),
                    'MEMCAMP_OUTPUT_TOKENS': str(cell.get('O', 1))})
    return env


def conditional_route(job_id, semantics):
    """B2: run | projection_only | gate (a6_feasible stays a hard gate)."""
    gate = plan.CONDITIONAL_JOBS.get(job_id)
    if gate is None:
        return 'run'
    if gate == 'per_peer_guard':
        return 'run' if semantics == 'per_peer' else 'projection_only'
    return 'gate'


def prior_receipt_ok(receipt, prior, manifest_sha, review_sha):
    """A prior job passes the ordering gate if it audited PASS, was recorded
    projection-only under non-per-peer semantics (B2), or was recorded
    search-complete when the preregistered boundary settled early (R3-A1)."""
    if receipt.get('job') != prior or receipt.get('manifest_sha256') != manifest_sha \
            or receipt.get('review_sha256') != review_sha:
        return False
    if receipt.get('status') == 'PASS':
        return True
    reason = receipt.get('projection_reason')
    if receipt.get('status') == 'PROJECTION_ONLY' and reason == 'per_peer_guard_semantics':
        return receipt.get('reviewed_semantics') in ('whole_input', 'union')
    if receipt.get('status') == 'SEARCH_COMPLETE' and reason in ('ladder_served_to_cap',
                                                                 'below_ladder_no_served_cell'):
        return receipt.get('job') == prior
    return False


def require_prior_jobs(job_id, manifest_sha, review_sha, trim_steps):
    for prior in plan.CAMPAIGN_ORDER[:plan.CAMPAIGN_ORDER.index(job_id)]:
        if prior in plan.trimmed_jobs(trim_steps):
            continue
        receipt = read(job_dir(prior) / 'completion.json')
        if not prior_receipt_ok(receipt, prior, manifest_sha, review_sha):
            raise ValueError(f'earlier job did not pass with identical sources: {prior}')


def derive_capacity_cell(job_id, prior_receipts):
    """B10/R2-B3: the cell of an OLMoE capacity slot follows deterministically
    from the SAME (model, arm)'s prior L=4096 search receipts via
    plan.next_capacity_cell. Confirm slots (k=5) run at L=1024 with the
    L=4096-derived C* (capped). Raises when the search is already complete —
    callers that want the recorded not-run path use capacity_cell_policy."""
    spec = plan.job_spec(job_id, 0)
    if spec['role'] not in ('capacity_search', 'capacity_confirm'):
        return None
    policy = capacity_cell_policy(job_id, prior_receipts)
    if 'search_complete' in policy:
        raise ValueError(f'capacity search slot {job_id} is already complete: '
                         f'{policy["search_complete"]}')
    return policy['cell']


def capacity_cell_policy(job_id, prior_receipts):
    """R3-B1: the cell policy for every capacity-bearing role.

    - role `capacity` (extension jobs C6QA/C6QB/C7GA/C7GB): DECLARED cell
      from plan.EXTENSION_CAPACITY_CELLS — preregistered per model, never
      derived from the OLMoE frontier (cross-model incomparability), never
      free-form.
    - role `capacity_search`/`capacity_confirm` (OLMoE slots): DERIVED from
      the same (model, arm) L=4096 search history.

    Returns {'cell': {...}, 'policy': 'declared'|'derived'} or
    {'search_complete': reason} when a search slot's boundary is already
    settled (whole ladder served, or below-ladder with nothing served) and
    the slot must be recorded not-run instead of raising.
    """
    spec = plan.job_spec(job_id, 0)
    if spec['role'] == 'capacity':
        declared = plan.EXTENSION_CAPACITY_CELLS.get(spec['model'])
        if declared is None:
            raise ValueError(f'no declared extension cell for model {spec["model"]}')
        return {'cell': dict(declared), 'policy': 'declared'}
    if spec['role'] not in ('capacity_search', 'capacity_confirm'):
        return None
    arm = spec['arm']
    history = same_arm_search_history(prior_receipts, arm, spec['model'])
    if spec['role'] == 'capacity_confirm':
        star = plan.c_star(history)
        if star is None:
            return {'search_complete': 'below_ladder_no_served_cell'}
        return {'cell': {'L': 1024, 'C': min(star, plan.LADDER_CONCURRENCY_CAP[1024]),
                         'O': 1}, 'policy': 'derived'}
    if not history:
        return {'cell': {'L': 4096, 'C': plan.ladder_concurrency(4096)[0], 'O': 1},
                'policy': 'derived'}
    cell = plan.next_capacity_cell(history, 4096)
    if cell is None:
        served = [C for C, outcome in history if outcome == 'served']
        reason = 'ladder_served_to_cap' if served else 'below_ladder_no_served_cell'
        return {'search_complete': reason}
    return {'cell': {'L': 4096, 'C': cell, 'O': 1}, 'policy': 'derived'}


def same_arm_search_history(prior_receipts, arm, model='olmoe'):
    """R2-B3/R3-B1.4: only the SAME (model, arm)'s L=4096 SEARCH receipts
    enter a derivation (cross-model and cross-arm pooling forbidden;
    confirm-role L=1024 cells excluded)."""
    history = []
    for receipt in prior_receipts:
        try:
            other = plan.job_spec(receipt.get('job', ''), 0)
        except ValueError:
            continue
        if other['arm'] != arm or other['model'] != model:
            continue
        probe = receipt.get('capacity_probe')
        if not probe or not probe.get('cell'):
            continue
        if probe['cell'].get('L') != 4096:
            continue
        history.append((probe['cell']['C'], probe.get('classification', 'censored')))
    return history


def audit_rederive_cell(job_id, prior_receipts, recorded_cell):
    """R2-B3/R2-A2: the audit re-derives the cell from the same receipts the
    start used and refuses any drift (the DESIGN-claimed re-derivation)."""
    expected = derive_capacity_cell(job_id, prior_receipts)
    if expected != recorded_cell:
        raise ValueError(f'capacity cell drift for {job_id}: recorded {recorded_cell}, '
                         f'derived {expected}')
    return expected


def validate_free_cell(role, cell):
    """R2-A7: explicit policy — derived slots never accept a free CLI cell;
    throughput-confirm cells stay operator-supplied (declared)."""
    if cell is None:
        return
    if role in ('capacity_search', 'capacity_confirm'):
        raise ValueError(f'role {role} derives its cell from prior receipts; '
                         'free CLI cells are rejected, not overridden')
    if role not in ('throughput_sweep', 'throughput_confirm'):
        raise ValueError('free-form cells are only accepted for confirm/sweep jobs')


def audit_a6_infeasible(payload_dir):
    """R2-B1: the A6-infeasible branch is a preregistered legitimate outcome
    (PLAN/FA6); its audit requires exactly the evidence subset the worker
    produces (fence_T0 + a6_infeasible) and lifts the floor into the
    receipt."""
    payload = Path(payload_dir)
    infeasible = payload / 'a6_infeasible.json'
    if not infeasible.is_file():
        return None
    if not (payload / 'fence_T0.json').is_file():
        raise RuntimeError('A6-infeasible evidence lacks fence_T0.json')
    record = json.loads(infeasible.read_text())
    kv_floor = record.get('kv_floor')
    if not kv_floor or kv_floor.get('feasible') is not False:
        raise RuntimeError('A6-infeasible record lacks an infeasible floor')
    return {'outcome': 'a6_infeasible', 'kv_floor': kv_floor,
            'finding': record.get('finding')}


def require_engine_blocks(role, engine_record):
    """R2-A6/R3-B1.3: every capacity-bearing role (search/confirm/extension
    `capacity`) must corroborate the configured-block question, so a None
    num_gpu_blocks is a hard audit failure there; elsewhere the absence is
    recorded honestly."""
    if role in ('capacity', 'capacity_search', 'capacity_confirm') \
            and engine_record.get('num_gpu_blocks') is None:
        raise RuntimeError('capacity arm audit requires the engine-reported '
                           'num_gpu_blocks (all engine paths failed)')
    return engine_record


def lift_kv_floor(payload_dir):
    """B3: lift the worker's A6 floor record into the receipt."""
    for name in ('a6_floor.json', 'a6_infeasible.json'):
        path = Path(payload_dir) / name
        if path.is_file():
            record = json.loads(path.read_text())
            return record.get('kv_floor', record)
    return None


def extract_free_t1_gib(payload_dir):
    """B11: A5's measured T1 free (rank 0) from the fencepost payload."""
    path = Path(payload_dir) / 'fence_T1.json'
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())
    for row in rows:
        if row.get('rank') == 0:
            return row['capture']['free_bytes'] / plan.GIB
    raise ValueError('fence_T1.json lacks rank 0')


def audit_l1_stream(directory):
    """B6: the L1 stream is required evidence and validated, not assumed."""
    path = Path(directory) / 'l1_stream.json'
    if not path.is_file():
        raise RuntimeError('L1 host-poller stream absent; run the job_l1 action '
                           'before job_audit (the #11 headline lens cannot be dark)')
    stream = read(path)
    return plan.validate_l1_stream(stream['samples'])


def audit_fenceposts(payload_dir, arm, role):
    """R2-B2: role-aware required sequence — T4 is WP-A/fill/pilot only
    (PLAN §1.4); capacity/throughput jobs audit exactly what they emit."""
    rows = []
    for name in plan.FENCEPOSTS:
        path = Path(payload_dir) / f'fence_{name}.json'
        if path.is_file():
            rows.extend(json.loads(path.read_text()))
    steady = role in plan.STEADY_ROLES
    return plan.validate_fencepost_sequence(
        rows, 'P' if arm in ('P', 'B-B', 'B-C', 'B-D', 'pace') else 'A',
        required=plan.required_fenceposts(
            'P' if arm in ('P', 'B-B', 'B-C', 'B-D', 'pace') else 'A', steady=steady))


def audit_reconciliation(directory, payload_dir):
    """B10 wiring: L1-vs-L2 bias at shared timestamps (within +/-2 s)."""
    stream = read(Path(directory) / 'l1_stream.json')
    l2 = []
    for name in ('T1', 'T2', 'T2b', 'T3', 'T4'):
        path = Path(payload_dir) / f'fence_{name}.json'
        if path.is_file():
            for row in json.loads(path.read_text()):
                if row.get('rank') == 0:
                    l2.append({'free_bytes': row['capture']['free_bytes'],
                               'mono_ns': row['mono_ns']})
    return plan.reconciliation_bias(stream['samples'], l2)


def aggregate_analyses(completions):
    """B10/R3-A2/R3-A3 wiring: pure cross-job analyses over receipts, with
    capacity histories and L1 maxima grouped by (model, arm) — cross-model
    pooling is forbidden — and the Tier-1 leg-(ii) peak alignment computed
    per model whenever both of its WP-B arms have per-die maxima."""
    durations = [row['duration_seconds'] for row in completions.values()
                 if row.get('status') == 'PASS']
    capacity_history = {}
    l1_max = {}
    for job, row in sorted(completions.items()):
        spec = plan.job_spec(job, 0)
        key = (spec['model'], spec['arm'])
        probe = row.get('capacity_probe')
        if probe and probe.get('cell'):
            capacity_history.setdefault(key, []).append(
                (probe['cell']['C'], probe.get('classification', 'censored')))
        if row.get('l1_max_used_per_die'):
            l1_max.setdefault(key, []).append(row['l1_max_used_per_die'])
    frontier = {f'{model}:{arm}': {'c_star': plan.c_star(history), 'history': history}
                for (model, arm), history in capacity_history.items()}
    alignments = {}
    for model, (native_arm, pace_arm) in plan.WP_B_MODEL_ARM_PAIRS.items():
        native_key, pace_key = (model, native_arm), (model, pace_arm)
        if native_key in l1_max and pace_key in l1_max:
            alignments[model] = plan.peak_alignment(
                {str(die): max(entry[die] for entry in l1_max[native_key])
                 for die in range(16)},
                {str(die): max(entry[die] for entry in l1_max[pace_key])
                 for die in range(16)})
    return {'durations': plan.summary(durations) if durations else None,
            'capacity_frontier': frontier,
            'peak_alignment_by_model': alignments,
            'peak_alignment_grouping': '(model, arm); cross-model pooling forbidden'}


# ------------------------------------------------------- precheck actions ---
def precheck_env():
    return {'MEMCAMP_ARMS_PATH': '/workspace/plan/arms.json',
            'MEMCAMP_ARMS_SHA256': plan.ARMS_SHA256,
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
            or record.get('arms_sha256') != plan.ARMS_SHA256
            or record.get('jobs') != len(plan.CAMPAIGN_ORDER)
            or record.get('memory_snapshot_sha256') != plan.MEMORY_SNAPSHOT_SHA256
            or record.get('metadata_helper_sha256') != plan.HELPER_SHA256
            or record.get('layout_sha256') != plan.LAYOUT_SHA256
            or record.get('trim_ladder') != [list(step) for step in plan.TRIM_LADDER]
            or record.get('capacity_ladders') != {str(L): plan.ladder_concurrency(L)
                                                  for L in plan.LADDER_LENGTHS}):
        raise RuntimeError('precheck evidence missing or inconsistent')
    plan.check_pace_schema(record['pace_schema'])
    if record['review_contract'] != {'schema': plan.REVIEW_SCHEMA, 'scope': plan.REVIEW_SCOPE,
                                     'groups': plan.REVIEW_GROUPS,
                                     'reviewed_files': sorted(plan.REVIEWED_FILES)}:
        raise RuntimeError('precheck review contract drifts from the shared constant')
    for job in plan.CAMPAIGN_ORDER:
        if record['ports'][job] != list(plan.ports(job)):
            raise RuntimeError('precheck port plan differs at ' + job)
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
    if (receipt.get('status') != 'PASS'
            or receipt.get('manifest_sha256') != manifest_sha
            or receipt.get('review_sha256') != review_sha
            or receipt.get('image_id') != plan.IMAGE
            or receipt.get('log_sha256') != sha(RESULTS / 'precheck' / 'container.log')
            or receipt.get('inspect_sha256') != sha(RESULTS / 'precheck' / 'inspect.json')):
        raise ValueError('matched no-device precheck required')
    return receipt


def snapshot(guard, job_id):
    shmem_port, master_port = plan.ports(job_id)
    guard.stage_info = lambda stage: ('native', plan.WORLD, (shmem_port, master_port))
    return guard.preflight('native-p16')


def check_snapshot(guard, snap, job_id):
    if snap.get('processes') != guard.parse_npu_info(snap['npu_smi_info']):
        raise ValueError('process table interpretation differs')
    shmem_port, master_port = plan.ports(job_id)
    if snap.get('ports') != [shmem_port, master_port] or \
            snap.get('devices') != list(guard.nodes(plan.WORLD)):
        raise ValueError('preflight port or device coverage differs')
    if set(snap['health']) != set(map(str, range(8))) or \
            set(snap['usage']) != set(map(str, range(8))):
        raise ValueError('health/usage card coverage differs')
    for card in range(8):
        guard.check_health(snap['health'][str(card)])
        guard.check_usage(snap['usage'][str(card)])


def a6_receipt(manifest_sha, review_sha):
    directory = job_dir('A6')
    if not (directory / 'completion.json').is_file():
        return None
    receipt = read(directory / 'completion.json')
    if (receipt.get('status'), receipt.get('manifest_sha256'),
            receipt.get('review_sha256')) != ('PASS', manifest_sha, review_sha):
        return None
    return receipt


# --------------------------------------------------------- L1 host poller ---
def l1_poller(runner, running, interval_s=plan.L1_INTERVAL_S, clock=None, sleep=None):
    """Host-side 1 Hz npu-smi loop; returns L1-tagged samples per die.

    `runner` (npu-smi text) and `running` (container-liveness probe) are
    injectable; sampler errors end the stream (recorded gap), never the job.
    """
    import time as _time
    clock = clock or _time.monotonic_ns
    sleep = sleep or _time.sleep
    samples = []
    errors = 0
    while True:
        try:
            rows = plan.parse_l1_usage(runner())
        except Exception:
            errors += 1
            break
        samples.extend(plan.l1_poller_record(rows, clock(), None))
        if not running():
            break
        sleep(interval_s)
    return samples, errors


def is_projection_dir(directory):
    """R2-A1: a projection-only job dir (no container, no payload) is a
    graceful no-op for both the watcher and the audit."""
    path = Path(directory) / 'completion.json'
    return path.is_file() and json.loads(path.read_text()).get('status') == 'PROJECTION_ONLY'


def job_l1(job_id, manifest_sha, review_sha):
    """B6: BLOCKING watcher action. Polls npu-smi from the host until the
    job container exits, then persists l1_stream.json on the main thread.
    Usage order: job_start -> job_l1 (blocks) -> job_audit. R2-A1: a
    projection-only job dir is a graceful no-op (there is no container)."""
    verify(manifest_sha, review_sha)
    directory = job_dir(job_id)
    if is_projection_dir(directory):
        print(json.dumps({'event': 'job_projection_only', 'job': job_id}), flush=True)
        return
    info = read(directory / 'container.json')

    def running():
        inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
        return bool(inspected['State'].get('Running'))

    samples, errors = l1_poller(lambda: command(['npu-smi', 'info'], timeout=15), running)
    publish(directory / 'l1_stream.json',
            {'schema': 'revision2.memory-campaign-l1-stream.v1', 'lens': 'L1',
             'samples': samples, 'sampler_errors': errors,
             'scope': 'host npu-smi 1 Hz per die; sampled maximum is a lower '
                      'bound; includes every process on the die'})
    print(json.dumps({'event': 'l1_persisted', 'job': job_id, 'samples': len(samples),
                      'sampler_errors': errors}), flush=True)


# ----------------------------------------------------------- job actions ---
def job_start(job_id, trim_steps, manifest_sha, review_sha, cell=None, l3=False):
    guard, review = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    spec = plan.job_spec(job_id, trim_steps)
    # B10: capacity cells are DERIVED, never free-form, except confirm cells
    # which the derivation rule itself defines from C*.
    prior = []
    for candidate in plan.CAMPAIGN_ORDER[:plan.CAMPAIGN_ORDER.index(job_id)]:
        if candidate in plan.trimmed_jobs(trim_steps):
            continue
        path = job_dir(candidate) / 'completion.json'
        if path.is_file():
            prior.append(read(path))
    # R2-A7/R3-B1.2: derived/declared slots reject free CLI cells explicitly
    # (never silently override); throughput confirm/sweep cells stay
    # operator-supplied.
    validate_free_cell(spec['role'], cell)
    policy = capacity_cell_policy(job_id, prior)
    if policy is not None and 'search_complete' in policy:
        # R3-A1: the preregistered boundary settled before this slot; record
        # the not-run slot and continue — never a deadlock, never silent.
        directory = job_dir(job_id)
        if directory.exists() or directory.is_symlink():
            raise ValueError('job already exists; no retries')
        directory.mkdir(parents=True, exist_ok=False)
        publish(directory / 'completion.json',
                {'schema': 'revision2.memory-campaign-completion.v1',
                 'status': 'SEARCH_COMPLETE', 'job': job_id, 'trim_steps': trim_steps,
                 'projection_reason': policy['search_complete'],
                 'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
                 'image_id': plan.IMAGE, 'duration_seconds': 0.0,
                 'duration_gate': plan.duration_gate(job_id, 0.0),
                 'quarantined': job_id in plan.QUARANTINED,
                 'scope': 'capacity search complete before this slot; boundary '
                          'recorded from prior receipts'})
        print(json.dumps({'event': 'job_search_complete', 'job': job_id,
                          'reason': policy['search_complete']}), flush=True)
        return
    if policy is not None:
        cell = policy['cell']
    require_prior_jobs(job_id, manifest_sha, review_sha, trim_steps)
    route = conditional_route(job_id, review['device_split_guard_semantics'])
    if route == 'gate':
        a6 = a6_receipt(manifest_sha, review_sha)
        if a6 is None or not a6.get('kv_floor', {}).get('feasible'):
            raise ValueError(f'{job_id} requires an A6-feasible receipt; GPT-OSS PACE '
                             'KV infeasibility is the reported outcome')
    directory = job_dir(job_id)
    if directory.exists() or directory.is_symlink():
        raise ValueError('job already exists; no retries')
    if route == 'projection_only':
        # B2: a legitimately-not-run conditional job is recorded, never run,
        # and neither blocks successors nor poisons the aggregate.
        directory.mkdir(parents=True, exist_ok=False)
        publish(directory / 'completion.json',
                {'schema': 'revision2.memory-campaign-completion.v1',
                 'status': 'PROJECTION_ONLY', 'job': job_id, 'trim_steps': trim_steps,
                 'projection_reason': 'per_peer_guard_semantics',
                 'reviewed_semantics': review['device_split_guard_semantics'],
                 'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
                 'image_id': plan.IMAGE, 'duration_seconds': 0.0,
                 'duration_gate': plan.duration_gate(job_id, 0.0),
                 'quarantined': job_id in plan.QUARANTINED,
                 'scope': 'B-D projection-only table row under whole-input/union '
                          'guard semantics (#9 whole-input min 88.95 MiB > 16 MiB slot)'})
        print(json.dumps({'event': 'job_projection_only', 'job': job_id}), flush=True)
        return
    a5_free = None
    if job_id == 'A6' or (spec['model'] == 'gpt_oss' and spec['arm'] in ('pace', 'P')):
        # B11/R2-A8: A6 and every GPT-OSS PACE arm resolve the A6 floor rule
        # from A5's measured T1 free.
        a5_free = extract_free_t1_gib(job_dir('A5') / 'payload')
        if a5_free is None:
            raise ValueError(f'{job_id} requires A5 audited first '
                             '(MEMCAMP_A6_FREE_T1_GIB producer)')
    kv_heap = plan.kv_heap_for(job_id, a5_free_t1_gib=a5_free)
    first = snapshot(guard, job_id)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'payload').mkdir()
    env = job_env(job_id, kv_heap, cell, l3, a5_free_t1_gib=a5_free)
    memory_limit = plan.C0_CGROUP_MEMORY_LIMIT if spec['role'] == 'c0_probe_cgroup' else None
    name = f'{NAME}-{job_id}'
    args = runtime_args(name, 'host', 'host', env, 'supervisor.py', guard, plan.WORLD,
                        directory / 'payload', memory_limit)
    publish(directory / 'preflight.json',
            {'source_manifest_sha256': manifest_sha, 'review_sha256': review_sha,
             'job': job_id, 'trim_steps': trim_steps, 'cell': cell,
             'cell_policy': policy['policy'] if policy else 'operator',
             'l3': bool(l3),
             'derived_cell': policy is not None and policy.get('policy') == 'derived',
             'kv_heap': kv_heap, 'first': first, 'docker_argv': args, 'env': env,
             'memory_limit': memory_limit})
    publish(directory / 'prelaunch_recheck.json', snapshot(guard, job_id))
    container = command(args).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('unknown container id: inspect, no automatic retry')
    publish(directory / 'container.json', {'id': container, 'name': name})
    print(json.dumps({'event': 'job_started', 'job': job_id, 'id': container,
                      'next': 'run job_l1 (blocks until container exit), then job_audit'}),
          flush=True)


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


def job_audit(job_id, trim_steps, manifest_sha, review_sha):
    guard, review = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    directory = job_dir(job_id)
    # R2-A1: projection-only dirs audit as a graceful no-op (the early-return
    # must be reachable for exactly the jobs it was written for).
    if is_projection_dir(directory):
        print(json.dumps({'event': 'job_projection_only', 'job': job_id}), flush=True)
        return
    before = read(directory / 'preflight.json')
    trim_steps = before.get('trim_steps', 0)
    spec = plan.job_spec(job_id, trim_steps)
    info = read(directory / 'container.json')
    inspected = json.loads(command(['docker', 'inspect', info['id']]))[0]
    if inspected['State'].get('Running'):
        print(json.dumps({'event': 'still_running', 'job': job_id}), flush=True)
        return
    logs = subprocess.run(['docker', 'logs', info['id']], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=30).stdout
    publish(directory / 'inspect.json', inspected)
    with (directory / 'container.log').open('x') as out:
        out.write(logs)
    inspect_contract(inspected, before['docker_argv'], before['env'], guard,
                     plan.WORLD, directory / 'payload', before.get('memory_limit'))
    rows = json_rows(logs)
    if not any(row.get('event') == 'job_natural_exit' for row in rows) or \
            re.search(r'\b507035\b|\bEZ9999\b|Traceback \(most recent call last\)', logs):
        raise RuntimeError('job did not complete naturally')
    check_snapshot(guard, before['first'], job_id)
    after = read(directory / 'prelaunch_recheck.json')
    check_snapshot(guard, after, job_id)
    plan.check_time_order(before['first']['utc'], after['utc'],
                          inspected['State']['StartedAt'], inspected['State']['FinishedAt'])
    duration = (plan.exact_ns(inspected['State']['FinishedAt'])
                - plan.exact_ns(inspected['State']['StartedAt'])) / 1e9
    gate = plan.duration_gate(job_id, duration)
    postflight = {'scope': 'read-only snapshot; not an exclusive reservation'}
    try:
        postflight['npu_smi_info'] = guard.command(['npu-smi', 'info'], timeout=15)
        postflight['parsed'] = guard.parse_npu_info(postflight['npu_smi_info'])
        postflight['status'] = 'OBSERVED_EMPTY_TARGET'
    except Exception as exc:
        postflight['status'] = 'UNCONFIRMED'
        postflight['error'] = type(exc).__name__ + ': ' + str(exc)
    publish(directory / 'postflight.json', postflight)
    payload_files = sorted(str(path.relative_to(directory / 'payload'))
                           for path in (directory / 'payload').rglob('*') if path.is_file())
    payload = directory / 'payload'
    receipt = {'schema': 'revision2.memory-campaign-completion.v1', 'status': 'PASS',
               'job': job_id, 'trim_steps': trim_steps, 'cell': before.get('cell'),
               'duration_seconds': duration, 'duration_gate': gate,
               'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
               'image_id': plan.IMAGE, 'quarantined': job_id in plan.QUARANTINED,
               'container_log_sha256': sha(directory / 'container.log'),
               'inspect_sha256': sha(directory / 'inspect.json'),
               'preflight_sha256': sha(directory / 'preflight.json'),
               'prelaunch_recheck_sha256': sha(directory / 'prelaunch_recheck.json'),
               'postflight_status': postflight['status'],
               'postflight_sha256': sha(directory / 'postflight.json'),
               'payload_files_sha256': {name: sha(payload / name) for name in payload_files}}
    if spec['role'] not in ('c0_probe_cgroup', 'c0_probe_npu_cap'):
        # R2-B1: the A6-infeasible branch audits on its actual evidence
        # subset (fence_T0 + a6_infeasible) — a preregistered legitimate
        # outcome that must earn a receipt, never a deadlock.
        infeasible = audit_a6_infeasible(payload)
        if infeasible is not None:
            # R3-A4: the branch's container ran, so the L1 stream is required
            # evidence here too (job_l1 precedes job_audit in the runbook).
            l1_summary = audit_l1_stream(directory)
            receipt['l1_max_used_bytes'] = l1_summary['max_used_bytes']
            receipt['l1_min_free_bytes'] = l1_summary['min_free_bytes']
            receipt.update(infeasible)
            publish(directory / 'completion.json', receipt)
            print(json.dumps({'event': 'job_audited', 'job': job_id,
                              'status': 'PASS', 'outcome': 'a6_infeasible',
                              'duration_seconds': duration,
                              'duration_warning': gate['warning']}), flush=True)
            return
        # B6: the L1 lens is required, validated evidence for every engine job.
        l1_summary = audit_l1_stream(directory)
        receipt['l1_max_used_bytes'] = l1_summary['max_used_bytes']
        receipt['l1_min_free_bytes'] = l1_summary['min_free_bytes']
        receipt['l1_dies'] = l1_summary['dies']
        receipt['l1_samples'] = l1_summary['samples']
        by_die = {}
        for row in read(directory / 'l1_stream.json')['samples']:
            by_die[row['die']] = max(by_die.get(row['die'], 0), row['used_bytes'])
        receipt['l1_max_used_per_die'] = [by_die[die] for die in range(plan.WORLD)]
        # B10 wiring: role-aware fencepost sequence (R2-B2) + L1/L2
        # reconciliation.
        receipt['fenceposts'] = audit_fenceposts(payload, spec['arm'], spec['role'])
        receipt['reconciliation'] = audit_reconciliation(directory, payload)
        # B9: the FA4 token gate record is required evidence for engine jobs.
        token_path = payload / 'token_gate.json'
        if not token_path.is_file():
            raise RuntimeError('FA4 token gate record absent for engine job ' + job_id)
        if not json.loads(token_path.read_text()).get('match'):
            raise RuntimeError('FA4 token gate FAILED for ' + job_id +
                               '; cell excluded, evidence preserved')
        # B8: engine-reported block count is required evidence; R2-A6: for
        # capacity arms a None block count is a hard failure (the arm exists
        # to corroborate the configured-block question).
        engine_path = payload / 'engine_runtime.json'
        if not engine_path.is_file():
            raise RuntimeError('engine runtime record absent for engine job ' + job_id)
        engine = json.loads(engine_path.read_text())
        require_engine_blocks(spec['role'], engine)
        receipt['engine_num_gpu_blocks'] = engine.get('num_gpu_blocks')
        # B8: the streak is lifted from the container log, never env.
        receipt['kv_streak_from_log'] = plan.kv_streak_from_log(logs)
        probe_path = payload / 'capacity_probe.json'
        if spec['role'] in ('capacity', 'capacity_search', 'capacity_confirm'):
            # R3-B1.1: every capacity-bearing role — including the four
            # extension jobs — must produce its probe evidence and have it
            # lifted, classified and re-checked against the cell policy.
            if not probe_path.is_file():
                raise RuntimeError('capacity probe record absent for capacity job '
                                   + job_id)
        if probe_path.is_file():
            probe = json.loads(probe_path.read_text())
            probe['insufficient_kv_streak'] = max(probe.get('insufficient_kv_streak', 0),
                                                  receipt['kv_streak_from_log'])
            probe['classification'] = plan.classify_probe(probe)
            receipt['capacity_probe'] = probe
            # R2-B3/R2-A2/R3-B1: audit-side re-check — the recorded cell must
            # equal what the same receipts derive/declare.
            prior = [read(job_dir(candidate) / 'completion.json')
                     for candidate in plan.CAMPAIGN_ORDER[:plan.CAMPAIGN_ORDER.index(job_id)]
                     if candidate not in plan.trimmed_jobs(trim_steps)
                     and (job_dir(candidate) / 'completion.json').is_file()]
            expected_policy = capacity_cell_policy(job_id, prior)
            if expected_policy is None or 'search_complete' in expected_policy:
                raise RuntimeError('capacity job audit requires a live cell policy')
            if expected_policy['cell'] != before.get('cell'):
                raise ValueError(f'capacity cell drift for {job_id}: recorded '
                                 f'{before.get("cell")}, policy {expected_policy}')
        # B3: the A6 floor rides the receipt.
        kv_floor = lift_kv_floor(payload)
        if kv_floor is not None:
            receipt['kv_floor'] = kv_floor
        if job_id == 'A5':
            # B11: produce the A6 floor input from the measured T1 free.
            receipt['a5_free_t1_gib'] = extract_free_t1_gib(payload)
    publish(directory / 'completion.json', receipt)
    print(json.dumps({'event': 'job_audited', 'job': job_id, 'status': 'PASS',
                      'duration_seconds': duration,
                      'duration_warning': gate['warning']}), flush=True)


def aggregate(manifest_sha, review_sha, trim_steps=0):
    guard, review = verify(manifest_sha, review_sha)
    require_precheck(manifest_sha, review_sha)
    effective = plan.effective_campaign(trim_steps)
    completions = {}
    for job_id in effective:
        directory = job_dir(job_id)
        receipt = read(directory / 'completion.json')
        if (receipt.get('job') != job_id
                or receipt.get('manifest_sha256') != manifest_sha
                or receipt.get('review_sha256') != review_sha
                or receipt.get('image_id') != plan.IMAGE):
            raise ValueError(f'job receipt does not bind identical evidence: {job_id}')
        if not prior_receipt_ok(receipt, job_id, manifest_sha, review_sha):
            raise ValueError(f'job receipt status not acceptable: {job_id}')
        if receipt.get('status') == 'PASS':
            if (receipt.get('container_log_sha256') != sha(directory / 'container.log')
                    or receipt.get('inspect_sha256') != sha(directory / 'inspect.json')
                    or receipt.get('preflight_sha256') != sha(directory / 'preflight.json')
                    or receipt.get('prelaunch_recheck_sha256') != sha(directory / 'prelaunch_recheck.json')
                    or receipt.get('postflight_sha256') != sha(directory / 'postflight.json')):
                raise ValueError(f'job evidence changed: {job_id}')
            for name, digest in receipt['payload_files_sha256'].items():
                if sha(directory / 'payload' / name) != digest:
                    raise ValueError(f'payload evidence changed: {job_id} {name}')
            check_snapshot(guard, read(directory / 'preflight.json')['first'], job_id)
            check_snapshot(guard, read(directory / 'prelaunch_recheck.json'), job_id)
        completions[job_id] = receipt
    c0_results = []
    for probe in ('C01', 'C02'):
        payload = job_dir(probe) / 'payload'
        for name in ('c0_cgroup_probe.json', 'c0_npu_cap_probe.json'):
            record = payload / name
            if record.is_file():
                row = json.loads(record.read_text())
                c0_results.append(row.get('outcome') or row)
    if len(c0_results) != 2:
        raise ValueError('aggregate requires both C0 probe outcomes')
    analyses = aggregate_analyses(completions)
    value = {'schema': 'revision2.memory-campaign-aggregate.v1', 'status': 'COMPLETE',
             'manifest_sha256': manifest_sha, 'review_sha256': review_sha,
             'image_id': plan.IMAGE, 'trim_steps': trim_steps,
             'jobs': {job: {'status': row['status'],
                            'duration_seconds': row.get('duration_seconds'),
                            'duration_warning': row.get('duration_gate', {}).get('warning'),
                            'quarantined': row.get('quarantined')}
                      for job, row in completions.items()},
             'projection_only_jobs': sorted(job for job, row in completions.items()
                                            if row.get('status') == 'PROJECTION_ONLY'),
             'search_complete_jobs': sorted(job for job, row in completions.items()
                                            if row.get('status') == 'SEARCH_COMPLETE'),
             'pilots_quarantined': sorted(plan.QUARANTINED),
             'analyses': analyses,
             'wording': plan.campaign_wording(c0_results),
             'scope': 'descriptive campaign rollup; peaks are sampled lower bounds; '
                      'lenses never summed; no causal attribution'}
    out = RESULTS / 'aggregate.json'
    if out.exists():
        raise ValueError('refuse to overwrite an existing aggregate')
    publish(out, value)
    print(json.dumps({'event': 'aggregated', 'jobs': len(completions),
                      'projection_only': value['projection_only_jobs'],
                      'search_complete': value['search_complete_jobs'],
                      'wording': value['wording']['equal_memory']['wording']}), flush=True)


def dispatch(args):
    """B1: single CLI dispatch point, signature-tested."""
    if args.action in ('precheck_arm', 'precheck_audit'):
        (precheck_arm if args.action == 'precheck_arm' else precheck_audit)(
            args.manifest_sha, args.review_sha)
    elif args.action == 'job_l1':
        job_l1(args.job, args.manifest_sha, args.review_sha)
    elif args.action == 'aggregate':
        aggregate(args.manifest_sha, args.review_sha, args.trim)
    elif args.action == 'job_start':
        if args.job is None:
            raise SystemExit('job actions require --job')
        cell = None
        if any((args.cell_L, args.cell_C, args.cell_O)):
            cell = {'L': args.cell_L or 2048, 'C': args.cell_C or 16, 'O': args.cell_O or 1}
        job_start(args.job, args.trim, args.manifest_sha, args.review_sha, cell, args.l3)
    elif args.action == 'job_audit':
        if args.job is None:
            raise SystemExit('job actions require --job')
        job_audit(args.job, args.trim, args.manifest_sha, args.review_sha)
    else:
        raise SystemExit('unknown action')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['precheck_arm', 'precheck_audit', 'job_start',
                                           'job_l1', 'job_audit', 'aggregate'])
    parser.add_argument('--job', choices=list(plan.CAMPAIGN_ORDER))
    parser.add_argument('--trim', type=int, default=0)
    parser.add_argument('--l3', action='store_true')
    parser.add_argument('--cell-L', type=int)
    parser.add_argument('--cell-C', type=int)
    parser.add_argument('--cell-O', type=int)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--review-sha', required=True)
    return parser


if __name__ == '__main__':
    dispatch(build_parser().parse_args())
