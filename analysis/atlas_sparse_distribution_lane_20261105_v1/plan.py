"""Pure CPU plans, cases binding, schedules, metrics and validators for the
concern-#8 sparse-distribution lane (PACE vs HCCL, 17 shapes, dual clock).

No framework/device/network imports; stdlib only. Every function here runs on
an ordinary laptop and is shared by the device worker, the no-device precheck,
the launcher audit, the aggregate and the local test suite, so the measured
structure cannot drift between those roles.

Cases are NOT copied into this lane: the machine-readable list of the reviewed
plan directory (atlas_sparse_distribution_plan_20261105_v1/cases.json) is bound
by SHA-256 and every guard/statistic is recomputed here from the embedded
16x16 matrices. The recomputation is authoritative; the values declared inside
cases.json are only cross-checked.
"""
import hashlib
import json
import os
import re
import statistics
from pathlib import Path

WORLD = 16
ROW_BYTES = 4096
DTYPE = 'torch.bfloat16'
SLOT_BYTES = 96 * 1024 ** 2
HEAP_BYTES = 4 * 1024 ** 3
GIB = 1024 ** 3
MIB = 1024 ** 2
UINT32 = 2 ** 32 - 1
CANARY = 173  # BF16-safe fill byte for tails and split sentinels
TAIL_SLACK_ROWS = 1  # (1024 + row_bytes - 1)//row_bytes with row_bytes 4096
PACE_CALLS_PER_CASE = 11  # 1 initial check + 2 warmups + 8 measured (lane precedent)
WARMUP_PER_ARM = 2
MEASURED_PER_ARM = 8
MEASURE_BLOCKS = 4  # (base, reverse) x 4, 2 calls per arm per block
ARRIVAL_SKEW_FLAG_NS = 5_000_000  # > 5 ms gate-quality flag (PLAN section 5)
TRIM_TRIGGER_SECONDS = 840  # single group > 14 min arms the trimming ladder

# Frozen provider identities (identical to the reverse-order lane and v3).
IMAGE = 'sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f'
SHMEM_SHA256 = 'c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71'
LIB_SHA256 = '24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9'
HELPER_SHA256 = '0d7fde8e4eb1f0c8c312412cfc2bb6ee6f5df116ea48e2e27aae2feadef962be'
LAYOUT_SHA256 = 'e844a5175ca3de3e4a54ef54723b76df6665607c4268ad29cc3ed541d7bf6154'

# Binding to the reviewed plan directory (cases referenced, never copied).
CASES_SHA256 = '3b7d831025b077e372ebc7fe4b935cbdd7bf6e20de817247099c574118258633'
PLAN_MD_SHA256 = '4c1a801cb9fc909e301bbff09c54d004646cef82adadf068f9d47769eafb9a36'
GEN_CASES_SHA256 = 'c34523101d19b3a618881cc43a1bb4f3826520eced8c6f4cfed38d87977d77e6'
WORKLOADS_SHA256 = '62d08e5d6cac30a6b9b8166f259713a94bf4171de014374ae8f0ba7bddd1c858'
LANE_DIR = Path(__file__).resolve().parent
LOCAL_PLAN_DIR = LANE_DIR.parent / 'atlas_sparse_distribution_plan_20261105_v1'
DEVICE_PLAN_DIR = Path('/workspace/plan')

# ------------------------------------------------------------- campaign ---
VARIANTS = ('plan_a', 'plan_b')
GROUPS_PLAN_A = tuple(f'job{index}' for index in range(6))
GROUPS_PLAN_B_A = tuple(f'pba{index}' for index in range(6))
GROUPS_PLAN_B_B = tuple(f'pbb{index}' for index in range(6))
GROUPS_PLAN_B = GROUPS_PLAN_B_A + GROUPS_PLAN_B_B
GROUPS = {'plan_a': GROUPS_PLAN_A, 'plan_b': GROUPS_PLAN_B}
CAMPAIGN_SLOTS = GROUPS_PLAN_A + GROUPS_PLAN_B  # disjoint port slots
SHMEM_PORT_BASE = 26601
MASTER_PORT_BASE = 26701

# Pre-registered trimming ladder (PLAN section 6): S8 -> R3 -> S5 -> R5.
LADDER = ('syn-unif-p25-r512',              # S8 scale-axis duplicate of S2
          'qwen-o1-call-036-dispatch',      # R3 median-typical control
          'syn-zipf-p50-r7680',             # S5 skew-axis companion of S3
          'qwen-o1-call-096-dispatch')      # R5 dense twin of R7
MAX_DROPS = len(LADDER)
MIN_CASES_PLAN_A = 17 - MAX_DROPS  # floor 13 cases (PLAN section 6)

# Plan B (only on SQ fail-closed): 17 cases split into two disjoint subgroups
# balanced by the PLAN's device-time ladder (small 18 s / mid 28 s / large 40 s)
# while keeping the R1/R2 anchor pair inside subgroup a. Canonical order kept.
PLAN_B_SUBGROUP_A_IDS = frozenset({
    'qwen-o1-call-038-dispatch', 'qwen-o1-call-039-combine',
    'qwen-o1-call-036-dispatch', 'qwen-o1-call-021-combine',
    'qwen-o1-call-096-dispatch', 'olmoe-o1-call-050-dispatch',
    'olmoe-o1-call-010-dispatch', 'syn-diag-512',
    'syn-unif-p25-r7680', 'syn-unif-p25-r512'})
PLAN_B_SUBGROUP_B_IDS = frozenset({
    'qwen-o1-call-163-combine', 'olmoe-o1-call-049-combine',
    'syn-unif-p50-r7680', 'syn-unif-p100-r7680', 'syn-zipf-p50-r7680',
    'syn-burst-p25-r7680', 'syn-perm-burst-r7680'})

# Historical anchor reference (descriptive cross-campaign drift only, never
# pooled). Rank-local max medians in ns from the two prior campaigns:
# forward order  = analysis_results/atlas_qwen_sparse_timing_device_20261029_v1
# reverse order  = analysis_results/atlas_qwen_sparse_reverse_timing_independent_20261030_v1
HISTORICAL_ANCHOR_RANK_LOCAL_MAX_NS = {
    'qwen-o1-call-038-dispatch': {
        'forward_job0': 259865, 'forward_job1': 265640, 'forward_job2': 266120,
        'reverse_job0': 542590, 'reverse_job1': 512565, 'reverse_job2': 457305,
        'forward_hccl': {'job0': 699245, 'job1': 547375, 'job2': 534490},
        'reverse_hccl': {'job0': 469110, 'job1': 472390, 'job2': 455805}},
    'qwen-o1-call-039-combine': {
        'forward_job0': 530375, 'forward_job1': 506450, 'forward_job2': 519765,
        'reverse_job0': 291570, 'reverse_job1': 439805, 'reverse_job2': 277305,
        'forward_hccl': {'job0': 517060, 'job1': 469300, 'job2': 430705},
        'reverse_hccl': {'job0': 607755, 'job1': 716390, 'job2': 535335}},
}
HISTORICAL_SCOPE = ('descriptive cross-campaign drift only; different day, '
                    'fabric and thermal state; no common-start gate in history; '
                    'do not pool with this campaign')

# Lane file set covered by the source manifest (review binds the same set).
FILES = {'DESIGN.md', 'plan.py', 'worker.py', 'supervisor.py', 'precheck.py',
         'launch.py', 'test_cpu.py', 'make_manifest.py',
         'observed_npu_info_fixture.json'}

# ------------------------------------------------------ review contract ---
# SINGLE contract constant: the device gate (launch.verify) consumes exactly
# this structure through plan.validate_review. This exists so the v3-class
# BLOCKER (launcher expecting a different review schema than reviewers issue)
# cannot recur: reviewer and launcher read one shared definition.
REVIEW_SCHEMA = 'revision2.sparse-distribution-review.v1'
REVIEW_STATUS_APPROVED = 'APPROVED'
REVIEW_SCOPE = 'sparse_distribution_pace_vs_hccl_p16'
REVIEW_GROUPS = {'plan_a': list(GROUPS_PLAN_A),
                 'plan_b': {'a': list(GROUPS_PLAN_B_A), 'b': list(GROUPS_PLAN_B_B)}}
REVIEWED_FILES = FILES | {'source_manifest.json'}
# PLAN section 9.3: the admission semantics of the frozen
# hpa_alltoallv_device_splits implementation (per-peer / whole-input / union)
# must be source-verified before any device action. The lane only sees the
# built .so, so the verification is carried by the independent review and
# asserted here; 'unverified' never opens the gate.
REVIEW_GUARD_SEMANTICS = ('per_peer', 'whole_input', 'union')


def validate_review(review, manifest_sha):
    """Validate an independent review against the single shared contract."""
    if not isinstance(review, dict):
        raise ValueError('review is not an object')
    if review.get('schema') != REVIEW_SCHEMA:
        raise ValueError('review schema differs from the shared contract')
    if review.get('status') != REVIEW_STATUS_APPROVED:
        raise ValueError('review is not APPROVED')
    if not isinstance(manifest_sha, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha):
        raise ValueError('invalid source manifest digest')
    if review.get('source_manifest_sha256') != manifest_sha:
        raise ValueError('review does not bind this source manifest')
    if review.get('cases_sha256') != CASES_SHA256:
        raise ValueError('review does not bind the frozen cases.json')
    if review.get('scope') != REVIEW_SCOPE:
        raise ValueError('review scope differs')
    files = review.get('reviewed_files')
    if not isinstance(files, list) or sorted(files) != sorted(REVIEWED_FILES):
        raise ValueError('reviewed file coverage differs from the lane source set')
    if review.get('groups') != REVIEW_GROUPS:
        raise ValueError('reviewed group plan differs from the registered campaign')
    if review.get('device_split_guard_semantics') not in REVIEW_GUARD_SEMANTICS:
        raise ValueError('device-split admission semantics not source-verified by review')
    return {'schema': REVIEW_SCHEMA, 'status': REVIEW_STATUS_APPROVED,
            'source_manifest_sha256': manifest_sha, 'cases_sha256': CASES_SHA256,
            'scope': REVIEW_SCOPE, 'reviewed_files': sorted(REVIEWED_FILES),
            'groups': REVIEW_GROUPS,
            'device_split_guard_semantics': review['device_split_guard_semantics']}


# ------------------------------------------------------------ cases ------
def _cases_path(path=None):
    if path is not None:
        return Path(path)
    override = os.environ.get('REVSD_CASES_PATH')
    candidates = ([Path(override)] if override else []) + \
                 [LOCAL_PLAN_DIR / 'cases.json', DEVICE_PLAN_DIR / 'cases.json']
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    raise ValueError('cases.json not found (set REVSD_CASES_PATH)')


# The bound cases file is frozen by SHA, so one successful default-resolution
# parse may be reused within a process; explicit path arguments bypass this
# cache (tests load mutated copies).
_CASES_CACHE = {}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != \
       (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('cases file changed while hashing')
    return digest.hexdigest()


def recompute_case_stats(send):
    """Authoritative statistics recomputed from an embedded 16x16 matrix."""
    if type(send) is not list or len(send) != WORLD or \
            any(type(row) is not list or len(row) != WORLD for row in send):
        raise ValueError('send matrix must be 16x16')
    if any(type(value) is not int or not 0 <= value <= 2 ** 31 - 1 for row in send for value in row):
        raise ValueError('send rows must be nonnegative int32')
    nonself = [send[i][j] for i in range(WORLD) for j in range(WORLD) if i != j]
    zeros = sum(1 for value in nonself if value == 0)
    maxpeer = max(max(row) for row in send) * ROW_BYTES
    whole = max(sum(row) for row in send) * ROW_BYTES
    agg = sum(sum(row) for row in send) * ROW_BYTES
    return {'empty_nonself_edges': zeros,
            'occupied_nonself_edges': 240 - zeros,
            'nonself_density': round((240 - zeros) / 240, 6),
            'per_peer_max_bytes': maxpeer,
            'per_peer_max_mib': round(maxpeer / MIB, 6),
            'whole_input_max_bytes': whole,
            'whole_input_max_mib': round(whole / MIB, 6),
            'aggregate_bytes': agg,
            'aggregate_mib': round(agg / MIB, 6),
            'per_rank_input_rows': [sum(row) for row in send],
            'per_rank_output_rows': [sum(send[i][j] for i in range(WORLD))
                                     for j in range(WORLD)],
            'guards_ok': bool(maxpeer <= SLOT_BYTES and whole <= SLOT_BYTES)}


def _check_declared(declared, recomputed, case_id):
    for key, value in recomputed.items():
        got = declared.get(key)
        if type(value) is float:
            if type(got) not in (int, float) or abs(got - value) > 1e-6:
                raise ValueError(f'declared stat differs ({case_id}): {key}')
        elif got != value:
            raise ValueError(f'declared stat differs ({case_id}): {key}')


def load_cases(path=None):
    """Load the bound cases.json and recompute every guard from its matrices."""
    cases_path = _cases_path(path)
    cache_key = str(cases_path) if path is None else None
    if cache_key is not None and cache_key in _CASES_CACHE:
        return _CASES_CACHE[cache_key]
    if file_sha256(cases_path) != CASES_SHA256:
        raise ValueError('cases.json identity differs from the frozen plan')
    document = json.loads(cases_path.read_text())
    if document.get('schema') != 'revision2.sparse-distribution-cases.v1':
        raise ValueError('cases schema differs')
    constraint = document.get('constraint', {})
    if (constraint.get('world_size'), constraint.get('row_bytes'), constraint.get('dtype'),
            constraint.get('slot_mib'), constraint.get('heap_gib'),
            constraint.get('single_node')) != (WORLD, ROW_BYTES, DTYPE, 96, 4, True):
        raise ValueError('cases constraint block differs from the frozen stage')
    if document.get('workload_sha256') != WORKLOADS_SHA256:
        raise ValueError('cases do not bind the frozen workloads.json')
    raw_cases = document.get('cases')
    if not isinstance(raw_cases, list) or len(raw_cases) != 17:
        raise ValueError('expected exactly the 17 reviewed cases')
    cases = []
    for case in raw_cases:
        if case.get('dtype') != DTYPE or case.get('row_bytes') != ROW_BYTES:
            raise ValueError('case dtype/row width differs')
        if case.get('pace_calls_per_rank') != PACE_CALLS_PER_CASE:
            raise ValueError('case PACE call count differs from the lane contract')
        send = case['send_matrix_rows']
        stats = recompute_case_stats(send)
        _check_declared(case, stats, case['id'])
        if not stats['guards_ok']:
            raise ValueError('case fails the slot/whole-input guard: ' + case['id'])
        if any(rows * ROW_BYTES > UINT32 for rows in stats['per_rank_output_rows']):
            raise ValueError('uint32 receive bound violated: ' + case['id'])
        if case.get('kind') not in ('real', 'synthetic'):
            raise ValueError('case kind differs')
        if case['kind'] == 'real' and not isinstance(case.get('source'), dict):
            raise ValueError('real case lacks source provenance')
        if case['kind'] == 'synthetic' and not isinstance(case.get('generator'), dict):
            raise ValueError('synthetic case lacks its closed-form generator')
        cases.append({'id': case['id'], 'kind': case['kind'], 'role': case.get('role'),
                      'scenario': case.get('scenario'), 'send_matrix_rows': send,
                      **stats})
    ids = [case['id'] for case in cases]
    if len(set(ids)) != 17:
        raise ValueError('case ids not distinct')
    # Cross-check the planning SQE estimates (models from the frozen 2-case
    # lane audit point: 22 calls / 704 MiB agg -> 110 SQE/channel maximum).
    budget = document.get('budget', {})
    calls = sum(PACE_CALLS_PER_CASE for _ in cases)
    agg_mib = sum(case['aggregate_bytes'] for case in cases) / MIB
    model_a = round(calls * (110.0 / 22.0), 1)
    model_b = round((110.0 / 704.0) * agg_mib, 1)
    declared = budget.get('sqe_channel_estimate', {})
    if (budget.get('cases'), budget.get('pace_calls_per_rank_per_group'),
            budget.get('hccl_calls_per_rank_per_group')) != (17, calls, calls):
        raise ValueError('cases budget call accounting differs')
    if abs(agg_mib - budget.get('group_aggregate_mib', -1)) > 1e-3:
        raise ValueError('cases budget aggregate accounting differs')
    if (abs(declared.get('model_a_per_call', -1) - model_a) > 0.05
            or abs(declared.get('model_b_bytes_proportional', -1) - model_b) > 0.05):
        raise ValueError('cases SQE planning estimates fail the cross-check')
    if declared.get('observed_ring_depth') != 2048 or declared.get('counter_bits') != 16:
        raise ValueError('cases SQE ring/counter constants differ from the audit')
    result = {'cases': cases, 'ids': tuple(ids), 'canonical': {case['id']: index for index, case in enumerate(cases)},
            'planning_estimate': {'calls': calls, 'aggregate_mib': agg_mib,
                                  'model_a_per_call_sum': model_a,
                                  'model_b_bytes_proportional': model_b,
                                  'observed_ring_depth': 2048, 'counter_bits': 16}}
    if cache_key is not None:
        _CASES_CACHE[cache_key] = result
    return result


def rank_splits(case, rank):
    matrix = case['send_matrix_rows']
    if type(rank) is not int or not 0 <= rank < len(matrix):
        raise ValueError('rank outside matrix')
    return list(matrix[rank]), [row[rank] for row in matrix]


def segment_bytes(case, source, destination):
    """Deterministic per-segment payload (reverse-lane seed scheme, BF16-safe)."""
    matrix = case['send_matrix_rows']
    if not (type(source) is int and type(destination) is int
            and 0 <= source < len(matrix) and 0 <= destination < len(matrix)):
        raise ValueError('segment endpoint outside matrix')
    rows = matrix[source][destination]
    if rows <= 0:
        raise ValueError('zero-row segment requested')
    seed = json.dumps(['shape-pilot-v1', case['id'], DTYPE, source, destination],
                      separators=(',', ':')).encode()
    raw = bytearray(hashlib.shake_256(seed).digest(rows * ROW_BYTES))
    # Clear bit0 and the sign of every BF16 high byte: exponent bounded, finite.
    raw[1::2] = raw[1::2].translate(bytes(value & 0x7e for value in range(256)))
    return bytes(raw)


def expected_bytes(case, rank, receive):
    """CPU oracle for one rank's input (receive=False) or output buffer."""
    send, recv = rank_splits(case, rank)
    counts = recv if receive else send
    parts = []
    for peer, count in enumerate(counts):
        if count:
            source, destination = (peer, rank) if receive else (rank, peer)
            parts.append(segment_bytes(case, source, destination))
    return b''.join(parts)


def peak_tensor_bytes(case, rank):
    """Live input + PACE output + HCCL output + tails/private splits."""
    send, recv = rank_splits(case, rank)
    return (sum(send) + 2 * sum(recv)) * ROW_BYTES + 3 * max(ROW_BYTES, 1024) \
        + 2 * (WORLD * 4 + 1024) + 4096


# --------------------------------------------------------- scheduling ----
def subgroup_of(group_key):
    if group_key in GROUPS_PLAN_A:
        return None
    if group_key in GROUPS_PLAN_B_A:
        return 'a'
    if group_key in GROUPS_PLAN_B_B:
        return 'b'
    raise ValueError('group outside the registered campaign')


def group_index(variant, group_key):
    if variant not in VARIANTS or group_key not in GROUPS[variant]:
        raise ValueError('variant/group outside the registered campaign')
    return GROUPS[variant].index(group_key)


def arm_first(group_key):
    """job0/2/4 (and pba/pbb even indices) run PACE first; odd run HCCL first.

    Six fresh groups per campaign give a 3/3 arm-order balance, repairing the
    historical 2:1 imbalance (PLAN section 4).
    """
    variant = 'plan_a' if group_key in GROUPS_PLAN_A else 'plan_b'
    return 'pace' if group_index(variant, group_key) % 2 == 0 else 'hccl'


def ports(group_key):
    if group_key not in CAMPAIGN_SLOTS:
        raise ValueError('group outside the reviewed plan')
    slot = CAMPAIGN_SLOTS.index(group_key)
    return SHMEM_PORT_BASE + 2 * slot, MASTER_PORT_BASE + 2 * slot


def campaign_case_ids(variant, subgroup, drops):
    """Effective canonical case list after the pre-registered ladder."""
    if variant not in VARIANTS:
        raise ValueError('unknown campaign variant')
    if type(drops) is not int or not 0 <= drops <= MAX_DROPS:
        raise ValueError('drops outside the pre-registered ladder')
    document = load_cases()
    ids = list(document['ids'])
    if subgroup is not None:
        if variant != 'plan_b' or subgroup not in ('a', 'b'):
            raise ValueError('subgroups exist only in plan_b')
        members = PLAN_B_SUBGROUP_A_IDS if subgroup == 'a' else PLAN_B_SUBGROUP_B_IDS
        ids = [case_id for case_id in ids if case_id in members]
    for case_id in LADDER[:drops]:
        if case_id in ids:
            ids.remove(case_id)
    if variant == 'plan_a' and len(ids) < MIN_CASES_PLAN_A:
        raise ValueError('plan_a trimming below the pre-registered 13-case floor')
    return tuple(ids)


def rotated_case_order(variant, group_key, drops):
    """Group g processes rotate(case_list, 3g mod len): anchors leave the head."""
    subgroup = subgroup_of(group_key)
    ids = campaign_case_ids(variant, subgroup, drops)
    offset = (3 * group_index(variant, group_key)) % len(ids)
    rotated = ids[offset:] + ids[:offset]
    canonical = load_cases()['canonical']
    return tuple((canonical[case_id], case_id) for case_id in rotated)


def iteration_plan(group_key):
    """(phase, arm, block, slot) sequence: warmup (base, reverse) then
    measured (base, reverse) x 4 alternating blocks (PLAN section 4)."""
    base = ('pace', 'hccl') if arm_first(group_key) == 'pace' else ('hccl', 'pace')
    reverse = tuple(reversed(base))
    sequence = []
    for pair in (base, reverse):
        for slot, arm in enumerate(pair):
            sequence.append(('warmup', arm, -1, slot))
    for block in range(MEASURE_BLOCKS):
        # Each measured block is the pair sequence (base, reverse): 2 calls
        # per arm per block, AB/BA alternation inside the block (PLAN section 4).
        for pair in (base, reverse):
            for slot, arm in enumerate(pair):
                sequence.append(('measure', arm, block, slot))
    if len(sequence) != 2 * (WARMUP_PER_ARM + MEASURED_PER_ARM):
        raise ValueError('iteration plan count differs from the lane contract')
    return tuple(sequence)


def gate_enabled(phase):
    """Only measured iterations carry the rank-0 flag + HCCL all-reduce(max)
    common start gate; warmups keep the historical ungated structure."""
    return phase == 'measure'


def generation_label(canonical_index):
    return PACE_CALLS_PER_CASE * canonical_index + 1


# ------------------------------------------------------ SQ budget gate ---
# Round-9/10: deterministic cold-start priming shape. job3 reproducibly hit
# HCCL aicpu 507018 when the group's FIRST alltoallv was the extreme-sparse
# diagonal case; the round-9 hypothesis test then showed SYNTHETIC priming
# traffic (syn-unif-p25-r512) itself trips the same fault (job0 crashed with
# that priming; barrier A1 passed concurrently; job0-2 had passed with NO
# priming), while REAL captured traffic as the first call is proven safe.
# The priming case is therefore a REAL dense anchor: qwen-o1-call-038-
# dispatch — the frozen lane's historical case 1, device-PASSed across many
# campaigns, and identical in shape to job0's own first case, so any
# residual priming effect coincides with the first measured case. One
# priming PACE call per rank runs BEFORE any measured case; every rank
# touches all 4 channels (dense matrix), and the call is INCLUDED in the
# fail-closed SQ budget (priming=True), keeping the runtime gate
# authoritative over true ring usage.
PRIMING_CASE_ID = 'qwen-o1-call-038-dispatch'
# Round-13: 5 priming iterations. Device evidence: job3's syn-diag-512 still
# tripped aicpu after ONE priming call, while the same case PASSES in
# job0-2 after ~160 preceding PACE calls — the kernel path needs more than
# a single warm call. 5 iterations on the real dense anchor (each followed
# by a full synchronize, 0.2 s settle after the last) stays far below the
# ~160-call proven-safe scale while multiplying the warm-through; the SQ
# budget counts all 5 (priming=True), keeping the fail-closed gate
# authoritative over true ring usage.
PRIMING_CALLS = 5


def priming_case():
    document = load_cases()
    return next(case for case in document['cases'] if case['id'] == PRIMING_CASE_ID)


def plan_call_budgets(case_ids, rank, helper, channels, priming=False):
    """Fail-closed cumulative SQE bound over the EXACT planned call sequence.

    `helper` is the frozen queue-metadata module and `channels` its captured
    quiescent channel records. Every case contributes
    PACE_CALLS_PER_CASE calls with its own per-peer byte vector; with
    priming=True one additional PRIMING_CASE call per rank is budgeted ahead
    of the sequence (round-9 cold-start warm-through); no queue draining is
    inferred, so the whole planned sequence must fit the SQ ring depth
    (measured at runtime; 2048 historically) and the 16-bit task counter.
    This recomputation is authoritative; the cases.json planning
    estimates are a separate cross-check only.
    """
    if any(row.get('sq_head') != 0 or row.get('sq_tail') != 0 or row.get('reserved') != 0
           for row in channels):
        raise ValueError('fresh zero head/tail/task-counter metadata required')
    document = load_cases()
    by_id = {case['id']: case for case in document['cases']}
    totals = [0] * len(channels)
    rows = []
    if priming:
        priming_case_data = priming_case()
        priming_bytes = [rows_ * ROW_BYTES
                         for rows_ in priming_case_data['send_matrix_rows'][rank]]
        priming_call = helper.submission_budget(priming_bytes, rank, channels)
        totals = [total + row['total_sqes'] * PRIMING_CALLS
                  for total, row in zip(totals, priming_call)]
        rows.append({'case': PRIMING_CASE_ID, 'pace_calls': PRIMING_CALLS,
                     'per_call_sqes': [row['total_sqes'] for row in priming_call],
                     'cumulative_after_case': list(totals),
                     'role': 'cold_start_priming'})
    for case_id in case_ids:
        case = by_id[case_id]
        peer_bytes = [rows_ * ROW_BYTES for rows_ in case['send_matrix_rows'][rank]]
        per_call = helper.submission_budget(peer_bytes, rank, channels)
        per_call_sqes = [row['total_sqes'] for row in per_call]
        totals = [total + sqes * PACE_CALLS_PER_CASE
                  for total, sqes in zip(totals, per_call_sqes)]
        rows.append({'case': case_id, 'pace_calls': PACE_CALLS_PER_CASE,
                     'per_call_sqes': per_call_sqes,
                     'cumulative_after_case': list(totals)})
    if any(total >= row['sq_depth'] or total >= 65536
           for total, row in zip(totals, channels)):
        raise ValueError('planned PACE call sequence could wrap the SQ ring or task counter')
    # The planning-estimate cross check itself already ran inside load_cases
    # (model A/B recomputed against the cases.json budget block); it rides
    # along in the persisted budget for auditability.
    estimate = document['planning_estimate']
    if estimate['calls'] != 17 * PACE_CALLS_PER_CASE:
        raise ValueError('planning-estimate call accounting differs')
    return {'pace_calls': len(case_ids) * PACE_CALLS_PER_CASE + (PRIMING_CALLS if priming else 0),
            'priming_calls': PRIMING_CALLS if priming else 0,
            'cases': rows, 'cumulative_sqes': totals,
            'sq_depth': [row['sq_depth'] for row in channels],
            'head_tail_drain_inferred': False, 'no_ring_or_task_counter_wrap': True,
            'planning_estimate_crosscheck': estimate}


# ---------------------------------------------------------- providers ----
def dladdr_provider_path(info_fname, info_sname=None):
    """Resolve a dladdr Dl_info file name to the provider object path.

    Only `fname` (the shared object containing the symbol) is a path;
    `sname` is the nearest symbol name (e.g. 'g_state') and is diagnostic
    only — it must never participate in the path comparison. Round-4 fix:
    the previous 'sname or fname' form realpath()'d the symbol name against
    the cwd and rejected every genuine provider inside the pinned image.
    """
    if not info_fname:
        raise ValueError('dladdr returned no containing file name')
    return os.path.realpath(os.fsdecode(info_fname))


def dladdr_symbol(info_sname):
    """Decode the dladdr symbol name for diagnostics; None when absent."""
    return os.fsdecode(info_sname) if info_sname else None


def discover_acl_library(maps_text):
    """Find the single loaded libascendcl provider from /proc/self/maps text.

    Read-only introspection of the already-loaded ACL runtime (torch_npu
    loads it before any worker code runs); no pointer is dereferenced.
    """
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split()
        if fields and fields[-1].endswith('/libascendcl.so'):
            paths.add(os.path.realpath(fields[-1]))
    if len(paths) != 1:
        raise ValueError('expected exactly one loaded libascendcl provider, found ' + str(len(paths)))
    path = paths.pop()
    if not path.startswith('/'):
        raise ValueError('unusable ACL provider path')
    return path


PACE_OPERATOR_NAME = 'cashflex::hpa_alltoallv_device_splits'
PACE_ARGUMENT_TYPES = ('Tensor', 'Tensor', 'Tensor', 'Tensor', 'int', 'int', 'int', 'int')
PACE_RETURN_TYPES = ('Tensor',)


def check_pace_schema(text):
    """Pin the frozen PACE device-split operator ABI from its schema string.

    Device-measured truth inside the pinned image:
    'cashflex::hpa_alltoallv_device_splits(Tensor _0, Tensor _1, Tensor _2, '
    'Tensor _3, int _4, int _5, int _6, int _7) -> Tensor _0'.
    The namespaced operator name and every argument/return type word are
    pinned exactly; the compiler-generated suffix names (`_0`, `_1`, ...) are
    not pinned, and the return segment is validated isomorphically to the
    argument segment (first type word per item, same item shape).
    """
    if not isinstance(text, str) or ' -> ' not in text or '(' not in text:
        raise ValueError('unrecognized operator schema string')
    head, returns = (part.strip() for part in text.split('->'))
    name = head[:head.index('(')].strip()
    if name != PACE_OPERATOR_NAME:
        raise ValueError('unexpected operator name: ' + name)
    body = head[head.index('(') + 1:head.rindex(')')]
    arguments = [item.strip() for item in body.split(',') if item.strip()]
    types = tuple(item.split()[0] for item in arguments)
    if len(arguments) != len(PACE_ARGUMENT_TYPES) or types != PACE_ARGUMENT_TYPES \
            or any(len(item.split()) != 2 for item in arguments):
        raise ValueError('PACE device-split argument ABI differs')
    return_items = [item.strip() for item in returns.split(',') if item.strip()]
    if len(return_items) != len(PACE_RETURN_TYPES) \
            or tuple(item.split()[0] for item in return_items) != PACE_RETURN_TYPES \
            or any(len(item.split()) != 2 for item in return_items):
        raise ValueError('PACE return ABI differs')
    return {'name': name, 'arguments': arguments, 'returns': list(PACE_RETURN_TYPES)}


# ------------------------------------------------------------- metrics ---
def _finite(values):
    for value in values:
        if type(value) not in (int, float) or value != value or value in (float('inf'), float('-inf')):
            raise ValueError('metric requires finite numbers')


def summary(values):
    values = list(values)
    if not values:
        raise ValueError('summary of nothing')
    _finite(values)
    if len(values) >= 2:
        q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    else:
        q1 = q3 = values[0]
    return {'n': len(values), 'median': statistics.median(values), 'q1': q1, 'q3': q3,
            'iqr': q3 - q1, 'min': min(values), 'max': max(values),
            'quantile_method': 'statistics.quantiles inclusive'}


def mad(values):
    """Median absolute deviation of the within-group 8 samples."""
    values = list(values)
    if not values:
        raise ValueError('mad of nothing')
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


METRIC_KEYS = ('rank_local_max_ns', 'makespan_ns', 'arrival_skew_ns')
DIRECTION_METRICS = ('rank_local_max_ns', 'makespan_ns')


def tie_threshold(hccl_median, pace_mad, hccl_mad):
    """PLAN section 8 tie band: max(1% x HCCL group median, 2 x arm MAD).

    PLAN words the MAD term as "2 x that arm's 8-sample MAD"; since the rule is
    applied symmetrically to both directions, this implementation takes the
    union (both arms' MADs), which widens the tie zone and is therefore the
    conservative reading against declaring a winner. Declared in DESIGN.md.
    """
    return max(0.01 * hccl_median, 2 * pace_mad, 2 * hccl_mad)


def group_direction(pace_summary_row, hccl_summary_row, metric):
    """Per-(case, group) direction with the pre-registered tie threshold."""
    if metric not in DIRECTION_METRICS:
        raise ValueError('direction rule applies to timing metrics only')
    m_pace = pace_summary_row[f'median_{metric}']
    m_hccl = hccl_summary_row[f'median_{metric}']
    threshold = tie_threshold(m_hccl, pace_summary_row[f'mad_{metric}'],
                              hccl_summary_row[f'mad_{metric}'])
    delta = m_pace - m_hccl
    if abs(delta) <= threshold:
        classification = 'tie'
    elif delta < 0:
        classification = 'pace_faster'
    else:
        classification = 'pace_slower'
    return {'metric': metric, 'delta_ns': delta, 'tie_threshold_ns': threshold,
            'pace_median_ns': m_pace, 'hccl_median_ns': m_hccl,
            'classification': classification}


def stable_classification(per_group):
    """Cross-group pre-registered rule: 6/6 same sign beyond tie, else unstable."""
    classes = [row['classification'] for row in per_group]
    if not classes:
        raise ValueError('no groups')
    counts = {name: classes.count(name) for name in ('pace_faster', 'pace_slower', 'tie')}
    if all(value == 'pace_faster' for value in classes):
        classification = 'pace_stable_faster'
    elif all(value == 'pace_slower' for value in classes):
        classification = 'pace_stable_slower'
    else:
        classification = 'unstable'
    flips = sum(1 for left, right in zip(classes, classes[1:]) if left != right)
    return {'classification': classification, 'per_group_classifications': classes,
            'class_counts': counts, 'adjacent_class_changes': flips}


# ---------------------------------------------------------- validators ---
CASE_RECORD_KEYS = ('pace', 'hccl', 'input', 'splits',
                    'input_tail', 'pace_tail', 'hccl_tail', 'send_tail', 'recv_tail')


def _check_timestamps(pairs):
    previous_end = None
    for start, end in pairs:
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError('invalid host timestamp pair')
        if previous_end is not None and start < previous_end:
            raise ValueError('non-chronological sample sequence')
        previous_end = end


def validate_case_record(record, canonical_index, position, variant, group_key, drops):
    """Validate one rank's per-case record against the frozen plan."""
    case_id = rotated_case_order(variant, group_key, drops)[position][1]
    expected_sequence = iteration_plan(group_key)
    if (record.get('event'), record.get('schema')) != ('case_samples', 'revision2.sparse-distribution-case.v1'):
        raise ValueError('case record identity differs')
    if (record.get('variant'), record.get('group'), record.get('world'),
            record.get('case'), record.get('case_index'), record.get('case_position'),
            record.get('generation'), record.get('generation_after_measurement'),
            record.get('arm_first')) != (
            variant, group_key, WORLD, case_id, canonical_index, position,
            generation_label(canonical_index), PACE_CALLS_PER_CASE * (canonical_index + 1),
            arm_first(group_key)):
        raise ValueError('case record plan binding differs')
    if record.get('clock') != 'same-host perf_counter_ns (single-container CLOCK_MONOTONIC)':
        raise ValueError('clock domain differs')
    if record.get('gate') != 'rank0_flag_plus_hccl_all_reduce_max_before_t0_measured_only':
        raise ValueError('start-gate declaration differs')
    if record.get('library_sha256') != LIB_SHA256 or record.get('shmem_sha256') != SHMEM_SHA256:
        raise ValueError('record provider identity differs')
    initial, final = record.get('initial', {}), record.get('final', {})
    for block in (initial, final):
        if block.get('exact') != dict.fromkeys(CASE_RECORD_KEYS, True):
            raise ValueError('bit/canary check reported false')
        if not re.fullmatch('[0-9a-f]{64}', block.get('expected_sha256', '')):
            raise ValueError('oracle digest missing')
    if initial.get('expected_sha256') != final.get('expected_sha256'):
        raise ValueError('oracle digest drifted within a case')
    if record.get('global_bad') != 0:
        raise ValueError('global failure count nonzero: ' + case_id)
    if record.get('validation_scope') != ('initial and final full checks only; '
                                          'intermediate repetitions not individually checked'):
        raise ValueError('validation scope differs')
    iterations = record.get('iterations')
    if not isinstance(iterations, list) or len(iterations) != len(expected_sequence):
        raise ValueError('iteration count differs from the plan')
    pairs = []
    for row, (phase, arm, block, slot) in zip(iterations, expected_sequence):
        if (row.get('phase'), row.get('arm'), row.get('block'), row.get('slot')) != (phase, arm, block, slot):
            raise ValueError('iteration order differs from the plan')
        if row.get('gated') is not gate_enabled(phase):
            raise ValueError('start-gate presence differs from the plan')
        pairs.append((row.get('t0_ns'), row.get('t1_ns')))
    _check_timestamps(pairs)
    return {'case_id': case_id, 'canonical_index': canonical_index,
            'iterations': [(row['t0_ns'], row['t1_ns']) for row in iterations],
            'budget_case_cumulative_sqes': record.get('budget_case_cumulative_sqes')}


def validate_case_budgets_per_rank(budgets_by_rank, canonical, rank):
    """Per-rank, per-channel cumulative-SQE structural validity.

    PLAN section 6 preregisters the SQ budget as PER RANK, PER CHANNEL
    (peers map to channels by peer % 4, so channel occupancy — and hence
    the cumulative arrays — legitimately DIFFER across ranks; real job0
    evidence: [539,682,682,649]/[561,594,627,484]/...). Cross-rank equality
    was never the preregistered standard. Structural invariants per rank:
    exactly 4 channel entries (world_channels(16, r) = min(16, 4)), each a
    nonnegative int, each below the observed ring depth 2048 (the runtime
    fail-closed gate against the MEASURED depth stays authoritative in
    plan_call_budgets; this is the audit-side sanity bound).
    """
    values = budgets_by_rank.get(rank)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f'per-rank budget array must have 4 channels '
                         f'(case {canonical}, rank {rank}): {values}')
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError(f'per-rank budget entries must be nonnegative ints '
                             f'(case {canonical}, rank {rank})')
        if value >= 2048:
            raise ValueError(f'per-rank cumulative SQE exceeds the observed ring '
                             f'depth 2048 (case {canonical}, rank {rank}): {value}')
    return values


def validate_channel_monotonicity(view, rotated):
    """Across the group's case order, every rank's per-channel cumulative
    SQE is non-decreasing (totals only accumulate)."""
    for rank in range(WORLD):
        previous = None
        for position, (canonical, _case_id) in enumerate(rotated):
            case_id = rotated[position][1]
            current = view[case_id]['budget_case_cumulative_sqes_by_rank'][rank]
            if previous is not None and any(cur < prev for prev, cur
                                            in zip(previous, current)):
                raise ValueError(f'per-channel cumulative SQE decreased across '
                                 f'cases (rank {rank}, case {canonical})')
            previous = current


def validate_group_records(records, variant, group_key, drops):
    """Validate all 16 rank records of one group; return per-case rank views."""
    if not records or {row.get('rank') for row in records} != set(range(WORLD)):
        raise ValueError('rank coverage differs')
    rotated = rotated_case_order(variant, group_key, drops)
    grouped = {(record['case_index'], record['case_position']): [] for record in records}
    for record in records:
        parsed = validate_case_record(record, record.get('case_index'),
                                      record.get('case_position'), variant, group_key, drops)
        grouped[(record['case_index'], record['case_position'])].append((record['rank'], parsed))
    expected_keys = {(canonical, position) for position, (canonical, _) in enumerate(rotated)}
    if set(grouped) != expected_keys:
        raise ValueError('group case coverage differs from the rotated plan')
    view = {}
    for (canonical, position), rows in grouped.items():
        if len(rows) != WORLD or {rank for rank, _ in rows} != set(range(WORLD)):
            raise ValueError('per-case rank coverage differs')
        by_rank = {rank: parsed['iterations'] for rank, parsed in rows}
        budgets_by_rank = {rank: parsed['budget_case_cumulative_sqes']
                           for rank, parsed in rows}
        # Round-7 fix: per-rank, per-channel validity (cross-rank equality
        # was wrong — channel occupancy differs by rank by construction).
        for rank in range(WORLD):
            validate_case_budgets_per_rank(budgets_by_rank, canonical, rank)
        view[rotated[position][1]] = {'canonical_index': canonical, 'position': position,
                                      'by_rank': by_rank,
                                      'budget_case_cumulative_sqes_by_rank': budgets_by_rank}
    validate_channel_monotonicity(view, rotated)
    return view


def measured_iteration_metrics(by_rank, group_key):
    """Cross-rank dual-clock metrics for each measured iteration (PLAN section 5)."""
    sequence = iteration_plan(group_key)
    if set(by_rank) != set(range(WORLD)):
        raise ValueError('rank coverage differs')
    for rank, rows in by_rank.items():
        if len(rows) != len(sequence):
            raise ValueError('per-rank iteration count differs')
    measured = []
    for index, (phase, arm, block, slot) in enumerate(sequence):
        if phase != 'measure':
            continue
        starts = [by_rank[rank][index][0] for rank in range(WORLD)]
        ends = [by_rank[rank][index][1] for rank in range(WORLD)]
        rank_local_max = max(end - start for start, end in zip(starts, ends))
        makespan = max(ends) - min(starts)
        skew = max(starts) - min(starts)
        measured.append({'arm': arm, 'block': block, 'slot': slot,
                         'rank_local_max_ns': rank_local_max,
                         'makespan_ns': makespan,
                         'arrival_skew_ns': skew,
                         'wait_gap_ns': makespan - rank_local_max,
                         'skew_gt_5ms': skew > ARRIVAL_SKEW_FLAG_NS})
    if len(measured) != MEASURED_PER_ARM * 2:
        raise ValueError('measured iteration count differs')
    return measured


def arm_group_metrics(measured):
    """median-of-8 summaries per arm for one (case, group)."""
    rows = {}
    for arm in ('pace', 'hccl'):
        samples = [row for row in measured if row['arm'] == arm]
        if len(samples) != MEASURED_PER_ARM:
            raise ValueError('measured per-arm count differs')
        entry = {'measured': samples}
        for key in METRIC_KEYS:
            values = [row[key] for row in samples]
            _finite(values)
            entry[f'median_{key}'] = statistics.median(values)
            entry[f'values_{key}'] = values
        entry['mad_rank_local_max_ns'] = mad(entry['values_rank_local_max_ns'])
        entry['mad_makespan_ns'] = mad(entry['values_makespan_ns'])
        entry['median_wait_gap_ns'] = entry['median_makespan_ns'] - entry['median_rank_local_max_ns']
        entry['arrival_skew_over_5ms_count'] = sum(1 for row in samples if row['skew_gt_5ms'])
        rows[arm] = entry
    return rows


def assemble_group_metrics(view, group_key):
    """Full per-group metrics for every case: dual-clock medians + directions."""
    result = {}
    for case_id, case_view in view.items():
        measured = measured_iteration_metrics(case_view['by_rank'], group_key)
        arms = arm_group_metrics(measured)
        entry = {'case_position': case_view['position'],
                 'canonical_index': case_view['canonical_index'],
                 'budget_case_cumulative_sqes_by_rank':
                     case_view['budget_case_cumulative_sqes_by_rank'],
                 'arms': arms,
                 'direction': {metric: group_direction(arms['pace'], arms['hccl'], metric)
                               for metric in DIRECTION_METRICS}}
        result[case_id] = entry
    return result


def trim_gate(prior_receipts, requested_drops):
    """Monotone-ladder admission for group_start: drops may only grow by one,
    and each increment must cite a prior group whose observed duration exceeded
    the 14-minute trigger (PLAN section 6)."""
    if type(requested_drops) is not int or not 0 <= requested_drops <= MAX_DROPS:
        raise ValueError('requested drops outside the ladder')
    if not prior_receipts:
        if requested_drops != 0:
            raise ValueError('the first group always runs the full plan')
        return None
    latest = prior_receipts[-1]
    prior_drops = latest.get('drops')
    if type(prior_drops) is not int or not 0 <= prior_drops <= MAX_DROPS:
        raise ValueError('prior receipt drops outside the ladder')
    if requested_drops < prior_drops:
        raise ValueError('drops may never decrease within a campaign')
    if requested_drops == prior_drops:
        return None
    if requested_drops > prior_drops + 1:
        raise ValueError('the ladder drops one case at a time')
    over_trigger = [row for row in prior_receipts
                    if type(row.get('duration_seconds')) in (int, float)
                    and row['duration_seconds'] > TRIM_TRIGGER_SECONDS
                    and row.get('drops') == requested_drops - 1]
    if not over_trigger:
        raise ValueError('ladder increment lacks a >14 min prior-group trigger')
    cited = max(over_trigger, key=lambda row: row['duration_seconds'])
    return {'dropped_case': LADDER[requested_drops - 1],
            'trigger_group': cited['group'],
            'trigger_duration_seconds': cited['duration_seconds'],
            'trigger_threshold_seconds': TRIM_TRIGGER_SECONDS}


# Round-14: preregistered device-limitation not-run groups. Device evidence
# (four reproducible job3 failures, 2026-11-05..07): the group starting at
# rotation offset 9 has syn-diag-512 as its FIRST case and reproducibly trips
# HCCL aicpu 507018 (RunAicpuRpcSrvLaunchV2_alltoallv); priming with 1 and
# then 5 real-anchor warm calls did not help (rounds 9/13 — the interaction
# is the case geometry on a cold group, not warm-through count); barrier A1
# passed concurrently (device healthy); the SAME case passes in job0-2 at
# later positions after ~160 preceding calls. job3 is therefore recorded as
# a known device-level limitation, not run: successors proceed, the
# aggregate accepts 5/6 + one device-limitation receipt, and the rotation
# offset 9 arm-order/case-position data point is declared MISSING in every
# output (direction rules judged on the groups that ran).
DEVICE_LIMITATION_GROUPS = frozenset({'job3'})
DEVICE_LIMITATION_REASONS = {'job3': 'syn_diagonal_early_case_aicpu_507018'}


def device_limitation_statement():
    return ('job3 (rotation offset 9, first case syn-diag-512) excluded due to '
            'aicpu 507018 device-level interaction; 3 groups with the same case '
            'at later positions PASS; priming 1/5 calls both ineffective; '
            'barrier A1 concurrent PASS')


def validate_completion(receipt, variant, group_key, manifest_sha, review_sha):
    """Validate one group receipt; drops come from the receipt itself because
    the pre-registered ladder may arm mid-campaign (later groups run fewer
    cases; every receipt still binds its own rotated plan)."""
    drops = receipt.get('drops')
    if type(drops) is not int or not 0 <= drops <= MAX_DROPS:
        raise ValueError('receipt drops outside the ladder')
    # Round-14: a preregistered device-limitation not-run receipt is a valid
    # terminal state for its group (bound to the registered reason).
    if receipt.get('status') == 'DEVICE_LIMITATION':
        if group_key not in DEVICE_LIMITATION_GROUPS:
            raise ValueError('device-limitation status on a non-registered group')
        if (receipt.get('schema'), receipt.get('variant'),
                receipt.get('group'), receipt.get('world'), receipt.get('drops'),
                receipt.get('manifest_sha256'), receipt.get('review_sha256'),
                receipt.get('image_id')) != (
                'revision2.sparse-distribution-completion.v1', variant, group_key,
                WORLD, drops, manifest_sha, review_sha, IMAGE):
            raise ValueError('device-limitation receipt identity differs')
        if receipt.get('device_limitation_reason') != DEVICE_LIMITATION_REASONS[group_key]:
            raise ValueError('device-limitation reason differs from the registry')
        return receipt  # no metrics/duration by construction
    if (receipt.get('schema'), receipt.get('status'), receipt.get('variant'),
            receipt.get('group'), receipt.get('world'), receipt.get('drops'),
            receipt.get('manifest_sha256'), receipt.get('review_sha256'),
            receipt.get('image_id')) != (
            'revision2.sparse-distribution-completion.v1', 'PASS', variant, group_key,
            WORLD, drops, manifest_sha, review_sha, IMAGE):
        raise ValueError('group receipt identity differs')
    if type(receipt.get('duration_seconds')) not in (int, float) or receipt['duration_seconds'] <= 0:
        raise ValueError('group receipt lacks an observed duration')
    if set(receipt.get('metrics', {})) != {case_id for _, case_id
                                           in rotated_case_order(variant, group_key, drops)}:
        raise ValueError('group receipt metric coverage differs from the rotated plan')
    return receipt


def metric_coverage_ok(receipt, variant, group_key):
    """Metric-set coverage for PASS receipts (device-limitation receipts
    carry no metrics by construction)."""
    return set(receipt.get('metrics', {})) == {case_id for _, case_id
                                               in rotated_case_order(variant, group_key,
                                                                     receipt.get('drops', 0))}


def campaign_rollup(group_metrics_by_group, variant, subgroup=None,
                    runnable_groups=None):
    """Cross-group analysis over the union of cases actually run.

    The ladder may trim later groups, so each case is analysed over the groups
    that ran it; cases missing from any campaign group can never reach the
    pre-registered 6/6 stability rule and are marked accordingly. When a
    group is a preregistered device limitation, `runnable_groups` carries the
    groups that could run (5/6) — coverage is judged against those, and the
    limitation is declared in the output rather than silently degrading every
    case to 'unstable'. Descriptive only; every original sample is retained.
    """
    if not group_metrics_by_group:
        raise ValueError('no groups')
    all_groups = runnable_groups if runnable_groups is not None else \
        [key for key in GROUPS[variant] if subgroup is None or subgroup_of(key) == subgroup]
    case_ids = []
    for metrics in group_metrics_by_group.values():
        for case_id in metrics:
            if case_id not in case_ids:
                case_ids.append(case_id)
    rollup = {}
    for case_id in case_ids:
        per_group = {key: metrics[case_id] for key, metrics in group_metrics_by_group.items()
                     if case_id in metrics}
        coverage_complete = sorted(per_group) == sorted(all_groups)
        entry = {'n_groups': len(per_group), 'expected_groups': len(all_groups),
                 'coverage_complete': coverage_complete,
                 'groups_running_case': sorted(per_group),
                 'case_positions': {key: row['case_position'] for key, row in per_group.items()}}
        for key in METRIC_KEYS + ('wait_gap_ns',):
            entry[key] = {}
            for arm in ('pace', 'hccl'):
                field = f'median_{key}'
                if key == 'wait_gap_ns':
                    values = [row['arms'][arm]['median_wait_gap_ns'] for row in per_group.values()]
                else:
                    values = [row['arms'][arm][field] for row in per_group.values()]
                entry[key][arm] = summary(values)
        entry['direction'] = {}
        for metric in DIRECTION_METRICS:
            per_group_direction = [row['direction'][metric] for row in per_group.values()]
            classified = stable_classification(per_group_direction)
            if not coverage_complete:
                # 6/6 same-sign stability is unreachable with reduced coverage.
                classified['classification'] = 'unstable'
                classified['reduced_coverage_note'] = ('case not run by every campaign group '
                                                       '(trimming ladder); 6/6 rule unreachable')
            entry['direction'][metric] = {**classified, 'per_group': per_group_direction}
        entry['arrival_skew_over_5ms_counts'] = {
            arm: sum(row['arms'][arm]['arrival_skew_over_5ms_count'] for row in per_group.values())
            for arm in ('pace', 'hccl')}
        if case_id in HISTORICAL_ANCHOR_RANK_LOCAL_MAX_NS:
            entry['historical_rank_local_max_reference'] = {
                'this_campaign_group_medians_us': {
                    key: round(row['arms']['pace']['median_rank_local_max_ns'] / 1e3, 3)
                    for key, row in per_group.items()},
                'history': HISTORICAL_ANCHOR_RANK_LOCAL_MAX_NS[case_id],
                'scope': HISTORICAL_SCOPE}
        rollup[case_id] = entry
    return rollup


AXIS_PAIRS = (
    ('density_axis', ('syn-diag-512', 'syn-unif-p25-r7680', 'syn-unif-p50-r7680',
                      'syn-unif-p100-r7680')),
    ('skew_axis', (('syn-unif-p50-r7680', 'syn-zipf-p50-r7680'),
                   ('syn-unif-p25-r7680', 'syn-burst-p25-r7680'))),
    ('scale_axis', (('syn-unif-p25-r7680', 'syn-unif-p25-r512'),
                    ('qwen-o1-call-038-dispatch', 'qwen-o1-call-096-dispatch'))),
    ('model_axis', (('qwen-o1-call-096-dispatch', 'olmoe-o1-call-050-dispatch'),
                    ('qwen-o1-call-163-combine', 'olmoe-o1-call-049-combine'))),
    ('direction_axis', (('qwen-o1-call-038-dispatch', 'qwen-o1-call-039-combine'),
                        ('qwen-o1-call-021-combine', 'qwen-o1-call-163-combine'))),
)


def axis_views(rollup):
    """Descriptive density/skew/scale/model/direction contrasts (no fitting)."""
    views = {}
    for name, items in AXIS_PAIRS:
        if name in ('density_axis',):
            views[name] = [case_id for case_id in items if case_id in rollup]
        else:
            views[name] = [pair for pair in items
                           if all(case_id in rollup for case_id in pair)]
    return views


# ------------------------------------------------- exact nanoseconds ------
ISO = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)$')


def exact_ns(value):
    """Parse explicit-offset RFC3339 into exact integer nanoseconds.

    Docker emits up to nine fractional digits; datetime parsing alone only
    keeps microseconds, so the sub-second field is handled as raw digits.
    """
    from datetime import datetime, timedelta
    match = ISO.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError('unrecognized timestamp')
    offset = match.group(3)
    if offset != 'Z' and (int(offset[1:3]) > 23 or int(offset[4:6]) > 59):
        raise ValueError('invalid UTC offset')
    naive = datetime.strptime(match.group(1), '%Y-%m-%dT%H:%M:%S')
    if offset == 'Z':
        shift = timedelta(0)
    else:
        sign = 1 if offset[0] == '+' else -1
        shift = sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6]))
    utc = naive - shift
    seconds = int((utc - datetime(1970, 1, 1)).total_seconds())
    return seconds * 10 ** 9 + int((match.group(2) or '').ljust(9, '0'))


def check_time_order(first_utc, second_utc, started_at, finished_at):
    """first preflight <= second preflight <= container start < finish."""
    for value in (started_at, finished_at):
        match = ISO.fullmatch(value) if isinstance(value, str) else None
        if match is None or match.group(3) not in ('Z', '+00:00'):
            raise ValueError('Docker timestamps must use explicit UTC')
    ordered = [exact_ns(value) for value in (first_utc, second_utc, started_at, finished_at)]
    if not ordered[0] <= ordered[1] <= ordered[2] < ordered[3]:
        raise ValueError('preflights and container timestamps are not in order')


def summary_from_receipts(receipts_by_group, variant, subgroup, manifest_sha, review_sha):
    """Validate a full campaign's receipts (mixed trim levels allowed by the
    ladder; preregistered device-limitation not-run receipts allowed for
    their registered groups) and roll up the analysis."""
    groups = [key for key in GROUPS[variant]
              if subgroup is None or subgroup_of(key) == subgroup]
    if sorted(receipts_by_group) != sorted(groups):
        raise ValueError('campaign receipt coverage differs')
    metrics, drops_timeline, device_limited = {}, {}, []
    for group_key in groups:
        receipt = receipts_by_group[group_key]
        validate_completion(receipt, variant, group_key, manifest_sha, review_sha)
        drops_timeline[group_key] = receipt.get('drops')
        if receipt.get('status') == 'DEVICE_LIMITATION':
            device_limited.append(group_key)
            continue
        metrics[group_key] = receipt['metrics']
    runnable = [key for key in groups if key not in device_limited]
    unexpected = [key for key in device_limited if key not in DEVICE_LIMITATION_GROUPS]
    if unexpected:
        raise ValueError('unregistered device-limitation receipts: ' + str(unexpected))
    rollup = campaign_rollup(metrics, variant, subgroup, runnable_groups=runnable)
    value = {'schema': 'revision2.sparse-distribution-aggregate.v1', 'variant': variant,
             'subgroup': subgroup, 'drops_by_group': drops_timeline,
             'groups': list(groups), 'cases': rollup, 'axis_views': axis_views(rollup),
             'runnable_groups': runnable,
             'device_limitation_jobs': sorted(device_limited),
             'device_limitation_statement': (device_limitation_statement()
                                             if device_limited else None),
             'result_interpretation_constraint': (
                 'rotation-offset-9 case-position data missing: direction rules '
                 'judged on the runnable groups (' + str(len(runnable)) + '/'
                 + str(len(groups)) + '), not 6/6; declared, not hidden'
                 if device_limited else None),
             'sample_unit': 'fresh process group; within-group repetitions are not independent',
             'statistics_scope': 'median/IQR/min/max of group medians only; no confidence '
                                 'intervals, no inferential tests, no model/serving claims',
             'keep_all_samples': True}
    return value
