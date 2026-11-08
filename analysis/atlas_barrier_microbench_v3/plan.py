"""Pure CPU plans, shapes, schedules, validators and metrics for barrier v3.

No framework/device/network imports. Every function here runs on an ordinary
laptop and is shared by the device worker, the no-device precheck, the
launcher audit and the local test suite, so the measured structure cannot
drift between those roles.
"""
import hashlib
import json
import os
import statistics

WORLD = 16
ARMS = ('A', 'B', 'C')
ARM_GROUPS = {'A': 5, 'B': 2, 'C': 3}
GROUP_ORDER = tuple((arm, index) for arm in ARMS for index in range(1, ARM_GROUPS[arm] + 1))

SHMEM_PORT_BASE = 26401
MASTER_PORT_BASE = 26501

# Frozen provider identities: identical to the v2 barrier lane.
SHMEM_SHA256 = 'c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71'
LIB_SHA256 = '24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9'
# Frozen queue-metadata helper and device layout (reverse-order lane identity).
HELPER_SHA256 = '0d7fde8e4eb1f0c8c312412cfc2bb6ee6f5df116ea48e2e27aae2feadef962be'
LAYOUT_SHA256 = 'e844a5175ca3de3e4a54ef54723b76df6665607c4268ad29cc3ed541d7bf6154'


def ports(arm, index):
    """Deterministic per-group (shmem, master) port pair; disjoint families."""
    if (arm, index) not in GROUP_ORDER:
        raise ValueError('group outside the reviewed plan')
    slot = GROUP_ORDER.index((arm, index))
    return SHMEM_PORT_BASE + 2 * slot, MASTER_PORT_BASE + 2 * slot


# ---------------------------------------------------------------- arm A ---
# Byte-for-byte the v2 empty-barrier loop, only re-hosted in fresh groups.
A_WARMUPS = 10
A_SAMPLES = 50
AB_HEAP_BYTES = 512 * 1024 ** 2
AB_SLOT_BYTES = 32 * 1024
AB_MARGIN_BYTES = 2 * 1024 ** 3  # v2's fixed pre-declared free-memory margin


# ---------------------------------------------------------------- arm B ---
B_PEER_ROW_BYTES = 1024  # int8 payload per peer per round through the PACE path
B_DTYPE = 'torch.int8'
B_BLOCKS = 5
B_BLOCK_ROUNDS = 10
B_WARMUP_ROUNDS = 3  # per sub-arm
B_SUBARMS = ('b1', 'b2')
B_MEASURED_PER_SUBARM = B_BLOCKS * B_BLOCK_ROUNDS


def arm_b_first_subarm(index):
    """Alternate which sub-arm leads per fresh group to cancel order bias."""
    return 'b1' if index % 2 == 1 else 'b2'


def arm_b_rounds(index):
    """(phase, sub) sequence: per-subarm warmups, then alternating 10-round blocks."""
    first = arm_b_first_subarm(index)
    second = 'b2' if first == 'b1' else 'b1'
    rounds = [('warmup', first)] * B_WARMUP_ROUNDS + [('warmup', second)] * B_WARMUP_ROUNDS
    for block in range(B_BLOCKS):
        lead, trail = (first, second) if block % 2 == 0 else (second, first)
        rounds += [('measure', lead)] * B_BLOCK_ROUNDS + [('measure', trail)] * B_BLOCK_ROUNDS
    return tuple(rounds)


# ---------------------------------------------------------------- arm C ---
C_K_VALUES = (0, 1, 2)
C_WARMUPS_PER_K = 2
C_MEASURED_PER_K = 30
C_BLOCK_ROUNDS = 10
C_BLOCKS = C_MEASURED_PER_K // C_BLOCK_ROUNDS
C_ROW_BYTES = 4096
C_DTYPE = 'torch.bfloat16'
C_SLOT_BYTES = 96 * 1024 ** 2
C_HEAP_BYTES = 4 * 1024 ** 3
C_GIB = 1024 ** 3
C_CASE_NAME = 'qwen-o1-call-038-dispatch'
C_CASE_SHA256 = '49c3fd0b3d5c55073da3e8195b8542cdd371722c83aba2b711d62bd2be68079c'
C_SOURCE_DATASET_SHA256 = '3954f89a0e91d85de9a4ffc0b214a9b91ab0f303d7d6ceab82fd729d492609bd'
# Exact send matrix of the frozen reverse-order lane's first case
# (qwen-o1-call-038-dispatch), copied from dataset.json of
# atlas_qwen_sparse_p16_candidate_v1 whose file digest is bound above.
CASE038_SEND_ROWS = (
    (44, 36, 33, 113, 11, 9, 23, 58, 9, 48, 16, 31, 31, 6, 37, 7),
    (5, 31, 11, 129, 2, 6, 34, 97, 1, 91, 3, 44, 2, 10, 29, 17),
    (0, 27, 24, 124, 1, 3, 58, 81, 0, 64, 4, 48, 0, 8, 50, 20),
    (0, 30, 21, 127, 3, 6, 62, 69, 0, 57, 6, 51, 0, 2, 60, 18),
    (0, 27, 19, 133, 5, 10, 64, 76, 0, 50, 2, 48, 0, 0, 63, 15),
    (0, 21, 19, 139, 5, 12, 64, 74, 0, 49, 3, 53, 0, 0, 61, 12),
    (0, 18, 23, 146, 3, 10, 63, 75, 0, 44, 7, 53, 0, 0, 59, 11),
    (0, 23, 21, 145, 0, 8, 64, 66, 0, 49, 6, 51, 0, 0, 67, 12),
    (0, 24, 24, 144, 1, 7, 64, 67, 0, 43, 4, 52, 0, 0, 71, 11),
    (0, 23, 24, 137, 0, 8, 64, 65, 0, 51, 3, 52, 0, 0, 73, 12),
    (0, 26, 21, 139, 0, 6, 64, 72, 0, 43, 4, 52, 0, 0, 74, 11),
    (0, 27, 22, 141, 1, 8, 64, 69, 0, 38, 5, 54, 0, 0, 72, 11),
    (0, 25, 22, 141, 0, 6, 64, 67, 0, 39, 6, 52, 0, 0, 78, 12),
    (0, 26, 18, 140, 0, 7, 64, 68, 0, 46, 6, 51, 0, 0, 74, 12),
    (0, 24, 19, 148, 0, 6, 63, 72, 0, 39, 5, 52, 0, 0, 72, 12),
    (0, 23, 19, 144, 0, 9, 64, 72, 0, 38, 6, 52, 0, 0, 74, 11),
)
CASE038_RECV_ROWS = (49, 411, 340, 2190, 32, 121, 943, 1148, 10, 789, 86, 796, 33, 26, 1014, 204)


def case038():
    """Validate and return the frozen reverse-lane dispatch case."""
    matrix = CASE038_SEND_ROWS
    if len(matrix) != WORLD or any(len(row) != WORLD for row in matrix):
        raise ValueError('case038 matrix must be 16x16')
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError('case038 rows must be nonnegative ints')
    row_sums = tuple(sum(row) for row in matrix)
    col_sums = tuple(sum(matrix[src][dst] for src in range(WORLD)) for dst in range(WORLD))
    if any(s != 512 for s in row_sums):
        raise ValueError('case038 row sums differ from the frozen 512-row inputs')
    if col_sums != CASE038_RECV_ROWS:
        raise ValueError('case038 column sums differ from the frozen receive rows')
    if any(value * C_ROW_BYTES > C_SLOT_BYTES for row in matrix for value in row):
        raise ValueError('case038 peer segment exceeds the 96 MiB slot')
    total_input = sum(row_sums) * C_ROW_BYTES
    if total_input > C_SLOT_BYTES or total_input > 2 ** 32 - 1:
        raise ValueError('case038 whole input violates the device-split envelope')
    return {'name': C_CASE_NAME, 'case_sha256': C_CASE_SHA256, 'dtype': C_DTYPE,
            'row_bytes': C_ROW_BYTES, 'send_matrix_rows': [list(row) for row in matrix],
            'recv_rows': list(CASE038_RECV_ROWS), 'slot_bytes': C_SLOT_BYTES,
            'heap_bytes': C_HEAP_BYTES, 'source_dataset_sha256': C_SOURCE_DATASET_SHA256}


def rank_splits(rank):
    case038()
    if type(rank) is not int or not 0 <= rank < WORLD:
        raise ValueError('rank outside the frozen case matrix')
    matrix = CASE038_SEND_ROWS
    return list(matrix[rank]), [matrix[src][rank] for src in range(WORLD)]


def arm_c_peak_tensor_bytes(rank):
    """Live input + PACE output + checker output + tails/private splits.

    Mirrors pilot_contract.peak_tensor_bytes of the original sparse lane for
    the single frozen case, so the admission margin matches its formula.
    """
    send, recv = rank_splits(rank)
    slack = 1024
    return (sum(send) + 2 * sum(recv)) * C_ROW_BYTES + 3 * max(C_ROW_BYTES, slack) \
        + 2 * (WORLD * 4 + slack) + 4096


def arm_c_iteration_plan():
    """(phase, k) sequence: per-k warmups, then alternating k-order blocks.

    Block k-order alternates by block position only (every group gets the
    identical sequence; each k already sees an identical even/odd generation
    mix), mirroring the frozen plan that all ranks execute.
    """
    sequence = []
    for k in C_K_VALUES:
        sequence += [('warmup', k)] * C_WARMUPS_PER_K
    for block in range(C_BLOCKS):
        order = C_K_VALUES if block % 2 == 0 else tuple(reversed(C_K_VALUES))
        for k in order:
            sequence += [('measure', k)] * C_BLOCK_ROUNDS
    return tuple(sequence)


def arm_c_checkpoints():
    """Iteration ordinals requiring bit-exact output/input/split checks:
    the first and the last iteration of each k value (reverse-lane precedent
    of one initial and one final complete check per arm)."""
    sequence = arm_c_iteration_plan()
    marks = set()
    for k in C_K_VALUES:
        positions = [i for i, (_, kk) in enumerate(sequence) if kk == k]
        if not positions:
            raise ValueError('k value missing from the plan')
        marks.add(positions[0])
        marks.add(positions[-1])
    return marks


# ------------------------------------------------------- shared helpers ---
# The frozen PACE device-split operator requires its input (and, by the
# verified reverse-order-lane precedent, the output) to sit in an explicit
# backing allocation with at least 1024 TAIL bytes beyond the live rows
# (device-observed contract, B1 container 42fe5bd9: "input requires an
# explicit backing allocation with at least 1024 tail bytes"). Tail rows
# follow the reverse lane's row-width conversion and are canary-filled;
# the canaries are verified untouched at every fail-closed checkpoint.
PACE_TAIL_BYTES = 1024
INT8_CANARY = -83
BF16_CANARY = 173
SPLIT_CANARY = 173          # reverse-order-lane int32 split-backing canary
SPLIT_ITEM_BYTES = 4        # split vectors are int32


def pace_tail_rows(row_bytes):
    if type(row_bytes) is not int or row_bytes <= 0:
        raise ValueError('row width must be a positive int')
    return (PACE_TAIL_BYTES + row_bytes - 1) // row_bytes


def pace_split_tail_elements(item_bytes=SPLIT_ITEM_BYTES):
    """Tail ELEMENTS for a split vector of the given element width.

    The operator requires >= 1024 tail bytes on every tensor argument,
    including the int32 split vectors (device-observed, B1 container
    15a6e1f33a5b); the reverse-order lane's verified form is int32 with 256
    tail elements (256 x 4 B = 1024 B) canary-filled with 173.
    """
    if type(item_bytes) is not int or item_bytes <= 0:
        raise ValueError('split element width must be a positive int')
    return (PACE_TAIL_BYTES + item_bytes - 1) // item_bytes


def required_free_bytes(arm, rank):
    if arm in ('A', 'B'):
        return AB_MARGIN_BYTES
    if arm == 'C':
        return C_HEAP_BYTES + arm_c_peak_tensor_bytes(rank) + C_GIB
    raise ValueError('unknown arm')


def arm_peer_bytes(arm, rank):
    """Per-peer byte vector for the SDMA submission budget of one PACE call."""
    if arm == 'A':
        return None
    if arm == 'B':
        return [B_PEER_ROW_BYTES] * WORLD
    if arm == 'C':
        send, _ = rank_splits(rank)
        return [rows * C_ROW_BYTES for rows in send]
    raise ValueError('unknown arm')


def arm_pace_calls(arm, index=0):
    if arm == 'A':
        return 0
    if arm == 'B':
        return len(arm_b_rounds(index))
    if arm == 'C':
        return len(arm_c_iteration_plan())
    raise ValueError('unknown arm')


def plan_call_budgets(arm, index, rank, peer_bytes, helper, channels):
    """Fail-closed cumulative SQE bound for this arm's exact PACE call plan.

    `helper` is the frozen queue-metadata module (submission_budget) and
    `channels` its captured quiescent channel records. Assumes, like the
    original lanes, that no queue draining is inferred: the whole planned
    call sequence must fit the SQ ring and the 16-bit task counter bound.
    """
    calls = arm_pace_calls(arm, index)
    if calls == 0:
        per_call = None
        totals = [0] * len(channels)
    else:
        if not isinstance(peer_bytes, list) or len(peer_bytes) != WORLD:
            raise ValueError('peer byte vector required for PACE arms')
        per_call = helper.submission_budget(peer_bytes, rank, channels)
        totals = [per_call[channel]['total_sqes'] * calls for channel in range(len(channels))]
    if any(row['sq_head'] != 0 or row['sq_tail'] != 0 or row['reserved'] != 0 for row in channels):
        raise ValueError('fresh zero head/tail/task-counter metadata required')
    if any(total >= row['sq_depth'] or total >= 65536 for total, row in zip(totals, channels)):
        raise ValueError('planned PACE call sequence could wrap the SQ ring or task counter')
    return {'pace_calls': calls, 'per_call': per_call, 'cumulative_sqes': totals,
            'sq_depth': [row['sq_depth'] for row in channels],
            'head_tail_drain_inferred': False, 'no_ring_or_task_counter_wrap': True}


def segment_bytes(case_name, dtype, source, destination, rows, row_bytes):
    """Deterministic per-segment payload, byte-identical to the reverse-order
    lane's oracle when (case_name, dtype, row_bytes) are the frozen case's:
    same seed tuple ['shape-pilot-v1', name, dtype, source, destination],
    same shake_256 stream, same BF16 exponent-safe high-byte mask.

    Zero-row segments are REJECTED on purpose (defense): sparse matrices
    legitimately contain them and callers must skip zero counts exactly
    like the reverse lane's host_values and the sparse lane's expected_bytes
    do (device failure C1/round-7 was a join that forgot the skip).
    """
    if type(rows) is not int or type(row_bytes) is not int:
        raise ValueError('segment geometry must be ints')
    if rows <= 0:
        raise ValueError('zero-row segment requested; callers must skip zero counts')
    if row_bytes <= 0 or row_bytes % 32:
        raise ValueError('segment row width outside the verified ABI')
    seed = json.dumps(['shape-pilot-v1', case_name, dtype, source, destination],
                      separators=(',', ':')).encode()
    raw = bytearray(hashlib.shake_256(seed).digest(rows * row_bytes))
    if dtype == 'torch.bfloat16':
        # Clear bit0 and the sign of every BF16 high byte: exponent bounded.
        raw[1::2] = raw[1::2].translate(bytes(value & 0x7e for value in range(256)))
    elif dtype != 'torch.int8':
        raise ValueError('unsupported payload dtype')
    return bytes(raw)


def discover_acl_library(maps_text):
    """Find the single loaded libascendcl provider from /proc/self/maps text.

    Read-only introspection of the already-loaded ACL runtime, in the same
    spirit as the RTLD_NOLOAD provider checks; no pointer is dereferenced.
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


# ------------------------------------------------- independent review ---
# Single source of truth for the independent-review binding consumed by
# launch.verify(). The review contract issues review.json with exactly these
# field names and structures; the launcher and the CPU tests both consume
# this constant, so the two schemas can never drift apart again.
REVIEW_STATUS = 'APPROVED'
REVIEW_SCOPE = 'barrier_v3_arms_abc_p16'
REVIEW_MANIFEST_FIELD = 'source_manifest_sha256'
REVIEW_FILES_FIELD = 'reviewed_files'
REVIEW_GROUPS_FIELD = 'groups'
REVIEW_GROUPS = {'arm_a': [f'A{i}' for i in range(1, ARM_GROUPS['A'] + 1)],
                 'arm_b': [f'B{i}' for i in range(1, ARM_GROUPS['B'] + 1)],
                 'arm_c': [f'C{i}' for i in range(1, ARM_GROUPS['C'] + 1)]}
REVIEW_CONTRACT = {'status': REVIEW_STATUS, 'scope': REVIEW_SCOPE,
                   'manifest_field': REVIEW_MANIFEST_FIELD, 'files_field': REVIEW_FILES_FIELD,
                   'groups_field': REVIEW_GROUPS_FIELD, 'groups_structure': REVIEW_GROUPS,
                   'manifest_extra_files': ('source_manifest.json',)}


def check_review_binding(review, manifest_sha, files):
    """Validate an independent review against REVIEW_CONTRACT; fail closed.

    `manifest_sha` is the SHA-256 of the current source_manifest.json file
    itself and `files` the frozen source set the review must cover.
    """
    if not isinstance(review, dict):
        raise ValueError('review is not an object')
    if review.get('status') != REVIEW_STATUS:
        raise ValueError('review status is not ' + REVIEW_STATUS)
    if review.get('scope') != REVIEW_SCOPE:
        raise ValueError('review scope differs from the lane contract')
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64 \
            or any(c not in '0123456789abcdef' for c in manifest_sha):
        raise ValueError('invalid manifest identity for review binding')
    if review.get(REVIEW_MANIFEST_FIELD) != manifest_sha:
        raise ValueError('review does not bind the current source manifest')
    reviewed = review.get(REVIEW_FILES_FIELD)
    if not isinstance(reviewed, list) or len(reviewed) != len(set(reviewed)) \
            or set(reviewed) != set(files) | set(REVIEW_CONTRACT['manifest_extra_files']):
        raise ValueError('reviewed file coverage differs from the frozen source set')
    if review.get(REVIEW_GROUPS_FIELD) != REVIEW_GROUPS:
        raise ValueError('reviewed group plan differs from the frozen group order')
    return review


# ------------------------------------------------------------- metrics ---
def _require_pairs(pairs):
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('empty or non-list sample sequence')
    previous_end = None
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2 or any(type(v) is not int for v in pair) \
                or pair[1] <= pair[0]:
            raise ValueError('invalid host timestamp pair')
        if previous_end is not None and pair[0] < previous_end:
            raise ValueError('non-chronological sample sequence')
        previous_end = pair[1]


def _aligned(by_rank, count):
    if set(by_rank) != set(range(WORLD)):
        raise ValueError('rank coverage differs')
    if any(len(rows) != count for rows in by_rank.values()):
        raise ValueError('per-rank iteration count differs')


def iteration_max_rank_us(by_rank, count):
    """Per-iteration maximum of rank-local elapsed times, in microseconds."""
    _aligned(by_rank, count)
    return [max(by_rank[rank][i][1] - by_rank[rank][i][0] for rank in range(WORLD)) / 1e3
            for i in range(count)]


def iteration_envelope_us(by_rank, count):
    """Per-iteration earliest-start to latest-finish envelope, microseconds."""
    _aligned(by_rank, count)
    return [(max(by_rank[rank][i][1] for rank in range(WORLD))
             - min(by_rank[rank][i][0] for rank in range(WORLD))) / 1e3 for i in range(count)]


def summary(values):
    values = list(values)
    if not values:
        raise ValueError('summary of nothing')
    for value in values:
        if type(value) not in (int, float) or value != value or value in (float('inf'), float('-inf')):
            raise ValueError('summary requires finite numbers')
    if len(values) >= 2:
        q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    else:
        q1 = q3 = values[0]
    return {'n': len(values), 'median': statistics.median(values), 'q1': q1, 'q3': q3,
            'iqr': q3 - q1, 'min': min(values), 'max': max(values),
            'quantile_method': 'statistics.quantiles inclusive'}


def arm_a_metrics(samples_by_rank):
    maxima = iteration_max_rank_us(samples_by_rank, A_SAMPLES)
    envelopes = iteration_envelope_us(samples_by_rank, A_SAMPLES)
    return {'max_rank_us': maxima, 'envelope_us': envelopes,
            'median_max_rank_us': statistics.median(maxima),
            'median_envelope_us': statistics.median(envelopes)}


def arm_b_metrics(rounds_by_rank, index):
    """Matched-subarm medians plus per-block increments for drift inspection."""
    expected = arm_b_rounds(index)
    _aligned(rounds_by_rank, len(expected))
    for rank in range(WORLD):
        if tuple((row[0], row[1]) for row in rounds_by_rank[rank]) != expected:
            raise ValueError('arm B round sequence differs from the plan')
    measured = {sub: [] for sub in B_SUBARMS}
    for i, (phase, sub) in enumerate(expected):
        if phase == 'measure':
            measured[sub].append(max(rounds_by_rank[rank][i][3] - rounds_by_rank[rank][i][2]
                                     for rank in range(WORLD)) / 1e3)
    if any(len(values) != B_MEASURED_PER_SUBARM for values in measured.values()):
        raise ValueError('measured round count differs')
    medians = {sub: statistics.median(measured[sub]) for sub in B_SUBARMS}
    blocks = []
    positions = [i for i, (phase, _) in enumerate(expected) if phase == 'measure']
    for block in range(B_BLOCKS):
        chunk = positions[block * 2 * B_BLOCK_ROUNDS:(block + 1) * 2 * B_BLOCK_ROUNDS]
        per_sub = {sub: [] for sub in B_SUBARMS}
        for i in chunk:
            sub = expected[i][1]
            per_sub[sub].append(max(rounds_by_rank[rank][i][3] - rounds_by_rank[rank][i][2]
                                    for rank in range(WORLD)) / 1e3)
        blocks.append({sub: statistics.median(per_sub[sub]) for sub in B_SUBARMS})
    return {'b1_max_rank_us': measured['b1'], 'b2_max_rank_us': measured['b2'],
            'b1_median_us': medians['b1'], 'b2_median_us': medians['b2'],
            'increment_median_us': medians['b1'] - medians['b2'], 'block_medians_us': blocks}


def arm_c_metrics(iterations_by_rank):
    expected = arm_c_iteration_plan()
    _aligned(iterations_by_rank, len(expected))
    for rank in range(WORLD):
        if tuple((row[0], row[1]) for row in iterations_by_rank[rank]) != expected:
            raise ValueError('arm C iteration sequence differs from the plan')
    per_k = {k: [] for k in C_K_VALUES}
    for i, (phase, k) in enumerate(expected):
        if phase == 'measure':
            per_k[k].append(max(iterations_by_rank[rank][i][3] - iterations_by_rank[rank][i][2]
                                for rank in range(WORLD)) / 1e3)
    if any(len(values) != C_MEASURED_PER_K for values in per_k.values()):
        raise ValueError('measured iteration count differs')
    medians = {k: statistics.median(per_k[k]) for k in C_K_VALUES}
    return {'k_max_rank_us': {str(k): per_k[k] for k in C_K_VALUES},
            'k_median_us': {str(k): medians[k] for k in C_K_VALUES},
            'slope_k1_minus_k0_us': medians[1] - medians[0],
            'slope_k2_minus_k0_over_2_us': (medians[2] - medians[0]) / 2}


# ----------------------------------------------------------- validators ---
def _check_common(record, arm, index, event):
    if not isinstance(record, dict):
        raise ValueError('record is not an object')
    if (record.get('event'), record.get('arm'), record.get('group'), record.get('world'),
            record.get('clock')) != (event, arm, index, WORLD, 'host_monotonic_ns'):
        raise ValueError('record identity differs')
    if record.get('library_sha256') != LIB_SHA256 or record.get('shmem_sha256') != SHMEM_SHA256:
        raise ValueError('record provider identity differs')


def validate_arm_a_records(records, index):
    by_rank = {}
    for record in records:
        _check_common(record, 'A', index, 'arm_a_samples')
        if record.get('warmups') != A_WARMUPS:
            raise ValueError('arm A warmup count differs')
        _require_pairs(record['samples'])
        if len(record['samples']) != A_SAMPLES:
            raise ValueError('arm A sample count differs')
        if record.get('scope') != 'ctypes_enqueue_plus_same_stream_synchronize':
            raise ValueError('arm A timing scope differs')
        by_rank[record['rank']] = record['samples']
    if len(records) != WORLD or set(by_rank) != set(range(WORLD)):
        raise ValueError('arm A rank coverage differs')
    return by_rank


def validate_arm_b_records(records, index):
    expected = arm_b_rounds(index)
    by_rank = {}
    for record in records:
        _check_common(record, 'B', index, 'arm_b_samples')
        if record.get('peer_row_bytes') != B_PEER_ROW_BYTES:
            raise ValueError('arm B payload geometry differs')
        if record.get('pace_calls') != arm_pace_calls('B', index):
            raise ValueError('arm B PACE call count differs')
        if not isinstance(record.get('budget_cumulative_sqes'), list) \
                or any(type(v) is not int or v < 0 for v in record['budget_cumulative_sqes']):
            raise ValueError('arm B budget identity differs')
        rounds = record.get('rounds')
        if not isinstance(rounds, list) or len(rounds) != len(expected):
            raise ValueError('arm B round count differs')
        pairs, flags = [], []
        for row, (phase, sub) in zip(rounds, expected):
            if (row.get('phase'), row.get('sub')) != (phase, sub):
                raise ValueError('arm B round order differs from the plan')
            for key in ('output_equal', 'input_equal', 'splits_equal',
                        'input_tail_equal', 'output_tail_equal', 'split_tail_equal'):
                if row.get(key) is not True:
                    raise ValueError('arm B fail-closed check reported false: ' + key)
            pairs.append([row['start_ns'], row['end_ns']])
            flags.append((phase, sub))
        _require_pairs(pairs)
        by_rank[record['rank']] = [(phase, sub, pair[0], pair[1])
                                   for (phase, sub), pair in zip(flags, pairs)]
    if len(records) != WORLD or set(by_rank) != set(range(WORLD)):
        raise ValueError('arm B rank coverage differs')
    return by_rank


def validate_arm_c_records(records, index):
    expected = arm_c_iteration_plan()
    marks = arm_c_checkpoints()
    by_rank = {}
    for record in records:
        _check_common(record, 'C', index, 'arm_c_samples')
        if (record.get('case'), record.get('row_bytes'), record.get('dtype'),
                record.get('slot_bytes'), record.get('heap_bytes')) != (
                C_CASE_NAME, C_ROW_BYTES, C_DTYPE, C_SLOT_BYTES, C_HEAP_BYTES):
            raise ValueError('arm C frozen case identity differs')
        if record.get('scope') != 'pace_full_call_plus_k_barriers_through_device_synchronize':
            raise ValueError('arm C timing scope differs')
        iterations = record.get('iterations')
        if not isinstance(iterations, list) or len(iterations) != len(expected):
            raise ValueError('arm C iteration count differs')
        pairs = []
        for row, (phase, k) in zip(iterations, expected):
            if (row.get('phase'), row.get('k')) != (phase, k):
                raise ValueError('arm C iteration order differs from the plan')
            if type(row.get('k')) is not int or row['k'] not in C_K_VALUES:
                raise ValueError('arm C k value outside the plan')
            pairs.append([row['start_ns'], row['end_ns']])
        _require_pairs(pairs)
        checks = record.get('checks')
        if not isinstance(checks, list) or sorted(row.get('index') for row in checks) != sorted(marks):
            raise ValueError('arm C checkpoint coverage differs from the plan')
        for row in checks:
            position = row.get('index')
            if type(position) is not int or not 0 <= position < len(expected):
                raise ValueError('arm C checkpoint index outside the plan')
            phase, k = expected[position]
            if (row.get('phase'), row.get('k')) != (phase, k):
                raise ValueError('arm C checkpoint does not bind its iteration')
            if any(row.get(key) is not True for key in ('output_equal', 'input_equal', 'splits_equal',
                                                        'input_tail_equal', 'output_tail_equal',
                                                        'split_tail_equal')):
                raise ValueError('arm C exact checkpoint failed')
        budget = record.get('budget_cumulative_sqes')
        calls = record.get('pace_calls')
        if not isinstance(budget, list) or any(type(v) is not int or v < 0 for v in budget) \
                or calls != arm_pace_calls('C', index):
            raise ValueError('arm C budget identity differs')
        by_rank[record['rank']] = [(phase, k, pair[0], pair[1])
                                   for (phase, k), pair in zip(expected, pairs)]
    if len(records) != WORLD or set(by_rank) != set(range(WORLD)):
        raise ValueError('arm C rank coverage differs')
    return by_rank
