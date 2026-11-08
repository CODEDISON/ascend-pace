"""Pure CPU campaign plan, contracts, validators and metrics for the
concern-#4 four-arm optimized-HCCL balanced baseline campaign.

No framework/device/network imports; stdlib only. Every function runs on an
ordinary laptop and is shared by the device worker, the no-device precheck,
the launcher audit, the aggregate and the local test suite, so the measured
structure cannot drift between those roles.

The reviewed plan directory (atlas_optimized_hccl_plan_20261105_v1) is bound
by SHA-256 (PLAN.md + arms.json); the 16-group Williams schedule, trimming
ladder, census, estimator and direction rule are transcribed here from
arms.json and re-validated programmatically (FrozenPinTests re-derive them
from the bound file on every CPU test run).
"""
import ast
import hashlib
import json
import os
import re
import statistics
from pathlib import Path

WORLD = 16

# ------------------------------------------------------------- campaign ---
ARMS = ('H', 'HmR', 'HOPT', 'F')
ARM_LABELS = {'H': 'native', 'HmR': 'no-resize', 'HOPT': 'optimized-hccl',
              'F': 'pace-forced-sdma'}
HCCL_ARMS = ('H', 'HmR', 'HOPT')  # real Work.wait HCCL arms; delegate to the
# frozen uninstrumented-timing driver with REV2_NO_RESIZE_ARM in:
DRIVER_ARM_ENV = {'H': 'native', 'HmR': 'no-resize', 'HOPT': 'no-resize'}

# One complete Williams design of 4 balanced sequences (arms.json schedule).
BLOCKS = {1: ('H', 'HmR', 'F', 'HOPT'),
          2: ('HmR', 'HOPT', 'H', 'F'),
          3: ('HOPT', 'F', 'HmR', 'H'),
          4: ('F', 'H', 'HOPT', 'HmR')}
GROUPS = tuple(f'g{index:02d}' for index in range(1, 17))
GROUP_REP = {}
GROUP_ARM = {}
for _rep, _arms in BLOCKS.items():
    for _position, _arm in enumerate(_arms, start=1):
        _key = f'g{len(GROUP_REP) + 1:02d}'
        GROUP_REP[_key] = _rep
        GROUP_ARM[_key] = _arm
CROSS_BLOCK_ADJACENCIES = (('HOPT', 'HmR'), ('F', 'HOPT'), ('H', 'F'))

# Pre-registered trimming ladder (downgrade only, fixed drop order).
LADDER = {'L0': (), 'L1': ('rep4',), 'L2': ('rep4', 'rep3')}
LADDER_FLOOR = 'L2'
TRIM_SINGLE_TRIGGER_S = 480.0     # any observed group container > 8 min
TRIM_CONSECUTIVE_TRIGGER_S = 360.0  # two consecutive groups > 6 min
WINDOW_SECONDS = 4 * 3600.0       # authorized exclusive device window

# Requests per fresh group (all arms, matched contract).
GATE_REQUESTS = 1
WARMUP_REQUESTS = 3
MEASURED_REQUESTS = 20
TOTAL_REQUESTS = 24
OUTPUT_TOKENS = 1
SEED = 2027

# Census (per rank). Frozen-predecessor formats, verified against the REAL
# persisted evidence (see frozen_census_fixture.json provenance):
# - HCCL arms: the frozen gate trace counts 64 DISTINCT collectives
#   (alltoallv_global_calls == 64, each executed by all 16 ranks); each
#   rank's capture record carries counts
#   {'collective': 64, 'forward': 2, 'method': 64, 'runner': 3} and events
#   with 64 method_enter rows (ALL backend_path 'All2AllV': 32 dispatch +
#   32 combine), 64 alltoallv_begin rows, 2 ALLTOALL selectors on 2
#   execute_model forwards. NOTE: the arms.json census parenthetical
#   "(2 All2AllV + 62 MC2)" does NOT match the six persisted groups (all
#   64 collectives ride All2AllV methods across 2 execute_model forwards);
#   the lane binds the empirical invariant. The 2+62 split belongs to the
#   F arm's logical-call accounting (sync-model qualification).
# - F arm: the sync-model session snapshot rows carry
#   counts['python_pace_calls']/['python_hccl_calls'] (62/2) and 64 call
#   records keyed by row['backend'] in {'PACE', 'HCCL'}.
CENSUS_LOGICAL_CALLS = 64
HCCL_CAPTURE_COUNTS = {'collective': 64, 'forward': 2, 'method': 64, 'runner': 3}
CENSUS_METHOD_BACKEND = 'All2AllV'
CENSUS_DISPATCH_METHODS = 32
CENSUS_COMBINE_METHODS = 32
CENSUS_SELECTOR_METHOD = 'ALLTOALL'
CENSUS_FORWARD_KIND = 'execute_model'
CENSUS_PACE = 62
CENSUS_NATIVE_ZERO_TOTAL = 2
PACE_COUNTS_KEYS = ('python_logical_calls', 'python_pace_calls', 'python_hccl_calls')

# Port plan: 16 campaign slots + probe + f_proof + two pilot slots.
AUX_JOBS = ('probe', 'f_proof', 'pilot0', 'pilot1')
CAMPAIGN_SLOTS = GROUPS + AUX_JOBS
SHMEM_PORT_BASE = 26801
MASTER_PORT_BASE = 26901


def ports(job):
    if job not in CAMPAIGN_SLOTS:
        raise ValueError('job outside the reviewed campaign')
    slot = CAMPAIGN_SLOTS.index(job)
    return SHMEM_PORT_BASE + 2 * slot, MASTER_PORT_BASE + 2 * slot


def validate_schedule(groups_by_id=None):
    """Re-derive every Williams property the plan claims (fail closed)."""
    if groups_by_id is not None:
        if [(row['id'], row['rep'], row['arm']) for row in groups_by_id] != \
                [(key, GROUP_REP[key], GROUP_ARM[key]) for key in GROUPS]:
            raise ValueError('arms.json schedule differs from the lane plan')
    if tuple(GROUPS) != tuple(f'g{i:02d}' for i in range(1, 17)):
        raise ValueError('group ids differ')
    for arm in ARMS:
        if sum(1 for key in GROUPS if GROUP_ARM[key] == arm) != 4:
            raise ValueError('each arm needs exactly four groups')
    leaders = [BLOCKS[rep][0] for rep in sorted(BLOCKS)]
    if sorted(leaders) != sorted(ARMS):
        raise ValueError('each arm must lead exactly once')
    for position in range(4):
        occupants = sorted(BLOCKS[rep][position] for rep in sorted(BLOCKS))
        if occupants != sorted(ARMS):
            raise ValueError('within-block position balance differs')
    within = []
    for rep in sorted(BLOCKS):
        within += list(zip(BLOCKS[rep], BLOCKS[rep][1:]))
    ordered = [pair for pair in _ORDERED_PAIRS()]
    if sorted(within) != sorted(ordered):
        raise ValueError('the 12 ordered adjacent pairs must occur exactly once')
    boundary = list(zip([BLOCKS[r][-1] for r in (1, 2, 3)],
                        [BLOCKS[r][0] for r in (2, 3, 4)]))
    if tuple(boundary) != CROSS_BLOCK_ADJACENCIES:
        raise ValueError('declared cross-block adjacencies differ')
    return True


def _ORDERED_PAIRS():
    return [(a, b) for a in ARMS for b in ARMS if a != b]


def ladder_groups(level):
    if level not in LADDER:
        raise ValueError('unknown ladder level')
    dropped = LADDER[level]
    return tuple(key for key in GROUPS if GROUP_REP[key] not in
                 tuple(int(name[3:]) for name in dropped))


def ladder_step(prior_receipts, requested_level):
    """Monotone ladder admission: L0 -> L1 -> L2 only, each increment citing
    prior observed receipts (single > 8 min, two consecutive > 6 min, or an
    extrapolation of rep3+rep4 beyond the authorized window)."""
    order = ('L0', 'L1', 'L2')
    if requested_level not in order:
        raise ValueError('unknown ladder level')
    if not prior_receipts and requested_level != 'L0':
        raise ValueError('the first group always starts at L0')
    levels = [receipt.get('ladder_level') for receipt in prior_receipts]
    if any(value not in order for value in levels):
        raise ValueError('prior receipt ladder level differs')
    previous = levels[-1] if levels else 'L0'
    if order.index(requested_level) < order.index(previous):
        raise ValueError('the ladder may only move down')
    if requested_level == previous:
        return None
    if order.index(requested_level) > order.index(previous) + 1:
        raise ValueError('the ladder moves one step at a time')
    durations = [receipt.get('duration_seconds') for receipt in prior_receipts]
    if any(type(value) not in (int, float) or value <= 0 for value in durations):
        raise ValueError('prior receipts lack observed durations')
    single = [d for d in durations if d > TRIM_SINGLE_TRIGGER_S]
    consecutive = any(a > TRIM_CONSECUTIVE_TRIGGER_S and b > TRIM_CONSECUTIVE_TRIGGER_S
                      for a, b in zip(durations, durations[1:]))
    mean_duration = statistics.median(durations)
    remaining = len(ladder_groups(previous)) - len(prior_receipts)
    extrapolated_end = sum(durations) + max(remaining, 0) * mean_duration
    if not single and not consecutive and extrapolated_end <= WINDOW_SECONDS:
        raise ValueError('ladder increment lacks the pre-registered trigger')
    return {'requested_level': requested_level, 'previous_level': previous,
            'single_over_trigger': [d for d in single],
            'consecutive_over_trigger': consecutive,
            'extrapolated_campaign_seconds': extrapolated_end,
            'window_seconds': WINDOW_SECONDS,
            'trigger_rule': 'single>8min or consecutive>6min or extrapolation beyond 4h'}


# ------------------------------------------------------------ O2 probe ---
# Pre-registered honorability probe contract. K1 is HCCL_BUFFSIZE (a real
# documented env knob of the pinned CANN: hcomm v9.0.0 docs, comm config
# priority "hcclBufferSize(comm) > HCCL_BUFFSIZE(global) > default"). K2 may
# be at most one additional knob, admissible ONLY if the probe receipt itself
# contains a pinned-image log/doc line naming the knob AND echoing the probe
# value; post-hoc invention is forbidden. No reliable echo evidence for any
# knob exists in the local materials, so the lane DEFAULTS TO P0 and only a
# produced, audited receipt can move it to P1.
PROBE_CANDIDATES = ('K1',)
PROBE_KNOBS = {'K1': 'HCCL_BUFFSIZE'}
PROBE_VALUE = '333'  # distinctive, never a plausible default (default 200/400)
PROBE_LOG_PATTERN = re.compile(r'(?i)[^a-z]HCCL_BUFFSIZE[^0-9]{0,40}333(?![0-9])')


def judge_probe_receipt(log_text, matched_lines):
    """Fail-closed adoption judge. A knob is honored ONLY if the probe's own
    HCCL log contains a line naming the knob and echoing the probe value."""
    if not isinstance(log_text, str) or not isinstance(matched_lines, list):
        raise ValueError('probe receipt shape differs')
    honored = []
    for line in matched_lines:
        if not isinstance(line, str) or not PROBE_LOG_PATTERN.search(line):
            raise ValueError('probe receipt line does not match the pre-registered pattern')
    lines = [raw for raw in log_text.splitlines() if PROBE_LOG_PATTERN.search(raw)]
    if lines == matched_lines and lines:
        honored.append('K1')
    elif matched_lines:
        raise ValueError('probe matched lines differ from the raw log rescan')
    return {'path': 'P1' if honored else 'P0', 'honored': honored,
            'knobs': dict(PROBE_KNOBS), 'probe_value': PROBE_VALUE,
            'criterion': 'log line naming HCCL_BUFFSIZE and echoing probe value 333'}


def envopt_malformed(env_delta):
    """P1 env delta must be exactly one HCCL_BUFFSIZE with a sane value."""
    if set(env_delta) != {'HCCL_BUFFSIZE'}:
        return True
    value = str(env_delta['HCCL_BUFFSIZE'])
    return not re.fullmatch(r'[1-9][0-9]{0,4}', value)


def hopt_env(path, pilot_selection=None):
    """Exact HOPT environment delta. P0: none (HOPT == HmR config, H4 null
    control). P1: the single frozen pilot-selected knob/value."""
    if path == 'P0':
        return {}
    if path == 'P1':
        if not isinstance(pilot_selection, dict) or \
                set(pilot_selection) != {'knob', 'value'} or \
                pilot_selection.get('knob') != 'HCCL_BUFFSIZE' or \
                not re.fullmatch(r'[1-9][0-9]{0,4}', str(pilot_selection.get('value', ''))):
            raise ValueError('P1 requires one frozen pilot selection for K1')
        return {'HCCL_BUFFSIZE': str(pilot_selection['value'])}
    raise ValueError('unknown probe path')


# ------------------------------------------------------------- identity ---
# Frozen pins, programmatically transcribed and FrozenPin-tested against the
# local source-of-truth artifacts (see test_cpu.py).
IMAGE = 'sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f'
SHMEM_SHA256 = 'c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71'
LIB_SHA256 = '24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9'
HELPER_SHA256 = '0d7fde8e4eb1f0c8c312412cfc2bb6ee6f5df116ea48e2e27aae2feadef962be'
LAYOUT_SHA256 = 'e844a5175ca3de3e4a54ef54723b76df6665607c4268ad29cc3ed541d7bf6154'
GUARD_LAUNCH_SHA = 'b4d96c96b5e304d8b0cf74c264d7b6006c158b018c7243bf055a74b79d91ffc2'
GUARD_MANIFEST_SHA = '288fb6913cab6181af5021505be0534b3105704c75358cb3252bf1cd31246d94'
HCCL_PATH = '/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so'
LIBHCCL_SHA = '1188f6acb2df4de12b887ab3d0a270c59d0b745e11f692929a6a6635c8cf3e4c'
# Frozen predecessor-lane source manifests (device-host mounted, SHA-checked).
CANDIDATE_MANIFEST = '64f1da2338e88bbab142ed49913df27c4d02f62d6252b4ef321ad77b0f9b2ae4'
RUNNER_MANIFEST = '365019d69a6e1f01f534976d9ecd0471c156cfa33b92d7b672d104ccce677be8'
TIMING_MANIFEST = '1bec868050fffdedf022aa1ec4185e77e8923c94a29f22c582fa08f712d78a8a'
SYNC_MODEL_MANIFEST = 'c755584d1d23c061840959e4d8b9ed8dbc217f86f0f000f0d33aad61d219c3c8'
# The sync-model lane's model_extension imports a runtime module FAMILY
# (contract/controller/frontend/padded_buffers/runtime/dependencies/
# queue_metadata + dataset.json/layout.json) that does NOT live in the
# sync-model lane directory. Its authoritative home is the sync
# qualification HARNESS (v3): model_contract.QUALIFICATION_SHA binds this
# exact manifest and sources() verifies per-module digests against it —
# the v2 qualification dir (fa29505c…) also carries the modules and
# satisfied imports at round-3/4, but sources() would reject it; the mount
# was corrected to v3 in round-5 (device failure lineage 54167a66/216a2400/
# d4e26ece).
QUALIFICATION_MANIFEST = 'f3b36ae0adf3906626f66cc93e094929aaeb24359cb394518807f6aa98c14c0b'
F_RUNTIME_FAMILY = ('contract.py', 'controller.py', 'frontend.py', 'padded_buffers.py',
                    'runtime.py', 'dependencies.py', 'queue_metadata.py',
                    'dataset.json', 'layout.json')
# Data files the delegated chain accesses BY PATH (AST audit of the frozen
# chain sources, round-5 root fix: the import closure cannot see path+SHA
# file reads — device failure d4e26ece was exactly one such file).
LEGACY_HOST_DIR = 'experiments/postreview_olmoe_o1_interaction_o32_20260821_v2'
MODEL_HOST_DIR = 'models/OLMoE-1B-7B-0125-Instruct'
MIXED_RESULTS_HOST_DIR = 'results/diagnostic/revision2_olmoe_o1_sync_qualification_20261028_v3/p16'
PROMPT_HELPER_SHA = '1c0d4073714d82ee8171fa6ab5dfe4f9cabcb6900b2cd7c8d243ea4338bf9764'
MIXED_COMPLETION_SHA = '263b2b59ec686f039c231953783db8648158f913bebac4e93ff3dacf79b6478a'
DELEGATED_DATA_FILES = (
    # (container path, kind, sha256 pin or None, consumer)
    ('/workspace/legacy/benchmark_olmoe_fixed_batch.py', 'file', PROMPT_HELPER_SHA,
     'run_capture/run_o1/run_timing PROMPT_SOURCE_SHA prompt helper'),
    ('/workspace/model', 'dir', None,
     'run_capture.check_model + prompt_batch model dir (digest-checked via '
     '/workspace/capture/model_manifest.json by the delegated flow)'),
    ('/workspace/capture/model_manifest.json', 'file', None,
     'run_capture.check_model manifest'),
    ('/workspace/capture/runtime_pins.json', 'file', None,
     'capture_contract.validate_workers runtime pin table'),
    ('/workspace/mixed_qualification/completion.json', 'file', MIXED_COMPLETION_SHA,
     'model_contract.sources prior mixed qualification receipt '
     '(env REV2_QUALIFICATION_COMPLETION_SHA256)'),
    ('/workspace/reference/request_raw.json', 'file', '40ea078535ccae4b70bf80b7affbe3f750a6a73d4e7048d00539437452bda7bc',
     'run_o1/run_timing fixed native O1 reference (== REFERENCE_RAW_SHA)'),    ('/workspace/qualification/dataset.json', 'file', None,
     'contract.dataset() qualified call matrices'),
    ('/workspace/qualification/layout.json', 'file', None,
     'dependencies layout pin (family member)'),
    ('/workspace/cashflex/cashflex/ops/hpa_alltoallv.py', 'file', None,
     'model_extension wrapper SHA check (BINARY-relative)'),
)
# The capture lane owns capture_contract/capture_core/run_capture/
# worker_extension — imported by BARE NAME from run_timing.py/run_o1.py/
# no_resize_worker.py (device-observed f_proof failure 216a2400:
# ModuleNotFoundError 'capture_contract'). Mounted read-only and on
# PYTHONPATH; its manifest was already pinned (arms.json capture_manifest).
CAPTURE_HOST_DIR = 'experiments/revision2_model_split_capture_20261028_v1'
CAPTURE_CONTAINER_DIR = '/workspace/capture'

# The seven module-home directories of the delegated chain, in the EXACT
# entry-script PYTHONPATH order (resolution semantics = sys.path order).
DELEGATED_MODULE_ROOTS = (
    '/workspace/lane',                # plan, restoration, worker (this lane)
    '/workspace/qualification',       # sync-model runtime family (round-3)
    '/workspace/no_resize_candidate', # patch_source (round-4 closure finding)
    '/workspace/capture',             # capture_contract et al. (round-4 fix)
    '/workspace/no_resize_lane',      # run_o1, no_resize_worker, patch_install, ...
    '/workspace/native_timing_lane',  # run_timing, timing_worker, timing_contract, bindings
    '/workspace/model_hook',          # model_extension + its lane-local modules
)
# Delegated-chain entry files whose bare-import transitive closure must
# resolve entirely inside DELEGATED_MODULE_ROOTS (proven in-container by the
# no-device precheck; device failures 54167a66/216a2400 were exactly
# closure members whose home was missing from the path).
DELEGATED_ENTRY_FILES = (
    '/workspace/native_timing_lane/run_timing.py',   # HCCL-arm delegation
    '/workspace/model_hook/model_extension.py',      # F-arm composition
    '/workspace/lane/worker.py',                     # the lane worker itself
)
# Delegated-chain environment contract (grep audit of os.environ across the
# frozen chain; device failure g01: REV2_NATIVE_TIMING_MANIFEST unset).
# (variable, arms, value, setter) — setter 'worker' means run_hccl_arm/
# run_f_arm must set it from the plan pin before delegating; 'group-env'
# means launch.group_env already provides it (precheck asserts presence).
DELEGATED_ENV = (
    ('REV2_NATIVE_TIMING_MANIFEST', ('H', 'HmR', 'HOPT'), 'TIMING_MANIFEST', 'worker'),
    ('REV2_NO_RESIZE_SOURCE_MANIFEST', ('H', 'HmR', 'HOPT'), 'RUNNER_MANIFEST', 'worker'),
    ('REV2_MODEL_MANIFEST_SHA256', ('F',), 'SYNC_MODEL_MANIFEST', 'worker'),
    ('REV2_NO_RESIZE_ARM', ('H', 'HmR', 'HOPT'), 'by arm (native/no-resize)', 'group-env'),
    ('REV2_BACKEND', ('H', 'HmR', 'HOPT'), 'hccl', 'group-env'),
    ('REV2_OUTPUT_TOKENS', ('H', 'HmR', 'HOPT'), '1', 'group-env'),
    ('REV2_RESULT_DIR', ('H', 'HmR', 'HOPT'), '/workspace/results', 'group-env'),
    ('VLLM_USE_V1', ('H', 'HmR', 'HOPT'), '1', 'group-env'),
    ('VLLM_ASCEND_ENABLE_FUSED_MC2', ('H', 'HmR', 'HOPT'), '0', 'group-env'),
    ('REV2_RUN_NONCE', ('F',), 'campaign-group nonce', 'group-env'),
    ('HPA_SHMEM_IP_PORT', ('F',), 'tcp://127.0.0.1:<shmem_port>', 'group-env'),
    ('REV2_QUALIFICATION_COMPLETION_SHA256', ('F',), 'MIXED_COMPLETION_SHA', 'group-env'),
)
THIRD_PARTY_ROOTS = frozenset(('torch', 'torch_npu', 'numpy', 'vllm', 'vllm_ascend',
                               'cashflex', 'transformers'))


def iter_bare_imports(source_text):
    """Root module names of every bare-name import in python source (AST)."""
    roots = set()
    for node in ast.walk(ast.parse(source_text)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split('.')[0])
    return roots


def resolve_delegated_closure(entry_paths, roots, stdlib_names, third_party=THIRD_PARTY_ROOTS):
    """AST transitive closure of the delegated chain over an ordered root map.

    `entry_paths` are entry FILES; `roots` is an ordered iterable of
    (container_dir, search_dir) pairs — first hit wins, exactly like
    sys.path. Returns {'edges': sorted (importer, module, container) rows,
    'unresolved': sorted (module, importer) rows}. stdlib_names should be
    sys.stdlib_module_names at call time (plus '__future__').
    """
    by_dir = {str(container): str(search) for container, search in roots}
    order = [str(container) for container, _ in roots]

    def find(name):
        for container in order:
            candidate = Path(by_dir[container]) / (name + '.py')
            if candidate.is_file():
                return container, candidate
        return None, None

    seen_files, edges, unresolved = set(), [], set()
    stack = [Path(entry) for entry in entry_paths]
    while stack:
        path = stack.pop()
        key = str(path)
        if key in seen_files:
            continue
        seen_files.add(key)
        text = path.read_text()
        for name in sorted(iter_bare_imports(text)):
            if name in stdlib_names or name in third_party or name == '__future__':
                continue
            container, found = find(name)
            if found is None:
                unresolved.add((name, path.name))
                edges.append((path.name, name, 'UNRESOLVED'))
            else:
                edges.append((path.name, name, container))
                if str(found) not in seen_files:
                    stack.append(found)
    return {'edges': sorted(set(edges)), 'unresolved': sorted(unresolved),
            'files': sorted(Path(key).name for key in seen_files)}
CAPTURE_MANIFEST = 'db33506846d1d007fdc2fd02d22c4f3f857a8322567015cc2ef89ea6b2588d5c'
PARITY_HOST_MANIFEST = 'f57b4fd4af500a89d639065c60f45c9f6664660cd8063311b592fe2e1861312a'
# Prior PASS receipts revalidated by exact hash (PLAN section 8.6).
PARITY_NATIVE_COMPLETION = '5265f7c20c3826cf681bf1cf835e6681ba3b94d90faf2765b88811bb2cb6d2fc'
PARITY_NORESIZE_COMPLETION = 'd2413c0b6b7dc3bd3e90a785427f66f57802f765642cd23f18dfc4b496fbaaf6'
SYNC_MODEL_COMPLETION = '3fbf5f5e5c99709a4b40f42f62d3b5963101379f4cfc41609f9925e75f431256'
REFERENCE_RAW_SHA = '40ea078535ccae4b70bf80b7affbe3f750a6a73d4e7048d00539437452bda7bc'
PROMPT_BATCH_SHA = 'ce5e1795710d16d49d4b65f3bffd4c1ad07556cc27d2d049f0dbd2b55eaebdc8'
# Bound plan documents.
ARMS_JSON_SHA = 'dbdb1aef0b72210f684cfe6b33e708103b4e964ea9fa5c64e232025033397232'
PLAN_MD_SHA = 'd3ac4df9be5015d8d93247f3b7aaa5b50e0bcaf7ec651ffc3c80455db30ace14'

# F-arm SHMEM geometry (sync-model v2 qualified).
SLOT_BYTES = 96 * 1024 ** 2
HEAP_BYTES = 4 * 1024 ** 3
F_SQ_BOUNDARIES = ('gate', 'warmup1', 'warmup2', 'warmup3')

# ------------------------------------------------------ review contract ---
# SINGLE contract constant (v3-class BLOCKER prevention): launch and the CPU
# tests consume exactly this structure through plan.validate_review.
REVIEW_SCHEMA = 'revision2.optimized-hccl-4arm-review.v1'
REVIEW_STATUS_APPROVED = 'APPROVED'
REVIEW_SCOPE = 'optimized_hccl_4arm_p16_olmoe_o1'
REVIEW_GROUPS = {'campaign': list(GROUPS),
                 'arm_map': {'g01': 'H', 'g02': 'HmR', 'g03': 'F', 'g04': 'HOPT',
                             'g05': 'HmR', 'g06': 'HOPT', 'g07': 'H', 'g08': 'F',
                             'g09': 'HOPT', 'g10': 'F', 'g11': 'HmR', 'g12': 'H',
                             'g13': 'F', 'g14': 'H', 'g15': 'HOPT', 'g16': 'HmR'}}
REVIEWED_FILES = None  # set below once FILES exists
FILES = {'DESIGN.md', 'plan.py', 'restoration.py', 'worker.py', 'supervisor.py',
         'precheck.py', 'launch.py', 'test_cpu.py', 'make_manifest.py',
         'observed_npu_info_fixture.json', 'frozen_census_fixture.json'}
REVIEWED_FILES = FILES | {'source_manifest.json'}
F_GATE_STATES = ('PROVEN', 'NOT_REQUIRED')  # NOT_REQUIRED only for an
# explicitly degraded three-arm campaign (review must say so).


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
    if review.get('arms_json_sha256') != ARMS_JSON_SHA:
        raise ValueError('review does not bind the frozen arms.json')
    if review.get('scope') != REVIEW_SCOPE:
        raise ValueError('review scope differs')
    files = review.get('reviewed_files')
    if not isinstance(files, list) or len(files) != len(set(files)) \
            or sorted(files) != sorted(REVIEWED_FILES):
        raise ValueError('reviewed file coverage differs from the lane source set')
    if review.get('groups') != REVIEW_GROUPS:
        raise ValueError('reviewed group plan differs from the registered campaign')
    if review.get('f_restoration_gate') not in F_GATE_STATES:
        raise ValueError('review does not state the F-arm restoration gate outcome')
    if review.get('f_restoration_gate') == 'NOT_REQUIRED' and review.get('degraded_three_arm') is not True:
        raise ValueError('three-arm degradation must be explicit in the review')
    if review.get('probe_path') not in ('P0', 'P1'):
        raise ValueError('review does not declare the probe path')
    if review.get('probe_path') == 'P1':
        env_delta = review.get('hopt_env')
        if not isinstance(env_delta, dict) or not env_delta \
                or envopt_malformed(env_delta):
            raise ValueError('P1 review must bind one frozen pilot-selected HCCL env')
    elif review.get('hopt_env') not in (None, {}):
        raise ValueError('P0 review must not carry an HOPT env delta')
    return {'schema': REVIEW_SCHEMA, 'status': REVIEW_STATUS_APPROVED,
            'source_manifest_sha256': manifest_sha, 'arms_json_sha256': ARMS_JSON_SHA,
            'scope': REVIEW_SCOPE, 'reviewed_files': sorted(REVIEWED_FILES),
            'groups': REVIEW_GROUPS, 'f_restoration_gate': review['f_restoration_gate'],
            'degraded_three_arm': review.get('degraded_three_arm') is True,
            'probe_path': review['probe_path'], 'hopt_env': review.get('hopt_env')}


def delegated_env(arm):
    """Env vars the WORKER must set for one arm, resolved from plan pins
    (the g01 device failure was REV2_NATIVE_TIMING_MANIFEST reaching
    timing_contract.check_source empty). REV2_NO_RESIZE_ARM rides along for
    the HCCL arms from DRIVER_ARM_ENV so the precheck simulation and the
    worker loop share one definition (worker's explicit line stays — it is
    idempotent under the same value)."""
    if arm not in ARMS:
        raise ValueError('unknown arm')
    env = {}
    for name, arms, source, setter in DELEGATED_ENV:
        if arm in arms and setter == 'worker':
            env[name] = globals()[source]
    if arm in HCCL_ARMS:
        env['REV2_NO_RESIZE_ARM'] = DRIVER_ARM_ENV[arm]
    if not env:
        raise ValueError('no worker-set delegated env for arm ' + arm)
    return env


def required_env_names(arm):
    """Every env var the delegated chain reads for this arm, whatever the
    setter — the precheck readiness proof asserts each is present."""
    return tuple(name for name, arms, _value, _setter in DELEGATED_ENV if arm in arms)


def prime_f_worker_env(store):
    """Prime the F-arm WORKER-PROCESS env (spawn defense, device failure
    e9f94a35): vLLM's multiproc executor spawns worker processes during
    engine init, so a variable set in the driver process AFTER LLM
    construction never reaches them — the spawned child's environ is a
    snapshot of the parent's at spawn time. Docker-level (-e) variables
    ARE present in every process from container start; the worker-settable
    manifest pins are setdefault-ed here (container/explicit values win)
    and the group-env-provided names are asserted present, fail-closed.
    `store` is a MutableMapping: os.environ inside the worker process, a
    plain dict in the CPU tests (spawn semantics simulated by a store that
    contains ONLY the docker-level variables)."""
    for name, value in delegated_env('F').items():
        store.setdefault(name, str(value))
    missing = [name for name in required_env_names('F') if not store.get(name, '')]
    if missing:
        raise ValueError('F worker env missing after priming: ' + ', '.join(missing))
    return {name: store[name] for name in required_env_names('F')}


# ---------------------------------------------------------- providers ----
def dladdr_provider_path(info_fname):
    """dladdr Dl_info file name -> provider object path (fname ONLY; the
    symbol name sname must never participate in the path comparison)."""
    if not info_fname:
        raise ValueError('dladdr returned no containing file name')
    return os.path.realpath(os.fsdecode(info_fname))


def discover_acl_library(maps_text):
    """Single loaded libascendcl provider from /proc/self/maps text."""
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
PACE_SCHEMA_TRUTH = ('cashflex::hpa_alltoallv_device_splits(Tensor _0, Tensor _1, '
                     'Tensor _2, Tensor _3, int _4, int _5, int _6, int _7) -> Tensor _0')


def check_pace_schema(text):
    """Pin the frozen PACE device-split operator ABI (device-measured truth
    string above; namespaced name pinned, per-item type words pinned,
    compiler suffix names not pinned, return segment isomorphic to args)."""
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


# ------------------------------------------------------------ census -----
def _sixteen_rank_rows(rows):
    if not isinstance(rows, list) or len(rows) != WORLD \
            or [row.get('rank', index) for index, row in enumerate(rows)] != list(range(WORLD)):
        raise ValueError('sixteen rank records required')
    return rows


def normalize_hccl_census(after_gate_rows, validated_trace):
    """Per-rank census from the frozen native capture evidence (real format).

    Global: validated_gate_trace['alltoallv_global_calls'] counts DISTINCT
    collectives and must equal 64 (run_timing.py:60 asserts the same; each
    collective is executed by all 16 ranks). Per rank: capture['counts']
    must equal the frozen counter set; the events must show 64 method_enter
    rows all under backend 'All2AllV' (32 dispatch + 32 combine), 64
    alltoallv_begin rows inside those methods, and 2 ALLTOALL selectors on
    2 execute_model forwards (capture_contract.validate_worker L107-186,
    real persisted sample: pair2-no-resize request_raw.json after_gate).
    Fail closed on any unknown shape.
    """
    if not isinstance(validated_trace, dict) or \
            validated_trace.get('alltoallv_global_calls') != CENSUS_LOGICAL_CALLS:
        raise ValueError('frozen gate trace global count differs from 64 distinct collectives')
    ranks = []
    for row in _sixteen_rank_rows(after_gate_rows):
        capture = row.get('capture')
        if not isinstance(capture, dict):
            raise ValueError('frozen per-rank capture record unavailable')
        if capture.get('counts') != HCCL_CAPTURE_COUNTS:
            raise ValueError('frozen per-rank recorder counters differ')
        events = capture.get('events')
        if not isinstance(events, list):
            raise ValueError('frozen per-rank event rows unavailable')
        methods = [event for event in events if event.get('kind') == 'method_enter']
        begins = [event for event in events if event.get('kind') == 'alltoallv_begin']
        selectors = [event for event in events if event.get('kind') == 'selector']
        forwards = [event for event in events if event.get('kind') == 'forward_enter']
        if len(methods) != CENSUS_LOGICAL_CALLS or len(begins) != CENSUS_LOGICAL_CALLS:
            raise ValueError('per-rank method/collective coverage differs from 64')
        by_position = {}
        method_ids = set()
        for event in methods:
            if event.get('backend_path') != CENSUS_METHOD_BACKEND:
                raise ValueError(f"unexpected backend_path: {event.get('backend_path')!r}")
            by_position[event.get('position')] = by_position.get(event.get('position'), 0) + 1
            method_ids.add(event.get('method_id'))
        if by_position != {'dispatch': CENSUS_DISPATCH_METHODS, 'combine': CENSUS_COMBINE_METHODS}:
            raise ValueError('per-rank dispatch/combine split differs from 32+32')
        for event in begins:
            if event.get('method_id') not in method_ids:
                raise ValueError('collective outside the captured methods')
        if [event.get('method') for event in selectors] != [CENSUS_SELECTOR_METHOD] * 2:
            raise ValueError('per-rank selector census differs from ALLTOALL x2')
        if [event.get('runner_kind') for event in forwards] != [CENSUS_FORWARD_KIND] * 2:
            raise ValueError('per-rank forward census differs from execute_model x2')
        ranks.append({'rank': row['rank'], 'alltoallv_class_calls': CENSUS_LOGICAL_CALLS,
                      'collectives': len(begins), 'method_backends': [CENSUS_METHOD_BACKEND],
                      'dispatch_methods': by_position['dispatch'],
                      'combine_methods': by_position['combine'],
                      'selector_methods': [CENSUS_SELECTOR_METHOD] * 2,
                      'forward_kinds': [CENSUS_FORWARD_KIND] * 2})
    return {'ranks': ranks, 'global_distinct_collectives': CENSUS_LOGICAL_CALLS,
            'source': 'frozen validated gate capture (events/counts)'}


def normalize_pace_census(snapshot_rows):
    """Per-rank census from the frozen sync-model session snapshot (real
    format): counts['python_pace_calls']/['python_hccl_calls'] primary,
    cross-checked against the 64 call records keyed by row['backend']
    ('PACE'/'HCCL'; model_state.py:88, snapshot L108-113; real persisted
    sample: sync-model v2 p16 payload request_raw.json['after'])."""
    ranks = []
    for row in _sixteen_rank_rows(snapshot_rows):
        counts = row.get('counts')
        if not isinstance(counts, dict) or \
                any(type(counts.get(key)) is not int for key in PACE_COUNTS_KEYS):
            raise ValueError('session snapshot counts unavailable')
        pace, hccl = counts['python_pace_calls'], counts['python_hccl_calls']
        if (pace + hccl, pace, hccl) != (CENSUS_LOGICAL_CALLS, CENSUS_PACE,
                                         CENSUS_NATIVE_ZERO_TOTAL):
            raise ValueError('session counts differ from the qualified 62+2 split')
        records = row.get('records')
        if isinstance(records, list):
            backends = [record.get('backend') for record in records]
            if len(backends) != CENSUS_LOGICAL_CALLS \
                    or backends.count('PACE') != CENSUS_PACE \
                    or backends.count('HCCL') != CENSUS_NATIVE_ZERO_TOTAL:
                raise ValueError('session call records contradict the counts')
        ranks.append({'rank': row['rank'], 'logical_calls': CENSUS_LOGICAL_CALLS,
                      'pace_calls': pace, 'hccl_calls': hccl})
    return {'ranks': ranks, 'source': 'sync-model session snapshot (counts + backend records)'}


def validate_hccl_census(trace):
    """64 distinct collectives per rank under the frozen capture invariant."""
    if not isinstance(trace, dict):
        raise ValueError('census trace is not an object')
    per_rank = trace.get('ranks')
    if not isinstance(per_rank, list) or len(per_rank) != WORLD:
        raise ValueError('census rank coverage differs')
    for row in per_rank:
        if (row.get('alltoallv_class_calls'), row.get('collectives'),
                row.get('dispatch_methods'), row.get('combine_methods')) != \
                (CENSUS_LOGICAL_CALLS, CENSUS_LOGICAL_CALLS,
                 CENSUS_DISPATCH_METHODS, CENSUS_COMBINE_METHODS):
            raise ValueError('HCCL census counts differ from the frozen gate invariant')
        if row.get('method_backends') != [CENSUS_METHOD_BACKEND] \
                or row.get('selector_methods') != [CENSUS_SELECTOR_METHOD] * 2 \
                or row.get('forward_kinds') != [CENSUS_FORWARD_KIND] * 2:
            raise ValueError('HCCL census backend/selector/forward labels differ')
    if trace.get('global_distinct_collectives') != CENSUS_LOGICAL_CALLS:
        raise ValueError('global distinct-collective count differs from 64')
    return True


def validate_pace_census(trace):
    """62 forced-SDMA PACE + 2 native HCCL zero-total calls per rank."""
    if not isinstance(trace, dict):
        raise ValueError('census trace is not an object')
    per_rank = trace.get('ranks')
    if not isinstance(per_rank, list) or len(per_rank) != WORLD:
        raise ValueError('census rank coverage differs')
    for row in per_rank:
        if (row.get('logical_calls'), row.get('pace_calls'), row.get('hccl_calls')) != \
                (CENSUS_LOGICAL_CALLS, CENSUS_PACE, CENSUS_NATIVE_ZERO_TOTAL):
            raise ValueError('PACE census counts differ from the qualified sync-model v2 trace')
    return True


def validate_sq_boundaries(boundaries):
    """F-arm request-boundary queue occupancy (per rank, per channel): the
    four boundary occupancy values (tail - head) must not be strictly
    increasing anywhere (no monotone accumulation across gate/w1/w2/w3).
    Raw values are retained; input shape is {label: [per-rank channel list]}.
    """
    if not isinstance(boundaries, dict) or sorted(boundaries) != sorted(F_SQ_BOUNDARIES):
        raise ValueError('SQ boundary coverage differs')
    for name in F_SQ_BOUNDARIES:
        ranks = boundaries[name]
        if not isinstance(ranks, list) or len(ranks) != WORLD:
            raise ValueError('SQ boundary rank coverage differs')
        for rank, channels in enumerate(ranks):
            if not isinstance(channels, list) or not channels:
                raise ValueError('SQ boundary channel record differs')
            for row in channels:
                if not isinstance(row, dict) or type(row.get('sq_head')) is not int \
                        or type(row.get('sq_tail')) is not int \
                        or row['sq_tail'] < row['sq_head']:
                    raise ValueError('SQ boundary head/tail differs')
    channel_count = len(boundaries[F_SQ_BOUNDARIES[0]][0])
    if any(len(channels) != channel_count for ranks in boundaries.values()
           for channels in ranks):
        raise ValueError('SQ boundary channel coverage differs')
    for rank in range(WORLD):
        for index in range(channel_count):
            series = [boundaries[name][rank][index]['sq_tail']
                      - boundaries[name][rank][index]['sq_head']
                      for name in F_SQ_BOUNDARIES]
            if all(b > a for a, b in zip(series, series[1:])):
                raise ValueError(f'monotone SQ occupancy accumulation across request '
                                 f'boundaries (rank {rank}, channel {index})')
    return True


# ------------------------------------------------------------ metrics ----
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
    values = list(values)
    if not values:
        raise ValueError('mad of nothing')
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def geomean(values):
    values = list(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError('geomean requires positive values')
    return statistics.geometric_mean(values)


CONTRASTS = {'C1': ('HmR', 'H'), 'C2': ('F', 'H'), 'C3': ('F', 'HOPT'),
             'C4': ('HOPT', 'HmR')}


def tie_threshold(treatment_medians, comparator_medians):
    """PLAN section 9: tau = max(1% x median of all 2k group medians,
    2 x MAD(treatment), 2 x MAD(comparator)) — symmetric conservative union."""
    all_medians = list(treatment_medians) + list(comparator_medians)
    return max(0.01 * statistics.median(all_medians),
               2 * mad(list(treatment_medians)), 2 * mad(list(comparator_medians)))


def direction_rule(treatment_medians, comparator_medians, level='L0'):
    """Unpaired k x k cross-arm group-median differences; L2 descriptive only."""
    k = len(treatment_medians)
    if k != len(comparator_medians) or k < 2:
        raise ValueError('direction rule needs balanced arms with >=2 groups')
    if level == 'L2':
        return {'level': 'L2', 'classification': 'descriptive_only',
                'note': 'no stable classification below 3 groups per arm'}
    expected = {'L0': 16, 'L1': 9}[level]
    deltas = [t - c for t in treatment_medians for c in comparator_medians]
    if len(deltas) != expected:
        raise ValueError('cross-arm difference count differs from the level')
    same_sign = sum(1 for d in deltas if d < 0)  # treatment faster
    opposite = sum(1 for d in deltas if d > 0)
    tau = tie_threshold(treatment_medians, comparator_medians)
    arm_delta = statistics.median(treatment_medians) - statistics.median(comparator_medians)
    if (same_sign == expected or opposite == expected) and abs(arm_delta) > tau:
        classification = 'stable-faster' if same_sign == expected else 'stable-slower'
    elif max(same_sign, opposite) >= expected - 2 and abs(arm_delta) > tau:
        classification = 'stable-with-flips'
    else:
        classification = 'unstable'
    return {'level': level, 'classification': classification,
            'treatment_faster_pairs': same_sign, 'comparator_faster_pairs': opposite,
            'tie_pairs': expected - same_sign - opposite,
            'arm_median_delta_ms': arm_delta, 'tie_threshold_ms': tau,
            'geomean_ratio': geomean(treatment_medians) / geomean(comparator_medians),
            'flip_counts_reported': True}


# Historical anchors: comparable-in-structure, NEVER pooled (descriptive
# drift context only; arms.json prior_data_anchors).
HISTORICAL_SIX_GROUP = {'native_medians_ms': [214.0339, 217.32487, 211.987395],
                        'no_resize_medians_ms': [213.24698, 220.011715, 202.499725],
                        'known_flaws': 'unbalanced order (first-position 2:1); two '
                                       'first-request memory jobs between the pair2 arms; '
                                       'mixed directions; no causal claim'}
HISTORICAL_ENDPOINT = {'hccl_ms': 199.326, 'pace_forced_sdma_c4_ms': 194.527,
                       'speedup': '1.025x [1.010, 1.038]',
                       'estimator': 'pooled medians, 8 fresh jobs per backend, '
                                    'different campaign and estimator; direction-only (H3)'}
HISTORICAL_SCOPE = ('cross-campaign drift context only; different day/fabric/thermal '
                    'state; never pooled, never used for significance')


# ---------------------------------------------------------- validators ---
REQUEST_PHASES = ('gate', 'warmup', 'measurement')


def validate_request_row(row, index):
    """One driver request row: phase by index, driver perf_counter_ns pair,
    canonical strictly increasing request id, 16 reference tokens."""
    if not isinstance(row, dict):
        raise ValueError('request row is not an object')
    phase = 'gate' if index == 0 else ('warmup' if index <= WARMUP_REQUESTS else 'measurement')
    if (row.get('index'), row.get('phase')) != (index, phase):
        raise ValueError('request phase/index differs from the frozen structure')
    start, end = row.get('start_ns'), row.get('end_ns')
    if type(start) is not int or type(end) is not int or end <= start:
        raise ValueError('invalid driver timer pair')
    if row.get('generate_elapsed_ns') != end - start:
        raise ValueError('driver elapsed differs from its own timestamps')
    if row.get('clock') != 'driver perf_counter_ns around llm.generate':
        raise ValueError('timer boundary declaration differs')
    if not isinstance(row.get('request_id'), str) or not row['request_id']:
        raise ValueError('request id missing')
    if row.get('output_tokens_equal_reference') is not True:
        raise ValueError('token equality against the native reference failed')
    if row.get('validated_outside_timer') is not True:
        raise ValueError('validation must sit outside the timer')
    return (start, end, row['request_id'])


def validate_request_records(rows):
    if not isinstance(rows, list) or len(rows) != TOTAL_REQUESTS:
        raise ValueError('group request count differs from the 24-request contract')
    pairs = [validate_request_row(row, index) for index, row in enumerate(rows)]
    starts = [p[0] for p in pairs]
    if any(b < a for a, b in zip(starts, starts[1:])):
        raise ValueError('request starts are not chronological')
    ids = [p[2] for p in pairs]
    if len(set(ids)) != TOTAL_REQUESTS:
        raise ValueError('request ids not distinct')
    measured = [(end - start) / 1e6 for start, end, _ in pairs[GATE_REQUESTS + WARMUP_REQUESTS:]]
    if len(measured) != MEASURED_REQUESTS:
        raise ValueError('measured request count differs')
    return {'measured_ms': measured, 'request_ids': ids}


def group_median_ms(rows):
    parsed = validate_request_records(rows)
    measured = parsed['measured_ms']
    if len(measured) != MEASURED_REQUESTS:
        raise ValueError('measured request count differs')
    return statistics.median(measured)


def validate_group_record(record, group_key, arm, level):
    """Worker group record: plan binding, census by arm, per-arm extras."""
    if group_key not in GROUPS or GROUP_ARM[group_key] != arm:
        raise ValueError('group/arm binding differs from the schedule')
    if group_key not in ladder_groups(level):
        raise ValueError('group dropped by the trimming ladder')
    if (record.get('event'), record.get('group'), record.get('arm'), record.get('world'),
            record.get('ladder_level'), record.get('image_id')) != (
            'group_samples', group_key, arm, WORLD, level, IMAGE):
        raise ValueError('group record identity differs')
    if record.get('clock') != 'driver perf_counter_ns around llm.generate':
        raise ValueError('clock domain differs')
    if arm in HCCL_ARMS:
        validate_hccl_census(record.get('census'))
        if not isinstance(record.get('hopt_env'), dict) \
                or any(key not in ('HCCL_BUFFSIZE',) for key in record['hopt_env']):
            raise ValueError('HOPT env delta outside the pre-registered knob set')
    else:
        validate_pace_census(record.get('census'))
        validate_sq_boundaries(record.get('sq_boundaries'))
        if (record.get('slot_bytes'), record.get('heap_bytes')) != (SLOT_BYTES, HEAP_BYTES):
            raise ValueError('F-arm SHMEM geometry differs from the qualified path')
        if record.get('f_restoration') != {'recorder_removed': True, 'hook_retained': True,
                                           'recorder_digest_unchanged': True}:
            raise ValueError('F-arm restoration proof flags differ')
    validate_request_records(record.get('requests'))
    return record


def validate_completion(receipt, group_key, manifest_sha, review_sha):
    if (receipt.get('schema'), receipt.get('status'), receipt.get('group'),
            receipt.get('world'), receipt.get('manifest_sha256'),
            receipt.get('review_sha256'), receipt.get('image_id')) != (
            'revision2.optimized-hccl-completion.v1', 'PASS', group_key, WORLD,
            manifest_sha, review_sha, IMAGE):
        raise ValueError('group receipt identity differs')
    arm = receipt.get('arm')
    if GROUP_ARM[group_key] != arm:
        raise ValueError('receipt arm differs from the schedule')
    if receipt.get('ladder_level') not in LADDER:
        raise ValueError('receipt ladder level differs')
    if type(receipt.get('duration_seconds')) not in (int, float) or receipt['duration_seconds'] <= 0:
        raise ValueError('receipt lacks an observed duration')
    if type(receipt.get('group_median_ms')) not in (int, float) or receipt['group_median_ms'] <= 0:
        raise ValueError('receipt lacks the group median')
    return receipt


# ------------------------------------------------- exact nanoseconds -----
ISO = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)$')


def exact_ns(value):
    """Explicit-offset RFC3339 into exact integer nanoseconds."""
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
    for value in (started_at, finished_at):
        match = ISO.fullmatch(value) if isinstance(value, str) else None
        if match is None or match.group(3) not in ('Z', '+00:00'):
            raise ValueError('Docker timestamps must use explicit UTC')
    ordered = [exact_ns(value) for value in (first_utc, second_utc, started_at, finished_at)]
    if not ordered[0] <= ordered[1] <= ordered[2] < ordered[3]:
        raise ValueError('preflights and container timestamps are not in order')


# ------------------------------------------------------------ rollup -----
def select_level_receipts(receipts_by_group, level, degraded_three_arm=False):
    """Per-level aggregation selection (sparse-lane per-subgroup precedent).

    For the requested ladder level, keep exactly the receipts of the level's
    group set (ladder_groups(level), minus F in a degraded campaign) and
    require complete coverage of that set. Because the ladder may arm
    mid-campaign, the level's groups may have run under an earlier (smaller
    index) level label; such receipts are admissible, their recorded labels
    are preserved in the rollup, and a receipt recorded at a LATER level
    than requested is refused (the ladder only moves down).
    """
    order = ('L0', 'L1', 'L2')
    if level not in order:
        raise ValueError('unknown ladder level')
    chosen = {}
    for group_key, receipt in receipts_by_group.items():
        if group_key not in GROUPS:
            raise ValueError(f'receipt group outside the schedule: {group_key}')
        if degraded_three_arm and GROUP_ARM[group_key] == 'F':
            raise ValueError('F receipts exist; degraded aggregation is not applicable')
        if group_key in ladder_groups(level):
            chosen[group_key] = receipt
    expected = [key for key in ladder_groups(level)
                if not degraded_three_arm or GROUP_ARM[key] != 'F']
    if sorted(chosen) != sorted(expected):
        raise ValueError(f'level {level} receipt coverage differs from the ladder group set')
    for group_key, receipt in chosen.items():
        recorded = receipt.get('ladder_level')
        if recorded not in order or order.index(recorded) > order.index(level):
            raise ValueError(f'receipt recorded at a later level than requested: {group_key}')
    return chosen


def campaign_rollup(receipts_by_group, level, degraded_three_arm=False):
    """Contrast table over the groups of one ladder level; history drift
    context attached but never pooled."""
    groups = [key for key in ladder_groups(level)
              if not degraded_three_arm or GROUP_ARM[key] != 'F']
    if sorted(receipts_by_group) != sorted(groups):
        raise ValueError('campaign receipt coverage differs')
    medians = {}
    for key in groups:
        arm = receipts_by_group[key]['arm']
        if GROUP_ARM[key] != arm:
            raise ValueError(f'receipt arm differs from the schedule: {key}')
        medians.setdefault(arm, []).append(receipts_by_group[key]['group_median_ms'])
    expected = {'L0': 4, 'L1': 3, 'L2': 2}[level]
    for arm, values in medians.items():
        if len(values) != expected:
            raise ValueError(f'arm {arm} has {len(values)} groups, expected {expected}')
    arms_summary = {arm: {'group_medians_ms': sorted(values), **summary(values)}
                    for arm, values in sorted(medians.items())}
    contrasts = {}
    for name, (treatment, comparator) in CONTRASTS.items():
        if treatment in medians and comparator in medians:
            contrasts[name] = {'pair': f'{treatment} vs {comparator}',
                               **direction_rule(medians[treatment], medians[comparator], level)}
    return {'schema': 'revision2.optimized-hccl-aggregate.v1', 'level': level,
            'reduced_replicate': level != 'L0',
            'degraded_three_arm': degraded_three_arm,
            'receipt_levels_by_group': {key: receipt.get('ladder_level')
                                        for key, receipt in sorted(receipts_by_group.items())},
            'arms': arms_summary, 'contrasts': contrasts,
            'historical_drift_context_only': {
                'six_group_20261028': HISTORICAL_SIX_GROUP,
                'endpoint_table': HISTORICAL_ENDPOINT,
                'scope': HISTORICAL_SCOPE,
                'policy': 'never pooled with this campaign; descriptive drift only'},
            'sample_unit': 'fresh process group; 20 in-group requests are dependent, not 20 replications',
            'statistics_scope': 'median/IQR/min/max of group medians; geomean ratio; '
                                'pre-registered direction rule; no p-values, no CI claims',
            'unstable_is_valid': True}
