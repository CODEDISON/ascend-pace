"""Pure CPU plans, arms binding, campaign grid, lens mathematics, validators
and wording rules for the joint #11/#12 memory campaign lane.

No framework/device/network imports; stdlib only. Every function here runs on
an ordinary laptop and is shared by the device worker, the no-device
precheck, the launcher audit, the aggregate and the local test suite, so the
measured structure cannot drift between those roles.

arms.json is NOT copied into this lane: the machine-readable companion of the
reviewed plan directory (atlas_memory_campaign_plan_20261105_v1/arms.json) is
bound by SHA-256 and every arithmetic fact that can be recomputed is
recomputed here; the recomputation is authoritative over declared prose.

Transcription pins were extracted PROGRAMMATICALLY from the barrier-v3 and
sparse-distribution (#8) lane sources and cross-verified against each other
before freezing (three prior transcription defects class: LIB_SHA256 hex
typo, IMAGE truncation, dladdr sname/fname swap - all prevented here by
construction and by FrozenPinTests).
"""
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

WORLD = 16
DIES = tuple(range(WORLD))
TOTAL_HBM_GIB = 64  # per die, Atlas 800T A3
MIB = 1024 ** 2
GIB = 1024 ** 3
BLOCK_TOKENS = 128
SEED = 2027

# ----------------------------------------------------- transcription pins ---
# Extracted programmatically (see module docstring); FrozenPinTests freezes
# each value verbatim, including the IMAGE 7+64 form.
IMAGE = 'sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f'
SHMEM_SHA256 = 'c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71'
LIB_SHA256 = '24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9'
HELPER_SHA256 = '0d7fde8e4eb1f0c8c312412cfc2bb6ee6f5df116ea48e2e27aae2feadef962be'
LAYOUT_SHA256 = 'e844a5175ca3de3e4a54ef54723b76df6665607c4268ad29cc3ed541d7bf6154'
GUARD_LAUNCH_SHA = 'b4d96c96b5e304d8b0cf74c264d7b6006c158b018c7243bf055a74b79d91ffc2'
GUARD_MANIFEST_SHA = '288fb6913cab6181af5021505be0534b3105704c75358cb3252bf1cd31246d94'
# L2 lens capture module (frozen across every memory lane; identical bytes in
# atlas_sdma_large_correctness_v1 and atlas_model_memory_pilot_v1).
MEMORY_SNAPSHOT_SHA256 = '3f02de28ce69e8be24fc6d0b3ab6f1ad728e859c03d1cdb9b4e88eaa853c3402'
MEMORY_API_SOURCE_SHA256 = {
    'mem_get_info': 'dc6b6763a6127c0a057a5e5f4f962bc94182e8a0697b446d5a8faeab35c9b058',
    'memory_stats': '9689894e9cb47ba6884f191e1b45326576441233475142108b05671866ed4947',
}

# Binding to the reviewed plan directory (referenced, never copied).
ARMS_SHA256 = 'e7fc2082abd82558cd97d46faf0e817bfe5f04dac7688d58d8570b3bceb5ae32'
PLAN_MD_SHA256 = 'c8f3371312ad94195cb121d68135972b8ae35070d9a3cede3b33bd56dcc0ff99'
LANE_DIR = Path(__file__).resolve().parent
LOCAL_PLAN_DIR = LANE_DIR.parent / 'atlas_memory_campaign_plan_20261105_v1'
DEVICE_PLAN_DIR = Path('/workspace/plan')

# ------------------------------------------------------------ invariants ---
# Three-lens instrumentation (PLAN section 1.3). Peak language is ALWAYS
# "sampled maximum (lower bound)"; lenses overlap and are never summed.
LENSES = ('L1', 'L2', 'L3')
L1_INTERVAL_S = 1.0
L3_RATES_HZ = (0.5, 2.0)
L3_ADOPTION_MAX_REQUEST_DELTA = 0.02  # median request-time delta <= 2%
L3_ADOPTION_MAX_FENCEPOST_DELTA = 64 * MIB
RECONCILIATION_WINDOW_S = 2.0
FENCEPOSTS = ('T0', 'T1', 'T2a', 'T2b', 'T2', 'T3', 'T4', 'T5')
FENCEPOST_LABELS = {
    'T0': 'container/driver start', 'T1': 'model loaded',
    'T2a': 'before SHMEM init (PACE)', 'T2b': 'after SHMEM init (PACE)',
    'T2': 'groups/backend initialized (native)', 'T3': 'first request complete',
    'T4': 'steady phase complete (WP-A)', 'T5': 'teardown complete',
}
WP_A_STEADY_REQUESTS = 8  # after the frozen first request; no peak resets

# Tier-1 matched-budget legs (PLAN section 3.1).
EPSILON_LEG2 = max(1 * GIB, 0.02 * TOTAL_HBM_GIB * GIB)  # 1.28 GiB (2% of 64 GiB)
FILL_FLOOR_F = 2 * GIB
FILL_STEP = 256 * MIB
PEAK_ADJUSTMENT_ROUNDS = 1

# Capacity frontier (PLAN section 3.3), preregistered.
EXHAUSTION_STREAK_ITERATIONS = 100
EXHAUSTION_WALL_CAP_MULTIPLE = 3
LADDER_LENGTHS = (1024, 4096)
LADDER_CONCURRENCY_FROM = 16
LADDER_CONCURRENCY_CAP = {1024: 2048, 4096: 512}
BINARY_DEPTH_MAX = 3
SEARCH_PROBES_PER_ARM_L = 4  # doubling sweep probes before refinement
CAPACITY_TOKENS = 1  # max_tokens: prefill-admission pressure

# Throughput frontier grid (PLAN section 3.4).
THROUGHPUT_CONCURRENCY = (16, 64, 256)
THROUGHPUT_LENGTHS = (1024, 4096)
THROUGHPUT_OUTPUT_TOKENS = (1, 32)

# ------------------------------------------------------------ campaign ------
# Job grid: id -> (workpackage, role, model, arm, conditional). A job is one
# fresh container/process group/16-die group with one preregistered config.
# Campaign order is the strict execution/gating order.
CAMPAIGN_ORDER = (
    'A0a', 'A0p',                                   # pilot, quarantined
    'C01', 'C02',                                   # equal-cap probes
    'A1', 'A2', 'A1r', 'A2r', 'A3', 'A4', 'A5', 'A6',   # WP-A formal
    'C1a', 'C1b',                                   # fill-probe pair
    'C2BA1', 'C2BA2', 'C2BA3', 'C2BA4', 'C2BA5',
    'C2BB1', 'C2BB2', 'C2BB3', 'C2BB4', 'C2BB5',
    'C2BC1', 'C2BC2', 'C2BC3', 'C2BC4', 'C2BC5',
    'C3BD1', 'C3BD2',                               # conditional on per_peer
    'C4BA', 'C4BB', 'C4BC',                         # throughput sweeps
    'C5BA', 'C5BB', 'C5BC',                         # fresh-group confirms
    'C6QA', 'C6QB', 'C6SA', 'C6SB',                 # Qwen3 WP-B (trim T1)
    'C7GA', 'C7GB', 'C7SA', 'C7SB',                 # GPT-OSS WP-B (T2, cond.)
)
SHMEM_PORT_BASE = 26801
MASTER_PORT_BASE = 26901
PILOT_JOBS = ('A0a', 'A0p')
QUARANTINED = frozenset(PILOT_JOBS)

# Preregistered per-job device-minute budgets (arms.json groups + WP-A grid).
JOB_BUDGET_MINUTES = {
    'A0a': 8, 'A0p': 8, 'C01': 6, 'C02': 6,
    'A1': 8, 'A2': 9, 'A1r': 8, 'A2r': 9, 'A3': 12, 'A4': 13, 'A5': 14, 'A6': 15,
    'C1a': 9, 'C1b': 9,
    'C2BA1': 7, 'C2BA2': 7, 'C2BA3': 7, 'C2BA4': 7, 'C2BA5': 7,
    'C2BB1': 7, 'C2BB2': 7, 'C2BB3': 7, 'C2BB4': 7, 'C2BB5': 7,
    'C2BC1': 7, 'C2BC2': 7, 'C2BC3': 7, 'C2BC4': 7, 'C2BC5': 7,
    'C3BD1': 7, 'C3BD2': 7,
    'C4BA': 14, 'C4BB': 14, 'C4BC': 14,
    'C5BA': 9, 'C5BB': 9, 'C5BC': 9,
    'C6QA': 12, 'C6QB': 12, 'C6SA': 12, 'C6SB': 12,
    'C7GA': 15, 'C7GB': 15, 'C7SA': 15, 'C7SB': 15,
}
DURATION_WARN_FACTOR = 1.25  # completion duration > budget*1.25 -> recorded warning

# Preregistered trim ladder (PLAN section 4): ordered drops, monotone only.
TRIM_LADDER = (
    ('drop', 'C7GA'), ('drop', 'C7GB'), ('drop', 'C7SA'), ('drop', 'C7SB'),  # T2
    ('drop', 'C6QA'), ('drop', 'C6QB'), ('drop', 'C6SA'), ('drop', 'C6SB'),  # T1
    ('drop', 'C2BA5'), ('drop', 'C2BB5'), ('drop', 'C2BC5'),  # L=1024 confirms
    ('drop', 'A1r'), ('drop', 'A2r'),                          # repeats
    ('drop', 'C2BC1'), ('drop', 'C2BC2'), ('drop', 'C2BC3'), ('drop', 'C2BC4'),
    ('drop', 'C4BC'), ('drop', 'C5BC'),                         # arm B-C
)
MAX_TRIM_STEPS = len(TRIM_LADDER)

ROLES = {
    'A0a': ('WP-A', 'wp_a_pilot', 'olmoe', 'A'),
    'A0p': ('WP-A', 'wp_a_pilot', 'olmoe', 'P'),
    'A1': ('WP-A', 'wp_a', 'olmoe', 'A'),
    'A2': ('WP-A', 'wp_a', 'olmoe', 'P'),
    'A1r': ('WP-A', 'wp_a', 'olmoe', 'A'),
    'A2r': ('WP-A', 'wp_a', 'olmoe', 'P'),
    'A3': ('WP-A', 'wp_a', 'qwen3', 'A'),
    'A4': ('WP-A', 'wp_a', 'qwen3', 'P'),
    'A5': ('WP-A', 'wp_a', 'gpt_oss', 'A'),
    'A6': ('WP-A', 'wp_a', 'gpt_oss', 'P'),
    'C01': ('WP-B', 'c0_probe_cgroup', None, None),
    'C02': ('WP-B', 'c0_probe_npu_cap', None, None),
    'C1a': ('WP-B', 'fill_probe', 'olmoe', 'A'),
    'C1b': ('WP-B', 'fill_probe', 'olmoe', 'P'),
    'C4BA': ('WP-B', 'throughput_sweep', 'olmoe', 'B-A'),
    'C4BB': ('WP-B', 'throughput_sweep', 'olmoe', 'B-B'),
    'C4BC': ('WP-B', 'throughput_sweep', 'olmoe', 'B-C'),
    'C5BA': ('WP-B', 'throughput_confirm', 'olmoe', 'B-A'),
    'C5BB': ('WP-B', 'throughput_confirm', 'olmoe', 'B-B'),
    'C5BC': ('WP-B', 'throughput_confirm', 'olmoe', 'B-C'),
    'C6QA': ('WP-B', 'capacity', 'qwen3', 'native'),
    'C6QB': ('WP-B', 'capacity', 'qwen3', 'pace'),
    'C6SA': ('WP-B', 'throughput_sweep', 'qwen3', 'native'),
    'C6SB': ('WP-B', 'throughput_sweep', 'qwen3', 'pace'),
    'C7GA': ('WP-B', 'capacity', 'gpt_oss', 'native'),
    'C7GB': ('WP-B', 'capacity', 'gpt_oss', 'pace'),
    'C7SA': ('WP-B', 'throughput_sweep', 'gpt_oss', 'native'),
    'C7SB': ('WP-B', 'throughput_sweep', 'gpt_oss', 'pace'),
}
for _arm in ('BA', 'BB', 'BC'):
    for _k in range(1, 6):
        _wpb_arm = {'BA': 'B-A', 'BB': 'B-B', 'BC': 'B-C'}[_arm]
        _role = 'capacity_confirm' if _k == 5 else 'capacity_search'
        ROLES[f'C2{_arm}{_k}'] = ('WP-B', _role, 'olmoe', _wpb_arm)
for _k in (1, 2):
    ROLES[f'C3BD{_k}'] = ('WP-B', 'capacity_search', 'olmoe', 'B-D')

# Conditional jobs and their gate facts (checked at start + audit).
CONDITIONAL_JOBS = {
    **{f'C3BD{k}': 'per_peer_guard' for k in (1, 2)},
    'C7GA': 'a6_feasible', 'C7GB': 'a6_feasible',
    'C7SA': 'a6_feasible', 'C7SB': 'a6_feasible',
}

# GPT-OSS A6 KV floor rule (PLAN section 2.1).
A6_HEAP_GIB = 16.5
A6_GUARD_GIB = 2
A6_MIN_KV_GIB = 2

MODELS = ('olmoe', 'qwen3', 'gpt_oss')
MODEL_PRIORITY = ('olmoe', 'qwen3', 'gpt_oss')
MODEL_NAMES = {'olmoe': 'OLMoE-1B-7B-0125-Instruct',
               'qwen3': 'Qwen3-30B-A3B', 'gpt_oss': 'GPT-OSS-120B'}
MODEL_PARALLEL = {'olmoe': 'TP16/EP16', 'qwen3': 'TP16/EP16', 'gpt_oss': 'DP2xTP8/EP16'}

# Frozen per-model engine template (OLMoE/Qwen3 from the budget-matched and
# first-memory lanes; GPT-OSS per Table endpoint-protocols DP2xTP8). Only the
# preregistered switches vary per arm/cell; everything else is frozen.
BASE_LLM_CONFIG = {
    'olmoe': {'tensor_parallel_size': 16, 'enable_expert_parallel': True,
              'distributed_executor_backend': 'mp', 'dtype': 'bfloat16',
              'trust_remote_code': True, 'enforce_eager': True,
              'disable_custom_all_reduce': True, 'gpu_memory_utilization': 0.50,
              'max_num_batched_tokens': 16384, 'block_size': BLOCK_TOKENS,
              'enable_chunked_prefill': False, 'enable_prefix_caching': False,
              'async_scheduling': False, 'seed': SEED,
              'num_gpu_blocks_override': None},
    'qwen3': None,   # inherited from olmoe template at freeze (same shape)
    'gpt_oss': None, # DP2xTP8: tensor_parallel_size 8, data_parallel_size 2
}
GPT_OSS_LLM_OVERRIDES = {'tensor_parallel_size': 8, 'data_parallel_size': 2}

# KV/heap splits per (workpackage, model, arm). KV in GiB (may be fractional).
WP_A_CONFIG = {
    ('olmoe', 'A'): {'kv_gib': 16.0, 'heap_gib': 0.0},
    ('olmoe', 'P'): {'kv_gib': 12.0, 'heap_gib': 4.0, 'slot_L_mib': 96},
    ('qwen3', 'A'): {'kv_gib': 16.0, 'heap_gib': 0.0},
    ('qwen3', 'P'): {'kv_gib': 7.5, 'heap_gib': 8.5, 'slot_L_mib': 256},
    ('gpt_oss', 'A'): {'kv_gib': 16.0, 'heap_gib': 0.0},
    ('gpt_oss', 'P'): {'kv_gib': None, 'heap_gib': 16.5, 'slot_L_mib': 512,
                       'kv_rule': 'a6_floor'},  # computed from A5 T1 free
}
WP_B_ARMS = {
    'B-A': {'type': 'native', 'kv_gib': 16.0, 'heap_gib': 0.0,
            'blocks_expected': 16384},
    'B-B': {'type': 'pace_table2', 'kv_gib': 12.0, 'heap_gib': 4.0,
            'slot_L_mib': 96, 'blocks_expected': 12288},
    'B-C': {'type': 'pace_right_sized', 'kv_gib': 12.75, 'heap_gib': 3.25,
            'slot_L_mib': 96},
    'B-D': {'type': 'pace_small_slot', 'kv_gib': 15.25, 'heap_gib': 0.75,
            'slot_L_mib': 16, 'conditional': 'per_peer_guard'},
}
# R2-A8: per-model WP-B mappings for the extension arms. The OLMoE B-B split
# (12+4, slot 96) is incoherent for Qwen3 under the reviewed whole-input
# semantics (#9 Qwen whole-input minimum 112.32 MiB > 96 MiB slot), so the
# Qwen3 PACE arms use the coherent A4 split (7.5+8.5, slot 256); GPT-OSS
# PACE arms use the A6 feasible floor (16.5 heap, slot 512).
WP_B_MODEL_OVERRIDES = {
    'qwen3': {'native': {'kv_gib': 16.0, 'heap_gib': 0.0},
              'pace': {'kv_gib': 7.5, 'heap_gib': 8.5, 'slot_L_mib': 256}},
    'gpt_oss': {'native': {'kv_gib': 16.0, 'heap_gib': 0.0},
                'pace': {'kv_gib': None, 'heap_gib': 16.5, 'slot_L_mib': 512,
                         'kv_rule': 'a6_floor'}},
}
# R3-B1: declarative cells for the four extension capacity jobs (role
# `capacity`). The (L, C) pair is a preregistered constant per model — one
# ladder-anchored probe at L=4096/C=16 (the ladder floor) — because an
# extension arm's cell is NOT derivable from the OLMoE frontier
# (cross-model incomparability); free CLI cells stay rejected.
EXTENSION_CAPACITY_CELLS = {
    'qwen3': {'L': 4096, 'C': 16, 'O': 1},
    'gpt_oss': {'L': 4096, 'C': 16, 'O': 1},
}
WP_B_MODEL_ARM_PAIRS = {
    'olmoe': ('B-A', 'B-B'),
    'qwen3': ('native', 'pace'),
    'gpt_oss': ('native', 'pace'),
}

OLMOE_BYTES_PER_BLOCK = MIB  # source-derived: BF16/TP16/16 layers/128 tokens

ENGINE_ENV_BASE = {'REV2_OUTPUT_TOKENS': '1', 'VLLM_USE_V1': '1',
                   'VLLM_ASCEND_ENABLE_FUSED_MC2': '0',
                   'TOKENIZERS_PARALLELISM': 'false'}


def ports(job_id):
    if job_id not in CAMPAIGN_ORDER:
        raise ValueError('job outside the registered campaign')
    slot = CAMPAIGN_ORDER.index(job_id)
    return SHMEM_PORT_BASE + 2 * slot, MASTER_PORT_BASE + 2 * slot


def job_spec(job_id, trim_steps=0):
    """Resolve one job's preregistered specification."""
    if job_id not in ROLES:
        raise ValueError('job outside the registered campaign')
    if type(trim_steps) is not int or not 0 <= trim_steps <= MAX_TRIM_STEPS:
        raise ValueError('trim steps outside the preregistered ladder')
    if job_id in trimmed_jobs(trim_steps):
        raise ValueError(f'job {job_id} is trimmed at level {trim_steps}')
    workpackage, role, model, arm = ROLES[job_id]
    return {'job': job_id, 'workpackage': workpackage, 'role': role,
            'model': model, 'arm': arm, 'budget_minutes': JOB_BUDGET_MINUTES[job_id],
            'conditional': CONDITIONAL_JOBS.get(job_id),
            'quarantined': job_id in QUARANTINED}


def trimmed_jobs(trim_steps):
    ids = {step[1] for step in TRIM_LADDER[:trim_steps]}
    return frozenset(ids)


def effective_campaign(trim_steps):
    return tuple(job for job in CAMPAIGN_ORDER if job not in trimmed_jobs(trim_steps))


def kv_heap_for(job_id, a5_free_t1_gib=None):
    """KV GiB / heap GiB / slot L for a job, resolving the A6 floor rule."""
    workpackage, role, model, arm = ROLES[job_id]
    if workpackage == 'WP-A':
        config = WP_A_CONFIG[(model, arm)]
    elif role in ('capacity', 'capacity_search', 'capacity_confirm',
                  'throughput_sweep', 'throughput_confirm', 'fill_probe'):
        # R2-A8: model-aware mapping — qwen3/gpt_oss extension arms use their
        # coherent splits, not the OLMoE B-B default.
        if arm in ('native', 'pace'):
            override = WP_B_MODEL_OVERRIDES.get(model, {}).get(arm)
            config = override if override is not None else \
                {'native': WP_B_ARMS['B-A'], 'pace': WP_B_ARMS['B-B']}[arm]
        elif arm in WP_B_ARMS:
            config = WP_B_ARMS[arm]
        else:
            return None
    else:
        return None
    if config.get('kv_rule') == 'a6_floor':
        return {**a6_kv_floor(a5_free_t1_gib), 'slot_L_mib': config['slot_L_mib'],
                'heap_gib': config['heap_gib']}
    return {'kv_gib': config['kv_gib'], 'heap_gib': config['heap_gib'],
            'slot_L_mib': config.get('slot_L_mib')}


def a6_kv_floor(a5_free_t1_gib):
    """A6 KV floor from A5's measured post-load free (T1), whole GiB down.

    kv = floor(free_T1 - heap 16.5 - guard 2); < 2 GiB -> infeasible (a
    reported outcome, not a campaign failure).
    """
    if type(a5_free_t1_gib) not in (int, float) or not 0 < a5_free_t1_gib <= TOTAL_HBM_GIB:
        raise ValueError('A5 T1 free measurement required for the A6 floor')
    kv = math.floor(a5_free_t1_gib - A6_HEAP_GIB - A6_GUARD_GIB)
    return {'kv_gib': float(kv), 'feasible': kv >= A6_MIN_KV_GIB,
            'infeasibility': None if kv >= A6_MIN_KV_GIB else
            'infeasible at Table 2 heap on 64 GB/die (staging leaves < 2 GiB KV)'}


def blocks_for(kv_gib, bytes_per_block=OLMOE_BYTES_PER_BLOCK):
    """Engine block-count arithmetic: floor(KV bytes / per-block bytes)."""
    blocks = int(kv_gib * GIB) // bytes_per_block
    return blocks


def llm_config(model, kv_gib, max_model_len, max_num_seqs, max_tokens):
    """The frozen engine template with only preregistered switches varying."""
    if model not in MODELS:
        raise ValueError('unknown model')
    config = dict(BASE_LLM_CONFIG['olmoe'])
    if model == 'gpt_oss':
        config.update(GPT_OSS_LLM_OVERRIDES)
    if kv_gib is None or type(kv_gib) not in (int, float) or kv_gib <= 0:
        raise ValueError('explicit positive kv_gib required (infeasible arms never '
                         'construct an engine)')
    config['kv_cache_memory_bytes'] = int(kv_gib * GIB)
    if max_model_len not in LADDER_LENGTHS and max_model_len != 2048:
        raise ValueError('max_model_len outside the preregistered set')
    config['max_model_len'] = max_model_len
    if type(max_num_seqs) is not int or not 16 <= max_num_seqs <= 2048:
        raise ValueError('max_num_seqs outside the preregistered ladder')
    config['max_num_seqs'] = max_num_seqs
    if max_tokens not in (CAPACITY_TOKENS,) + THROUGHPUT_OUTPUT_TOKENS:
        raise ValueError('max_tokens outside the preregistered grid')
    return config


# ----------------------------------------------------------- arms binding ---
_ARMS_CACHE = {}


def _arms_path(path=None):
    if path is not None:
        return Path(path)
    override = os.environ.get('MEMCAMP_ARMS_PATH')
    candidates = ([Path(override)] if override else []) + \
                 [LOCAL_PLAN_DIR / 'arms.json', DEVICE_PLAN_DIR / 'arms.json']
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    raise ValueError('arms.json not found (set MEMCAMP_ARMS_PATH)')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != \
       (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('arms file changed while hashing')
    return digest.hexdigest()


def load_arms(path=None):
    """Bind arms.json by SHA and recompute every checkable arithmetic fact."""
    arms_path = _arms_path(path)
    cache_key = str(arms_path) if path is None else None
    if cache_key is not None and cache_key in _ARMS_CACHE:
        return _ARMS_CACHE[cache_key]
    if file_sha256(arms_path) != ARMS_SHA256:
        raise ValueError('arms.json identity differs from the frozen plan')
    document = json.loads(arms_path.read_text())
    if document.get('campaign') != 'atlas_memory_campaign_plan_20261105_v1' \
            or document.get('status') != 'planned_only_no_device_execution':
        raise ValueError('arms campaign identity differs')
    instrument = document['invariants']['instrumentation']
    if (instrument['L3_background_sampler'].find('0.5-2 Hz') < 0
            or instrument['L1_host_poller'].find('1 Hz') < 0):
        raise ValueError('instrumentation lens constants differ')
    fencepost_entries = ' | '.join(document['invariants']['fenceposts'])
    for fencepost in FENCEPOSTS:
        if fencepost_entries.find(fencepost + ' ') < 0 and fencepost not in fencepost_entries:
            raise ValueError('fencepost coverage differs: ' + fencepost)
    models = document['models']
    if sorted(models) != sorted(MODELS) or models['olmoe']['kv_bytes_per_block_per_rank'] != MIB:
        raise ValueError('model block arithmetic basis differs')
    if models['olmoe']['native_blocks_at_16gib'] != blocks_for(16.0):
        raise ValueError('OLMoE 16 GiB block count differs from arithmetic')
    wpb = document['workpackages']['WP-B']
    arms_by_id = {row['id']: row for row in wpb['arms']}
    if sorted(arms_by_id) != ['B-A', 'B-B', 'B-C', 'B-D']:
        raise ValueError('WP-B arm coverage differs')
    for arm_id, contract in WP_B_ARMS.items():
        row = arms_by_id[arm_id]
        if row['kv_gib'] != contract['kv_gib'] or row['heap_gib'] != contract['heap_gib']:
            raise ValueError(f'{arm_id} KV/heap split differs from the lane contract')
        if row.get('slot_L_mib') != contract.get('slot_L_mib'):
            raise ValueError(f'{arm_id} slot L differs')
    # Config-alignment leg: native KV == PACE KV + heap (whole-nominal sum).
    for arm_id in ('B-A', 'B-B', 'B-C', 'B-D'):
        row = arms_by_id[arm_id]
        if abs(row['kv_gib'] + row['heap_gib'] - 16.0) > 1e-9:
            raise ValueError(f'{arm_id} violates the config-alignment leg (KV+heap != 16)')
    if arms_by_id['B-B'].get('blocks_expected') != blocks_for(12.0):
        raise ValueError('B-B block count differs from arithmetic')
    ladder = wpb['capacity_frontier']['ladder']
    if ladder['max_model_len'] != list(LADDER_LENGTHS) \
            or ladder['concurrency_from'] != LADDER_CONCURRENCY_FROM \
            or {int(k): v for k, v in ladder['concurrency_cap'].items()} != LADDER_CONCURRENCY_CAP \
            or ladder['max_tokens'] != CAPACITY_TOKENS:
        raise ValueError('capacity ladder constants differ')
    grid = wpb['throughput_frontier']['cells']
    if grid['concurrency'] != list(THROUGHPUT_CONCURRENCY) \
            or grid['max_model_len'] != list(THROUGHPUT_LENGTHS) \
            or grid['output_tokens'] != list(THROUGHPUT_OUTPUT_TOKENS):
        raise ValueError('throughput grid differs')
    caps = wpb['capacity_frontier']
    if caps['exhaustion_definition'].find('>=100 consecutive') < 0 \
            or caps['exhaustion_definition'].find('3x median') < 0:
        raise ValueError('exhaustion definition differs')
    if document['model_priority'] != list(MODEL_PRIORITY):
        raise ValueError('model priority differs')
    trim = document['trim_ladder_preregistered']
    if len(trim) != 5 or trim[0].find('C7') < 0 or trim[1].find('C6') < 0 \
            or trim[2].find('L=1024') < 0 or trim[3].find('A1r') < 0 \
            or trim[4].find('B-C') < 0:
        raise ValueError('preregistered trim ladder differs')
    budget = document['budget_summary_minutes']
    if budget['core'] != 348 or budget['full'] != 470:
        raise ValueError('budget summary differs from the frozen plan')
    wpa_budget = sum(JOB_BUDGET_MINUTES[job] for job in
                     ('A1', 'A2', 'A1r', 'A2r', 'A3', 'A4', 'A5', 'A6'))
    # PLAN 2.4 states "~106" for pilot(16) + formal(88) = 104 with rounding;
    # tolerate the plan's own declared slack, nothing more.
    if not 0 <= (document['workpackages']['WP-A']['budget_minutes']
                 - (wpa_budget + 2 * JOB_BUDGET_MINUTES['A0a'])) <= 4:
        raise ValueError('WP-A job budgets disagree with the plan total')
    if document['workpackages']['WP-A']['groups'][0]['id'] != 'A0' \
            or document['workpackages']['WP-A']['groups'][0]['role'] != 'pilot_excluded_from_formal':
        raise ValueError('A0 pilot must stay quarantined')
    result = {'document': document, 'arms': arms_by_id, 'models': models}
    if cache_key is not None:
        _ARMS_CACHE[cache_key] = result
    return result


# ------------------------------------------------------ capacity frontier ---
def ladder_concurrency(L):
    """Doubling sweep from 16 up to the storage-bounded cap."""
    if L not in LADDER_CONCURRENCY_CAP:
        raise ValueError('L outside the preregistered ladder')
    cap = LADDER_CONCURRENCY_CAP[L]
    values = []
    value = LADDER_CONCURRENCY_FROM
    while value <= cap:
        values.append(value)
        value *= 2
    values.append(cap)  # the cap itself when the doubling skips it
    return sorted(set(values))


def storage_blocks(C, L, block_tokens=BLOCK_TOKENS):
    """blocks = C * ceil(L / block_tokens) + 1 (PLAN section 3.3)."""
    return C * math.ceil(L / block_tokens) + 1


def next_capacity_cell(history, L):
    """Next probe C from recorded outcomes: doubling until the FIRST failure,
    then binary refinement (midpoint, depth <= 3) between the best served C
    and the first failure. None when the search is complete. Refinement
    probes are arbitrary integers (the doubling ladder is powers of two)."""
    if L not in LADDER_CONCURRENCY_CAP:
        raise ValueError('L outside the preregistered ladder')
    ladder = ladder_concurrency(L)
    served = {C for C, outcome in history if outcome == 'served'}
    failed = {C for C, outcome in history if outcome != 'served'}
    if not history:
        return ladder[0]
    if not failed:
        for C in ladder:
            if C not in served:
                return C
        return None  # whole ladder served: boundary at/above the cap
    first_fail = min(failed)
    lower = max((C for C in served if C < first_fail), default=None)
    if lower is None:
        return None  # even the first cell failed: boundary below the ladder
    upper = first_fail
    if upper - lower <= 1:
        return None  # boundary pinned to adjacent integers
    if len(history) >= SEARCH_PROBES_PER_ARM_L + BINARY_DEPTH_MAX:
        return None  # preregistered search depth exhausted (4 doubling + 3 binary)
    return (lower + upper) // 2


def classify_probe(record):
    """Operational exhaustion classifier (preregistered, PLAN section 3.3).

    'served' | 'exhausted' | 'fault' | 'censored' — never retried.
    """
    if record.get('fault'):
        return 'fault'
    completed, submitted = record.get('completed'), record.get('submitted')
    if type(completed) is not int or type(submitted) is not int or submitted <= 0:
        raise ValueError('probe completion accounting missing')
    streak = record.get('insufficient_kv_streak', 0)
    median = record.get('median_completion_s')
    wall = record.get('wall_s')
    if completed >= submitted:
        return 'served'
    if streak >= EXHAUSTION_STREAK_ITERATIONS and median and wall \
            and wall > EXHAUSTION_WALL_CAP_MULTIPLE * median:
        return 'exhausted'
    if wall is not None and median is not None and wall > EXHAUSTION_WALL_CAP_MULTIPLE * median:
        return 'censored'  # hit the wall cap without the KV streak: censored, not exhausted
    return 'incomplete' if completed == 0 else 'censored'


def c_star(history):
    """Max fully-served C; boundary description retained separately."""
    served = sorted(C for C, outcome in history if outcome == 'served')
    return served[-1] if served else None


# B8: preregistered engine-log patterns for the KV-exhaustion streak. Each
# matching scheduler-log line counts as one scheduler iteration (declared
# proxy); the streak is the max run of consecutive matching lines.
INSUFFICIENT_KV_PATTERNS = ('insufficient kv', 'not enough kv', 'kv cache blocks',
                            'not enough free kv', 'kv cache is full')


def kv_streak_from_log(text):
    """Max consecutive insufficient-KV scheduler-log lines (B8 producer)."""
    if not isinstance(text, str):
        raise ValueError('log text required')
    best = streak = 0
    for line in text.splitlines():
        lowered = line.lower()
        if any(pattern in lowered for pattern in INSUFFICIENT_KV_PATTERNS):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


# B9: frozen per-model reference token digests for the FA4 gate. The OLMoE
# L=1024 anchor is the frozen sync-model prompt batch; per-model token
# references live under /workspace/reference on device. Models without a
# frozen reference use intra-job determinism (two identical temperature-0
# generates must match token-for-token), declared per record.
PROMPT_SHA_ANCHORS = {1024: 'ce5e1795710d16d49d4b65f3bffd4c1ad07556cc27d2d049f0dbd2b55eaebdc8'}


def token_gate_verdict(reference_tokens, observed_tokens):
    """FA4 gate: exact token-id match against the reference (B9)."""
    if observed_tokens is None:
        raise ValueError('observed tokens required for the FA4 gate')
    if reference_tokens is None:
        raise ValueError('reference tokens required for the FA4 gate')
    return {'match': bool(list(reference_tokens) == list(observed_tokens)),
            'reference_len': len(list(reference_tokens)),
            'observed_len': len(list(observed_tokens)),
            'scope': 'reference token match (FA4): mismatch excludes the cell, '
                     'evidence kept'}


# ------------------------------------------------------------ lens rules ---
def l3_adoption(pass_on, pass_off):
    """A0 pilot gate: median request-time delta <= 2% AND fencepost free
    deltas <= 64 MiB at every shared fencepost. Both directions reported."""
    times_on, times_off = pass_on['request_times'], pass_off['request_times']
    if len(times_on) != len(times_off) or not times_on:
        raise ValueError('pilot passes must share the request count')
    deltas = [abs(a - b) / b for a, b in zip(times_on, times_off) if b > 0]
    time_ok = statistics.median(deltas) <= L3_ADOPTION_MAX_REQUEST_DELTA
    fence_on = pass_on['fencepost_free_bytes']
    fence_off = pass_off['fencepost_free_bytes']
    if set(fence_on) != set(fence_off):
        raise ValueError('pilot fencepost coverage differs')
    fence_deltas = {name: abs(fence_on[name] - fence_off[name]) for name in fence_on}
    fence_ok = all(value <= L3_ADOPTION_MAX_FENCEPOST_DELTA for value in fence_deltas.values())
    return {'adopt': bool(time_ok and fence_ok), 'median_request_delta': statistics.median(deltas),
            'max_fencepost_free_delta_bytes': max(fence_deltas.values()),
            'fencepost_free_deltas_bytes': fence_deltas,
            'rule': 'median request-time delta <= 2% and fencepost free delta <= 64 MiB '
                    '(L3 adopted only if both hold; pilot quarantined either way)'}


def reconciliation_bias(l1_samples, l2_stamps, total_bytes=None, window_s=RECONCILIATION_WINDOW_S):
    """Pair L1 (host, per die) with L2 fencepost stamps within +/-2 s.

    Bias is reported per lens pair per die (min/median/max); never averaged
    away and never summed across lenses.
    """
    if total_bytes is None:
        total_bytes = TOTAL_HBM_GIB * GIB
    pairs = []
    used = set()
    for l1 in sorted(l1_samples, key=lambda row: row['mono_ns']):
        for index, l2 in enumerate(l2_stamps):
            if index in used:
                continue
            if abs(l1['mono_ns'] - l2['mono_ns']) <= window_s * 1e9:
                pairs.append((l1, l2))
                used.add(index)
                break
    if not pairs:
        return {'pairs': 0, 'bias_bytes': [], 'scope': 'no shared timestamps within 2 s'}
    biases = []
    for l1, l2 in pairs:
        l2_used = (total_bytes - l2['free_bytes'])
        biases.append({'die': l1['die'], 'l1_used_bytes': l1['used_bytes'],
                       'l2_used_bytes': l2_used,
                       'bias_bytes': l1['used_bytes'] - l2_used,
                       'delta_ns': abs(l1['mono_ns'] - l2['mono_ns'])})
    values = [row['bias_bytes'] for row in biases]
    return {'pairs': len(pairs), 'bias_bytes': biases,
            'bias_summary_bytes': {'min': min(values), 'median': statistics.median(values),
                                   'max': max(values)},
            'scope': 'per-lens-pair bias, reported not averaged; lenses never summed'}


def peak_alignment(native_max_used_per_die, pace_max_used_per_die, adjustment_rounds=0):
    """Tier-1 leg (ii): per-die sampled max used within epsilon, one
    preregistered adjustment round allowed before the wording degrades."""
    if set(native_max_used_per_die) != set(pace_max_used_per_die) or not native_max_used_per_die:
        raise ValueError('per-die coverage differs between arms')
    if adjustment_rounds > PEAK_ADJUSTMENT_ROUNDS:
        raise ValueError('more than the one preregistered adjustment round')
    deltas = {die: abs(native_max_used_per_die[die] - pace_max_used_per_die[die])
              for die in native_max_used_per_die}
    aligned = all(value <= EPSILON_LEG2 for value in deltas.values())
    wording = ('matched configured budget with measured-peak/headroom alignment'
               if aligned else
               'aligned configuration, measured peaks reported separately')
    return {'aligned': aligned, 'epsilon_bytes': EPSILON_LEG2,
            'per_die_delta_bytes': deltas, 'adjustment_rounds': adjustment_rounds,
            'wording': wording}


def fill_probe_result(filler_bytes, steps, free_at_stop):
    """Tier-1 leg (iii): filler to device-free floor F; filler is not KV."""
    reached = free_at_stop <= FILL_FLOOR_F + FILL_STEP
    return {'filler_bytes': filler_bytes, 'steps': steps, 'free_at_stop_bytes': free_at_stop,
            'floor_f_bytes': FILL_FLOOR_F, 'floor_reached_within_one_step': bool(reached),
            'scope': 'equal remaining headroom at F within one step; not a hard ceiling'}


def duration_gate(job_id, duration_seconds):
    """Recorded per-job duration warning (never a silent overrun)."""
    budget_s = JOB_BUDGET_MINUTES[job_id] * 60
    return {'job': job_id, 'duration_seconds': duration_seconds,
            'budget_seconds': budget_s,
            'warning': bool(duration_seconds > budget_s * DURATION_WARN_FACTOR)}


# -------------------------------------------------------------- records ----
L1_SAMPLE_KEYS = ('schema', 'lens', 'die', 'used_bytes', 'free_bytes', 'total_bytes',
                  'mono_ns', 'utc')
L3_SAMPLE_KEYS = ('schema', 'lens', 'rank', 'free_bytes', 'total_bytes', 'mono_ns')


def validate_l1_stream(records):
    """L1 host poller stream: 1 Hz per die, monotone, well-formed."""
    if not isinstance(records, list) or not records:
        raise ValueError('L1 stream empty')
    per_die = {}
    for row in records:
        if any(key not in row for key in L1_SAMPLE_KEYS) or row.get('lens') != 'L1':
            raise ValueError('L1 sample shape differs')
        if row['schema'] != 'revision2.memory-campaign-l1-sample.v1':
            raise ValueError('L1 sample schema differs')
        if row['die'] not in DIES or not 0 <= row['free_bytes'] <= row['total_bytes']:
            raise ValueError('L1 sample die/free differs')
        if not 0 <= row['used_bytes'] <= row['total_bytes']:
            raise ValueError('L1 usage outside total')
        per_die.setdefault(row['die'], []).append(row['mono_ns'])
    for die, stamps in per_die.items():
        if any(later <= earlier for earlier, later in zip(stamps, stamps[1:])):
            raise ValueError(f'L1 stream non-monotone for die {die}')
    return {'dies': sorted(per_die), 'samples': len(records),
            'max_used_bytes': max(row['used_bytes'] for row in records),
            'min_free_bytes': min(row['free_bytes'] for row in records),
            'scope': 'sampled maximum (lower bound); includes every process on the die'}


def l2_fencepost_free(record, fencepost):
    """Extract the free-bytes value of one fencepost capture for pairing."""
    if record.get('lens') != 'L2' or record.get('fencepost') != fencepost:
        raise ValueError('L2 fencepost record differs')
    capture = record.get('capture', {})
    if type(capture.get('free_bytes')) is not int or capture.get('total_bytes') is None:
        raise ValueError('L2 capture missing free/total')
    return capture['free_bytes']


def required_fenceposts(arm, steady):
    """Role-aware required fencepost sequence (R2-B2; PLAN §1.4: T4 is
    steady-phase-complete, WP-A only). Capacity/throughput jobs stop at T3
    and teardown at T5."""
    pace = arm in ('P', 'B-B', 'B-C', 'B-D', 'pace')
    sequence = ['T0', 'T1', 'T2a', 'T2b'] if pace else ['T0', 'T1', 'T2']
    sequence += ['T3']
    if steady:
        sequence += ['T4']
    sequence += ['T5']
    return tuple(sequence)


STEADY_ROLES = ('wp_a', 'wp_a_pilot', 'fill_probe')


def validate_fencepost_sequence(fenceposts, arm, required=None):
    """Validate the (role-aware) fencepost sequence for the arm, in order,
    with L2 captures."""
    if required is None:
        required = required_fenceposts(arm, steady=True)
    if tuple(row.get('fencepost') for row in fenceposts) != tuple(required):
        raise ValueError(f'fencepost sequence differs for arm {arm}: '
                         f'{tuple(row.get("fencepost") for row in fenceposts)}')
    for row in fenceposts:
        capture = row.get('capture', {})
        if row.get('lens') != 'L2' or not isinstance(capture, dict):
            raise ValueError('fencepost capture missing')
        if capture.get('api_source_sha256') != MEMORY_API_SOURCE_SHA256:
            raise ValueError('memory API source identity differs at ' + row.get('fencepost'))
        allocator = capture.get('allocator_bytes', {})
        for kind in ('allocated_bytes', 'reserved_bytes'):
            if allocator.get(f'{kind}.all.peak', -1) < allocator.get(f'{kind}.all.current', 0):
                raise ValueError('allocator peak below current at ' + row.get('fencepost'))
    return [row['fencepost'] for row in fenceposts]


def validate_memory_snapshot_module(digest):
    if digest != MEMORY_SNAPSHOT_SHA256:
        raise ValueError('L2 memory_snapshot module digest differs')


# ---------------------------------------------------------- c0 probes ------
C0_CGROUP_MEMORY_LIMIT = '2g'  # far below any planned NPU allocation


def c0_outcome(probe, over_allocation_failed, error_recorded):
    """Fail-closed probe semantics: a cap counts ONLY if the deliberate
    over-allocation actually failed AND the error is recorded."""
    if probe not in ('C0-1', 'C0-2'):
        raise ValueError('unknown probe')
    enforced = bool(over_allocation_failed and error_recorded)
    return {'probe': probe, 'enforced': enforced,
            'expected': False,
            'note': 'cap counts only if over-allocation fails with recorded error'}


def equal_memory_wording(c0_results):
    """The preregistered wording switch driven by the C0 probes alone."""
    enforced = any(row['enforced'] for row in c0_results)
    if enforced:
        return {'wording': 'enforced equal ceiling',
                'basis': 'a C0 probe demonstrated enforcement (unexpected; FA1 falsified)'}
    return {'wording': 'matched configured budget with measured-peak/headroom alignment',
            'basis': 'no C0 probe demonstrated enforcement; Tier-1 mandatory'}


# ---------------------------------------------------------- review binding --
FILES = {'DESIGN.md', 'plan.py', 'worker.py', 'supervisor.py', 'precheck.py',
         'launch.py', 'test_cpu.py', 'make_manifest.py', 'wp_a_profile.py',
         'wp_b_frontier.py', 'observed_npu_info_fixture.json'}
REVIEW_SCHEMA = 'revision2.memory-campaign-review.v1'
REVIEW_STATUS_APPROVED = 'APPROVED'
REVIEW_SCOPE = 'memory_campaign_wp_a_wp_b_p16'
REVIEW_GROUPS = {'campaign_order': list(CAMPAIGN_ORDER),
                 'conditional': {'per_peer_guard': ['C3BD1', 'C3BD2'],
                                 'a6_feasible': ['C7GA', 'C7GB', 'C7SA', 'C7SB']}}
REVIEWED_FILES = FILES | {'source_manifest.json'}
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
    if review.get('arms_sha256') != ARMS_SHA256:
        raise ValueError('review does not bind the frozen arms.json')
    if review.get('scope') != REVIEW_SCOPE:
        raise ValueError('review scope differs')
    files = review.get('reviewed_files')
    if not isinstance(files, list) or sorted(files) != sorted(REVIEWED_FILES):
        raise ValueError('reviewed file coverage differs from the lane source set')
    if review.get('groups') != REVIEW_GROUPS:
        raise ValueError('reviewed campaign/conditional plan differs')
    if review.get('device_split_guard_semantics') not in REVIEW_GUARD_SEMANTICS:
        raise ValueError('device-split admission semantics not source-verified by review')
    return {'schema': REVIEW_SCHEMA, 'status': REVIEW_STATUS_APPROVED,
            'source_manifest_sha256': manifest_sha, 'arms_sha256': ARMS_SHA256,
            'scope': REVIEW_SCOPE, 'reviewed_files': sorted(REVIEWED_FILES),
            'groups': REVIEW_GROUPS,
            'device_split_guard_semantics': review['device_split_guard_semantics']}


# ------------------------------------------------------ providers/schema ---
PACE_OPERATOR_NAME = 'cashflex::hpa_alltoallv_device_splits'
PACE_ARGUMENT_TYPES = ('Tensor', 'Tensor', 'Tensor', 'Tensor', 'int', 'int', 'int', 'int')
PACE_RETURN_TYPES = ('Tensor',)


def check_pace_schema(text):
    """Pin the frozen PACE device-split operator ABI from its schema string.

    Device-measured truth inside the pinned image:
    'cashflex::hpa_alltoallv_device_splits(Tensor _0, Tensor _1, Tensor _2, '
    'Tensor _3, int _4, int _5, int _6, int _7) -> Tensor _0'.
    The namespaced operator name and every argument/return type word are
    pinned exactly; the compiler-generated suffix names are not pinned, and
    the return segment is validated isomorphically to the argument segment.
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


def dladdr_provider_path(info_fname, info_sname=None):
    """Resolve a dladdr Dl_info file name to the provider object path.

    Only `fname` (the shared object containing the symbol) is a path;
    `sname` is the nearest symbol name and is diagnostic only — it must
    never participate in the path comparison.
    """
    if not info_fname:
        raise ValueError('dladdr returned no containing file name')
    return os.path.realpath(os.fsdecode(info_fname))


def dladdr_symbol(info_sname):
    """Decode the dladdr symbol name for diagnostics; None when absent."""
    return os.fsdecode(info_sname) if info_sname else None


def discover_acl_library(maps_text):
    """Find the single loaded libascendcl provider from /proc/self/maps text."""
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


# ------------------------------------------------------- L1 poller parse ---
CHIP_ROW = re.compile(
    r'^\|\s*(\d+)\s+(\d+)\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+'
    r'(\d+)\s*/\s*(\d+)\s*\|$')


def parse_l1_usage(text):
    """Per-die HBM usage from an `npu-smi info` inventory snapshot.

    Rows: | NPU Chip | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |
    with usage cells 'used / total'. Returns die -> (used_bytes, total_bytes)
    where die = Phy-ID (0..15), MB treated as MiB (npu-smi convention).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('empty npu-smi snapshot')
    rows = {}
    for raw in text.splitlines():
        match = CHIP_ROW.fullmatch(raw.strip())
        if match is None:
            continue
        chip_col, die, _bus, _aicore, mem_used, mem_total, hbm_used, hbm_total = match.groups()
        rows[int(die)] = {'used_bytes': int(hbm_used) * MIB,
                          'total_bytes': int(hbm_total) * MIB,
                          'chip': int(chip_col), 'memory_used_bytes': int(mem_used) * MIB}
    if sorted(rows) != list(DIES):
        raise ValueError(f'npu-smi snapshot does not cover all 16 dies: {sorted(rows)}')
    return rows


def l1_poller_record(rows, mono_ns, utc):
    samples = []
    for die in DIES:
        row = rows[die]
        samples.append({'schema': 'revision2.memory-campaign-l1-sample.v1', 'lens': 'L1',
                        'die': die, 'used_bytes': row['used_bytes'],
                        'free_bytes': row['total_bytes'] - row['used_bytes'],
                        'total_bytes': row['total_bytes'], 'mono_ns': mono_ns, 'utc': utc})
    return samples


# ------------------------------------------------- exact nanosecond time ---
ISO = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)$')


def exact_ns(value):
    """Parse explicit-offset RFC3339 into exact integer nanoseconds."""
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


# ------------------------------------------------------------ aggregates ---
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


WORDING_SHEET = {
    'peak': 'sampled maximum (lower bound); torch allocator excludes direct '
            'SHMEM/ACL/CANN allocations; lenses overlap and are never summed',
    'delta': 'descriptive native-vs-PACE delta; arms differ in diagnostic '
             'instrumentation, no causal transport-only attribution',
    'blocks': 'configured-block difference only; usable capacity is bounded '
              'by the measured exhaustion boundary, never stated as a percent loss',
    'equal_memory': 'mandatory Tier-1 wording unless a C0 probe proves enforcement',
    'gpt_oss_l': 'no per-peer L validation for GPT-OSS (no captured matrices)',
    'scope': 'fixed prompts, one node, O in {1,32}; synchronous diagnostic '
             'PACE path only; no async/multi-node/safety conclusions',
}


def campaign_wording(c0_results):
    return {'equal_memory': equal_memory_wording(c0_results), 'sheet': dict(WORDING_SHEET)}

