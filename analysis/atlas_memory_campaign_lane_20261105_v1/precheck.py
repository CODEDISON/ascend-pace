"""No-device import/ABI/plan evidence; creates no stream and calls no device API.

Proves inside the pinned image: provider digests plus the RTLD_NOLOAD/dladdr
identity of the already-loaded SHMEM provider (fname-only path semantics),
the frozen PACE operator schema (namespaced form), the L2 memory_snapshot
module digest and its pinned API-source hashes, the full arms.json binding
with every recomputable arithmetic fact, the complete campaign grid (job
ids/order/budgets/ports/conditionals/trim ladder), the capacity ladder and
exhaustion constants, and the shared review-contract constant the device
gate consumes.
"""
import ctypes
import hashlib
import importlib.util
import json
import os

import torch
import torch_npu
import cashflex  # noqa: F401

import plan

SHMEM_PATH = '/workspace/shmem/build_cann9_vllm/lib/libshmem.so'
HELPER_PATH = '/workspace/queue_metadata/metadata.py'
LAYOUT_PATH = '/workspace/sdma_large/layout.json'
SNAPSHOT_PATH = '/workspace/memory_snapshot/memory_snapshot.py'

with open(SHMEM_PATH, 'rb') as stream:
    shmem_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
if shmem_hash != plan.SHMEM_SHA256:
    raise RuntimeError('provider mismatch')

provider = ctypes.CDLL(SHMEM_PATH, mode=os.RTLD_NOLOAD | os.RTLD_NOW)


class DlInfo(ctypes.Structure):
    _fields_ = [('fname', ctypes.c_char_p), ('fbase', ctypes.c_void_p),
                ('sname', ctypes.c_char_p), ('saddr', ctypes.c_void_p)]


dladdr = ctypes.CDLL(None).dladdr
dladdr.argtypes, dladdr.restype = [ctypes.c_void_p, ctypes.POINTER(DlInfo)], ctypes.c_int
info = DlInfo()
symbol = provider.g_state  # host state object export; its address is only resolved
if dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)) != 1:
    raise RuntimeError('dladdr did not locate the export')
# The containing FILE name is the provider path; the symbol name is carried
# as a diagnostic only and never participates in the path comparison.
if plan.dladdr_provider_path(info.fname, info.sname) != os.path.realpath(SHMEM_PATH):
    raise RuntimeError('SHMEM export resolves to unexpected provider')

if hashlib.sha256(open(HELPER_PATH, 'rb').read()).hexdigest() != plan.HELPER_SHA256:
    raise RuntimeError('queue-metadata helper digest differs')
spec = importlib.util.spec_from_file_location('memcamp_precheck_helper', HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
layout = helper.load_layout(LAYOUT_PATH, plan.LAYOUT_SHA256)
helper.require_export(SHMEM_PATH, 'g_state', 'OBJECT', layout['state']['size'])
acl_path = plan.discover_acl_library(open('/proc/self/maps').read())
with open(acl_path, 'rb') as stream:
    acl_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
helper.require_export(acl_path, 'aclrtMemcpy', 'FUNC')

# L2 lens module identity: the frozen memory_snapshot capture implementation.
if hashlib.sha256(open(SNAPSHOT_PATH, 'rb').read()).hexdigest() != plan.MEMORY_SNAPSHOT_SHA256:
    raise RuntimeError('L2 memory_snapshot module digest differs')
snapshot_spec = importlib.util.spec_from_file_location('memcamp_precheck_snapshot', SNAPSHOT_PATH)
snapshot = importlib.util.module_from_spec(snapshot_spec)
snapshot_spec.loader.exec_module(snapshot)
if snapshot.API_SOURCE_SHA256 != plan.MEMORY_API_SOURCE_SHA256:
    raise RuntimeError('L2 memory API source pins differ')

# Pin the frozen PACE device-split ABI (namespaced device truth).
pace_schema = str(torch.ops.cashflex.hpa_alltoallv_device_splits.default._schema)
plan.check_pace_schema(pace_schema)

# Full campaign identity: arms binding + every job's spec, budget, ports,
# conditional gate and the complete trim ladder.
arms = plan.load_arms()
specs = {job: plan.job_spec(job, 0) for job in plan.CAMPAIGN_ORDER}
budget_total = sum(plan.JOB_BUDGET_MINUTES[job] for job in plan.CAMPAIGN_ORDER)
for trim_steps in range(plan.MAX_TRIM_STEPS + 1):
    plan.effective_campaign(trim_steps)
for L in plan.LADDER_LENGTHS:
    plan.ladder_concurrency(L)
ladders = {str(L): plan.ladder_concurrency(L) for L in plan.LADDER_LENGTHS}
kv_heap_olmoe = {job: plan.kv_heap_for(job) for job in
                 ('A1', 'A2', 'C1a', 'C1b', 'C2BA1', 'C2BB1', 'C2BC1', 'C4BA')}
a6_examples = {str(free): plan.a6_kv_floor(free) for free in (40.0, 36.0, 19.0)}

print(json.dumps({'event': 'NO_DEVICE_PRECHECK_PASS', 'provider_path': SHMEM_PATH,
                  'provider_sha256': shmem_hash,
                  'dladdr_symbol': plan.dladdr_symbol(info.sname),
                  'pace_schema': pace_schema,
                  'memory_snapshot_sha256': plan.MEMORY_SNAPSHOT_SHA256,
                  'memory_api_source_sha256': plan.MEMORY_API_SOURCE_SHA256,
                  'arms_sha256': plan.ARMS_SHA256,
                  'arms_crosschecks': 'binding + arithmetic recomputation pass',
                  'jobs': len(plan.CAMPAIGN_ORDER), 'budget_minutes_total': budget_total,
                  'job_specs': {job: {'role': row['role'], 'model': row['model'],
                                      'arm': row['arm'], 'budget_minutes': row['budget_minutes'],
                                      'conditional': row['conditional'],
                                      'quarantined': row['quarantined']}
                                for job, row in specs.items()},
                  'ports': {job: list(plan.ports(job)) for job in plan.CAMPAIGN_ORDER},
                  'kv_heap_examples': kv_heap_olmoe, 'a6_floor_examples': a6_examples,
                  'capacity_ladders': ladders,
                  'capacity_constants': {'streak': plan.EXHAUSTION_STREAK_ITERATIONS,
                                         'wall_cap_multiple': plan.EXHAUSTION_WALL_CAP_MULTIPLE,
                                         'binary_depth': plan.BINARY_DEPTH_MAX,
                                         'search_probes': plan.SEARCH_PROBES_PER_ARM_L},
                  'throughput_grid': {'C': list(plan.THROUGHPUT_CONCURRENCY),
                                      'L': list(plan.THROUGHPUT_LENGTHS),
                                      'O': list(plan.THROUGHPUT_OUTPUT_TOKENS)},
                  'tier1': {'epsilon_bytes': plan.EPSILON_LEG2,
                            'fill_floor_bytes': plan.FILL_FLOOR_F,
                            'adjustment_rounds': plan.PEAK_ADJUSTMENT_ROUNDS},
                  'l3_rule': {'max_request_delta': plan.L3_ADOPTION_MAX_REQUEST_DELTA,
                              'max_fencepost_delta_bytes': plan.L3_ADOPTION_MAX_FENCEPOST_DELTA},
                  'trim_ladder': [list(step) for step in plan.TRIM_LADDER],
                  'review_contract': {'schema': plan.REVIEW_SCHEMA, 'scope': plan.REVIEW_SCOPE,
                                      'groups': plan.REVIEW_GROUPS,
                                      'reviewed_files': sorted(plan.REVIEWED_FILES)},
                  'metadata_helper_sha256': plan.HELPER_SHA256,
                  'layout_sha256': plan.LAYOUT_SHA256,
                  'acl_library_path': acl_path, 'acl_library_sha256': acl_hash,
                  'torch_version': torch.__version__,
                  'torch_npu_version': getattr(torch_npu, '__version__', 'unknown'),
                  'scope': 'imports, no-load provider identity, PACE schema, L2 module '
                           'identity, arms/campaign binding and review contract only; no '
                           'device initialized, no engine constructed'}), flush=True)
