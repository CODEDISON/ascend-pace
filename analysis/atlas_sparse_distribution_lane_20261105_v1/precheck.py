"""No-device import/ABI/plan evidence; creates no stream and calls no device API.

Proves inside the reviewed image: provider digests plus the RTLD_NOLOAD/dladdr
identity of the already-loaded SHMEM/ACL providers, the frozen PACE operator
schema, the full 17-case plan identity (guards recomputed from the embedded
matrices against the SHA-bound cases.json), every group's rotated order and
iteration plan for both campaign variants, the SQ planning-estimate cross
check, and the shared review-contract constant the device gate consumes.
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

with open(SHMEM_PATH, 'rb') as stream:
    shmem_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
if shmem_hash != plan.SHMEM_SHA256:
    raise RuntimeError('provider mismatch')

# NOLOAD identity: the provider must already be loaded by cashflex; no
# duplicate SHMEM state is created and no symbol is called.
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
spec = importlib.util.spec_from_file_location('sparse_dist_precheck_helper', HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
layout = helper.load_layout(LAYOUT_PATH, plan.LAYOUT_SHA256)
helper.require_export(SHMEM_PATH, 'g_state', 'OBJECT', layout['state']['size'])
acl_path = plan.discover_acl_library(open('/proc/self/maps').read())
with open(acl_path, 'rb') as stream:
    acl_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
helper.require_export(acl_path, 'aclrtMemcpy', 'FUNC')

# Pin the frozen PACE device-split ABI both arms rely on.
pace_schema = str(torch.ops.cashflex.hpa_alltoallv_device_splits.default._schema)
plan.check_pace_schema(pace_schema)

# Full plan identity: guards recomputed, rotations and iteration plans for
# every group of both campaign variants and every legal trim level.
document = plan.load_cases()
rotation_heads = {}
iteration_counts = {}
for variant in plan.VARIANTS:
    for group_key in plan.GROUPS[variant]:
        for drops in range(plan.MAX_DROPS + 1):
            order = plan.rotated_case_order(variant, group_key, drops)
            if drops == 0:
                rotation_heads[f'{variant}:{group_key}'] = [case_id for _, case_id in order[:3]]
            iteration_counts[f'{variant}:{group_key}:d{drops}'] = \
                len(order) * plan.PACE_CALLS_PER_CASE
        plan.iteration_plan(group_key)
ladder_remain = [len(plan.campaign_case_ids('plan_a', None, drops))
                 for drops in range(plan.MAX_DROPS + 1)]
subgroups = {name: len(plan.campaign_case_ids('plan_b', name, 0)) for name in ('a', 'b')}

print(json.dumps({'event': 'NO_DEVICE_PRECHECK_PASS', 'provider_path': SHMEM_PATH,
                  'provider_sha256': shmem_hash,
                  'dladdr_symbol': plan.dladdr_symbol(info.sname),
                  'pace_schema': pace_schema,
                  'cases_sha256': plan.CASES_SHA256, 'cases_count': len(document['cases']),
                  'workload_sha256': plan.WORKLOADS_SHA256,
                  'guards_recomputed_from_matrices': True,
                  'planning_estimate_crosscheck': document['planning_estimate'],
                  'rotation_heads': rotation_heads,
                  'pace_calls_per_case': plan.PACE_CALLS_PER_CASE,
                  'iteration_call_counts': iteration_counts,
                  'trim_ladder': list(plan.LADDER), 'ladder_case_counts': ladder_remain,
                  'plan_b_subgroup_sizes': subgroups,
                  'groups': {variant: list(plan.GROUPS[variant]) for variant in plan.VARIANTS},
                  'ports': {key: list(plan.ports(key)) for key in plan.CAMPAIGN_SLOTS},
                  'review_contract': {'schema': plan.REVIEW_SCHEMA, 'scope': plan.REVIEW_SCOPE,
                                      'groups': plan.REVIEW_GROUPS,
                                      'reviewed_files': sorted(plan.REVIEWED_FILES)},
                  'metadata_helper_sha256': plan.HELPER_SHA256,
                  'layout_sha256': plan.LAYOUT_SHA256,
                  'acl_library_path': acl_path, 'acl_library_sha256': acl_hash,
                  'torch_version': torch.__version__,
                  'torch_npu_version': getattr(torch_npu, '__version__', 'unknown'),
                  'scope': 'imports, no-load provider identity, PACE op schema, plan identity '
                           'and review contract only; no device initialized'}), flush=True)
