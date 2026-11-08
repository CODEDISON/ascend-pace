"""No-device import/ABI/plan evidence; creates no stream and calls no device API."""
import ctypes
import hashlib
import importlib.util
import inspect
import json
import os

import torch
import torch_npu
import cashflex  # noqa: F401
from torch_npu.npu.streams import Stream

import plan

SHMEM_PATH = '/workspace/shmem/build_cann9_vllm/lib/libshmem.so'
HELPER_PATH = '/workspace/queue_metadata/metadata.py'
LAYOUT_PATH = '/workspace/sdma_large/layout.json'

with open(SHMEM_PATH, 'rb') as f:
    shmem_hash = hashlib.file_digest(f, 'sha256').hexdigest()
if shmem_hash != plan.SHMEM_SHA256:
    raise RuntimeError('provider mismatch')
source = inspect.getsource(Stream._as_parameter_.fget)
if 'ctypes.c_void_p(self.npu_stream)' not in source:
    raise RuntimeError('unexpected current-stream ctypes ABI')
provider = ctypes.CDLL(SHMEM_PATH, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
function = provider.aclshmemx_barrier_all_on_stream


class DlInfo(ctypes.Structure):
    _fields_ = [('fname', ctypes.c_char_p), ('fbase', ctypes.c_void_p),
                ('sname', ctypes.c_char_p), ('saddr', ctypes.c_void_p)]


dladdr = ctypes.CDLL(None).dladdr
dladdr.argtypes, dladdr.restype = [ctypes.c_void_p, ctypes.POINTER(DlInfo)], ctypes.c_int
info = DlInfo()
if dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)) != 1:
    raise RuntimeError('dladdr did not locate the function')
if os.path.realpath(os.fsdecode(info.fname)) != os.path.realpath(SHMEM_PATH):
    raise RuntimeError('barrier symbol resolves to unexpected provider')

# Pin the frozen PACE device-split ABI the producer arms rely on.
pace_schema = str(torch.ops.cashflex.hpa_alltoallv_device_splits.default._schema)
plan.check_pace_schema(pace_schema)
case = plan.case038()
for index in range(1, plan.ARM_GROUPS['B'] + 1):
    plan.arm_b_rounds(index)
for index in range(1, plan.ARM_GROUPS['C'] + 1):
    plan.arm_c_iteration_plan()
    plan.arm_c_checkpoints()

if hashlib.sha256(open(HELPER_PATH, 'rb').read()).hexdigest() != plan.HELPER_SHA256:
    raise RuntimeError('queue-metadata helper digest differs')
spec = importlib.util.spec_from_file_location('v3_precheck_helper', HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
layout = helper.load_layout(LAYOUT_PATH, plan.LAYOUT_SHA256)
helper.require_export(SHMEM_PATH, 'g_state', 'OBJECT', layout['state']['size'])
acl_path = plan.discover_acl_library(open('/proc/self/maps').read())
with open(acl_path, 'rb') as f:
    acl_hash = hashlib.file_digest(f, 'sha256').hexdigest()
helper.require_export(acl_path, 'aclrtMemcpy', 'FUNC')
print(json.dumps({'event': 'NO_DEVICE_PRECHECK_PASS', 'provider_path': os.fsdecode(info.fname),
                  'provider_sha256': shmem_hash, 'stream_parameter_source': source,
                  'pace_schema': pace_schema, 'case038': {'name': case['name'],
                  'case_sha256': case['case_sha256'], 'source_dataset_sha256': case['source_dataset_sha256'],
                  'row_bytes': case['row_bytes'], 'dtype': case['dtype']},
                  'metadata_helper_sha256': plan.HELPER_SHA256, 'layout_sha256': plan.LAYOUT_SHA256,
                  'acl_library_path': acl_path, 'acl_library_sha256': acl_hash,
                  'arm_b_rounds': [plan.arm_pace_calls('B', i) for i in range(1, plan.ARM_GROUPS['B'] + 1)],
                  'arm_c_iterations': [plan.arm_pace_calls('C', i) for i in range(1, plan.ARM_GROUPS['C'] + 1)],
                  'torch_version': torch.__version__,
                  'torch_npu_version': getattr(torch_npu, '__version__', 'unknown'),
                  'scope': 'imports, no-load provider, stream ABI, PACE op schema and frozen plan identity only'}), flush=True)
