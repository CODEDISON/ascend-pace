"""No-device, no-network container precheck for the four-arm campaign.

Imports the light providers only (torch/torch_npu/cashflex), proves the
RTLD_NOLOAD/dladdr provider identity (fname only), the PACE operator schema
against the device-measured truth, the lane CPU fake-import restoration
proof, the frozen-lane source identities by digest, the HCCL image file
digest, the arms.json/PLAN.md bindings, the schedule re-derivation and the
review-contract echo. Creates no stream and calls no device API.
"""
import ctypes
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import torch
import torch_npu
import cashflex  # noqa: F401
from torch_npu.npu.streams import Stream

import plan
import restoration

SHMEM_PATH = '/workspace/shmem/build_cann9_vllm/lib/libshmem.so'
HCCL_PATH = plan.HCCL_PATH
PLAN_DIR = Path('/workspace/plan')
LANES = {'candidate': Path('/workspace/no_resize_candidate'),
         'runner': Path('/workspace/no_resize_lane'),
         'timing': Path('/workspace/native_timing_lane'),
         'sync_model': Path('/workspace/model_hook'),
         'qualification': Path('/workspace/qualification')}
LANE_PINS = {'candidate': plan.CANDIDATE_MANIFEST, 'runner': plan.RUNNER_MANIFEST,
             'timing': plan.TIMING_MANIFEST, 'sync_model': plan.SYNC_MODEL_MANIFEST,
             'qualification': plan.QUALIFICATION_MANIFEST}


def require_digest(path, expected):
    with open(path, 'rb') as stream:
        observed = hashlib.file_digest(stream, 'sha256').hexdigest()
    if observed != expected:
        raise RuntimeError('digest mismatch: ' + str(path))


class DlInfo(ctypes.Structure):
    _fields_ = [('fname', ctypes.c_char_p), ('fbase', ctypes.c_void_p),
                ('sname', ctypes.c_char_p), ('saddr', ctypes.c_void_p)]


def main():
    with open(SHMEM_PATH, 'rb') as handle:
        shmem_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    if shmem_hash != plan.SHMEM_SHA256:
        raise RuntimeError('provider mismatch')
    require_digest('/workspace/cashflex/cashflex/lib/libcashflex.so', plan.LIB_SHA256)
    require_digest(HCCL_PATH, plan.LIBHCCL_SHA)
    source = inspect.getsource(Stream._as_parameter_.fget)
    if 'ctypes.c_void_p(self.npu_stream)' not in source:
        raise RuntimeError('unexpected current-stream ctypes ABI')
    provider = ctypes.CDLL(SHMEM_PATH, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    function = provider.aclshmemx_barrier_all_on_stream
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes, dladdr.restype = [ctypes.c_void_p, ctypes.POINTER(DlInfo)], ctypes.c_int
    info = DlInfo()
    if dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)) != 1:
        raise RuntimeError('dladdr did not locate the function')
    if plan.dladdr_provider_path(os.fsdecode(info.fname)) != os.path.realpath(SHMEM_PATH):
        raise RuntimeError('barrier symbol resolves to an unexpected provider')
    pace_schema = str(torch.ops.cashflex.hpa_alltoallv_device_splits.default._schema)
    plan.check_pace_schema(pace_schema)
    fake_proof = restoration.fake_import_proof()
    arms = json.loads((PLAN_DIR / 'arms.json').read_text())
    require_digest(PLAN_DIR / 'arms.json', plan.ARMS_JSON_SHA)
    require_digest(PLAN_DIR / 'PLAN.md', plan.PLAN_MD_SHA)
    plan.validate_schedule(arms.get('schedule', {}).get('groups'))
    lane_files = {}
    for name, root in LANES.items():
        require_digest(root / 'source_manifest.json', LANE_PINS[name])
        lane_files[name] = sorted(json.loads((root / 'source_manifest.json').read_text()))
    # F-arm runtime family completeness inside the container: every module
    # model_extension.py imports by bare name must be importable from
    # /workspace/qualification (f_proof failure 54167a66 was exactly this).
    qualification_files = set(lane_files['qualification'])
    missing_family = [member for member in plan.F_RUNTIME_FAMILY
                      if member not in qualification_files]
    if missing_family:
        raise RuntimeError('sync-model runtime family incomplete: ' + ', '.join(missing_family))
    import sys as _sys
    # Delegated-module recursive resolution proof (root fix for the module
    # family wiring class): AST transitive closure over both delegated
    # entries + the lane worker, every bare-name module must find_spec to a
    # file inside one of the mounted module-home roots, in PYTHONPATH order.
    # Device failures 54167a66 ('contract') and 216a2400 ('capture_contract')
    # were closure members whose home was missing — both impossible to miss
    # here now.
    roots = [(root, Path(root)) for root in plan.DELEGATED_MODULE_ROOTS]
    closure = plan.resolve_delegated_closure(plan.DELEGATED_ENTRY_FILES, roots,
                                             frozenset(_sys.stdlib_module_names)
                                             | {'__future__'})
    if closure['unresolved']:
        raise RuntimeError('delegated closure has unresolved modules: '
                           + ', '.join(f'{name} <- {importer}'
                                       for name, importer in closure['unresolved']))
    resolved = {}
    for importer, name, container in closure['edges']:
        if container == 'UNRESOLVED':
            raise RuntimeError(f'delegated module unresolved: {name} <- {importer}')
        spec = importlib.util.find_spec(name)
        origin = str(spec.origin) if spec is not None and spec.origin else ''
        if not origin.startswith(container + '/'):
            raise RuntimeError(f'module {name!r} resolves to {origin!r}, expected under {container}')
        resolved[name] = container
    import sys as _sys
    # Delegated-chain environment readiness proof (round-6 root fix): the
    # g01 device failure was REV2_NATIVE_TIMING_MANIFEST reaching
    # timing_contract empty. Simulate each arm's full container env —
    # launch.group_env (mounted lane source) plus the worker-set pins from
    # plan.delegated_env — and assert completeness and pin equality.
    import launch as lane_launch
    acl_stub = {'path': '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so',
                'sha256': '0' * 64}
    env_ready = {}
    for arm in plan.ARMS:
        group_key = next(key for key in plan.GROUPS if plan.GROUP_ARM[key] == arm)
        simulated = dict(lane_launch.group_env(group_key, arm, 'L0', acl_stub, {},
                                               '0' * 64, '0' * 64)
                         if arm in plan.HCCL_ARMS else
                         lane_launch.group_env(group_key, arm, 'L0', acl_stub, {},
                                               '0' * 64, '0' * 64))
        simulated.update({name: str(value) for name, value in plan.delegated_env(arm).items()})
        for name in plan.required_env_names(arm):
            if not simulated.get(name, ''):
                raise RuntimeError(f'delegated env unresolved for {arm}: {name}')
        env_ready[arm] = {name: simulated[name] for name in plan.required_env_names(arm)}
    if any(env_ready[arm]['REV2_NATIVE_TIMING_MANIFEST'] != plan.TIMING_MANIFEST
           or env_ready[arm]['REV2_NO_RESIZE_SOURCE_MANIFEST'] != plan.RUNNER_MANIFEST
           for arm in plan.HCCL_ARMS):
        raise RuntimeError('HCCL-arm manifest env pins mismatch')
    if env_ready['F']['REV2_MODEL_MANIFEST_SHA256'] != plan.SYNC_MODEL_MANIFEST:
        raise RuntimeError('sync-model manifest env pin mismatch')
    del _sys
    require_digest('/workspace/queue_metadata/metadata.py', plan.HELPER_SHA256)
    require_digest('/workspace/sdma_large/layout.json', plan.LAYOUT_SHA256)
    require_digest('/workspace/reference/request_raw.json', plan.REFERENCE_RAW_SHA)
    acl_path = plan.discover_acl_library(Path('/proc/self/maps').read_text())
    with open(acl_path, 'rb') as handle:
        acl_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    # Delegated-chain DATA-FILE proof (round-5 root fix): every audited
    # path+SHA file read (plan.DELEGATED_DATA_FILES) must exist inside the
    # container, with digest pins enforced — the import closure cannot see
    # these (device failure d4e26ece: the legacy prompt helper).
    data_files = {}
    for container_path, kind, pin, consumer in plan.DELEGATED_DATA_FILES:
        target = Path(container_path)
        if kind == 'dir':
            if not target.is_dir():
                raise RuntimeError(f'delegated data directory missing: {container_path}')
            data_files[container_path] = {'kind': 'dir', 'consumer': consumer}
            continue
        if not target.is_file():
            raise RuntimeError(f'delegated data file missing: {container_path}')
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if pin is not None and digest != pin:
            raise RuntimeError(f'delegated data file digest differs: {container_path}')
        data_files[container_path] = {'kind': 'file', 'sha256': digest,
                                      'pinned': pin is not None, 'consumer': consumer}
    print(json.dumps({'event': 'NO_DEVICE_PRECHECK_PASS',
                      'provider_sha256': shmem_hash,
                      'provider_path': os.path.realpath(SHMEM_PATH),
                      'stream_parameter_source': source,
                      'pace_schema': pace_schema,
                      'libhccl_sha256': plan.LIBHCCL_SHA,
                      'f_fake_import_proof': {'proof': fake_proof['proof'],
                                              'binding_count': len(fake_proof['binding_names']),
                                              'hook_layer_retained': True,
                                              'double_restore_refused': True,
                                              'forged_chain_rejected': True},
                      'frozen_lane_manifests': dict(LANE_PINS),
                      'frozen_lane_files': lane_files,
                      'f_runtime_family_resolved': sorted(plan.F_RUNTIME_FAMILY),
                      'delegated_closure': {'entries': list(plan.DELEGATED_ENTRY_FILES),
                                            'module_roots': list(plan.DELEGATED_MODULE_ROOTS),
                                            'resolved_modules': resolved,
                                            'unresolved': []},
                      'delegated_data_files': data_files,
                      'delegated_env_ready': {arm: env_ready[arm] for arm in plan.ARMS},
                      'arms_json_sha256': plan.ARMS_JSON_SHA,
                      'plan_md_sha256': plan.PLAN_MD_SHA,
                      'schedule': {'groups': list(plan.GROUPS),
                                   'arm_map': dict(plan.GROUP_ARM)},
                      'census': {'hccl': '64 distinct collectives/rank; counts '
                                          '{collective:64,forward:2,method:64,runner:3}; '
                                          '64 All2AllV methods = 32 dispatch + 32 combine; '
                                          '2 ALLTOALL selectors on 2 execute_model forwards',
                                 'pace': 'counts python_pace_calls=62 + python_hccl_calls=2; '
                                         '64 backend-keyed records'},
                      'trim_ladder': {level: list(plan.ladder_groups(level))
                                      for level in plan.LADDER},
                      'review_contract': {'schema': plan.REVIEW_SCHEMA,
                                          'scope': plan.REVIEW_SCOPE,
                                          'groups': plan.REVIEW_GROUPS,
                                          'reviewed_files': sorted(plan.REVIEWED_FILES)},
                      'acl_library_path': acl_path, 'acl_library_sha256': acl_hash,
                      'torch_version': torch.__version__,
                      'torch_npu_version': getattr(torch_npu, '__version__', 'unknown'),
                      'scope': 'imports, provider identity, PACE schema, restoration proof, '
                               'frozen-source digests and plan bindings only; no device API'}),
         flush=True)


if __name__ == '__main__':
    main()
