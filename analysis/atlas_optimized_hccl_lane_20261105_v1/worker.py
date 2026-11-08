"""Four-arm optimized-HCCL campaign worker (one fresh container per group).

Arms H/HmR/HOPT delegate to the frozen, independently reviewed
uninstrumented-timing driver (run_timing.py of the 1bec8680… lane) with
REV2_NO_RESIZE_ARM native / no-resize; HOPT adds only the pre-registered
HCCL env delta (P0: none — identical configuration to HmR). The delegated
flow owns the captured gate, the native ten-binding capture removal, W3/I20
and every frozen validator; this worker afterwards normalizes the persisted
evidence into the lane's group record.

Arm F composes the frozen sync-model v1 lane (PACE hook + 4 GiB SHMEM heap,
manifest c755584d…) with this lane's RecorderBindings machinery: recorder
wrappers are removed after the gate, the PACE hook is retained through the
timing region, the SQ request-boundary occupancy check runs across
gate/w1/w2/w3 per rank, and validation always sits outside the timer.

Timing boundary (all arms): driver perf_counter_ns immediately around
llm.generate; no events, synchronize, profiler or allocator query inside
the measured path.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import plan
import restoration

LANE = Path('/workspace/lane')
RUNNER = Path('/workspace/no_resize_lane')             # corrected parity v2 365019d6…
TIMING = Path('/workspace/native_timing_lane')         # uninstrumented timing 1bec8680…
SYNC = Path('/workspace/model_hook')                   # sync-model v1 c755584d…
HELPER_PATH = Path('/workspace/queue_metadata/metadata.py')
LAYOUT_PATH = Path('/workspace/sdma_large/layout.json')
REFERENCE = Path('/workspace/reference/request_raw.json')
RESULTS_DIR = Path(os.environ.get('REV4_RESULT_DIR', '/workspace/results'))


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('unsafe or missing file: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_module(path, digest, name):
    if sha(path) != digest:
        raise ValueError('fixed dependency source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry_sha(root, manifest_pin, name):
    if sha(root / 'source_manifest.json') != manifest_pin:
        raise ValueError('frozen lane manifest differs: ' + str(root))
    manifest = json.loads((root / 'source_manifest.json').read_text())
    return manifest[name]


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def load_helper():
    return checked_module(HELPER_PATH, plan.HELPER_SHA256, 'lane_queue_helper')


# The census adapters (normalize_hccl_census / normalize_pace_census) live in
# plan.py so the worker, the launcher audit and the CPU tests share one
# implementation against the frozen predecessors' real evidence formats.


# ------------------------------------------------------------- F arm -----
def build_extension(frozen_cls):
    class Revision4PaceTimingExtension(frozen_cls):
        """Frozen sync-model session + lane recorder removal."""

        def install_revision4_pace_timing(self):
            if getattr(self, '_revision4_bindings', None) is not None:
                raise RuntimeError('installation is one-shot')
            # Spawn defense FIRST (device failure e9f94a35): this method runs
            # inside a vLLM-spawned worker process whose environ is a
            # snapshot of the parent's at spawn time — anything the driver
            # set after LLM construction is absent here. Prime the F-arm
            # contract (pins via setdefault; group-env names asserted).
            self._revision4_env_primed = plan.prime_f_worker_env(os.environ)
            from vllm_ascend import ascend_forward_context as context
            from vllm_ascend.worker import model_runner_v1 as runner_module
            from vllm_ascend.ops.fused_moe import token_dispatcher as dispatcher
            items = restoration.targets(self, context, runner_module, dispatcher)
            bindings = restoration.RecorderBindings(items)
            installed = super().install_revision2_sync_model()
            adopted = bindings.adopt_hook_layer()
            self._revision4_bindings = bindings
            return {'rank': installed.get('rank'), 'adopted': adopted,
                    'frozen_install': installed,
                    'slot_bytes': plan.SLOT_BYTES, 'heap_bytes': plan.HEAP_BYTES}

        def revision4_remove_recorder(self):
            snapshot = super().revision2_sync_model_snapshot()
            digest = hashlib.sha256(json.dumps(snapshot.get('records', snapshot),
                                               sort_keys=True).encode()).hexdigest()
            self._revision4_gate_capture_digest = digest
            restored = self._revision4_bindings.check_wrapped().restore()
            return {'rank': snapshot.get('rank'), 'restored': restored,
                    'gate_capture_sha256': digest,
                    'recorder_removed': True, 'hook_retained': True}

        def revision4_pace_timing_snapshot(self):
            snapshot = super().revision2_sync_model_snapshot()
            digest = hashlib.sha256(json.dumps(snapshot.get('records', snapshot),
                                               sort_keys=True).encode()).hexdigest()
            unchanged = digest == getattr(self, '_revision4_gate_capture_digest', None)
            # census inputs in the session snapshot's real format: counts
            # (python_pace_calls/python_hccl_calls) + backend-keyed records
            return {'rank': snapshot.get('rank'),
                    'bindings': self._revision4_bindings.verify(),
                    'recorder_digest': digest,
                    'recorder_digest_unchanged': unchanged,
                    'census_row': {'counts': snapshot.get('counts'),
                                   'records': snapshot.get('records')}}

        def revision4_sq_boundary(self, label):
            import torch
            import torch_npu
            rank = torch.distributed.get_rank()
            torch_npu.npu.synchronize(rank)
            helper = load_helper()
            queues = helper.capture(LAYOUT_PATH, plan.LAYOUT_SHA256, '/workspace/shmem',
                                    os.environ['REV4_ACL_LIBRARY_PATH'],
                                    os.environ['REV4_ACL_LIBRARY_SHA256'], rank, plan.WORLD,
                                    quiescent=True)
            return {'rank': rank, 'label': label, 'channels': queues['channels']}

    return Revision4PaceTimingExtension


def _load_extension_base():
    module = checked_module(SYNC / 'model_extension.py',
                            entry_sha(SYNC, plan.SYNC_MODEL_MANIFEST, 'model_extension.py'),
                            'frozen_sync_extension')
    return module.Revision2SyncModelExtension


# Composed once per process (driver and every spawned vLLM worker) from the
# SHA-checked frozen base; deterministic under the pinned mounts.
Revision4PaceTimingExtension = build_extension(_load_extension_base())


def run_f_arm(group_key, level, proof_only=False):
    """F flow: install -> gate+census -> recorder removal -> one-request
    restoration proof -> (proof_only: stop here) -> W3/I20 with per-rank SQ
    boundary occupancy check."""
    from vllm import LLM, SamplingParams
    if sha(REFERENCE) != plan.REFERENCE_RAW_SHA:
        raise ValueError('reference request_raw changed')
    run_o1 = checked_module(RUNNER / 'run_o1.py',
                            entry_sha(RUNNER, plan.RUNNER_MANIFEST, 'run_o1.py'), 'frozen_run_o1')
    reference = json.loads(REFERENCE.read_text())
    if reference['prompt_batch_sha256'] != plan.PROMPT_BATCH_SHA \
            or reference['output_tokens'] != plan.OUTPUT_TOKENS:
        raise ValueError('reference scope differs')
    model = Path('/workspace/model')
    helper_path = Path('/workspace/legacy/benchmark_olmoe_fixed_batch.py')
    if sha(helper_path) != run_o1.PROMPT_SOURCE_SHA:
        raise ValueError('prompt helper changed')
    spec = importlib.util.spec_from_file_location('fixed_prompt_helper', helper_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    prompts = prompt_module.prompt_batch(str(model), 1024, 16, 'heterogeneous')
    if prompt_module.prompt_hash(prompts) != plan.PROMPT_BATCH_SHA:
        raise ValueError('prompt content changed')
    # Delegated env contract for F, set BEFORE LLM construction: vLLM spawns
    # worker processes during engine init, and a spawned child's environ is a
    # snapshot of the parent's at spawn time — anything set later is absent
    # in the child (device failure e9f94a35). The install method re-primes
    # defensively (setdefault) inside each worker process as well.
    for name, value in plan.delegated_env('F').items():
        os.environ[name] = str(value)
    llm = LLM(model=str(model), tensor_parallel_size=16, enable_expert_parallel=True,
              distributed_executor_backend='mp', dtype='bfloat16', trust_remote_code=True,
              enforce_eager=True, disable_custom_all_reduce=True, gpu_memory_utilization=0.50,
              max_model_len=2048, max_num_batched_tokens=16384, max_num_seqs=16, block_size=128,
              enable_chunked_prefill=False, enable_prefix_caching=False, async_scheduling=False,
              seed=plan.SEED, worker_extension_cls='worker.Revision4PaceTimingExtension')
    install = llm.collective_rpc('install_revision4_pace_timing', timeout=None)
    write_new(RESULTS_DIR / 'installation.json', install)
    inputs = [{'prompt_token_ids': p} for p in prompts]
    sampling = SamplingParams(temperature=0.0, max_tokens=plan.OUTPUT_TOKENS, ignore_eos=True)

    def generate(index, phase):
        start = time.perf_counter_ns()
        outputs = llm.generate(inputs, sampling, use_tqdm=False)
        end = time.perf_counter_ns()
        tokens_equal = _tokens_equal_reference(outputs, reference['generated_token_ids'])
        return {'index': index, 'phase': phase, 'start_ns': start, 'end_ns': end,
                'generate_elapsed_ns': end - start,
                'clock': 'driver perf_counter_ns around llm.generate',
                'request_id': outputs[0].request_id,
                'output_tokens_equal_reference': tokens_equal,
                'validated_outside_timer': True, 'returned': outputs}

    boundaries = {}
    gate = generate(0, 'gate')
    if not gate['output_tokens_equal_reference']:
        raise RuntimeError('F gate request differs from the native reference')
    boundaries['gate'] = llm.collective_rpc('revision4_sq_boundary', args=('gate',), timeout=None)
    write_new(RESULTS_DIR / 'request_00.json', _plain(gate))
    removed = llm.collective_rpc('revision4_remove_recorder', timeout=None)
    write_new(RESULTS_DIR / 'capture_removed.json', removed)
    census_rows = llm.collective_rpc('revision4_pace_timing_snapshot', timeout=None)
    census = plan.normalize_pace_census([{'rank': row.get('rank', index),
                                          'counts': row['census_row']['counts'],
                                          'records': row['census_row']['records']}
                                         for index, row in enumerate(census_rows)])
    plan.validate_pace_census(census)
    proof = generate(1, 'warmup')  # the one-request restoration proof (untimed)
    if not proof['output_tokens_equal_reference']:
        raise RuntimeError('one-request restoration proof differs from the reference')
    boundaries['warmup1'] = llm.collective_rpc('revision4_sq_boundary', args=('warmup1',), timeout=None)
    write_new(RESULTS_DIR / 'restoration_proof_request.json', _plain(proof))
    if proof_only:
        check = llm.collective_rpc('revision4_pace_timing_snapshot', timeout=None)
        if not all(row.get('recorder_digest_unchanged') for row in check):
            raise RuntimeError('recorder evidence changed after the restoration proof')
        finish = llm.collective_rpc('finish_revision2_sync_model', timeout=None)
        write_new(RESULTS_DIR / 'finalization.json', finish)
        write_new(RESULTS_DIR / 'f_proof_result.json',
                  {'recorder_removed': True, 'hook_retained': True,
                   'tokens_equal_reference': True,
                   'restored': [row.get('bindings') for row in removed]})
        print(json.dumps({'event': 'F_PROOF_PASS', 'group': 'f_proof', 'arm': 'F',
                          'recorder_removed': True, 'hook_retained': True,
                          'tokens_equal_reference': True}), flush=True)
        return None
    rows = [gate, proof]
    for index in range(2, plan.TOTAL_REQUESTS):
        phase = 'warmup' if index <= plan.WARMUP_REQUESTS else 'measurement'
        row = generate(index, phase)
        if not row['output_tokens_equal_reference']:
            raise RuntimeError('request tokens differ from the reference: %d' % index)
        rows.append(row)
        write_new(RESULTS_DIR / ('request_%02d.json' % index), _plain(row))
        if phase == 'warmup':
            label = 'warmup%d' % (index - 1)
            boundaries[label] = llm.collective_rpc('revision4_sq_boundary', args=(label,), timeout=None)
    post = llm.collective_rpc('revision4_pace_timing_snapshot', timeout=None)
    write_new(RESULTS_DIR / 'after_timing.json', post)
    if not all(row.get('recorder_digest_unchanged') for row in post):
        raise RuntimeError('recorder evidence changed during the timing region')
    # Explicit rank-order assertion per boundary before the field is dropped
    # (collective_rpc ordering is rank order, but never assumed silently).
    for label, value in boundaries.items():
        if [row['rank'] for row in value] != list(range(plan.WORLD)):
            raise RuntimeError(f'SQ boundary rows not in rank order: {label}')
    sq_boundaries = {label: [row['channels'] for row in value]
                     for label, value in boundaries.items()}
    plan.validate_sq_boundaries(sq_boundaries)
    finish = llm.collective_rpc('finish_revision2_sync_model', timeout=None)
    write_new(RESULTS_DIR / 'finalization.json', finish)
    return {'event': 'group_samples', 'group': group_key, 'arm': 'F', 'world': plan.WORLD,
            'ladder_level': level, 'image_id': plan.IMAGE,
            'clock': 'driver perf_counter_ns around llm.generate',
            'census': census, 'slot_bytes': plan.SLOT_BYTES, 'heap_bytes': plan.HEAP_BYTES,
            'f_restoration': {'recorder_removed': True, 'hook_retained': True,
                              'recorder_digest_unchanged': True},
            'sq_boundaries': sq_boundaries,
            'requests': [{k: v for k, v in row.items() if k != 'returned'} for row in rows],
            'restoration_proof': {'request_index': 1, 'tokens_equal_reference': True},
            'scope': 'PACE hook retained, recorder removed before W3/I20; '
                     'validation outside the timer'}


# ---------------------------------------------------------- HCCL arms ----
def run_hccl_arm(arm, group_key, level, hopt_env):
    """Delegate to the frozen uninstrumented-timing driver, then normalize."""
    if arm not in plan.HCCL_ARMS:
        raise ValueError('HCCL delegation requires an HCCL arm')
    # The delegated chain's env contract (grep audit of the frozen sources):
    # timing_contract reads REV2_NATIVE_TIMING_MANIFEST, run_timing/run_o1
    # read REV2_NO_RESIZE_SOURCE_MANIFEST/ARM/BACKEND/OUTPUT_TOKENS/RESULT_DIR
    # and the VLLM_* switches. Group env carries most of them; the two
    # manifest SHAs are set here from the plan pins, saved/restored around
    # the delegated run (device failure g01: the timing manifest was unset).
    saved = {}
    for name, value in plan.delegated_env(arm).items():
        saved[name] = os.environ.get(name)
        os.environ[name] = str(value)
    os.environ['REV2_NO_RESIZE_ARM'] = plan.DRIVER_ARM_ENV[arm]
    try:
        driver = checked_module(TIMING / 'run_timing.py',
                                entry_sha(TIMING, plan.TIMING_MANIFEST, 'run_timing.py'),
                                'frozen_native_timing_driver')
        driver.main()  # owns gate, native capture removal, W3/I20, frozen validators
    finally:
        for name, previous in saved.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    result = json.loads((RESULTS_DIR / 'result.json').read_text())
    if result.get('status') != 'PASS':
        raise RuntimeError('delegated timing driver did not pass')
    raw = json.loads((RESULTS_DIR / 'request_raw.json').read_text())
    trace = json.loads((RESULTS_DIR / 'validated_gate_trace.json').read_text())
    census = plan.normalize_hccl_census(raw['after_gate'], trace)
    plan.validate_hccl_census(census)
    requests = []
    for index in range(plan.TOTAL_REQUESTS):
        row = json.loads((RESULTS_DIR / ('request_%02d.json' % index)).read_text())
        requests.append({'index': index, 'phase': row['phase'], 'start_ns': row['start_ns'],
                         'end_ns': row['end_ns'], 'generate_elapsed_ns': row['generate_elapsed_ns'],
                         'clock': 'driver perf_counter_ns around llm.generate',
                         'request_id': row['returned'][0]['request_id'],
                         'output_tokens_equal_reference': True,
                         'validated_outside_timer': True})
    return {'event': 'group_samples', 'group': group_key, 'arm': arm, 'world': plan.WORLD,
            'ladder_level': level, 'image_id': plan.IMAGE,
            'clock': 'driver perf_counter_ns around llm.generate',
            'census': census, 'hopt_env': dict(hopt_env), 'requests': requests,
            'delegated_flow': 'frozen run_timing (uninstrumented native timing)',
            'scope': 'gate/W3/I20 with native capture removed before timing'}


def _tokens_equal_reference(outputs, reference_tokens):
    if len(outputs) != len(reference_tokens):
        return False
    for item, expected in zip(outputs, reference_tokens):
        if list(item.outputs[0].token_ids) != expected or not item.finished:
            return False
    return True


def _plain(row):
    value = {k: v for k, v in row.items() if k != 'returned'}
    value['returned'] = [{'request_id': item.request_id, 'finished': item.finished,
                          'outputs': [{'index': choice.index, 'token_ids': list(choice.token_ids)}
                                      for choice in item.outputs]} for item in row['returned']]
    return value


def main():
    arm = os.environ['REV4_ARM']
    group_key = os.environ['REV4_GROUP']
    level = os.environ.get('REV4_LADDER_LEVEL', 'L0')
    proof_only = os.environ.get('REV4_MODE') == 'f_proof'
    if proof_only:
        if arm != 'F' or group_key != 'f_proof':
            raise ValueError('f_proof mode requires the dedicated proof job binding')
    elif arm not in plan.ARMS or group_key not in plan.GROUPS or plan.GROUP_ARM[group_key] != arm:
        raise ValueError('arm/group binding differs from the reviewed schedule')
    if not proof_only and (level not in plan.LADDER or group_key not in plan.ladder_groups(level)):
        raise ValueError('ladder level/group binding differs')
    if sha(REFERENCE) != plan.REFERENCE_RAW_SHA:
        raise ValueError('reference request_raw changed')
    if not RESULTS_DIR.is_dir() or any(RESULTS_DIR.iterdir()):
        raise ValueError('a new empty evidence directory is required')
    if arm in plan.HCCL_ARMS:
        hopt = json.loads(os.environ.get('REV4_HOPT_ENV', '{}'))
        if arm != 'HOPT' and hopt:
            raise ValueError('only HOPT may carry an HCCL env delta')
        record = run_hccl_arm(arm, group_key, level, hopt)
    else:
        if not os.environ.get('HPA_SHMEM_IP_PORT'):
            raise ValueError('F arm requires the SHMEM rendezvous env')
        proof_only = os.environ.get('REV4_MODE') == 'f_proof'
        record = run_f_arm(group_key, level, proof_only)
        if record is None:
            return  # f_proof mode already emitted its own pass marker
    plan.validate_group_record(record, group_key, arm, level)
    median = plan.group_median_ms(record['requests'])
    write_new(RESULTS_DIR / 'group_record.json', record)
    print(json.dumps({'event': 'group_summary', 'group': group_key, 'arm': arm,
                      'world': plan.WORLD, 'ladder_level': level,
                      'group_median_ms': median, 'image_id': plan.IMAGE,
                      'record_sha256': sha(RESULTS_DIR / 'group_record.json')}), flush=True)


if __name__ == '__main__':
    main()
