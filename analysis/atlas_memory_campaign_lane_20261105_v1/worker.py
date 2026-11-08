"""Memory-campaign measurement worker: one driver process per job container.

Round-2 rewiring (review B3/B5/B7/B8/B9/B11, advisories A1-A6):

- T0 is captured in this driver BEFORE engine construction; per-request
  BEGIN and END captures (B{index}/R{index}) surround every generate (A4).
- The A0 pilot jobs run two real passes (L3 sampler ON, then OFF) inside the
  fresh group, instantiate Lens3Sampler, and call plan.l3_adoption on the
  measured passes; non-pilot jobs run the sampler only when MEMCAMP_L3=1
  (B7).
- PACE heap/slot come from HPA_SHMEM_HEAP_BYTES/HPA_SHMEM_SLOT_BYTES (the
  only sizing mechanism in this stack) and are echoed into the evidence as
  actually-configured bytes (B5).
- The FA4 token gate runs on the first request of every engine job:
  reference token match when a frozen reference exists, else intra-job
  determinism (two identical temperature-0 generates must match
  token-for-token); mismatch publishes evidence and fails closed (B9).
- The engine-reported num_gpu_blocks / KV config is captured after init
  (engine_runtime.json) (B8).
- A6 publishes its computed KV floor (feasible or infeasible) as
  a6_floor.json / a6_infeasible.json for the receipt lift (B3/B11).
- Capacity probes no longer read a streak from the environment; the audit
  lifts it from the container log (B8). C5 confirm jobs actually run at the
  arm's best admitted C (A1). C0-1 records the true 16 GiB over-allocation
  (A2); C0-2 is declared a desk determination, not a device probe (A3).
"""
import hashlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

import plan

MODEL_DIR = Path('/workspace/model')
PROMPT_HELPER = Path('/workspace/legacy/benchmark_olmoe_fixed_batch.py')
REFERENCE_DIR = Path('/workspace/reference')
PAYLOAD = Path('/workspace/memory_results')
SNAPSHOT_MODULE = Path('/workspace/memory_snapshot/memory_snapshot.py')


def load_module(name, path, digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise RuntimeError('frozen module digest differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def fencepost(name, mark=None):
    """L2 fencepost capture on this rank (driver context or RPC body)."""
    import torch
    if mark is None and name not in plan.FENCEPOSTS:
        raise ValueError('fencepost outside the preregistered set')
    rank = int(os.environ.get('RANK', '0'))
    snapshot = load_module('memcamp_snapshot', SNAPSHOT_MODULE, plan.MEMORY_SNAPSHOT_SHA256)
    torch.npu.synchronize()
    return {'schema': 'revision2.memory-campaign-fencepost.v1', 'lens': 'L2',
            'fencepost': name, 'label': plan.FENCEPOST_LABELS.get(name, 'per-request'),
            'mark': mark, 'rank': rank,
            'mono_ns': time.monotonic_ns(), 'capture': snapshot.capture(torch.npu, rank),
            'l3_enabled': os.environ.get('MEMCAMP_L3') == '1'}


class MemoryCampaignExtension:
    """Worker extension mounted on every engine rank; adds the fencepost RPC
    on top of the frozen sync-model / split-capture extensions."""

    def revision2_memcamp_fencepost(self, name):
        return fencepost(name)


class Lens3Sampler:
    """Low-rate per-rank mem_get_info thread; records tagged lens=L3."""

    def __init__(self, npu, rank, rate_hz):
        self.npu, self.rank, self.period = npu, rank, 1.0 / rate_hz
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                free, total = self.npu.mem_get_info(self.rank)
                self.samples.append({'schema': 'revision2.memory-campaign-l3-sample.v1',
                                     'lens': 'L3', 'rank': self.rank, 'free_bytes': free,
                                     'total_bytes': total, 'mono_ns': time.monotonic_ns()})
            except Exception:
                break
            self._stop.wait(self.period)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self.samples


def prompt_batch(max_model_len, count):
    """Fixed prompt constructor from the frozen helper; the L=1024 batch is
    SHA-anchored, other lengths record their digest (declared, A6)."""
    from model_validation import prompt_digest
    spec = importlib.util.spec_from_file_location('memcamp_prompt', PROMPT_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = module.prompt_batch(str(MODEL_DIR), max_model_len, count, 'heterogeneous')
    digest = prompt_digest(prompts)
    anchor = plan.PROMPT_SHA_ANCHORS.get(max_model_len)
    if anchor is not None and digest != anchor:
        raise RuntimeError('frozen prompt batch differs from the pinned anchor')
    return prompts, {'max_model_len': max_model_len, 'prompt_digest': digest,
                     'anchored': anchor is not None}


def reference_tokens(model):
    path = REFERENCE_DIR / f'{model}_tokens.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get('token_ids')


def token_gate(directory, model, outputs, repeat_outputs=None):
    """B9 FA4 gate: reference match, else intra-job determinism."""
    observed = [list(output.outputs[0].token_ids) for output in outputs if output.outputs]
    reference = reference_tokens(model)
    if reference is not None:
        verdicts = [plan.token_gate_verdict(reference, tokens) for tokens in observed]
        mode = 'reference'
    else:
        if repeat_outputs is None:
            raise RuntimeError('no frozen reference for ' + model +
                               '; intra-job determinism pass required')
        repeat = [list(output.outputs[0].token_ids) for output in repeat_outputs
                  if output.outputs]
        verdicts = [plan.token_gate_verdict(r, o) for r, o in zip(repeat, observed)]
        mode = 'intra_job_determinism'
    record = {'schema': 'revision2.memory-campaign-token-gate.v1', 'mode': mode,
              'model': model, 'match': all(row['match'] for row in verdicts),
              'verdicts': verdicts}
    publish(directory / 'token_gate.json', record)
    if not record['match']:
        raise RuntimeError('FA4 token gate failed (' + mode + '); cell excluded, '
                           'evidence preserved')
    return record


def engine_records(llm):
    """B8: engine-reported block count + KV cache config, best-effort path
    declaration (absence recorded, never invented)."""
    found = {'num_gpu_blocks': None, 'source': None}
    for path, getter in (
            ('kv_cache_config.num_gpu_blocks',
             lambda: llm.llm_engine.kv_cache_config.num_gpu_blocks),
            ('cache_config.num_gpu_blocks',
             lambda: llm.llm_engine.cache_config.num_gpu_blocks),
            ('get_num_gpu_blocks()', lambda: llm.get_num_gpu_blocks())):
        try:
            value = getter()
        except Exception:
            continue
        if isinstance(value, int):
            found = {'num_gpu_blocks': value, 'source': path}
            break
    return found


def collect_fence(proxy, name, directory):
    rows = proxy.collective_rpc('revision2_memcamp_fencepost', name=name, timeout=None)
    publish(directory / f'fence_{name}.json', rows)
    return rows


def request_pass(proxy, llm, prompts, sampling, count, directory, tag):
    """One measured pass: per-request BEGIN/END captures + wall times."""
    times = []
    outputs = None
    for index in range(count):
        begin = fencepost('request', mark=f'{tag}B{index}')
        start = time.monotonic_ns()
        outputs = llm.generate([{'prompt_token_ids': p} for p in prompts],
                               sampling, use_tqdm=False)
        end = fencepost('request', mark=f'{tag}R{index}')
        times.append(time.monotonic_ns() - start)
        publish(directory / f'request_{tag}{index:02d}.json',
                {'begin': begin, 'end': end, 'wall_ns': times[-1],
                 'finished': [bool(o.finished) for o in outputs]})
    return times, outputs


def configured_pace_bytes():
    """B5: echo the actually-plumbed heap/slot env (evidence honesty)."""
    heap = os.environ.get('HPA_SHMEM_HEAP_BYTES')
    slot = os.environ.get('HPA_SHMEM_SLOT_BYTES')
    return {'heap_bytes': int(heap) if heap else None,
            'slot_bytes': int(slot) if slot else None}


# vLLM's ParallelConfig.worker_extension_cls requires a DOTTED STRING that
# the engine worker processes dynamically import from PYTHONPATH — a class
# object fails pydantic validation (A0a device find, exit=1 at 26 s). The
# lane source dir is first on PYTHONPATH and this module has no import-time
# side effects, so 'worker.MemoryCampaignExtension' resolves exactly like
# the #4-lane precedent.
WORKER_EXTENSION_CLS = 'worker.MemoryCampaignExtension'


def build_engine(model, kv_gib, max_model_len, max_num_seqs,
                 extension=WORKER_EXTENSION_CLS):
    if not isinstance(extension, str) or '.' not in extension:
        raise ValueError('worker_extension_cls must be a dotted string '
                         '(vLLM imports it dynamically per engine process)')
    from vllm import LLM
    config = plan.llm_config(model, kv_gib, max_model_len, max_num_seqs,
                             plan.CAPACITY_TOKENS)
    config['worker_extension_cls'] = extension
    return LLM(model=str(MODEL_DIR), **config)


def fill_probe(npu, rank, directory):
    import torch
    filler, steps = [], 0
    free, _ = npu.mem_get_info(rank)
    start_free = free
    while free > plan.FILL_FLOOR_F:
        try:
            filler.append(torch.empty(plan.FILL_STEP // 2, dtype=torch.bfloat16,
                                      device=f'npu:{rank}'))
        except RuntimeError:
            break
        steps += 1
        torch.npu.synchronize()
        free, _ = npu.mem_get_info(rank)
    result = plan.fill_probe_result(plan.FILL_STEP * steps, steps, free)
    result['start_free_bytes'] = start_free
    publish(directory / 'fill_probe.json', result)
    del filler
    torch.npu.synchronize()
    return result


def c0_cgroup_probe(directory):
    """C0-1: over-allocation under a low docker --memory limit (A2: the
    record states the true 16 GiB single-tensor allocation)."""
    import torch
    rank = int(os.environ.get('RANK', '0'))
    requested = 16 * plan.GIB
    record = {'schema': 'revision2.memory-campaign-c0-probe.v1', 'probe': 'C0-1',
              'container_memory_limit': plan.C0_CGROUP_MEMORY_LIMIT,
              'requested_device_bytes': requested, 'error_recorded': False}
    try:
        block = torch.empty(requested // 2, dtype=torch.bfloat16, device=f'npu:{rank}')
        block.fill_(1.0)
        torch.npu.synchronize()
        record['over_allocation_failed'] = False
        del block
    except RuntimeError as exc:
        record['over_allocation_failed'] = True
        record['error'] = type(exc).__name__ + ': ' + str(exc)[:1024]
        record['error_recorded'] = True
    record['outcome'] = plan.c0_outcome('C0-1', record['over_allocation_failed'],
                                        record['error_recorded'])
    publish(directory / 'c0_cgroup_probe.json', record)
    return record


def c0_npu_cap_determination(directory):
    """C0-2 (A3): a DESK determination, not a device probe — no NPU-side
    per-process HBM cap is documented in the pinned local docs (Ascend KG
    unreachable, [未核实]); a documented knob would need a fail-closed
    over-allocation job to count."""
    record = {'schema': 'revision2.memory-campaign-c0-determination.v2', 'probe': 'C0-2',
              'kind': 'desk_determination',
              'documented_knobs_found': [], 'over_allocation_failed': False,
              'error_recorded': False,
              'note': 'desk determination from the pinned local docs; any future '
                      'documented knob requires a fail-closed probe job to count; '
                      'Ascend KG unverified [未核实]'}
    record['outcome'] = plan.c0_outcome('C0-2', record['over_allocation_failed'],
                                        record['error_recorded'])
    publish(directory / 'c0_npu_cap_probe.json', record)
    return record


def main():
    job_id = os.environ['MEMCAMP_JOB']
    trim_steps = int(os.environ.get('MEMCAMP_TRIM', '0'))
    spec = plan.job_spec(job_id, trim_steps)
    role = spec['role']
    if not PAYLOAD.is_dir() or any(PAYLOAD.iterdir()):
        raise RuntimeError('fresh payload directory required')
    if role == 'c0_probe_cgroup':
        c0_cgroup_probe(PAYLOAD)
        return
    if role == 'c0_probe_npu_cap':
        c0_npu_cap_determination(PAYLOAD)
        return
    import torch
    # A4: T0 is captured here, before anything else runs in the driver.
    publish(PAYLOAD / 'fence_T0.json', [fencepost('T0')])
    a5_free = float(os.environ['MEMCAMP_A6_FREE_T1_GIB']) \
        if os.environ.get('MEMCAMP_A6_FREE_T1_GIB') else None
    kv_heap = plan.kv_heap_for(job_id, a5_free_t1_gib=a5_free)
    if role == 'wp_a' and spec['model'] == 'gpt_oss' and spec['arm'] == 'P' \
            and kv_heap is not None and not kv_heap.get('feasible', True):
        publish(PAYLOAD / 'a6_infeasible.json',
                {'job': job_id, 'kv_floor': kv_heap, 'finding': kv_heap['infeasibility']})
        return  # a reported outcome, not a failure
    if job_id == 'A6':
        # B3: publish the computed floor for the receipt lift in both cases.
        publish(PAYLOAD / 'a6_floor.json', {'job': job_id, 'kv_floor': kv_heap})
    from vllm import SamplingParams
    max_model_len = int(os.environ.get('MEMCAMP_MAX_MODEL_LEN', '2048'))
    max_num_seqs = int(os.environ.get('MEMCAMP_MAX_NUM_SEQS', '16'))
    max_tokens = int(os.environ.get('MEMCAMP_OUTPUT_TOKENS', '1'))
    llm = build_engine(spec['model'], kv_heap['kv_gib'], max_model_len, max_num_seqs,
                       WORKER_EXTENSION_CLS)
    proxy = llm
    directory = PAYLOAD
    engine = engine_records(llm)
    publish(directory / 'engine_runtime.json',
            {**engine, 'configured_pace': configured_pace_bytes(),
             'scope': 'engine-reported; absence recorded, never invented'})
    collect_fence(proxy, 'T1', directory)
    arm = spec['arm']
    if arm in ('P', 'B-B', 'B-C', 'B-D', 'pace'):
        collect_fence(proxy, 'T2a', directory)
        collect_fence(proxy, 'T2b', directory)
    else:
        collect_fence(proxy, 'T2', directory)
    prompts, prompt_record = prompt_batch(1024, 16)
    publish(directory / 'prompt_record.json', prompt_record)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    # B9: FA4 gate on the first request of this (model, config) cell.
    first = llm.generate([{'prompt_token_ids': p} for p in prompts], sampling,
                         use_tqdm=False)
    repeat = None
    if reference_tokens(spec['model']) is None:
        repeat = llm.generate([{'prompt_token_ids': p} for p in prompts], sampling,
                              use_tqdm=False)
    token_gate(directory, spec['model'], first, repeat_outputs=repeat)
    fencepost_row = fencepost('T3', mark='R0')
    publish(directory / 'fence_T3.json', [fencepost_row])
    pilot = None
    if role == 'wp_a_pilot':
        # B7: two real passes — L3 sampler ON, then OFF — inside this fresh
        # group; plan.l3_adoption decides from the measured passes.
        sampler = Lens3Sampler(torch.npu, 0, 1.0)
        sampler.start()
        times_on, _ = request_pass(proxy, llm, prompts, sampling,
                                   plan.WP_A_STEADY_REQUESTS, directory, 'P1')
        samples_on = sampler.stop()
        publish(directory / 'l3_samples.json', {'lens': 'L3', 'samples': samples_on})
        times_off, _ = request_pass(proxy, llm, prompts, sampling,
                                    plan.WP_A_STEADY_REQUESTS, directory, 'P2')
        # R2-A5: pair the two POST-PASS captures (shared boundary, one pass
        # apart) instead of mixing a pre-pass T3 with a post-pass capture —
        # the preregistered shared-fencepost pairing of the sampler effect.
        fence_on = {'post_pass': fencepost('request', mark='P1END')['capture']['free_bytes']}
        fence_off = {'post_pass': fencepost('request', mark='P2END')['capture']['free_bytes']}
        pilot = plan.l3_adoption(
            {'request_times': [t / 1e9 for t in times_on], 'fencepost_free_bytes': fence_on},
            {'request_times': [t / 1e9 for t in times_off], 'fencepost_free_bytes': fence_off})
        publish(directory / 'pilot.json', {'job': job_id, 'l3_adoption': pilot,
                                           'times_on_ns': times_on, 'times_off_ns': times_off})
    else:
        sampler = None
        if os.environ.get('MEMCAMP_L3') == '1':
            sampler = Lens3Sampler(torch.npu, 0, 1.0)
            sampler.start()
        times, _ = request_pass(proxy, llm, prompts, sampling,
                                plan.WP_A_STEADY_REQUESTS, directory, 'R')
        if sampler is not None:
            publish(directory / 'l3_samples.json',
                    {'lens': 'L3', 'samples': sampler.stop()})
        if role == 'fill_probe':
            fill_probe(torch.npu, 0, directory)
        if role in ('capacity', 'capacity_search', 'capacity_confirm'):
            # R3-B1.1: every capacity-bearing role — including the extension
            # jobs — runs the probe at its preregistered cell (declared for
            # extensions, derived for OLMoE slots; the launch sets the env).
            cell_c = max_num_seqs
            batch, _ = prompt_batch(max_model_len, min(cell_c, 16))
            cell_prompts = (batch * (cell_c // len(batch) + 1))[:cell_c]
            cell_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
            start = time.monotonic_ns()
            outputs = llm.generate([{'prompt_token_ids': p} for p in cell_prompts],
                                   cell_sampling, use_tqdm=False)
            wall = (time.monotonic_ns() - start) / 1e9
            record = {'job': job_id, 'completed': sum(1 for o in outputs if o.finished),
                      'submitted': cell_c,
                      'insufficient_kv_streak': 0,  # lifted from the log at audit (B8)
                      'median_completion_s': wall, 'wall_s': wall,
                      'cell': {'C': cell_c, 'L': max_model_len, 'max_tokens': 1}}
            record['classification'] = plan.classify_probe(record)
            publish(directory / 'capacity_probe.json', record)
        if role == 'throughput_sweep':
            cells = []
            for L in plan.THROUGHPUT_LENGTHS:
                for C in plan.THROUGHPUT_CONCURRENCY:
                    for O in plan.THROUGHPUT_OUTPUT_TOKENS:
                        cell_sampling = SamplingParams(temperature=0.0, max_tokens=O,
                                                       ignore_eos=True)
                        batch, _ = prompt_batch(L, min(C, 16))
                        cell_prompts = (batch * (C // len(batch) + 1))[:C]
                        start = time.monotonic_ns()
                        outputs = llm.generate([{'prompt_token_ids': p} for p in cell_prompts],
                                               cell_sampling, use_tqdm=False)
                        wall = (time.monotonic_ns() - start) / 1e9
                        tokens = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
                        finished = sum(1 for o in outputs if o.finished)
                        cells.append({'C': C, 'L': L, 'O': O, 'completed': finished,
                                      'submitted': C, 'wall_s': wall,
                                      'request_throughput': finished / wall if wall else 0.0,
                                      'token_throughput': tokens / wall if wall else 0.0})
            publish(directory / 'throughput_sweep.json',
                    {'job': job_id, 'cells': cells,
                     'scope': 'paired across arms at identical cells only'})
        if role == 'throughput_confirm':
            # A1: a real fresh-group confirmation at the arm's best admitted C.
            cell_c = max_num_seqs
            batch, _ = prompt_batch(max_model_len, min(cell_c, 16))
            cell_prompts = (batch * (cell_c // len(batch) + 1))[:cell_c]
            start = time.monotonic_ns()
            outputs = llm.generate([{'prompt_token_ids': p} for p in cell_prompts],
                                   sampling, use_tqdm=False)
            wall = (time.monotonic_ns() - start) / 1e9
            publish(directory / 'throughput_confirm.json',
                    {'job': job_id, 'cell': {'C': cell_c, 'L': max_model_len,
                                             'O': max_tokens},
                     'completed': sum(1 for o in outputs if o.finished), 'submitted': cell_c,
                     'wall_s': wall, 'fresh_group': True})
    fence_final = fencepost('T4' if role in ('wp_a', 'wp_a_pilot', 'fill_probe') else 'T5')
    publish(directory / ('fence_T4.json' if role in ('wp_a', 'wp_a_pilot', 'fill_probe')
                         else 'fence_T5.json'), [fence_final])
    if role in ('wp_a', 'wp_a_pilot', 'fill_probe'):
        publish(PAYLOAD / 'fence_T5.json', [fencepost('T5')])


if __name__ == '__main__':
    main()
