"""Sparse-distribution PACE-vs-HCCL worker: one rank per process, by env.

Per case (from the rotated group plan): one initial full PACE+HCCL correctness
check against the CPU oracle (bit-exact, five tail sentinels), 2 warmups per
arm WITHOUT the start gate (lane structure), 8 measured iterations per arm in
(base, reverse) x 4 alternating blocks, each measured iteration preceded by a
rank-0 flag + HCCL all-reduce(max) common start gate, then one final full
check plus a global all-reduce(SUM, bad) failure count. Both timing口径 are
produced from the same batch: per-rank t0/t1 give rank-local API->sync maxima
and cross-rank earliest-start/latest-finish makespans.

Fail-closed throughout: digest checks, RTLD_NOLOAD provider identity, memory
admission all-reduce before SHMEM init, quiescent queue capture plus a
fail-closed SQE budget over the exact planned call sequence, HCCL liveness
before and after the measured region, finalize only on the clean flag, no
retries or signals. Clock: same-container perf_counter_ns (CLOCK_MONOTONIC).
"""
import ctypes
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import cashflex  # noqa: F401
from cashflex.ops.hpa_alltoallv import hpa_shmem_finalize, hpa_shmem_init

import plan

SHMEM_PATH = '/workspace/shmem/build_cann9_vllm/lib/libshmem.so'
CASHFLEX_PATH = '/workspace/cashflex/cashflex/lib/libcashflex.so'
HELPER_PATH = '/workspace/queue_metadata/metadata.py'
LAYOUT_PATH = '/workspace/sdma_large/layout.json'
SHMEM_ROOT = '/workspace/shmem'
PAYLOAD = Path('/results')


def require_digest(path, expected):
    with open(path, 'rb') as stream:
        observed = hashlib.file_digest(stream, 'sha256').hexdigest()
    if observed != expected:
        raise RuntimeError('library digest mismatch: ' + path)


def load_helper():
    require_digest(HELPER_PATH, plan.HELPER_SHA256)
    spec = importlib.util.spec_from_file_location('sparse_dist_verified_queue_helper', HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verified_acl_library():
    acl_path = plan.discover_acl_library(Path('/proc/self/maps').read_text())
    require_digest(acl_path, os.environ['REVSD_ACL_LIBRARY_SHA256'])
    if acl_path != os.environ['REVSD_ACL_LIBRARY_PATH']:
        raise RuntimeError('loaded ACL provider differs from the reviewed identity')
    return acl_path


def liveness(dist, world, rank, marker, label):
    marker.fill_(1)
    dist.all_reduce(marker)
    torch.npu.synchronize()
    if int(marker.cpu().item()) != world:
        raise RuntimeError(label)


def bytes_digest(raw):
    return hashlib.sha256(raw).hexdigest()


def timed_call(arm, values, out_p, out_h, rd, sd, send, recv, npu, dist, gate_flag, rank, gated):
    npu.synchronize()
    if gated:
        # Common start gate, identical for both arms (PLAN section 5). It is a
        # HCCL collective and may warm the HCCL context in the HCCL arm; this
        # non-neutrality is declared in DESIGN.md and the record.
        gate_flag.fill_(1 if rank == 0 else 0)
        dist.all_reduce(gate_flag, op=dist.ReduceOp.MAX)
    start = time.perf_counter_ns()
    if arm == 'pace':
        torch.ops.cashflex.hpa_alltoallv_device_splits(values, out_p, rd, sd, 1, 0, -1, 0)
    else:
        dist.all_to_all_single(out_h, values, output_split_sizes=recv, input_split_sizes=send)
    npu.synchronize()
    return start, time.perf_counter_ns()


def full_check(values, out_p, out_h, ib, pb, hb, sb, rb, send, recv, nr_in, nr_out,
               world, expected_out, input_reference, torch):
    """Bit-exact oracle comparison plus the five tail sentinels."""
    exact = {
        'pace': bool(torch.equal(out_p.cpu().view(torch.uint8), expected_out)),
        'hccl': bool(torch.equal(out_h.cpu().view(torch.uint8), expected_out)),
        'input': bool(torch.equal(values.cpu().view(torch.uint8), input_reference)),
        'splits': bool(sb[:world].cpu().tolist() == send and rb[:world].cpu().tolist() == recv),
        'input_tail': bool(torch.all(ib[nr_in:].cpu() == plan.CANARY).item()),
        'pace_tail': bool(torch.all(pb[nr_out:].cpu() == plan.CANARY).item()),
        'hccl_tail': bool(torch.all(hb[nr_out:].cpu() == plan.CANARY).item()),
        'send_tail': bool(torch.all(sb[world:].cpu() == plan.CANARY).item()),
        'recv_tail': bool(torch.all(rb[world:].cpu() == plan.CANARY).item()),
    }
    return exact


def uint8_rows(raw, rows, width):
    """0-byte-safe uint8 row matrix.

    torch.frombuffer rejects empty buffers (device job0 find: a zero-RECEIVE
    rank — e.g. olmoe-o1-call-010 rank 11 — has a 0-byte expected output),
    so empty payloads become an explicitly-shaped 0-row tensor. Bit-exact
    comparisons then hold naturally: the arm's actual output must also be
    0 rows or the existing equality check fails closed.
    """
    if rows == 0 or len(raw) == 0:
        return torch.empty((0, width), dtype=torch.uint8)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(rows, width)


def run_case(case, canonical_index, position, variant, group_key, rank, world,
             torch, dist, npu, gate_flag, budget_cumulative, directory):
    device = f'npu:{rank}'
    elements = plan.ROW_BYTES // 2
    send, recv = plan.rank_splits(case, rank)
    nr_in, nr_out = sum(send), sum(recv)
    # Runner-embedded self-check: rebuild the splits from the embedded matrix.
    rebuilt_recv = [case['send_matrix_rows'][source][rank] for source in range(world)]
    if rebuilt_recv != recv or case['per_rank_output_rows'][rank] != sum(recv):
        raise RuntimeError('embedded matrix/split self-check failed: ' + case['id'])
    ib = torch.full((nr_in + plan.TAIL_SLACK_ROWS, elements), plan.CANARY,
                    dtype=torch.bfloat16, device=device)
    pb = torch.full((nr_out + plan.TAIL_SLACK_ROWS, elements), plan.CANARY,
                    dtype=torch.bfloat16, device=device)
    hb = torch.full((nr_out + plan.TAIL_SLACK_ROWS, elements), plan.CANARY,
                    dtype=torch.bfloat16, device=device)
    values, out_p, out_h = ib[:nr_in], pb[:nr_out], hb[:nr_out]
    input_raw = bytearray(plan.expected_bytes(case, rank, receive=False))
    expected_raw = plan.expected_bytes(case, rank, receive=True)
    if nr_in:  # a zero-SEND rank has a 0-row input; nothing to stage
        values.copy_(torch.frombuffer(input_raw, dtype=torch.uint8)
                     .view(torch.bfloat16).reshape(nr_in, elements))
    # uint8 column count is the byte width (elements counts BF16 halves);
    # both sides are 0-row safe for zero-send / zero-receive ranks.
    expected_out = uint8_rows(expected_raw, nr_out, elements * 2)
    input_reference = uint8_rows(input_raw, nr_in, elements * 2)
    sb = torch.full((world + 256,), plan.CANARY, dtype=torch.int32, device=device)
    rb = torch.full((world + 256,), plan.CANARY, dtype=torch.int32, device=device)
    sb[:world].copy_(torch.tensor(send, dtype=torch.int32))
    rb[:world].copy_(torch.tensor(recv, dtype=torch.int32))
    npu.synchronize()

    # Initial complete check: one PACE call then one HCCL call (PACE first in
    # every group, as in the lane; order influence is recorded via arm_first).
    torch.ops.cashflex.hpa_alltoallv_device_splits(values, out_p, rb[:world], sb[:world], 1, 0, -1, 0)
    npu.synchronize()
    dist.all_to_all_single(out_h, values, output_split_sizes=recv, input_split_sizes=send)
    npu.synchronize()
    initial = full_check(values, out_p, out_h, ib, pb, hb, sb, rb, send, recv,
                         nr_in, nr_out, world, expected_out, input_reference, torch)
    initial_digest = {'expected_sha256': bytes_digest(expected_raw),
                      'pace_output_sha256': bytes_digest(out_p.cpu().view(torch.uint8).numpy().tobytes()),
                      'hccl_output_sha256': bytes_digest(out_h.cpu().view(torch.uint8).numpy().tobytes())}
    if not all(initial.values()):
        raise RuntimeError('initial bit/canary check failed: ' + case['id'])

    iterations = []
    for phase, arm, block, slot in plan.iteration_plan(group_key):
        gated = plan.gate_enabled(phase)
        start, end = timed_call(arm, values, out_p, out_h, rb[:world], sb[:world],
                                send, recv, npu, dist, gate_flag, rank, gated)
        if end <= start:
            raise RuntimeError('invalid host timer')
        iterations.append({'phase': phase, 'arm': arm, 'block': block, 'slot': slot,
                           'gated': gated, 't0_ns': start, 't1_ns': end})

    final = full_check(values, out_p, out_h, ib, pb, hb, sb, rb, send, recv,
                       nr_in, nr_out, world, expected_out, input_reference, torch)
    final_digest = {'expected_sha256': bytes_digest(expected_raw),
                    'pace_output_sha256': bytes_digest(out_p.cpu().view(torch.uint8).numpy().tobytes()),
                    'hccl_output_sha256': bytes_digest(out_h.cpu().view(torch.uint8).numpy().tobytes())}
    bad = torch.tensor([int(not all(final.values()))], dtype=torch.int32, device=device)
    dist.all_reduce(bad, op=dist.ReduceOp.SUM)
    global_bad = int(bad.cpu().item())
    record = {'event': 'case_samples', 'schema': 'revision2.sparse-distribution-case.v1',
              'variant': variant, 'group': group_key, 'rank': rank, 'world': world,
              'case': case['id'], 'case_index': canonical_index, 'case_position': position,
              'kind': case['kind'], 'role': case['role'], 'dtype': plan.DTYPE,
              'row_bytes': plan.ROW_BYTES, 'generation': plan.generation_label(canonical_index),
              'generation_after_measurement': plan.PACE_CALLS_PER_CASE * (canonical_index + 1),
              'arm_first': plan.arm_first(group_key),
              'clock': 'same-host perf_counter_ns (single-container CLOCK_MONOTONIC)',
              'gate': 'rank0_flag_plus_hccl_all_reduce_max_before_t0_measured_only',
              'pace_api': 'device splits, forced SDMA', 'hccl_api': 'host split lists',
              'initial': {'exact': initial, **initial_digest},
              'final': {'exact': final, **final_digest},
              'iterations': iterations, 'global_bad': global_bad,
              'budget_case_cumulative_sqes': budget_cumulative,
              'validation_scope': 'initial and final full checks only; '
                                  'intermediate repetitions not individually checked',
              'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256}
    publish(directory / f'case-{canonical_index:03d}.json', record)
    print(json.dumps(record), flush=True)
    if global_bad:
        raise RuntimeError('global bit/canary failure: ' + case['id'])
    return record


def cold_start_prime(torch, dist, npu, rank, world, directory):
    """Round-9/10/11: one untimed/unchecked PACE + HCCL warm-through call
    BEFORE any measured case, on the REAL dense anchor case
    (plan.PRIMING_CASE_ID). Root cause chain: job3 hit HCCL aicpu 507018
    when the group's first alltoallv was the extreme-sparse diagonal case;
    the round-9 synthetic priming itself tripped the fault (round-10: real
    anchor case instead); round-11: the priming tensors lacked the 1024-byte
    tail backing the device-split ABI requires (input requires an explicit
    backing allocation with at least 1024 tail bytes — rank 7 device find),
    so the priming buffers now use EXACTLY the measured-case preparation
    pattern: padded backing + row slice for values/out, +256 split backing
    with canary tails for sd/rd."""
    case = plan.priming_case()
    send, recv = plan.rank_splits(case, rank)
    elements = plan.ROW_BYTES // 2
    device = f'npu:{rank}'
    # Barrier-v3 verified pattern (arm_b/arm_c rounds 4-6, identical to this
    # lane's run_case preparation since round 1): the device-split ABI
    # requires >= 1024 tail bytes behind the input/output and the split
    # vectors; backings are sliced, tails stay canary.
    values_backing = torch.full((sum(send) + plan.TAIL_SLACK_ROWS, elements),
                                plan.CANARY, dtype=torch.bfloat16, device=device)
    out_backing = torch.full((sum(recv) + plan.TAIL_SLACK_ROWS, elements),
                             plan.CANARY, dtype=torch.bfloat16, device=device)
    values, out = values_backing[:sum(send)], out_backing[:sum(recv)]
    sd_backing = torch.full((world + 256,), plan.CANARY, dtype=torch.int32, device=device)
    rd_backing = torch.full((world + 256,), plan.CANARY, dtype=torch.int32, device=device)
    sd, rd = sd_backing[:world], rd_backing[:world]
    sd.copy_(torch.tensor(send, dtype=torch.int32))
    rd.copy_(torch.tensor(recv, dtype=torch.int32))
    npu.synchronize()
    # Round-13: 5 warm-through iterations (one priming call proved
    # insufficient on device — job3 still tripped aicpu; job0-2 show the
    # same case passes after ~160 preceding calls). Each iteration is a
    # complete PACE call followed by a full synchronize so the queue
    # settles before the next; all 5 are budgeted (priming=True).
    for _ in range(plan.PRIMING_CALLS):
        torch.ops.cashflex.hpa_alltoallv_device_splits(values, out, rd, sd, 1, 0, -1, 0)
        npu.synchronize()
    dist.all_to_all_single(out, values, output_split_sizes=recv, input_split_sizes=send)
    # Drain: full synchronize pair plus a settle pause so the SDMA queue is
    # quiet before the budgeted measured sequence begins (0.2 s, round-13).
    npu.synchronize()
    npu.synchronize()
    time.sleep(0.2)
    event = {'event': 'cold_start_priming', 'rank': rank, 'world': world,
             'priming_case': plan.PRIMING_CASE_ID,
             'priming_calls': plan.PRIMING_CALLS,
             'tail_backing_rows': plan.TAIL_SLACK_ROWS,
             'split_tail_entries': 256,
             'timed': False, 'validated': False,
             'scope': 'cold-start kernel warm-through outside the measured '
                      'case sequence; included in the SQ budget (priming=True); '
                      'buffers carry the ABI-required tail backing'}
    print(json.dumps(event), flush=True)
    if directory is not None:
        publish(directory / 'cold_start_priming.json', event)
    return event


def main():
    variant = os.environ['REVSD_VARIANT']
    group_key = os.environ['REVSD_GROUP']
    drops = int(os.environ['REVSD_DROPS'])
    if os.environ.get('REVSD_CASES_SHA256') != plan.CASES_SHA256:
        raise RuntimeError('cases identity environment differs')
    document = plan.load_cases()
    case_order = plan.rotated_case_order(variant, group_key, drops)
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    if world != plan.WORLD or int(os.environ['LOCAL_RANK']) != rank or not 0 <= rank < world:
        raise ValueError('rank/world configuration not permitted')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != ','.join(map(str, range(world))):
        raise ValueError('incorrect visible devices')
    require_digest(SHMEM_PATH, plan.SHMEM_SHA256)
    require_digest(CASHFLEX_PATH, plan.LIB_SHA256)
    # NOLOAD: identity check on the provider already loaded by cashflex; the
    # start gate uses HCCL, and no SHMEM barrier symbol is called here.
    ctypes.CDLL(SHMEM_PATH, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    peak = max(plan.peak_tensor_bytes(case, rank) for case in document['cases']
               if case['id'] in {case_id for _, case_id in case_order})
    required = plan.HEAP_BYTES + peak + plan.GIB
    free_bytes, total_bytes = torch.npu.mem_get_info(rank)
    adequate = isinstance(free_bytes, int) and isinstance(total_bytes, int) \
        and 0 < free_bytes <= total_bytes and free_bytes >= required
    memory_failure = torch.tensor([int(not adequate)], dtype=torch.int32, device=f'npu:{rank}')
    dist.all_reduce(memory_failure)
    torch.npu.synchronize()
    if int(memory_failure.cpu().item()):
        dist.destroy_process_group()
        raise RuntimeError('a rank lacks the predeclared free-memory margin')
    reference = torch.empty(1, device=f'npu:{rank}')
    if not hpa_shmem_init(reference, rank=rank, world_size=world,
                          ip_port=os.environ['HPA_SHMEM_IP_PORT'],
                          heap_bytes=plan.HEAP_BYTES, slot_bytes=plan.SLOT_BYTES):
        raise RuntimeError('SHMEM init returned false')
    clean = False
    try:
        directory = PAYLOAD / f'rank-{rank:02d}'
        directory.mkdir(parents=False, exist_ok=False)
        free_after, _ = torch.npu.mem_get_info(rank)
        if free_after < peak + plan.GIB:
            raise RuntimeError('insufficient free memory after SHMEM init')
        helper = load_helper()
        acl_path = verified_acl_library()
        # Quiescent queue metadata plus the fail-closed SQE budget over the
        # EXACT planned call sequence of this group, before any payload.
        queues = helper.capture(LAYOUT_PATH, plan.LAYOUT_SHA256, SHMEM_ROOT,
                                acl_path, os.environ['REVSD_ACL_LIBRARY_SHA256'],
                                rank, world, quiescent=True)
        case_ids = [case_id for _, case_id in case_order]
        try:
            budget = plan.plan_call_budgets(case_ids, rank, helper, queues['channels'],
                                            priming=True)
        except ValueError as exc:
            publish(directory / 'sq_gate_reject.json',
                    {'variant': variant, 'group': group_key, 'rank': rank,
                     'reason': 'SQ_BUDGET_FAIL_CLOSED: ' + str(exc),
                     'channels': queues['channels'], 'case_ids': case_ids})
            raise
        publish(directory / 'queue_metadata.json', queues)
        publish(directory / 'budget.json', budget)
        publish(directory / 'admission.json',
                {'variant': variant, 'group': group_key, 'rank': rank, 'world': world,
                 'drops': drops, 'heap_bytes': plan.HEAP_BYTES, 'slot_bytes': plan.SLOT_BYTES,
                 'required_free_bytes': required, 'free_bytes_before_init': free_bytes,
                 'free_bytes_after_init': free_after, 'peak_tensor_bytes': peak,
                 'cases': len(case_ids), 'cases_sha256': plan.CASES_SHA256})
        # Round-9: cold-start warm-through BEFORE any measured case (the
        # aicpu 507018 job3 root cause); budgeted above with priming=True.
        cold_start_prime(torch, dist, torch.npu, rank, world, directory)
        gate_flag = torch.zeros(1, dtype=torch.int32, device=f'npu:{rank}')
        marker = torch.ones(1, dtype=torch.int32, device=f'npu:{rank}')
        liveness(dist, world, rank, marker, 'rendezvous all-reduce failed')
        cumulative_by_index = {row['case']: row['cumulative_after_case'] for row in budget['cases']}
        for position, (canonical_index, case_id) in enumerate(case_order):
            case = next(c for c in document['cases'] if c['id'] == case_id)
            run_case(case, canonical_index, position, variant, group_key, rank, world,
                     torch, dist, torch.npu, gate_flag, cumulative_by_index[case_id], directory)
        liveness(dist, world, rank, marker, 'post-measurement group liveness failed')
        clean = True
    finally:
        # Any fault leaves SHMEM unfinalized: no extra device work after a failure.
        if clean:
            torch.npu.synchronize()
            hpa_shmem_finalize(reference)
            dist.destroy_process_group()
    print(json.dumps({'event': 'rank_pass', 'variant': variant, 'group': group_key,
                      'rank': rank, 'world': world, 'cases': len(case_order),
                      'drops': drops, 'performance_eligible': False}), flush=True)


if __name__ == '__main__':
    main()
