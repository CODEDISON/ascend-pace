"""v3 barrier/control-cost worker: one rank per process, arm A/B/C by env.

Arm A repeats the v2 empty-barrier loop verbatim inside a fresh group.
Arm B drives one small real PACE producer call per round through the SHMEM
write path, quiets the stream, then times either barrier+sync (b1) or sync
only (b2) in alternating blocks, checking peer-visible payload every round.
Arm C replays the reverse-order lane's frozen Qwen 038 dispatch shape and
appends k in {0,1,2} stream-ordered all-world barriers after each complete
PACE call, timing host API entry through device synchronize.
All arms use host CLOCK_MONOTONIC nanoseconds and fail closed: no finalize,
retry or extra work after any fault.
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
    spec = importlib.util.spec_from_file_location('v3_verified_queue_helper', HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verified_acl_library():
    acl_path = plan.discover_acl_library(Path('/proc/self/maps').read_text())
    require_digest(acl_path, os.environ['REV3_ACL_LIBRARY_SHA256'])
    if acl_path != os.environ['REV3_ACL_LIBRARY_PATH']:
        raise RuntimeError('loaded ACL provider differs from the reviewed identity')
    return acl_path


def liveness(dist, world, rank, marker, label):
    marker.fill_(1)
    dist.all_reduce(marker)
    torch.npu.synchronize()
    if int(marker.cpu().item()) != world:
        raise RuntimeError(label)


def arm_a_measure(world, rank, group, stream, barrier):
    samples = []
    for index in range(plan.A_WARMUPS + plan.A_SAMPLES):
        start_ns = time.monotonic_ns()
        barrier(stream._as_parameter_)
        stream.synchronize()
        end_ns = time.monotonic_ns()
        if end_ns <= start_ns:
            raise RuntimeError('invalid host timer')
        if index >= plan.A_WARMUPS:
            samples.append([start_ns, end_ns])
    return {'event': 'arm_a_samples', 'arm': 'A', 'group': group, 'rank': rank, 'world': world,
            'warmups': plan.A_WARMUPS, 'samples': samples, 'clock': 'host_monotonic_ns',
            'scope': 'ctypes_enqueue_plus_same_stream_synchronize',
            'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256}


def arm_b_measure(world, rank, group, stream, barrier):
    device = f'npu:{rank}'
    elements = plan.B_PEER_ROW_BYTES
    # The frozen operator requires >= 1024 tail bytes of explicit backing
    # beyond the live rows (reverse-order-lane precedent); the tail rows are
    # canary-filled and verified untouched every round. Payload semantics
    # are unchanged: `values` sees exactly the same 16 x 1024 B live rows.
    tail_rows = plan.pace_tail_rows(plan.B_PEER_ROW_BYTES)
    input_backing = torch.full((world + tail_rows, elements), plan.INT8_CANARY,
                               dtype=torch.int8, device=device)
    output_backing = torch.full((world + tail_rows, elements), plan.INT8_CANARY,
                                dtype=torch.int8, device=device)
    values, out = input_backing[:world], output_backing[:world]
    for peer in range(world):
        raw = plan.segment_bytes('arm-b-producers', plan.B_DTYPE, rank, peer, 1,
                                 plan.B_PEER_ROW_BYTES)
        values[peer].copy_(torch.frombuffer(raw, dtype=torch.int8))
    expected_out = b''.join(plan.segment_bytes('arm-b-producers', plan.B_DTYPE, peer, rank, 1,
                                               plan.B_PEER_ROW_BYTES)
                            for peer in range(world))
    expected_values = b''.join(plan.segment_bytes('arm-b-producers', plan.B_DTYPE, rank, peer, 1,
                                                  plan.B_PEER_ROW_BYTES)
                               for peer in range(world))
    # Split vectors share the same explicit-backing tail contract (B1 device
    # failure 15a6e1f33a5b: "split vector requires an explicit backing
    # allocation with at least 1024 tail bytes"); reverse-order-lane form:
    # int32 backing with 256 canary tail elements (256 x 4 B = 1024 B).
    split_tail = plan.pace_split_tail_elements()
    send_backing = torch.full((world + split_tail,), plan.SPLIT_CANARY,
                              dtype=torch.int32, device=device)
    recv_backing = torch.full((world + split_tail,), plan.SPLIT_CANARY,
                              dtype=torch.int32, device=device)
    sd, rd = send_backing[:world], recv_backing[:world]
    # The canary backing replaces round-4's torch.ones allocation, so the
    # live split values must be restored explicitly after the slice (the
    # arm-B semantics: every peer sends and receives exactly one row),
    # mirroring arm C's live fills below. Without this the live splits stay
    # SPLIT_CANARY=173 and the operator's split-sum check fails on round 1.
    sd.fill_(1)
    rd.fill_(1)
    torch.npu.synchronize()
    rounds = []
    for phase, sub in plan.arm_b_rounds(group):
        torch.ops.cashflex.hpa_alltoallv_device_splits(values, out, rd, sd, 1, 0, -1, 0)
        stream.synchronize()  # producer quiet before the measured region
        start_ns = time.monotonic_ns()
        if sub == 'b1':
            barrier(stream._as_parameter_)
        stream.synchronize()
        end_ns = time.monotonic_ns()
        if end_ns <= start_ns:
            raise RuntimeError('invalid host timer')
        output_equal = bool(torch.equal(out.cpu().view(torch.uint8),
                                        torch.frombuffer(bytearray(expected_out), dtype=torch.uint8).reshape(world, elements)))
        input_equal = bool(torch.equal(values.cpu().view(torch.uint8),
                                       torch.frombuffer(bytearray(expected_values), dtype=torch.uint8).reshape(world, elements)))
        splits_equal = bool(sd.cpu().tolist() == [1] * world and rd.cpu().tolist() == [1] * world)
        input_tail_equal = bool(torch.all(input_backing[world:].cpu() == plan.INT8_CANARY).item())
        output_tail_equal = bool(torch.all(output_backing[world:].cpu() == plan.INT8_CANARY).item())
        split_tail_equal = bool(torch.all(send_backing[world:].cpu() == plan.SPLIT_CANARY).item()
                                 and torch.all(recv_backing[world:].cpu() == plan.SPLIT_CANARY).item())
        if not (output_equal and input_equal and splits_equal
                and input_tail_equal and output_tail_equal and split_tail_equal):
            raise RuntimeError('arm B payload visibility check failed')
        rounds.append({'phase': phase, 'sub': sub, 'start_ns': start_ns, 'end_ns': end_ns,
                       'output_equal': output_equal, 'input_equal': input_equal,
                       'splits_equal': splits_equal, 'input_tail_equal': input_tail_equal,
                       'output_tail_equal': output_tail_equal, 'split_tail_equal': split_tail_equal,
                       'tail_rows': tail_rows, 'split_tail_elements': split_tail})
    return {'event': 'arm_b_samples', 'arm': 'B', 'group': group, 'rank': rank, 'world': world,
            'clock': 'host_monotonic_ns', 'peer_row_bytes': plan.B_PEER_ROW_BYTES,
            'input_tail_rows': tail_rows, 'input_tail_bytes': tail_rows * plan.B_PEER_ROW_BYTES,
            'split_tail_elements': split_tail, 'split_tail_bytes': split_tail * plan.SPLIT_ITEM_BYTES,
            'subarm_scope': {'b1': 'ctypes_barrier_enqueue_plus_same_stream_synchronize',
                             'b2': 'same_stream_synchronize_only'},
            'producer': 'one forced-SDMA PACE device-split call per round, then same-stream quiet',
            'rounds': rounds, 'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256}


def arm_c_measure(world, rank, group, stream, barrier):
    device = f'npu:{rank}'
    case = plan.case038()
    send, recv = plan.rank_splits(rank)
    elements = plan.C_ROW_BYTES // 2
    rows_in, rows_out = sum(send), sum(recv)
    # Same explicit-backing tail contract as arm B (reverse-order-lane
    # precedent): >= 1024 tail bytes beyond the live rows on input AND
    # output, canary-filled and verified untouched at every checkpoint.
    tail_rows = plan.pace_tail_rows(plan.C_ROW_BYTES)
    input_backing = torch.full((rows_in + tail_rows, elements), plan.BF16_CANARY,
                               dtype=torch.bfloat16, device=device)
    output_backing = torch.full((rows_out + tail_rows, elements), plan.BF16_CANARY,
                                dtype=torch.bfloat16, device=device)
    values, out = input_backing[:rows_in], output_backing[:rows_out]
    for peer in range(world):
        if send[peer]:
            raw = plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, rank, peer,
                                     send[peer], plan.C_ROW_BYTES)
            start = sum(send[:peer])
            values[start:start + send[peer]].copy_(
                torch.frombuffer(raw, dtype=torch.uint8).view(torch.bfloat16).reshape(send[peer], elements))
    # Zero-count segments are SKIPPED exactly like the reverse lane's
    # host_values oracle: case038 is sparse (122 zero-row segments), and a
    # join without the skip is what killed C1 on device at the oracle build.
    expected_out = b''.join(plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, peer, rank,
                                               recv[peer], plan.C_ROW_BYTES)
                            for peer in range(world) if recv[peer])
    expected_values = b''.join(plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, rank, peer,
                                                  send[peer], plan.C_ROW_BYTES)
                               for peer in range(world) if send[peer])
    # Split vectors share the same explicit-backing tail contract (see arm B;
    # reverse-order-lane form: int32 backing + 256 canary tail elements).
    split_tail = plan.pace_split_tail_elements()
    send_backing = torch.full((world + split_tail,), plan.SPLIT_CANARY,
                              dtype=torch.int32, device=device)
    recv_backing = torch.full((world + split_tail,), plan.SPLIT_CANARY,
                              dtype=torch.int32, device=device)
    sd, rd = send_backing[:world], recv_backing[:world]
    sd.copy_(torch.tensor(send, dtype=torch.int32))
    rd.copy_(torch.tensor(recv, dtype=torch.int32))
    torch.npu.synchronize()
    sequence = plan.arm_c_iteration_plan()
    checkpoints = plan.arm_c_checkpoints()
    iterations, checks = [], []
    for index, (phase, k) in enumerate(sequence):
        torch.npu.synchronize()
        start_ns = time.monotonic_ns()
        torch.ops.cashflex.hpa_alltoallv_device_splits(values, out, rd, sd, 1, 0, -1, 0)
        for _ in range(k):
            barrier(stream._as_parameter_)
        torch.npu.synchronize()
        end_ns = time.monotonic_ns()
        if end_ns <= start_ns:
            raise RuntimeError('invalid host timer')
        if index in checkpoints:
            # Same uint8->BF16 view-then-reshape unit discipline as the send
            # side fill above: frombuffer(uint8).view(bfloat16) halves the
            # element count to the row width, THEN reshape to (rows, 2048);
            # both torch.equal sides stay BF16 (rows, elements). A uint8-
            # direct reshape(rows, 2048) needs only half the oracle bytes —
            # the exact C1 device failure (round-8).
            output_equal = bool(torch.equal(out.cpu(),
                                            torch.frombuffer(bytearray(expected_out), dtype=torch.uint8).view(torch.bfloat16).reshape(rows_out, elements)))
            input_equal = bool(torch.equal(values.cpu(),
                                           torch.frombuffer(bytearray(expected_values), dtype=torch.uint8).view(torch.bfloat16).reshape(rows_in, elements)))
            splits_equal = bool(sd.cpu().tolist() == send and rd.cpu().tolist() == recv)
            input_tail_equal = bool(torch.all(input_backing[rows_in:].cpu() == plan.BF16_CANARY).item())
            output_tail_equal = bool(torch.all(output_backing[rows_out:].cpu() == plan.BF16_CANARY).item())
            split_tail_equal = bool(torch.all(send_backing[world:].cpu() == plan.SPLIT_CANARY).item()
                                     and torch.all(recv_backing[world:].cpu() == plan.SPLIT_CANARY).item())
            if not (output_equal and input_equal and splits_equal
                    and input_tail_equal and output_tail_equal and split_tail_equal):
                raise RuntimeError('arm C exact output/input/split check failed')
            checks.append({'index': index, 'phase': phase, 'k': k,
                           'output_equal': output_equal, 'input_equal': input_equal,
                           'splits_equal': splits_equal, 'input_tail_equal': input_tail_equal,
                           'output_tail_equal': output_tail_equal, 'split_tail_equal': split_tail_equal})
        iterations.append({'phase': phase, 'k': k, 'start_ns': start_ns, 'end_ns': end_ns})
    return {'event': 'arm_c_samples', 'arm': 'C', 'group': group, 'rank': rank, 'world': world,
            'clock': 'host_monotonic_ns', 'case': case['name'], 'row_bytes': plan.C_ROW_BYTES,
            'dtype': plan.C_DTYPE, 'slot_bytes': case['slot_bytes'], 'heap_bytes': case['heap_bytes'],
            'input_tail_rows': tail_rows, 'input_tail_bytes': tail_rows * plan.C_ROW_BYTES,
            'split_tail_elements': split_tail, 'split_tail_bytes': split_tail * plan.SPLIT_ITEM_BYTES,
            'scope': 'pace_full_call_plus_k_barriers_through_device_synchronize',
            'iterations': iterations, 'checks': checks,
            'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256}


def main():
    arm = os.environ['REV3_ARM']
    group = int(os.environ['REV3_GROUP_INDEX'])
    if arm not in plan.ARMS or (arm, group) not in plan.GROUP_ORDER:
        raise ValueError('arm/group outside the reviewed plan')
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    if world != plan.WORLD or int(os.environ['LOCAL_RANK']) != rank or not 0 <= rank < world:
        raise ValueError('rank/world configuration not permitted')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != ','.join(map(str, range(world))):
        raise ValueError('incorrect visible devices')
    require_digest(SHMEM_PATH, plan.SHMEM_SHA256)
    require_digest(CASHFLEX_PATH, plan.LIB_SHA256)
    # NOLOAD ensures calls reach a provider already loaded by cashflex; no duplicate SHMEM state.
    provider = ctypes.CDLL(SHMEM_PATH, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    barrier = provider.aclshmemx_barrier_all_on_stream
    barrier.argtypes, barrier.restype = [ctypes.c_void_p], None
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    heap_bytes = plan.C_HEAP_BYTES if arm == 'C' else plan.AB_HEAP_BYTES
    slot_bytes = plan.C_SLOT_BYTES if arm == 'C' else plan.AB_SLOT_BYTES
    free_bytes, total_bytes = torch.npu.mem_get_info(rank)
    required = plan.required_free_bytes(arm, rank)
    adequate = isinstance(free_bytes, int) and isinstance(total_bytes, int) \
        and 0 < free_bytes <= total_bytes and free_bytes >= required
    # All ranks must agree on available memory before any SHMEM initialization.
    memory_failure = torch.tensor([int(not adequate)], dtype=torch.int32, device=f'npu:{rank}')
    dist.all_reduce(memory_failure)
    torch.npu.synchronize()
    if int(memory_failure.cpu().item()):
        dist.destroy_process_group()
        raise RuntimeError('a rank lacks the predeclared free-memory margin')
    reference = torch.empty(1, device=f'npu:{rank}')
    if not hpa_shmem_init(reference, rank=rank, world_size=world, ip_port=os.environ['HPA_SHMEM_IP_PORT'],
                          heap_bytes=heap_bytes, slot_bytes=slot_bytes):
        raise RuntimeError('SHMEM init returned false')
    clean = False
    try:
        directory = PAYLOAD / f'rank-{rank:02d}'
        directory.mkdir(parents=False, exist_ok=False)
        stream = torch.npu.current_stream()
        if not isinstance(stream.npu_stream, int) or stream.npu_stream == 0:
            raise RuntimeError('invalid NPU stream handle')
        # Flush any pending torch software queue before direct ACL submissions.
        stream.synchronize()
        torch.npu.synchronize()
        # Quiescent queue metadata plus a fail-closed submission budget for
        # this arm's exact PACE call plan, captured before any payload.
        helper = load_helper()
        acl_path = verified_acl_library()
        acl_sha = os.environ['REV3_ACL_LIBRARY_SHA256']
        queues = helper.capture(LAYOUT_PATH, plan.LAYOUT_SHA256, SHMEM_ROOT,
                                acl_path, acl_sha, rank, world, quiescent=True)
        budget = plan.plan_call_budgets(arm, group, rank, plan.arm_peer_bytes(arm, rank),
                                        helper, queues['channels'])
        publish(directory / 'queue_metadata.json', queues)
        publish(directory / 'budget.json', budget)
        publish(directory / 'admission.json',
                {'arm': arm, 'group': group, 'rank': rank, 'world': world,
                 'heap_bytes': heap_bytes, 'slot_bytes': slot_bytes,
                 'required_free_bytes': required, 'free_bytes_before_init': free_bytes})
        if arm == 'C':
            free_after, _ = torch.npu.mem_get_info(rank)
            if free_after < plan.arm_c_peak_tensor_bytes(rank) + plan.C_GIB:
                raise RuntimeError('insufficient free memory after SHMEM init')
        marker = torch.ones(1, dtype=torch.int32, device=f'npu:{rank}')
        liveness(dist, world, rank, marker, 'rendezvous all-reduce failed')
        if arm == 'A':
            record = arm_a_measure(world, rank, group, stream, barrier)
        elif arm == 'B':
            record = arm_b_measure(world, rank, group, stream, barrier)
        else:
            record = arm_c_measure(world, rank, group, stream, barrier)
        record['pace_calls'] = budget['pace_calls']
        record['budget_cumulative_sqes'] = budget['cumulative_sqes']
        liveness(dist, world, rank, marker, 'post-measurement group liveness failed')
        clean = True
    finally:
        # Device-side fault: do not issue finalization or additional work.
        if clean:
            torch.npu.synchronize()
            hpa_shmem_finalize(reference)
            dist.destroy_process_group()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
