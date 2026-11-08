"""Bounded SHMEM world barrier pilot; never executes the failed Alltoallv graph.

One process per rank. Host wall time includes ctypes enqueue and stream sync.
This is a barrier-only diagnostic, not the incremental cost inside PACE.
"""
import ctypes
import hashlib
import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import cashflex  # noqa: F401
from cashflex.ops.hpa_alltoallv import hpa_shmem_finalize, hpa_shmem_init

SHMEM_PATH = '/workspace/shmem/build_cann9_vllm/lib/libshmem.so'
SHMEM_SHA256 = 'c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71'
LIB_SHA256 = '24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9'
CASHFLEX_PATH = '/workspace/cashflex/cashflex/lib/libcashflex.so'


def require_digest(path, expected):
    with open(path, 'rb') as stream:
        observed = hashlib.file_digest(stream, 'sha256').hexdigest()
    if observed != expected:
        raise RuntimeError('library digest mismatch: ' + path)


def main():
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    if world not in (2, 16) or int(os.environ['LOCAL_RANK']) != rank or not 0 <= rank < world:
        raise ValueError('rank/world configuration not permitted')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != ','.join(map(str, range(world))):
        raise ValueError('incorrect visible devices')
    require_digest(SHMEM_PATH, SHMEM_SHA256)
    require_digest(CASHFLEX_PATH, LIB_SHA256)
    # NOLOAD ensures calls reach a provider already loaded by cashflex; no duplicate SHMEM state.
    provider = ctypes.CDLL(SHMEM_PATH, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    barrier = provider.aclshmemx_barrier_all_on_stream
    barrier.argtypes, barrier.restype = [ctypes.c_void_p], None
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    free_bytes, total_bytes = torch.npu.mem_get_info(rank)
    adequate = isinstance(free_bytes, int) and isinstance(total_bytes, int) and 0 < free_bytes <= total_bytes and free_bytes >= 2 * 1024 ** 3
    # All ranks must agree on available memory before any SHMEM initialization.
    memory_failure = torch.tensor([int(not adequate)], dtype=torch.int32, device=f'npu:{rank}')
    dist.all_reduce(memory_failure)
    torch.npu.synchronize()
    if int(memory_failure.cpu().item()):
        dist.destroy_process_group()
        raise RuntimeError('a rank lacks the predeclared 2 GiB free-memory margin')
    reference = torch.empty(1, device=f'npu:{rank}')
    if not hpa_shmem_init(reference, rank=rank, world_size=world, ip_port=os.environ['HPA_SHMEM_IP_PORT'],
                          heap_bytes=512 * 1024 * 1024, slot_bytes=32 * 1024):
        raise RuntimeError('SHMEM init returned false')
    clean = False
    try:
        stream = torch.npu.current_stream()
        if not isinstance(stream.npu_stream, int) or stream.npu_stream == 0:
            raise RuntimeError('invalid NPU stream handle')
        # Flush any pending torch software queue before direct ACL submissions.
        stream.synchronize()
        marker = torch.ones(1, dtype=torch.int32, device=f'npu:{rank}')
        dist.all_reduce(marker)
        torch.npu.synchronize()
        if int(marker.cpu().item()) != world:
            raise RuntimeError('rendezvous all-reduce failed')
        samples = []
        for index in range(60):
            start_ns = time.monotonic_ns()
            barrier(stream._as_parameter_)
            stream.synchronize()
            end_ns = time.monotonic_ns()
            if end_ns <= start_ns:
                raise RuntimeError('invalid host timer')
            if index >= 10:
                samples.append([start_ns, end_ns])
        # All ranks must reach a normal HCCL collective after the barrier cohort.
        marker.fill_(1)
        dist.all_reduce(marker)
        torch.npu.synchronize()
        if int(marker.cpu().item()) != world:
            raise RuntimeError('post-barrier group liveness failed')
        clean = True
    finally:
        # Device-side fault: do not issue finalization or additional work.
        if clean:
            torch.npu.synchronize()
            hpa_shmem_finalize(reference)
            dist.destroy_process_group()
    print(json.dumps({'event': 'barrier_samples', 'rank': rank, 'world': world,
                      'warmups': 10, 'samples': samples, 'clock': 'host_monotonic_ns',
                      'scope': 'ctypes_enqueue_plus_same_stream_synchronize',
                      'library_sha256': LIB_SHA256, 'shmem_sha256': SHMEM_SHA256}), flush=True)


if __name__ == '__main__':
    main()
