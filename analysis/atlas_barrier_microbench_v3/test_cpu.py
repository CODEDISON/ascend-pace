"""Host logic tests only; no torch, device, processes or network.

Covers the supervisor contract, the frozen plan/schedule/metric functions,
the fail-closed record validators, the launcher container inspection
contract, the admission-guard parser on an observed all-idle snapshot, the
pure discovery/ABI helpers, and the arm B worker preparation/measurement
flow over a mocked torch surface (byte-window fake tensors injected into
sys.modules only for the duration of the worker import; oracle PACE calls).
Everything here runs on an ordinary laptop.
"""
import importlib.util
import json
import statistics
import sys
import types
import unittest
from pathlib import Path

import plan
import launch
from supervisor import supervise

ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent
spec = importlib.util.spec_from_file_location('v3_guard', ROOT.parent / 'atlas_group_admission_device_v3' / 'launch.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
fixture = json.loads((ROOT / 'observed_npu_info_fixture.json').read_text())['npu_smi_info']

SCRIPT = ('set -euo pipefail; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
          'export PYTHONPATH=/workspace/cashflex:/workspace/barrier:${PYTHONPATH:-}; '
          'export LD_LIBRARY_PATH=/workspace/cashflex/cashflex/lib:/workspace/shmem/build_cann9_vllm/lib:${LD_LIBRARY_PATH:-}; '
          'exec python3 -B -u /workspace/barrier/supervisor.py')
# Device-measured schema string of the frozen PACE device-split operator
# (pinned image, identical precheck mounts/PYTHONPATH, rc=0); frozen as the
# ABI-pin test fixture.
PACE_SCHEMA_FIXTURE = ('cashflex::hpa_alltoallv_device_splits(Tensor _0, Tensor _1, Tensor _2, '
                       'Tensor _3, int _4, int _5, int _6, int _7) -> Tensor _0')


def container_info(devices, mounts, env, payload=None, script=SCRIPT):
    host = {'NetworkMode': 'host' if devices else 'none', 'IpcMode': 'host' if devices else 'private',
            'Devices': [{'PathOnHost': d, 'PathInContainer': d, 'CgroupPermissions': 'rwm'} for d in devices]}
    argv = ['docker', 'run', '-d', 'bash', '-lc', script]
    return {'Image': launch.IMAGE, 'HostConfig': host,
            'State': {'Running': False, 'ExitCode': 0, 'OOMKilled': False, 'Error': ''},
            'Config': {'Env': [f'{k}={v}' for k, v in env.items()], 'Cmd': argv[-3:]},
            'Mounts': [{'Destination': t, 'Source': s, 'RW': w} for t, (s, w) in mounts.items()]}


def base_env():
    shmem_port, master_port = plan.ports('B', 1)
    return {'WORLD_SIZE': '16', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(master_port),
            'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(16))),
            'HPA_SHMEM_IP_PORT': f'tcp://127.0.0.1:{shmem_port}', 'HPA_SHMEM_WORKSPACE_PLANES': '2',
            'REV3_ARM': 'B', 'REV3_GROUP_INDEX': '1',
            'REV3_ACL_LIBRARY_PATH': '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so',
            'REV3_ACL_LIBRARY_SHA256': '0' * 64, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}


def synthetic_pairs(count, start=1_000_000, gap=5_000, span=1_000):
    pairs, cursor = [], start
    for _ in range(count):
        pairs.append([cursor, cursor + span])
        cursor += gap
    return pairs


class Child:
    pid = 123

    def __init__(self, codes):
        self.codes = list(codes)

    def poll(self):
        return self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]


class SupervisorTests(unittest.TestCase):
    def env(self, arm='A', group='1', world='16'):
        return {'WORLD_SIZE': world, 'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(16))),
                'REV3_ARM': arm, 'REV3_GROUP_INDEX': group}

    def test_normal_group_spawns_all_sixteen(self):
        calls, output = [], []
        def popen(argv, env):
            calls.append((argv, env))
            return Child([None, 0])
        observed, errors = supervise(self.env(), popen, lambda x: None, output.append)
        self.assertEqual(observed, {rank: 0 for rank in range(16)})
        self.assertFalse(errors)
        self.assertEqual([call[1]['RANK'] for call in calls], [str(r) for r in range(16)])
        self.assertEqual(calls[0][0][-1], '/workspace/barrier/worker.py')

    def test_spawn_failure_still_waits_started_children(self):
        output = []
        def popen(argv, env):
            if env['RANK'] == '15':
                raise OSError('failed')
            return Child([None, None, 0])
        observed, errors = supervise(self.env(), popen, lambda x: None, output.append)
        self.assertEqual(set(observed), set(range(15)))
        self.assertEqual(errors[0]['rank'], 15)
        self.assertTrue(any('no signaling' in x for x in output))

    def test_rejects_bad_world_visibility_arm_or_group(self):
        for env in (self.env(world='2'), self.env(arm='A', group='6'), self.env(arm='D'),
                    self.env(group='0'), {**self.env(), 'ASCEND_RT_VISIBLE_DEVICES': '0,1'}):
            with self.assertRaises(ValueError):
                supervise(env, lambda *a, **k: self.fail('started'), lambda x: None)


class PlanStructureTests(unittest.TestCase):
    def test_group_order_and_ports(self):
        self.assertEqual(plan.ARM_GROUPS, {'A': 5, 'B': 2, 'C': 3})
        self.assertEqual(len(plan.GROUP_ORDER), 10)
        seen = set()
        for arm, index in plan.GROUP_ORDER:
            shmem_port, master_port = plan.ports(arm, index)
            self.assertNotIn((shmem_port, master_port), seen)
            seen.add((shmem_port, master_port))
            self.assertTrue(26401 <= shmem_port < 26501 and 26501 <= master_port < 26601)
        with self.assertRaises(ValueError):
            plan.ports('A', 6)

    def test_arm_b_round_schedule(self):
        for index in (1, 2):
            rounds = plan.arm_b_rounds(index)
            self.assertEqual(len(rounds), 106)
            self.assertEqual(sum(1 for p, _ in rounds if p == 'warmup'), 6)
            measured = [(p, s) for p, s in rounds if p == 'measure']
            self.assertEqual(len(measured), 100)
            for sub in ('b1', 'b2'):
                self.assertEqual(sum(1 for _, s in measured if s == sub), 50)
            # measured rounds come in 10-round blocks with alternating lead sub-arm
            lead = [measured[0][1], measured[20][1], measured[40][1], measured[60][1], measured[80][1]]
            self.assertEqual(len(set(lead)), 2)
            first = plan.arm_b_first_subarm(index)
            self.assertEqual(measured[0][1], first)
        self.assertNotEqual(plan.arm_b_rounds(1)[6], plan.arm_b_rounds(2)[6])

    def test_arm_c_iteration_schedule(self):
        sequence = plan.arm_c_iteration_plan()
        self.assertEqual(len(sequence), 96)
        self.assertEqual(sum(1 for p, _ in sequence if p == 'warmup'), 6)
        for k in plan.C_K_VALUES:
            self.assertEqual(sum(1 for p, kk in sequence if p == 'measure' and kk == k), 30)
        marks = plan.arm_c_checkpoints()
        self.assertEqual(len(marks), 6)
        for k in plan.C_K_VALUES:
            positions = [i for i, (_, kk) in enumerate(sequence) if kk == k]
            self.assertIn(positions[0], marks)
            self.assertIn(positions[-1], marks)
        first_block = [k for p, k in sequence[6:36]]
        self.assertEqual(first_block, [0] * 10 + [1] * 10 + [2] * 10)
        second_block = [k for p, k in sequence[36:66]]
        self.assertEqual(second_block, [2] * 10 + [1] * 10 + [0] * 10)

    def test_case038_matrix_invariants(self):
        case = plan.case038()
        self.assertEqual(case['name'], 'qwen-o1-call-038-dispatch')
        send, recv = plan.rank_splits(0)
        self.assertEqual(send, list(plan.CASE038_SEND_ROWS[0]))
        self.assertEqual(recv[0], plan.CASE038_SEND_ROWS[0][0])
        self.assertEqual(sum(case['recv_rows']), 16 * 512)
        for rank in range(16):
            send, recv = plan.rank_splits(rank)
            self.assertEqual(sum(send), 512)
            self.assertEqual(sum(recv), case['recv_rows'][rank])
            self.assertGreater(plan.arm_c_peak_tensor_bytes(rank), 0)
            self.assertEqual(plan.arm_peer_bytes('C', rank), [rows * 4096 for rows in send])

    def test_arm_b_shape_fits_verified_slot_envelope(self):
        self.assertEqual(sum(plan.arm_peer_bytes('B', 0)), 16 * 1024)
        self.assertLess(sum(plan.arm_peer_bytes('B', 0)), plan.AB_SLOT_BYTES)
        self.assertEqual(plan.arm_peer_bytes('A', 0), None)

    def test_pace_tail_backing_math(self):
        # B1 device failure 42fe5bd9: the operator requires >= 1024 explicit
        # tail bytes; tail rows follow the reverse-order-lane conversion.
        self.assertEqual(plan.PACE_TAIL_BYTES, 1024)
        self.assertEqual(plan.pace_tail_rows(1024), 1)
        self.assertEqual(plan.pace_tail_rows(4096), 1)
        self.assertEqual(plan.pace_tail_rows(300), 4)
        for width in (64, 1024, 2048, 4096):
            self.assertGreaterEqual(plan.pace_tail_rows(width) * width, 1024)
            self.assertLess(plan.pace_tail_rows(width) * width, 1024 + width)
        with self.assertRaises(ValueError):
            plan.pace_tail_rows(0)
        with self.assertRaises(ValueError):
            plan.pace_tail_rows('4096')
        # split vectors: >= 1024 tail BYTES converted to tail ELEMENTS by the
        # element width; the reverse-lane verified form is int32 + 256 canary
        # elements (256 x 4 B = 1024 B)
        self.assertEqual(plan.pace_split_tail_elements(4), 256)
        self.assertEqual(plan.pace_split_tail_elements(8), 128)
        self.assertEqual(plan.pace_split_tail_elements(2), 512)
        self.assertEqual(plan.pace_split_tail_elements(), 256)
        for width in (1, 2, 4, 8):
            self.assertGreaterEqual(plan.pace_split_tail_elements(width) * width, 1024)
            self.assertLess(plan.pace_split_tail_elements(width) * width, 1024 + width)
        with self.assertRaises(ValueError):
            plan.pace_split_tail_elements(0)
        self.assertEqual(plan.SPLIT_CANARY, 173)
        self.assertEqual(plan.SPLIT_ITEM_BYTES, 4)

    def test_segment_bytes_deterministic_and_bf16_safe(self):
        left = plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, 3, 7, 4, 4096)
        right = plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, 3, 7, 4, 4096)
        other = plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, 7, 3, 4, 4096)
        self.assertEqual(left, right)
        self.assertNotEqual(left, other)
        self.assertEqual(len(left), 4 * 4096)
        self.assertTrue(all(byte & 0x7e == byte for byte in left[1::2]))
        for rows, width in ((0, 4096), (-1, 4096), (1, 33)):
            with self.assertRaises(ValueError):
                plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, 3, 7, rows, width)
        with self.assertRaises(ValueError):
            plan.segment_bytes(plan.C_CASE_NAME, 'torch.float16', 3, 7, 1, 4096)

    def test_case038_real_geometry_regression(self):
        """C1 device failure regression: the sparse case038 matrix has 122
        zero-row segments; the CPU oracle joins must SKIP them (like the
        reverse lane's host_values), and every nonzero (rank, peer) segment
        must build cleanly and match the reverse-order lane's oracle
        byte-for-byte (same seed scheme, shake stream and BF16 mask)."""
        import importlib.util
        reverse = ANALYSIS_ROOT / 'atlas_qwen_sparse_p16_candidate_v1'
        spec = importlib.util.spec_from_file_location('reverse_pilot_contract',
                                                      reverse / 'pilot_contract.py')
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        case = pilot.selected('p16-qwen')['cases'][0]
        self.assertEqual(case['name'], plan.C_CASE_NAME)
        nonzero = 0
        rejected_skips = 0
        for rank in range(plan.WORLD):
            send, recv = plan.rank_splits(rank)
            for counts, source_of in ((send, rank), (recv, None)):
                for peer, rows in enumerate(counts):
                    if rows == 0:
                        rejected_skips += 1
                        with self.assertRaises(ValueError):
                            plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE,
                                               source_of if source_of is not None else peer,
                                               peer if source_of is not None else rank,
                                               rows, plan.C_ROW_BYTES)
                        continue
                    source = rank if counts is send else peer
                    destination = peer if counts is send else rank
                    mine = plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, source,
                                              destination, rows, plan.C_ROW_BYTES)
                    theirs = bytes(pilot.segment_bytes(case, source, destination))
                    self.assertEqual(len(mine), rows * plan.C_ROW_BYTES)
                    self.assertTrue(all(b & 0x7e == b for b in mine[1::2]))
                    nonzero += 1
        matrix_zeros = sum(1 for row in plan.CASE038_SEND_ROWS for value in row if value == 0)
        self.assertEqual(matrix_zeros, 61)                     # sparse matrix cells
        self.assertEqual(rejected_skips, 2 * matrix_zeros)     # send-side + recv-side joins
        self.assertEqual(nonzero, 2 * (16 * 16 - matrix_zeros))
        # byte-identity with the reverse lane's verified oracle, sampled
        for source, destination in ((0, 3), (3, 4), (7, 7), (15, 2), (1, 9)):
            rows = plan.CASE038_SEND_ROWS[source][destination]
            self.assertEqual(
                plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, source, destination,
                                   rows, plan.C_ROW_BYTES),
                bytes(pilot.segment_bytes(case, source, destination)))
        # full-geometry join totals equal the live-row byte counts per rank
        for rank in range(plan.WORLD):
            send, recv = plan.rank_splits(rank)
            expected_out = b''.join(plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, peer, rank,
                                                       recv[peer], plan.C_ROW_BYTES)
                                    for peer in range(plan.WORLD) if recv[peer])
            self.assertEqual(len(expected_out), sum(recv) * plan.C_ROW_BYTES)

    def test_budget_gate_fail_closed(self):
        class Helper:
            @staticmethod
            def submission_budget(peer_bytes, rank, channels):
                return [{'total_sqes': 5} for _ in channels]
        channels = [{'sq_head': 0, 'sq_tail': 0, 'reserved': 0, 'sq_depth': 1000} for _ in range(4)]
        budget = plan.plan_call_budgets('C', 1, 0, [1024] * 16, Helper, channels)
        self.assertEqual(budget['pace_calls'], 96)
        self.assertEqual(budget['cumulative_sqes'], [480] * 4)
        tight = [{'sq_head': 0, 'sq_tail': 0, 'reserved': 0, 'sq_depth': 479} for _ in range(4)]
        with self.assertRaises(ValueError):
            plan.plan_call_budgets('C', 1, 0, [1024] * 16, Helper, tight)
        dirty = [{'sq_head': 1, 'sq_tail': 0, 'reserved': 0, 'sq_depth': 1000} for _ in range(4)]
        with self.assertRaises(ValueError):
            plan.plan_call_budgets('B', 1, 0, [1024] * 16, Helper, dirty)
        empty = plan.plan_call_budgets('A', 1, 0, None, Helper, channels)
        self.assertEqual(empty['cumulative_sqes'], [0, 0, 0, 0])

    def test_discover_acl_library(self):
        path = '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so'
        text = '\n'.join(['7f00-7f10 r--p 00000000 fd:01 1234 ' + path,
                          '7f10-7f30 r-xp 00001000 fd:01 1234 ' + path,
                          '7f00-7f10 r--p 00000000 fd:01 99 /usr/lib/libother.so'])
        self.assertEqual(plan.discover_acl_library(text), path)
        with self.assertRaises(ValueError):
            plan.discover_acl_library('7f00-7f10 r--p 00000000 fd:01 1 /usr/lib/libother.so')
        with self.assertRaises(ValueError):
            plan.discover_acl_library(text + '\n7f40-7f50 r-xp 0 fd:01 5 /opt/other/libascendcl.so')

    def test_check_pace_schema(self):
        parsed = plan.check_pace_schema(PACE_SCHEMA_FIXTURE)
        self.assertEqual(parsed['name'], 'cashflex::hpa_alltoallv_device_splits')
        self.assertEqual(len(parsed['arguments']), 8)
        self.assertEqual(parsed['returns'], ['Tensor'])
        for label, bad in (
                ('namespace stripped', PACE_SCHEMA_FIXTURE.replace('cashflex::', '')),
                ('argument type', PACE_SCHEMA_FIXTURE.replace('int _6', 'str _6')),
                ('return type', PACE_SCHEMA_FIXTURE.replace('-> Tensor _0', '-> int _0')),
                ('missing argument', PACE_SCHEMA_FIXTURE.replace(', int _7', '')),
                ('multiple returns', PACE_SCHEMA_FIXTURE.replace('-> Tensor _0', '-> (Tensor _0, Tensor _1)')),
                ('other operator', PACE_SCHEMA_FIXTURE.replace('cashflex::hpa_alltoallv_device_splits',
                                                               'cashflex::other_op'))):
            with self.assertRaises(ValueError, msg=label):
                plan.check_pace_schema(bad)


class MetricTests(unittest.TestCase):
    def test_summary_matches_inclusive_quantiles(self):
        result = plan.summary([3.0, 1.0, 2.0, 5.0, 4.0])
        q1, _, q3 = statistics.quantiles([3.0, 1.0, 2.0, 5.0, 4.0], n=4, method='inclusive')
        self.assertEqual(result['median'], 3.0)
        self.assertEqual(result['q1'], q1)
        self.assertEqual(result['q3'], q3)
        self.assertEqual(result['iqr'], q3 - q1)
        self.assertEqual(result['n'], 5)
        single = plan.summary([7.5])
        self.assertEqual(single['median'], 7.5)
        self.assertEqual(single['iqr'], 0.0)
        with self.assertRaises(ValueError):
            plan.summary([])

    def by_rank(self, count, offsets):
        return {rank: [[1_000 + 10 * i, 2_000 + offsets[rank] + 10 * i]
                       for i in range(count)] for rank in offsets}

    def test_iteration_max_rank_and_envelope(self):
        offsets = {rank: (50 if rank == 15 else 0) for rank in range(plan.WORLD)}
        by_rank = self.by_rank(3, offsets)
        self.assertEqual(plan.iteration_max_rank_us(by_rank, 3), [1.05] * 3)
        self.assertEqual(plan.iteration_envelope_us(by_rank, 3), [1.05] * 3)
        with self.assertRaises(ValueError):
            plan.iteration_max_rank_us({0: [[1, 2]]}, 1)
        with self.assertRaises(ValueError):
            plan.iteration_max_rank_us({rank: [[1, 2]] for rank in range(16)}, 2)

    def arm_b_by_rank(self, index):
        rounds = []
        cursor = 0
        for phase, sub in plan.arm_b_rounds(index):
            span = 3_000 if sub == 'b1' else 1_000
            rounds.append((phase, sub, cursor, cursor + span))
            cursor += 10_000
        return {rank: list(rounds) for rank in range(plan.WORLD)}

    def test_arm_b_metrics_increment(self):
        metrics = plan.arm_b_metrics(self.arm_b_by_rank(1), 1)
        self.assertEqual(metrics['b1_median_us'], 3.0)
        self.assertEqual(metrics['b2_median_us'], 1.0)
        self.assertEqual(metrics['increment_median_us'], 2.0)
        self.assertEqual(len(metrics['block_medians_us']), plan.B_BLOCKS)
        for block in metrics['block_medians_us']:
            self.assertEqual(block['b1'] - block['b2'], 2.0)
        mutated = self.arm_b_by_rank(1)
        mutated[0][6] = ('measure', 'b2', 0, 1)
        with self.assertRaises(ValueError):
            plan.arm_b_metrics(mutated, 1)

    def test_arm_c_metrics_slopes(self):
        cursor = 0
        rows = []
        for phase, k in plan.arm_c_iteration_plan():
            span = 10_000 + 5_000 * k
            rows.append((phase, k, cursor, cursor + span))
            cursor += 100_000
        metrics = plan.arm_c_metrics({rank: list(rows) for rank in range(plan.WORLD)})
        self.assertEqual(metrics['k_median_us'], {'0': 10.0, '1': 15.0, '2': 20.0})
        self.assertEqual(metrics['slope_k1_minus_k0_us'], 5.0)
        self.assertEqual(metrics['slope_k2_minus_k0_over_2_us'], 5.0)


class ValidatorTests(unittest.TestCase):
    def arm_a_records(self):
        records = []
        for rank in range(plan.WORLD):
            records.append({'event': 'arm_a_samples', 'arm': 'A', 'group': 1, 'rank': rank,
                            'world': 16, 'warmups': plan.A_WARMUPS,
                            'samples': synthetic_pairs(plan.A_SAMPLES),
                            'clock': 'host_monotonic_ns',
                            'scope': 'ctypes_enqueue_plus_same_stream_synchronize',
                            'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256,
                            'pace_calls': 0, 'budget_cumulative_sqes': [0] * 4})
        return records

    def test_arm_a_valid_and_mutations_fail(self):
        by_rank = plan.validate_arm_a_records(self.arm_a_records(), 1)
        self.assertEqual(set(by_rank), set(range(16)))
        metrics = plan.arm_a_metrics(by_rank)
        self.assertEqual(len(metrics['max_rank_us']), plan.A_SAMPLES)
        for mutate in ('count', 'timer', 'order', 'identity', 'scope'):
            records = self.arm_a_records()
            if mutate == 'count':
                records[0]['samples'] = records[0]['samples'][:-1]
            elif mutate == 'timer':
                records[3]['samples'][2] = [500, 500]
            elif mutate == 'order':
                records[5]['samples'][2] = [1, 2]
            elif mutate == 'identity':
                records[7]['world'] = 2
            else:
                records[9]['scope'] = 'something_else'
            with self.assertRaises(ValueError):
                plan.validate_arm_a_records(records, 1)

    def arm_b_records(self, index=1):
        records, cursor = [], 0
        rows = []
        for phase, sub in plan.arm_b_rounds(index):
            span = 4_000 if sub == 'b1' else 2_000
            rows.append({'phase': phase, 'sub': sub, 'start_ns': cursor, 'end_ns': cursor + span,
                         'output_equal': True, 'input_equal': True, 'splits_equal': True,
                         'input_tail_equal': True, 'output_tail_equal': True,
                         'split_tail_equal': True, 'tail_rows': 1, 'split_tail_elements': 256})
            cursor += 10_000
        for rank in range(plan.WORLD):
            records.append({'event': 'arm_b_samples', 'arm': 'B', 'group': index, 'rank': rank,
                            'world': 16, 'clock': 'host_monotonic_ns',
                            'peer_row_bytes': plan.B_PEER_ROW_BYTES, 'pace_calls': 106,
                            'input_tail_rows': 1, 'input_tail_bytes': 1024,
                            'budget_cumulative_sqes': [530] * 4, 'rounds': [dict(row) for row in rows],
                            'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256})
        return records

    def test_arm_b_valid_and_mutations_fail(self):
        plan.validate_arm_b_records(self.arm_b_records(), 1)
        for mutate in ('flag', 'order', 'count', 'peer_bytes', 'tail_false', 'split_tail_false'):
            records = self.arm_b_records()
            if mutate == 'flag':
                records[2]['rounds'][10]['output_equal'] = False
            elif mutate == 'order':
                records[4]['rounds'][6], records[4]['rounds'][7] = records[4]['rounds'][7], records[4]['rounds'][6]
            elif mutate == 'count':
                records[6]['rounds'] = records[6]['rounds'][:-1]
            elif mutate == 'peer_bytes':
                records[8]['peer_row_bytes'] = 4096
            elif mutate == 'tail_false':
                records[9]['rounds'][3]['input_tail_equal'] = False
            else:
                records[10]['rounds'][5]['split_tail_equal'] = False
            with self.assertRaises(ValueError):
                plan.validate_arm_b_records(records, 1)

    def arm_c_records(self, index=1):
        records, cursor = [], 0
        rows = []
        for phase, k in plan.arm_c_iteration_plan():
            rows.append({'phase': phase, 'k': k, 'start_ns': cursor, 'end_ns': cursor + 9_000})
            cursor += 50_000
        marks = plan.arm_c_checkpoints()
        checks = []
        for i in sorted(marks):
            phase, k = plan.arm_c_iteration_plan()[i]
            checks.append({'index': i, 'phase': phase, 'k': k, 'output_equal': True,
                           'input_equal': True, 'splits_equal': True,
                           'input_tail_equal': True, 'output_tail_equal': True,
                           'split_tail_equal': True})
        for rank in range(plan.WORLD):
            records.append({'event': 'arm_c_samples', 'arm': 'C', 'group': index, 'rank': rank,
                            'world': 16, 'clock': 'host_monotonic_ns', 'case': plan.C_CASE_NAME,
                            'row_bytes': plan.C_ROW_BYTES, 'dtype': plan.C_DTYPE,
                            'slot_bytes': plan.C_SLOT_BYTES, 'heap_bytes': plan.C_HEAP_BYTES,
                            'scope': 'pace_full_call_plus_k_barriers_through_device_synchronize',
                            'iterations': [dict(row) for row in rows], 'checks': [dict(row) for row in checks],
                            'pace_calls': 96, 'budget_cumulative_sqes': [480] * 4,
                            'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256})
        return records

    def test_arm_c_valid_and_mutations_fail(self):
        plan.validate_arm_c_records(self.arm_c_records(), 1)
        for mutate in ('k_order', 'checkpoint_missing', 'checkpoint_false', 'checkpoint_unbound',
                       'checkpoint_tail_false', 'checkpoint_split_tail_false', 'case', 'calls'):
            records = self.arm_c_records()
            if mutate == 'k_order':
                records[1]['iterations'][6]['k'] = 2
            elif mutate == 'checkpoint_missing':
                records[2]['checks'] = records[2]['checks'][:-1]
            elif mutate == 'checkpoint_false':
                records[3]['checks'][0]['input_equal'] = False
            elif mutate == 'checkpoint_unbound':
                records[4]['checks'][1]['k'] = 2
            elif mutate == 'checkpoint_tail_false':
                records[5]['checks'][0]['output_tail_equal'] = False
            elif mutate == 'checkpoint_split_tail_false':
                records[6]['checks'][1]['split_tail_equal'] = False
            elif mutate == 'case':
                records[7]['row_bytes'] = 8192
            else:
                records[8]['pace_calls'] = 95
            with self.assertRaises(ValueError):
                plan.validate_arm_c_records(records, 1)


class LaunchContractTests(unittest.TestCase):
    def test_inspect_contract_accepts_arm_container(self):
        devices = list(guard.nodes(16))
        env = base_env()
        mounts = launch.expected_mounts(Path('/results/host/payload'))
        info = container_info(devices, mounts, env, Path('/results/host/payload'))
        argv = ['docker', 'run', launch.IMAGE, 'bash', '-lc', SCRIPT]
        launch.inspect_contract(info, argv, env, guard, 16, Path('/results/host/payload'))

    def test_inspect_contract_accepts_precheck_container(self):
        env = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
        script = ('set -euo pipefail; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
                  'export PYTHONPATH=/workspace/cashflex:/workspace/barrier:${PYTHONPATH:-}; '
                  'export LD_LIBRARY_PATH=/workspace/cashflex/cashflex/lib:/workspace/shmem/build_cann9_vllm/lib:${LD_LIBRARY_PATH:-}; '
                  'exec python3 -B -u /workspace/barrier/precheck.py')
        info = container_info([], launch.expected_mounts(), env, script=script)
        argv = ['docker', 'run', launch.IMAGE, 'bash', '-lc', script]
        launch.inspect_contract(info, argv, env)

    def test_inspect_contract_rejects_deviations(self):
        env = base_env()
        devices = list(guard.nodes(16))
        payload = Path('/results/host/payload')
        argv = ['docker', 'run', launch.IMAGE, 'bash', '-lc', SCRIPT]
        broken = container_info(devices, launch.expected_mounts(payload), env, payload)
        launch.inspect_contract(broken, argv, env, guard, 16, payload)
        cases = []
        writable = dict(broken)
        writable['Mounts'] = [dict(row, RW=True) for row in broken['Mounts']]
        cases.append(('writable mount', writable))
        extra_device = dict(broken)
        extra_device['HostConfig'] = dict(broken['HostConfig'], DeviceRequests=[{}])
        cases.append(('device request', extra_device))
        bad_env = dict(broken)
        bad_env['Config'] = dict(broken['Config'], Env=['REV3_ARM=Z'])
        cases.append(('environment', bad_env))
        running = dict(broken)
        running['State'] = dict(broken['State'], Running=True)
        cases.append(('still running', running))
        for label, info in cases:
            with self.assertRaises(RuntimeError, msg=label):
                launch.inspect_contract(info, argv, env, guard, 16, payload)


class GuardParserTests(unittest.TestCase):
    def test_all_idle_observed_table_passes(self):
        parsed = guard.parse_npu_info(fixture)
        self.assertEqual(parsed['empty_npu_cards'], list(range(8)))
        self.assertEqual(parsed['covered_die_count'], 16)

    def test_occupied_or_incomplete_tables_fail_closed(self):
        occupied = fixture.replace('No running processes found in NPU 3',
                                   '3       0                 | 4242      | intruder         | 512')
        with self.assertRaises(RuntimeError):
            guard.parse_npu_info(occupied)
        missing = '\n'.join(line for line in fixture.splitlines()
                            if 'No running processes found in NPU 7' not in line)
        with self.assertRaises(RuntimeError):
            guard.parse_npu_info(missing)
        unhealthy = fixture.replace('| 4     Ascend910           | OK', '| 4     Ascend910           | WARN', 1)
        with self.assertRaises(RuntimeError):
            guard.parse_npu_info(unhealthy)


class VerifyTests(unittest.TestCase):
    """Full launch.verify() over a synthetic lane tree: the BLOCKER test gap.

    Builds a temporary source tree (real frozen files + synthetic provider
    files + stub guard) and drives the exact verify() code path the device
    lane uses, against one contract-conforming APPROVED review and five
    non-conforming variants, plus source-tree hygiene rejections.
    """

    GUARD_CODE = ("FILES = {'launch.py'}\nRESULTS = None\n\n"
                  "def manifest(root, expected, files=None):\n"
                  "    return {}\n")

    def setUp(self):
        import hashlib
        import shutil
        import tempfile
        self.hash = hashlib.sha256
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        source = root / 'src'
        source.mkdir()
        for name in launch.FILES:
            shutil.copy(ROOT / name, source / name)
        shutil.copy(ROOT / 'source_manifest.json', source / 'source_manifest.json')
        lib = root / 'libcashflex.so'
        lib.write_bytes(b'synthetic-cashflex')
        shmem = root / 'libshmem.so'
        shmem.write_bytes(b'synthetic-shmem')
        helper = root / 'metadata.py'
        helper.write_text('# synthetic helper\n')
        layout = root / 'layout.json'
        layout.write_text('{}\n')
        guard_dir = root / 'guard'
        guard_dir.mkdir()
        (guard_dir / 'launch.py').write_text(self.GUARD_CODE)
        (guard_dir / 'source_manifest.json').write_text(
            json.dumps({'launch.py': self.hash(self.GUARD_CODE.encode()).hexdigest()}, indent=2) + '\n')
        self.source = source
        self.manifest_sha = self.hash((source / 'source_manifest.json').read_bytes()).hexdigest()
        self.env = launch.LaneEnv(
            source, lib, shmem, helper, layout, guard_dir,
            self.hash((guard_dir / 'launch.py').read_bytes()).hexdigest(),
            self.hash((guard_dir / 'source_manifest.json').read_bytes()).hexdigest(),
            self.hash(lib.read_bytes()).hexdigest(), self.hash(shmem.read_bytes()).hexdigest(),
            self.hash(helper.read_bytes()).hexdigest(), self.hash(layout.read_bytes()).hexdigest())

    def tearDown(self):
        self.tmp.cleanup()

    def compliant_review(self):
        return {'status': 'APPROVED', 'scope': plan.REVIEW_SCOPE,
                'source_manifest_sha256': self.manifest_sha,
                'reviewed_files': sorted(launch.FILES | {'source_manifest.json'}),
                'groups': {key: list(value) for key, value in plan.REVIEW_GROUPS.items()},
                'findings': [], 'notes': 'synthetic conformance fixture'}

    def install_review(self, review):
        path = self.source / 'review.json'
        path.write_text(json.dumps(review, indent=2, sort_keys=True) + '\n')
        return self.hash(path.read_bytes()).hexdigest()

    def test_conforming_review_opens_the_gate(self):
        review_sha = self.install_review(self.compliant_review())
        guard = launch.verify(self.manifest_sha, review_sha, self.env)
        self.assertTrue(hasattr(guard, 'manifest'))

    def test_non_conforming_reviews_are_rejected(self):
        variants = {}
        rejected = self.compliant_review()
        rejected['status'] = 'REJECTED'
        variants['status'] = rejected
        wrong_manifest = self.compliant_review()
        wrong_manifest['source_manifest_sha256'] = '0' * 64
        variants['manifest'] = wrong_manifest
        wrong_scope = self.compliant_review()
        wrong_scope['scope'] = 'some_other_scope'
        variants['scope'] = wrong_scope
        missing_files = self.compliant_review()
        missing_files['reviewed_files'] = sorted(set(missing_files['reviewed_files']) - {'worker.py'})
        variants['files'] = missing_files
        wrong_groups = self.compliant_review()
        wrong_groups['groups'] = {k: (v[:-1] if k == 'arm_c' else v)
                                  for k, v in wrong_groups['groups'].items()}
        variants['groups'] = wrong_groups
        for label, review in variants.items():
            review_sha = self.install_review(review)
            with self.assertRaises(ValueError, msg=label):
                launch.verify(self.manifest_sha, review_sha, self.env)

    def test_review_binding_contract_shape(self):
        self.assertEqual(plan.REVIEW_GROUPS, {'arm_a': ['A1', 'A2', 'A3', 'A4', 'A5'],
                                              'arm_b': ['B1', 'B2'],
                                              'arm_c': ['C1', 'C2', 'C3']})
        self.assertEqual(plan.REVIEW_CONTRACT['manifest_field'], 'source_manifest_sha256')
        review = self.compliant_review()
        plan.check_review_binding(review, self.manifest_sha, launch.FILES)
        duplicate = dict(review, reviewed_files=review['reviewed_files'] + ['plan.py'])
        with self.assertRaises(ValueError):
            plan.check_review_binding(duplicate, self.manifest_sha, launch.FILES)

    def test_stray_tree_entries_are_rejected(self):
        review_sha = self.install_review(self.compliant_review())
        (self.source / 'old_version.py').write_text('stale\n')
        with self.assertRaises(ValueError):
            launch.verify(self.manifest_sha, review_sha, self.env)
        (self.source / 'old_version.py').unlink()
        launch.verify(self.manifest_sha, review_sha, self.env)
        (self.source / 'scratch').mkdir()
        with self.assertRaises(ValueError):
            launch.verify(self.manifest_sha, review_sha, self.env)
        (self.source / 'scratch').rmdir()
        # __pycache__ subtrees stay permitted (the launcher itself may create them)
        cache = self.source / '__pycache__'
        cache.mkdir()
        (cache / 'plan.cpython-313.pyc').write_bytes(b'\x00')
        launch.verify(self.manifest_sha, review_sha, self.env)

    def test_tampered_provider_or_launcher_fails(self):
        review_sha = self.install_review(self.compliant_review())
        tampered = launch.LaneEnv(**{**vars(self.env), 'lib_sha': '1' * 64})
        with self.assertRaises(ValueError):
            launch.verify(self.manifest_sha, review_sha, tampered)
        drifted = launch.LaneEnv(**{**vars(self.env), 'layout_sha': '2' * 64})
        with self.assertRaises(ValueError):
            launch.verify(self.manifest_sha, review_sha, drifted)
        with open(self.source / 'launch.py', 'a') as handle:
            handle.write('# local drift\n')
        with self.assertRaises(ValueError):
            launch.verify(self.manifest_sha, review_sha, self.env)


# ------------------------------------- arm B worker flow over mock torch ---
# Round-4/5 both hid preparation defects (missing explicit tails, then the
# dropped live split values) in arm_b_measure's untested path. The mocks
# below replace the torch/torch_npu/cashflex surface with a pure-python
# byte-window implementation whose slices and rows share storage with the
# parent buffer (like device views into one backing allocation); the fake
# PACE call fabricates received rows from the CPU oracle. It is NOT a device
# simulation — the worker's own wiring is what is under test.

class FakeTensor:
    """Byte-window tensor: slices and row indexing share parent storage."""

    WIDTH = {'torch.int8': 1, 'torch.int32': 4, 'torch.uint8': 1}
    SIGNED = {'torch.int8': True, 'torch.int32': True, 'torch.uint8': False}

    def __init__(self, data, dtype, shape, base=None, window=None):
        self.dtype, self.shape = dtype, tuple(shape)
        if base is None:
            self.base, self.window = bytearray(data), (0, len(data))
        else:
            self.base, self.window = base, window

    @property
    def nbytes(self):
        return self.window[1] - self.window[0]

    @property
    def bytes(self):
        return bytes(self.base[self.window[0]:self.window[1]])

    def encode(self, value):
        return int(value).to_bytes(self.WIDTH[self.dtype], 'little',
                                   signed=self.SIGNED[self.dtype])

    def __getitem__(self, key):
        rows, item = self.shape[0], self.nbytes // self.shape[0]
        if isinstance(key, int):
            lo = self.window[0] + key * item
            return type(self)(b'', self.dtype, self.shape[1:],
                              base=self.base, window=(lo, lo + item))
        start, stop, step = key.indices(rows)
        assert step == 1
        lo = self.window[0] + start * item
        return type(self)(b'', self.dtype, (stop - start,) + self.shape[1:],
                          base=self.base, window=(lo, lo + (stop - start) * item))

    def copy_(self, other):
        assert self.shape == other.shape and self.nbytes == other.nbytes
        self.base[self.window[0]:self.window[1]] = other.bytes
        return self

    def fill_(self, value):
        pattern = self.encode(value)
        self.base[self.window[0]:self.window[1]] = \
            bytearray(pattern * (self.nbytes // len(pattern)))
        return self

    def cpu(self):
        return self

    def view(self, dtype):
        assert self.WIDTH[dtype] == self.WIDTH[self.dtype]
        return type(self)(b'', dtype, self.shape, base=self.base, window=self.window)

    def reshape(self, *shape):
        assert shape[0] * shape[1] * self.WIDTH[self.dtype] == self.nbytes
        return type(self)(b'', self.dtype, shape, base=self.base, window=self.window)

    def tolist(self):
        width = self.WIDTH[self.dtype]
        raw = self.bytes
        return [int.from_bytes(raw[i:i + width], 'little',
                               signed=self.SIGNED[self.dtype])
                for i in range(0, len(raw), width)]

    def __eq__(self, scalar):
        pattern = self.encode(scalar)
        raw = self.bytes
        return FakeEq(len(raw) % len(pattern) == 0
                      and raw == pattern * (len(raw) // len(pattern)))


class FakeEq:
    def __init__(self, matches):
        self.matches = matches

    def item(self):
        return self.matches


class FakeTorchSurface:
    """Bound methods become the fake torch module's namespace."""

    int8, int32, uint8 = 'torch.int8', 'torch.int32', 'torch.uint8'

    def __init__(self):
        self.created = []
        self.npu = types.SimpleNamespace(synchronize=lambda: None)
        self.ops = types.SimpleNamespace(cashflex=types.SimpleNamespace(
            hpa_alltoallv_device_splits=fake_pace_call))

    def full(self, shape, fill, dtype, device=None):
        width = FakeTensor.WIDTH[dtype]
        count = 1
        for dim in shape:
            count *= dim
        data = int(fill).to_bytes(width, 'little', signed=FakeTensor.SIGNED[dtype]) * count
        tensor = FakeTensor(data, dtype, tuple(shape))
        self.created.append(tensor)
        return tensor

    def frombuffer(self, buffer, dtype):
        width = FakeTensor.WIDTH[dtype]
        return FakeTensor(bytes(buffer), dtype, (len(buffer) // width,))

    def equal(self, left, right):
        return left.shape == right.shape and left.bytes == right.bytes

    def all(self, operand):
        assert isinstance(operand, FakeEq)
        return operand


PACE_CONTEXT = {}


def fake_pace_call(values, out, rd, sd, *args):
    """Oracle PACE: records the live splits presented at call time, then
    fabricates each peer's rows from the deterministic CPU oracle."""
    PACE_CONTEXT['pace_calls'] += 1
    PACE_CONTEXT['splits_at_call'].append((sd.tolist(), rd.tolist()))
    PACE_CONTEXT['shapes_at_call'].append((sd.shape, rd.shape, values.shape, out.shape))
    if PACE_CONTEXT['pollute_live_splits'] and not PACE_CONTEXT['polluted']:
        PACE_CONTEXT['polluted'] = True
        # the round-5 defect state at check time: live region canary-filled
        sd.fill_(plan.SPLIT_CANARY)
    offset = 0
    for peer, rows in enumerate(rd.tolist()):
        raw = plan.segment_bytes('arm-b-producers', plan.B_DTYPE, peer, PACE_CONTEXT['rank'], rows,
                                 plan.B_PEER_ROW_BYTES)
        out.base[out.window[0] + offset:out.window[0] + offset + len(raw)] = raw
        offset += len(raw)
    assert offset == out.nbytes
    return out


class ArmBWorkerFlowTests(unittest.TestCase):
    """arm_b_measure end-to-end over the mocked torch surface.

    Round-5 BLOCKER regression cover: the canary backing slice dropped the
    live split values (1 -> 173) with no compensating fill; without the
    explicit fill after the slice this flow fails closed at the first round.
    """

    def load_worker(self):
        surface = FakeTorchSurface()
        torch_module = types.ModuleType('torch')
        for name in ('int8', 'int32', 'uint8', 'full', 'frombuffer', 'equal', 'all'):
            setattr(torch_module, name, getattr(surface, name))
        torch_module.npu = surface.npu
        torch_module.ops = surface.ops
        dist_module = types.ModuleType('torch.distributed')
        cashflex_module = types.ModuleType('cashflex')
        ops_module = types.ModuleType('cashflex.ops')
        hpa_module = types.ModuleType('cashflex.ops.hpa_alltoallv')
        hpa_module.hpa_shmem_init = lambda *args, **kwargs: True
        hpa_module.hpa_shmem_finalize = lambda *args, **kwargs: True
        cashflex_module.ops = ops_module
        ops_module.hpa_alltoallv = hpa_module
        names = ('torch', 'torch.distributed', 'torch_npu', 'cashflex',
                 'cashflex.ops', 'cashflex.ops.hpa_alltoallv')
        fakes = {'torch': torch_module, 'torch.distributed': dist_module,
                 'torch_npu': types.ModuleType('torch_npu'),
                 'cashflex': cashflex_module, 'cashflex.ops': ops_module,
                 'cashflex.ops.hpa_alltoallv': hpa_module}
        saved = {name: sys.modules.get(name) for name in names}
        sys.modules.update(fakes)
        try:
            sys.modules.pop('worker', None)  # bind a fresh worker to these fakes
            worker = importlib.import_module('worker')
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return worker, surface

    def run_arm_b(self, rank=3, group=1, pollute=False):
        worker, surface = self.load_worker()
        PACE_CONTEXT.clear()
        PACE_CONTEXT.update({'rank': rank, 'pace_calls': 0,
                             'pollute_live_splits': pollute, 'polluted': False,
                             'splits_at_call': [], 'shapes_at_call': []})
        stream = types.SimpleNamespace(_as_parameter_=0xC0FFEE,
                                       synchronize=lambda: None)
        barrier_calls = []
        record = worker.arm_b_measure(plan.WORLD, rank, group, stream,
                                      barrier_calls.append)
        return record, surface, barrier_calls

    def test_live_splits_restored_after_canary_backing(self):
        record, surface, barrier_calls = self.run_arm_b()
        rounds = plan.arm_b_rounds(1)
        split_tail = plan.pace_split_tail_elements()
        # exactly one oracle PACE call per scheduled round
        self.assertEqual(PACE_CONTEXT['pace_calls'], len(rounds))
        # the operator must see one row per peer at every call — never the
        # canary (16 x 173 = 2768 rows against input.size(0)=16 is the
        # device-fatal split-sum the frozen TORCH_CHECK rejects)
        for sd_values, rd_values in PACE_CONTEXT['splits_at_call']:
            self.assertEqual(sd_values, [1] * plan.WORLD)
            self.assertEqual(rd_values, [1] * plan.WORLD)
        for shapes in PACE_CONTEXT['shapes_at_call']:
            self.assertEqual(shapes, ((plan.WORLD,), (plan.WORLD,),
                                      (plan.WORLD, plan.B_PEER_ROW_BYTES),
                                      (plan.WORLD, plan.B_PEER_ROW_BYTES)))
        # both int32 backings: live prefix restored to 1, canary tail untouched
        backings = [tensor for tensor in surface.created
                    if tensor.dtype == 'torch.int32'
                    and tensor.shape == (plan.WORLD + split_tail,)]
        self.assertEqual(len(backings), 2)
        for backing in backings:
            self.assertEqual(backing[:plan.WORLD].tolist(), [1] * plan.WORLD)
            self.assertEqual(backing[plan.WORLD:].tolist(),
                             [plan.SPLIT_CANARY] * split_tail)
        # round records: every fail-closed flag true, tail constants bound
        self.assertEqual(len(record['rounds']), len(rounds))
        for row in record['rounds']:
            self.assertTrue(all(row[key] for key in
                                ('output_equal', 'input_equal', 'splits_equal',
                                 'input_tail_equal', 'output_tail_equal',
                                 'split_tail_equal')))
            self.assertEqual(row['tail_rows'],
                             plan.pace_tail_rows(plan.B_PEER_ROW_BYTES))
            self.assertEqual(row['split_tail_elements'], split_tail)
        self.assertEqual(record['input_tail_bytes'], plan.PACE_TAIL_BYTES)
        self.assertEqual(record['split_tail_bytes'],
                         split_tail * plan.SPLIT_ITEM_BYTES)
        self.assertEqual(len(barrier_calls),
                         sum(1 for _, sub in rounds if sub == 'b1'))

    def test_canary_polluted_live_splits_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_arm_b(pollute=True)
        # the live fill did run (call-time splits were 1); the injected
        # canary pollution was refused by the very first round's check
        self.assertEqual(PACE_CONTEXT['pace_calls'], 1)
        self.assertEqual(PACE_CONTEXT['splits_at_call'][0][0], [1] * plan.WORLD)


# ------------------------------------- arm C worker flow over mock torch ---

class FakeBFloat16:
    """Marker dtype for the BF16 fake (2-byte items, byte-wise equality)."""
    pass


class ArmCFakeTensor(FakeTensor):
    """Adds bfloat16 support and dtype-changing views to the byte-window
    tensor: view(bfloat16) halves the element count, view(uint8) doubles it,
    exactly like torch; equality stays byte-wise (finite BF16 payloads make
    element equality == byte equality)."""

    WIDTH = dict(FakeTensor.WIDTH, **{'torch.bfloat16': 2})
    SIGNED = dict(FakeTensor.SIGNED, **{'torch.bfloat16': False})

    def encode(self, value):
        width = self.WIDTH[self.dtype]
        return int(value).to_bytes(width, 'little')

    def view(self, dtype):
        if self.WIDTH[dtype] == self.WIDTH[self.dtype]:
            shape = self.shape
        elif self.WIDTH[dtype] == 2 * self.WIDTH[self.dtype]:
            shape = (self.shape[0] * 2,) if len(self.shape) == 1 else (self.shape[0], self.shape[1] * 2)
        else:
            shape = (self.shape[0] // 2,) if len(self.shape) == 1 else (self.shape[0], self.shape[1] // 2)
        return ArmCFakeTensor(b'', dtype, shape, base=self.base, window=self.window)

    def __eq__(self, scalar):
        raw = self.bytes
        pattern = self.encode(scalar)
        return FakeEq(len(raw) % len(pattern) == 0
                      and raw == pattern * (len(raw) // len(pattern)))


class ArmCFakeTorchSurface(FakeTorchSurface):
    bfloat16 = 'torch.bfloat16'
    int32, int8, uint8 = 'torch.int32', 'torch.int8', 'torch.uint8'

    def __init__(self):
        super().__init__()
        self.compared = []

    def full(self, shape, fill, dtype, device=None):
        width = ArmCFakeTensor.WIDTH[dtype]
        count = 1
        for dim in shape:
            count *= dim
        value = int(fill) & (2 ** (8 * width) - 1)
        data = value.to_bytes(width, 'little') * count
        tensor = ArmCFakeTensor(data, dtype, tuple(shape))
        self.created.append(tensor)
        return tensor

    def frombuffer(self, buffer, dtype):
        width = ArmCFakeTensor.WIDTH[dtype]
        tensor = ArmCFakeTensor(bytes(buffer), dtype, (len(buffer) // width,))
        self.created.append(tensor)
        return tensor

    def equal(self, left, right):
        self.compared.append((left.shape, left.nbytes, right.shape, right.nbytes))
        return left.shape == right.shape and left.bytes == right.bytes

    def tensor(self, data, dtype):
        width = ArmCFakeTensor.WIDTH[dtype]
        raw = b''.join(int(value).to_bytes(width, 'little', signed=True) for value in data)
        return ArmCFakeTensor(raw, dtype, (len(data),))


PACE_C_CONTEXT = {}


def fake_pace_call_c(values, out, rd, sd, *args):
    """Oracle PACE for arm C: fabricates the rank's received rows from the
    deterministic CPU oracle into the output backing's live window."""
    PACE_C_CONTEXT['pace_calls'] += 1
    rank = PACE_C_CONTEXT['rank']
    _, recv = plan.rank_splits(rank)
    offset = 0
    for peer, rows in enumerate(recv):
        if rows:
            raw = plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, peer, rank,
                                     rows, plan.C_ROW_BYTES)
            out.base[out.window[0] + offset:out.window[0] + offset + len(raw)] = raw
            offset += len(raw)
    assert offset == out.nbytes
    return out


class ArmCWorkerFlowTests(unittest.TestCase):
    """arm_c_measure end-to-end over the mocked torch surface (reduced
    schedule: one k, 2 warmups + 3 measured — the plan's checkpoint rows are
    exercised because first/last iteration of the k arm are checkpoints).

    Round-8 BLOCKER regression cover: the checkpoint comparison built the
    expected tensor with a uint8-direct reshape(rows, elements) — only half
    the oracle bytes, shape-invalid on device; the flow must build it with
    frombuffer(uint8).view(bfloat16).reshape(rows, elements) and compare
    equal against a byte-identical oracle replay."""

    def patched_plan(self):
        """Reduce the arm-C schedule to 1 k value x (2 warmup + 3 measured)."""
        saved = {name: getattr(plan, name) for name in
                 ('C_K_VALUES', 'C_WARMUPS_PER_K', 'C_MEASURED_PER_K',
                  'C_BLOCK_ROUNDS', 'arm_c_iteration_plan', 'arm_c_checkpoints')}
        plan.C_K_VALUES = (0,)
        plan.C_WARMUPS_PER_K = 2
        plan.C_MEASURED_PER_K = 3
        plan.C_BLOCK_ROUNDS = 3
        original_plan = plan.arm_c_iteration_plan
        original_marks = plan.arm_c_checkpoints

        def reduced_plan():
            sequence = []
            for k in plan.C_K_VALUES:
                sequence += [('warmup', k)] * plan.C_WARMUPS_PER_K
            for block in range(max(1, plan.C_MEASURED_PER_K // plan.C_BLOCK_ROUNDS)):
                for k in plan.C_K_VALUES:
                    sequence += [('measure', k)] * plan.C_BLOCK_ROUNDS
            return tuple(sequence)

        def reduced_checkpoints():
            sequence = reduced_plan()
            marks = set()
            for k in plan.C_K_VALUES:
                positions = [i for i, (_, kk) in enumerate(sequence) if kk == k]
                marks.add(positions[0])
                marks.add(positions[-1])
            return marks

        plan.arm_c_iteration_plan = reduced_plan
        plan.arm_c_checkpoints = reduced_checkpoints
        return saved

    def restore_plan(self, saved):
        for name, value in saved.items():
            setattr(plan, name, value)

    def load_worker(self):
        surface = ArmCFakeTorchSurface()
        surface.ops = types.SimpleNamespace(cashflex=types.SimpleNamespace(
            hpa_alltoallv_device_splits=fake_pace_call_c))
        torch_module = types.ModuleType('torch')
        for name in ('int8', 'int32', 'uint8', 'bfloat16', 'full', 'frombuffer',
                     'equal', 'all', 'tensor'):
            setattr(torch_module, name, getattr(surface, name))
        torch_module.npu = surface.npu
        torch_module.ops = surface.ops
        cashflex_module = types.ModuleType('cashflex')
        ops_module = types.ModuleType('cashflex.ops')
        hpa_module = types.ModuleType('cashflex.ops.hpa_alltoallv')
        hpa_module.hpa_shmem_init = lambda *args, **kwargs: True
        hpa_module.hpa_shmem_finalize = lambda *args, **kwargs: True
        cashflex_module.ops = ops_module
        ops_module.hpa_alltoallv = hpa_module
        names = ('torch', 'torch.distributed', 'torch_npu', 'cashflex',
                 'cashflex.ops', 'cashflex.ops.hpa_alltoallv')
        fakes = {'torch': torch_module, 'torch.distributed': types.ModuleType('torch.distributed'),
                 'torch_npu': types.ModuleType('torch_npu'),
                 'cashflex': cashflex_module, 'cashflex.ops': ops_module,
                 'cashflex.ops.hpa_alltoallv': hpa_module}
        saved_modules = {name: sys.modules.get(name) for name in names}
        sys.modules.update(fakes)
        try:
            sys.modules.pop('worker', None)
            worker = importlib.import_module('worker')
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return worker, surface

    def run_arm_c(self, rank=7, group=1, monkeypatched_reshape=None):
        saved_plan = self.patched_plan()
        worker, surface = self.load_worker()
        PACE_C_CONTEXT.clear()
        PACE_C_CONTEXT.update({'rank': rank, 'pace_calls': 0})
        stream = types.SimpleNamespace(_as_parameter_=0xC0FFEE,
                                       synchronize=lambda: None)
        barrier_calls = []
        if monkeypatched_reshape is not None:
            original = worker.torch if hasattr(worker, 'torch') else None
        try:
            record = worker.arm_c_measure(plan.WORLD, rank, group, stream,
                                          barrier_calls.append)
        finally:
            self.restore_plan(saved_plan)
        return record, surface, barrier_calls

    def test_arm_c_checkpoint_expected_shapes_and_oracle_replay(self):
        record, surface, barrier_calls = self.run_arm_c()
        rank = 7
        send, recv = plan.rank_splits(rank)
        rows_in, rows_out = sum(send), sum(recv)
        elements = plan.C_ROW_BYTES // 2
        # the oracle replay: byte-identical expected buffers
        expected_out = b''.join(plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, peer, rank,
                                                   recv[peer], plan.C_ROW_BYTES)
                                for peer in range(plan.WORLD) if recv[peer])
        expected_values = b''.join(plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, rank, peer,
                                                      send[peer], plan.C_ROW_BYTES)
                                   for peer in range(plan.WORLD) if send[peer])
        self.assertEqual(len(expected_out), rows_out * plan.C_ROW_BYTES)
        self.assertEqual(len(expected_values), rows_in * plan.C_ROW_BYTES)
        # the comparison tensors recorded by the worker must exist with the
        # right shape and full oracle byte length (unit regression: a uint8
        # reshape(rows, 2048) yields half the bytes and fails the flow)
        # the checkpoint torch.equal calls compared BOTH sides at full oracle
        # byte length with identical (rows, elements) shapes — the unit
        # regression: a uint8 reshape(rows, 2048) would compare half bytes
        seen_shapes = {(shape, nbytes) for _ls, _ln, shape, nbytes in surface.compared}
        self.assertIn(((rows_out, elements), rows_out * plan.C_ROW_BYTES), seen_shapes)
        self.assertIn(((rows_in, elements), rows_in * plan.C_ROW_BYTES), seen_shapes)
        for _ls, ln, _rs, rn in surface.compared:
            self.assertEqual(ln % plan.C_ROW_BYTES, 0)
            self.assertEqual(rn % plan.C_ROW_BYTES, 0)
        # the fake PACE replay wrote oracle bytes into the output backing,
        # and every checkpoint row passed all six fail-closed checks
        self.assertGreaterEqual(PACE_C_CONTEXT['pace_calls'], 5)
        for row in record['checks']:
            self.assertTrue(all(row[key] for key in
                                ('output_equal', 'input_equal', 'splits_equal',
                                 'input_tail_equal', 'output_tail_equal',
                                 'split_tail_equal')))
        # k=0 reduced plan: no barriers appended, 5 iterations recorded
        self.assertEqual(barrier_calls, [])
        self.assertEqual(len(record['iterations']), 5)
        # the recorded iterations exactly match the reduced plan structure
        self.assertEqual([row['phase'] for row in record['iterations']],
                         ['warmup', 'warmup', 'measure', 'measure', 'measure'])

    def test_arm_c_uint8_direct_reshape_variant_is_shape_invalid(self):
        """The round-8 defect form (uint8-direct reshape with element
        columns) must be structurally rejected: it needs half the oracle
        bytes, so the fake surface's reshape assertion fails closed."""
        rank = 7
        _, recv = plan.rank_splits(rank)
        rows_out, elements = sum(recv), plan.C_ROW_BYTES // 2
        expected_out = b''.join(plan.segment_bytes(plan.C_CASE_NAME, plan.C_DTYPE, peer, rank,
                                                   recv[peer], plan.C_ROW_BYTES)
                                for peer in range(plan.WORLD) if recv[peer])
        surface = ArmCFakeTorchSurface()
        broken = surface.frombuffer(bytearray(expected_out), surface.uint8)
        with self.assertRaises(AssertionError):
            broken.reshape(rows_out, elements)  # exactly the device failure
        good = surface.frombuffer(bytearray(expected_out), surface.uint8) \
            .view(surface.bfloat16).reshape(rows_out, elements)
        self.assertEqual(good.shape, (rows_out, elements))
        self.assertEqual(good.nbytes, rows_out * plan.C_ROW_BYTES)


if __name__ == '__main__':
    unittest.main()
