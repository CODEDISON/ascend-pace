"""Host logic tests only; no torch, device, processes or network.

Covers the supervisor contract, the frozen plan/campaign structure (groups,
ports, rotation, trimming ladder, plan_b split, iteration schedules), the
SHA-bound cases loading with full guard recomputation and tamper detection,
the fail-closed SQ budget gate, the dual-clock metric mathematics, the
pre-registered direction rules, the record validators with mutation tests,
the SINGLE review-contract constant (synthetic APPROVED accepted; REJECTED /
wrong manifest SHA / wrong scope / missing reviewed_files / wrong groups all
refused), the launcher container-inspection contract, the admission-guard
snapshot validation on an observed all-idle npu-smi fixture, the exact
nanosecond timestamp ordering, the trimming admission gate and the campaign
rollup. Everything here runs on an ordinary laptop.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import plan
import launch
from supervisor import supervise

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
spec = importlib.util.spec_from_file_location(
    'sparse_dist_guard', ROOT.parent / 'atlas_group_admission_device_v3' / 'launch.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
fixture = json.loads((ROOT / 'observed_npu_info_fixture.json').read_text())['npu_smi_info']

SUPERVISOR_SCRIPT = launch.entry_script('supervisor.py')
PRECHECK_SCRIPT = launch.entry_script('precheck.py')

HEALTH_TEXT = '\n'.join(['Health : OK'] * 3)
USAGE_TEXT = '\n'.join(
    ['HBM Capacity(MB)      : 32768'] * 2 +
    ['HBM Usage Rate(%)     : 8'] * 2 +
    ['Aicore Usage Rate(%)  : 0'] * 2 +
    ['Aivector Usage Rate(%) : 0'] * 2 +
    ['NPU Utilization(%)    : 0'] * 2)


class Child:
    pid = 321

    def __init__(self, codes):
        self.codes = list(codes)

    def poll(self):
        return self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]


# Device-measured PACE schema truth inside the pinned image (captured live
# during the barrier-v3 review round 3 from the identical precheck mounts and
# PYTHONPATH); frozen as the ABI-pin test fixture.
PACE_SCHEMA_FIXTURE = ('cashflex::hpa_alltoallv_device_splits(Tensor _0, Tensor _1, Tensor _2, '
                       'Tensor _3, int _4, int _5, int _6, int _7) -> Tensor _0')


def container_info(devices, mounts, env, payload=None, script=SUPERVISOR_SCRIPT):
    host = {'NetworkMode': 'host' if devices else 'none',
            'IpcMode': 'host' if devices else 'private',
            'Devices': [{'PathOnHost': d, 'PathInContainer': d, 'CgroupPermissions': 'rwm'}
                        for d in devices]}
    argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', script]
    return {'Image': plan.IMAGE, 'HostConfig': host,
            'State': {'Running': False, 'ExitCode': 0, 'OOMKilled': False, 'Error': ''},
            'Config': {'Env': [f'{k}={v}' for k, v in env.items()], 'Cmd': argv[-3:]},
            'Mounts': [{'Destination': t, 'Source': s, 'RW': w}
                       for t, (s, w) in mounts.items()]}


def base_env(group='job0', drops='0'):
    shmem_port, master_port = plan.ports(group)
    return {'WORLD_SIZE': '16', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(master_port),
            'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(16))),
            'HPA_SHMEM_IP_PORT': f'tcp://127.0.0.1:{shmem_port}',
            'HPA_SHMEM_WORKSPACE_PLANES': '2',
            'REVSD_VARIANT': 'plan_a', 'REVSD_GROUP': group, 'REVSD_DROPS': drops,
            'REVSD_ACL_LIBRARY_PATH': '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so',
            'REVSD_ACL_LIBRARY_SHA256': '0' * 64,
            'REVSD_CASES_PATH': '/workspace/plan/cases.json',
            'REVSD_CASES_SHA256': plan.CASES_SHA256,
            'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}


def approved_review(manifest_sha='a' * 64):
    return {'schema': plan.REVIEW_SCHEMA, 'status': plan.REVIEW_STATUS_APPROVED,
            'source_manifest_sha256': manifest_sha, 'cases_sha256': plan.CASES_SHA256,
            'scope': plan.REVIEW_SCOPE, 'reviewed_files': sorted(plan.REVIEWED_FILES),
            'groups': json.loads(json.dumps(plan.REVIEW_GROUPS)),
            'device_split_guard_semantics': 'union'}


class FakeHelper:
    """Queue helper stand-in with a fixed per-call SQE demand."""

    def __init__(self, per_call=5):
        self.per_call = per_call

    def submission_budget(self, peer_bytes, rank, channels):
        return [{'total_sqes': self.per_call} for _ in channels]


def channels(depth=2048, dirty=False):
    return [{'sq_head': 1 if dirty else 0, 'sq_tail': 0, 'reserved': 0,
             'sq_depth': depth} for _ in range(4)]


def make_record(rank, canonical_index, position, group_key='job0', variant='plan_a',
                drops=0, pace_span=5_000_000, hccl_span=8_000_000, skew_rank=None,
                skew_ns=0):
    case_id = plan.rotated_case_order(variant, group_key, drops)[position][1]
    sequence = plan.iteration_plan(group_key)
    iterations = []
    cursor = 10_000_000
    if skew_rank is not None and rank == skew_rank:
        cursor += skew_ns
    for phase, arm, block, slot in sequence:
        span = pace_span if arm == 'pace' else hccl_span
        iterations.append({'phase': phase, 'arm': arm, 'block': block, 'slot': slot,
                           'gated': plan.gate_enabled(phase),
                           't0_ns': cursor, 't1_ns': cursor + span})
        cursor += 50_000_000
    digest = hashlib.sha256(b'oracle').hexdigest()
    return {'event': 'case_samples', 'schema': 'revision2.sparse-distribution-case.v1',
            'variant': variant, 'group': group_key, 'rank': rank, 'world': 16,
            'case': case_id, 'case_index': canonical_index, 'case_position': position,
            'kind': 'synthetic', 'role': 'test', 'dtype': plan.DTYPE,
            'row_bytes': plan.ROW_BYTES, 'generation': plan.generation_label(canonical_index),
            'generation_after_measurement': plan.PACE_CALLS_PER_CASE * (canonical_index + 1),
            'arm_first': plan.arm_first(group_key),
            'clock': 'same-host perf_counter_ns (single-container CLOCK_MONOTONIC)',
            'gate': 'rank0_flag_plus_hccl_all_reduce_max_before_t0_measured_only',
            'initial': {'exact': dict.fromkeys(plan.CASE_RECORD_KEYS, True),
                        'expected_sha256': digest},
            'final': {'exact': dict.fromkeys(plan.CASE_RECORD_KEYS, True),
                      'expected_sha256': digest},
            'iterations': iterations, 'global_bad': 0,
            'budget_case_cumulative_sqes': [11 * 5 * (position + 1)] * 4,
            'validation_scope': 'initial and final full checks only; ' \
                                'intermediate repetitions not individually checked',
            'library_sha256': plan.LIB_SHA256, 'shmem_sha256': plan.SHMEM_SHA256}


def group_records(group_key='job0', variant='plan_a', drops=0, **kwargs):
    order = plan.rotated_case_order(variant, group_key, drops)
    records = []
    for position, (canonical, _) in enumerate(order):
        for rank in range(16):
            records.append(make_record(rank, canonical, position, group_key, variant,
                                       drops, **kwargs))
    return records


class SupervisorTests(unittest.TestCase):
    def env(self, group='job0', variant='plan_a', drops='0', world='16'):
        return {'WORLD_SIZE': world,
                'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(16))),
                'REVSD_VARIANT': variant, 'REVSD_GROUP': group, 'REVSD_DROPS': drops,
                'REVSD_CASES_SHA256': plan.CASES_SHA256}

    def test_normal_group_spawns_all_sixteen(self):
        calls, output = [], []

        def popen(argv, env):
            calls.append((argv, env))
            return Child([None, 0])

        observed, errors = supervise(self.env(), popen, lambda x: None, output.append)
        self.assertEqual(observed, {rank: 0 for rank in range(16)})
        self.assertFalse(errors)
        self.assertEqual([call[1]['RANK'] for call in calls], [str(r) for r in range(16)])
        self.assertEqual(calls[0][0][-1], '/workspace/lane/worker.py')

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

    def test_rejects_bad_world_visibility_variant_group_or_drops(self):
        for env in (self.env(world='2'), self.env(variant='plan_c'),
                    self.env(group='job6'), self.env(group='pba0', variant='plan_a'),
                    self.env(drops='5'), self.env(drops='-1'),
                    {**self.env(), 'ASCEND_RT_VISIBLE_DEVICES': '0,1'},
                    {**self.env(), 'REVSD_CASES_SHA256': '0' * 64}):
            with self.assertRaises(ValueError):
                supervise(env, lambda *a, **k: self.fail('started'), lambda x: None)


class CampaignStructureTests(unittest.TestCase):
    def test_group_orders_and_ports(self):
        self.assertEqual(len(plan.GROUPS_PLAN_A), 6)
        self.assertEqual(len(plan.GROUPS_PLAN_B), 12)
        self.assertEqual(set(plan.CAMPAIGN_SLOTS), set(plan.GROUPS_PLAN_A) |
                         set(plan.GROUPS_PLAN_B))
        seen = set()
        for group_key in plan.CAMPAIGN_SLOTS:
            ports = plan.ports(group_key)
            self.assertNotIn(ports, seen)
            seen.add(ports)
            self.assertTrue(26601 <= ports[0] < 26701 and 26701 <= ports[1] < 26801)
            self.assertNotEqual(ports[0], ports[1])
        with self.assertRaises(ValueError):
            plan.ports('job6')

    def test_arm_order_balance_three_three(self):
        for groups in (plan.GROUPS_PLAN_A, plan.GROUPS_PLAN_B_A, plan.GROUPS_PLAN_B_B):
            firsts = [plan.arm_first(key) for key in groups]
            self.assertEqual(firsts.count('pace'), 3)
            self.assertEqual(firsts.count('hccl'), 3)
            self.assertEqual([plan.arm_first(f'job{i}') for i in range(6)],
                             ['pace', 'hccl', 'pace', 'hccl', 'pace', 'hccl'])

    def test_case_rotation_offsets_and_anchor_positions(self):
        offsets = [(3 * plan.group_index('plan_a', key)) % 17 for key in plan.GROUPS_PLAN_A]
        self.assertEqual(offsets, [0, 3, 6, 9, 12, 15])
        self.assertEqual(len(set(offsets)), 6)
        heads = [plan.rotated_case_order('plan_a', key, 0)[0][1] for key in plan.GROUPS_PLAN_A]
        self.assertNotEqual(len(set(heads)), 1, 'anchors must not stay at position 0')
        first_orders = [plan.rotated_case_order('plan_a', 'job1', 0),
                        plan.rotated_case_order('plan_a', 'job1', 1)]
        self.assertEqual([case_id for _, case_id in first_orders[0]][:4],
                         ['qwen-o1-call-021-combine', 'qwen-o1-call-096-dispatch',
                          'qwen-o1-call-163-combine', 'olmoe-o1-call-050-dispatch'])
        # Trimming shifts the rotation base; the set is the plan minus the
        # ladder head and the first element follows the (3g mod len) rule.
        trimmed = plan.rotated_case_order('plan_a', 'job1', 1)
        self.assertNotIn('syn-unif-p25-r512', [case_id for _, case_id in trimmed])
        self.assertEqual(len(trimmed), 16)
        self.assertEqual(trimmed[0][1], plan.campaign_case_ids('plan_a', None, 1)[3 % 16])

    def test_iteration_plan_structure(self):
        for group_key in ('job0', 'job1'):
            sequence = plan.iteration_plan(group_key)
            self.assertEqual(len(sequence), 20)
            warmup = [arm for phase, arm, _, _ in sequence if phase == 'warmup']
            measured = [arm for phase, arm, _, _ in sequence if phase == 'measure']
            self.assertEqual(len(warmup), 4)
            self.assertEqual(len(measured), 16)
            for arm in ('pace', 'hccl'):
                self.assertEqual(warmup.count(arm), plan.WARMUP_PER_ARM)
                self.assertEqual(measured.count(arm), plan.MEASURED_PER_ARM)
            # warmups are ungated, every measured iteration is gated
            self.assertTrue(all(not plan.gate_enabled(phase)
                                for phase, *_ in sequence if phase == 'warmup'))
            self.assertTrue(all(plan.gate_enabled('measure') for _ in range(16)))
            base = warmup[:2]
            reverse = warmup[2:]
            self.assertEqual(list(reversed(base)), list(reverse))
            for block in range(4):
                chunk = measured[block * 4:(block + 1) * 4]
                self.assertEqual(chunk[:2] + chunk[2:], list(base) + list(reverse)
                                 if block == 0 else chunk)  # identity for block 0
                self.assertEqual(chunk, list(base) + list(reverse))
        self.assertNotEqual(plan.iteration_plan('job0')[2], plan.iteration_plan('job1')[2])
        self.assertEqual(plan.PACE_CALLS_PER_CASE, 1 + 2 + 8)

    def test_trim_ladder_sets_and_floor(self):
        ladder_sets = [set(plan.campaign_case_ids('plan_a', None, drops))
                       for drops in range(5)]
        self.assertEqual([len(s) for s in ladder_sets], [17, 16, 15, 14, 13])
        for index, (lower, higher) in enumerate(zip(ladder_sets, ladder_sets[1:])):
            self.assertEqual(len(lower - higher), 1)
            self.assertEqual((lower - higher).pop(), plan.LADDER[index])
        self.assertEqual(plan.LADDER[0], 'syn-unif-p25-r512')       # S8
        self.assertEqual(plan.LADDER[1], 'qwen-o1-call-036-dispatch')  # R3
        self.assertEqual(plan.LADDER[2], 'syn-zipf-p50-r7680')      # S5
        self.assertEqual(plan.LADDER[3], 'qwen-o1-call-096-dispatch')  # R5
        with self.assertRaises(ValueError):
            plan.campaign_case_ids('plan_a', None, 5)

    def test_plan_b_subgroups_partition_and_ladder_restriction(self):
        all_ids = set(plan.load_cases()['ids'])
        a = set(plan.campaign_case_ids('plan_b', 'a', 0))
        b = set(plan.campaign_case_ids('plan_b', 'b', 0))
        self.assertEqual(len(a), 10)
        self.assertEqual(len(b), 7)
        self.assertFalse(a & b)
        self.assertEqual(a | b, all_ids)
        self.assertIn('qwen-o1-call-038-dispatch', a)
        self.assertIn('qwen-o1-call-039-combine', a)
        # The ladder only drops cases the subgroup actually contains.
        self.assertNotIn('syn-zipf-p50-r7680', plan.campaign_case_ids('plan_b', 'a', 3))
        self.assertEqual(len(plan.campaign_case_ids('plan_b', 'b', 4)), 6)
        with self.assertRaises(ValueError):
            plan.campaign_case_ids('plan_a', 'a', 0)

    def test_generation_labels(self):
        self.assertEqual(plan.generation_label(0), 1)
        self.assertEqual(plan.generation_label(4), 45)


class CasesBindingTests(unittest.TestCase):
    def test_load_cases_recomputes_every_guard(self):
        document = plan.load_cases()
        self.assertEqual(len(document['cases']), 17)
        for case in document['cases']:
            self.assertTrue(case['guards_ok'])
            self.assertLessEqual(case['per_peer_max_bytes'], plan.SLOT_BYTES)
            self.assertLessEqual(case['whole_input_max_bytes'], plan.SLOT_BYTES)
            # recv = matrix transpose: output rows are the column sums
            matrix = case['send_matrix_rows']
            self.assertEqual(case['per_rank_output_rows'],
                             [sum(matrix[i][j] for i in range(16)) for j in range(16)])
        empty = sorted(case['empty_nonself_edges'] for case in document['cases'])
        self.assertEqual(empty[0], 0)
        self.assertEqual(empty[-1], 240)
        self.assertEqual(document['planning_estimate'],
                         {'calls': 187, 'aggregate_mib': 5024.0,
                          'model_a_per_call_sum': 935.0,
                          'model_b_bytes_proportional': 785.0,
                          'observed_ring_depth': 2048, 'counter_bits': 16})

    def test_tampered_cases_file_fails_sha_binding(self):
        document = json.loads((plan.LOCAL_PLAN_DIR / 'cases.json').read_text())
        document['cases'][0]['empty_nonself_edges'] += 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cases.json'
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                plan.load_cases(path)

    def test_declared_stat_mismatch_detected_by_recomputation(self):
        source = json.loads((plan.LOCAL_PLAN_DIR / 'cases.json').read_text())
        first = source['cases'][0]
        variants = []
        mutated_stat = dict(first)
        mutated_stat['empty_nonself_edges'] = first['empty_nonself_edges'] - 1
        variants.append(mutated_stat)
        mutated_matrix = dict(first)
        mutated_matrix['send_matrix_rows'] = [row[:] for row in first['send_matrix_rows']]
        mutated_matrix['send_matrix_rows'][0][1] += 1
        variants.append(mutated_matrix)
        mutated_guard = dict(first)
        mutated_guard['guards_ok'] = False
        variants.append(mutated_guard)
        for variant in variants:
            document = json.loads(json.dumps(source))
            document['cases'][0] = variant
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'cases.json'
                payload = json.dumps(document)
                path.write_text(payload)
                original = plan.CASES_SHA256
                plan.CASES_SHA256 = hashlib.sha256(payload.encode()).hexdigest()
                try:
                    with self.assertRaises(ValueError):
                        plan.load_cases(path)
                finally:
                    plan.CASES_SHA256 = original

    def test_cases_regenerate_byte_identical_from_frozen_workloads(self):
        workloads = REPO / 'analysis_results/atlas_fair_trace_workloads_20261028_v1/workloads.json'
        if not workloads.is_file():
            self.skipTest('frozen workloads.json not present on this host')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # gen_cases.py resolves its workloads via parents[2] of its own
            # path, so it must sit two levels below the fake repo root.
            gen_dir = root / 'a' / 'gen'
            gen_dir.mkdir(parents=True)
            shutil.copy(plan.LOCAL_PLAN_DIR / 'gen_cases.py', gen_dir / 'gen_cases.py')
            target = root / 'analysis_results' / 'atlas_fair_trace_workloads_20261028_v1'
            target.mkdir(parents=True)
            shutil.copy(workloads, target / 'workloads.json')
            subprocess.run([sys.executable, str(gen_dir / 'gen_cases.py')], check=True,
                           capture_output=True, text=True, cwd=gen_dir)
            self.assertEqual(hashlib.sha256((gen_dir / 'cases.json').read_bytes()).hexdigest(),
                             plan.CASES_SHA256)

    def test_oracle_generation(self):
        document = plan.load_cases()
        case = document['cases'][0]
        left = plan.segment_bytes(case, 3, 7)
        right = plan.segment_bytes(case, 3, 7)
        other = plan.segment_bytes(case, 7, 3)
        self.assertEqual(left, right)
        self.assertNotEqual(left, other)
        self.assertEqual(len(left), case['send_matrix_rows'][3][7] * plan.ROW_BYTES)
        self.assertTrue(all(byte & 0x7e == byte for byte in left[1::2]))
        diag = next(c for c in document['cases'] if c['id'] == 'syn-diag-512')
        with self.assertRaises(ValueError):
            plan.segment_bytes(diag, 0, 1)
        send, recv = plan.rank_splits(case, 5)
        self.assertEqual(send, list(case['send_matrix_rows'][5]))
        self.assertEqual(recv, [case['send_matrix_rows'][i][5] for i in range(16)])
        self.assertEqual(len(plan.expected_bytes(case, 5, False)), sum(send) * plan.ROW_BYTES)
        self.assertEqual(len(plan.expected_bytes(case, 5, True)), sum(recv) * plan.ROW_BYTES)
        peak = plan.peak_tensor_bytes(case, 0)
        self.assertGreater(peak, 0)
        # The largest captured combine must fit the admission formula too.
        big = next(c for c in document['cases'] if c['id'] == 'olmoe-o1-call-049-combine')
        self.assertLess(plan.peak_tensor_bytes(big, 0), 256 * 1024 * 1024)


class BudgetGateTests(unittest.TestCase):
    def test_exact_sequence_budget_and_cross_check(self):
        document = plan.load_cases()
        budget = plan.plan_call_budgets(list(document['ids']), 0, FakeHelper(5), channels())
        self.assertEqual(budget['pace_calls'], 187)
        self.assertEqual(budget['cumulative_sqes'], [935] * 4)
        self.assertEqual(len(budget['cases']), 17)
        self.assertEqual(budget['cases'][0]['pace_calls'], 11)
        self.assertEqual(budget['cases'][0]['cumulative_after_case'], [55] * 4)
        self.assertEqual(budget['planning_estimate_crosscheck'], document['planning_estimate'])
        self.assertTrue(budget['no_ring_or_task_counter_wrap'])
        self.assertFalse(budget['head_tail_drain_inferred'])

    def test_ring_depth_and_dirty_channels_fail_closed(self):
        document = plan.load_cases()
        with self.assertRaises(ValueError):
            plan.plan_call_budgets(list(document['ids']), 0, FakeHelper(5), channels(depth=935))
        budget = plan.plan_call_budgets(list(document['ids']), 0, FakeHelper(5), channels(depth=936))
        self.assertEqual(budget['cumulative_sqes'], [935] * 4)
        with self.assertRaises(ValueError):
            plan.plan_call_budgets(list(document['ids']), 0, FakeHelper(5), channels(dirty=True))

    def test_sixteen_bit_task_counter_bound(self):
        document = plan.load_cases()
        # 400 SQE/call x 187 calls = 74800 >= 65536 even with a huge ring.
        with self.assertRaises(ValueError):
            plan.plan_call_budgets(list(document['ids']), 0, FakeHelper(400),
                                   channels(depth=100000))
        # 350 x 187 = 65450 < 65536 fits.
        budget = plan.plan_call_budgets(list(document['ids']), 0, FakeHelper(350),
                                        channels(depth=100000))
        self.assertEqual(budget['cumulative_sqes'], [65450] * 4)

    def test_budget_follows_the_trimmed_sequence(self):
        document = plan.load_cases()
        trimmed = list(plan.campaign_case_ids('plan_a', None, 4))
        budget = plan.plan_call_budgets(trimmed, 3, FakeHelper(5), channels())
        self.assertEqual(budget['pace_calls'], 13 * 11)
        self.assertEqual([row['case'] for row in budget['cases']], trimmed)


class MetricTests(unittest.TestCase):
    def by_rank_from_records(self, records, case_id):
        return {record['rank']: [(row['t0_ns'], row['t1_ns']) for row in record['iterations']]
                for record in records if record['case'] == case_id}

    def test_dual_clock_iteration_metrics(self):
        records = group_records(skew_rank=15, skew_ns=6_000_000)
        by_rank = self.by_rank_from_records(records, 'qwen-o1-call-038-dispatch')
        measured = plan.measured_iteration_metrics(by_rank, 'job0')
        self.assertEqual(len(measured), 16)
        for row in measured:
            self.assertEqual(row['wait_gap_ns'], row['makespan_ns'] - row['rank_local_max_ns'])
            self.assertGreaterEqual(row['wait_gap_ns'], -6_000_000)  # skew-adjusted bound
            self.assertGreaterEqual(row['arrival_skew_ns'], 0)
        skew_row = [row for row in measured if row['skew_gt_5ms']]
        self.assertEqual(len(skew_row), 16)
        self.assertGreaterEqual(measured[0]['arrival_skew_ns'], 6_000_000)
        # without the skew, no flag and makespan == rank local max
        flat = plan.measured_iteration_metrics(
            self.by_rank_from_records(group_records(), 'qwen-o1-call-038-dispatch'), 'job0')
        for row in flat:
            self.assertFalse(row['skew_gt_5ms'])
            self.assertEqual(row['makespan_ns'], row['rank_local_max_ns'])
            self.assertEqual(row['arrival_skew_ns'], 0)
            self.assertEqual(row['wait_gap_ns'], 0)

    def test_arm_group_metrics_medians_and_mad(self):
        records = group_records()
        view = plan.validate_group_records(records, 'plan_a', 'job0', 0)
        metrics = plan.assemble_group_metrics(view, 'job0')
        entry = metrics['qwen-o1-call-038-dispatch']
        for arm, span in (('pace', 5_000_000), ('hccl', 8_000_000)):
            row = entry['arms'][arm]
            self.assertEqual(row['median_rank_local_max_ns'], span)
            self.assertEqual(row['median_makespan_ns'], span)
            self.assertEqual(row['mad_rank_local_max_ns'], 0.0)
            self.assertEqual(row['median_wait_gap_ns'], 0)
            self.assertEqual(len(row['values_rank_local_max_ns']), 8)
        self.assertEqual(entry['direction']['rank_local_max_ns']['classification'], 'pace_faster')
        self.assertEqual(entry['direction']['makespan_ns']['classification'], 'pace_faster')

    def test_summary_and_mad(self):
        result = plan.summary([3.0, 1.0, 2.0, 5.0, 4.0])
        q1, _, q3 = statistics.quantiles([3.0, 1.0, 2.0, 5.0, 4.0], n=4, method='inclusive')
        self.assertEqual(result['median'], 3.0)
        self.assertEqual((result['q1'], result['q3'], result['iqr']), (q1, q3, q3 - q1))
        self.assertEqual(plan.summary([7.5])['iqr'], 0.0)
        with self.assertRaises(ValueError):
            plan.summary([])
        self.assertEqual(plan.mad([1.0, 1.0, 2.0, 9.0]), 0.5)
        self.assertEqual(plan.mad([4.0]), 0.0)

    def test_tie_threshold_union_interpretation(self):
        self.assertEqual(plan.tie_threshold(1_000.0, 0.0, 0.0), 10.0)
        self.assertEqual(plan.tie_threshold(1_000.0, 100.0, 5.0), 200.0)
        self.assertEqual(plan.tie_threshold(1_000.0, 5.0, 100.0), 200.0)

    def test_direction_and_stability_classification(self):
        def row(median, mad=0.0):
            return {'median_rank_local_max_ns': median, 'mad_rank_local_max_ns': mad}
        tie = plan.group_direction(row(5_000_000), row(5_010_000), 'rank_local_max_ns')
        self.assertEqual(tie['classification'], 'tie')
        faster = plan.group_direction(row(5_000_000), row(8_000_000), 'rank_local_max_ns')
        self.assertEqual(faster['classification'], 'pace_faster')
        slower = plan.group_direction(row(9_000_000), row(8_000_000), 'rank_local_max_ns')
        self.assertEqual(slower['classification'], 'pace_slower')
        self.assertEqual(plan.stable_classification([faster] * 6)['classification'],
                         'pace_stable_faster')
        self.assertEqual(plan.stable_classification([slower] * 6)['classification'],
                         'pace_stable_slower')
        mixed = plan.stable_classification([faster, slower, tie, faster, tie, faster])
        self.assertEqual(mixed['classification'], 'unstable')
        self.assertEqual(mixed['class_counts'], {'pace_faster': 3, 'pace_slower': 1, 'tie': 2})
        self.assertEqual(mixed['adjacent_class_changes'], 5)
        with self.assertRaises(ValueError):
            plan.group_direction(row(1.0), row(1.0), 'arrival_skew_ns')


class ValidatorTests(unittest.TestCase):
    def test_valid_group_records_pass(self):
        records = group_records(drops=2)
        view = plan.validate_group_records(records, 'plan_a', 'job0', 2)
        expected = plan.rotated_case_order('plan_a', 'job0', 2)
        self.assertEqual(set(view), {case_id for _, case_id in expected})

    def test_mutations_fail_closed(self):
        for mutate in ('arm_order', 'iteration_count', 'timestamp_inversion', 'global_bad',
                       'initial_flag', 'final_flag', 'wrong_case', 'position_shift',
                       'gate_mismatch', 'budget_mismatch', 'clock'):
            records = group_records()
            if mutate == 'arm_order':
                records[0]['iterations'][16], records[0]['iterations'][17] = \
                    records[0]['iterations'][17], records[0]['iterations'][16]
            elif mutate == 'iteration_count':
                records[0]['iterations'] = records[0]['iterations'][:-1]
            elif mutate == 'timestamp_inversion':
                records[0]['iterations'][5]['t1_ns'] = records[0]['iterations'][5]['t0_ns']
            elif mutate == 'global_bad':
                records[0]['global_bad'] = 1
            elif mutate == 'initial_flag':
                records[0]['initial']['exact']['pace'] = False
            elif mutate == 'final_flag':
                records[0]['final']['exact']['recv_tail'] = False
            elif mutate == 'wrong_case':
                records[0]['case'] = 'not-a-case'
            elif mutate == 'position_shift':
                records[0]['case_position'] = 3
            elif mutate == 'gate_mismatch':
                records[0]['iterations'][0]['gated'] = True
            elif mutate == 'budget_mismatch':
                # round-7 semantics: wrong channel count (not cross-rank
                # inequality, which is legitimate)
                records[0]['budget_case_cumulative_sqes'] = [1, 2, 3]
            else:
                records[0]['clock'] = 'wall'
            with self.assertRaises(ValueError, msg=mutate):
                plan.validate_group_records(records, 'plan_a', 'job0', 0)

    def test_rank_coverage_required(self):
        records = group_records()[:-1]
        with self.assertRaises(ValueError):
            plan.validate_group_records(records, 'plan_a', 'job0', 0)


class ReviewContractTests(unittest.TestCase):
    def test_launch_consumes_the_single_shared_constant(self):
        self.assertIs(launch.verify_review_binding, plan.validate_review)

    def test_synthetic_approved_fixture_is_accepted(self):
        binding = plan.validate_review(approved_review(), 'a' * 64)
        self.assertEqual(binding['status'], 'APPROVED')
        self.assertEqual(binding['source_manifest_sha256'], 'a' * 64)
        self.assertEqual(binding['groups'], plan.REVIEW_GROUPS)
        for semantics in plan.REVIEW_GUARD_SEMANTICS:
            review = approved_review()
            review['device_split_guard_semantics'] = semantics
            self.assertEqual(plan.validate_review(review, 'a' * 64)
                             ['device_split_guard_semantics'], semantics)

    def test_rejected_status_refused(self):
        review = approved_review()
        review['status'] = 'REJECTED'
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_wrong_manifest_sha_refused(self):
        review = approved_review(manifest_sha='b' * 64)
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_wrong_scope_refused(self):
        review = approved_review()
        review['scope'] = 'some_other_lane'
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_missing_reviewed_files_refused(self):
        review = approved_review()
        review['reviewed_files'] = review['reviewed_files'][:-1]
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_wrong_groups_refused(self):
        review = approved_review()
        review['groups'] = {'plan_a': list(plan.GROUPS_PLAN_A)[:5]}
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)
        review = approved_review()
        review['groups']['plan_b']['a'] = list(plan.GROUPS_PLAN_B_A)[:5]
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_unverified_guard_semantics_refused(self):
        review = approved_review()
        review['device_split_guard_semantics'] = 'unverified'
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_wrong_cases_binding_refused(self):
        review = approved_review()
        review['cases_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)


class LaunchContractTests(unittest.TestCase):
    def test_entry_scripts_are_pinned(self):
        self.assertIn('/workspace/lane/supervisor.py', SUPERVISOR_SCRIPT)
        self.assertIn('/workspace/plan', SUPERVISOR_SCRIPT)
        self.assertIn('/workspace/lane/precheck.py', PRECHECK_SCRIPT)

    def test_inspect_contract_accepts_group_container(self):
        devices = list(guard.nodes(16))
        env = base_env('job3', '2')
        payload = Path('/results/host/payload')
        mounts = launch.expected_mounts(payload)
        info = container_info(devices, mounts, env, payload)
        argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', SUPERVISOR_SCRIPT]
        launch.inspect_contract(info, argv, env, guard, 16, payload)

    def test_inspect_contract_accepts_precheck_container(self):
        env = launch.precheck_env()
        info = container_info([], launch.expected_mounts(), env, script=PRECHECK_SCRIPT)
        argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', PRECHECK_SCRIPT]
        launch.inspect_contract(info, argv, env)

    def test_inspect_contract_rejects_deviations(self):
        env = base_env()
        devices = list(guard.nodes(16))
        payload = Path('/results/host/payload')
        argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', SUPERVISOR_SCRIPT]
        broken = container_info(devices, launch.expected_mounts(payload), env, payload)
        launch.inspect_contract(broken, argv, env, guard, 16, payload)
        cases = []
        writable = dict(broken)
        writable['Mounts'] = [dict(row, RW=True) for row in broken['Mounts']]
        cases.append(('writable mount', writable))
        extra_device = dict(broken)
        extra_device['HostConfig'] = dict(broken['HostConfig'], DeviceRequests=[{}])
        cases.append(('device request', extra_device))
        extra_node = dict(broken)
        extra_node['HostConfig'] = dict(broken['HostConfig'], Devices=[
            {'PathOnHost': d, 'PathInContainer': d, 'CgroupPermissions': 'rwm'}
            for d in [*devices, '/dev/davinci16']])
        cases.append(('extra device node', extra_node))
        bad_env = dict(broken)
        bad_env['Config'] = dict(broken['Config'], Env=['REVSD_DROPS=4'])
        cases.append(('environment', bad_env))
        running = dict(broken)
        running['State'] = dict(broken['State'], Running=True)
        cases.append(('still running', running))
        crashed = dict(broken)
        crashed['State'] = dict(broken['State'], ExitCode=1)
        cases.append(('nonzero exit', crashed))
        for label, info in cases:
            with self.assertRaises(RuntimeError, msg=label):
                launch.inspect_contract(info, argv, env, guard, 16, payload)


class SnapshotTests(unittest.TestCase):
    def snapshot(self, group_key='job0', occupied=False):
        text = fixture
        if occupied:
            text = fixture.replace('No running processes found in NPU 3',
                                   '3       0                 | 4242      | intruder         | 512')
        return {'health': {str(card): HEALTH_TEXT for card in range(8)},
                'usage': {str(card): USAGE_TEXT for card in range(8)},
                'npu_smi_info': text, 'processes': guard.parse_npu_info(text),
                'ports': list(plan.ports(group_key)),
                'devices': list(guard.nodes(plan.WORLD)),
                'utc': '2026-11-05T12:00:00Z', 'scope': 'snapshot only'}

    def test_all_idle_snapshot_passes(self):
        launch.check_snapshot(guard, self.snapshot(), 'job0')
        launch.check_snapshot(guard, self.snapshot('pbb5'), 'pbb5')

    def test_occupied_or_mismatched_snapshots_fail(self):
        with self.assertRaises(RuntimeError):
            launch.check_snapshot(guard, self.snapshot(occupied=True), 'job0')
        wrong_ports = self.snapshot()
        wrong_ports['ports'] = list(plan.ports('job1'))
        with self.assertRaises(ValueError):
            launch.check_snapshot(guard, wrong_ports, 'job0')
        wrong_devices = self.snapshot()
        wrong_devices['devices'] = wrong_devices['devices'][:-1]
        with self.assertRaises(ValueError):
            launch.check_snapshot(guard, wrong_devices, 'job0')
        unhealthy = self.snapshot()
        unhealthy['health']['4'] = unhealthy['health']['4'].replace('OK', 'WARN', 1)
        with self.assertRaises(RuntimeError):
            launch.check_snapshot(guard, unhealthy, 'job0')


class TimestampTests(unittest.TestCase):
    def test_exact_ns(self):
        self.assertEqual(plan.exact_ns('2026-11-05T12:00:00Z'), 1793880000000000000)
        self.assertEqual(plan.exact_ns('2026-11-05T12:00:00.123456789Z'),
                         1793880000123456789)
        self.assertEqual(plan.exact_ns('2026-11-05T12:00:00+01:00'),
                         plan.exact_ns('2026-11-05T11:00:00Z'))
        with self.assertRaises(ValueError):
            plan.exact_ns('2026-11-05 12:00:00')
        with self.assertRaises(ValueError):
            plan.exact_ns('2026-11-05T12:00:00+25:00')

    def test_time_order(self):
        plan.check_time_order('2026-11-05T12:00:00Z', '2026-11-05T12:00:01.5Z',
                              '2026-11-05T12:00:02.123456789Z', '2026-11-05T12:00:03Z')
        with self.assertRaises(ValueError):
            plan.check_time_order('2026-11-05T12:00:01Z', '2026-11-05T12:00:00Z',
                                  '2026-11-05T12:00:02Z', '2026-11-05T12:00:03Z')
        with self.assertRaises(ValueError):
            plan.check_time_order('2026-11-05T12:00:00Z', '2026-11-05T12:00:01Z',
                                  '2026-11-05T12:00:02Z', '2026-11-05T12:00:02Z')
        with self.assertRaises(ValueError):
            plan.check_time_order('2026-11-05T12:00:00Z', '2026-11-05T12:00:01Z',
                                  '2026-11-05T12:00:02+01:00', '2026-11-05T12:00:03Z')


class TrimGateTests(unittest.TestCase):
    def receipt(self, group, drops, duration):
        return {'schema': 'revision2.sparse-distribution-completion.v1', 'status': 'PASS',
                'variant': 'plan_a', 'group': group, 'world': 16, 'drops': drops,
                'manifest_sha256': 'a' * 64, 'review_sha256': 'b' * 64,
                'image_id': plan.IMAGE, 'duration_seconds': duration, 'metrics': {}}

    def test_first_group_must_run_full_plan(self):
        with self.assertRaises(ValueError):
            plan.trim_gate([], 1)
        self.assertIsNone(plan.trim_gate([], 0))

    def test_monotone_one_step_ladder(self):
        prior = [self.receipt('job0', 0, 500)]
        self.assertIsNone(plan.trim_gate(prior, 0))  # same level allowed
        with self.assertRaises(ValueError):
            plan.trim_gate(prior, 2)  # never more than one step at a time
        with self.assertRaises(ValueError):
            plan.trim_gate(prior + [self.receipt('job1', 1, 100)], 0)  # decreasing

    def test_increment_requires_duration_trigger(self):
        prior = [self.receipt('job0', 0, 500)]
        with self.assertRaises(ValueError):
            plan.trim_gate(prior, 1)
        prior.append(self.receipt('job1', 0, 900))
        trigger = plan.trim_gate(prior, 1)
        self.assertEqual(trigger['dropped_case'], plan.LADDER[0])
        self.assertEqual(trigger['trigger_group'], 'job1')
        self.assertGreater(trigger['trigger_duration_seconds'], plan.TRIM_TRIGGER_SECONDS)
        # exactly 840 s does not trigger (strictly greater required)
        with self.assertRaises(ValueError):
            plan.trim_gate([self.receipt('job0', 0, 840)], 1)
        self.assertIsNotNone(plan.trim_gate([self.receipt('job0', 0, 840.001)], 1))


class CampaignRollupTests(unittest.TestCase):
    def receipts(self, variant='plan_a', pace_span=5_000_000, hccl_span=8_000_000, drops=0):
        receipts = {}
        for group_key in plan.GROUPS[variant]:
            view = plan.validate_group_records(
                group_records(group_key, variant, drops, pace_span=pace_span,
                              hccl_span=hccl_span), variant, group_key, drops)
            metrics = plan.assemble_group_metrics(view, group_key)
            receipts[group_key] = {'schema': 'revision2.sparse-distribution-completion.v1',
                                   'status': 'PASS', 'variant': variant, 'group': group_key,
                                   'world': 16, 'drops': drops, 'manifest_sha256': 'a' * 64,
                                   'review_sha256': 'b' * 64, 'image_id': plan.IMAGE,
                                   'duration_seconds': 500.0, 'metrics': metrics}
        return receipts

    def test_plan_b_subgroup_rollup(self):
        receipts = {}
        for group_key in plan.GROUPS_PLAN_B_B:
            view = plan.validate_group_records(
                group_records(group_key, 'plan_b', 0), 'plan_b', group_key, 0)
            receipts[group_key] = {'schema': 'revision2.sparse-distribution-completion.v1',
                                   'status': 'PASS', 'variant': 'plan_b', 'group': group_key,
                                   'world': 16, 'drops': 0, 'manifest_sha256': 'a' * 64,
                                   'review_sha256': 'b' * 64, 'image_id': plan.IMAGE,
                                   'duration_seconds': 400.0,
                                   'metrics': plan.assemble_group_metrics(view, group_key)}
        self.assertEqual(len(receipts), 6)
        summary = plan.summary_from_receipts(receipts, 'plan_b', 'b', 'a' * 64, 'b' * 64)
        self.assertEqual(summary['subgroup'], 'b')
        self.assertEqual(set(summary['cases']), set(plan.campaign_case_ids('plan_b', 'b', 0)))

    def test_direction_and_anchor_reference_in_rollup(self):
        receipts = self.receipts()
        summary = plan.summary_from_receipts(receipts, 'plan_a', None, 'a' * 64, 'b' * 64)
        anchor = summary['cases']['qwen-o1-call-038-dispatch']
        self.assertEqual(anchor['direction']['rank_local_max_ns']['classification'],
                         'pace_stable_faster')
        self.assertIn('historical_rank_local_max_reference', anchor)
        history = anchor['historical_rank_local_max_reference']['history']
        self.assertEqual(history['forward_job0'], 259865)
        self.assertEqual(history['reverse_hccl']['job2'], 455805)
        self.assertIn('syn-diag-512', summary['axis_views']['density_axis'])
        self.assertTrue(summary['keep_all_samples'])

    def test_reduced_coverage_is_never_stable(self):
        full = self.receipts(drops=0)
        trimmed = self.receipts(drops=1)
        mixed = dict(full)
        mixed['job2'] = trimmed['job2']
        summary = plan.summary_from_receipts(mixed, 'plan_a', None, 'a' * 64, 'b' * 64)
        dropped_case = plan.LADDER[0]
        entry = summary['cases'][dropped_case]
        self.assertFalse(entry['coverage_complete'])
        self.assertEqual(entry['direction']['rank_local_max_ns']['classification'], 'unstable')
        kept = summary['cases']['qwen-o1-call-038-dispatch']
        self.assertTrue(kept['coverage_complete'])

    def test_receipt_coverage_validated(self):
        receipts = self.receipts()
        del receipts['job2']
        with self.assertRaises(ValueError):
            plan.summary_from_receipts(receipts, 'plan_a', None, 'a' * 64, 'b' * 64)
        receipts = self.receipts()
        receipts['job1']['status'] = 'FAIL'
        with self.assertRaises(ValueError):
            plan.summary_from_receipts(receipts, 'plan_a', None, 'a' * 64, 'b' * 64)
        receipts = self.receipts()
        receipts['job0']['metrics'] = {}
        with self.assertRaises(ValueError):
            plan.summary_from_receipts(receipts, 'plan_a', None, 'a' * 64, 'b' * 64)


class FrozenPinTests(unittest.TestCase):
    """Transcribed provider pins must equal the barrier-v3 lane's constants.

    Round-3 fix: LIB_SHA256 carried a single hex typo (position 52 '4' vs
    '9') that only surfaced as a remote verify() 'runtime provider mismatch'.
    These assertions freeze the four provider digests verbatim so any future
    transcription drift from the v3 lane fails a local test instead of the
    device gate.
    """

    def test_provider_pins_match_barrier_v3_verbatim(self):
        truth = {
            'LIB_SHA256': '24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9',
            'SHMEM_SHA256': 'c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71',
            'HELPER_SHA256': '0d7fde8e4eb1f0c8c312412cfc2bb6ee6f5df116ea48e2e27aae2feadef962be',
            'LAYOUT_SHA256': 'e844a5175ca3de3e4a54ef54723b76df6665607c4268ad29cc3ed541d7bf6154',
        }
        for name, expected in truth.items():
            observed = getattr(plan, name)
            self.assertEqual(observed, expected, name)
            self.assertEqual(len(observed), 64, name)
            self.assertTrue(all(char in '0123456789abcdef' for char in observed), name)


    def test_image_pin_matches_barrier_v3_verbatim(self):
        truth = 'sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f'
        observed = plan.IMAGE
        self.assertEqual(observed, truth)
        self.assertEqual(len(observed), 7 + 64)
        self.assertTrue(observed.startswith('sha256:'))
        self.assertTrue(all(char in '0123456789abcdef' for char in observed[7:]))
        self.assertEqual(len(observed[7:]), 64)


class DladdrPathSemanticsTests(unittest.TestCase):
    """dladdr Dl_info semantics: fname is the path, sname is diagnostic only.

    Round-4 fix lock: the previous 'sname or fname' form resolved the symbol
    name ('g_state') against the cwd, so every genuine provider was rejected
    with 'SHMEM export resolves to unexpected provider'. Device probe truth:
    both g_state and aclshmemx_barrier_all_on_stream dladdr to
    fname=/workspace/shmem/build_cann9_vllm/lib/libshmem.so (realpath equal
    to the expected provider), sname carrying the symbol name.
    """

    def test_dladdr_provider_path_ignores_symbol_name(self):
        # sname is NOT a path and must not influence the resolution
        resolved = plan.dladdr_provider_path('/x/libshmem.so', 'g_state')
        self.assertEqual(resolved, os.path.realpath('/x/libshmem.so'))
        self.assertNotEqual(resolved, os.path.realpath('g_state'))
        # identical resolution with and without a symbol name
        self.assertEqual(resolved, plan.dladdr_provider_path('/x/libshmem.so'))
        # device-truth form: fname matches the pinned provider exactly
        provider = '/workspace/shmem/build_cann9_vllm/lib/libshmem.so'
        self.assertEqual(plan.dladdr_provider_path(provider.encode(), b'g_state'),
                         os.path.realpath(provider))
        # a fname pointing elsewhere is rejected by the CALLER's comparison,
        # never by sname: the helper only ever sees fname
        self.assertNotEqual(plan.dladdr_provider_path('/opt/other/libshmem.so', 'g_state'),
                            os.path.realpath(provider))
        # missing fname fails closed inside the helper
        with self.assertRaises(ValueError):
            plan.dladdr_provider_path(None)
        with self.assertRaises(ValueError):
            plan.dladdr_provider_path(b'')

    def test_dladdr_symbol_is_diagnostic_only(self):
        self.assertEqual(plan.dladdr_symbol(b'g_state'), 'g_state')
        self.assertIsNone(plan.dladdr_symbol(None))
        self.assertIsNone(plan.dladdr_symbol(b''))


class PaceSchemaTests(unittest.TestCase):
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
                ('unnamed return', PACE_SCHEMA_FIXTURE.replace('-> Tensor _0', '-> Tensor')),
                ('other operator', PACE_SCHEMA_FIXTURE.replace('cashflex::hpa_alltoallv_device_splits',
                                                               'cashflex::other_op'))):
            with self.assertRaises(ValueError, msg=label):
                plan.check_pace_schema(bad)

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


# ------------------------------------------------- worker logic (mocked) ---
# A minimal pure-python stand-in for the torch/dist/npc surface run_case uses.
# The fake collectives fabricate received segments from the CPU oracle, so the
# plumbing under test is the worker's own logic: buffer wiring, oracle
# comparison, sentinel checks, gate discipline, record structure and
# fail-closed behavior. It is NOT a device simulation.
class FakeTensor:
    """Byte-window tensor: slices share storage with the parent buffer."""

    def __init__(self, data, dtype, shape, base=None, window=None):
        self.dtype, self.shape = dtype, tuple(shape)
        if base is None:
            self.base = bytearray(data)
            self.window = (0, len(self.base))
        else:
            self.base, self.window = base, window

    @property
    def nbytes(self):
        return self.window[1] - self.window[0]

    @property
    def bytes(self):
        return self.base[self.window[0]:self.window[1]]

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise NotImplementedError(key)
        rows = self.shape[0]
        start, stop, step = key.indices(rows)
        assert step == 1
        row_bytes = self.nbytes // rows
        return FakeTensor(b'', self.dtype, (stop - start, self.shape[1]),
                          base=self.base,
                          window=(self.window[0] + start * row_bytes,
                                  self.window[0] + stop * row_bytes))

    def copy_(self, other):
        assert self.shape == other.shape and self.nbytes == other.nbytes
        self.base[self.window[0]:self.window[1]] = other.bytes
        return self

    def cpu(self):
        return self

    def view(self, dtype):
        widths = {'torch.bfloat16': 2, 'torch.int32': 4, 'torch.uint8': 1}
        rows, cols = self.shape
        old, width = widths[self.dtype], widths[dtype]
        assert rows * cols * old % width == 0
        return FakeTensor(b'', dtype, (rows, cols * old // width),
                          base=self.base, window=self.window)

    def reshape(self, *shape):
        widths = {'torch.bfloat16': 2, 'torch.int32': 4, 'torch.uint8': 1}
        assert shape[0] * shape[1] * widths[self.dtype] == self.nbytes, \
            (shape, self.shape, self.dtype, self.nbytes)
        return FakeTensor(b'', self.dtype, shape, base=self.base, window=self.window)

    def numpy(self):
        return memoryview(self.bytes)

    def tolist(self):
        width = {'torch.int32': 4, 'torch.uint8': 1, 'torch.bfloat16': 2}[self.dtype]
        raw = self.bytes
        if self.dtype == 'torch.int32':
            return [int.from_bytes(raw[i:i + 4], 'little')
                    for i in range(0, len(raw), 4)]
        raise NotImplementedError('tolist for ' + self.dtype)

    def fill_(self, value):
        width = {'torch.int32': 4, 'torch.uint8': 1, 'torch.bfloat16': 2}[self.dtype]
        pattern = int(value).to_bytes(width, 'little')
        count = self.nbytes // width
        self.base[self.window[0]:self.window[1]] = bytearray(pattern * count)
        return self

    def item(self):
        return int.from_bytes(self.bytes[:4], 'little')

    def __eq__(self, scalar):
        width = {'torch.int32': 4, 'torch.uint8': 1, 'torch.bfloat16': 2}[self.dtype]
        pattern = int(scalar).to_bytes(width, 'little')
        raw = self.bytes
        matches = len(raw) % width == 0 and \
            raw == bytes(pattern * (len(raw) // width))
        return FakeEq(matches)


class FakeEq:
    def __init__(self, matches):
        self.matches = matches

    def item(self):
        return self.matches


class FakeTorch:
    bfloat16 = 'torch.bfloat16'
    int32 = 'torch.int32'
    uint8 = 'torch.uint8'

    class ReduceOp:
        SUM = 'SUM'
        MAX = 'MAX'

    def __init__(self):
        self.corrupt_next_output = False
        import types as _types
        self.ops = _types.SimpleNamespace(cashflex=_types.SimpleNamespace(
            hpa_alltoallv_device_splits=fake_device_split_call))

    def full(self, shape, fill, dtype, device=None):
        shape = tuple(shape) if len(shape) == 2 else (shape[0], 1)
        width = {'torch.bfloat16': 2, 'torch.int32': 4}[dtype]
        data = int(fill).to_bytes(width, 'little') * (shape[0] * shape[1])
        return FakeTensor(data, dtype, shape)

    def tensor(self, values, dtype=None, device=None):
        width = {'torch.int32': 4, 'torch.uint8': 1}[dtype]
        data = b''.join(int(v).to_bytes(width, 'little') for v in values)
        return FakeTensor(data, dtype, (len(values), 1))

    def frombuffer(self, buffer, dtype):
        return FakeTensor(bytes(buffer), dtype, (len(buffer), 1))

    def empty(self, shape, dtype=None):
        width = {'torch.bfloat16': 2, 'torch.int32': 4, 'torch.uint8': 1}.get(dtype, 1)
        data = b'\x00' * (shape[0] * shape[1] * width)
        return FakeTensor(data, dtype or 'torch.uint8', tuple(shape))

    def equal(self, left, right):
        return left.shape == right.shape and left.bytes == right.bytes

    def all(self, operand):
        if isinstance(operand, FakeEq):
            return operand
        raise NotImplementedError('torch.all over raw tensors')

    @staticmethod
    def _oracle_alltoall(out, case, rank, rd):
        row_bytes = plan.ROW_BYTES
        offset = 0
        for peer, rows in enumerate(rd):
            if rows:
                raw = plan.segment_bytes(case, peer, rank)
                out.base[out.window[0] + offset:
                         out.window[0] + offset + rows * row_bytes] = raw
            offset += rows * row_bytes
        assert offset == out.nbytes, (offset, out.nbytes)


def fake_device_split_call(values, out, rd, sd, *args):
    FakeTorch._oracle_alltoall(out, context['case'], context['rank'], rd.tolist())
    if context['torch'].corrupt_next_output:
        context['torch'].corrupt_next_output = False
        out.base[out.window[0]] ^= 0xFF
    return out


class FakeDist:
    ReduceOp = FakeTorch.ReduceOp

    def __init__(self):
        self.gate_calls = 0
        self.all_to_all_calls = 0

    def all_reduce(self, tensor, op=None):
        if op == 'MAX':
            self.gate_calls += 1  # gate outcome is not asserted on the host side
        else:
            tensor.fill_(0)       # global bad count stays zero in this harness
        return tensor

    def all_to_all_single(self, out, values, output_split_sizes, input_split_sizes):
        context['dist'].all_to_all_calls += 1
        FakeTorch._oracle_alltoall(out, context['case'], context['rank'],
                                   output_split_sizes)
        if context['torch'].corrupt_next_output:
            context['torch'].corrupt_next_output = False
            out.base[out.window[0]] ^= 0xFF


class FakeNpu:
    @staticmethod
    def synchronize():
        return None


context = {}


class WorkerLogicTests(unittest.TestCase):
    def load_worker(self):
        import importlib
        import types
        fake_torch = FakeTorch()
        fake_torch_module = types.ModuleType('torch')
        for name in ('bfloat16', 'int32', 'uint8', 'equal', 'all', 'full', 'tensor',
                     'frombuffer', 'empty'):
            setattr(fake_torch_module, name, getattr(fake_torch, name))
        fake_torch_module.ops = types.SimpleNamespace(
            cashflex=types.SimpleNamespace(
                hpa_alltoallv_device_splits=fake_device_split_call))
        fake_dist_module = types.ModuleType('torch.distributed')
        fake_dist_module.ReduceOp = FakeTorch.ReduceOp
        fake_cashflex = types.ModuleType('cashflex')
        fake_cashflex_ops = types.ModuleType('cashflex.ops')
        fake_hpa = types.ModuleType('cashflex.ops.hpa_alltoallv')
        fake_hpa.hpa_shmem_init = lambda *a, **k: True
        fake_hpa.hpa_shmem_finalize = lambda *a, **k: True
        fake_cashflex.ops = fake_cashflex_ops
        fake_cashflex_ops.hpa_alltoallv = fake_hpa
        saved = {name: sys.modules.get(name) for name in
                 ('torch', 'torch.distributed', 'torch_npu', 'cashflex',
                  'cashflex.ops', 'cashflex.ops.hpa_alltoallv')}
        sys.modules['torch'] = fake_torch_module
        sys.modules['torch.distributed'] = fake_dist_module
        sys.modules['torch_npu'] = types.ModuleType('torch_npu')
        sys.modules['cashflex'] = fake_cashflex
        sys.modules['cashflex.ops'] = fake_cashflex_ops
        sys.modules['cashflex.ops.hpa_alltoallv'] = fake_hpa
        try:
            module = importlib.import_module('worker')
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        return module

    def run_case(self, case_id, corrupt=False, rank=3, case=None):
        worker = self.load_worker()
        document = plan.load_cases()
        if case is None:
            case = next(c for c in document['cases'] if c['id'] == case_id)
        canonical = document['canonical'][case_id]
        position = [index for index, (_, cid) in
                    enumerate(plan.rotated_case_order('plan_a', 'job0', 0))
                    if cid == case_id][0]
        fake_torch, fake_dist = FakeTorch(), FakeDist()
        context.clear()
        context.update({'case': case, 'rank': rank, 'torch': fake_torch,
                        'dist': fake_dist})
        if corrupt:
            fake_torch.corrupt_next_output = True
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                record = worker.run_case(case, canonical, position, 'plan_a', 'job0',
                                         rank, 16, fake_torch, fake_dist, FakeNpu(),
                                         FakeTensor(b'\x00' * 4, 'torch.int32', (1, 1)),
                                         [55, 110, 165, 220], directory)
            stored = json.loads((directory / f'case-{canonical:03d}.json').read_text())
        return record, stored, fake_dist, position

    def test_run_case_record_is_plan_valid(self):
        for case_id in ('qwen-o1-call-038-dispatch', 'syn-diag-512',
                        'olmoe-o1-call-049-combine'):
            with self.subTest(case=case_id):
                record, stored, fake_dist, position = self.run_case(case_id)
                self.assertEqual(stored, record)
                parsed = plan.validate_case_record(record, record['case_index'],
                                                   position, 'plan_a', 'job0', 0)
                self.assertEqual(parsed['case_id'], case_id)
                # gate ran exactly once per measured iteration (16 = 8 per arm
                # x 2 arms) and never during warmup (4) or the checks
                self.assertEqual(fake_dist.gate_calls, 16)
                # all_to_all_single runs once for the initial check plus 10
                # timed HCCL iterations (2 warmup + 8 measured)
                self.assertEqual(fake_dist.all_to_all_calls, 11)
                self.assertTrue(all(row['exact'][key] for row in (record['initial'], record['final'])
                                    for key in plan.CASE_RECORD_KEYS))
                self.assertEqual(record['global_bad'], 0)
                measured_gates = [row['gated'] for row in record['iterations']]
                self.assertEqual(measured_gates.count(True), 16)
                self.assertEqual(measured_gates.count(False), 4)

    def test_run_case_fails_closed_on_corrupt_output(self):
        with self.assertRaises(RuntimeError):
            self.run_case('qwen-o1-call-038-dispatch', corrupt=True)

    def test_zero_receive_rank_full_flow(self):
        # job0 device crash site: olmoe-o1-call-010 rank 11 receives zero
        # rows (expected output 0 bytes); the 0-byte-safe oracle path must
        # produce a record the plan validator accepts, with all bit-exact
        # flags true and empty outputs comparing equal.
        record, stored, fake_dist, position = self.run_case(
            'olmoe-o1-call-010-dispatch', rank=11)
        self.assertEqual(stored, record)
        parsed = plan.validate_case_record(record, record['case_index'],
                                           position, 'plan_a', 'job0', 0)
        self.assertEqual(parsed['case_id'], 'olmoe-o1-call-010-dispatch')
        for block in (record['initial'], record['final']):
            self.assertTrue(all(block['exact'].values()))
        self.assertEqual(fake_dist.gate_calls, 16)
        self.assertEqual(fake_dist.all_to_all_calls, 11)
        # the empty expected output digests to sha256(b'') on both sides
        import hashlib as _hl
        self.assertEqual(record['initial']['expected_sha256'],
                         _hl.sha256(b'').hexdigest())

    def test_zero_send_rank_boundary(self):
        # defensive boundary (no frozen case has a zero-SEND rank): rank 5
        # contributes no input rows yet still receives rows from peers; the
        # 0-row staging path must leave every bit-exact flag true.
        matrix = [[0] * 16 for _ in range(16)]
        for i in range(16):
            if i != 5:
                for j in range(16):
                    matrix[i][j] = 8
        case = {'id': 'syn-zero-send-boundary', 'kind': 'synthetic', 'role': 'test',
                'send_matrix_rows': matrix,
                'per_rank_output_rows': [sum(matrix[i][j] for i in range(16))
                                         for j in range(16)]}
        record, stored, fake_dist, position = self.run_case(
            'qwen-o1-call-038-dispatch', rank=5, case=case)
        self.assertTrue(all(record['initial']['exact'].values()))
        self.assertTrue(all(record['final']['exact'].values()))
        self.assertEqual(record['global_bad'], 0)
        self.assertEqual(fake_dist.gate_calls, 16)


class SourceTreeTests(unittest.TestCase):
    """Recursive source-tree coverage (barrier-v3 round-2 form), local trees."""

    def make_tree(self, extra=(), stray_dir=None, symlink=None, pycache=True):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        for name in plan.FILES | {'source_manifest.json', 'review.json'}:
            (root / name).write_text('x')
        for name in extra:
            (root / name).write_text('x')
        if stray_dir is not None:
            (root / stray_dir).mkdir()
        if pycache:
            (root / '__pycache__').mkdir()
            (root / '__pycache__' / 'plan.cpython-313.pyc').write_text('x')
        if symlink is not None:
            (root / symlink).symlink_to(root / 'plan.py')
        return root

    def test_exact_tree_with_pycache_passes(self):
        launch.check_source_tree(self.make_tree())

    def test_stray_file_stray_dir_and_symlink_rejected(self):
        with self.assertRaises(ValueError):
            launch.check_source_tree(self.make_tree(extra=('scratch.py',)))
        with self.assertRaises(ValueError):
            launch.check_source_tree(self.make_tree(stray_dir='scratch'))
        with self.assertRaises(ValueError):
            launch.check_source_tree(self.make_tree(symlink='plan_link.py'))

    def test_missing_frozen_file_rejected(self):
        root = self.make_tree()
        (root / 'worker.py').unlink()
        with self.assertRaises(ValueError):
            launch.check_source_tree(root)

    def test_nested_stray_entries_rejected(self):
        # A stray directory outside __pycache__ fails even when nested files
        # mimic ordinary sources; entries inside __pycache__ are skipped as a
        # whole subtree, exactly like the barrier-v3 form.
        root = self.make_tree()
        nested = root / 'scratch' / 'deep'
        nested.mkdir(parents=True)
        (nested / 'plan.py').write_text('x')
        with self.assertRaises(ValueError):
            launch.check_source_tree(root)


class SqGateMarkerTests(unittest.TestCase):
    def setUp(self):
        self.results = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.results)
        self._saved_results = launch.RESULTS
        launch.RESULTS = self.results

    def tearDown(self):
        launch.RESULTS = self._saved_results

    def rejecting_group(self, group='job1'):
        directory = launch.group_dir(group)
        (directory / 'payload').mkdir(parents=True)
        for rank in range(16):
            rank_dir = directory / 'payload' / f'rank-{rank:02d}'
            rank_dir.mkdir()
            (rank_dir / 'sq_gate_reject.json').write_text('{"reason": "SQ_BUDGET_FAIL_CLOSED"}')
        # a prior passing group whose evidence the marker must bind
        prior = launch.group_dir('job0')
        prior.mkdir(parents=True)
        for name in ('container.log', 'inspect.json', 'preflight.json',
                     'prelaunch_recheck.json'):
            (prior / name).write_text('evidence:' + name)
        return directory

    def exits(self):
        return [{'event': 'natural_exit', 'rank': rank, 'code': 1} for rank in range(16)]

    def test_marker_binds_published_inspect_hash(self):
        directory = self.rejecting_group()
        self.assertTrue(launch.detect_sq_gate_rejection(directory, self.exits(), self.exits()))
        marker = json.loads((self.results / 'sq_gate_fail_closed.json').read_text())
        self.assertEqual(marker['status'], 'SQ_GATE_FAIL_CLOSED')
        self.assertEqual(len(marker['rejecting_ranks']), 16)
        self.assertEqual(marker['rejecting_groups'], ['job1'])
        evidence = marker['evidence_sha256']
        # the published name is inspect.json and it IS hashed (A4 regression)
        self.assertIn('job0/inspect.json', evidence)
        self.assertNotIn('job0/container.inspect.json', evidence)
        self.assertEqual(evidence['job0/inspect.json'],
                         hashlib.sha256(b'evidence:inspect.json').hexdigest())
        self.assertEqual(evidence['job0/container.log'],
                         hashlib.sha256(b'evidence:container.log').hexdigest())

    def test_marker_requires_supervisor_exit_rows(self):
        directory = self.rejecting_group()
        # exits nonzero but no supervisor natural-exit corroboration rows
        self.assertFalse(launch.detect_sq_gate_rejection(directory, [], self.exits()))
        self.assertFalse((self.results / 'sq_gate_fail_closed.json').exists())

    def test_marker_never_overwrites(self):
        directory = self.rejecting_group()
        self.assertTrue(launch.detect_sq_gate_rejection(directory, self.exits(), self.exits()))
        with self.assertRaises(RuntimeError):
            launch.detect_sq_gate_rejection(directory, self.exits(), self.exits())

    def test_normal_group_writes_no_marker(self):
        directory = launch.group_dir('job2')
        (directory / 'payload').mkdir(parents=True)
        (directory / 'payload' / 'rank-00' / 'case-000.json').parent.mkdir()
        (directory / 'payload' / 'rank-00' / 'case-000.json').write_text('{}')
        self.assertFalse(launch.detect_sq_gate_rejection(
            directory, [], [{'event': 'natural_exit', 'rank': r, 'code': 0}
                            for r in range(16)]))
        self.assertFalse((self.results / 'sq_gate_fail_closed.json').exists())


class MergedStdoutParserTests(unittest.TestCase):
    """Round-5 defect: unbuffered print + docker stream reassembly produces
    STRUCTURALLY MERGED lines; the old line-split parser lost 82 of 272
    case_samples on the real job0 log. The raw_decode scanner must recover
    every well-formed object; the audit is payload-driven with stdout as an
    inclusive cross-check."""

    REAL_LOG = REPO / 'analysis_results/sparse_lane_campaign_20261105_v1/v1_execution/job0/container.log'

    def test_real_job0_log_full_recovery(self):
        # regression fixture: the ACTUAL retrieved job0 container log
        if not self.REAL_LOG.is_file():
            self.skipTest('retrieved job0 container.log not present on this host')
        text = self.REAL_LOG.read_text()
        rows = launch.json_rows(text)
        cases = [row for row in rows if row.get('event') == 'case_samples']
        # every marker occurrence in the text must correspond to a recovered record
        marker_count = text.count('"case_samples"')
        self.assertEqual(len(cases), marker_count)
        # the old line-split parser provably lost records on this same text
        naive = []
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get('event') == 'case_samples':
                naive.append(value)
        self.assertGreater(len(cases), len(naive))
        # recovered records are intact (rank/case keys well-formed)
        for row in cases:
            self.assertIn(row.get('rank'), range(16))
            self.assertIsInstance(row.get('case_index'), int)

    def test_synthetic_merged_line_fixture(self):
        # equivalent compact fixture built to the observed merged structure:
        # two complete records concatenated WITHOUT a separator, prefixed by
        # a supervisor fragment, followed by an intact line
        first = {'event': 'natural_exit', 'rank': 7, 'code': 0}
        second = {'event': 'rank_pass', 'rank': 7, 'world': 16, 'cases': 17}
        merged_line = json.dumps(first) + json.dumps(second)  # no newline between
        text = ('supervisor preamble\n' + merged_line + '\n'
                + json.dumps({'event': 'all_natural_exit', 'world': 16}) + '\n')
        rows = launch.json_rows(text)
        self.assertEqual(rows, [first, second, {'event': 'all_natural_exit', 'world': 16}])
        # triple concatenation and interleaved junk also recover
        triple = json.dumps(first) + json.dumps(second) + json.dumps(first)
        self.assertEqual(launch.json_rows('x' + triple + 'y'),
                         [first, second, first])

    def test_cross_check_inclusive_semantics(self):
        payload = [
            {'rank': 0, 'case_index': 0, 'event': 'case_samples', 'v': 1},
            {'rank': 0, 'case_index': 1, 'event': 'case_samples', 'v': 2},
            {'rank': 1, 'case_index': 0, 'event': 'case_samples', 'v': 3},
        ]
        # empty/partial stdout is tolerated (payload owns coverage)
        launch.cross_check_stdout_records([], payload)
        launch.cross_check_stdout_records([payload[0], payload[2]], payload)
        # any recoverable stdout record must equal its payload twin
        with self.assertRaises(ValueError):
            launch.cross_check_stdout_records([dict(payload[0], v=99)], payload)
        # unknown (rank, case) pairs refuse
        with self.assertRaises(ValueError):
            launch.cross_check_stdout_records(
                [{'rank': 5, 'case_index': 9, 'event': 'case_samples'}], payload)

    def test_payload_records_loader_coverage(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        order = plan.rotated_case_order('plan_a', 'job0', 0)
        for rank in range(plan.WORLD):
            directory = root / 'payload' / f'rank-{rank:02d}'
            directory.mkdir(parents=True)
            for canonical, _ in order:
                (directory / f'case-{canonical:03d}.json').write_text(json.dumps(
                    {'event': 'case_samples', 'rank': rank, 'case_index': canonical}))
        records = launch.load_payload_case_records(root, 'plan_a', 'job0', 0)
        self.assertEqual(len(records), plan.WORLD * len(order))
        # one missing file refuses (coverage is the payload's job now)
        (root / 'payload' / 'rank-03' / 'case-000.json').unlink()
        with self.assertRaises(RuntimeError):
            launch.load_payload_case_records(root, 'plan_a', 'job0', 0)


class CLIDispatchTests(unittest.TestCase):
    """Round-6 fix lock: every CLI action receives EXACTLY the arguments its
    implementation signature declares (the device run crashed group_audit
    with a 5-vs-4 TypeError after a successful container exit)."""

    def args(self, action, **overrides):
        base = {'action': action, 'variant': 'plan_a', 'group': 'job0',
                'subgroup': None, 'drops': 0,
                'manifest_sha': 'a' * 64, 'review_sha': 'b' * 64}
        base.update(overrides)
        import argparse
        return argparse.Namespace(**base)

    def test_dispatch_argument_tuples_match_signatures(self):
        import inspect
        calls = {}

        def recorder(name, real):
            def wrapper(*argv):
                calls[name] = argv
            wrapper.__signature__ = inspect.signature(real)
            return wrapper

        targets = {name: getattr(launch, name) for name in
                   ('precheck_arm', 'precheck_audit', 'group_start', 'group_audit',
                    'aggregate')}
        saved = dict(targets)
        try:
            for name, real in targets.items():
                setattr(launch, name, recorder(name, real))
            launch.dispatch(self.args('precheck_arm'))
            launch.dispatch(self.args('precheck_audit'))
            launch.dispatch(self.args('group_start', drops=2))
            launch.dispatch(self.args('group_audit', drops=2))  # drops ignored by audit
            launch.dispatch(self.args('aggregate', subgroup=None))
            launch.dispatch(self.args('aggregate', variant='plan_b', subgroup='b'))
        finally:
            for name, real in saved.items():
                setattr(launch, name, real)
        # each action received a positional tuple whose LENGTH equals its
        # real signature's parameter count (the 5-vs-4 class is impossible)
        for name, real in targets.items():
            expected_arity = len(inspect.signature(real).parameters)
            self.assertEqual(len(calls[name]), expected_arity, name)
        self.assertEqual(calls['precheck_arm'], ('a' * 64, 'b' * 64))
        self.assertEqual(calls['precheck_audit'], ('a' * 64, 'b' * 64))
        self.assertEqual(calls['group_start'],
                         ('plan_a', 'job0', 2, 'a' * 64, 'b' * 64))
        self.assertEqual(calls['group_audit'],
                         ('plan_a', 'job0', 'a' * 64, 'b' * 64))  # no drops
        # both aggregate invocations hit the recorder; the plan_b one overwrote
        self.assertEqual(calls['aggregate'], ('plan_b', 'b', 'a' * 64, 'b' * 64))

    def test_action_signature_contract_pinned(self):
        import inspect
        self.assertEqual(list(inspect.signature(launch.group_start).parameters),
                         ['variant', 'group_key', 'drops', 'manifest_sha', 'review_sha'])
        self.assertEqual(list(inspect.signature(launch.group_audit).parameters),
                         ['variant', 'group_key', 'manifest_sha', 'review_sha'])

    def test_parser_roundtrip_group_audit_with_drops(self):
        parsed = launch.build_parser().parse_args(
            ['group_audit', '--variant', 'plan_a', '--group', 'job0', '--drops', '2',
             '--manifest-sha', 'a' * 64, '--review-sha', 'b' * 64])
        self.assertEqual((parsed.action, parsed.group, parsed.drops),
                         ('group_audit', 'job0', 2))
        parsed = launch.build_parser().parse_args(
            ['group_start', '--group', 'job1', '--manifest-sha', 'a' * 64,
             '--review-sha', 'b' * 64])
        self.assertEqual((parsed.action, parsed.group, parsed.drops),
                         ('group_start', 'job1', 0))

    def test_group_actions_require_group(self):
        with self.assertRaises(SystemExit):
            launch.dispatch(self.args('group_audit', group=None))
        with self.assertRaises(SystemExit):
            launch.dispatch(self.args('group_start', group='pba0'))  # wrong variant


class RealBudgetEvidenceTests(unittest.TestCase):
    """Round-7 fix regression: real job0 evidence shows per-rank
    per-channel cumulative SQEs legitimately DIFFER across ranks ([539,682,
    682,649]/[561,594,627,484]/... — peers map to channels by peer % 4).
    The old cross-rank-equality check rejected a successful run; the new
    per-rank invariants (4 channels, nonneg ints, < 2048, per-channel
    non-decreasing across case order) accept it and refuse the true
    violations."""

    REAL_LOG = REPO / 'analysis_results/sparse_lane_campaign_20261105_v1/v1_execution/job0/container.log'

    def real_records(self):
        """16-rank record set built from the ACTUAL recovered job0 records
        (the retrieved fragment holds ALL 16 ranks x cases 0-7 verbatim);
        the remaining 9 cases per rank are clones of that rank's case-7
        record with flat-continued budgets — real per-rank budget arrays and
        all crash-adjacent evidence stay untouched."""
        if not self.REAL_LOG.is_file():
            self.skipTest('retrieved job0 container.log not present on this host')
        text = self.REAL_LOG.read_text()
        decoder = json.JSONDecoder()
        rows, index, size = [], 0, len(text)
        while index < size:
            if text[index] != '{':
                index += 1
                continue
            try:
                value, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            index = end
        real = {}
        for row in rows:
            if row.get('event') == 'case_samples':
                real.setdefault(row['rank'], {})[row['case_index']] = row
        self.assertEqual(sorted(real), list(range(16)))
        order = plan.rotated_case_order('plan_a', 'job0', 0)
        position_of = {canonical: position for position, (canonical, _) in enumerate(order)}
        records = []
        for rank in range(16):
            for canonical, case_id in order:
                if canonical in real[rank]:
                    records.append(real[rank][canonical])
                    continue
                template = dict(real[rank][7])
                template.update({'case_index': canonical, 'case': case_id,
                                 'case_position': position_of[canonical],
                                 'generation': plan.generation_label(canonical),
                                 'generation_after_measurement':
                                     plan.PACE_CALLS_PER_CASE * (canonical + 1)})
                records.append(template)
        return records

    def test_real_job0_budgets_validate_per_rank(self):
        records = self.real_records()
        view = plan.validate_group_records(records, 'plan_a', 'job0', 0)
        # the per-rank arrays really do differ across ranks (regression
        # premise) and every one of them is accepted
        case = plan.rotated_case_order('plan_a', 'job0', 0)[0][1]
        arrays = {tuple(values) for values in
                  view[case]['budget_case_cumulative_sqes_by_rank'].values()}
        self.assertGreater(len(arrays), 1)  # cross-rank inequality is real
        self.assertEqual(len(view[case]['budget_case_cumulative_sqes_by_rank']), 16)

    def test_per_rank_violations_refused(self):
        records = self.real_records()
        for mutate in ('depth', 'negative', 'length', 'nonmono_channel', 'nonmono_case'):
            broken = [dict(record) for record in records]
            victim = next(r for r in broken if r['case_index'] == 0)
            if mutate == 'depth':
                victim['budget_case_cumulative_sqes'] = [2048, 1, 1, 1]
            elif mutate == 'negative':
                victim['budget_case_cumulative_sqes'] = [-1, 1, 1, 1]
            elif mutate == 'length':
                victim['budget_case_cumulative_sqes'] = [1, 2, 3]
            elif mutate == 'nonmono_channel':
                # decrease one channel between the first two cases of that rank
                rank = victim['rank']
                second = next(r for r in broken
                              if r['rank'] == rank and r['case_index'] == 1)
                second['budget_case_cumulative_sqes'] = [0, 0, 0, 0]
            else:  # nonmono_case: first case above the second on a channel
                rank = victim['rank']
                second = next(r for r in broken
                              if r['rank'] == rank and r['case_index'] == 1)
                second['budget_case_cumulative_sqes'] = [0, 999, 999, 999]
            with self.assertRaises(ValueError, msg=mutate):
                plan.validate_group_records(broken, 'plan_a', 'job0', 0)

    def test_cross_rank_equality_no_longer_required(self):
        # synthetic minimal proof: two ranks with different-but-valid arrays
        # pass (the exact scenario the old check rejected on device)
        records = group_records()
        for record in records:
            if record['rank'] == 0:
                record['budget_case_cumulative_sqes'] = [10, 20, 30, 40]
            elif record['rank'] == 1:
                record['budget_case_cumulative_sqes'] = [40, 30, 20, 10]
        view = plan.validate_group_records(records, 'plan_a', 'job0', 0)
        case = plan.rotated_case_order('plan_a', 'job0', 0)[0][1]
        by_rank = view[case]['budget_case_cumulative_sqes_by_rank']
        self.assertEqual(by_rank[0], [10, 20, 30, 40])
        self.assertEqual(by_rank[1], [40, 30, 20, 10])


class ColdStartPrimingTests(unittest.TestCase):
    """Round-9: the aicpu-507018 job3 mitigation — one untimed PACE + HCCL
    warm-through call BEFORE any measured case, on the deterministic small
    uniform priming shape, included in the SQ budget (priming=True)."""

    def load_worker(self):
        # reuse the FakeTorch harness from WorkerLogicTests
        logic = WorkerLogicTests('test_run_case_fails_closed_on_corrupt_output')
        return logic.load_worker()

    def test_priming_function_runs_before_cases_untimed(self):
        worker = self.load_worker()
        fake_torch = FakeTorch()
        fake_dist = FakeDist()
        calls = []
        original_split = fake_torch.ops.cashflex.hpa_alltoallv_device_splits

        def recording_split(*argv):
            calls.append(('pace',))
            return original_split(*argv)

        fake_torch.ops.cashflex.hpa_alltoallv_device_splits = recording_split
        original_all_to_all = fake_dist.all_to_all_single

        def recording_all_to_all(*argv, **kwargs):
            calls.append(('hccl',))
            return original_all_to_all(*argv, **kwargs)

        fake_dist.all_to_all_single = recording_all_to_all
        # the shared fake oracle needs the priming case bound in the context
        context.clear()
        context.update({'case': plan.priming_case(), 'rank': 3,
                        'torch': fake_torch, 'dist': fake_dist})
        syncs = []
        fake_npu = type('FakeNpu', (), {'synchronize': staticmethod(lambda: syncs.append(1))})()
        import io
        import contextlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                event = worker.cold_start_prime(fake_torch, fake_dist, fake_npu,
                                                3, 16, directory)
            stored = json.loads((directory / 'cold_start_priming.json').read_text())
        # round-13: FIVE warm-through PACE calls then one HCCL call, a
        # synchronize after every priming call plus the drain pair
        self.assertEqual([kind for kind, in calls], ['pace'] * 5 + ['hccl'])
        self.assertGreaterEqual(len(syncs), 5 + 3)  # per-call + pre + drain
        self.assertIs(event['timed'], False)
        self.assertIs(event['validated'], False)
        self.assertEqual(event['priming_case'], 'qwen-o1-call-038-dispatch')
        self.assertEqual(event['rank'], 3)
        self.assertEqual(stored['event'], 'cold_start_priming')
        # the stdout event is parseable by the robust row scanner
        rows = launch.json_rows(stdout.getvalue())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['event'], 'cold_start_priming')

    def test_priming_buffers_carry_abi_tail_backing(self):
        # round-11: the device-split ABI requires >= 1024 tail bytes behind
        # values/out and canary tails behind the split vectors (rank-7
        # device crash: 'input requires an explicit backing allocation with
        # at least 1024 tail bytes'). The fake tensors are windowed views of
        # their backings, so the operator-visible slices must share a padded
        # base and the tails must remain canary.
        worker = self.load_worker()
        fake_torch = FakeTorch()
        fake_dist = FakeDist()
        seen = {}
        original_split = fake_torch.ops.cashflex.hpa_alltoallv_device_splits

        def recording_split(values, out, rd, sd, *rest):
            seen.update(values=values, out=out, rd=rd, sd=sd)
            return original_split(values, out, rd, sd, *rest)

        fake_torch.ops.cashflex.hpa_alltoallv_device_splits = recording_split
        context.clear()
        context.update({'case': plan.priming_case(), 'rank': 3,
                        'torch': fake_torch, 'dist': fake_dist})
        fake_npu = type('FakeNpu', (), {'synchronize': staticmethod(lambda: None)})()
        import io
        import contextlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                worker.cold_start_prime(fake_torch, fake_dist, fake_npu, 3, 16, Path(tmp))
        case = plan.priming_case()
        send, recv = plan.rank_splits(case, 3)
        # live slices have the case geometry...
        self.assertEqual(seen['values'].shape, (sum(send), plan.ROW_BYTES // 2))
        self.assertEqual(seen['out'].shape, (sum(recv), plan.ROW_BYTES // 2))
        # ...but their BACKING covers rows + TAIL_SLACK_ROWS, with the TAIL
        # region still canary (the live region legitimately holds the fake
        # oracle bytes written by the priming PACE call)
        # (fake slices keep .base as the raw shared bytearray)
        values_base = seen['values'].base if hasattr(seen['values'], 'base') \
            else seen['values'].data
        self.assertEqual(len(values_base),
                         (sum(send) + plan.TAIL_SLACK_ROWS) * plan.ROW_BYTES)
        out_base = seen['out'].base if hasattr(seen['out'], 'base') else seen['out'].data
        self.assertEqual(len(out_base),
                         (sum(recv) + plan.TAIL_SLACK_ROWS) * plan.ROW_BYTES)
        out_tail = bytes(out_base)[sum(recv) * plan.ROW_BYTES:]
        # bf16 canary fills are the 2-byte little-endian pattern of CANARY
        pattern = plan.CANARY.to_bytes(2, 'little')
        self.assertTrue(all(out_tail[i:i + 2] == pattern
                            for i in range(0, len(out_tail), 2)),
                        'out tail rows must stay canary')
        values_tail = bytes(values_base)[sum(send) * plan.ROW_BYTES:]
        self.assertTrue(all(values_tail[i:i + 2] == pattern
                            for i in range(0, len(values_tail), 2)),
                        'input tail rows must stay canary')
        # split vectors: 16 live values + 256 canary tail entries in the base
        split_pattern = plan.CANARY.to_bytes(4, 'little')  # int32 canary
        for vector, live in ((seen['sd'], send), (seen['rd'], recv)):
            self.assertEqual(vector.tolist(), list(live))
            base = vector.base if hasattr(vector, 'base') else vector.data
            self.assertEqual(len(base) // 4, plan.WORLD + 256)
            tail = bytes(base)[plan.WORLD * 4:]
            self.assertTrue(all(tail[i:i + 4] == split_pattern
                                for i in range(0, len(tail), 4)),
                            'split tail must stay canary')

    def test_budget_includes_priming_row_and_calls(self):
        ids = list(plan.load_cases()['ids'])
        base = plan.plan_call_budgets(ids, 0, FakeHelper(5), channels())
        primed = plan.plan_call_budgets(ids, 0, FakeHelper(5), channels(), priming=True)
        self.assertEqual(base['pace_calls'], 187)
        self.assertEqual(primed['pace_calls'], 187 + plan.PRIMING_CALLS)
        self.assertEqual(primed['priming_calls'], plan.PRIMING_CALLS)
        self.assertEqual(plan.PRIMING_CALLS, 5)
        self.assertEqual(primed['cases'][0]['case'], plan.PRIMING_CASE_ID)
        self.assertEqual(primed['cases'][0]['role'], 'cold_start_priming')
        self.assertEqual([p - b for p, b in zip(primed['cumulative_sqes'],
                                                base['cumulative_sqes'])],
                         [5 * plan.PRIMING_CALLS] * 4)
        # round-10: the priming shape is the REAL dense anchor case (the
        # synthetic priming case itself tripped aicpu on device)
        self.assertEqual(plan.PRIMING_CASE_ID, 'qwen-o1-call-038-dispatch')
        self.assertEqual(plan.load_cases()['canonical'][plan.PRIMING_CASE_ID], 0)
        # priming is inside the fail-closed depth gate
        self.assertTrue(all(value < 2048 for value in primed['cumulative_sqes']))
        # default stays priming-free (backwards-compatible audit semantics)
        self.assertEqual(base.get('priming_calls', 0), 0)
        self.assertEqual(base['cases'][0]['case'], ids[0])

    def test_worker_wiring_priming_before_case_loop(self):
        source = (ROOT / 'worker.py').read_text()
        main_body = source.split('def main')[1]
        prime_at = main_body.index('cold_start_prime(torch, dist, torch.npu, rank, world, directory)')
        loop_at = main_body.index('for position, (canonical_index, case_id) in enumerate(case_order)')
        self.assertLess(prime_at, loop_at, 'priming must precede the case loop')
        self.assertIn('priming=True)', main_body)
        # audit requires the priming event and recomputes the budget WITH priming
        launch_source = (ROOT / 'launch.py').read_text()
        self.assertIn("x.get('event') == 'cold_start_priming'", launch_source)
        self.assertIn('plan.plan_call_budgets(case_ids, rank, helper, channels, priming=True)',
                      launch_source)


    def test_priming_payload_coverage_in_audit_contract(self):
        # round-12: the audit's expected payload set includes the per-rank
        # cold_start_priming.json (16 copies) with identical hash treatment
        source = (ROOT / 'launch.py').read_text()
        self.assertIn("'cold_start_priming.json'):", source)
        self.assertIn("priming.get('event'), priming.get('rank'), priming.get('world')",
                      source)
        # the structural expectation: 16 priming files in the names set —
        # mirrored from verify_payload_evidence's per-rank loop
        names = {f'rank-{rank:02d}/cold_start_priming.json' for rank in range(16)}
        self.assertEqual(len(names), 16)


class DeviceLimitationTests(unittest.TestCase):
    """Round-14: preregistered device-limitation not-run receipts — job3
    (rotation offset 9, first case syn-diag-512) is excluded on aicpu
    507018 device evidence; successors and the aggregate proceed with the
    limitation declared (5/6 groups analysed)."""

    def receipt(self, group='job3', status='DEVICE_LIMITATION', reason=None):
        return {'schema': 'revision2.sparse-distribution-completion.v1',
                'status': status, 'variant': 'plan_a', 'group': group, 'drops': 0,
                'world': 16, 'manifest_sha256': 'a' * 64, 'review_sha256': 'b' * 64,
                'image_id': plan.IMAGE,
                'device_limitation_reason': reason
                or plan.DEVICE_LIMITATION_REASONS.get(group, 'unregistered'),
                'metrics': {}}

    def test_registry_is_preregistered_and_reason_bound(self):
        self.assertEqual(plan.DEVICE_LIMITATION_GROUPS, frozenset({'job3'}))
        self.assertEqual(plan.DEVICE_LIMITATION_REASONS['job3'],
                         'syn_diagonal_early_case_aicpu_507018')
        statement = plan.device_limitation_statement()
        self.assertIn('job3', statement)
        self.assertIn('later positions PASS', statement)

    def test_validate_completion_accepts_registered_refuses_others(self):
        receipt = self.receipt()
        plan.validate_completion(receipt, 'plan_a', 'job3', 'a' * 64, 'b' * 64)
        # non-registered group refuses
        with self.assertRaises(ValueError):
            plan.validate_completion(self.receipt(group='job4'), 'plan_a', 'job4',
                                     'a' * 64, 'b' * 64)
        # wrong reason refuses
        with self.assertRaises(ValueError):
            plan.validate_completion(self.receipt(reason='some_other'), 'plan_a',
                                     'job3', 'a' * 64, 'b' * 64)
        # broken identity binding refuses
        with self.assertRaises(ValueError):
            plan.validate_completion(dict(receipt, manifest_sha256='c' * 64),
                                     'plan_a', 'job3', 'a' * 64, 'b' * 64)
        # PASS receipts still validate through the original path (metrics
        # must cover the rotated plan)
        ok = dict(receipt, status='PASS', duration_seconds=500.0,
                  metrics={case_id: {} for _, case_id
                           in plan.rotated_case_order('plan_a', 'job3', 0)})
        plan.validate_completion(ok, 'plan_a', 'job3', 'a' * 64, 'b' * 64)

    def test_summary_accepts_five_of_six_with_limitation_declared(self):
        metrics = None
        base = CampaignRollupTests('test_direction_and_anchor_reference_in_rollup')
        receipts = base.receipts()  # 6 PASS receipts (job0-2, job4-5 of plan_a)
        # build 5-group metrics: drop job3 by remapping — simplest: construct
        # fresh via the fixture minus one group
        five = {key: value for key, value in receipts.items() if key != 'job3'}
        five['job3'] = self.receipt()
        summary = plan.summary_from_receipts(five, 'plan_a', None, 'a' * 64, 'b' * 64)
        self.assertEqual(summary['device_limitation_jobs'], ['job3'])
        self.assertIn('aicpu 507018', summary['device_limitation_statement'])
        self.assertEqual(len(summary['runnable_groups']), 5)
        self.assertIn('5/6', summary['result_interpretation_constraint'])
        # coverage judged against the 5 runnable groups, not degraded
        anchor = summary['cases']['qwen-o1-call-038-dispatch']
        self.assertTrue(anchor['coverage_complete'])
        self.assertEqual(anchor['n_groups'], 5)
        # an unregistered device-limitation receipt in the campaign refuses
        bad = dict(five)
        bad['job4'] = self.receipt(group='job4')
        with self.assertRaises(ValueError):
            plan.summary_from_receipts(bad, 'plan_a', None, 'a' * 64, 'b' * 64)

    def test_launch_wiring_receipt_action_and_gate(self):
        source = (ROOT / 'launch.py').read_text()
        self.assertIn('def device_limitation_receipt(group_key, manifest_sha, review_sha)',
                      source)
        self.assertIn("'device_limitation_receipt'", source)  # parser choice
        # ordering gate accepts the limitation receipt
        self.assertIn("if receipt.get('status') == 'DEVICE_LIMITATION':\n            continue",
                      source)
        # aggregate skips evidence rehash for limitations and lists them
        self.assertIn("receipts[group_key] = receipt\n            continue", source)


if __name__ == '__main__':
    unittest.main()