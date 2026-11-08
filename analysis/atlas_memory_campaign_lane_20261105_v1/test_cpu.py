"""Host logic tests only; no torch, device, processes or network.

Covers the transcription pins (IMAGE 71-char, four provider digests, GUARD_*,
L2 memory_snapshot identity), the namespaced PACE schema fixture and its
rejection variants, the dladdr fname-only path semantics, the supervisor
contract, the campaign grid (jobs/order/budgets/ports/conditionals/pilot
quarantine/trim ladder), the SHA-bound arms.json with arithmetic
recomputation and tamper rejection, per-job config resolution incl. the A6
KV-floor rule, the capacity ladder/exhaustion/censoring mathematics, the
three-lens rules (L3 adoption, L1/L2 reconciliation, Tier-1 peak alignment,
fill probe, duration gate), the L1 poller parser/stream validators, the
fencepost validator, the SINGLE review-contract constant (APPROVED accepted;
REJECTED / wrong manifest / wrong scope / missing files / wrong groups /
wrong arms binding / unverified semantics refused), the launcher container
contract incl. the C0-1 memory-limit variant, the recursive source-tree
coverage, and the exact-nanosecond timestamp ordering.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
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
    'memcamp_guard', ROOT.parent / 'atlas_group_admission_device_v3' / 'launch.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
fixture = json.loads((ROOT / 'observed_npu_info_fixture.json').read_text())['npu_smi_info']

SUPERVISOR_SCRIPT = launch.entry_script('supervisor.py')
PRECHECK_SCRIPT = launch.entry_script('precheck.py')

# Device-measured PACE schema truth inside the pinned image (captured live
# during the barrier-v3 review round 3); frozen as the ABI-pin test fixture.
PACE_SCHEMA_FIXTURE = ('cashflex::hpa_alltoallv_device_splits(Tensor _0, Tensor _1, Tensor _2, '
                       'Tensor _3, int _4, int _5, int _6, int _7) -> Tensor _0')

HEALTH_TEXT = '\n'.join(['Health : OK'] * 3)
USAGE_TEXT = '\n'.join(
    ['HBM Capacity(MB)      : 65536'] * 2 +
    ['HBM Usage Rate(%)     : 8'] * 2 +
    ['Aicore Usage Rate(%)  : 0'] * 2 +
    ['Aivector Usage Rate(%) : 0'] * 2 +
    ['NPU Utilization(%)    : 0'] * 2)


class Child:
    pid = 777

    def __init__(self, codes):
        self.codes = list(codes)

    def poll(self):
        return self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]


def container_info(devices, mounts, env, payload=None, script=SUPERVISOR_SCRIPT,
                   memory=None):
    host = {'NetworkMode': 'host' if devices else 'none',
            'IpcMode': 'host' if devices else 'private', 'Memory': memory,
            'Devices': [{'PathOnHost': d, 'PathInContainer': d, 'CgroupPermissions': 'rwm'}
                        for d in devices]}
    argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', script]
    return {'Image': plan.IMAGE, 'HostConfig': host,
            'State': {'Running': False, 'ExitCode': 0, 'OOMKilled': False, 'Error': ''},
            'Config': {'Env': [f'{k}={v}' for k, v in env.items()], 'Cmd': argv[-3:]},
            'Mounts': [{'Destination': t, 'Source': s, 'RW': w}
                       for t, (s, w) in mounts.items()]}


def base_env(job='A1', trim='0', l3='0'):
    shmem_port, master_port = plan.ports(job)
    env = {'MEMCAMP_JOB': job, 'MEMCAMP_TRIM': trim,
           'MEMCAMP_ARMS_SHA256': plan.ARMS_SHA256, 'MEMCAMP_L3': l3,
           'WORLD_SIZE': '16', 'MASTER_ADDR': '127.0.0.1',
           'MASTER_PORT': str(master_port),
           'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(16))),
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
    if job in ('A2', 'A4', 'A6', 'C1b', 'C2BB1'):
        env.update({'REV2_BACKEND': 'sync-diagnostic',
                    'HPA_SHMEM_IP_PORT': f'tcp://127.0.0.1:{shmem_port}',
                    'HPA_SHMEM_WORKSPACE_PLANES': '2'})
    else:
        env['REV2_BACKEND'] = 'hccl'
    env.update(plan.ENGINE_ENV_BASE)
    return env


def approved_review(manifest_sha='a' * 64):
    return {'schema': plan.REVIEW_SCHEMA, 'status': plan.REVIEW_STATUS_APPROVED,
            'source_manifest_sha256': manifest_sha, 'arms_sha256': plan.ARMS_SHA256,
            'scope': plan.REVIEW_SCOPE, 'reviewed_files': sorted(plan.REVIEWED_FILES),
            'groups': json.loads(json.dumps(plan.REVIEW_GROUPS)),
            'device_split_guard_semantics': 'whole_input'}


class FrozenPinTests(unittest.TestCase):
    """Transcribed pins must equal the barrier-v3 / #8-lane values verbatim."""

    def test_provider_pins_match_barrier_v3_verbatim(self):
        truth = {
            'LIB_SHA256': '24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9',
            'SHMEM_SHA256': 'c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71',
            'HELPER_SHA256': '0d7fde8e4eb1f0c8c312412cfc2bb6ee6f5df116ea48e2e27aae2feadef962be',
            'LAYOUT_SHA256': 'e844a5175ca3de3e4a54ef54723b76df6665607c4268ad29cc3ed541d7bf6154',
            'GUARD_LAUNCH_SHA': 'b4d96c96b5e304d8b0cf74c264d7b6006c158b018c7243bf055a74b79d91ffc2',
            'GUARD_MANIFEST_SHA': '288fb6913cab6181af5021505be0534b3105704c75358cb3252bf1cd31246d94',
            'MEMORY_SNAPSHOT_SHA256': '3f02de28ce69e8be24fc6d0b3ab6f1ad728e859c03d1cdb9b4e88eaa853c3402',
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

    def test_memory_api_source_pins(self):
        self.assertEqual(plan.MEMORY_API_SOURCE_SHA256,
                         {'mem_get_info': 'dc6b6763a6127c0a057a5e5f4f962bc94182e8a0697b446d5a8faeab35c9b058',
                          'memory_stats': '9689894e9cb47ba6884f191e1b45326576441233475142108b05671866ed4947'})


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


class DladdrPathSemanticsTests(unittest.TestCase):
    def test_dladdr_provider_path_ignores_symbol_name(self):
        resolved = plan.dladdr_provider_path('/x/libshmem.so', 'g_state')
        self.assertEqual(resolved, os.path.realpath('/x/libshmem.so'))
        self.assertNotEqual(resolved, os.path.realpath('g_state'))
        self.assertEqual(resolved, plan.dladdr_provider_path('/x/libshmem.so'))
        provider = '/workspace/shmem/build_cann9_vllm/lib/libshmem.so'
        self.assertEqual(plan.dladdr_provider_path(provider.encode(), b'g_state'),
                         os.path.realpath(provider))
        with self.assertRaises(ValueError):
            plan.dladdr_provider_path(None)
        with self.assertRaises(ValueError):
            plan.dladdr_provider_path(b'')

    def test_dladdr_symbol_is_diagnostic_only(self):
        self.assertEqual(plan.dladdr_symbol(b'g_state'), 'g_state')
        self.assertIsNone(plan.dladdr_symbol(None))
        self.assertIsNone(plan.dladdr_symbol(b''))

    def test_discover_acl_library(self):
        path = '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so'
        text = '\n'.join(['7f00-7f10 r--p 00000000 fd:01 1234 ' + path,
                          '7f10-7f30 r-xp 00001000 fd:01 1234 ' + path,
                          '7f00-7f10 r--p 00000000 fd:01 99 /usr/lib/libother.so'])
        self.assertEqual(plan.discover_acl_library(text), path)
        with self.assertRaises(ValueError):
            plan.discover_acl_library('7f00-7f10 r--p 0 fd:01 1 /usr/lib/libother.so')


class SupervisorTests(unittest.TestCase):
    def env(self, job='A1', trim='0', l3='0'):
        return {'MEMCAMP_JOB': job, 'MEMCAMP_TRIM': trim,
                'MEMCAMP_ARMS_SHA256': plan.ARMS_SHA256, 'MEMCAMP_L3': l3}

    def test_normal_job_spawns_driver(self):
        calls, output = [], []

        def popen(argv, env):
            calls.append((argv, env))
            return Child([None, 0])

        observed, errors = supervise(self.env(), popen, lambda x: None, output.append)
        self.assertEqual(observed, {0: 0})
        self.assertFalse(errors)
        self.assertEqual(calls[0][0][-1], '/workspace/lane/worker.py')
        self.assertEqual(calls[0][1]['PYTHONUNBUFFERED'], '1')

    def test_driver_failure_preserves_evidence(self):
        output = []
        observed, errors = supervise(self.env(), lambda *a, **k: Child([1]),
                                     lambda x: None, output.append)
        self.assertEqual(observed, {0: 1})
        self.assertTrue(any('no signaling' in x for x in output))

    def test_spawn_failure_recorded(self):
        output = []

        def popen(argv, env):
            raise OSError('failed')

        observed, errors = supervise(self.env(), popen, lambda x: None, output.append)
        self.assertEqual(observed, {0: -1})
        self.assertTrue(errors)

    def test_rejects_bad_job_trim_arms_or_l3(self):
        for env in (self.env(job='Z9'), self.env(trim='-1'), self.env(trim='x'),
                    self.env(l3=''), self.env(l3='maybe'),
                    {**self.env(), 'MEMCAMP_ARMS_SHA256': '0' * 64}):
            with self.assertRaises(ValueError):
                supervise(env, lambda *a, **k: self.fail('started'), lambda x: None)


class CampaignStructureTests(unittest.TestCase):
    def test_campaign_order_unique_and_gated(self):
        self.assertEqual(len(plan.CAMPAIGN_ORDER), len(set(plan.CAMPAIGN_ORDER)))
        self.assertEqual(len(plan.CAMPAIGN_ORDER), 45)
        self.assertEqual(plan.CAMPAIGN_ORDER[:2], ('A0a', 'A0p'))
        self.assertEqual(plan.CAMPAIGN_ORDER[2:4], ('C01', 'C02'))
        for job in plan.CAMPAIGN_ORDER:
            plan.job_spec(job, 0)
        with self.assertRaises(ValueError):
            plan.job_spec('A9', 0)

    def test_ports_disjoint_and_ranged(self):
        seen = set()
        for job in plan.CAMPAIGN_ORDER:
            ports = plan.ports(job)
            self.assertNotIn(ports, seen)
            seen.add(ports)
            self.assertTrue(26801 <= ports[0] < 26901 and 26901 <= ports[1] < 27001)
        with self.assertRaises(ValueError):
            plan.ports('A9')

    def test_budget_minutes_cover_every_job(self):
        self.assertEqual(set(plan.JOB_BUDGET_MINUTES), set(plan.CAMPAIGN_ORDER))
        for job, minutes in plan.JOB_BUDGET_MINUTES.items():
            self.assertGreater(minutes, 0, job)
        # PLAN section 4 core arithmetic anchors
        self.assertEqual(plan.JOB_BUDGET_MINUTES['C01'] + plan.JOB_BUDGET_MINUTES['C02'], 12)
        self.assertEqual(sum(plan.JOB_BUDGET_MINUTES[j] for j in ('C2BA1', 'C2BA2', 'C2BA3',
                                                                  'C2BA4', 'C2BA5')), 35)

    def test_pilot_quarantine_and_conditionals(self):
        self.assertEqual(plan.QUARANTINED, frozenset(('A0a', 'A0p')))
        for job in plan.QUARANTINED:
            self.assertTrue(plan.job_spec(job, 0)['quarantined'])
        self.assertFalse(plan.job_spec('A1', 0)['quarantined'])
        self.assertEqual(sorted(plan.CONDITIONAL_JOBS),
                         ['C3BD1', 'C3BD2', 'C7GA', 'C7GB', 'C7SA', 'C7SB'])
        self.assertEqual(plan.CONDITIONAL_JOBS['C3BD1'], 'per_peer_guard')
        self.assertEqual(plan.CONDITIONAL_JOBS['C7SA'], 'a6_feasible')

    def test_trim_ladder_monotone_removal(self):
        full = set(plan.CAMPAIGN_ORDER)
        previous = full
        for steps in range(1, plan.MAX_TRIM_STEPS + 1):
            current = set(plan.effective_campaign(steps))
            dropped = previous - current
            self.assertEqual(len(dropped), 1, f'step {steps} must drop exactly one job')
            self.assertEqual(dropped.pop(), plan.TRIM_LADDER[steps - 1][1])
            previous = current
        # full ladder drops: C7(4) + C6(4) + confirms(3) + repeats(2) + B-C(6)
        self.assertEqual(len(plan.TRIM_LADDER), 19)
        self.assertEqual(plan.TRIM_LADDER[0][1], 'C7GA')
        self.assertEqual(plan.TRIM_LADDER[18][1], 'C5BC')
        # MAX_TRIM_STEPS == ladder length; one step past it is refused
        self.assertEqual(plan.MAX_TRIM_STEPS, len(plan.TRIM_LADDER))
        with self.assertRaises(ValueError):
            plan.job_spec('A1', plan.MAX_TRIM_STEPS + 1)

    def test_trimmed_job_refused(self):
        with self.assertRaises(ValueError):
            plan.job_spec('C7GA', 4)
        self.assertIsNotNone(plan.job_spec('A1', 4))


class ArmsBindingTests(unittest.TestCase):
    def test_load_arms_recomputes_every_fact(self):
        document = plan.load_arms()
        self.assertEqual(sorted(document['arms']), ['B-A', 'B-B', 'B-C', 'B-D'])
        self.assertEqual(document['models']['olmoe']['kv_bytes_per_block_per_rank'],
                         plan.MIB)

    def test_tampered_arms_file_fails_sha_binding(self):
        document = json.loads((plan.LOCAL_PLAN_DIR / 'arms.json').read_text())
        document['models']['olmoe']['native_blocks_at_16gib'] = 16383
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'arms.json'
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                plan.load_arms(path)

    def test_arithmetic_drift_detected_despite_matching_sha(self):
        source = json.loads((plan.LOCAL_PLAN_DIR / 'arms.json').read_text())
        variants = []
        broken = json.loads(json.dumps(source))
        broken['workpackages']['WP-B']['arms'][1]['kv_gib'] = 11.0  # sum != 16
        variants.append(broken)
        broken = json.loads(json.dumps(source))
        broken['workpackages']['WP-B']['capacity_frontier']['ladder']['concurrency_cap']['4096'] = 1024
        variants.append(broken)
        broken = json.loads(json.dumps(source))
        broken['model_priority'] = ['gpt_oss', 'olmoe', 'qwen3']
        variants.append(broken)
        broken = json.loads(json.dumps(source))
        broken['trim_ladder_preregistered'] = broken['trim_ladder_preregistered'][::-1]
        variants.append(broken)
        broken = json.loads(json.dumps(source))
        broken['budget_summary_minutes']['core'] = 1
        variants.append(broken)
        broken = json.loads(json.dumps(source))
        broken['workpackages']['WP-B']['throughput_frontier']['cells']['output_tokens'] = [1, 2]
        variants.append(broken)
        for variant in variants:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'arms.json'
                payload = json.dumps(variant)
                path.write_text(payload)
                original = plan.ARMS_SHA256
                plan.ARMS_SHA256 = hashlib.sha256(payload.encode()).hexdigest()
                try:
                    with self.assertRaises(ValueError):
                        plan.load_arms(path)
                finally:
                    plan.ARMS_SHA256 = original


class ConfigTests(unittest.TestCase):
    def test_kv_heap_resolution_per_job(self):
        self.assertEqual(plan.kv_heap_for('A1'), {'kv_gib': 16.0, 'heap_gib': 0.0,
                                                  'slot_L_mib': None})
        self.assertEqual(plan.kv_heap_for('A2'), {'kv_gib': 12.0, 'heap_gib': 4.0,
                                                  'slot_L_mib': 96})
        self.assertEqual(plan.kv_heap_for('A4'), {'kv_gib': 7.5, 'heap_gib': 8.5,
                                                  'slot_L_mib': 256})
        self.assertEqual(plan.kv_heap_for('C2BA1')['kv_gib'], 16.0)
        self.assertEqual(plan.kv_heap_for('C2BC1'), {'kv_gib': 12.75, 'heap_gib': 3.25,
                                                     'slot_L_mib': 96})
        self.assertIsNone(plan.kv_heap_for('C01'))

    def test_config_alignment_leg(self):
        for arm in ('B-A', 'B-B', 'B-C', 'B-D'):
            row = plan.WP_B_ARMS[arm]
            self.assertAlmostEqual(row['kv_gib'] + row['heap_gib'], 16.0, places=9)

    def test_block_arithmetic(self):
        self.assertEqual(plan.blocks_for(16.0), 16384)
        self.assertEqual(plan.blocks_for(12.0), 12288)
        self.assertEqual(plan.blocks_for(12.75), 13056)
        self.assertEqual(plan.storage_blocks(2048, 1024), 2048 * 8 + 1)
        self.assertEqual(plan.storage_blocks(512, 4096), 512 * 32 + 1)

    def test_a6_floor_rule(self):
        feasible = plan.a6_kv_floor(40.0)
        self.assertEqual(feasible['kv_gib'], 21.0)
        self.assertTrue(feasible['feasible'])
        tight = plan.a6_kv_floor(36.6)
        self.assertEqual(tight['kv_gib'], 18.0)
        infeasible = plan.a6_kv_floor(19.0)
        self.assertFalse(infeasible['feasible'])
        self.assertIn('infeasible', infeasible['infeasibility'])
        with self.assertRaises(ValueError):
            plan.a6_kv_floor(None)
        a6 = plan.kv_heap_for('A6', a5_free_t1_gib=37.0)
        self.assertEqual(a6['kv_gib'], 18.0)
        self.assertEqual(a6['slot_L_mib'], 512)

    def test_llm_config_template_and_gpt_overrides(self):
        base = plan.llm_config('olmoe', 16.0, 2048, 16, 1)
        self.assertEqual((base['tensor_parallel_size'], base['enable_expert_parallel'],
                          base['seed'], base['block_size']), (16, True, 2027, 128))
        self.assertEqual(base['kv_cache_memory_bytes'], 16 * plan.GIB)
        self.assertIsNone(base['num_gpu_blocks_override'])
        gpt = plan.llm_config('gpt_oss', 2.0, 4096, 64, 1)
        self.assertEqual((gpt['tensor_parallel_size'], gpt['data_parallel_size']), (8, 2))
        with self.assertRaises(ValueError):
            plan.llm_config('unknown', 16.0, 2048, 16, 1)
        with self.assertRaises(ValueError):
            plan.llm_config('olmoe', None, 2048, 16, 1)
        with self.assertRaises(ValueError):
            plan.llm_config('olmoe', 16.0, 8192, 16, 1)
        with self.assertRaises(ValueError):
            plan.llm_config('olmoe', 16.0, 2048, 8, 1)
        with self.assertRaises(ValueError):
            plan.llm_config('olmoe', 16.0, 2048, 16, 7)


class LadderTests(unittest.TestCase):
    def test_ladder_concurrency(self):
        self.assertEqual(plan.ladder_concurrency(1024), [16, 32, 64, 128, 256, 512, 1024, 2048])
        self.assertEqual(plan.ladder_concurrency(4096), [16, 32, 64, 128, 256, 512])
        with self.assertRaises(ValueError):
            plan.ladder_concurrency(8192)

    def test_next_cell_doubling_then_binary_with_depth_cap(self):
        history = []
        sequence = []
        for _ in range(10):
            cell = plan.next_capacity_cell(history, 4096)
            if cell is None:
                break
            sequence.append(cell)
            history.append((cell, 'served' if cell < 300 else 'exhausted'))
        # doubling 16..256 served, 512 fails, then refine 384 (fails): the
        # preregistered depth (4 doubling + 3 binary = 7 probes) is exhausted.
        self.assertEqual(sequence, [16, 32, 64, 128, 256, 512, 384])
        self.assertEqual(len(history), plan.SEARCH_PROBES_PER_ARM_L + plan.BINARY_DEPTH_MAX)
        self.assertIsNone(plan.next_capacity_cell(history, 4096))
        self.assertEqual(plan.c_star(history), 256)

    def test_next_cell_first_failure_and_refinement(self):
        history = [(16, 'served'), (32, 'served'), (64, 'exhausted')]
        self.assertEqual(plan.next_capacity_cell(history, 4096), 48)
        history.append((48, 'served'))
        self.assertEqual(plan.next_capacity_cell(history, 4096), 56)
        history.append((56, 'exhausted'))
        self.assertEqual(plan.next_capacity_cell(history, 4096), 52)
        history.append((52, 'exhausted'))
        self.assertEqual(plan.next_capacity_cell(history, 4096), 50)  # 7th and last probe
        history.append((50, 'exhausted'))
        self.assertIsNone(plan.next_capacity_cell(history, 4096))  # depth cap
        self.assertEqual(plan.c_star(history), 48)

    def test_first_cell_failure_ends_search(self):
        self.assertIsNone(plan.next_capacity_cell([(16, 'exhausted')], 4096))

    def test_whole_ladder_served(self):
        history = [(C, 'served') for C in plan.ladder_concurrency(4096)]
        self.assertIsNone(plan.next_capacity_cell(history, 4096))

    def test_classify_probe_outcomes(self):
        self.assertEqual(plan.classify_probe({'completed': 64, 'submitted': 64}), 'served')
        self.assertEqual(plan.classify_probe(
            {'completed': 10, 'submitted': 64, 'insufficient_kv_streak': 150,
             'median_completion_s': 10.0, 'wall_s': 100.0}), 'exhausted')
        self.assertEqual(plan.classify_probe(
            {'completed': 10, 'submitted': 64, 'insufficient_kv_streak': 5,
             'median_completion_s': 10.0, 'wall_s': 100.0}), 'censored')
        self.assertEqual(plan.classify_probe({'fault': True, 'completed': 0,
                                              'submitted': 64}), 'fault')
        self.assertEqual(plan.classify_probe(
            {'completed': 5, 'submitted': 64, 'insufficient_kv_streak': 200,
             'median_completion_s': 10.0, 'wall_s': 20.0}), 'censored')
        with self.assertRaises(ValueError):
            plan.classify_probe({'completed': 'x', 'submitted': 64})


class LensRuleTests(unittest.TestCase):
    def pilot(self, factor=1.0, free_delta=0):
        base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        times_on = [value * factor for value in base]
        fence_off = {'T1': 40 * plan.GIB, 'T3': 38 * plan.GIB}
        fence_on = {name: value - free_delta for name, value in fence_off.items()}
        return {'request_times': times_on, 'fencepost_free_bytes': fence_on}, \
               {'request_times': base, 'fencepost_free_bytes': fence_off}

    def test_l3_adoption_rule(self):
        on, off = self.pilot(factor=1.01, free_delta=0)
        self.assertTrue(plan.l3_adoption(on, off)['adopt'])
        on, off = self.pilot(factor=1.03, free_delta=0)
        self.assertFalse(plan.l3_adoption(on, off)['adopt'])  # 3% > 2%
        on, off = self.pilot(factor=1.01, free_delta=plan.L3_ADOPTION_MAX_FENCEPOST_DELTA)
        self.assertTrue(plan.l3_adoption(on, off)['adopt'])
        on, off = self.pilot(factor=1.01, free_delta=plan.L3_ADOPTION_MAX_FENCEPOST_DELTA + 1)
        result = plan.l3_adoption(on, off)
        self.assertFalse(result['adopt'])
        self.assertGreater(result['max_fencepost_free_delta_bytes'],
                           plan.L3_ADOPTION_MAX_FENCEPOST_DELTA)
        with self.assertRaises(ValueError):
            plan.l3_adoption({'request_times': [], 'fencepost_free_bytes': {}},
                             {'request_times': [], 'fencepost_free_bytes': {}})

    def test_reconciliation_bias(self):
        total = 64 * plan.GIB
        l1 = [{'die': 0, 'used_bytes': 30 * plan.GIB, 'mono_ns': 1_000_000_000}]
        l2 = [{'free_bytes': total - 28 * plan.GIB, 'mono_ns': 1_000_500_000}]
        result = plan.reconciliation_bias(l1, l2, total_bytes=total)
        self.assertEqual(result['pairs'], 1)
        self.assertEqual(result['bias_summary_bytes']['median'], 2 * plan.GIB)
        l2_far = [{'free_bytes': total - 28 * plan.GIB, 'mono_ns': 9_000_000_000}]
        self.assertEqual(plan.reconciliation_bias(l1, l2_far)['pairs'], 0)

    def test_peak_alignment_epsilon_and_wording(self):
        total = 64 * plan.GIB
        native = {die: 40 * plan.GIB + die * plan.MIB for die in range(16)}
        pace = {die: 40 * plan.GIB + die * plan.MIB + 500 * plan.MIB for die in range(16)}
        self.assertLessEqual(plan.EPSILON_LEG2, 2 * 0.02 * 64 * plan.GIB)
        aligned = plan.peak_alignment(native, pace)
        self.assertTrue(aligned['aligned'])
        self.assertIn('matched configured budget', aligned['wording'])
        pace_far = {die: value + 4 * plan.GIB for die, value in pace.items()}
        degraded = plan.peak_alignment(native, pace_far)
        self.assertFalse(degraded['aligned'])
        self.assertIn('reported separately', degraded['wording'])
        with self.assertRaises(ValueError):
            plan.peak_alignment(native, pace, adjustment_rounds=2)

    def test_fill_probe_and_duration_gate(self):
        result = plan.fill_probe_result(18 * plan.GIB, 72, 2 * plan.GIB)
        self.assertTrue(result['floor_reached_within_one_step'])
        self.assertEqual(plan.fill_probe_result(0, 0, 3 * plan.GIB)['steps'], 0)
        gate = plan.duration_gate('A1', 500)
        self.assertFalse(gate['warning'])
        self.assertTrue(plan.duration_gate('A1', 8 * 60 * 1.26)['warning'])
        self.assertEqual(plan.duration_gate('C01', 10 ** 6)['job'], 'C01')


class L1PollerTests(unittest.TestCase):
    def test_parse_l1_usage_on_observed_fixture(self):
        rows = plan.parse_l1_usage(fixture)
        self.assertEqual(sorted(rows), list(range(16)))
        self.assertEqual(rows[0]['total_bytes'], 65536 * plan.MIB)
        self.assertEqual(rows[0]['used_bytes'], 3142 * plan.MIB)
        self.assertTrue({row['chip'] for row in rows.values()} <= {0, 1})

    def test_parse_rejects_incomplete_snapshots(self):
        lines = [line for line in fixture.splitlines()]
        missing = '\n'.join(line for line in lines
                            if not line.strip().startswith('| 0     0  '))
        with self.assertRaises(ValueError):
            plan.parse_l1_usage(missing)
        with self.assertRaises(ValueError):
            plan.parse_l1_usage('')

    def test_poller_stream_valid(self):
        rows = plan.parse_l1_usage(fixture)
        clock_values = iter([10 ** 12, 10 ** 12 + 10 ** 9, 10 ** 12 + 2 * 10 ** 9])
        samples = []
        for _ in range(3):
            samples.extend(plan.l1_poller_record(rows, next(clock_values), None))
        stream = plan.validate_l1_stream(samples)
        self.assertEqual(stream['dies'], list(range(16)))
        self.assertEqual(stream['samples'], 48)
        self.assertEqual(stream['max_used_bytes'], max(row['used_bytes'] for row in rows.values()))
        broken = [dict(row) for row in samples]
        broken[17]['used_bytes'] = broken[17]['total_bytes'] + 1
        with self.assertRaises(ValueError):
            plan.validate_l1_stream(broken)
        nonmonotone = [dict(row) for row in samples]
        nonmonotone[17]['mono_ns'] = 0
        with self.assertRaises(ValueError):
            plan.validate_l1_stream(nonmonotone)

    def test_launch_poller_loop_stops_on_dead_container(self):
        calls = {'npu': 0, 'running': 0}

        def runner():
            calls['npu'] += 1
            return fixture

        def running():
            calls['running'] += 1
            return calls['running'] < 3

        samples, errors = launch.l1_poller(runner, running, interval_s=0,
                                           clock=lambda: 10 ** 12, sleep=lambda s: None)
        self.assertEqual(errors, 0)
        self.assertEqual(len(samples), 16 * 3)

        def broken_runner():
            raise RuntimeError('npu-smi unavailable')

        samples, errors = launch.l1_poller(broken_runner, lambda: True, interval_s=0)
        self.assertEqual((samples, errors), ([], 1))


class FencepostTests(unittest.TestCase):
    def capture(self, free=40 * plan.GIB):
        return {'free_bytes': free, 'total_bytes': 64 * plan.GIB,
                'allocator_bytes': {f'{kind}.all.{state}': 1024
                                    for kind in ('allocated_bytes', 'reserved_bytes',
                                                 'active_bytes', 'inactive_split_bytes')
                                    for state in ('current', 'peak')},
                'api_source_sha256': dict(plan.MEMORY_API_SOURCE_SHA256)}

    def fence(self, name):
        return {'lens': 'L2', 'fencepost': name, 'capture': self.capture()}

    def test_native_and_pace_sequences(self):
        native = ('T0', 'T1', 'T2', 'T3', 'T4', 'T5')
        pace = ('T0', 'T1', 'T2a', 'T2b', 'T3', 'T4', 'T5')
        self.assertEqual(plan.validate_fencepost_sequence(
            [self.fence(name) for name in native], 'A'), list(native))
        self.assertEqual(plan.validate_fencepost_sequence(
            [self.fence(name) for name in pace], 'P'), list(pace))

    def test_mutations_rejected(self):
        native = [self.fence(name) for name in ('T0', 'T1', 'T2', 'T3', 'T4', 'T5')]
        for mutate in ('order', 'missing', 'api', 'peak', 'lens'):
            rows = [dict(row) for row in native]
            if mutate == 'order':
                rows[2], rows[3] = rows[3], rows[2]
            elif mutate == 'missing':
                rows = rows[:-1]
            elif mutate == 'api':
                rows[1]['capture'] = {**rows[1]['capture'],
                                      'api_source_sha256': {'mem_get_info': 'x'}}
            elif mutate == 'peak':
                rows[0]['capture']['allocator_bytes']['allocated_bytes.all.peak'] = 0
            else:
                rows[0]['lens'] = 'L3'
            with self.assertRaises(ValueError, msg=mutate):
                plan.validate_fencepost_sequence(rows, 'A')

    def test_l2_fencepost_free_accessor(self):
        row = self.fence('T1')
        self.assertEqual(plan.l2_fencepost_free(row, 'T1'), 40 * plan.GIB)
        with self.assertRaises(ValueError):
            plan.l2_fencepost_free(row, 'T3')


class ReviewContractTests(unittest.TestCase):
    def test_launch_consumes_the_single_shared_constant(self):
        self.assertIs(launch.verify_review_binding, plan.validate_review)

    def test_synthetic_approved_fixture_is_accepted(self):
        binding = plan.validate_review(approved_review(), 'a' * 64)
        self.assertEqual(binding['status'], 'APPROVED')
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
        with self.assertRaises(ValueError):
            plan.validate_review(approved_review(manifest_sha='b' * 64), 'a' * 64)

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
        review['groups'] = {'campaign_order': list(plan.CAMPAIGN_ORDER)[:44]}
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)
        review = approved_review()
        review['groups']['conditional']['a6_feasible'] = ['C7GA']
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_wrong_arms_binding_refused(self):
        review = approved_review()
        review['arms_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)

    def test_unverified_guard_semantics_refused(self):
        review = approved_review()
        review['device_split_guard_semantics'] = 'unverified'
        with self.assertRaises(ValueError):
            plan.validate_review(review, 'a' * 64)


class WordingTests(unittest.TestCase):
    def test_c0_outcome_fail_closed_semantics(self):
        no_cap = plan.c0_outcome('C0-1', over_allocation_failed=False, error_recorded=False)
        self.assertFalse(no_cap['enforced'])
        failed_unrecorded = plan.c0_outcome('C0-1', over_allocation_failed=True,
                                            error_recorded=False)
        self.assertFalse(failed_unrecorded['enforced'])
        enforced = plan.c0_outcome('C0-2', over_allocation_failed=True, error_recorded=True)
        self.assertTrue(enforced['enforced'])
        with self.assertRaises(ValueError):
            plan.c0_outcome('C0-3', False, False)

    def test_equal_memory_wording_switch(self):
        tier1 = plan.equal_memory_wording([{'enforced': False}, {'enforced': False}])
        self.assertIn('matched configured budget', tier1['wording'])
        enforced = plan.equal_memory_wording([{'enforced': True}])
        self.assertEqual(enforced['wording'], 'enforced equal ceiling')
        campaign = plan.campaign_wording([{'enforced': False}])
        self.assertIn('sampled maximum (lower bound)', campaign['sheet']['peak'])
        self.assertIn('never stated as a percent loss', campaign['sheet']['blocks'])


class LaunchContractTests(unittest.TestCase):
    def test_entry_scripts_are_pinned(self):
        self.assertIn('/workspace/lane/supervisor.py', SUPERVISOR_SCRIPT)
        self.assertIn('/workspace/plan', SUPERVISOR_SCRIPT)
        self.assertIn('/workspace/lane/precheck.py', PRECHECK_SCRIPT)

    def test_inspect_contract_accepts_engine_job(self):
        devices = list(guard.nodes(16))
        env = base_env('A2', '0')
        payload = Path('/results/host/payload')
        mounts = launch.expected_mounts(payload)
        info = container_info(devices, mounts, env, payload)
        argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', SUPERVISOR_SCRIPT]
        launch.inspect_contract(info, argv, env, guard, 16, payload)

    def test_inspect_contract_accepts_c0_memory_limit_container(self):
        env = base_env('C01', '0')
        payload = Path('/results/host/payload')
        info = container_info(list(guard.nodes(16)), launch.expected_mounts(payload), env,
                              payload, memory=2 * 1024 ** 3)
        argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', SUPERVISOR_SCRIPT]
        launch.inspect_contract(info, argv, env, guard, 16, payload,
                                memory_limit=plan.C0_CGROUP_MEMORY_LIMIT)

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
        bad_env = dict(broken)
        bad_env['Config'] = dict(broken['Config'], Env=['MEMCAMP_L3=maybe'])
        cases.append(('environment', bad_env))
        running = dict(broken)
        running['State'] = dict(broken['State'], Running=True)
        cases.append(('still running', running))
        c0_without_limit = dict(broken)
        cases.append(('c0 memory limit mismatch',
                      {**broken, 'HostConfig': dict(broken['HostConfig'], Memory=0)}))
        for label, info in cases:
            with self.assertRaises(RuntimeError, msg=label):
                if label == 'c0 memory limit mismatch':
                    launch.inspect_contract(info, argv, env, guard, 16, payload,
                                            memory_limit=plan.C0_CGROUP_MEMORY_LIMIT)
                else:
                    launch.inspect_contract(info, argv, env, guard, 16, payload)


class ConditionalRouteTests(unittest.TestCase):
    """B2: conditional jobs route run/projection-only instead of deadlocking."""

    def test_per_peer_route(self):
        self.assertEqual(launch.conditional_route('C3BD1', 'per_peer'), 'run')
        self.assertEqual(launch.conditional_route('C3BD2', 'whole_input'), 'projection_only')
        self.assertEqual(launch.conditional_route('C3BD1', 'union'), 'projection_only')
        self.assertEqual(launch.conditional_route('A1', 'whole_input'), 'run')

    def test_a6_gate_is_hard(self):
        self.assertEqual(launch.conditional_route('C7GA', 'per_peer'), 'gate')
        self.assertEqual(launch.conditional_route('C7SB', 'whole_input'), 'gate')

    def test_projection_receipt_satisfies_ordering(self):
        receipt = {'job': 'C3BD1', 'manifest_sha256': 'a' * 64, 'review_sha256': 'b' * 64,
                   'status': 'PROJECTION_ONLY',
                   'projection_reason': 'per_peer_guard_semantics',
                   'reviewed_semantics': 'whole_input'}
        self.assertTrue(launch.prior_receipt_ok(receipt, 'C3BD1', 'a' * 64, 'b' * 64))
        # wrong job binding / manifests refuse
        self.assertFalse(launch.prior_receipt_ok(receipt, 'C3BD2', 'a' * 64, 'b' * 64))
        self.assertFalse(launch.prior_receipt_ok(receipt, 'C3BD1', 'c' * 64, 'b' * 64))
        # projection without the semantics justification refuses
        bad = dict(receipt, reviewed_semantics='per_peer')
        self.assertFalse(launch.prior_receipt_ok(bad, 'C3BD1', 'a' * 64, 'b' * 64))
        bad = dict(receipt, projection_reason='other')
        self.assertFalse(launch.prior_receipt_ok(bad, 'C3BD1', 'a' * 64, 'b' * 64))
        # PASS receipts still satisfy; unknown statuses refuse
        ok = dict(receipt, status='PASS')
        self.assertTrue(launch.prior_receipt_ok(ok, 'C3BD1', 'a' * 64, 'b' * 64))
        bad = dict(receipt, status='FAILED')
        self.assertFalse(launch.prior_receipt_ok(bad, 'C3BD1', 'a' * 64, 'b' * 64))


class CLIDispatchTests(unittest.TestCase):
    """B1: every CLI action reaches its implementation with the exact
    signature the implementation declares (the round-1 TypeError class)."""

    def args(self, action, **overrides):
        base = {'action': action, 'job': None, 'trim': 0, 'l3': False,
                'cell_L': None, 'cell_C': None, 'cell_O': None,
                'manifest_sha': 'a' * 64, 'review_sha': 'b' * 64}
        base.update(overrides)
        return argparse_namespace(**base)

    def test_dispatch_signatures(self):
        calls = {'job_start': []}

        def fake_job_start(job, trim, manifest, review, cell, l3):
            calls['job_start'].append((job, trim, manifest, review, cell, l3))

        def fake_job_audit(job, trim, manifest, review):
            calls['job_audit'] = (job, trim, manifest, review)

        def fake_job_l1(job, manifest, review):
            calls['job_l1'] = (job, manifest, review)

        def fake_precheck(manifest, review):
            calls['precheck_arm'] = (manifest, review)

        def fake_aggregate(manifest, review, trim=0):
            calls['aggregate'] = (manifest, review, trim)

        saved = {name: getattr(launch, name) for name in
                 ('job_start', 'job_audit', 'job_l1', 'precheck_arm', 'precheck_audit',
                  'aggregate')}
        launch.job_start, launch.job_audit, launch.job_l1 = fake_job_start, fake_job_audit, fake_job_l1
        launch.precheck_arm, launch.precheck_audit = fake_precheck, fake_precheck
        launch.aggregate = fake_aggregate
        try:
            launch.dispatch(self.args('job_start', job='A1'))
            launch.dispatch(self.args('job_audit', job='A1', trim=2))
            launch.dispatch(self.args('job_l1', job='A1'))
            launch.dispatch(self.args('precheck_arm'))
            launch.dispatch(self.args('aggregate', trim=3))
            launch.dispatch(self.args('job_start', job='C5BA', cell_C=64))
        finally:
            for name, value in saved.items():
                setattr(launch, name, value)
        first_call = {'job': 'A1', 'cell': None}
        self.assertEqual(calls['job_start'][0][:4], ('A1', 0, 'a' * 64, 'b' * 64))
        self.assertIsNone(calls['job_start'][0][4])
        self.assertEqual(calls['job_audit'], ('A1', 2, 'a' * 64, 'b' * 64))
        self.assertEqual(calls['job_l1'], ('A1', 'a' * 64, 'b' * 64))
        self.assertEqual(calls['aggregate'], ('a' * 64, 'b' * 64, 3))
        second = dict(zip(('job', 'trim', 'manifest', 'review', 'cell', 'l3'),
                          calls['job_start'][1]))
        self.assertEqual(second['job'], 'C5BA')
        self.assertEqual(second['cell'], {'L': 2048, 'C': 64, 'O': 1})

    def test_parser_accepts_job_l1_action(self):
        parsed = launch.build_parser().parse_args(
            ['job_l1', '--job', 'A2', '--manifest-sha', 'a' * 64, '--review-sha', 'b' * 64])
        self.assertEqual((parsed.action, parsed.job), ('job_l1', 'A2'))


def argparse_namespace(**kwargs):
    import argparse
    return argparse.Namespace(**kwargs)


class ReceiptLiftTests(unittest.TestCase):
    """B3 + B11: producers and consumers of the A6 floor / A5 T1 free."""

    def job_payload(self, files):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        for name, value in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        return root

    def test_lift_kv_floor(self):
        floor = {'kv_gib': 18.0, 'feasible': True, 'infeasibility': None}
        self.assertEqual(launch.lift_kv_floor(self.job_payload(
            {'a6_floor.json': {'job': 'A6', 'kv_floor': floor}})), floor)
        infeasible = {'kv_gib': 0.0, 'feasible': False, 'infeasibility': 'infeasible...'}
        self.assertEqual(launch.lift_kv_floor(self.job_payload(
            {'a6_infeasible.json': {'kv_floor': infeasible}})), infeasible)
        self.assertIsNone(launch.lift_kv_floor(self.job_payload({})))

    def test_extract_free_t1_gib(self):
        fence = [{'rank': 0, 'capture': {'free_bytes': 40 * plan.GIB}},
                 {'rank': 1, 'capture': {'free_bytes': 39 * plan.GIB}}]
        self.assertEqual(launch.extract_free_t1_gib(
            self.job_payload({'fence_T1.json': fence})), 40.0)
        self.assertIsNone(launch.extract_free_t1_gib(self.job_payload({})))
        with self.assertRaises(ValueError):
            launch.extract_free_t1_gib(self.job_payload(
                {'fence_T1.json': [{'rank': 1, 'capture': {'free_bytes': 1}}]}))


class MountsTests(unittest.TestCase):
    """B4: engine jobs mount model/legacy/sync-model/reference."""

    def test_engine_mounts_present(self):
        mounts = launch.expected_mounts(Path('/results/host/payload'))
        for target in ('/workspace/model', '/workspace/legacy', '/workspace/model_hook',
                       '/workspace/reference', '/workspace/first_memory',
                       '/workspace/budget_matched', '/workspace/memory_snapshot'):
            self.assertIn(target, mounts, target)
        self.assertFalse(all(rw for _s, rw in mounts.values()))

    def test_pythonpath_carries_model_hook_and_legacy(self):
        script = launch.entry_script('supervisor.py')
        self.assertIn('/workspace/model_hook', script)
        self.assertIn('/workspace/legacy', script)


class JobEnvTests(unittest.TestCase):
    """B5: heap/slot plumbed per the arm contract; B11: A6 floor input."""

    def test_pace_heap_slot_bytes(self):
        cases = {
            'A2': {'kv_gib': 12.0, 'heap_gib': 4.0, 'slot_L_mib': 96},
            'C2BB1': {'kv_gib': 12.0, 'heap_gib': 4.0, 'slot_L_mib': 96},
            'C2BC1': {'kv_gib': 12.75, 'heap_gib': 3.25, 'slot_L_mib': 96},
            'C3BD1': {'kv_gib': 15.25, 'heap_gib': 0.75, 'slot_L_mib': 16},
        }
        for job, kv_heap in cases.items():
            env = launch.job_env(job, kv_heap)
            self.assertEqual(env['HPA_SHMEM_HEAP_BYTES'],
                             str(int(kv_heap['heap_gib'] * plan.GIB)), job)
            self.assertEqual(env['HPA_SHMEM_SLOT_BYTES'],
                             str(int(kv_heap['slot_L_mib'] * plan.MIB)), job)
            self.assertEqual(env['REV2_BACKEND'], 'sync-diagnostic', job)

    def test_native_arm_has_no_pace_heap(self):
        env = launch.job_env('A1', {'kv_gib': 16.0, 'heap_gib': 0.0, 'slot_L_mib': None})
        self.assertEqual(env['REV2_BACKEND'], 'hccl')
        self.assertNotIn('HPA_SHMEM_HEAP_BYTES', env)
        self.assertNotIn('HPA_SHMEM_SLOT_BYTES', env)

    def test_a6_floor_input_env(self):
        kv_heap = plan.kv_heap_for('A6', a5_free_t1_gib=40.0)
        env = launch.job_env('A6', kv_heap, a5_free_t1_gib=40.0)
        self.assertEqual(env['MEMCAMP_A6_FREE_T1_GIB'], '40.0')
        self.assertEqual(env['HPA_SHMEM_HEAP_BYTES'], str(int(16.5 * plan.GIB)))
        env_no_input = launch.job_env('A6', kv_heap)
        self.assertNotIn('MEMCAMP_A6_FREE_T1_GIB', env_no_input)


class L1AuditTests(unittest.TestCase):
    """B6: the L1 stream is required, validated evidence at audit."""

    def stream_dir(self, samples):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        if samples is not None:
            (root / 'l1_stream.json').write_text(json.dumps(
                {'schema': 'revision2.memory-campaign-l1-stream.v1', 'lens': 'L1',
                 'samples': samples, 'sampler_errors': 0}))
        return root

    def samples(self, count=2):
        rows = []
        for tick in range(count):
            for die in range(16):
                rows.append({'schema': 'revision2.memory-campaign-l1-sample.v1',
                             'lens': 'L1', 'die': die, 'used_bytes': 30 * plan.GIB,
                             'free_bytes': 34 * plan.GIB, 'total_bytes': 64 * plan.GIB,
                             'mono_ns': 10 ** 12 + tick * 10 ** 9, 'utc': None})
        return rows

    def test_missing_stream_fails_audit_helper(self):
        with self.assertRaises(RuntimeError):
            launch.audit_l1_stream(self.stream_dir(None))

    def test_valid_stream_summarized(self):
        summary = launch.audit_l1_stream(self.stream_dir(self.samples()))
        self.assertEqual(summary['dies'], list(range(16)))
        self.assertEqual(summary['max_used_bytes'], 30 * plan.GIB)


class SourceWiringTests(unittest.TestCase):
    """Lock the round-2 producer wiring in worker.py against silent removal
    (the worker itself imports torch and cannot run on this host)."""

    def worker_source(self):
        return (Path(__file__).resolve().parent / 'worker.py').read_text()

    def test_b7_pilot_and_sampler_wiring(self):
        source = self.worker_source()
        self.assertIn('Lens3Sampler(torch.npu, 0, 1.0)', source)      # instantiated
        self.assertIn('sampler.start()', source)
        self.assertIn("plan.l3_adoption(", source)                    # called device-side
        self.assertIn("'P1'", source)                                 # pass tags
        self.assertIn("'P2'", source)
        self.assertIn("'P1END'", source)                              # R2-A5 post-pass pairing
        self.assertIn("'P2END'", source)
        self.assertIn('pilot.json', source)

    def test_b9_token_gate_wiring(self):
        source = self.worker_source()
        self.assertIn('token_gate(directory', source)
        self.assertIn('token_gate.json', source)
        self.assertIn('reference_tokens', source)
        self.assertIn('intra_job_determinism', source)

    def test_b8_engine_records_and_streak_policy(self):
        source = self.worker_source()
        self.assertIn('engine_records(llm)', source)
        self.assertIn('engine_runtime.json', source)
        self.assertNotIn("os.environ['MEMCAMP_KV_STREAK']", source)  # never env
        self.assertNotIn("os.environ.get('MEMCAMP_KV_STREAK'", source)
        self.assertIn("'insufficient_kv_streak': 0,  # lifted from the log at audit", source)

    def test_b5_a6_and_heap_echo_wiring(self):
        source = self.worker_source()
        self.assertIn('configured_pace_bytes()', source)
        self.assertIn('a6_floor.json', source)
        self.assertIn('a6_infeasible.json', source)
        self.assertIn("fence_T0.json", source)                       # A4: T0 captured

    def test_a1_confirm_runs_real_cell(self):
        source = self.worker_source()
        self.assertIn("role == 'throughput_confirm'", source)
        self.assertIn('throughput_confirm.json', source)

    def test_launch_wiring_of_plan_functions(self):
        source = (Path(__file__).resolve().parent / 'launch.py').read_text()
        for symbol in ('plan.next_capacity_cell', 'plan.c_star', 'plan.peak_alignment',
                       'plan.reconciliation_bias', 'plan.validate_fencepost_sequence',
                       'plan.kv_streak_from_log', 'plan.summary'):
            self.assertIn(symbol + '(', source, symbol)


class StreakTests(unittest.TestCase):
    """B8: the KV-exhaustion streak is lifted from the engine log."""

    def test_consecutive_pattern_lines(self):
        text = '\n'.join(['ok'] + ['WARNING: insufficient KV cache blocks for admission'] * 150
                         + ['ok', 'not enough KV blocks again', 'done'])
        self.assertEqual(plan.kv_streak_from_log(text), 150)
        self.assertEqual(plan.kv_streak_from_log('nothing here'), 0)
        self.assertEqual(plan.kv_streak_from_log(
            'insufficient KV\nnoise\ninsufficient KV\ninsufficient kv'), 2)
        with self.assertRaises(ValueError):
            plan.kv_streak_from_log(None)


class TokenGateTests(unittest.TestCase):
    """B9: the FA4 verdict math."""

    def test_verdicts(self):
        match = plan.token_gate_verdict([1, 2, 3], [1, 2, 3])
        self.assertTrue(match['match'])
        mismatch = plan.token_gate_verdict([1, 2, 3], [1, 2, 4])
        self.assertFalse(mismatch['match'])
        self.assertEqual(mismatch['observed_len'], 3)
        with self.assertRaises(ValueError):
            plan.token_gate_verdict([1], None)
        with self.assertRaises(ValueError):
            plan.token_gate_verdict(None, [1])


class CellDerivationTests(unittest.TestCase):
    """B10: capacity cells derive deterministically from prior receipts."""

    def receipt(self, job, C, classification):
        return {'job': job, 'status': 'PASS',
                'capacity_probe': {'cell': {'C': C, 'L': 4096, 'max_tokens': 1},
                                   'classification': classification}}

    def test_search_slots_follow_next_capacity_cell(self):
        self.assertEqual(launch.derive_capacity_cell('C2BA1', []),
                         {'L': 4096, 'C': 16, 'O': 1})
        history = [self.receipt('C2BA1', 16, 'served')]
        self.assertEqual(launch.derive_capacity_cell('C2BA2', history)['C'], 32)
        history.append(self.receipt('C2BA2', 32, 'served'))
        history.append(self.receipt('C2BA3', 64, 'exhausted'))
        self.assertEqual(launch.derive_capacity_cell('C2BA4', history)['C'], 48)
        # search complete (whole ladder served): a further search slot refuses
        served_all = [self.receipt('C2BA1', C, 'served')
                      for C in plan.ladder_concurrency(4096)]
        with self.assertRaises(ValueError):
            launch.derive_capacity_cell('C2BA4', served_all)

    def test_confirm_slot_uses_c_star_at_l1024(self):
        history = [self.receipt('C2BB1', 16, 'served'),
                   self.receipt('C2BB2', 32, 'served'),
                   self.receipt('C2BB3', 64, 'exhausted'),
                   self.receipt('C2BB4', 48, 'exhausted')]
        self.assertEqual(launch.derive_capacity_cell('C2BB5', history),
                         {'L': 1024, 'C': 32, 'O': 1})
        with self.assertRaises(ValueError):
            launch.derive_capacity_cell('C2BB5', [])  # no served cell yet

    def test_non_capacity_roles_have_no_cell(self):
        self.assertIsNone(launch.derive_capacity_cell('A1', []))
        self.assertIsNone(launch.derive_capacity_cell('C4BA', []))


class AggregateAnalysesTests(unittest.TestCase):
    """B10: c_star / peak alignment / duration summary wired into aggregate."""

    def completion(self, job, arm, duration, l1=None, probe=None):
        return {'job': job, 'status': 'PASS', 'duration_seconds': duration,
                'duration_gate': {'warning': False}, 'arm': arm,
                'l1_max_used_per_die': l1, 'capacity_probe': probe}

    def test_c_star_and_alignment(self):
        native_l1 = [40 * plan.GIB] * 16
        pace_l1 = [40 * plan.GIB + 512 * plan.MIB] * 16
        completions = {
            'C2BA1': self.completion('C2BA1', 'B-A', 400.0, native_l1,
                                     {'cell': {'C': 16, 'L': 4096}, 'classification': 'served'}),
            'C2BA2': self.completion('C2BA2', 'B-A', 420.0, native_l1,
                                     {'cell': {'C': 32, 'L': 4096}, 'classification': 'served'}),
            'C2BB1': self.completion('C2BB1', 'B-B', 430.0, pace_l1,
                                     {'cell': {'C': 16, 'L': 4096}, 'classification': 'served'}),
            'C2BB2': self.completion('C2BB2', 'B-B', 440.0, pace_l1,
                                     {'cell': {'C': 32, 'L': 4096}, 'classification': 'exhausted'}),
            'A1': self.completion('A1', 'A', 500.0),
        }
        analyses = launch.aggregate_analyses(completions)
        self.assertEqual(analyses['capacity_frontier']['olmoe:B-A']['c_star'], 32)
        self.assertEqual(analyses['capacity_frontier']['olmoe:B-B']['c_star'], 16)
        self.assertTrue(analyses['peak_alignment_by_model']['olmoe']['aligned'])
        self.assertEqual(analyses['durations']['n'], 5)


class AnalysisScriptsTests(unittest.TestCase):
    """A7: the frozen analysis scripts run over synthetic receipts."""

    def results_tree(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        jobs = root / 'jobs'
        for job, row in (
                ('A1', {'status': 'PASS', 'quarantined': False, 'duration_seconds': 500.0,
                        'duration_gate': {'warning': False},
                        'l1_max_used_bytes': 30 * plan.GIB,
                        'l1_min_free_bytes': 34 * plan.GIB,
                        'reconciliation': {'bias_summary_bytes': {'median': 0}},
                        'engine_num_gpu_blocks': 16384}),
                ('A2', {'status': 'PASS', 'quarantined': False, 'duration_seconds': 540.0,
                        'duration_gate': {'warning': False},
                        'l1_max_used_bytes': 31 * plan.GIB,
                        'l1_min_free_bytes': 33 * plan.GIB,
                        'reconciliation': {}, 'engine_num_gpu_blocks': 12288,
                        'kv_floor': None})):
            directory = jobs / job
            directory.mkdir(parents=True)
            (directory / 'completion.json').write_text(json.dumps(row))
        return root

    def test_wp_a_profile_runs(self):
        import wp_a_profile
        root = self.results_tree()
        rows = wp_a_profile.load_completions(root)
        table = wp_a_profile.profile(rows)
        self.assertIn('olmoe', table)
        self.assertEqual(table['olmoe']['native_vs_pace_sampled_max_used_delta_bytes'],
                         1 * plan.GIB)
        self.assertIn('sampled maximum (lower bound)', json.dumps(table))
        # without the jobs root the SHMEM-init delta stays honestly absent
        self.assertIsNone(table['olmoe']['shmem_init_free_delta_bytes'])

    def test_wp_a_profile_computes_shmem_init_delta(self):
        # R2-A3: with the jobs root the headline T2a->T2b column is computed
        import wp_a_profile
        root = self.results_tree()
        payload = root / 'jobs' / 'A2' / 'payload'
        payload.mkdir(parents=True)
        (payload / 'fence_T2a.json').write_text(json.dumps(
            [{'rank': 0, 'capture': {'free_bytes': 40 * plan.GIB}}]))
        (payload / 'fence_T2b.json').write_text(json.dumps(
            [{'rank': 0, 'capture': {'free_bytes': 36 * plan.GIB}}]))
        rows = wp_a_profile.load_completions(root)
        table = wp_a_profile.profile(rows, jobs_root=root)
        self.assertEqual(table['olmoe']['shmem_init_free_delta_bytes'], 4 * plan.GIB)

    def test_wp_b_frontier_runs(self):
        import wp_b_frontier
        value = wp_b_frontier.blocks_table()
        self.assertEqual(value['B-A']['blocks_expected'], 16384)
        self.assertEqual(value['B-C']['blocks_expected'], 13056)
        self.assertTrue(value['B-D']['projection_only'])   # projection-only row
        self.assertEqual(value['B-D']['blocks_expected'], 15616)  # configured arithmetic still shown


class SourceTreeTests(unittest.TestCase):
    def make_tree(self, extra=(), stray_dir=None, symlink=None):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        for name in plan.FILES | {'source_manifest.json', 'review.json'}:
            (root / name).write_text('x')
        for name in extra:
            (root / name).write_text('x')
        if stray_dir is not None:
            (root / stray_dir).mkdir()
        (root / '__pycache__').mkdir()
        (root / '__pycache__' / 'plan.pyc').write_text('x')
        if symlink is not None:
            (root / symlink).symlink_to(root / 'plan.py')
        return root

    def test_exact_tree_with_pycache_passes(self):
        launch.check_source_tree(self.make_tree())

    def test_stray_file_dir_symlink_and_missing_rejected(self):
        with self.assertRaises(ValueError):
            launch.check_source_tree(self.make_tree(extra=('scratch.py',)))
        with self.assertRaises(ValueError):
            launch.check_source_tree(self.make_tree(stray_dir='scratch'))
        with self.assertRaises(ValueError):
            launch.check_source_tree(self.make_tree(symlink='plan_link.py'))
        root = self.make_tree()
        (root / 'worker.py').unlink()
        with self.assertRaises(ValueError):
            launch.check_source_tree(root)


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


class RoleAwareFencepostTests(unittest.TestCase):
    """R2-B2: the audit requires exactly the sequence each role emits
    (T4 is WP-A/fill/pilot only, PLAN section 1.4)."""

    def capture(self, free=40 * plan.GIB):
        return {'free_bytes': free, 'total_bytes': 64 * plan.GIB,
                'allocator_bytes': {f'{kind}.all.{state}': 1024
                                    for kind in ('allocated_bytes', 'reserved_bytes',
                                                 'active_bytes', 'inactive_split_bytes')
                                    for state in ('current', 'peak')},
                'api_source_sha256': dict(plan.MEMORY_API_SOURCE_SHA256)}

    def payload(self, names, arm):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        pace = arm in ('P', 'B-B', 'B-C', 'B-D', 'pace')
        for name in names:
            fence = 'T2a' if (name == 'T2' and pace) else name
            (root / f'fence_{fence}.json').write_text(json.dumps(
                [{'lens': 'L2', 'fencepost': fence, 'capture': self.capture()}]))
        return root

    def test_wp_a_sequences_still_required_in_full(self):
        for arm in ('A', 'P'):
            pace = arm == 'P'
            names = ['T0', 'T1'] + (['T2a', 'T2b'] if pace else ['T2']) + \
                ['T3', 'T4', 'T5']
            result = launch.audit_fenceposts(self.payload(names, arm), arm, 'wp_a')
            self.assertEqual(result, names)
            # missing T4 fails for WP-A
            without_t4 = [name for name in names if name != 'T4']
            with self.assertRaises(ValueError):
                launch.audit_fenceposts(self.payload(without_t4, arm), arm, 'wp_a')

    def test_capacity_role_sequence_accepted_without_t4(self):
        # the reviewer's simulated capacity sequence must be accepted
        result = launch.audit_fenceposts(self.payload(
            ['T0', 'T1', 'T2', 'T3', 'T5'], 'A'), 'A', 'capacity_search')
        self.assertEqual(result, ['T0', 'T1', 'T2', 'T3', 'T5'])
        result = launch.audit_fenceposts(self.payload(
            ['T0', 'T1', 'T2a', 'T2b', 'T3', 'T5'], 'P'), 'B-B', 'capacity_confirm')
        self.assertEqual(result, ['T0', 'T1', 'T2a', 'T2b', 'T3', 'T5'])
        # an unexpected T4 in a capacity job also fails (exact-sequence rule)
        with self.assertRaises(ValueError):
            launch.audit_fenceposts(self.payload(
                ['T0', 'T1', 'T2', 'T3', 'T4', 'T5'], 'A'), 'A', 'capacity_search')

    def test_throughput_and_fill_roles(self):
        self.assertEqual(launch.audit_fenceposts(self.payload(
            ['T0', 'T1', 'T2', 'T3', 'T5'], 'A'), 'A', 'throughput_sweep'),
            ['T0', 'T1', 'T2', 'T3', 'T5'])
        self.assertEqual(launch.audit_fenceposts(self.payload(
            ['T0', 'T1', 'T2', 'T3', 'T4', 'T5'], 'A'), 'A', 'fill_probe'),
            ['T0', 'T1', 'T2', 'T3', 'T4', 'T5'])


class CrossArmDerivationTests(unittest.TestCase):
    """R2-B3: derivation pools only the SAME arm's L=4096 search receipts."""

    def receipt(self, job, C, classification, L=4096):
        return {'job': job, 'status': 'PASS',
                'capacity_probe': {'cell': {'C': C, 'L': L, 'max_tokens': 1},
                                   'classification': classification}}

    def ba_history_complete(self):
        return [self.receipt('C2BA1', 16, 'served'),
                self.receipt('C2BA2', 32, 'served'),
                self.receipt('C2BA3', 64, 'exhausted'),
                self.receipt('C2BA4', 48, 'exhausted'),
                self.receipt('C2BA5', 32, 'served', L=1024)]  # confirm cell

    def test_foreign_arm_history_does_not_contaminate(self):
        # B-A's complete history (incl. its L=1024 confirm) is present, yet
        # C2BB1 (B-B, empty own history) must still derive C=16.
        self.assertEqual(launch.derive_capacity_cell('C2BB1', self.ba_history_complete()),
                         {'L': 4096, 'C': 16, 'O': 1})

    def test_own_history_drives_derivation(self):
        bb = self.ba_history_complete() + [self.receipt('C2BB1', 16, 'served')]
        self.assertEqual(launch.derive_capacity_cell('C2BB2', bb)['C'], 32)

    def test_confirm_cells_excluded_from_search_derivation(self):
        # L=1024 cells (any arm, including the same arm) never enter the
        # L=4096 search history.
        history = self.ba_history_complete()
        only_l1024 = [self.receipt('C2BA5', 2048, 'served', L=1024)]
        self.assertEqual(launch.same_arm_search_history(history + only_l1024, 'B-A'),
                         [(16, 'served'), (32, 'served'), (64, 'exhausted'),
                          (48, 'exhausted')])

    def test_confirm_c_star_uses_l4096_history_only(self):
        history = self.ba_history_complete()
        self.assertEqual(launch.derive_capacity_cell('C2BA5', history),
                         {'L': 1024, 'C': 32, 'O': 1})  # c* of the L=4096 probes

    def test_audit_rederivation_match_and_drift(self):
        history = self.ba_history_complete()
        self.assertEqual(launch.audit_rederive_cell('C2BB1', history,
                                                    {'L': 4096, 'C': 16, 'O': 1}),
                         {'L': 4096, 'C': 16, 'O': 1})
        with self.assertRaises(ValueError):
            launch.audit_rederive_cell('C2BB1', history, {'L': 4096, 'C': 96, 'O': 1})


class A6InfeasibleAuditTests(unittest.TestCase):
    """R2-B1: the preregistered infeasible outcome audits, gates and lists."""

    def payload(self, files):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        for name, value in files.items():
            (root / name).write_text(json.dumps(value))
        return root

    def fence_t0(self):
        return [{'lens': 'L2', 'fencepost': 'T0', 'rank': 0, 'capture': {}}]

    def test_infeasible_payload_audits_to_receipt(self):
        floor = {'kv_gib': 0.0, 'feasible': False, 'infeasibility': 'infeasible at '
                                                                    'Table 2 heap'}
        payload = self.payload({'fence_T0.json': self.fence_t0(),
                                'a6_infeasible.json': {'kv_floor': floor,
                                                       'finding': floor['infeasibility']}})
        outcome = launch.audit_a6_infeasible(payload)
        self.assertEqual(outcome['outcome'], 'a6_infeasible')
        self.assertFalse(outcome['kv_floor']['feasible'])
        # the resulting PASS receipt satisfies the ordering gate...
        receipt = {'job': 'A6', 'manifest_sha256': 'a' * 64, 'review_sha256': 'b' * 64,
                   'status': 'PASS', **outcome}
        self.assertTrue(launch.prior_receipt_ok(receipt, 'A6', 'a' * 64, 'b' * 64))
        # ...and the C7 conditional stays a hard gate that refuses on it
        self.assertEqual(launch.conditional_route('C7GA', 'whole_input'), 'gate')

    def test_missing_evidence_fails_closed(self):
        floor = {'kv_gib': 0.0, 'feasible': False}
        with self.assertRaises(RuntimeError):
            launch.audit_a6_infeasible(self.payload(
                {'a6_infeasible.json': {'kv_floor': floor}}))  # no fence_T0
        with self.assertRaises(RuntimeError):
            launch.audit_a6_infeasible(self.payload(
                {'fence_T0.json': self.fence_t0(),
                 'a6_infeasible.json': {'kv_floor': {'feasible': True}}}))
        self.assertIsNone(launch.audit_a6_infeasible(self.payload(
            {'fence_T0.json': self.fence_t0()})))


class FreeCellPolicyTests(unittest.TestCase):
    """R2-A7: derived slots reject free cells explicitly."""

    def test_policy(self):
        with self.assertRaises(ValueError):
            launch.validate_free_cell('capacity_search', {'L': 4096, 'C': 16, 'O': 1})
        with self.assertRaises(ValueError):
            launch.validate_free_cell('capacity_confirm', {'L': 1024, 'C': 16, 'O': 1})
        launch.validate_free_cell('capacity_search', None)   # derived: no cell input
        launch.validate_free_cell('throughput_confirm', {'L': 4096, 'C': 64, 'O': 32})
        launch.validate_free_cell('throughput_sweep', {'L': 1024, 'C': 16, 'O': 1})
        with self.assertRaises(ValueError):
            launch.validate_free_cell('wp_a', {'L': 2048, 'C': 16, 'O': 1})


class EngineBlocksRequirementTests(unittest.TestCase):
    """R2-A6: capacity arms require a non-None engine block count."""

    def test_requirement(self):
        with self.assertRaises(RuntimeError):
            launch.require_engine_blocks('capacity_search', {'num_gpu_blocks': None})
        self.assertEqual(launch.require_engine_blocks(
            'capacity_confirm', {'num_gpu_blocks': 12288})['num_gpu_blocks'], 12288)
        self.assertIsNone(launch.require_engine_blocks(
            'wp_a', {'num_gpu_blocks': None})['num_gpu_blocks'])


class ModelSplitTests(unittest.TestCase):
    """R2-A8: Qwen3/GPT-OSS WP-B arms use their coherent splits."""

    def test_qwen3_extension_arms(self):
        self.assertEqual(plan.kv_heap_for('C6QB'),
                         {'kv_gib': 7.5, 'heap_gib': 8.5, 'slot_L_mib': 256})
        self.assertEqual(plan.kv_heap_for('C6QA'),
                         {'kv_gib': 16.0, 'heap_gib': 0.0, 'slot_L_mib': None})
        self.assertEqual(plan.kv_heap_for('C6SB')['slot_L_mib'], 256)

    def test_gpt_oss_pace_uses_a6_floor(self):
        pace = plan.kv_heap_for('C7GB', a5_free_t1_gib=40.0)
        self.assertEqual((pace['kv_gib'], pace['heap_gib'], pace['slot_L_mib']),
                         (21.0, 16.5, 512))
        self.assertEqual(plan.kv_heap_for('C7GA'),
                         {'kv_gib': 16.0, 'heap_gib': 0.0, 'slot_L_mib': None})

    def test_olmoe_arms_unchanged(self):
        self.assertEqual(plan.kv_heap_for('C2BB1')['slot_L_mib'], 96)
        self.assertEqual(plan.kv_heap_for('C2BB1')['kv_gib'], 12.0)


class ProjectionSkipTests(unittest.TestCase):
    """R2-A1: projection-only dirs are a graceful no-op, reachable."""

    def test_is_projection_dir(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        self.assertFalse(launch.is_projection_dir(root))
        (root / 'completion.json').write_text(json.dumps(
            {'status': 'PROJECTION_ONLY', 'job': 'C3BD1'}))
        self.assertTrue(launch.is_projection_dir(root))
        (root / 'completion.json').write_text(json.dumps({'status': 'PASS'}))
        self.assertFalse(launch.is_projection_dir(root))


class ExtensionCapacityTests(unittest.TestCase):
    """R3-B1: the four extension capacity jobs produce real capacity
    evidence via a DECLARED preregistered cell, never derived from the
    OLMoE frontier and never free-form."""

    def receipt(self, job, C, classification, L=4096):
        return {'job': job, 'status': 'PASS',
                'capacity_probe': {'cell': {'C': C, 'L': L, 'max_tokens': 1},
                                   'classification': classification}}

    def test_declared_cells_for_extension_jobs(self):
        for job in ('C6QA', 'C6QB', 'C7GA', 'C7GB'):
            policy = launch.capacity_cell_policy(job, [])
            self.assertEqual(policy, {'cell': {'L': 4096, 'C': 16, 'O': 1},
                                      'policy': 'declared'}, job)

    def test_olmoe_frontier_present_extensions_still_declared(self):
        # the (model, arm) reverse test: a full OLMoE frontier in the prior
        # receipts must not move a Qwen extension cell by one step
        olmoe = [self.receipt('C2BA1', 16, 'served'), self.receipt('C2BA2', 32, 'served'),
                 self.receipt('C2BA3', 64, 'exhausted'), self.receipt('C2BA4', 48, 'served'),
                 self.receipt('C2BA5', 32, 'served', L=1024)]
        self.assertEqual(launch.capacity_cell_policy('C6QB', olmoe),
                         {'cell': {'L': 4096, 'C': 16, 'O': 1}, 'policy': 'declared'})
        self.assertEqual(launch.capacity_cell_policy('C7GB', olmoe)['policy'], 'declared')

    def test_extension_jobs_reject_free_cells(self):
        for role in ('capacity',):
            with self.assertRaises(ValueError):
                launch.validate_free_cell(role, {'L': 4096, 'C': 16, 'O': 1})
        launch.validate_free_cell('capacity', None)

    def test_extension_block_count_hard_gate(self):
        # R3-B1.3: require_engine_blocks covers role `capacity`
        with self.assertRaises(RuntimeError):
            launch.require_engine_blocks('capacity', {'num_gpu_blocks': None})
        self.assertEqual(launch.require_engine_blocks(
            'capacity', {'num_gpu_blocks': 8192})['num_gpu_blocks'], 8192)

    def test_worker_probe_branch_includes_extension_role(self):
        # R3-B1.1 wiring lock: the extension role runs the probe path
        source = (Path(__file__).resolve().parent / 'worker.py').read_text()
        self.assertIn("role in ('capacity', 'capacity_search', 'capacity_confirm')",
                      source)

    def test_extension_audit_requires_probe_evidence(self):
        # the audit-side requirement is structural: role capacity is in the
        # probe-required set (mirrored from job_audit)
        required_roles = ('capacity', 'capacity_search', 'capacity_confirm')
        self.assertIn(plan.job_spec('C6QA', 0)['role'], required_roles)
        self.assertIn(plan.job_spec('C7GB', 0)['role'], required_roles)

    def test_search_history_filters_by_model(self):
        olmoe = [self.receipt('C2BA1', 16, 'served')]
        qwen = [self.receipt('C6QA', 16, 'served')]
        self.assertEqual(launch.same_arm_search_history(olmoe, 'B-A', 'olmoe'),
                         [(16, 'served')])
        # cross-model history invisible even when the arm string matches
        self.assertEqual(launch.same_arm_search_history(olmoe + qwen, 'native', 'qwen3'),
                         [(16, 'served')])
        self.assertEqual(launch.same_arm_search_history(olmoe + qwen, 'B-A', 'olmoe'),
                         [(16, 'served')])


class SearchCompleteTests(unittest.TestCase):
    """R3-A1: a settled boundary produces a recorded not-run receipt."""

    def receipt(self, job, C, classification, L=4096):
        return {'job': job, 'status': 'PASS',
                'capacity_probe': {'cell': {'C': C, 'L': L, 'max_tokens': 1},
                                   'classification': classification}}

    def test_first_cell_failure_settles_remaining_search_slots(self):
        history = [self.receipt('C2BA1', 16, 'exhausted')]
        policy = launch.capacity_cell_policy('C2BA2', history)
        self.assertEqual(policy, {'search_complete': 'below_ladder_no_served_cell'})
        # and a confirm slot with nothing served settles the same way
        self.assertEqual(launch.capacity_cell_policy('C2BA5', history),
                         {'search_complete': 'below_ladder_no_served_cell'})

    def test_whole_ladder_served_settles(self):
        history = [self.receipt('C2BA1', C, 'served')
                   for C in plan.ladder_concurrency(4096)]
        self.assertEqual(launch.capacity_cell_policy('C2BA4', history),
                         {'search_complete': 'ladder_served_to_cap'})

    def test_search_complete_receipt_satisfies_ordering(self):
        receipt = {'job': 'C2BA2', 'manifest_sha256': 'a' * 64, 'review_sha256': 'b' * 64,
                   'status': 'SEARCH_COMPLETE',
                   'projection_reason': 'below_ladder_no_served_cell'}
        self.assertTrue(launch.prior_receipt_ok(receipt, 'C2BA2', 'a' * 64, 'b' * 64))
        bad = dict(receipt, projection_reason='laziness')
        self.assertFalse(launch.prior_receipt_ok(bad, 'C2BA2', 'a' * 64, 'b' * 64))
        bad = dict(receipt, status='FAILED')
        self.assertFalse(launch.prior_receipt_ok(bad, 'C2BA2', 'a' * 64, 'b' * 64))


class ExtensionAlignmentTests(unittest.TestCase):
    """R3-A3: per-model Tier-1 leg-(ii) alignment in the aggregate."""

    def completion(self, job, model, arm, l1):
        return {'job': job, 'status': 'PASS', 'duration_seconds': 100.0,
                'duration_gate': {'warning': False}, 'arm': arm, 'model': model,
                'l1_max_used_per_die': l1}

    def test_qwen_alignment_computed_with_model_grouping(self):
        native_l1 = [40 * plan.GIB] * 16
        pace_l1 = [40 * plan.GIB + 300 * plan.MIB] * 16
        completions = {
            'C6QA': self.completion('C6QA', 'qwen3', 'native', native_l1),
            'C6QB': self.completion('C6QB', 'qwen3', 'pace', pace_l1),
            'C2BA1': self.completion('C2BA1', 'olmoe', 'B-A', native_l1),
            # a rogue OLMoE B-B die-max must not enter the qwen alignment
            'C2BB1': self.completion('C2BB1', 'olmoe', 'B-B',
                                     [44 * plan.GIB] * 16),
        }
        analyses = launch.aggregate_analyses(completions)
        self.assertTrue(analyses['peak_alignment_by_model']['qwen3']['aligned'])
        self.assertFalse(analyses['peak_alignment_by_model']['olmoe']['aligned'])
        self.assertNotIn('gpt_oss', analyses['peak_alignment_by_model'])


    def test_a4_infeasible_branch_requires_l1(self):
        # R3-A4 wiring lock: the A6-infeasible audit path calls the L1
        # stream validator before publishing its receipt (uniformity)
        source = (Path(__file__).resolve().parent / 'launch.py').read_text()
        self.assertIn('infeasible = audit_a6_infeasible(payload)', source)
        self.assertIn('l1_summary = audit_l1_stream(directory)', source)


class PlanAttributeLockTests(unittest.TestCase):
    """Anti-recurrence lock for plan-attribute typos (round-4 device find:
    precheck.py referenced a FENCE-POINT-spelled constant that exists only
    as FENCEPOST — a slip that surfaced inside the device precheck
    container). Every plan.<IDENT> reference in the device-side sources must
    resolve on the plan module, and the FENCEPOST constant family must be
    spelled consistently everywhere."""

    DEVICE_SOURCES = ('worker.py', 'precheck.py', 'launch.py', 'supervisor.py',
                      'wp_a_profile.py', 'wp_b_frontier.py')

    def test_every_plan_attribute_reference_resolves(self):
        import re as _re
        pattern = _re.compile(r'\bplan\.([A-Za-z_][A-Za-z0-9_]*)')
        missing = []
        checked = 0
        for name in self.DEVICE_SOURCES:
            text = (ROOT / name).read_text()
            for ident in pattern.findall(text):
                if ident == 'py':
                    continue  # the filename token 'plan.py' in prose/strings
                checked += 1
                if not hasattr(plan, ident):
                    missing.append(f'{name}: plan.{ident}')
        self.assertEqual(missing, [], f'unresolvable plan references: {missing}')
        self.assertGreater(checked, 100)  # the sweep genuinely covers the sources

    def test_fencepost_family_spelled_consistently(self):
        # the misspelled FENCE-POINT form must not exist in any shipped
        # source (test_cpu.py is excluded only because this docstring
        # documents the defect by name)
        for name in (*self.DEVICE_SOURCES, 'plan.py', 'DESIGN.md'):
            self.assertNotIn('FENCE' + 'POINT', (ROOT / name).read_text(), name)
        for ident in ('FENCEPOSTS', 'FENCEPOST_LABELS', 'L3_ADOPTION_MAX_FENCEPOST_DELTA'):
            self.assertTrue(hasattr(plan, ident), ident)


class WorkerExtensionStringTests(unittest.TestCase):
    """Round-7 fix lock: vLLM ParallelConfig.worker_extension_cls requires a
    DOTTED STRING (a class object fails pydantic validation — A0a device
    find). build_engine must pass a string whose dotted path resolves to the
    top-level extension class, and the whole engine config must contain only
    JSON-primitive values."""

    def load_worker_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'memcamp_worker_under_test', ROOT / 'worker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_extension_is_dotted_string_and_resolves(self):
        worker = self.load_worker_module()
        self.assertIsInstance(worker.WORKER_EXTENSION_CLS, str)
        module_name, _, class_name = worker.WORKER_EXTENSION_CLS.rpartition('.')
        self.assertEqual(module_name, 'worker')
        self.assertEqual(class_name, 'MemoryCampaignExtension')
        # the class exists at the worker module's top level and is a class
        self.assertTrue(hasattr(worker, class_name))
        self.assertTrue(isinstance(getattr(worker, class_name), type))

    def test_build_engine_string_contract_via_fake_llm(self):
        import types
        import sys as _sys
        worker = self.load_worker_module()
        captured = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_vllm = types.ModuleType('vllm')
        fake_vllm.LLM = FakeLLM
        saved = _sys.modules.get('vllm')
        _sys.modules['vllm'] = fake_vllm
        try:
            engine = worker.build_engine('olmoe', 16.0, 2048, 16)
        finally:
            if saved is None:
                _sys.modules.pop('vllm', None)
            else:
                _sys.modules['vllm'] = saved
        self.assertIsInstance(engine, FakeLLM)
        # the pydantic contract: a string, not a class object
        self.assertIsInstance(captured['worker_extension_cls'], str)
        self.assertEqual(captured['worker_extension_cls'],
                         'worker.MemoryCampaignExtension')
        # and every other config value is a JSON primitive
        for key, value in captured.items():
            self.assertIn(type(value), (str, int, float, bool, type(None)), key)

    def test_build_engine_rejects_class_objects(self):
        worker = self.load_worker_module()
        with self.assertRaises(ValueError):
            worker.build_engine('olmoe', 16.0, 2048, 16,
                                extension=worker.MemoryCampaignExtension)
        with self.assertRaises(ValueError):
            worker.build_engine('olmoe', 16.0, 2048, 16, extension='NoDot')

    def test_call_site_uses_the_string_constant(self):
        source = (ROOT / 'worker.py').read_text()
        self.assertIn('build_engine(spec[\'model\'], kv_heap[\'kv_gib\'], '
                      'max_model_len, max_num_seqs,\n                       '
                      'WORKER_EXTENSION_CLS)', source)
        self.assertNotIn('MemoryCampaignExtension)', source.split('def main')[1])


if __name__ == '__main__':
    unittest.main()
