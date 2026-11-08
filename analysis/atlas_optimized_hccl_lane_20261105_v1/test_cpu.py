"""Host logic tests only; no torch/vllm import, no device, no network.

Covers the frozen pins against local source-of-truth artifacts
(FrozenPinTests), the Williams schedule re-derivation against arms.json, the
review-contract single source (one conformance fixture + five non-conforming
variants), the trimming ladder, the O2 probe judge, census/SQ/request
validators, the estimator and direction rule, the restoration state machine
and its CPU fake-import proof, the supervisor contract, the container
inspection contract, tree coverage and the admission-guard parser on the
observed all-idle 16-die fixture.
"""
import hashlib
import importlib.util
import json
import statistics
import unittest
from pathlib import Path

import plan
import restoration
from supervisor import supervise

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT.parent
WORKSPACE = ANALYSIS.parent
spec = importlib.util.spec_from_file_location('opt_guard', ANALYSIS / 'atlas_group_admission_device_v3' / 'launch.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
fixture = json.loads((ROOT / 'observed_npu_info_fixture.json').read_text())['npu_smi_info']
PACE_SCHEMA_TRUTH = plan.PACE_SCHEMA_TRUTH


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenPinTests(unittest.TestCase):
    """Every pin re-derived from the local source-of-truth artifacts."""

    def test_provider_pins(self):
        self.assertEqual(sha_file(ANALYSIS / 'atlas_group_admission_device_v3/launch.py'),
                         plan.GUARD_LAUNCH_SHA)
        self.assertEqual(sha_file(ANALYSIS / 'atlas_group_admission_device_v3/source_manifest.json'),
                         plan.GUARD_MANIFEST_SHA)
        self.assertEqual(sha_file(ANALYSIS / 'atlas_sdma_queue_metadata_v1/metadata.py'),
                         plan.HELPER_SHA256)
        self.assertEqual(sha_file(ANALYSIS / 'atlas_sdma_large_correctness_v1/layout.json'),
                         plan.LAYOUT_SHA256)
        self.assertEqual(plan.IMAGE,
                         'sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f')

    def test_frozen_lane_manifest_pins(self):
        pairs = (('atlas_native_hccl_no_resize_v1/source_manifest.json', plan.CANDIDATE_MANIFEST),
                 ('atlas_native_hccl_no_resize_o1_v2/source_manifest.json', plan.RUNNER_MANIFEST),
                 ('atlas_native_hccl_no_resize_timing_v1/source_manifest.json', plan.TIMING_MANIFEST),
                 ('atlas_olmoe_o1_sync_model_v1/source_manifest.json', plan.SYNC_MODEL_MANIFEST))
        for name, pin in pairs:
            self.assertEqual(sha_file(ANALYSIS / name), pin, name)
        # the pinned manifests must actually cover their key entries
        for name, pin, member in (
                ('atlas_native_hccl_no_resize_o1_v2/source_manifest.json', plan.RUNNER_MANIFEST, 'patch_install.py'),
                ('atlas_native_hccl_no_resize_timing_v1/source_manifest.json', plan.TIMING_MANIFEST, 'run_timing.py'),
                ('atlas_olmoe_o1_sync_model_v1/source_manifest.json', plan.SYNC_MODEL_MANIFEST, 'model_extension.py')):
            manifest = json.loads((ANALYSIS / name).read_text())
            self.assertEqual(sha_file(ANALYSIS / name.rsplit('/', 1)[0] / member), manifest[member])

    def test_receipt_and_document_pins(self):
        results = WORKSPACE / 'analysis_results'
        self.assertEqual(sha_file(results / 'atlas_native_no_resize_o1_lane_20261028_v2/native/completion.json'),
                         plan.PARITY_NATIVE_COMPLETION)
        self.assertEqual(sha_file(results / 'atlas_native_no_resize_o1_lane_20261028_v2/no-resize/completion.json'),
                         plan.PARITY_NORESIZE_COMPLETION)
        self.assertEqual(sha_file(results / 'atlas_olmoe_o1_sync_model_20261028_v2/p16/completion.json'),
                         plan.SYNC_MODEL_COMPLETION)
        for name, pin in (('arms.json', plan.ARMS_JSON_SHA), ('PLAN.md', plan.PLAN_MD_SHA)):
            self.assertEqual(sha_file(ANALYSIS / 'atlas_optimized_hccl_plan_20261105_v1' / name), pin)
        # census numbers equal the qualified sync-model raw-audit counts
        self.assertEqual((plan.CENSUS_PACE, plan.CENSUS_NATIVE_ZERO_TOTAL), (62, 2))

    def test_pins_match_arms_json(self):
        arms = json.loads((ANALYSIS / 'atlas_optimized_hccl_plan_20261105_v1/arms.json').read_text())
        frozen = arms['frozen_identities']
        self.assertEqual(frozen['runtime_image'], plan.IMAGE)
        self.assertTrue(frozen['libhccl_so'].endswith(plan.LIBHCCL_SHA))
        self.assertEqual(frozen['libshmem_so'], plan.SHMEM_SHA256)
        # Both digests below were transcription typos in the original
        # arms.json, caught by this lane's programmatic pin transcription and
        # corrected by the plan owner on 2026-11-05; they now equal the
        # actual frozen files character-for-character (strictly asserted).
        self.assertEqual(frozen['queue_metadata_helper'], plan.HELPER_SHA256)
        self.assertEqual(sha_file(ANALYSIS / 'atlas_sdma_queue_metadata_v1/metadata.py'),
                         plan.HELPER_SHA256)
        self.assertEqual(frozen['device_layout_json'], plan.LAYOUT_SHA256)
        self.assertEqual(frozen['reference_request_raw_sha256'], plan.REFERENCE_RAW_SHA)
        self.assertEqual(sha_file(WORKSPACE / 'analysis_results/atlas_model_split_capture_lane_20261028_v1/hccl-o1/request_raw.json'),
                         plan.REFERENCE_RAW_SHA)
        self.assertEqual(frozen['no_resize_timing_worker_manifest'], plan.TIMING_MANIFEST)
        self.assertEqual(frozen['corrected_parity_v2_manifest'], plan.RUNNER_MANIFEST)
        self.assertEqual(frozen['frozen_patch_candidate_manifest'], plan.CANDIDATE_MANIFEST)
        self.assertEqual(frozen['capture_manifest'], plan.CAPTURE_MANIFEST)


class ScheduleTests(unittest.TestCase):
    def test_williams_properties(self):
        self.assertTrue(plan.validate_schedule())
        arms = json.loads((ANALYSIS / 'atlas_optimized_hccl_plan_20261105_v1/arms.json').read_text())
        self.assertTrue(plan.validate_schedule(arms['schedule']['groups']))

    def test_ladder_groups(self):
        self.assertEqual(len(plan.ladder_groups('L0')), 16)
        self.assertEqual(len(plan.ladder_groups('L1')), 12)
        self.assertEqual(len(plan.ladder_groups('L2')), 8)
        self.assertNotIn('g13', plan.ladder_groups('L1'))
        self.assertNotIn('g09', plan.ladder_groups('L2'))
        for level in ('L1', 'L2'):
            for key in plan.ladder_groups(level):
                self.assertIn(key, plan.ladder_groups('L0'))

    def test_ports_unique(self):
        seen = set()
        for job in plan.CAMPAIGN_SLOTS:
            pair = plan.ports(job)
            self.assertNotIn(pair, seen)
            seen.add(pair)
        self.assertEqual(len(plan.CAMPAIGN_SLOTS), 20)

    def test_ladder_step_gates(self):
        receipts = [{'ladder_level': 'L0', 'duration_seconds': 300.0} for _ in range(4)]
        with self.assertRaises(ValueError):
            plan.ladder_step(receipts, 'L1')  # no trigger
        hot = [{'ladder_level': 'L0', 'duration_seconds': 500.0}] + receipts[:3]
        trigger = plan.ladder_step(hot, 'L1')
        self.assertEqual(trigger['requested_level'], 'L1')
        self.assertEqual(trigger['single_over_trigger'], [500.0])
        consecutive = [{'ladder_level': 'L0', 'duration_seconds': 400.0},
                       {'ladder_level': 'L0', 'duration_seconds': 400.0}]
        self.assertIsNotNone(plan.ladder_step(consecutive, 'L1'))
        with self.assertRaises(ValueError):
            plan.ladder_step(receipts, 'L2')  # one step at a time
        with self.assertRaises(ValueError):
            plan.ladder_step([], 'L1')


class ProbeTests(unittest.TestCase):
    def test_p0_when_no_receipt(self):
        judgment = plan.judge_probe_receipt('', [])
        self.assertEqual(judgment['path'], 'P0')

    def test_p1_only_on_value_echo(self):
        line = '[INFO] hccl config: HCCL_BUFFSIZE set to 333 by user env'
        self.assertEqual(plan.judge_probe_receipt(line, [line])['path'], 'P1')
        wrong_value = '[INFO] hccl config: HCCL_BUFFSIZE set to 200 (default)'
        with self.assertRaises(ValueError):
            plan.judge_probe_receipt(wrong_value, [wrong_value])
        # a forged receipt (matched lines absent from the raw log) is rejected
        with self.assertRaises(ValueError):
            plan.judge_probe_receipt('', [line])

    def test_hopt_env_paths(self):
        self.assertEqual(plan.hopt_env('P0'), {})
        self.assertEqual(plan.hopt_env('P1', {'knob': 'HCCL_BUFFSIZE', 'value': '512'}),
                         {'HCCL_BUFFSIZE': '512'})
        for bad in ({'knob': 'HCCL_WHAT', 'value': '512'}, {'knob': 'HCCL_BUFFSIZE', 'value': 'x'}):
            with self.assertRaises(ValueError):
                plan.hopt_env('P1', bad)
        with self.assertRaises(ValueError):
            plan.hopt_env('P2')


class ReviewContractTests(unittest.TestCase):
    def compliant(self, manifest_sha=None):
        return {'schema': plan.REVIEW_SCHEMA, 'status': 'APPROVED',
                'scope': plan.REVIEW_SCOPE,
                'source_manifest_sha256': manifest_sha or 'a' * 64,
                'arms_json_sha256': plan.ARMS_JSON_SHA,
                'reviewed_files': sorted(plan.REVIEWED_FILES),
                'groups': json.loads(json.dumps(plan.REVIEW_GROUPS)),
                'f_restoration_gate': 'PROVEN', 'degraded_three_arm': False,
                'probe_path': 'P0', 'hopt_env': None}

    def test_conforming_review_passes(self):
        review = plan.validate_review(self.compliant(), 'a' * 64)
        self.assertEqual(review['probe_path'], 'P0')
        self.assertFalse(review['degraded_three_arm'])

    def test_five_variants_rejected(self):
        variants = {}
        rejected = self.compliant()
        rejected['status'] = 'REJECTED'
        variants['status'] = rejected
        wrong_manifest = self.compliant()
        wrong_manifest['source_manifest_sha256'] = 'b' * 64
        variants['manifest'] = wrong_manifest
        wrong_scope = self.compliant()
        wrong_scope['scope'] = 'some_other_scope'
        variants['scope'] = wrong_scope
        missing_files = self.compliant()
        missing_files['reviewed_files'] = sorted(set(missing_files['reviewed_files']) - {'worker.py'})
        variants['files'] = missing_files
        wrong_groups = self.compliant()
        wrong_groups['groups'] = json.loads(json.dumps(plan.REVIEW_GROUPS))
        wrong_groups['groups']['arm_map']['g16'] = 'F'
        variants['groups'] = wrong_groups
        for label, review in variants.items():
            with self.assertRaises(ValueError, msg=label):
                plan.validate_review(review, 'a' * 64)

    def test_degradation_and_p1_bindings(self):
        degraded = self.compliant()
        degraded['f_restoration_gate'] = 'NOT_REQUIRED'
        with self.assertRaises(ValueError):
            plan.validate_review(degraded, 'a' * 64)
        degraded['degraded_three_arm'] = True
        plan.validate_review(degraded, 'a' * 64)
        p1 = self.compliant()
        p1['probe_path'] = 'P1'
        p1['hopt_env'] = {'HCCL_BUFFSIZE': '512'}
        result = plan.validate_review(p1, 'a' * 64)
        self.assertEqual(result['hopt_env'], {'HCCL_BUFFSIZE': '512'})
        bad_value = self.compliant()
        bad_value['probe_path'] = 'P1'
        bad_value['hopt_env'] = {'HCCL_BUFFSIZE': 'zero'}
        with self.assertRaises(ValueError):
            plan.validate_review(bad_value, 'a' * 64)
        p1_orphan = self.compliant()
        p1_orphan['probe_path'] = 'P1'
        with self.assertRaises(ValueError):
            plan.validate_review(p1_orphan, 'a' * 64)


class CensusAndValidatorTests(unittest.TestCase):
    def hccl_census(self):
        # Row shape produced by plan.normalize_hccl_census (real frozen format).
        return {'ranks': [{'rank': rank, 'alltoallv_class_calls': 64, 'collectives': 64,
                           'method_backends': [plan.CENSUS_METHOD_BACKEND],
                           'dispatch_methods': 32, 'combine_methods': 32,
                           'selector_methods': [plan.CENSUS_SELECTOR_METHOD] * 2,
                           'forward_kinds': [plan.CENSUS_FORWARD_KIND] * 2}
                          for rank in range(16)],
                'global_distinct_collectives': 64}

    def pace_census(self):
        return {'ranks': [{'rank': rank, 'logical_calls': 64, 'pace_calls': 62,
                           'hccl_calls': 2} for rank in range(16)]}

    def test_census_validators(self):
        plan.validate_hccl_census(self.hccl_census())
        plan.validate_pace_census(self.pace_census())
        broken = self.hccl_census()
        broken['ranks'][3]['dispatch_methods'] = 31
        with self.assertRaises(ValueError):
            plan.validate_hccl_census(broken)
        short = self.pace_census()
        short['ranks'] = short['ranks'][:15]
        with self.assertRaises(ValueError):
            plan.validate_pace_census(short)
        noglobal = self.hccl_census()
        del noglobal['global_distinct_collectives']
        with self.assertRaises(ValueError):
            plan.validate_hccl_census(noglobal)

    def sq_boundaries(self, series=(0, 0, 0, 0)):
        boundaries = {}
        for index, label in enumerate(plan.F_SQ_BOUNDARIES):
            boundaries[label] = [[{'sq_head': 0, 'sq_tail': series[index]}]
                                 for _ in range(plan.WORLD)]
        return boundaries

    def test_sq_boundaries(self):
        plan.validate_sq_boundaries(self.sq_boundaries())
        plan.validate_sq_boundaries(self.sq_boundaries((5, 5, 5, 5)))
        plan.validate_sq_boundaries(self.sq_boundaries((4, 6, 2, 2)))
        with self.assertRaises(ValueError):
            plan.validate_sq_boundaries(self.sq_boundaries((0, 1, 2, 3)))
        with self.assertRaises(ValueError):
            plan.validate_sq_boundaries({k: v for k, v in self.sq_boundaries().items()
                                         if k != 'warmup3'})

    def requests(self):
        rows = []
        for index in range(plan.TOTAL_REQUESTS):
            phase = 'gate' if index == 0 else ('warmup' if index <= plan.WARMUP_REQUESTS
                                               else 'measurement')
            rows.append({'index': index, 'phase': phase,
                         'start_ns': 1_000_000 + index * 10_000_000,
                         'end_ns': 1_000_000 + index * 10_000_000 + 200_000_000,
                         'generate_elapsed_ns': 200_000_000,
                         'clock': 'driver perf_counter_ns around llm.generate',
                         'request_id': f'req-{index:02d}',
                         'output_tokens_equal_reference': True,
                         'validated_outside_timer': True})
        return rows

    def group_record(self, arm='H', group='g01', level='L0'):
        record = {'event': 'group_samples', 'group': group, 'arm': arm, 'world': 16,
                  'ladder_level': level, 'image_id': plan.IMAGE,
                  'clock': 'driver perf_counter_ns around llm.generate',
                  'census': self.hccl_census() if arm in plan.HCCL_ARMS else self.pace_census(),
                  'requests': self.requests()}
        if arm in plan.HCCL_ARMS:
            record['hopt_env'] = {} if arm != 'HOPT' else {}
        else:
            record.update({'slot_bytes': plan.SLOT_BYTES, 'heap_bytes': plan.HEAP_BYTES,
                           'f_restoration': {'recorder_removed': True, 'hook_retained': True,
                                             'recorder_digest_unchanged': True},
                           'sq_boundaries': self.sq_boundaries()})
        return record

    def test_group_record_validation(self):
        for arm, group in (('H', 'g01'), ('HmR', 'g02'), ('HOPT', 'g04'), ('F', 'g03'),
                           ('H', 'g12'), ('F', 'g13'), ('HmR', 'g16')):
            plan.validate_group_record(self.group_record(arm, group), group, arm, 'L0')
        for mutate in ('arm_mismatch', 'census', 'request_phase', 'timer', 'sq_missing'):
            record = self.group_record('F', 'g03')
            if mutate == 'arm_mismatch':
                with self.assertRaises(ValueError):
                    plan.validate_group_record(self.group_record('H', 'g03'), 'g03', 'H', 'L0')
                continue
            if mutate == 'census':
                record['census']['ranks'][0]['pace_calls'] = 61
            elif mutate == 'request_phase':
                record['requests'][2]['phase'] = 'measurement'
            elif mutate == 'timer':
                record['requests'][7]['end_ns'] = record['requests'][7]['start_ns']
            else:
                del record['sq_boundaries']
            with self.assertRaises(ValueError):
                plan.validate_group_record(record, 'g03', 'F', 'L0')
        with self.assertRaises(ValueError):
            plan.validate_group_record(self.group_record('H', 'g13'), 'g13', 'H', 'L2')

    def test_request_median(self):
        rows = self.requests()
        self.assertEqual(plan.group_median_ms(rows), 200.0)
        rows[10]['end_ns'] += 50_000_000
        rows[10]['generate_elapsed_ns'] += 50_000_000
        self.assertEqual(plan.group_median_ms(rows), 200.0)


LOCAL_MODULE_MIRRORS = {
    '/workspace/lane': 'atlas_optimized_hccl_lane_20261105_v1',
    '/workspace/qualification': 'atlas_olmoe_o1_sync_qualification_v3',
    '/workspace/no_resize_candidate': 'atlas_native_hccl_no_resize_v1',
    '/workspace/capture': 'atlas_model_split_capture_v1',
    '/workspace/no_resize_lane': 'atlas_native_hccl_no_resize_o1_v2',
    '/workspace/native_timing_lane': 'atlas_native_hccl_no_resize_timing_v1',
    '/workspace/model_hook': 'atlas_olmoe_o1_sync_model_v1',
}


class DelegatedClosureTests(unittest.TestCase):
    """Root-fix proof: the delegated chain's bare-import transitive closure
    resolves completely under the mounted module-home roots (device
    failures 54167a66 'contract' and 216a2400 'capture_contract' were
    closure members whose home was missing)."""

    @classmethod
    def setUpClass(cls):
        import sys
        roots = [(container, ANALYSIS / mirror)
                 for container, mirror in LOCAL_MODULE_MIRRORS.items()]
        entries = [ANALYSIS / LOCAL_MODULE_MIRRORS['/workspace/native_timing_lane'] / 'run_timing.py',
                   ANALYSIS / LOCAL_MODULE_MIRRORS['/workspace/model_hook'] / 'model_extension.py',
                   ANALYSIS / 'atlas_optimized_hccl_lane_20261105_v1' / 'worker.py']
        cls.closure = plan.resolve_delegated_closure(
            entries, roots, frozenset(sys.stdlib_module_names) | {'__future__'})

    def test_closure_has_no_unresolved_modules(self):
        self.assertEqual(self.closure['unresolved'], [])

    def test_closure_edge_table(self):
        edges = {(importer, name, home) for importer, name, home in self.closure['edges']}
        for expected in (
                ('run_o1.py', 'capture_contract', '/workspace/capture'),
                ('run_timing.py', 'capture_contract', '/workspace/capture'),
                ('run_capture.py', 'capture_contract', '/workspace/capture'),
                ('no_resize_worker.py', 'worker_extension', '/workspace/capture'),
                ('no_resize_worker.py', 'patch_source', '/workspace/no_resize_candidate'),
                ('bindings.py', 'patch_install', '/workspace/no_resize_lane'),
                ('model_extension.py', 'contract', '/workspace/qualification'),
                ('model_extension.py', 'frontend', '/workspace/qualification'),
                ('model_extension.py', 'model_state', '/workspace/model_hook'),
                ('model_contract.py', 'dependencies', '/workspace/qualification'),
                ('runtime.py', 'padded_buffers', '/workspace/qualification'),
                ('worker.py', 'plan', '/workspace/lane'),
                ('worker.py', 'restoration', '/workspace/lane')):
            self.assertIn(expected, edges, expected)
        homes = {home for _, _, home in self.closure['edges']}
        self.assertTrue(homes <= set(plan.DELEGATED_MODULE_ROOTS),
                        f'closure escaped the mounted roots: {homes - set(plan.DELEGATED_MODULE_ROOTS)}')

    def test_every_container_module_exists_locally(self):
        for importer, name, home in self.closure['edges']:
            mirror = ANALYSIS / LOCAL_MODULE_MIRRORS[home]
            self.assertTrue((mirror / (name + '.py')).is_file(),
                            f'{name} missing from {home}')

    def test_iter_bare_imports(self):
        source = ('import os, json\n'
                  'from pathlib import Path\n'
                  'from capture_contract import frozen_sources\n'
                  'from . import sibling\n'
                  'import torch.nn.functional\n')
        self.assertEqual(plan.iter_bare_imports(source),
                         {'os', 'json', 'pathlib', 'capture_contract', 'torch'})

    def test_delegated_pythonpath_order(self):
        import launch
        script = launch.entry_script('/workspace/lane/worker.py')
        positions = [script.index(root) for root in plan.DELEGATED_MODULE_ROOTS]
        self.assertEqual(positions, sorted(positions),
                         'module roots must appear in DELEGATED_MODULE_ROOTS order')
        self.assertLess(script.index('/workspace/qualification'),
                        script.index('/workspace/model_hook'))
        self.assertLess(script.index('/workspace/qualification'),
                        script.index('/workspace/queue_metadata'))
        for root in plan.DELEGATED_MODULE_ROOTS:
            self.assertIn(root, launch.DELEGATED_PYTHONPATH)

    def test_capture_lane_mounted_and_pinned(self):
        import launch
        self.assertEqual(sha_file(ANALYSIS / 'atlas_model_split_capture_v1/source_manifest.json'),
                         plan.CAPTURE_MANIFEST)
        mounts = dict((host, target) for host, target, _ in launch.fixed_mounts())
        self.assertEqual(mounts.get(launch.CAPTURE), plan.CAPTURE_CONTAINER_DIR)
        self.assertEqual(plan.CAPTURE_CONTAINER_DIR, '/workspace/capture')


class FrozenCensusFixtureTests(unittest.TestCase):
    """BLOCKER regression tests: the census adapters against the frozen
    predecessors' REAL persisted evidence formats."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT / 'frozen_census_fixture.json').read_text())

    def test_fixture_provenance_still_binds_sources(self):
        for group in ('hccl_sources', 'pace_sources'):
            for name, entry in self.fixture['provenance'][group].items():
                path = WORKSPACE / entry['path']
                self.assertEqual(sha_file(path), entry['sha256'],
                                 f'{group}/{name} drifted')

    def test_hccl_adapter_on_real_persisted_evidence(self):
        # the FULL 11 MB persisted request_raw.json (six-group campaign,
        # pair2-no-resize) plus its validated gate trace, unexcerpted
        root = WORKSPACE / 'analysis_results/atlas_native_no_resize_timing_lane_20261028_v1/pair2-no-resize/payload'
        raw = json.loads((root / 'request_raw.json').read_text())
        trace = json.loads((root / 'validated_gate_trace.json').read_text())
        census = plan.normalize_hccl_census(raw['after_gate'], trace)
        plan.validate_hccl_census(census)
        self.assertEqual(census['global_distinct_collectives'], 64)

    def test_hccl_adapter_on_shipped_fixture_and_mutations(self):
        census = plan.normalize_hccl_census(self.fixture['hccl']['after_gate'],
                                            self.fixture['hccl']['validated_trace'])
        plan.validate_hccl_census(census)
        rows, trace = self.fixture['hccl']['after_gate'], self.fixture['hccl']['validated_trace']
        mutations = {}
        wrong_global = dict(trace, alltoallv_global_calls=1024)
        mutations['global_1024'] = (rows, wrong_global)
        broken_counts = json.loads(json.dumps(rows))
        broken_counts[0]['capture']['counts']['collective'] = 63
        mutations['counts'] = (broken_counts, trace)
        broken_backend = json.loads(json.dumps(rows))
        for event in broken_backend[0]['capture']['events']:
            if event['kind'] == 'method_enter':
                event['backend_path'] = 'MC2'
                break
        mutations['backend'] = (broken_backend, trace)
        broken_selector = json.loads(json.dumps(rows))
        for event in broken_selector[0]['capture']['events']:
            if event['kind'] == 'selector':
                event['method'] = 'MC2'
                break
        mutations['selector'] = (broken_selector, trace)
        broken_position = json.loads(json.dumps(rows))
        for event in broken_position[0]['capture']['events']:
            if event['kind'] == 'method_enter' and event['position'] == 'combine':
                event['position'] = 'dispatch'
                break
        mutations['position_split'] = (broken_position, trace)
        orphan = json.loads(json.dumps(rows))
        orphan[0]['capture']['events'] = orphan[0]['capture']['events'][:-1]
        mutations['missing_events'] = (orphan, trace)
        dropped = json.loads(json.dumps(rows))[:15]
        mutations['rank_coverage'] = (dropped, trace)
        for label, (mutated_rows, mutated_trace) in mutations.items():
            with self.assertRaises(ValueError, msg=label):
                plan.normalize_hccl_census(mutated_rows, mutated_trace)

    def test_pace_adapter_on_shipped_fixture_and_mutations(self):
        census = plan.normalize_pace_census(self.fixture['pace']['snapshot_rows'])
        plan.validate_pace_census(census)
        rows = self.fixture['pace']['snapshot_rows']
        mutations = {}
        wrong_counts = json.loads(json.dumps(rows))
        wrong_counts[0]['counts']['python_pace_calls'] = 61
        mutations['counts'] = wrong_counts
        contradicting = json.loads(json.dumps(rows))
        contradicting[5]['records'][0]['backend'] = 'HCCL'
        mutations['records_contradict'] = contradicting
        no_counts = json.loads(json.dumps(rows))
        del no_counts[2]['counts']
        mutations['counts_missing'] = no_counts
        for label, mutated in mutations.items():
            with self.assertRaises(ValueError, msg=label):
                plan.normalize_pace_census(mutated)

    def test_arms_json_census_parenthetical_is_empirically_wrong(self):
        # Recorded finding: arms.json's census parenthetical "(2 All2AllV +
        # 62 MC2)" does not match the six persisted groups, where all 64
        # collectives ride All2AllV methods (32 dispatch + 32 combine) on
        # 2 execute_model forwards; the lane binds the empirical invariant.
        arms = json.loads((ANALYSIS / 'atlas_optimized_hccl_plan_20261105_v1/arms.json').read_text())
        hccl_text = [arm['census_expectation'] for arm in arms['arms']
                     if arm['id'] in plan.HCCL_ARMS][0]
        self.assertIn('2 All2AllV + 62 MC2', hccl_text)
        self.assertEqual(plan.HCCL_CAPTURE_COUNTS,
                         {'collective': 64, 'forward': 2, 'method': 64, 'runner': 3})
        self.assertEqual((plan.CENSUS_DISPATCH_METHODS, plan.CENSUS_COMBINE_METHODS), (32, 32))


class EstimatorTests(unittest.TestCase):
    def test_summary_and_mad(self):
        result = plan.summary([3.0, 1.0, 2.0, 5.0, 4.0])
        q1, _, q3 = statistics.quantiles([3.0, 1.0, 2.0, 5.0, 4.0], n=4, method='inclusive')
        self.assertEqual((result['median'], result['q1'], result['q3']), (3.0, q1, q3))
        self.assertEqual(plan.mad([1.0, 2.0, 3.0, 4.0]), 1.0)
        with self.assertRaises(ValueError):
            plan.summary([])

    def test_direction_rule_stable_and_unstable(self):
        treatment = [100.0, 101.0, 99.0, 100.5]
        comparator = [110.0, 111.0, 109.0, 110.5]
        result = plan.direction_rule(treatment, comparator, 'L0')
        self.assertEqual(result['classification'], 'stable-faster')
        self.assertEqual(result['treatment_faster_pairs'], 16)
        noisy = [100.0, 112.0, 99.0, 100.5]
        self.assertEqual(plan.direction_rule(noisy, comparator, 'L0')['classification'],
                         'unstable')
        flips = [100.0, 101.0, 99.0, 110.4]
        self.assertEqual(plan.direction_rule(flips, comparator, 'L0')['classification'],
                         'stable-with-flips')
        tie = [110.0, 110.001, 109.999, 110.0]
        self.assertIn(plan.direction_rule(tie, comparator[:3] + [110.0], 'L0')['classification'],
                      ('unstable', 'stable-with-flips'))
        self.assertEqual(plan.direction_rule(treatment[:3], comparator[:3], 'L1')
                         ['treatment_faster_pairs'], 9)
        self.assertEqual(plan.direction_rule(treatment[:2], comparator[:2], 'L2')['classification'],
                         'descriptive_only')
        with self.assertRaises(ValueError):
            plan.direction_rule(treatment, comparator[:3], 'L0')

    def test_select_level_receipts_per_level_mode(self):
        def receipt(key, level):
            return {'schema': 'revision2.optimized-hccl-completion.v1', 'status': 'PASS',
                    'group': key, 'arm': plan.GROUP_ARM[key], 'world': 16,
                    'ladder_level': level, 'manifest_sha256': 'a' * 64,
                    'review_sha256': 'b' * 64, 'image_id': plan.IMAGE,
                    'duration_seconds': 120.0, 'group_median_ms': 200.0}
        # complete L0 campaign
        on_disk = {key: receipt(key, 'L0') for key in plan.ladder_groups('L0')}
        self.assertEqual(sorted(plan.select_level_receipts(on_disk, 'L0')),
                         sorted(plan.ladder_groups('L0')))
        # the pre-registered L1 subset of a complete L0 campaign is a fixed
        # (non-data-driven) subset and is therefore aggregable too
        self.assertEqual(sorted(plan.select_level_receipts(on_disk, 'L1')),
                         sorted(plan.ladder_groups('L1')))
        # mid-campaign trim after g08: g01..g08 at L0, g09..g12 at L1
        mixed = {**{key: receipt(key, 'L0') for key in plan.ladder_groups('L0')[:8]},
                 **{key: receipt(key, 'L1') for key in plan.ladder_groups('L1')[8:]}}
        chosen_l1 = plan.select_level_receipts(mixed, 'L1')
        self.assertEqual(sorted(chosen_l1), sorted(plan.ladder_groups('L1')))
        chosen_l2 = plan.select_level_receipts(mixed, 'L2')
        self.assertEqual(sorted(chosen_l2), sorted(plan.ladder_groups('L2')))
        # a receipt recorded at a LATER level than requested is refused
        late = dict(mixed)
        late['g01'] = receipt('g01', 'L2')
        with self.assertRaises(ValueError):
            plan.select_level_receipts(late, 'L1')
        with self.assertRaises(ValueError):
            plan.select_level_receipts(mixed, 'L0')  # g09..g12 recorded beyond L0
        degraded = {key: receipt(key, 'L1') for key in plan.ladder_groups('L1')
                    if plan.GROUP_ARM[key] != 'F'}
        self.assertEqual(sorted(plan.select_level_receipts(degraded, 'L1', True)),
                         sorted(key for key in plan.ladder_groups('L1')
                                if plan.GROUP_ARM[key] != 'F'))
        with_f = dict(degraded, g03=receipt('g03', 'L1'))
        with self.assertRaises(ValueError):
            plan.select_level_receipts(with_f, 'L1', True)

    def test_campaign_rollup_bindings(self):
        receipts = {key: {'schema': 'revision2.optimized-hccl-completion.v1', 'status': 'PASS',
                          'group': key, 'arm': plan.GROUP_ARM[key], 'world': 16,
                          'ladder_level': 'L0', 'manifest_sha256': 'a' * 64,
                          'review_sha256': 'b' * 64, 'image_id': plan.IMAGE,
                          'duration_seconds': 120.0,
                          'group_median_ms': 200.0 + 5.0 * plan.GROUPS.index(key)}
                    for key in plan.GROUPS}
        rollup = plan.campaign_rollup(receipts, 'L0')
        self.assertEqual(sorted(rollup['arms']), sorted(plan.ARMS))
        self.assertIn('C3', rollup['contrasts'])
        self.assertEqual(rollup['contrasts']['C3']['pair'], 'F vs HOPT')
        self.assertTrue(rollup['historical_drift_context_only']['policy'].startswith('never pooled'))
        with self.assertRaises(ValueError):
            plan.campaign_rollup({**receipts, 'g16': {**receipts['g16'], 'arm': 'H'}}, 'L0')

    def test_completion_validation(self):
        receipt = {'schema': 'revision2.optimized-hccl-completion.v1', 'status': 'PASS',
                   'group': 'g01', 'arm': 'H', 'world': 16, 'ladder_level': 'L0',
                   'manifest_sha256': 'a' * 64, 'review_sha256': 'b' * 64,
                   'image_id': plan.IMAGE, 'duration_seconds': 120.0,
                   'group_median_ms': 200.0}
        plan.validate_completion(receipt, 'g01', 'a' * 64, 'b' * 64)
        for mutate in ('status', 'arm', 'duration', 'median'):
            broken = json.loads(json.dumps(receipt))
            if mutate == 'status':
                broken['status'] = 'FAIL'
            elif mutate == 'arm':
                broken['arm'] = 'F'
            elif mutate == 'duration':
                broken['duration_seconds'] = 0
            else:
                del broken['group_median_ms']
            with self.assertRaises(ValueError):
                plan.validate_completion(broken, 'g01', 'a' * 64, 'b' * 64)


class RestorationTests(unittest.TestCase):
    def test_fake_import_proof(self):
        proof = restoration.fake_import_proof()
        self.assertEqual(proof['proof'], 'cpu_fake_import_restoration')
        self.assertEqual(len(proof['binding_names']), 10)
        self.assertTrue(proof['hook_layer_retained'])
        self.assertTrue(proof['double_restore_refused'])
        self.assertTrue(proof['forged_chain_rejected'])

    def test_binding_names_are_the_native_ten(self):
        self.assertEqual(restoration.NAMES, (
            'runner.execute_model', 'runner._dummy_run', 'context.forward', 'runner_module.forward',
            'context.selector', 'dispatcher.helper', 'all2all.dispatch', 'all2all.combine',
            'mc2.dispatch', 'mc2.combine'))

    def test_cyclic_chain(self):
        from types import CodeType  # noqa: F401
        import types

        class Fn:
            def __init__(self, name):
                self.__code__ = types.SimpleNamespace(co_name=name, co_filename='x',
                                                      co_firstlineno=1, co_code=b'\x01')

        a, b = Fn('a'), Fn('b')
        a.__wrapped__ = b
        b.__wrapped__ = a
        with self.assertRaises(restoration.RestorationError):
            restoration.chain_of(a)


class ProviderAndSchemaTests(unittest.TestCase):
    def test_pace_schema_truth_fixture(self):
        parsed = plan.check_pace_schema(PACE_SCHEMA_TRUTH)
        self.assertEqual(parsed['name'], plan.PACE_OPERATOR_NAME)
        self.assertEqual(len(parsed['arguments']), 8)
        self.assertEqual(parsed['returns'], ['Tensor'])
        for label, bad in (('namespace', PACE_SCHEMA_TRUTH.replace('cashflex::', '')),
                           ('arg type', PACE_SCHEMA_TRUTH.replace('int _6', 'str _6')),
                           ('return type', PACE_SCHEMA_TRUTH.replace('-> Tensor _0', '-> int _0')),
                           ('arity', PACE_SCHEMA_TRUTH.replace(', int _7', ''))):
            with self.assertRaises(ValueError, msg=label):
                plan.check_pace_schema(bad)

    def test_dladdr_fname_only(self):
        self.assertEqual(plan.dladdr_provider_path('/usr/lib/libshmem.so'), '/usr/lib/libshmem.so')
        with self.assertRaises(ValueError):
            plan.dladdr_provider_path(None)
        with self.assertRaises(ValueError):
            plan.dladdr_provider_path('')

    def test_discover_acl(self):
        path = '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so'
        text = '\n'.join(['7f00-7f10 r--p 0 fd:01 1 ' + path, '7f10-7f30 r-xp 0 fd:01 1 ' + path,
                          '7f00 r--p 0 fd:01 2 /usr/lib/libother.so'])
        self.assertEqual(plan.discover_acl_library(text), path)
        with self.assertRaises(ValueError):
            plan.discover_acl_library('7f00 r--p 0 fd:01 2 /usr/lib/libother.so')

    def test_exact_ns_ordering(self):
        self.assertEqual(plan.exact_ns('1970-01-01T00:00:01Z'), 1_000_000_000)
        self.assertEqual(plan.exact_ns('1970-01-01T00:00:00.5Z'), 500_000_000)
        with self.assertRaises(ValueError):
            plan.exact_ns('not-a-time')
        plan.check_time_order('2026-11-05T01:00:00Z', '2026-11-05T01:01:00Z',
                              '2026-11-05T01:01:01Z', '2026-11-05T01:05:00Z')
        with self.assertRaises(ValueError):
            plan.check_time_order('2026-11-05T01:02:00Z', '2026-11-05T01:01:00Z',
                                  '2026-11-05T01:01:01Z', '2026-11-05T01:05:00Z')


class SupervisorTests(unittest.TestCase):
    def test_natural_flow(self):
        output = []

        class Child:
            pid = 4242
            returncode = 0

            def __init__(self):
                self.calls = 0

            def poll(self):
                self.calls += 1
                return None if self.calls < 3 else 0

        child = Child()

        def popen(argv):
            self.assertTrue(argv[-1].endswith('/workspace/lane/worker.py'))
            return child

        code = supervise(entry='/workspace/lane/worker.py', popen=popen,
                         live=lambda: [], sleep=lambda x: None, emit=output.append, reap=False)
        self.assertEqual(code, 0)
        self.assertEqual(output[0]['event'], 'driver_started')
        self.assertEqual(output[-1]['event'], 'supervisor_complete')

    def test_live_peers_are_preserved(self):
        state = {'peers': [{'pid': 9, 'name': 'vllm', 'state': 'S'}], 'calls': 0}

        class Child:
            pid = 1
            returncode = 0

            def poll(self):
                return 0

        def live():
            state['calls'] += 1
            return [] if state['calls'] > 2 else state['peers']

        output = []
        code = supervise(entry='x', popen=lambda argv: Child(), live=live,
                         sleep=lambda x: None, emit=output.append, reap=False)
        self.assertEqual(code, 0)
        self.assertTrue(any(row['event'] == 'peers_still_live' for row in output))


class LaunchContractTests(unittest.TestCase):
    def test_mount_map_and_env_helpers_pure(self):
        import launch
        mounts = launch.expected_mounts(Path('/tmp/payload'))
        self.assertTrue(all(not rw for target, (_, rw) in mounts.items()
                            if target != '/workspace/results'))
        self.assertTrue(mounts['/workspace/results'][1])
        self.assertEqual(set(mounts) & {'/workspace/lane', '/workspace/plan',
                                        '/workspace/reference', '/workspace/model_hook'},
                         {'/workspace/lane', '/workspace/plan', '/workspace/reference',
                          '/workspace/model_hook'})

    def test_probe_entry_uses_heredoc_and_value(self):
        import launch
        script = launch.probe_entry_script()
        self.assertIn('REV4PROBE', script)
        self.assertIn("os.environ.get('HCCL_BUFFSIZE')", script)
        self.assertIn('PROBE_PAYLOAD_DIR', script)

    def test_group_env_shapes(self):
        import launch
        acl = {'path': '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so',
               'sha256': '0' * 64}
        env = launch.group_env('g03', 'F', 'L0', acl, {}, 'a' * 64, 'b' * 64)
        self.assertIn('HPA_SHMEM_IP_PORT', env)
        self.assertEqual(env['REV4_ARM'], 'F')
        self.assertEqual(env['VLLM_ASCEND_ENABLE_FUSED_MC2'], '0')
        hccl = launch.group_env('g01', 'H', 'L0', acl, {}, 'a' * 64, 'b' * 64)
        self.assertNotIn('HPA_SHMEM_IP_PORT', hccl)
        hopt = launch.group_env('g04', 'HOPT', 'L0', acl, {'HCCL_BUFFSIZE': '512'},
                                'a' * 64, 'b' * 64)
        self.assertEqual(hopt['HCCL_BUFFSIZE'], '512')
        self.assertEqual(launch.nonce('g01'), launch.nonce('g01'))
        self.assertNotEqual(launch.nonce('g01'), launch.nonce('g02'))

    def test_f_runtime_family_wiring(self):
        """f_proof failure 54167a66 regression: the sync-model runtime family
        must be mounted and on PYTHONPATH before model_hook."""
        import launch
        qualification_host = WORKSPACE / 'analysis/atlas_olmoe_o1_sync_qualification_v3'
        self.assertEqual(sha_file(qualification_host / 'source_manifest.json'),
                         plan.QUALIFICATION_MANIFEST)
        mounts = dict((host, target) for host, target, _ in launch.fixed_mounts())
        self.assertEqual(mounts.get(launch.QUALIFICATION), '/workspace/qualification')
        script = launch.entry_script('/workspace/lane/worker.py')
        self.assertIn('/workspace/qualification', script)
        self.assertLess(script.index('/workspace/qualification'),
                        script.index('/workspace/model_hook'),
                        'family dir must precede model_hook on PYTHONPATH')
        self.assertLess(script.index('/workspace/qualification'),
                        script.index('/workspace/queue_metadata'),
                        'family-internal queue_metadata sibling must win')
        # family completeness against the local mirror of the qualification dir
        present = {p.name for p in qualification_host.iterdir() if p.is_file()}
        missing = [member for member in plan.F_RUNTIME_FAMILY if member not in present]
        self.assertEqual(missing, [], 'sync-model runtime family incomplete locally')
        # every family member is covered by the qualification lane's own manifest
        manifest = json.loads((qualification_host / 'source_manifest.json').read_text())
        self.assertEqual(set(plan.F_RUNTIME_FAMILY) - set(manifest), set())

    def test_f_worker_env_priming_under_spawn_semantics(self):
        """e9f94a35 regression: the install RPC runs inside a vLLM-spawned
        worker whose environ is the parent's snapshot at spawn time. Prime
        with a store holding ONLY docker-level vars (spawn simulation):
        the manifest pin is filled, group-env names resolve; a store with
        nothing fails closed; pre-set values win over pins (setdefault)."""
        docker_env = {'REV2_RUN_NONCE': 'a' * 32,
                      'HPA_SHMEM_IP_PORT': 'tcp://127.0.0.1:26807',
                      'REV2_QUALIFICATION_COMPLETION_SHA256': plan.MIXED_COMPLETION_SHA}
        primed = plan.prime_f_worker_env(dict(docker_env))
        self.assertEqual(primed['REV2_MODEL_MANIFEST_SHA256'], plan.SYNC_MODEL_MANIFEST)
        self.assertEqual(primed['REV2_QUALIFICATION_COMPLETION_SHA256'],
                         plan.MIXED_COMPLETION_SHA)
        # setdefault semantics: an explicit value is never overwritten
        explicit = dict(docker_env, REV2_MODEL_MANIFEST_SHA256='f' * 64)
        self.assertEqual(plan.prime_f_worker_env(explicit)['REV2_MODEL_MANIFEST_SHA256'],
                         'f' * 64)
        # nothing present -> fail closed naming the group-env variables
        # (the pin is fillable, the rendezvous env is not)
        with self.assertRaises(ValueError) as caught:
            plan.prime_f_worker_env({})
        message = str(caught.exception)
        self.assertIn('REV2_RUN_NONCE', message)
        self.assertIn('HPA_SHMEM_IP_PORT', message)
        self.assertNotIn('REV2_MODEL_MANIFEST_SHA256', message)

    def test_f_env_priming_wiring_order(self):
        """The prime call sits BEFORE the frozen super() install (worker
        process), and the driver sets the F pins BEFORE LLM construction
        (parent process, pre-spawn snapshot)."""
        worker_source = (ANALYSIS / 'atlas_optimized_hccl_lane_20261105_v1'
                         / 'worker.py').read_text()
        prime_call = worker_source.index('plan.prime_f_worker_env(os.environ)')
        super_call = worker_source.index('super().install_revision2_sync_model()')
        self.assertLess(prime_call, super_call)
        pin_loop = worker_source.index("for name, value in plan.delegated_env('F').items():")
        llm_construction = worker_source.index('llm = LLM(')
        self.assertLess(pin_loop, llm_construction,
                        'F pins must be set before LLM spawns worker processes')

    def test_delegated_env_contract_wiring(self):
        """g01 regression: the delegated chain's env vars are audited
        (plan.DELEGATED_ENV), worker source sets every worker-setter var,
        and simulated arm envs are complete with plan pins."""
        worker_source = (ANALYSIS / 'atlas_optimized_hccl_lane_20261105_v1'
                         / 'worker.py').read_text()
        for name, arms, _value, setter in plan.DELEGATED_ENV:
            self.assertTrue(arms, f'{name} bound to no arm')
        # the worker sets every worker-setter var through the generic
        # plan.delegated_env loop in BOTH arm paths (no per-name literals
        # to drift); the loop itself is the wiring under test
        self.assertGreaterEqual(worker_source.count("plan.delegated_env("), 2)
        self.assertIn('os.environ[name] = str(value)', worker_source)
        # worker-set pins resolve from plan constants
        self.assertEqual(plan.delegated_env('H'), {
            'REV2_NATIVE_TIMING_MANIFEST': plan.TIMING_MANIFEST,
            'REV2_NO_RESIZE_SOURCE_MANIFEST': plan.RUNNER_MANIFEST,
            'REV2_NO_RESIZE_ARM': 'native'})
        self.assertEqual(plan.delegated_env('HOPT')['REV2_NO_RESIZE_ARM'], 'no-resize')
        self.assertEqual(plan.delegated_env('HmR')['REV2_NO_RESIZE_ARM'], 'no-resize')
        self.assertNotIn('REV2_NO_RESIZE_ARM', plan.delegated_env('F'))
        self.assertEqual(plan.delegated_env('F'),
                         {'REV2_MODEL_MANIFEST_SHA256': plan.SYNC_MODEL_MANIFEST})
        for arm in plan.ARMS:
            self.assertTrue(plan.required_env_names(arm))
        # the audit list covers every variable the frozen chain reads
        audited = {name for name, _a, _v, _s in plan.DELEGATED_ENV}
        for required in ('REV2_NATIVE_TIMING_MANIFEST', 'REV2_NO_RESIZE_SOURCE_MANIFEST',
                         'REV2_NO_RESIZE_ARM', 'REV2_BACKEND', 'REV2_OUTPUT_TOKENS',
                         'REV2_RESULT_DIR', 'VLLM_USE_V1', 'VLLM_ASCEND_ENABLE_FUSED_MC2',
                         'REV2_MODEL_MANIFEST_SHA256', 'REV2_RUN_NONCE',
                         'HPA_SHMEM_IP_PORT', 'REV2_QUALIFICATION_COMPLETION_SHA256'):
            self.assertIn(required, audited)
        # simulated group env + worker pins = complete per-arm env
        import launch
        acl_stub = {'path': '/x', 'sha256': '0' * 64}
        for arm in plan.ARMS:
            group_key = next(key for key in plan.GROUPS if plan.GROUP_ARM[key] == arm)
            simulated = dict(launch.group_env(group_key, arm, 'L0', acl_stub, {},
                                              '0' * 64, '0' * 64))
            simulated.update({k: str(v) for k, v in plan.delegated_env(arm).items()})
            simulated['REV2_NO_RESIZE_ARM'] = plan.DRIVER_ARM_ENV.get(arm, simulated.get(
                'REV2_NO_RESIZE_ARM', ''))
            for name in plan.required_env_names(arm):
                self.assertNotEqual(simulated.get(name, ''), f'{arm}/{name} empty')

    def test_delegated_data_files_wiring(self):
        """d4e26ece regression: the delegated chain's path+SHA data files are
        audited (plan.DELEGATED_DATA_FILES), their host homes mounted, and
        every locally verifiable pin matches ground truth."""
        import launch
        mounts = dict((host, target) for host, target, _ in launch.fixed_mounts())
        self.assertEqual(mounts.get(launch.LEGACY), '/workspace/legacy')
        self.assertEqual(mounts.get(launch.MODEL), '/workspace/model')
        self.assertEqual(mounts.get(launch.MIXED_RESULTS), '/workspace/mixed_qualification')
        self.assertEqual(mounts.get(launch.QUALIFICATION), '/workspace/qualification')
        # qualification must be the v3 HARNESS (model_contract binds f3b36ae0)
        self.assertEqual(sha_file(ANALYSIS / 'atlas_olmoe_o1_sync_qualification_v3'
                                  / 'source_manifest.json'), plan.QUALIFICATION_MANIFEST)
        # mixed qualification receipt pin vs the local frozen result
        self.assertEqual(sha_file(WORKSPACE / 'analysis_results/atlas_olmoe_o1_sync_'
                                  'qualification_20261028_v3/p16/completion.json'),
                         plan.MIXED_COMPLETION_SHA)
        # prompt-helper pin equals the frozen chain's own PROMPT_SOURCE_SHA
        run_capture = (ANALYSIS / 'atlas_model_split_capture_v1/run_capture.py').read_text()
        self.assertIn(plan.PROMPT_HELPER_SHA, run_capture)
        # the audit covers every audited container path with the right kinds
        paths = {path for path, _kind, _pin, _c in plan.DELEGATED_DATA_FILES}
        self.assertIn('/workspace/legacy/benchmark_olmoe_fixed_batch.py', paths)
        self.assertIn('/workspace/mixed_qualification/completion.json', paths)
        self.assertIn('/workspace/reference/request_raw.json', paths)
        kinds = {path: kind for path, kind, _pin, _c in plan.DELEGATED_DATA_FILES}
        self.assertEqual(kinds['/workspace/model'], 'dir')
        # F env binds the mixed receipt digest for model_contract.sources()
        acl = {'path': '/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so',
               'sha256': '0' * 64}
        env = launch.group_env('g03', 'F', 'L0', acl, {}, 'a' * 64, 'b' * 64)
        self.assertEqual(env['REV2_QUALIFICATION_COMPLETION_SHA256'], plan.MIXED_COMPLETION_SHA)

    def test_probe_contract_accepts_conforming_container(self):
        # ADVISORY-1 regression: probe_audit must call inspect_contract WITH
        # guard/world/payload (probe containers run host net/ipc, 16 devices
        # and the rw payload mount); a probe-shaped container now passes.
        import launch
        payload = Path('/tmp/probe-payload')
        devices = list(guard.nodes(plan.WORLD))
        mounts = launch.expected_mounts(payload)
        env = {'WORLD_SIZE': '16', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '26833',
               'RANK': '0', 'ASCEND_RT_VISIBLE_DEVICES': ','.join(map(str, range(16))),
               'HCCL_BUFFSIZE': plan.PROBE_VALUE, 'PROBE_PAYLOAD_DIR': '/workspace/results',
               'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
        argv = ['docker', 'run', plan.IMAGE, 'bash', '-lc', launch.probe_entry_script()]
        info = {'Image': plan.IMAGE,
                'HostConfig': {'NetworkMode': 'host', 'IpcMode': 'host',
                               'Devices': [{'PathOnHost': d, 'PathInContainer': d,
                                            'CgroupPermissions': 'rwm'} for d in devices]},
                'State': {'Running': False, 'ExitCode': 0, 'OOMKilled': False, 'Error': ''},
                'Config': {'Env': [f'{k}={v}' for k, v in env.items()], 'Cmd': argv[-3:]},
                'Mounts': [{'Destination': t, 'Source': s, 'RW': w}
                           for t, (s, w) in mounts.items()]}
        launch.inspect_contract(info, argv, env, guard, plan.WORLD, payload)
        # the previously broken no-guard call form must reject the same info
        with self.assertRaises(RuntimeError):
            launch.inspect_contract(info, argv, env)

    def test_probe_mount_target_is_results(self):
        import launch
        mounts = launch.expected_mounts(Path('/tmp/x'))
        self.assertIn('/workspace/results', mounts)
        self.assertNotIn('/workspace/probe_results', mounts)

    def test_tree_coverage_rejects_strays(self):
        import launch
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in plan.FILES:
                (root / name).write_text('x\n')
            (root / 'source_manifest.json').write_text('{}\n')
            (root / 'review.json').write_text('{}\n')
            launch.check_source_tree(root)
            (root / 'scratch').mkdir()
            with self.assertRaises(ValueError):
                launch.check_source_tree(root)
            (root / 'scratch').rmdir()
            (root / 'old.py').write_text('x\n')
            with self.assertRaises(ValueError):
                launch.check_source_tree(root)
            (root / 'old.py').unlink()
            cache = root / '__pycache__'
            cache.mkdir()
            (cache / 'plan.pyc').write_bytes(b'\x00')
            launch.check_source_tree(root)
            # ADVISORY-4: symlinks are now rejected UNCONDITIONALLY, even
            # inside __pycache__ (only regular files are permitted there).
            (cache / 'alias.pyc').symlink_to(root / 'plan.py')
            with self.assertRaises(ValueError):
                launch.check_source_tree(root)
            (cache / 'alias.pyc').unlink()
            launch.check_source_tree(root)


class GuardParserTests(unittest.TestCase):
    def test_all_idle_observed_table_passes(self):
        parsed = guard.parse_npu_info(fixture)
        self.assertEqual(parsed['empty_npu_cards'], list(range(8)))
        self.assertEqual(parsed['covered_die_count'], 16)

    def test_occupied_or_missing_fails_closed(self):
        occupied = fixture.replace('No running processes found in NPU 3',
                                   '3       0                 | 4242      | intruder         | 512')
        with self.assertRaises(RuntimeError):
            guard.parse_npu_info(occupied)
        missing = '\n'.join(line for line in fixture.splitlines()
                            if 'No running processes found in NPU 7' not in line)
        with self.assertRaises(RuntimeError):
            guard.parse_npu_info(missing)


if __name__ == '__main__':
    unittest.main()
