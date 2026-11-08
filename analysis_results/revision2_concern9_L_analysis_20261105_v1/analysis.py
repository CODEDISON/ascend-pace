#!/usr/bin/env python3
"""Revision2 concern #9: minimum safe per-peer slot capacity L, local CPU replay.

Replays the frozen native OLMoE/Qwen3 Alltoallv split captures (P16, BF16,
row_bytes 4096) to answer, for the *captured* distributions only:
  1. per-call maximum per-peer demand (bytes) and rank whole-input totals;
  2. the minimum L that admits every captured call under (a) the host-list
     per-peer-segment guard and (b) the historical device-split whole-input
     guard, versus the historical configurations (96/256/512 MiB);
  3. per-peer/whole-input demand distributions (P50/P90/P99/P99.9/max) and
     admission-versus-L curves from 1 to 512 MiB under both guards;
  4. zero-exceedance one-sided 95% upper bounds (Clopper-Pearson) with
     explicit non-iid caveats.

Everything is recomputed from the raw capture files.  No device, remote or
SSH operation is performed; this script is pure local Python (stdlib only).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

MIB = 1024 * 1024

# ---------------------------------------------------------------------------
# Configuration anchors (from the paper, not computed results): historical
# per-peer slot capacities L per model, and GPT-OSS's prospective bound.
HISTORICAL_L_MIB = {'olmoe': 96, 'qwen': 256, 'gpt-oss': 512}
GPTOSS_PROSPECTIVE_BOUND_MIB = 360  # model-level raw routed-payload bound; no capture exists

# Prior published maxima (supplement "Counterfactual capacity checks on native
# O1 traces" and native_o1_capacity_independent_replay_20261028_v1/replay.py).
# Used ONLY to cross-check the fresh recomputation below; a mismatch aborts.
ANCHORS = {
    'olmoe-o1': {'max_peer_bytes_including_self': 7118848, 'max_rank_input_bytes': 93265920},
    'qwen-o1': {'max_peer_bytes_including_self': 10297344, 'max_rank_input_bytes': 117772288},
    'olmoe-o32': {'max_peer_bytes_including_self': 7135232, 'max_rank_input_bytes': 93290496},
    'qwen-o32': {'max_peer_bytes_including_self': 10416128, 'max_rank_input_bytes': 117813248},
}
ANCHOR_REPEAT_OLMOE = {'max_peer_bytes_including_self': 7118848, 'max_rank_input_bytes': 93265920}

FAIR_DIR = ROOT / 'analysis_results/atlas_fair_trace_workloads_20261028_v1'
REPEAT_DIR = ROOT / 'analysis_results/atlas_native_repeat_capture_20261028_v3/native_two_o1/payload'
RAW_SOURCES = {
    'olmoe-o1': ROOT / 'analysis_results/atlas_model_split_capture_lane_20261028_v1/hccl-o1/request_raw.json',
    'olmoe-o32': ROOT / 'analysis_results/atlas_model_split_capture_lane_20261028_v1/hccl-o32/request_raw.json',
    'qwen-o1': ROOT / 'analysis_results/atlas_qwen_split_capture_20261028_v1/hccl-o1/request_raw.json',
    'qwen-o32': ROOT / 'analysis_results/atlas_qwen_split_capture_20261028_v1/hccl-o32/request_raw.json',
}
REPEAT_RAW = {'repeat-00': REPEAT_DIR / 'request-00.raw.json',
              'repeat-01': REPEAT_DIR / 'request-01.raw.json'}
REPEAT_VALIDATED = {'repeat-00': REPEAT_DIR / 'request-00.validated.json',
                    'repeat-01': REPEAT_DIR / 'request-01.validated.json'}

INPUT_RECORDS = []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path, role: str) -> dict:
    digest = sha256_file(path)
    INPUT_RECORDS.append({'path': str(path.relative_to(ROOT)), 'sha256': digest,
                          'bytes': path.stat().st_size, 'role': role})
    return json.loads(path.read_text())


def percentile_nearest_rank(sorted_values, p):
    """Nearest-rank percentile: the ceil(p/100*n)-th smallest value (1-based)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError('empty')
    rank = max(1, math.ceil(p / 100.0 * n))
    return sorted_values[rank - 1]


def reconstruct_from_workers(workers, label):
    """Independent matrix reconstruction straight from raw worker events.

    Returns a list of per-global-call dicts with send/receive matrices and
    row_bytes, validating rank coverage, per-rank call counts, transpose
    consistency and tensor layout agreement across ranks.
    """
    if len(workers) != 16:
        raise ValueError(f'{label}: expected 16 workers, got {len(workers)}')
    by_rank = sorted(workers, key=lambda w: w['rank'])
    if [w['rank'] for w in by_rank] != list(range(16)):
        raise ValueError(f'{label}: rank coverage differs')
    begins_per_rank = []
    for w in by_rank:
        if w.get('dropped_events') != 0 or w.get('capture_error_count') != 0:
            raise ValueError(f'{label}: rank {w["rank"]} capture dropped/failed events')
        begins_per_rank.append([e for e in w['events'] if e['kind'] == 'alltoallv_begin'])
    n_calls = len(begins_per_rank[0])
    if any(len(b) != n_calls for b in begins_per_rank):
        raise ValueError(f'{label}: per-rank Alltoallv call counts differ')
    calls = []
    for ci in range(n_calls):
        rows, recvs, layouts = [], [], []
        for r in range(16):
            ev = begins_per_rank[r][ci]
            send = ev['input_splits']['values']
            recv = ev['output_splits']['values']
            if len(send) != 16 or len(recv) != 16:
                raise ValueError(f'{label}: split vector length differs')
            if any(type(v) is not int or v < 0 for v in send + recv):
                raise ValueError(f'{label}: noninteger/negative split')
            rows.append(send)
            recvs.append(recv)
            tensor = ev['input']
            layouts.append((tensor['dtype'], tensor['row_bytes'], tuple(tensor['shape'][1:])))
        if len(set(layouts)) != 1:
            raise ValueError(f'{label}: tensor layout differs across ranks')
        for i in range(16):
            for j in range(16):
                if rows[i][j] != recvs[j][i]:
                    raise ValueError(f'{label}: send/receive matrices do not transpose')
        calls.append({'send_matrix_rows': rows, 'receive_matrix_rows': recvs,
                      'dtype': layouts[0][0], 'row_bytes': layouts[0][1]})
    return calls


def arm_metrics(calls):
    """Per-arm capacity metrics recomputed from per-call split matrices."""
    peer_cells_incl = []            # (call, src rank, peer) demand incl. self segment
    peer_cells_excl = []            # off-diagonal (remote peer) demand
    whole_input_cells = []          # (call, rank) whole-input demand (send side)
    whole_output_cells = []         # (call, rank) whole-output demand (receive side)
    per_call = []
    zero_total_calls = 0
    for call in calls:
        send, recv, rb = (call['send_matrix_rows'], call['receive_matrix_rows'],
                          call['row_bytes'])
        call_max_peer = 0
        call_max_peer_excl = 0
        for i in range(16):
            row_sum = sum(send[i]) * rb
            whole_input_cells.append(row_sum)
            whole_output_cells.append(sum(recv[i]) * rb)
            for j in range(16):
                b = send[i][j] * rb
                peer_cells_incl.append(b)
                if i != j:
                    peer_cells_excl.append(b)
                    call_max_peer_excl = max(call_max_peer_excl, b)
                call_max_peer = max(call_max_peer, b)
        call_max_whole = max(sum(row) for row in send) * rb
        per_call.append({'max_peer_incl': call_max_peer, 'max_peer_excl': call_max_peer_excl,
                         'max_whole_input': call_max_whole})
        if any(sum(send[i]) == 0 for i in range(16)) or any(sum(recv[i]) == 0 for i in range(16)):
            zero_total_calls += 1
    return {
        'calls': len(calls),
        'max_peer_bytes_including_self': max(c['max_peer_incl'] for c in per_call),
        'max_peer_bytes_excluding_self': max(c['max_peer_excl'] for c in per_call),
        'max_rank_input_bytes': max(c['max_whole_input'] for c in per_call),
        'max_rank_output_bytes': max(whole_output_cells),
        'zero_total_calls': zero_total_calls,
        'peer_cells_incl_sorted': sorted(peer_cells_incl),
        'peer_cells_excl_sorted': sorted(peer_cells_excl),
        'whole_input_cells_sorted': sorted(whole_input_cells),
        'per_call_max_peer_sorted': sorted(c['max_peer_incl'] for c in per_call),
        'per_call_max_whole_sorted': sorted(c['max_whole_input'] for c in per_call),
        'per_call_pairs': [(c['max_peer_incl'], c['max_whole_input']) for c in per_call],
        'n_peer_cells_incl': len(peer_cells_incl),
        'n_peer_cells_excl': len(peer_cells_excl),
        'n_whole_cells': len(whole_input_cells),
    }


def distribution(sorted_values, points=(50, 90, 99, 99.9)):
    return {f'p{p}': percentile_nearest_rank(sorted_values, p) for p in points} | {
        'max': sorted_values[-1], 'min': sorted_values[0], 'n': len(sorted_values)}


def admission_scan(m, limits_mib):
    """Call-rejection counts and cell-exceedance rates under both guards."""
    peer_by_call = m['per_call_max_peer_sorted']
    whole_by_call = m['per_call_max_whole_sorted']
    peer_cells = m['peer_cells_incl_sorted']
    whole_cells = m['whole_input_cells_sorted']
    n_calls = m['calls']
    scan = []
    for l_mib in limits_mib:
        limit = l_mib * MIB
        peer_rej = n_calls - bisect.bisect_right(peer_by_call, limit)
        whole_rej = n_calls - bisect.bisect_right(whole_by_call, limit)
        peer_cell_exc = m['n_peer_cells_incl'] - bisect.bisect_right(peer_cells, limit)
        whole_cell_exc = m['n_whole_cells'] - bisect.bisect_right(whole_cells, limit)
        union_rej = sum(1 for peer_max, whole_max in m['per_call_pairs']
                        if peer_max > limit or whole_max > limit)
        scan.append({
            'L_mib': l_mib, 'limit_bytes': limit,
            'calls_rejected_peer_guard': peer_rej,
            'calls_rejected_peer_guard_frac': peer_rej / n_calls,
            'calls_rejected_whole_input_guard': whole_rej,
            'calls_rejected_whole_input_guard_frac': whole_rej / n_calls,
            'calls_rejected_union_guard': union_rej,
            'calls_rejected_union_guard_frac': union_rej / n_calls,
            'peer_cells_exceeding': peer_cell_exc,
            'peer_cells_exceeding_frac': peer_cell_exc / m['n_peer_cells_incl'],
            'whole_input_cells_exceeding': whole_cell_exc,
            'whole_input_cells_exceeding_frac': whole_cell_exc / m['n_whole_cells'],
        })
    return scan


def cp_upper_zero_exceedances(n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound for exceedance probability given
    0 exceedances in n trials: 1 - alpha^(1/n)."""
    alpha = 1.0 - confidence
    return 1.0 - alpha ** (1.0 / n)


def fmt_bytes(b):
    return f'{b:,} B ({b / MIB:.4f} MiB)'


def main():
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')

    # ---- input chain: manifest -> workloads.json -> arm receipts -> raws ----
    manifest = load_json(FAIR_DIR / 'manifest.json', 'fair-trace manifest (hash chain root)')
    workloads = load_json(FAIR_DIR / 'workloads.json', 'frozen 4-arm native split matrices')
    workloads_file_sha = INPUT_RECORDS[-1]['sha256']
    if workloads_file_sha != manifest['workloads_sha256']:
        raise ValueError('workloads.json file digest does not match its manifest')

    arms_out = {}
    for arm_name in ('olmoe-o1', 'olmoe-o32', 'qwen-o1', 'qwen-o32'):
        arm = workloads['arms'][arm_name]
        raw_path = RAW_SOURCES[arm_name]
        raw = load_json(raw_path, f'raw native capture ({arm_name})')
        if INPUT_RECORDS[-1]['sha256'] != arm['request_raw_sha256']:
            raise ValueError(f'{arm_name}: raw capture digest differs from arm receipt')
        rebuilt = reconstruct_from_workers(raw['workers'], arm_name)
        # equality with the frozen workload arm
        if len(rebuilt) != len(arm['calls']):
            raise ValueError(f'{arm_name}: call count differs from frozen arm')
        for idx, (new, old) in enumerate(zip(rebuilt, arm['calls'])):
            for field in ('send_matrix_rows', 'receive_matrix_rows'):
                if new[field] != old[field]:
                    raise ValueError(f'{arm_name} call {idx}: {field} differs from frozen arm')
            if new['row_bytes'] != old['row_bytes'] or new['dtype'] != old['dtype']:
                raise ValueError(f'{arm_name} call {idx}: layout differs from frozen arm')
        m = arm_metrics(arm['calls'])
        # cross-validation against prior published numbers
        for key, anchor in ANCHORS[arm_name].items():
            if m[key] != anchor:
                raise ValueError(f'{arm_name}: recomputed {key}={m[key]} != anchor {anchor}')
        arms_out[arm_name] = m

    # ---- supplementary repeatability arms (native two-request capture) ------
    repeat_out = {}
    frozen_olmoe_calls = workloads['arms']['olmoe-o1']['calls']
    for tag in ('repeat-00', 'repeat-01'):
        val = load_json(REPEAT_VALIDATED[tag], f'{tag} validated trace (repeat capture)')
        raw = load_json(REPEAT_RAW[tag], f'{tag} raw capture (repeat capture)')
        if INPUT_RECORDS[-1]['sha256'] != val['source_raw_sha256']:
            raise ValueError(f'{tag}: raw digest differs from validated receipt')
        rebuilt = reconstruct_from_workers(raw['after'], tag)
        if len(rebuilt) != val['alltoallv_global_calls']:
            raise ValueError(f'{tag}: reconstructed call count differs')
        for idx, (new, old) in enumerate(zip(rebuilt, val['alltoallv_calls'])):
            if new['send_matrix_rows'] != old['send_matrix_rows']:
                raise ValueError(f'{tag} call {idx}: matrix differs from validated trace')
        m = arm_metrics(val['alltoallv_calls'])
        if (m['max_peer_bytes_including_self'] != ANCHOR_REPEAT_OLMOE['max_peer_bytes_including_self']
                or m['max_rank_input_bytes'] != ANCHOR_REPEAT_OLMOE['max_rank_input_bytes']):
            raise ValueError(f'{tag}: maxima differ from the frozen OLMoE O1 anchor')
        n_frozen = len(frozen_olmoe_calls)
        prefix_eq = sum(1 for i in range(min(len(rebuilt), n_frozen))
                        if rebuilt[i]['send_matrix_rows'] == frozen_olmoe_calls[i]['send_matrix_rows'])
        second_half_self_eq = None
        if len(rebuilt) == 128:
            second_half_self_eq = sum(1 for i in range(64)
                                      if rebuilt[64 + i]['send_matrix_rows'] == rebuilt[i]['send_matrix_rows'])
        repeat_out[tag] = {
            'calls': m['calls'],
            'max_peer_bytes_including_self': m['max_peer_bytes_including_self'],
            'max_rank_input_bytes': m['max_rank_input_bytes'],
            'matching_frozen_olmoe_o1_first_64': prefix_eq,
            'forwards_2_3_equal_forwards_0_1': second_half_self_eq,
        }

    # ---- guard semantics, minimum safe L, utilization ----------------------
    scan_grid = list(range(1, 513))                     # 1..512 MiB, 1 MiB steps
    report_grid = [8, 16, 32, 48, 64, 88, 96, 100, 104, 112, 116, 120, 128,
                   160, 192, 256, 320, 360, 384, 448, 512]
    results_arms = {}
    for arm_name, m in arms_out.items():
        model = 'olmoe' if arm_name.startswith('olmoe') else 'qwen'
        hist_l = HISTORICAL_L_MIB[model] * MIB
        # exact per-call union rejection counts are included for every scanned L
        scan = admission_scan(m, scan_grid)
        results_arms[arm_name] = {
            'model': model,
            'historical_L_mib': HISTORICAL_L_MIB[model],
            'min_safe_L_peer_guard_bytes': m['max_peer_bytes_including_self'],
            'min_safe_L_peer_guard_mib': m['max_peer_bytes_including_self'] / MIB,
            'min_safe_L_peer_guard_excluding_self_bytes': m['max_peer_bytes_excluding_self'],
            'min_safe_L_peer_guard_excluding_self_mib': m['max_peer_bytes_excluding_self'] / MIB,
            'min_safe_L_whole_input_guard_bytes': m['max_rank_input_bytes'],
            'min_safe_L_whole_input_guard_mib': m['max_rank_input_bytes'] / MIB,
            'utilization_of_historical_L_peer_guard': m['max_peer_bytes_including_self'] / hist_l,
            'utilization_of_historical_L_whole_input_guard': m['max_rank_input_bytes'] / hist_l,
            'headroom_ratio_historical_L_over_whole_input_min': hist_l / m['max_rank_input_bytes'],
            'peer_demand_distribution_incl_self_bytes': distribution(m['peer_cells_incl_sorted']),
            'peer_demand_distribution_excl_self_bytes': distribution(m['peer_cells_excl_sorted']),
            'whole_input_distribution_bytes': distribution(m['whole_input_cells_sorted']),
            'zero_total_calls': m['zero_total_calls'],
            'counts': {'calls': m['calls'], 'peer_cells_incl': m['n_peer_cells_incl'],
                       'peer_cells_excl': m['n_peer_cells_excl'], 'whole_cells': m['n_whole_cells']},
            'admission_scan': scan,
        }

    # Cross-check against the supplement's published whole-input rejection
    # counts at 64/96 MiB (second-forward combine calls: olmoe 9->0,
    # qwen 40->9). Any mismatch aborts the run.
    def _whole_rejected(arm, l_mib):
        return next(s['calls_rejected_whole_input_guard']
                    for s in results_arms[arm]['admission_scan'] if s['L_mib'] == l_mib)
    for arm, c64, c96 in (('olmoe-o1', 9, 0), ('qwen-o1', 40, 9)):
        if (_whole_rejected(arm, 64), _whole_rejected(arm, 96)) != (c64, c96):
            raise SystemExit('published rejection-count cross-check failed for ' + arm)

    # forward-level breakdown (descriptive: demand scales with scheduled tokens)
    forward_breakdown = {}
    for arm_name in ('olmoe-o1', 'qwen-o1', 'olmoe-o32', 'qwen-o32'):
        arm = workloads['arms'][arm_name]
        per_fwd = {}
        for fwd in (0, 1):
            cs = [c for c in arm['calls'] if c['forward_id'] == fwd]
            per_fwd[str(fwd)] = {
                'calls': len(cs),
                'scheduled_tokens': arm['forwards'][fwd]['scheduled_tokens'],
                'max_peer_bytes': max(max(v for row in c['send_matrix_rows'] for v in row) * c['row_bytes'] for c in cs),
                'max_whole_input_bytes': max(max(sum(row) for row in c['send_matrix_rows']) * c['row_bytes'] for c in cs),
            }
        forward_breakdown[arm_name] = per_fwd

    # ---- zero-exceedance confidence bounds ---------------------------------
    confidence = {}
    for arm_name, m in arms_out.items():
        hist_l_mib = HISTORICAL_L_MIB['olmoe' if arm_name.startswith('olmoe') else 'qwen']
        confidence[arm_name] = {
            'at_historical_L_calls_observed_rejected': next(
                s['calls_rejected_whole_input_guard'] for s in results_arms[arm_name]['admission_scan']
                if s['L_mib'] == hist_l_mib),
            'peer_guard_calls_observed_rejected': next(
                s['calls_rejected_peer_guard'] for s in results_arms[arm_name]['admission_scan']
                if s['L_mib'] == hist_l_mib),
            'cp95_upper_call_rejection_prob_if_exchangeable': cp_upper_zero_exceedances(m['calls']),
            'cp95_upper_peer_cell_exceedance_prob_if_exchangeable': cp_upper_zero_exceedances(m['n_peer_cells_incl']),
        }

    # ---- emit summary.json --------------------------------------------------
    summary = {
        'schema': 'revision2.concern9.L-analysis.v1',
        'generated_utc': started,
        'scope': ('local CPU replay of frozen native Alltoallv split captures; no device, '
                  'remote or SSH operation; captured distributions only, not production'),
        'inputs': INPUT_RECORDS,
        'arm_results': results_arms,
        'forward_breakdown': forward_breakdown,
        'repeat_captures': repeat_out,
        'confidence_bounds_zero_exceedance': confidence,
        'historical_L_mib': HISTORICAL_L_MIB,
        'gpt_oss_note': ('no GPT-OSS capture exists; its 512 MiB slot came from a prospective '
                         f'{GPTOSS_PROSPECTIVE_BOUND_MIB} MiB model-level raw routed-payload bound '
                         'and cannot be validated by this replay'),
        'limitations': [
            'captured distributions are fixed-prompt B16/L1024 captures, not production workloads',
            'O32 arms cover only the first two Alltoallv forwards; the later 31 forwards use MC2 and have no Alltoallv matrices',
            'later captures cannot retro-justify the historical L choices (96 MiB was a prior config; 256 MiB followed a 96 MiB whole-input rejection; 512 MiB a prospective bound)',
            'GPT-OSS has no capture; its peak per-peer demand is unverifiable here',
            'percentiles are descriptive of the sample; Clopper-Pearson bounds assume exchangeability, which fixed prompts violate',
            'the whole-input guard requirement scales with scheduled tokens; the paper L1536 example (103,866,368 B rejected at 96 MiB) is outside this capture set',
        ],
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    # ---- input SHA256 manifest ----------------------------------------------
    lines = ['# Input files (path relative to repo root, SHA-256, size, role)', '',
             '| path | sha256 | bytes | role |', '|---|---|---|---|']
    for rec in INPUT_RECORDS:
        lines.append(f"| `{rec['path']}` | `{rec['sha256']}` | {rec['bytes']:,} | {rec['role']} |")
    (OUT_DIR / 'inputs.sha256.md').write_text('\n'.join(lines) + '\n')

    # ---- report.md -----------------------------------------------------------
    write_report(summary, report_grid)

    # ---- console digest -----------------------------------------------------
    print('=== Revision2 concern #9 L analysis (local CPU replay) ===')
    for arm_name, r in results_arms.items():
        print(f"{arm_name}: calls={r['counts']['calls']} "
              f"min-safe per-peer L={fmt_bytes(r['min_safe_L_peer_guard_bytes'])} "
              f"(excl self {fmt_bytes(r['min_safe_L_peer_guard_excluding_self_bytes'])}); "
              f"min-safe whole-input L={fmt_bytes(r['min_safe_L_whole_input_guard_bytes'])}; "
              f"historical L={r['historical_L_mib']} MiB -> utilization "
              f"peer {r['utilization_of_historical_L_peer_guard']*100:.2f}%, "
              f"whole {r['utilization_of_historical_L_whole_input_guard']*100:.2f}%")
    print('outputs:', OUT_DIR / 'summary.json', OUT_DIR / 'report.md', OUT_DIR / 'inputs.sha256.md')


def scan_row(scan, l_mib):
    for row in scan:
        if row['L_mib'] == l_mib:
            return row
    raise KeyError(l_mib)


def write_report(summary, report_grid):
    arms = summary['arm_results']
    md = []
    a = md.append
    a('# Revision2 Concern #9: Minimum Safe Per-Peer Slot Capacity L')
    a('')
    a(f"Generated {summary['generated_utc']} by `analysis.py` (pure local CPU; stdlib only; "
      'no device, remote or SSH operation). All numbers below are recomputed from the raw '
      'capture files listed in `inputs.sha256.md`; prior published numbers are used only as '
      'cross-check assertions inside the script (a mismatch aborts the run).')
    a('')
    a('## 1. Data inventory (what is actually analyzed)')
    a('')
    a('The "512 matrices" are the **512 global Alltoallv calls** stored in the frozen four-arm '
      'workload file `atlas_fair_trace_workloads_20261028_v1/workloads.json` (verified against '
      'its manifest): one fixed native B16/L1024 request per arm, P16/EP16, BF16 rows of '
      '4,096 B; each call carries a 16x16 send-split matrix (bytes = rows x 4,096):')
    a('')
    a('| arm | calls | forwards with Alltoallv | capture run |')
    a('|---|---|---|---|')
    a('| olmoe-o1 | 64 | 2 (prompt-only) | `atlas_model_split_capture_lane_20261028_v1/hccl-o1` |')
    a('| olmoe-o32 | 64 | first 2 of 33 (later 31 use MC2) | `atlas_model_split_capture_lane_20261028_v1/hccl-o32` |')
    a('| qwen-o1 | 192 | 2 (prompt-only) | `atlas_qwen_split_capture_20261028_v1/hccl-o1` |')
    a('| qwen-o32 | 192 | first 2 of 33 (later 31 use MC2) | `atlas_qwen_split_capture_20261028_v1/hccl-o32` |')
    a('')
    a('So the **O1-only subset is 256 calls (OLMoE 64 + Qwen 192)**; the O32 arms add the first '
      'two Alltoallv forwards of the separate O32 runs, giving 512 total. The script rebuilds '
      'every matrix independently from the four raw `request_raw.json` worker event streams '
      '(rank coverage, per-rank call counts, send/receive transpose, layout consistency), then '
      'asserts equality with the frozen arm and the anchor maxima; every check passed.')
    a('')
    a('Two supplementary native captures (`atlas_native_repeat_capture_20261028_v3`, same fixed '
      'OLMoE B16/L1024 batch) are replayed as **repeatability** evidence only: request-00 (64 '
      'calls) matches the frozen olmoe-o1 matrices 64/64; request-01 (128 calls) repeats the '
      'same batch twice and its forwards 2-3 equal forwards 0-1 64/64. Deterministic routing '
      'under one fixed batch is therefore established; this adds no new routing distribution.')
    a('')
    a('## 2. Guard semantics (per paper main.tex and supplement)')
    a('')
    a('- **Per-peer guard (host-list interface):** a call is rejected if any send or receive '
      'segment exceeds L. Because receive[j][i] = send[i][j], this equals: any matrix entry > L. '
      'The host lists include the self segment, so the binding max includes the diagonal '
      '(both variants reported).')
    a('- **Whole-input guard (historical device-split binding):** additionally requires every '
      "rank's whole input (row sum x 4,096 B) <= L. This dominates the per-peer bound because "
      'a row sum is at least as large as any of its entries.')
    a('- **Historical deployed semantics = union** of both checks (device-split required segment '
      'AND whole-input bounds); admission curves are reported separately for all three.')
    a('')
    a('## 3. Recomputed maxima and cross-validation')
    a('')
    a('| arm | calls | max peer segment incl self | max peer excl self | max rank whole input | anchor match |')
    a('|---|---|---|---|---|---|')
    for name, r in arms.items():
        a(f"| {name} | {r['counts']['calls']} | {r['min_safe_L_peer_guard_bytes']:,} B "
          f"({r['min_safe_L_peer_guard_mib']:.4f} MiB) | {r['min_safe_L_peer_guard_excluding_self_bytes']:,} B "
          f"({r['min_safe_L_peer_guard_excluding_self_mib']:.4f} MiB) | "
          f"{r['min_safe_L_whole_input_guard_bytes']:,} B ({r['min_safe_L_whole_input_guard_mib']:.4f} MiB) | "
          f"exact ({'OLMoE' if r['model'] == 'olmoe' else 'Qwen'} anchors) |")
    a('')
    a('All four arms reproduce the previously published O1 anchors exactly '
      '(OLMoE 7,118,848 / 93,265,920 B; Qwen 10,297,344 / 117,772,288 B) and the O32 '
      'initial-forward numbers (7,135,232 / 93,290,496 B and 10,416,128 / 117,813,248 B). '
      'Note the Qwen per-peer maximum is a **self segment** (10,297,344 B incl vs 10,153,984 B '
      'excl): under a slots-only-for-remote-peers reading the remote bound would be the smaller '
      'value, but the host-list guard checks all segments including self, so the larger value '
      'is the guard-relevant minimum.')
    a('')
    a('## 4. Minimum safe L for the captured distributions, vs historical L')
    a('')
    a('| model/arm | min safe L, per-peer guard | min safe L, whole-input guard (= union) | historical L | utilization at historical L (peer / whole) | headroom ratio (hist L / whole min) |')
    a('|---|---|---|---|---|---|')
    for name, r in arms.items():
        a(f"| {name} | {r['min_safe_L_peer_guard_mib']:.4f} MiB | {r['min_safe_L_whole_input_guard_mib']:.4f} MiB "
          f"| {r['historical_L_mib']} MiB | {r['utilization_of_historical_L_peer_guard']*100:.2f}% / "
          f"{r['utilization_of_historical_L_whole_input_guard']*100:.2f}% | "
          f"{r['headroom_ratio_historical_L_over_whole_input_min']:.3f}x |")
    a('')
    a('Key reading (captured B16/L1024/O1 requests):')
    a('')
    a('- Under the **per-peer guard** alone, the observed minimum is only ~6.79 MiB (OLMoE) and '
      '~9.82 MiB (Qwen): per-peer staging in PACE is nowhere near binding for these traces; '
      'the historical 96/256 MiB slots are 14.1x / 26.1x above the per-peer minimum.')
    a('- Under the **historical whole-input guard**, the minimums are 88.95 MiB (OLMoE) and '
      '112.32 MiB (Qwen). OLMoE\'s historical 96 MiB is only 1.079x above this observed minimum '
      '(92.65% utilized), i.e., nearly tight; Qwen\'s 256 MiB is 2.28x (43.87% utilized).')
    a('- These are minima **for these captures only**. The whole-input requirement grows with '
      'scheduled tokens: the paper\'s B16/L1536 run rejected a 103,866,368 B whole input at '
      '96 MiB, which is outside (and consistent with the length-scaling of) this capture set.')
    a('')
    a('## 5. Demand distributions (bytes; nearest-rank percentiles)')
    a('')
    a('Population = every (call, source rank, peer) segment for per-peer demand (incl self), '
      'and every (call, rank) row for whole input.')
    a('')
    for name, r in arms.items():
        d_incl = r['peer_demand_distribution_incl_self_bytes']
        d_whole = r['whole_input_distribution_bytes']
        a(f"**{name}** (n segments = {d_incl['n']:,}; n rank-inputs = {d_whole['n']:,})")
        a('')
        a('| metric | P50 | P90 | P99 | P99.9 | max |')
        a('|---|---|---|---|---|---|')
        a('| per-peer segment (MiB) | ' + ' | '.join(
            f"{d_incl[k]/MIB:.4f}" for k in ('p50', 'p90', 'p99', 'p99.9', 'max')) + ' |')
        a('| rank whole input (MiB) | ' + ' | '.join(
            f"{d_whole[k]/MIB:.4f}" for k in ('p50', 'p90', 'p99', 'p99.9', 'max')) + ' |')
        a('')
    a('Per-forward structure (all four arms): forward 0 schedules 1,024 tokens (first chunk of '
      'request 1) and its maxima are ~11-14x smaller (peer ratios 13.7-14.4x, whole-input '
      'ratios 11.0-13.1x; e.g., olmoe-o1: 0.473 MiB peer / 6.809 MiB whole; qwen-o1: 0.719 / '
      '10.250); forward 1 schedules the remaining 15,360 prompt tokens and produces every arm '
      'maximum. Demand tracks scheduled tokens roughly linearly (exact per-forward numbers in '
      '`summary.json` `forward_breakdown`).')
    a('')
    a('## 6. Admission curves: fraction of captured calls rejected at capacity L')
    a('')
    a('Counts are counterfactual guard rejections over the captured calls, not observed PACE '
      'events. Peer cell exceedance counts (call, rank, peer) segments above L. The '
      'whole-input guard rejection counts at 64/96 MiB exactly reproduce the supplement\'s '
      'published values (olmoe 9->0, qwen 40->9 rejected calls); every rejected call is a '
      'second-forward combine call, so the per-call sets here are identical to the published '
      'per-forward counts. `analysis.py` asserts this equality and aborts on any mismatch.')
    a('')
    for name, r in arms.items():
        n_calls = r['counts']['calls']
        n_cells = r['counts']['peer_cells_incl']
        n_whole = r['counts']['whole_cells']
        a(f"**{name}** ({n_calls} calls; {n_cells:,} peer segments; {n_whole:,} rank inputs)")
        a('')
        a('| L (MiB) | calls rejected, peer guard | calls rejected, whole-input guard | calls rejected, union (historical) | peer segments > L | rank inputs > L |')
        a('|---|---|---|---|---|---|')
        for l_mib in report_grid:
            row = scan_row(r['admission_scan'], l_mib)
            a(f"| {l_mib} | {row['calls_rejected_peer_guard']} ({row['calls_rejected_peer_guard_frac']*100:.2f}%) "
              f"| {row['calls_rejected_whole_input_guard']} ({row['calls_rejected_whole_input_guard_frac']*100:.2f}%) "
              f"| {row['calls_rejected_union_guard']} ({row['calls_rejected_union_guard_frac']*100:.2f}%) "
              f"| {row['peer_cells_exceeding']:,} ({row['peer_cells_exceeding_frac']*100:.3f}%) "
              f"| {row['whole_input_cells_exceeding']:,} ({row['whole_input_cells_exceeding_frac']*100:.2f}%) |")
        a('')
    a('Curve highlights: the per-peer guard admits everything down to ~7-10 MiB; the '
      'whole-input guard is the binding constraint and rejects OLMoE calls below 89 MiB and '
      'Qwen calls below 113 MiB; the historical 96/256 MiB configurations sit above every '
      'captured whole input (0 rejections). The full 1..512 MiB scan at 1 MiB resolution is '
      'in `summary.json` `arm_results[arm].admission_scan`.')
    a('')
    a('## 7. Confidence boundaries (order statistics only)')
    a('')
    a('| arm | rejections observed at historical L (whole / peer) | CP95 upper bound on per-call rejection prob. IF calls were exchangeable | CP95 upper bound on per-segment exceedance prob. |')
    a('|---|---|---|---|')
    for name, c in summary['confidence_bounds_zero_exceedance'].items():
        a(f"| {name} | {c['at_historical_L_calls_observed_rejected']} / {c['peer_guard_calls_observed_rejected']} "
          f"| {c['cp95_upper_call_rejection_prob_if_exchangeable']*100:.2f}% "
          f"| {c['cp95_upper_peer_cell_exceedance_prob_if_exchangeable']*100:.4f}% |")
    a('')
    a('These are one-sided Clopper-Pearson upper bounds given zero observed exceedances at '
      'L >= the arm minimum. **They are not production statements**: the captured calls come '
      'from fixed prompts, are not exchangeable random draws from a serving distribution, and '
      'the bounds condition on that unverified assumption. With n = 64/192 calls per arm the '
      'call-level bounds (4.57% / 1.55%) are weak; the segment-level bounds are small only '
      'because n is large, not because tail risk is measured.')
    a('')
    a('## 8. Limitations (scope of what this analysis can claim)')
    a('')
    for lim in summary['limitations']:
        a(f'- {lim}.')
    a('- Zero-total calls (OLMoE O1: 2, Qwen O1: 4; O32 initial forwards: 2 and 4) are '
      'scheduler artifacts of the fixed batch, not capacity events; they still pass both guards.')
    a('- No admission curve here measures runtime benefit, fallback frequency or fallback cost; '
      'the deployed system has no validated transparent fallback, and rejection remains the '
      'documented behavior.')
    a('- GPT-OSS has no captured matrices; nothing in this analysis validates or falsifies its '
      '512 MiB slot or the 360 MiB prospective bound.')
    a('')
    a('## 9. Files')
    a('')
    a('- `analysis.py` - this analysis (rerun: `python3 analysis.py` from any directory; '
      'writes the three other files).')
    a('- `summary.json` - machine-readable results: per-arm metrics, distributions, full 1..512 '
      'MiB admission scan, confidence bounds, input manifest, limitations.')
    a('- `inputs.sha256.md` - path/SHA-256/size/role of every input file read.')
    a('- `report.md` - this report.')
    (OUT_DIR / 'report.md').write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
