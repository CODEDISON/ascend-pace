#!/usr/bin/env python3
"""Revision2 concern #9 (v2): capacity demand of synthetic routing-distribution
families, local CPU analysis.

v1 (analysis_results/revision2_concern9_L_analysis_20261105_v1) answered, for
the *captured* OLMoE/Qwen distributions only, the minimum safe per-peer slot
capacity L (per-peer guard: 6.79-9.93 MiB; binding whole-input guard:
88.95/112.32 MiB).  The gap v2 fills: *different routing distributions* --
how much per-peer / whole-input capacity would a parameterized family of
routing distributions need at the captured token scale?

Method (all pure local Python, stdlib only; no device, remote or SSH):
  1. Reload and re-verify the same frozen hash chain as v1 (manifest ->
     workloads.json -> four raw request_raw.json), rebuilding every 16x16
     split matrix from the raw worker events and asserting equality with the
     frozen arms and the v1 published anchor maxima.
  2. Derive the fwd1 volume conversion from the verified data: fwd1 schedules
     15,360 tokens (B16/L1024 minus the 1,024-token first chunk), every
     Alltoallv call carries scheduled_tokens x 8 rows of 4,096 B (top-8
     expansion), i.e. 122,880 rows = 480 MiB per call, 7,680 rows mean per
     rank.  The per-rank row totals of each arm's worst fwd1 combine call
     (the call that sets the whole-input maximum) become the reference
     "rank-total profile".
  3. Generate deterministic synthetic routing families on that profile:
     uniform-dense; uniform-sparse (density 1/2, 1/4, 1/16); Zipf-skewed
     (s = 0.5, 1.0, 1.5); single-peer burst (hotspot); diagonal-dominant
     (self share q = 50%, 90%) -- at volume scales {0.5x, 1x, 2x} of the
     reference call (rank totals scaled, total rows preserved exactly by
     largest-remainder rounding).
  4. For every family x scale: per-peer max segment (= min safe per-peer L),
     whole-input max (= min safe whole-input / union L), receive-side column
     max, utilization vs the historical L (96/256/512 MiB), and admission
     behavior of the family-shaped call population on an L grid.
  5. Position the *captured* distributions inside the family spectrum using
     per-row concentration metrics (peakedness 16*max_share, HHI, support
     size, self share) and rank-total imbalance, cross-referencing v1.

Everything is deterministic (largest-remainder rounding with fixed
tie-breaking; no RNG anywhere).  Rerunning reproduces all numbers except the
generation timestamp.
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
V1_DIR = ROOT / 'analysis_results/revision2_concern9_L_analysis_20261105_v1'

MIB = 1024 * 1024
P = 16
ROW_BYTES = 4096
TOPK = 8                      # rows per scheduled token per Alltoallv call (asserted below)

# Configuration anchors (paper, not computed).  Historical per-peer slot
# capacities L per model; the deployed (union) guard is dominated by the
# whole-input check, see v1 section 2.
HISTORICAL_L_MIB = {'olmoe': 96, 'qwen': 256, 'gpt-oss': 512}
HISTORICAL_L_GRID_MIB = [96, 256, 512]

# v1 published anchors (recomputed independently there from the same raws).
# Used ONLY as cross-check assertions; a mismatch aborts.
ANCHORS = {
    'olmoe-o1': {'max_peer_bytes_including_self': 7118848, 'max_rank_input_bytes': 93265920},
    'qwen-o1': {'max_peer_bytes_including_self': 10297344, 'max_rank_input_bytes': 117772288},
    'olmoe-o32': {'max_peer_bytes_including_self': 7135232, 'max_rank_input_bytes': 93290496},
    'qwen-o32': {'max_peer_bytes_including_self': 10416128, 'max_rank_input_bytes': 117813248},
}
# v1/supplement published whole-input rejection counts at 64/96 MiB for the
# O1 arms (second-forward combine calls); the family-augmented replay at 1x
# must reproduce them exactly because family construction preserves row sums.
V1_PUBLISHED_REJECTIONS = {'olmoe-o1': {64: 9, 96: 0}, 'qwen-o1': {64: 40, 96: 9}}

FAIR_DIR = ROOT / 'analysis_results/atlas_fair_trace_workloads_20261028_v1'
RAW_SOURCES = {
    'olmoe-o1': ROOT / 'analysis_results/atlas_model_split_capture_lane_20261028_v1/hccl-o1/request_raw.json',
    'olmoe-o32': ROOT / 'analysis_results/atlas_model_split_capture_lane_20261028_v1/hccl-o32/request_raw.json',
    'qwen-o1': ROOT / 'analysis_results/atlas_qwen_split_capture_20261028_v1/hccl-o1/request_raw.json',
    'qwen-o32': ROOT / 'analysis_results/atlas_qwen_split_capture_20261028_v1/hccl-o32/request_raw.json',
}

SCALE_FACTORS = [0.5, 1.0, 2.0]
ADMISSION_GRID_MIB = [8, 16, 32, 48, 64, 88, 96, 112, 128, 192, 256, 512]

INPUT_RECORDS = []


# ---------------------------------------------------------------------------
# Data loading / verification (chain identical to v1)
# ---------------------------------------------------------------------------

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


def reconstruct_from_workers(workers, label):
    """Independent matrix reconstruction straight from raw worker events."""
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


# ---------------------------------------------------------------------------
# Pure numeric helpers (independently re-checkable)
# ---------------------------------------------------------------------------

def percentile_nearest_rank(sorted_values, p):
    """Nearest-rank percentile: the ceil(p/100*n)-th smallest value (1-based)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError('empty')
    rank = max(1, math.ceil(p / 100.0 * n))
    return sorted_values[rank - 1]


def largest_remainder(total, weights):
    """Integer allocation of `total` over `weights` (sum>0) by the
    largest-remainder method; ties broken by ascending index.  Deterministic;
    returns list[int] with sum == total whenever total is an int >= 0."""
    s = sum(weights)
    if s <= 0:
        raise ValueError('weights must be positive')
    raw = [total * w / s for w in weights]
    base = [math.floor(x) for x in raw]
    rem = int(total) - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - base[i]), i))
    for k in range(rem):
        base[order[k]] += 1
    return base


def zipf_weights(s, n=P):
    """Normalized Zipf(s) weights over ranks 1..n (descending)."""
    w = [1.0 / ((t + 1) ** s) for t in range(n)]
    tot = sum(w)
    return [x / tot for x in w]


def scale_profile(profile, m):
    """Scale every rank total by m, preserving the grand total exactly as
    round(sum(profile)*m) via largest-remainder rounding."""
    total = sum(profile)
    target = int(round(total * m))
    return largest_remainder(target, [p / total for p in profile])


# ---------------------------------------------------------------------------
# Synthetic routing-distribution families
# ---------------------------------------------------------------------------

def row_weights(family, i):
    """Peer weights (length 16, indexed by destination peer j) for source
    rank i under `family`.  Every construction is deterministic and
    rotation-based; no RNG.  Off-self rotation offset +1 avoids artificial
    self bias in the sparse/Zipf families (the separate diagonal family
    models self dominance explicitly)."""
    kind = family['kind']
    if kind == 'uniform_dense':
        return [1.0 / P] * P
    if kind == 'uniform_sparse':
        k = family['k']                                # active peers per rank
        w = [0.0] * P
        for t in range(k):
            w[(i + 1 + t * (P // k)) % P] = 1.0 / k    # strided, off-self
        return w
    if kind == 'zipf':
        z = zipf_weights(family['s'])
        w = [0.0] * P
        for t in range(P):
            w[(i + 1 + t) % P] = z[t]                  # circulant, hottest = i+1
        return w
    if kind == 'single_peer_burst':
        return [1.0 if j == 0 else 0.0 for j in range(P)]   # hotspot: all -> peer 0
    if kind == 'diagonal_dominant':
        q = family['q']
        w = [(1.0 - q) / (P - 1)] * P
        w[i] = q
        return w
    raise ValueError(kind)


FAMILIES = [
    {'id': 'uniform_dense', 'label': 'uniform dense (all 16 peers equal)',
     'kind': 'uniform_dense'},
    {'id': 'sparse_p1_2', 'label': 'uniform sparse p=1/2 (8 peers)',
     'kind': 'uniform_sparse', 'k': 8},
    {'id': 'zipf_s0.5', 'label': 'Zipf s=0.5 (16 peers, tilted)',
     'kind': 'zipf', 's': 0.5},
    {'id': 'sparse_p1_4', 'label': 'uniform sparse p=1/4 (4 peers)',
     'kind': 'uniform_sparse', 'k': 4},
    {'id': 'zipf_s1.0', 'label': 'Zipf s=1.0 (16 peers, tilted)',
     'kind': 'zipf', 's': 1.0},
    {'id': 'zipf_s1.5', 'label': 'Zipf s=1.5 (16 peers, strongly tilted)',
     'kind': 'zipf', 's': 1.5},
    {'id': 'diag_q0.5', 'label': 'diagonal-dominant q=50% self',
     'kind': 'diagonal_dominant', 'q': 0.5},
    {'id': 'diag_q0.9', 'label': 'diagonal-dominant q=90% self',
     'kind': 'diagonal_dominant', 'q': 0.9},
    {'id': 'sparse_p1_16', 'label': 'uniform sparse p=1/16 (1 peer, rotated permutation)',
     'kind': 'uniform_sparse', 'k': 1},
    {'id': 'single_burst', 'label': 'single-peer burst (every rank -> peer 0 hotspot)',
     'kind': 'single_peer_burst'},
]
# analytic per-row shape constants, computed (not hardcoded) from the weight
# vectors: peakedness = 16 x max share; HHI = sum of squared shares
for _f in FAMILIES:
    _w = row_weights(_f, 0)
    _f['peak'] = P * max(_w)
    _f['hhi'] = sum(x * x for x in _w)


def family_matrix(family, profile):
    """16x16 integer row-count matrix: row i distributes profile[i] rows over
    peers per the family weights (largest-remainder, row sums preserved)."""
    return [largest_remainder(profile[i], row_weights(family, i)) for i in range(P)]


def matrix_metrics(matrix, row_bytes=ROW_BYTES):
    row_sums = [sum(r) for r in matrix]
    col_sums = [sum(matrix[i][j] for i in range(P)) for j in range(P)]
    max_cell_rows = max(max(r) for r in matrix)
    return {
        'max_cell_bytes': max_cell_rows * row_bytes,
        'max_row_bytes': max(row_sums) * row_bytes,
        'max_col_bytes': max(col_sums) * row_bytes,
        'total_rows': sum(row_sums),
        'max_cell_rows': max_cell_rows,
        'max_row_rows': max(row_sums),
    }


# ---------------------------------------------------------------------------
# Captured-distribution positioning metrics
# ---------------------------------------------------------------------------

def row_shape_stats(rows):
    """Concentration stats for one nonzero send row: peakedness (16*max
    share), HHI, support (#nonzero peers), self share."""
    s = sum(rows)
    if s == 0:
        return None
    shares = [v / s for v in rows]
    return {
        'peakedness': P * max(shares),
        'hhi': sum(x * x for x in shares),
        'support': sum(1 for v in rows if v > 0),
        # peer-0 share, kept only as a structural liveness check (not the true
        # self share; summary self_share is computed inline as r[i]/s).
        'peer0_share': rows[0] / s if len(rows) == P else None,
    }


def captured_positioning(calls):
    """Per (forward, position) concentration metrics over all captured calls.
    peakedness/HHI are row-share metrics (row_bytes cancels); rank imbalance
    is per call (nonzero calls only)."""
    out = {}
    for fwd in (0, 1):
        for pos in ('dispatch', 'combine'):
            cs = [c for c in calls if c['forward_id'] == fwd and c['position'] == pos]
            if not cs:
                continue
            peaks, hhis, supports, selfshares, imbs = [], [], [], [], []
            max_cell = (0, None, None, None, False)
            for c in cs:
                rows = c['send_matrix_rows']
                rs = [sum(r) for r in rows]
                tot = sum(rs)
                if tot > 0:
                    imbs.append(max(rs) / (tot / P))
                for i, r in enumerate(rows):
                    st = row_shape_stats(r)
                    if st is None:
                        continue
                    # rows are absolute-peer indexed; self = index i
                    s = sum(r)
                    peaks.append(P * max(r) / s)
                    hhis.append(sum((v / s) ** 2 for v in r))
                    supports.append(sum(1 for v in r if v > 0))
                    selfshares.append(r[i] / s)
                    for j, v in enumerate(r):
                        if v > max_cell[0]:
                            max_cell = (v, i, j, s, i == j)
            pk, hh, ss = sorted(peaks), sorted(hhis), sorted(selfshares)
            out[f'fwd{fwd}_{pos}'] = {
                'calls': len(cs),
                'nonzero_rows': len(peaks),
                'peakedness': {'p50': percentile_nearest_rank(pk, 50),
                               'p90': percentile_nearest_rank(pk, 90),
                               'p99': percentile_nearest_rank(pk, 99),
                               'max': pk[-1]},
                'hhi': {'p50': percentile_nearest_rank(hh, 50),
                        'p90': percentile_nearest_rank(hh, 90), 'max': hh[-1]},
                'support': {'min': min(supports), 'median': sorted(supports)[len(supports) // 2],
                            'max': max(supports), 'of': P},
                'self_share': {'p90': percentile_nearest_rank(ss, 90), 'max': ss[-1]},
                'rank_imbalance_max_over_mean': {'min': min(imbs), 'max': max(imbs)},
                'global_max_cell': {'rows': max_cell[0], 'row': max_cell[1], 'peer': max_cell[2],
                                    'row_sum': max_cell[3], 'is_self': max_cell[4],
                                    'row_peakedness': P * max_cell[0] / max_cell[3]},
            }
    return out


# ---------------------------------------------------------------------------
# Family-shaped replay of the captured call population
# ---------------------------------------------------------------------------

def family_replay(family, calls, m):
    """Apply the family shape to every captured call's rank-total profile at
    scale m.  Row sums (hence whole-input demand) are preserved; only the
    across-peer split changes.  Returns per-call max-cell bytes and the
    unchanged per-call max row-sum bytes."""
    per_call_max_cell = []
    per_call_max_whole = []
    for c in calls:
        prof = [sum(r) for r in c['send_matrix_rows']]
        scaled = scale_profile(prof, m) if m != 1.0 else prof
        mm = family_matrix(family, scaled)
        per_call_max_cell.append(max(max(r) for r in mm) * ROW_BYTES)
        per_call_max_whole.append(max(sum(r) for r in mm) * ROW_BYTES)
    return per_call_max_cell, per_call_max_whole


def admission_from_maxima(max_cell_list, max_whole_list, grid_mib):
    """Rejection counts on an L grid: peer guard (max cell), whole-input guard
    (max row sum), union (historical deployed semantics)."""
    cells = sorted(max_cell_list)
    wholes = sorted(max_whole_list)
    n = len(cells)
    scan = []
    for l_mib in grid_mib:
        lim = l_mib * MIB
        peer_rej = n - bisect.bisect_right(cells, lim)
        whole_rej = n - bisect.bisect_right(wholes, lim)
        scan.append({
            'L_mib': l_mib,
            'calls_rejected_peer_guard': peer_rej,
            'calls_rejected_whole_input_guard': whole_rej,
            'calls_rejected_union_guard': sum(
                1 for a, b in zip(max_cell_list, max_whole_list) if a > lim or b > lim),
        })
    return scan


def fmt_mib(b):
    return f'{b / MIB:.4f}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')

    # ---- verify the same hash chain as v1 -------------------------------
    manifest = load_json(FAIR_DIR / 'manifest.json', 'fair-trace manifest (hash chain root)')
    workloads = load_json(FAIR_DIR / 'workloads.json', 'frozen 4-arm native split matrices')
    if INPUT_RECORDS[-1]['sha256'] != manifest['workloads_sha256']:
        raise ValueError('workloads.json file digest does not match its manifest')

    arms = {}
    for arm_name in ('olmoe-o1', 'olmoe-o32', 'qwen-o1', 'qwen-o32'):
        arm = workloads['arms'][arm_name]
        raw = load_json(RAW_SOURCES[arm_name], f'raw native capture ({arm_name})')
        if INPUT_RECORDS[-1]['sha256'] != arm['request_raw_sha256']:
            raise ValueError(f'{arm_name}: raw capture digest differs from arm receipt')
        rebuilt = reconstruct_from_workers(raw['workers'], arm_name)
        if len(rebuilt) != len(arm['calls']):
            raise ValueError(f'{arm_name}: call count differs from frozen arm')
        for idx, (new, old) in enumerate(zip(rebuilt, arm['calls'])):
            if new['send_matrix_rows'] != old['send_matrix_rows']:
                raise ValueError(f'{arm_name} call {idx}: matrix differs from frozen arm')
            if new['row_bytes'] != old['row_bytes']:
                raise ValueError(f'{arm_name} call {idx}: row_bytes differs')
        calls = arm['calls']
        max_cell = max(max(v for r in c['send_matrix_rows'] for v in r) * c['row_bytes']
                       for c in calls)
        max_whole = max(max(sum(r) for r in c['send_matrix_rows']) * c['row_bytes']
                        for c in calls)
        if max_cell != ANCHORS[arm_name]['max_peer_bytes_including_self']:
            raise ValueError(f'{arm_name}: max peer != v1 anchor')
        if max_whole != ANCHORS[arm_name]['max_rank_input_bytes']:
            raise ValueError(f'{arm_name}: max whole-input != v1 anchor')
        arms[arm_name] = {'calls': calls, 'forwards': arm['forwards']}

    # ---- fwd token->row conversion (asserted, not assumed) ---------------
    conversion = {'scheduled_tokens_fwd0': 1024, 'scheduled_tokens_fwd1': {},
                  'rows_per_token': TOPK, 'row_bytes': ROW_BYTES}
    for arm_name, a in arms.items():
        for f in a['forwards'][:2]:
            total_rows = {sum(sum(r) for r in c['send_matrix_rows'])
                          for c in a['calls'] if c['forward_id'] == f['forward_id']}
            if len(total_rows) != 1:
                raise ValueError(f'{arm_name} fwd{f["forward_id"]}: call totals differ')
            if total_rows.pop() != f['scheduled_tokens'] * TOPK:
                raise ValueError(f'{arm_name} fwd{f["forward_id"]}: total rows != '
                                 'scheduled_tokens*8; conversion premise broken')
        conversion['scheduled_tokens_fwd1'][arm_name] = a['forwards'][1]['scheduled_tokens']

    # ---- reference rank-total profiles (worst fwd1 combine call) ---------
    reference = {}
    for arm_name in ('olmoe-o1', 'qwen-o1'):          # O1 arms = published v1 anchors
        a = arms[arm_name]
        fwd1c = [c for c in a['calls'] if c['forward_id'] == 1 and c['position'] == 'combine']
        worst = max(fwd1c, key=lambda c: max(sum(r) for r in c['send_matrix_rows']))
        prof = [sum(r) for r in worst['send_matrix_rows']]
        reference[arm_name] = {
            'profile_rows': prof,
            'total_rows': sum(prof),
            'total_bytes': sum(prof) * ROW_BYTES,
            'max_rank_rows': max(prof),
            'max_rank_share_of_total': max(prof) / sum(prof),
            'whole_input_min_bytes_at_1x': max(prof) * ROW_BYTES,
        }
        # tie to v1: uniform-dense 1x must reproduce the arm's whole-input max
        if reference[arm_name]['whole_input_min_bytes_at_1x'] != \
                ANCHORS[arm_name]['max_rank_input_bytes']:
            raise ValueError(f'{arm_name}: worst fwd1 call does not set the v1 whole-input max')
    o32_note = {}
    for arm_name in ('olmoe-o32', 'qwen-o32'):
        a = arms[arm_name]
        fwd1c = [c for c in a['calls'] if c['forward_id'] == 1 and c['position'] == 'combine']
        o32_note[arm_name] = max(max(sum(r) for r in c['send_matrix_rows']) * ROW_BYTES
                                 for c in fwd1c) / MIB

    # ---- family x scale grid ---------------------------------------------
    family_results = {}
    for base in ('olmoe-o1', 'qwen-o1'):
        ref = reference[base]
        prof1 = ref['profile_rows']
        for m in SCALE_FACTORS:
            prof = scale_profile(prof1, m)
            whole_min = max(prof) * ROW_BYTES
            whole_cells = {}
            for fam in FAMILIES:
                mat = family_matrix(fam, prof)
                mm = matrix_metrics(mat)
                # structural invariants (abort on violation)
                if [sum(r) for r in mat] != prof:
                    raise ValueError(f'{fam["id"]}: row sums not preserved')
                if any(v != int(v) or v < 0 for r in mat for v in r):
                    raise ValueError(f'{fam["id"]}: noninteger/negative cell')
                if mm['max_row_bytes'] != whole_min:
                    raise ValueError(f'{fam["id"]}: whole-input min differs across families')
                floor_rows = largest_remainder(max(prof), [1.0 / P] * P)[0]
                entry = {
                    'label': fam['label'],
                    'max_cell_bytes': mm['max_cell_bytes'],
                    'max_cell_mib': mm['max_cell_bytes'] / MIB,
                    'max_row_bytes': mm['max_row_bytes'],
                    'max_col_bytes': mm['max_col_bytes'],
                    'max_col_mib': mm['max_col_bytes'] / MIB,
                    'per_peer_vs_uniform_dense_floor': mm['max_cell_rows'] / floor_rows,
                    'util_vs_hist_L': {str(l): mm['max_cell_bytes'] / (l * MIB)
                                       for l in HISTORICAL_L_GRID_MIB},
                    'admits_peer_guard': {str(l): mm['max_cell_bytes'] <= l * MIB
                                          for l in HISTORICAL_L_GRID_MIB},
                }
                # exact-construction check: the global max cell must come from
                # the max-total row (largest-remainder allocation is monotone
                # in the row total for fixed weights).
                imax = prof.index(max(prof))
                w = row_weights(fam, imax)
                expect = max(largest_remainder(prof[imax], w))
                if mm['max_cell_rows'] != expect:
                    raise ValueError(f'{fam["id"]}: max cell not from max-total row construction')
                if fam['kind'] == 'single_peer_burst' and mm['max_col_bytes'] != sum(prof) * ROW_BYTES:
                    raise ValueError('single-peer burst: hotspot column must carry the full total')
                whole_cells[fam['id']] = entry
            family_results.setdefault(base, {})[str(m)] = {
                'scaled_profile_rows': prof,
                'total_rows': sum(prof),
                'total_mib': sum(prof) * ROW_BYTES / MIB,
                'whole_input_min_bytes': whole_min,
                'whole_input_min_mib': whole_min / MIB,
                'whole_decomposition': {
                    'call_total_mib': sum(prof) * ROW_BYTES / MIB,
                    'max_rank_share': max(prof) / sum(prof),
                    'note': 'whole_min = call_total x max_rank_share (rank imbalance), '
                            'identical for every family (row sums preserved)'},
                'admits_whole_guard': {str(l): whole_min <= l * MIB
                                       for l in HISTORICAL_L_GRID_MIB},
                'util_whole_vs_hist_L': {str(l): whole_min / (l * MIB)
                                         for l in HISTORICAL_L_GRID_MIB},
                'families': whole_cells,
            }

    # ---- family-shaped admission replay over the captured call population
    replay = {}
    for base in ('olmoe-o1', 'qwen-o1'):
        calls = arms[base]['calls']
        for m in SCALE_FACTORS:
            per_family_cells = {}
            whole_list = None
            for fam in FAMILIES:
                cells, wholes = family_replay(fam, calls, m)
                if whole_list is None:
                    whole_list = wholes
                elif wholes != whole_list:
                    raise ValueError('whole-input list differs across families')
                per_family_cells[fam['id']] = cells
                replay.setdefault(base, {}).setdefault(str(m), {})[fam['id']] = {
                    'per_call_max_cell_bytes': {
                        'min': min(cells), 'p50': percentile_nearest_rank(sorted(cells), 50),
                        'p90': percentile_nearest_rank(sorted(cells), 90), 'max': max(cells)},
                    'set_min_safe_peer_L_bytes': max(cells),
                }
                # set-level per-peer min must equal the worst-call matrix value
                if max(cells) != family_results[base][str(m)]['families'][fam['id']]['max_cell_bytes']:
                    raise ValueError(f'{fam["id"]}: set-level per-peer min != worst-call value')
            scans = {fid: admission_from_maxima(cl, whole_list, ADMISSION_GRID_MIB)
                     for fid, cl in per_family_cells.items()}
            # structural identity: per call max cell <= max row sum, so the
            # union (historical) rejections must equal the whole-input ones.
            for fid, sc in scans.items():
                for row in sc:
                    if row['calls_rejected_union_guard'] != row['calls_rejected_whole_input_guard']:
                        raise ValueError(f'{fid}: union rejections differ from whole-input rejections')
            # whole/union guard rejection counts are family-invariant; assert
            # the v1/supplement published counts at 1x (row sums preserved).
            if m == 1.0:
                for l_mib, expect in V1_PUBLISHED_REJECTIONS[base].items():
                    row = next(r for r in scans['uniform_dense'] if r['L_mib'] == l_mib)
                    if row['calls_rejected_whole_input_guard'] != expect:
                        raise SystemExit(f'{base}: family replay rejects v1 published counts')
            replay[base][str(m)]['_admission_scan'] = {
                'grid_mib': ADMISSION_GRID_MIB,
                'whole_input_rejections_family_invariant': {
                    r['L_mib']: r['calls_rejected_whole_input_guard'] for r in scans['uniform_dense']},
                'peer_guard_rejections_by_family': {
                    fid: {r['L_mib']: r['calls_rejected_peer_guard'] for r in sc}
                    for fid, sc in scans.items()},
            }

    # ---- captured positioning --------------------------------------------
    positioning = {arm_name: captured_positioning(a['calls']) for arm_name, a in arms.items()}

    # captured vs dense-uniform floor (fwd1, per-peer and whole-input)
    captured_vs_floor = {}
    for base in ('olmoe-o1', 'qwen-o1'):
        calls = arms[base]['calls']
        fwd1 = [c for c in calls if c['forward_id'] == 1]
        max_cell_rows = max(max(v for r in c['send_matrix_rows'] for v in r) for c in fwd1)
        max_row_rows = max(max(sum(r) for r in c['send_matrix_rows']) for c in fwd1)
        captured_vs_floor[base] = {
            'fwd1_max_cell_bytes': max_cell_rows * ROW_BYTES,
            'fwd1_max_cell_mib': max_cell_rows * ROW_BYTES / MIB,
            'uniform_dense_floor_bytes': math.ceil(max_row_rows / P) * ROW_BYTES,
            'per_peer_ratio_over_floor': (max_cell_rows * ROW_BYTES) /
                                         (math.ceil(max_row_rows / P) * ROW_BYTES),
            'fwd1_max_whole_bytes': max_row_rows * ROW_BYTES,
            'rank_balanced_floor_bytes': (sum(sum(r) for r in fwd1[0]['send_matrix_rows']) //
                                          P) * ROW_BYTES,
        }

    # ---- break-even scales vs historical L (union guard) -----------------
    breakeven = {}
    for base in ('olmoe-o1', 'qwen-o1'):
        whole_1x = reference[base]['whole_input_min_bytes_at_1x']
        tok = reference[base]['total_rows'] // TOPK
        breakeven[base] = {
            'fwd1_scheduled_tokens': tok,
            'scale_breaking_L': {str(l): l * MIB / whole_1x for l in HISTORICAL_L_GRID_MIB},
            'tokens_breaking_L': {str(l): (l * MIB / whole_1x) * tok
                                  for l in HISTORICAL_L_GRID_MIB},
        }

    # ---- emit summary.json -------------------------------------------------
    summary = {
        'schema': 'revision2.concern9.L-analysis.v2',
        'generated_utc': started,
        'scope': ('local CPU synthesis on top of the frozen native Alltoallv split '
                  'captures; no device, remote or SSH operation; synthetic families '
                  'are parameterized constructs, not production traces'),
        'relation_to_v1': ('v1 = captured distributions only; v2 adds the '
                           '"distribution family -> capacity demand" mapping and '
                           'positions the captured distributions inside that family '
                           'spectrum. v1 directory is untouched.'),
        'inputs': INPUT_RECORDS,
        'conversion': conversion | {
            'note': ('every Alltoallv call carries scheduled_tokens x 8 rows '
                     '(top-8 expert expansion) of 4,096 B; fwd1 = 15,360 tokens = '
                     '122,880 rows = 480 MiB per call, 7,680 rows mean per rank; '
                     'asserted against every captured call'),
        },
        'reference_profiles': reference,
        'o32_robustness_whole_min_mib': o32_note,
        'families': [{'id': f['id'], 'label': f['label'],
                      'analytic_peakedness_16x_max_share': f['peak'],
                      'analytic_hhi': f['hhi'],
                      'kind': f['kind']} for f in FAMILIES],
        'family_results': family_results,
        'family_replay_percentiles': replay,
        'captured_positioning': positioning,
        'captured_vs_uniform_floor': captured_vs_floor,
        'breakeven_scales_vs_historical_L': breakeven,
        'historical_L_mib': HISTORICAL_L_MIB,
        'limitations': [
            'synthetic families are parameterized constructs anchored to the captured token scale; they are not production routing distributions and no claim is made that any of them occurs in serving',
            'rank-total profiles are held at the captured worst fwd1 combine call per O1 arm (row sums preserved by construction), so whole-input results are family-invariant by design; only per-peer and receive-side results differ across families',
            'this analysis cannot retro-justify the historical L choices (96 MiB was a prior config; 256 MiB followed a 96 MiB whole-input rejection; 512 MiB a prospective bound)',
            'single-node fixed-prompt B16/L1024 token scale only: fwd1 = 15,360 scheduled tokens; the paper L1536 example (103,866,368 B whole input) is outside this set',
            'GPT-OSS still has no capture; nothing here validates its 512 MiB slot or the 360 MiB prospective bound',
            'largest-remainder rounding moves individual cells by <1 row (4 KiB); all totals are preserved exactly',
            'the hotspot (single-peer burst) receive-side column max is a staging upper bound under a receive-whole-input reading, which the historical guard (send-side whole-input) did not check',
        ],
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    # ---- input SHA256 manifest ---------------------------------------------
    lines = ['# Input files (path relative to repo root, SHA-256, size, role)', '',
             '| path | sha256 | bytes | role |', '|---|---|---|---|']
    for rec in INPUT_RECORDS:
        lines.append(f"| `{rec['path']}` | `{rec['sha256']}` | {rec['bytes']:,} | {rec['role']} |")
    (OUT_DIR / 'inputs.sha256.md').write_text('\n'.join(lines) + '\n')

    # ---- report.md -----------------------------------------------------------
    write_report(summary)

    # ---- console digest ----------------------------------------------------
    print('=== Revision2 concern #9 L analysis v2 (synthetic routing families) ===')
    for base in ('olmoe-o1', 'qwen-o1'):
        r1 = family_results[base]['1.0']
        print(f"{base}: fwd1 total {r1['total_mib']:.2f} MiB/call; whole min-L "
              f"{r1['whole_input_min_mib']:.4f} MiB (family-invariant); per-peer min-L "
              f"uniform {r1['families']['uniform_dense']['max_cell_mib']:.4f} -> "
              f"single-peer {r1['families']['single_burst']['max_cell_mib']:.4f} MiB")
        cv = captured_vs_floor[base]
        print(f"  captured fwd1 per-peer max {cv['fwd1_max_cell_mib']:.4f} MiB = "
              f"{cv['per_peer_ratio_over_floor']:.3f}x uniform-dense floor")
    print('outputs:', OUT_DIR / 'summary.json', OUT_DIR / 'report.md', OUT_DIR / 'inputs.sha256.md')


def scan_row(scan, l_mib):
    for row in scan:
        if row['L_mib'] == l_mib:
            return row
    raise KeyError(l_mib)


def write_report(summary):
    md = []
    a = md.append
    a('# Revision2 Concern #9 (v2): Capacity Demand of Routing-Distribution Families')
    a('')
    a(f"Generated {summary['generated_utc']} by `analysis.py` (pure local CPU; stdlib only; "
      'no device, remote or SSH operation). This extends v1 '
      '(`revision2_concern9_L_analysis_20261105_v1`, untouched) which answered the same '
      'question **for the captured distributions only**; v2 fills the gap "how much capacity '
      'would *different* routing distributions need" with a deterministic synthetic family '
      'grid anchored to the captured token scale, and positions the captured distributions '
      'inside that spectrum. All input data is re-verified through the same hash chain and '
      'raw-event reconstruction as v1, and the v1 anchor maxima are asserted (mismatch aborts).')
    a('')
    a('## 1. Question and headline answer')
    a('')
    a('**Q:** how does the required per-peer slot capacity L (and the binding whole-input '
      'requirement) change across routing distributions at the captured scale?')
    a('')
    a('**A:** the requirement decomposes into two nearly independent factors:')
    a('')
    a('- **Whole-input (binding, historical union guard): depends only on per-rank totals** '
      '– i.e., token scale x rank imbalance – and is *exactly family-invariant* here by '
      'construction (families redistribute each rank\'s rows across peers; row sums are '
      'preserved). At 1x fwd1 scale it is 88.95 MiB (OLMoE base) / 112.32 MiB (Qwen base), '
      'reproducing v1\'s captured minima exactly.')
    a('- **Per-peer guard: depends on routing shape**, spanning a 16x range from the '
      'uniform-dense floor (max rank rows / 16) to the single-peer worst case (the full max '
      'rank rows). Captured routing sits at the benign end: its fwd1 per-peer maximum is '
      'only 1.22x (OLMoE) / 1.40x (Qwen) above the uniform-dense floor.')
    a('')
    a('Because any matrix entry is bounded by its row sum, the per-peer requirement can '
      '**never exceed** the whole-input requirement: routing shape can never justify an L '
      'larger than the whole-input need; it only determines how much *smaller* than that L '
      'could be if the whole-input guard were dropped.')
    a('')
    a('## 2. Data, verification and the token->row conversion')
    a('')
    a('Same inventory as v1: four frozen native arms (olmoe/qwen x o1/o32, P16, BF16, '
      'row 4,096 B) re-verified end-to-end (manifest -> workloads.json digest -> raw '
      '`request_raw.json` digests -> independent matrix reconstruction -> equality with the '
      'frozen arms -> v1 anchor maxima). Every check passed; files in `inputs.sha256.md`.')
    a('')
    a('**Conversion (asserted against every captured call):** each Alltoallv call carries '
      '`scheduled_tokens x 8` rows (top-8 expert expansion) of 4,096 B. Forward 1 schedules '
      '15,360 tokens (B16/L1024 minus the 1,024-token first chunk), so every fwd1 call moves '
      '122,880 rows = **480 MiB**, mean 7,680 rows (30 MiB) per rank. Dispatch calls '
      'distribute these perfectly evenly (every rank exactly 7,680 rows in the O1 arms); '
      'combine calls are rank-imbalanced, and the whole-input maxima always come from '
      'combine calls.')
    a('')
    a('**Reference rank-total profiles** (the worst fwd1 combine call per O1 arm; this is '
      'the call that sets v1\'s whole-input maximum, asserted):')
    a('')
    a('| base arm | worst call | rank totals (rows) | total | max rank | max-rank share | whole-input min @1x |')
    a('|---|---|---|---|---|---|---|')
    for base, ref in summary['reference_profiles'].items():
        a(f"| {base} | fwd1 combine | {ref['max_rank_rows']} max of 16 rows "
          f"(full profile in `summary.json`) | {ref['total_rows']:,} rows "
          f"({ref['total_bytes']/MIB:.0f} MiB) | {ref['max_rank_rows']:,} "
          f"({ref['whole_input_min_bytes_at_1x']/MIB:.4f} MiB) | "
          f"{ref['max_rank_share_of_total']*100:.2f}% | "
          f"{ref['whole_input_min_bytes_at_1x']/MIB:.4f} MiB |")
    a('')
    a(f"O32 robustness: the O32 worst fwd1 whole-input maxima are "
      f"{summary['o32_robustness_whole_min_mib']['olmoe-o32']:.4f} / "
      f"{summary['o32_robustness_whole_min_mib']['qwen-o32']:.4f} MiB (<0.04% above the O1 "
      'bases), so the O1 profiles are used as the canonical bases.')
    a('')
    a('**Scales:** each family is generated at {0.5x, 1x, 2x} of the reference call: every '
      'rank total is scaled by m with the grand total preserved exactly '
      '(`round(T x m)`) by largest-remainder rounding; cells are integer row counts, row '
      'sums exact.')
    a('')
    a('## 3. Family definitions (all deterministic; no RNG)')
    a('')
    a('| family | construction (per source rank i) | analytic peakedness (16 x max share) | HHI |')
    a('|---|---|---|---|')
    for f in summary['families']:
        a(f"| `{f['id']}` | {f['label']} | {f['analytic_peakedness_16x_max_share']:.3f} | "
          f"{f['analytic_hhi']:.4f} |")
    a('')
    a('Uniform = uniform over the rank\'s *own* total (rows keep the captured imbalance); '
      'sparse supports are strided and off-self (rank i\'s peers are (i+1), (i+1+16/k), ...); '
      'Zipf weights w_t = (t+1)^-s / normalizer are assigned circulant with the hottest peer '
      'at i+1 (no artificial self bias); diagonal-dominant puts share q on the self cell and '
      'spreads 1-q uniformly over the other 15; single-peer burst sends every rank\'s whole '
      'input to peer 0 (hotspot). `sparse_p1_16` (rotated single peer, a permutation) and '
      '`single_burst` (all to the same peer) have identical send-side demand and differ only '
      'in the receive-side column maximum.')
    a('')
    a('## 4. Distribution family -> capacity demand')
    a('')
    for base in ('qwen-o1', 'olmoe-o1'):
        a(f'### Base: {base} reference profile')
        a('')
        for m in ('0.5', '1.0', '2.0'):
            r = summary['family_results'][base][m]
            a(f'**Scale {m}x** - call total {r["total_mib"]:.2f} MiB; whole-input min L '
              f'(= historical union guard, family-invariant) **{r["whole_input_min_mib"]:.4f} MiB** '
              f'= {r["total_mib"]:.2f} MiB x {r["whole_decomposition"]["max_rank_share"]*100:.2f}% '
              f'max-rank share; admits at historical L: '
              + ', '.join(f'{l}: {"yes" if r["admits_whole_guard"][l] else "NO"}'
                          for l in ('96', '256', '512')) + '.')
            a('')
            a('| family | per-peer min L (MiB) | ratio vs uniform floor | recv-side col max (MiB) | peer guard admits @96/256/512 |')
            a('|---|---|---|---|---|')
            order = ['uniform_dense', 'sparse_p1_2', 'zipf_s0.5', 'sparse_p1_4', 'zipf_s1.0',
                     'zipf_s1.5', 'diag_q0.5', 'diag_q0.9', 'sparse_p1_16', 'single_burst']
            for fid in order:
                e = r['families'][fid]
                adm = '/'.join('Y' if e['admits_peer_guard'][l] else 'N'
                               for l in ('96', '256', '512'))
                a(f"| `{fid}` | {e['max_cell_mib']:.4f} | "
                  f"{e['per_peer_vs_uniform_dense_floor']:.2f}x | {e['max_col_mib']:.4f} | {adm} |")
            a('')
    a('Reading the grid (Qwen base, 1x): per-peer min L spans **7.02 MiB** (uniform dense) '
      'to **112.32 MiB** (single-peer = the whole-input value), with Zipf s=1.0 at 33.2 MiB '
      'and 90% self-dominance at 101.1 MiB; OLMoE base spans 5.56-88.95 MiB. The per-peer '
      'requirement is capped by the whole-input requirement in every family. The receive-side '
      'column maximum is within 0.1% of call total/16 (30.03 MiB; largest-remainder '
      'tie bias favors low column indices) for uniform dense, moderately above '
      'it for the rotated sparse/Zipf/diagonal families (36.8-103.6 MiB at 1x Qwen), and '
      'equals the **full call total (480 MiB at 1x)** for the single-peer hotspot - relevant '
      'only under a receive-side whole-staging reading, which the historical (send-side) '
      'whole-input guard did not check.')
    a('')
    a('## 5. Admission behavior on the L grid (family-shaped call populations)')
    a('')
    a('Each family shape was also applied to **every captured call** of the base arm '
      '(64 OLMoE / 192 Qwen calls, fwd0+fwd1) at each scale. Because row sums are preserved, '
      'the whole-input/union rejection counts equal the captured ones at 1x - the replay '
      'asserts the v1/supplement published counts (olmoe 64 MiB: 9, 96 MiB: 0; qwen: 40, 9) '
      'and aborts on mismatch. Only the peer-guard counts move with family shape; per-call '
      'max-cell percentiles (1x scale):')
    a('')
    a('| base | family | per-call max cell P50 / P90 / max (MiB) |')
    a('|---|---|---|')
    for base in ('qwen-o1', 'olmoe-o1'):
        rp = summary['family_replay_percentiles'][base]['1.0']
        for fid in ('uniform_dense', 'zipf_s0.5', 'zipf_s1.0', 'diag_q0.9', 'single_burst'):
            d = rp[fid]['per_call_max_cell_bytes']
            a(f"| {base} | `{fid}` | {d['p50']/MIB:.4f} / {d['p90']/MIB:.4f} / "
              f"{d['max']/MIB:.4f} |")
    a('')
    a('Set-level per-peer minima equal the worst-call values of section 4 (asserted in the '
      'script). Full grids are in `summary.json` `family_replay_percentiles`.')
    a('')
    a('**Break-even vs historical L (union guard, any family):** since the union requirement '
      'is family-invariant, the historical configurations break at a pure scale multiple:')
    a('')
    a('| base | fwd1 tokens | breaks 96 MiB at | breaks 256 MiB at | breaks 512 MiB at |')
    a('|---|---|---|---|---|')
    for base, be in summary['breakeven_scales_vs_historical_L'].items():
        a(f"| {base} | {be['fwd1_scheduled_tokens']:,} | "
          f"{be['scale_breaking_L']['96']:.2f}x ({be['tokens_breaking_L']['96']:,.0f} tok) | "
          f"{be['scale_breaking_L']['256']:.2f}x ({be['tokens_breaking_L']['256']:,.0f} tok) | "
          f"{be['scale_breaking_L']['512']:.2f}x ({be['tokens_breaking_L']['512']:,.0f} tok) |")
    a('')
    a('At 2x fwd1 scale (~30.7k scheduled tokens) every family still admits under 256/512 MiB; '
      '512 MiB first breaks at ~4.6x (Qwen base, ~70k tokens) / ~5.8x (OLMoE base) regardless '
      'of routing shape. This bounds what the synthetic families can say about GPT-OSS\'s '
      '512 MiB slot: at captured-like token scale no family - not even the single-peer '
      'hotspot - needs more than 512 MiB; only larger token scales do, and then for every '
      'family alike.')
    a('')
    a('## 6. Where the captured distributions sit in the family spectrum')
    a('')
    a('Row-share metrics over all captured fwd1 rows (peakedness = 16 x max share; HHI with '
      'uniform = 0.0625):')
    a('')
    a('| arm x fwd1 | peakedness P50 / P90 / P99 / max | HHI P50 / max | support (of 16) | rank imbalance (max/mean) | global max cell |')
    a('|---|---|---|---|---|---|')
    for arm_name in ('olmoe-o1', 'qwen-o1'):
        for key in ('fwd1_dispatch', 'fwd1_combine'):
            d = summary['captured_positioning'][arm_name][key]
            pk, hh = d['peakedness'], d['hhi']
            g = d['global_max_cell']
            a(f"| {arm_name} {key.replace('fwd1_', '')} | {pk['p50']:.2f} / {pk['p90']:.2f} / "
              f"{pk['p99']:.2f} / {pk['max']:.2f} | {hh['p50']:.4f} / {hh['max']:.4f} | "
              f"{d['support']['min']}-{d['support']['max']} | "
              f"{d['rank_imbalance_max_over_mean']['min']:.2f}-{d['rank_imbalance_max_over_mean']['max']:.2f} | "
              f"{g['rows']:,} rows "
              f"({'self' if g['is_self'] else 'remote'} cell, row peak {g['row_peakedness']:.2f}) |")
    a('')
    ladder = ' < '.join(f"{f['id']} {f['analytic_peakedness_16x_max_share']:.2f}"
                        for f in summary['families'])
    a(f'Family ladder for comparison (analytic peakedness): {ladder}.')
    a('')
    a('**Positioning conclusions:**')
    a('')
    a('- **Support is dense**: captured rows send to 13-16 of 16 peers - far from every '
      'sparse family; the routing is an "all-peers, mildly tilted" point.')
    a('- **Typical concentration is between uniform and Zipf(0.5)**: P50 peakedness 1.56 '
      '(combine) / 2.30-2.67 (dispatch), HHI 0.068-0.099 vs uniform 0.0625.')
    a('- **Tail rows reach Zipf(1.0)-equivalent**: P99 peakedness 3.1-5.9, max 3.4-3.6 '
      '(OLMoE) and 5.2-10.2 (Qwen, whose global max cell is a *self* cell in a balanced '
      'dispatch row - the only echo of the diagonal-dominant family in the captures).')
    a('- **But the per-peer maxima land in *large* rows with mild internal concentration** '
      '(OLMoE combine row peak 1.26; Qwen combine row peak 1.61), so the realized per-peer '
      f"requirement is only {summary['captured_vs_uniform_floor']['olmoe-o1']['per_peer_ratio_over_floor']:.2f}x "
      f"(OLMoE) / {summary['captured_vs_uniform_floor']['qwen-o1']['per_peer_ratio_over_floor']:.2f}x "
      '(Qwen) the uniform-dense floor - i.e., captured routing sits at the *benign* end of '
      'the per-peer spectrum, much closer to uniform-dense than to any skewed family.')
    a('- **The binding whole-input requirement is set by rank imbalance, not peer skew**: '
      'captured rank totals are perfectly balanced in dispatch (1.00) but 1.50-2.96 / '
      '1.65-3.74x imbalanced in combine (OLMoE / Qwen), which alone lifts the whole-input '
      'need from the rank-balanced 30 MiB floor to 88.95/112.32 MiB at 1x. Rank imbalance, '
      'not routing shape, is what makes the historical 96/256 MiB slots nearly-tight/'
      'moderately-utilized in v1.')
    a('')
    a('## 7. Cross-reference to v1')
    a('')
    a('| quantity | v1 (captured) | v2 re-derivation |')
    a('|---|---|---|')
    a('| OLMoE per-peer max (incl self) | 7,118,848 B = 6.7891 MiB | same anchor, asserted; '
      '= 1.22x uniform-dense floor 5.5625 MiB |')
    a('| Qwen per-peer max (incl self) | 10,297,344 B = 9.8203 MiB | same anchor, asserted; '
      '= 1.40x uniform-dense floor 7.0234 MiB (ceil-row caliber, as in the tables) |')
    a('| OLMoE whole-input max | 93,265,920 B = 88.9453 MiB | = uniform-dense 1x whole min '
      '(family-invariant) = 480 MiB x 18.53% max-rank share |')
    a('| Qwen whole-input max | 117,772,288 B = 112.3164 MiB | = uniform-dense 1x whole min '
      '(family-invariant) = 480 MiB x 23.40% max-rank share |')
    a('| v1 admission counts at 64/96 MiB | olmoe 9/0, qwen 40/9 | reproduced exactly by the '
      'family replay at 1x (asserted) |')
    a('| v1 utilization at historical L | olmoe 92.65%, qwen 43.87% (whole) | identical at 1x '
      'for every family (whole guard family-invariant) |')
    a('')
    a('## 8. Limitations')
    a('')
    for lim in summary['limitations']:
        a(f'- {lim}.')
    a('- No admission result here measures runtime benefit, fallback frequency or fallback '
      'cost; rejection remains the documented deployed behavior.')
    a('- fwd0 rows include tiny single-peer artifacts (support 1, peakedness 16) of the '
      '1,024-token first chunk; they are reported in `summary.json` but excluded from the '
      'positioning conclusions above, which use fwd1.')
    a('')
    a('## 9. Files')
    a('')
    a('- `analysis.py` - this analysis (rerun: `python3 analysis.py`; writes the three other '
      'files; deterministic apart from the timestamp).')
    a('- `summary.json` - machine-readable results: conversion, reference profiles, full '
      'family x scale grid, replay percentiles, captured positioning, break-evens, '
      'limitations, input manifest.')
    a('- `inputs.sha256.md` - path/SHA-256/size/role of every input file read.')
    a('- `report.md` - this report.')
    (OUT_DIR / 'report.md').write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
