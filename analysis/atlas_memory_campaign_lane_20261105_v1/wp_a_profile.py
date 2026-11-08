"""Frozen WP-A profile analysis (A7): per model x arm watermark timeline.

Pure CPU over the audited receipts (completion.json per job). Produces the
preregistered WP-A table: per model, native-vs-PACE sampled-max-used delta
and the SHMEM-init free delta (T2a->T2b), every peak labeled "sampled
maximum (lower bound)" with the lens stated. Never sums lenses; no causal
attribution. Frozen with the lane manifest before the first formal job.

Usage (device host): python3 wp_a_profile.py <results_root> <out_json>
"""
import json
import sys
from pathlib import Path

import plan


def load_completions(results_root):
    jobs = Path(results_root) / 'jobs'
    rows = {}
    for directory in sorted(jobs.iterdir()) if jobs.is_dir() else []:
        path = directory / 'completion.json'
        if path.is_file():
            rows[directory.name] = json.loads(path.read_text())
    return rows


def shmem_init_delta(payload_dir):
    """PACE only: free at T2a minus free at T2b (rank 0)."""
    payload = Path(payload_dir)
    frees = {}
    for name in ('T2a', 'T2b'):
        path = payload / f'fence_{name}.json'
        if not path.is_file():
            return None
        rows = json.loads(path.read_text())
        frees[name] = next(row['capture']['free_bytes'] for row in rows
                           if row.get('rank') == 0)
    return frees['T2a'] - frees['T2b']


def profile(rows, jobs_root=None):
    models = {}
    for job, row in sorted(rows.items()):
        if row.get('status') != 'PASS' or row.get('quarantined'):
            continue
        spec = plan.job_spec(job, 0)
        if spec['workpackage'] != 'WP-A':
            continue
        entry = {'job': job, 'duration_seconds': row.get('duration_seconds'),
                 'duration_warning': row.get('duration_gate', {}).get('warning'),
                 'l1_max_used_bytes': row.get('l1_max_used_bytes'),
                 'l1_min_free_bytes': row.get('l1_min_free_bytes'),
                 'reconciliation_bias_summary': (row.get('reconciliation') or {})
                     .get('bias_summary_bytes'),
                 'engine_num_gpu_blocks': row.get('engine_num_gpu_blocks'),
                 'kv_floor': row.get('kv_floor'),
                 'a5_free_t1_gib': row.get('a5_free_t1_gib')}
        models.setdefault(spec['model'], {}).setdefault(spec['arm'], []).append(entry)
    table = {}
    for model, arms in models.items():
        native = arms.get('A') or arms.get('B-A')
        pace = arms.get('P') or arms.get('B-B')
        delta = None
        if native and pace:
            native_max = max(row['l1_max_used_bytes'] for row in native
                             if row.get('l1_max_used_bytes') is not None)
            pace_max = max(row['l1_max_used_bytes'] for row in pace
                           if row.get('l1_max_used_bytes') is not None)
            if native_max is not None and pace_max is not None:
                delta = pace_max - native_max
        # R2-A3: the SHMEM-init free delta (T2a->T2b) is a headline column
        # (PLAN §2.3) — compute it from the PACE arm's fencepost payload when
        # the jobs root is available.
        shmem_delta = None
        if pace and jobs_root is not None:
            for row in pace:
                shmem_delta = shmem_init_delta(Path(jobs_root) / 'jobs' / row['job'] / 'payload')
                if shmem_delta is not None:
                    break
        table[model] = {'arms': arms,
                        'native_vs_pace_sampled_max_used_delta_bytes': delta,
                        'shmem_init_free_delta_bytes': shmem_delta,
                        'wording': {'peak': plan.WORDING_SHEET['peak'],
                                    'delta': plan.WORDING_SHEET['delta']}}
    return table


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: wp_a_profile.py <results_root> <out_json>')
    rows = load_completions(sys.argv[1])
    value = {'schema': 'revision2.memory-campaign-wp-a-profile.v1',
             'table': profile(rows, jobs_root=sys.argv[1]), 'jobs': sorted(rows),
             'scope': 'sampled maxima are lower bounds; lenses never summed; '
                      'no causal transport-only attribution'}
    out = Path(sys.argv[2])
    if out.exists():
        raise SystemExit('refuse to overwrite an existing profile')
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'event': 'wp_a_profile_written', 'models': sorted(value['table'])}))


if __name__ == '__main__':
    main()
