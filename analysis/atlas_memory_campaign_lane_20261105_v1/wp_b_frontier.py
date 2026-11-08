"""Frozen WP-B frontier analysis (A7): capacity/throughput/fill/wording.

Pure CPU over the audited receipts. Produces: (1) the capacity frontier
table (C* per arm x L with the full probe history and boundary behavior),
(2) the throughput cell table, (3) the fill-probe headroom table, (4) the
preregistered wording sheet mapping every number to permitted sentences.
Configured-block differences only — never a percent "usable loss" claim.
Frozen with the lane manifest before the first formal job.

Usage (device host): python3 wp_b_frontier.py <results_root> <out_json>
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


def frontier(rows):
    history = {}
    sweeps = []
    fills = []
    for job, row in sorted(rows.items()):
        if row.get('status') != 'PASS':
            continue
        spec = plan.job_spec(job, 0)
        if spec['workpackage'] != 'WP-B':
            continue
        probe = row.get('capacity_probe')
        if probe and probe.get('cell'):
            # R3-A2: group by (model, arm) — cross-model pooling forbidden
            key = f"{spec['model']}:{spec['arm']}"
            history.setdefault(key, []).append(
                {'C': probe['cell']['C'], 'L': probe['cell']['L'],
                 'classification': probe.get('classification'),
                 'completed': probe.get('completed'), 'submitted': probe.get('submitted'),
                 'kv_streak': row.get('kv_streak_from_log'),
                 'boundary': 'clean queueing or fault is read from the '
                             'classification + fault fields, never assumed'})
        payload = Path(job_dir_payload(rows, job))
        sweep = payload / 'throughput_sweep.json'
        if sweep.is_file():
            sweeps.append({'job': job, 'arm': spec['arm'],
                           'cells': json.loads(sweep.read_text())['cells']})
        fill = payload / 'fill_probe.json'
        if fill.is_file():
            fills.append({'job': job, 'arm': spec['arm'],
                          **json.loads(fill.read_text())})
    return {'capacity': {arm: {'c_star': plan.c_star([(row['C'], row['classification'])
                                                      for row in rows_]),
                               'probes': rows_}
                         for arm, rows_ in history.items()},
            'throughput_cells': sweeps, 'fill_probes': fills}


def job_dir_payload(rows, job):
    return rows[job].get('_payload_dir', f'jobs/{job}/payload')


def blocks_table():
    """Configured-block arithmetic only (never a 'usable loss' percent)."""
    return {arm: {'kv_gib': row['kv_gib'], 'heap_gib': row['heap_gib'],
                  'slot_L_mib': row.get('slot_L_mib'),
                  'blocks_expected': row.get('blocks_expected',
                                             plan.blocks_for(row['kv_gib'])
                                             if row['kv_gib'] else None),
                  'projection_only': 'conditional' in row}
            for arm, row in plan.WP_B_ARMS.items()}


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: wp_b_frontier.py <results_root> <out_json>')
    rows = load_completions(sys.argv[1])
    for job, row in rows.items():
        row['_payload_dir'] = str(Path(sys.argv[1]) / 'jobs' / job / 'payload')
    value = {'schema': 'revision2.memory-campaign-wp-b-frontier.v1',
             'frontier': frontier(rows), 'configured_blocks': blocks_table(),
             'wording': {'blocks': plan.WORDING_SHEET['blocks'],
                         'equal_memory': plan.WORDING_SHEET['equal_memory'],
                         'scope': plan.WORDING_SHEET['scope']},
             'scope': 'configured-block differences, measured exhaustion '
                      'boundaries and watermark deltas only; no percent-loss claim'}
    out = Path(sys.argv[2])
    if out.exists():
        raise SystemExit('refuse to overwrite an existing frontier')
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'event': 'wp_b_frontier_written',
                      'arms': sorted(value['frontier']['capacity'])}))


if __name__ == '__main__':
    main()
