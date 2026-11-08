"""Independent CPU-only verification of a completed barrier microbenchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def verify(path):
    receipt = json.loads((path / 'completion.json').read_text())
    assert receipt['status'] == 'PASS'
    for name, field in [('container.log', 'container_log_sha256'), ('inspect.json', 'inspect_sha256'), ('postflight.json', 'postflight_sha256')]:
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == receipt[field], name
    samples = []
    for line in (path / 'container.log').read_text().splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict) and row.get('event') == 'barrier_samples':
            samples.append(row)
    world, n = receipt['world'], receipt['samples']
    assert len(samples) == world and {row['rank'] for row in samples} == set(range(world))
    for row in samples:
        values = row['samples']
        assert len(values) == n and all(pair[1] > pair[0] and (i == 0 or pair[0] >= values[i-1][1]) for i, pair in enumerate(values))
    maxima = [max(row['samples'][i][1] - row['samples'][i][0] for row in samples) / 1000 for i in range(n)]
    envelopes = [(max(row['samples'][i][1] for row in samples) - min(row['samples'][i][0] for row in samples)) / 1000 for i in range(n)]
    for field, recalculated in [('max_rank_us', maxima), ('group_envelope_us', envelopes)]:
        assert all(abs(a-b) < 1e-9 for a, b in zip(receipt[field], recalculated)) and len(receipt[field]) == n
    return {'world': world, 'samples': n, 'max_rank_median_us': statistics.median(maxima),
            'max_rank_range_us': [min(maxima), max(maxima)], 'envelope_median_us': statistics.median(envelopes),
            'envelope_range_us': [min(envelopes), max(envelopes)], 'postflight_status': receipt['postflight_status'],
            'scope': 'one process group; consecutive samples; host enqueue plus stream synchronization and peer waiting'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
