"""Regenerate source_manifest.json over the frozen v3 source set.

Run only from the v3 source directory after every intentional edit; the
launcher pins the manifest digest, so any drift fails closed at startup.
"""
import hashlib
import json
from pathlib import Path

FILES = {'DESIGN.md', 'plan.py', 'worker.py', 'supervisor.py', 'precheck.py', 'launch.py',
         'test_cpu.py', 'make_manifest.py', 'observed_npu_info_fixture.json'}

root = Path(__file__).resolve().parent
rows = {}
for name in sorted(FILES):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit('missing or unsafe source file: ' + name)
    rows[name] = hashlib.sha256(path.read_bytes()).hexdigest()
out = root / 'source_manifest.json'
out.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
print(json.dumps({'event': 'manifest_written', 'files': len(rows),
                  'manifest_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
