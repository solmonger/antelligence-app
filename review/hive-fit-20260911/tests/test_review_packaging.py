"""Behavioral regressions for public review bundle portability."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_covers_every_executable_snapshot_file():
    manifest = json.loads((ROOT / 'SOURCE-MANIFEST.json').read_text())
    expected = {str(p.relative_to(ROOT)) for folder in ['backend', 'scripts', 'pilot', 'tests']
                for p in (ROOT / folder).glob('*.py')}
    assert expected <= set(manifest)
    for rel in expected:
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == manifest[rel]['sha256']


def test_generated_matrix_can_be_relocated_and_audited(tmp_path):
    original = tmp_path / 'original'
    completed = subprocess.run([sys.executable, str(ROOT / 'pilot/run.py'),
                                '--output', str(original)], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = json.loads((original / 'summary.json').read_text())
    relocated = tmp_path / 'relocated'
    original.rename(relocated)
    rows = [json.loads(line) for line in (relocated / 'rows.jsonl').read_text().splitlines()]
    assert all(not Path(row['trace_path']).is_absolute() for row in rows)
    audited = subprocess.run([sys.executable, str(ROOT / 'pilot/audit.py'), str(relocated)],
                             cwd=ROOT, capture_output=True, text=True)
    assert audited.returncode == 0, audited.stdout + audited.stderr
    assert json.loads((relocated / 'summary.json').read_text()) == expected
