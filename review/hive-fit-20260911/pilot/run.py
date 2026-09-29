"""Run the deterministic review pilot without the original machine or checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fit_pilot import run_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory; existing results are never overwritten')
    args = parser.parse_args()
    for rel, entry in json.loads((ROOT / 'SOURCE-MANIFEST.json').read_text()).items():
        digest = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise SystemExit('Snapshot hash mismatch: ' + rel)
    out = args.output.resolve()
    run_matrix(out)
    subprocess.run([sys.executable, str(ROOT / 'pilot/audit.py'), str(out)], check=True)
    result = json.loads((out / 'summary.json').read_text())
    if result['rows_verified'] != 480:
        raise SystemExit('Incomplete assigned matrix')
    print('Verified raw outcomes:', result['rows_verified'])
    print('Results:', out)


if __name__ == '__main__':
    main()
