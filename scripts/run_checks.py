"""Run offline implementation, evidence, and publication checks without API calls."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ['navigator.py', 'check'],
    ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    ['scripts/verify_evidence.py'],
    ['scripts/scan_repository.py'],
    ['navigator.py', 'summary'],
]
if __name__ == '__main__':
    for command in COMMANDS:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
