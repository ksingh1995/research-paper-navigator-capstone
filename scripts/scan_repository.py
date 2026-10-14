"""Scan only publishable code/evidence paths. Never print matching secret values.

This is a heuristic prepublication check, not a confidentiality certification.
Local presentation, PDF, cache and output directories are outside the inventory.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {'.env.example', '.gitignore', 'README.md', 'requirements.txt', 'navigator.py'}
TOP_DIRS = {'src', 'tests', 'scripts', 'docs', 'evidence'}
PATTERNS = {
    'provider_or_github_token': re.compile(r'(?:sk-or-v1-|sk-proj-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{16,}'),
    'legacy_openai_token': re.compile(r'\bsk-[A-Za-z0-9]{24,}\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'private_home_path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+[^\s"\n]+|/(?:home|Users)/[^\s"\n]+'),
    'literal_bearer_token': re.compile(r'Bearer\s+[A-Za-z0-9_-]{24,}'),
}


def publishable_files(root=ROOT):
    for file in sorted(root.rglob('*')):
        if not file.is_file():
            continue
        rel = file.relative_to(root)
        if '__pycache__' in rel.parts:
            continue
        if (len(rel.parts) == 1 and rel.name in TOP_FILES) or rel.parts[0] in TOP_DIRS:
            yield file
        elif rel.as_posix() in {'data/corpus_manifest.json', 'data/sample_papers/README.md'}:
            yield file


def scan(root=ROOT):
    findings, count = [], 0
    for file in publishable_files(root):
        count += 1
        rel = file.relative_to(root).as_posix()
        if file.suffix.lower() not in {'.py', '.md', '.txt', '.json', '.log', '.mmd', '.example', ''} and rel != '.gitignore':
            findings.append({'file': rel, 'category': 'unexpected_file_type'})
            continue
        if re.search(r'voice.?over|recording.?script|confidential|(?:^|/)private(?:/|\.)', rel, re.I):
            findings.append({'file': rel, 'category': 'excluded_filename'})
        text = file.read_text(encoding='utf8')
        for category, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append({'file': rel, 'category': category})
        if file.name == '.env.example':
            for line in text.splitlines():
                if line.startswith('OPENROUTER_API_KEY=') and line.split('=', 1)[1].strip():
                    findings.append({'file': rel, 'category': 'nonempty_example_key'})
    return {'files_scanned': count, 'findings': findings,
            'note': 'Explicit publication inventory only. Private local files and Git history are not scanned.'}


if __name__ == '__main__':
    report = scan()
    print(json.dumps(report, indent=2))
    sys.exit(bool(report['findings']))
