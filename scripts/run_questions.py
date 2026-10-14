"""Preview recorded questions, or explicitly run them through the current system.

This does not reproduce historical baseline/agent arms or run an LLM judge.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load_cases(path, group=None):
    data = json.loads(path.read_text(encoding='utf8'))
    if isinstance(data, dict):
        if group is None or group not in data:
            raise ValueError('This test set requires --group with one of: ' + ', '.join(data))
        data = data[group]
    if not isinstance(data, list) or not data:
        raise ValueError('Expected a non-empty list of question objects')
    for case in data:
        if not isinstance(case, dict) or not isinstance(case.get('question'), str) or not case['question'].strip():
            raise ValueError('Each case needs a non-empty question string')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-set', type=Path, required=True)
    parser.add_argument('--group', help='Select a named group, e.g. control in the Week 6 set')
    parser.add_argument('--execute', action='store_true', help='Opt in to provider calls and possible charges')
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.cache')
    parser.add_argument('--model')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/current_run.json')
    args = parser.parse_args()
    try:
        cases = load_cases(args.test_set, args.group)
        if not args.execute:
            for i, case in enumerate(cases, 1):
                print(f"{case.get('id', case.get('item_id', i))}: {case['question']}")
            print('\nPREVIEW ONLY: no API calls. --execute runs the current system and can incur charges.')
            return
        if any(case.get('overlay') for case in cases):
            raise ValueError('Synthetic PDF-overlay audit cases need the original audit harness; they cannot run here.')
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / 'outputs').resolve()):
            raise ValueError('Save new user answers only under this repository outputs/ folder (ignored by Git).')
        if output.exists():
            raise ValueError('Output exists; choose a new --output name to preserve earlier answers.')
        from research_navigator.cli import Navigator, default_corpus
        nav = Navigator(args.corpus or default_corpus(), args.cache_dir, model=args.model)
        result = {'system': 'current final CLI', 'historical_arm_reproduction': False,
                  'judge_performed': False, 'records': []}
        output.parent.mkdir(parents=True, exist_ok=True)
        for i, case in enumerate(cases, 1):
            # Never pass expected sources, title labels, or grading notes to Navigator.
            answer = nav.ask(case['question'], use_cache=not args.no_cache)
            result['records'].append({'case_id': case.get('id', case.get('item_id', str(i))), **answer})
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
            print(f"Saved case {i}/{len(cases)}; route={answer.get('route')}")
        print('Saved current-system results. No historical score or judge pass is implied.')
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
