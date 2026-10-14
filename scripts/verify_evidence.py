"""Check packaged historical evidence structure, not factual answer accuracy."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = ROOT / 'evidence'
    parsed = {p.relative_to(evidence).as_posix(): json.loads(p.read_text(encoding='utf8'))
              for p in evidence.rglob('*.json')}
    for week in range(2, 8):
        assert f'week{week}_results.json' in parsed, f'Missing Week {week} results'
    expected = {'checkpoint_2_1_test_set.json': 4, 'checkpoint_3_1_test_set.json': 8,
                'checkpoint_4_1_broad_direction_test_set.json': 6, 'checkpoint_5_1_test_set.json': 3,
                'checkpoint_7_1_test_set.json': 7}
    for name, count in expected.items():
        assert len(parsed[f'test_sets/{name}']) == count, name
    week6 = parsed['test_sets/checkpoint_6_1_test_set.json']
    assert len(week6['control']) == 3
    assert len(week6['security']) == 9
    week7 = parsed['week7_results.json']
    paired = [r for r in week7['records'] if r['arm'] in ('baseline', 'hardened')]
    assert len(paired) == 14
    assert len([r for r in week7['records'] if r['arm'] == 'cached_repeat']) == 1
    log = (evidence / 'week7_answers_verbatim.log').read_text(encoding='utf8')
    for record in paired:
        assert record['answer'].strip() in log, f"Missing logged answer: {record['case_id']} {record['arm']}"
    for name in ('final_interface_method.json', 'final_interface_comparison.json', 'final_interface_repeat.json'):
        assert name in parsed, name
    assert parsed['final_interface_repeat.json']['metrics']['api_calls'] == 0
    for name in ('checkpoint_2_1_retrieval.log', 'checkpoint_2_1_hybrid_retrieval.log',
                 'checkpoint_3_1_evaluation.log', 'checkpoint_4_1_broad_direction.log',
                 'checkpoint_5_1_agent.log', 'checkpoint_6_1_agent.log'):
        assert (evidence / 'logs' / name).is_file(), name
    print(f'Evidence checks passed: {len(parsed)} JSON files, recorded test sets, six logs, and 14 final Week 7 answers.')


if __name__ == '__main__':
    main()
