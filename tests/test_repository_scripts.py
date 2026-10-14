"""Offline checks for the new question runner and publication scanner."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RepositoryScriptTests(unittest.TestCase):
    def test_week5_question_loading(self):
        run = module('run_questions')
        cases = run.load_cases(ROOT / 'evidence/test_sets/checkpoint_5_1_test_set.json')
        self.assertEqual(len(cases), 3)
        self.assertIn('question', cases[0])

    def test_week6_requires_explicit_group(self):
        run = module('run_questions')
        path = ROOT / 'evidence/test_sets/checkpoint_6_1_test_set.json'
        with self.assertRaises(ValueError):
            run.load_cases(path)
        self.assertEqual(len(run.load_cases(path, 'control')), 3)

    def test_scanner_reports_categories_without_secret_values(self):
        scanner = module('scan_repository')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            token = 'sk' + '-or-v1-' + 'a' * 64
            (root / 'README.md').write_text(token, encoding='utf8')
            report = scanner.scan(root)
            self.assertEqual(report['findings'][0]['category'], 'provider_or_github_token')
            self.assertNotIn(token, json.dumps(report))

    def test_scanner_requires_empty_example_key(self):
        scanner = module('scan_repository')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '.env.example').write_text('OPENROUTER_API_KEY=not-empty', encoding='utf8')
            self.assertTrue(scanner.scan(root)['findings'])


if __name__ == '__main__':
    unittest.main()
