import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import query_parser
from app.settings import PARSE_SYSTEM_PROMPT
from app.schemas.query import ParsedQuery
from scripts import run_dataset


class PromptSelectionTests(unittest.TestCase):
    def test_selected_prompt_is_sent_to_model_and_default_is_restored(self):
        parsed = ParsedQuery.model_validate({'conditions': {
            'departure': {'place': None, 'airline': None}, 'arrival': {'place': None, 'airline': None},
            'travel_date': None, 'baggage_type': None}, 'items': []})
        with patch.object(query_parser, 'generate_structured_langchain', return_value=parsed) as generate:
            with query_parser.use_parse_system_prompt('parse_system_ver1.txt'):
                query_parser.parse_query('test question')
            self.assertEqual(generate.call_args.args[0],
                             (query_parser.PROMPTS_DIR / 'parse_system_ver1.txt').read_text(encoding='utf-8'))
            self.assertEqual(generate.call_args.kwargs['role'], 'query')
        self.assertEqual(query_parser.active_parse_system_prompt(), PARSE_SYSTEM_PROMPT)

    def test_runner_snapshots_actual_prompt_without_sending_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'reference.json'
            dataset.write_text(json.dumps({'category': [{'id': 'q', 'question': 'question only',
                                                        'answer': {'secret_reference': True}}]}), encoding='utf-8')
            def answer(question, **kwargs):
                self.assertEqual(question, 'question only')
                self.assertEqual(query_parser.active_parse_system_prompt(), 'parse_system_ver1.txt')
                return {'answer': {'items': []}}
            with patch.object(run_dataset, 'answer_question', side_effect=answer):
                summary = run_dataset.run_dataset(dataset, root / 'output', 5, 'parse_system_ver1.txt')
            saved = json.loads((root / 'output/attempt_1/prompts.json').read_text(encoding='utf-8'))
            self.assertIn('parse_system_ver1.txt', saved)
            self.assertNotIn(PARSE_SYSTEM_PROMPT, saved)
            self.assertEqual(summary['parse_system_prompt'], 'parse_system_ver1.txt')
            self.assertEqual(query_parser.active_parse_system_prompt(), PARSE_SYSTEM_PROMPT)

    def test_missing_or_invalid_prompt_fails_before_attempt_creation(self):
        for name in ['does_not_exist.txt', '../parse_system.txt']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'output'
                with self.assertRaises(ValueError):
                    run_dataset.run_dataset(Path('not_read.json'), target, parse_system_prompt=name)
                self.assertFalse(target.exists())

    def test_exception_restores_default_prompt(self):
        with self.assertRaises(RuntimeError):
            with query_parser.use_parse_system_prompt('parse_system_ver1.txt'):
                raise RuntimeError('stop')
        self.assertEqual(query_parser.active_parse_system_prompt(), PARSE_SYSTEM_PROMPT)

    def test_default_prompt_comes_from_config_and_exists(self):
        self.assertEqual(PARSE_SYSTEM_PROMPT, 'parse_system_en_ver2.txt')
        self.assertEqual(query_parser.active_parse_system_prompt(), PARSE_SYSTEM_PROMPT)
        self.assertTrue((query_parser.PROMPTS_DIR / PARSE_SYSTEM_PROMPT).read_text(encoding='utf-8').strip())


if __name__ == '__main__':
    unittest.main()

