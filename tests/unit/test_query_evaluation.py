import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.query_evaluation import evaluate_queries
from scripts.evaluate_dataset import evaluate_dataset, export_report


def parsed():
    return {'conditions': {'departure': {'place': '한국', 'airline': None},
                           'arrival': {'place': '일본', 'airline': None},
                           'travel_date': None, 'baggage_type': '기내'},
            'items': [{'name': '샴푸', 'details': '200ml'}, {'name': '선크림', 'details': None}]}


class QueryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'queries.json'
        self.cases = {'q': {'id': 'q', 'question': '질문'}}
        self.gold = parsed()
        self.save()

    def save(self):
        self.path.write_text(json.dumps({'category': [{'id': 'q', 'question': '질문',
                              'expected_parsed': self.gold}]}, ensure_ascii=False), encoding='utf-8')

    def evaluate(self, prediction):
        return evaluate_queries(self.path, self.cases, {'q': {'parsed': prediction, 'status': 'success'}})

    def test_exact_match_ignores_runtime_item_id_and_item_order(self):
        prediction = parsed()
        prediction['items'].reverse()
        prediction['items'][0]['item_id'] = 'item_1'
        result = self.evaluate(prediction)
        for key in ['query_exact_match', 'condition_field_accuracy', 'item_precision', 'item_recall', 'details_accuracy']:
            self.assertEqual(result['metrics'][key], 1)

    def test_null_and_missing_and_invented_fields_are_distinct(self):
        prediction = parsed()
        prediction['conditions']['departure']['place'] = None
        prediction['conditions']['arrival']['airline'] = '架空航空'
        del prediction['conditions']['travel_date']
        result = self.evaluate(prediction)
        self.assertEqual(result['metrics']['condition_omission_count'], 1)
        self.assertEqual(result['metrics']['condition_addition_count'], 1)
        self.assertAlmostEqual(result['metrics']['condition_field_accuracy'], 3 / 6)
        self.assertEqual(result['metrics']['query_exact_match'], 0)
        self.assertEqual(result['metrics']['valid_parsed_questions'], 0)

    def test_missing_and_extra_items_are_in_precision_recall_denominators(self):
        prediction = parsed()
        prediction['items'] = [prediction['items'][0], {'name': '추가물품', 'details': None}]
        result = self.evaluate(prediction)
        self.assertEqual(result['metrics']['item_precision'], 0.5)
        self.assertEqual(result['metrics']['item_recall'], 0.5)
        self.assertEqual(result['metrics']['details_accuracy'], 0.5)

    def test_different_details_are_not_semantically_guessed(self):
        prediction = parsed()
        prediction['items'][0]['details'] = '0.2L'
        result = self.evaluate(prediction)
        self.assertEqual(result['metrics']['nonnull_details_accuracy'], 0)
        self.assertEqual(result['metrics']['null_details_accuracy'], 1)
        self.assertEqual(result['metrics']['items_exact_match'], 1)
        self.assertEqual(result['metrics']['query_exact_match'], 0)

    def test_missing_record_is_not_excluded(self):
        result = evaluate_queries(self.path, self.cases, {})
        self.assertEqual(result['metrics']['query_exact_match'], 0)
        self.assertEqual(result['metrics']['item_recall'], 0)
        self.assertEqual(result['metrics']['condition_field_accuracy'], 0)

    def test_question_and_schema_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '질문이 다릅니다'):
            evaluate_queries(self.path, {'q': {'question': '다른질문'}}, {})
        del self.gold['conditions']['travel_date']
        self.save()
        with self.assertRaisesRegex(ValueError, '스키마'):
            self.evaluate(parsed())

    def test_empty_item_list_is_exact_when_schema_is_valid(self):
        self.gold['items'] = []
        self.save()
        result = self.evaluate(copy.deepcopy(self.gold))
        self.assertEqual(result['metrics']['query_exact_match'], 1)
        self.assertIsNone(result['metrics']['details_accuracy'])

    def test_integrated_export_separates_query_and_final_scores(self):
        dataset = self.root / 'questions.json'
        dataset.write_text(json.dumps({'category': [{'id': 'q', 'question': '질문',
                            'answer': {'items': []}}]}, ensure_ascii=False), encoding='utf-8')
        run = self.root / 'run'
        (run / 'records').mkdir(parents=True)
        (run / 'records/q.json').write_text(json.dumps({'id': 'q', 'question': '질문', 'status': 'success',
                    'parsed': parsed(), 'answer': {'items': []}, 'groups': []}, ensure_ascii=False), encoding='utf-8')
        report = evaluate_dataset(dataset, run, query_dataset_path=self.path)
        folder = self.root / 'evaluate'
        export_report(report, folder)
        query = json.loads((folder / 'query_structure/report.json').read_text(encoding='utf-8'))
        final = json.loads((folder / 'final_answer/report.json').read_text(encoding='utf-8'))
        self.assertEqual(query['metrics']['query_exact_match'], 1)
        self.assertNotIn('query_structure', final)
        self.assertTrue((folder / 'query_structure/conditions.csv').is_file())
        self.assertIn('질의 구조화', (folder / 'summary.csv').read_text(encoding='utf-8-sig'))


if __name__ == '__main__':
    unittest.main()
