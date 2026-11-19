"""오프라인 평가의 순위·분모·실패 처리·한국어 토큰화 회귀 테스트."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_dataset import evaluate_dataset, export_report, match_items, source_id


def item(name='배터리', source=None, **kwargs):
    return dict(name=name, carry_on='조건부 가능', checked_baggage='불가',
                notes='반드시 전원을 완전히 끄고 안전하게 포장해야 합니다', source=source, **kwargs)


def group(name, ids):
    return {'item': {'name': name}, 'documents': [{'id': key, 'metadata': {}} for key in ids]}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'reference.json'
        self.run = self.root / 'attempt_1'
        (self.run / 'records').mkdir(parents=True)

    def evaluate(self, gold, predicted=None, groups=None, status='success', write_record=True,
                 ks=(5,), source_key='chunk_id'):
        question = '배터리를 가져갈 수 있나요?'
        reference = {'category': [{'id': 'q', 'question': question, 'answer': {'items': gold}}]}
        self.dataset.write_text(json.dumps(reference, ensure_ascii=False), encoding='utf-8')
        if write_record:
            record = {'id': 'q', 'question': question, 'status': status,
                      'answer': {'items': predicted if predicted is not None else copy.deepcopy(gold)}}
            if groups is not None:
                record['groups'] = groups
            (self.run / 'records/q.json').write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
        return evaluate_dataset(self.dataset, self.run, ks, source_key)

    def test_rank_metrics_use_search_order_not_citation_subset_or_id_sort(self):
        report = self.evaluate([item(source=[{'chunk_id': 'a#0'}, {'chunk_id': 'b#0'}])],
                               groups=[group('배터리', ['z#0', 'b#0', 'a#0'])], ks=(1, 3, 5))
        scores = report['retrieval']['metrics']
        self.assertEqual(scores['recall@1'], 0)
        self.assertEqual(scores['recall@3'], 1)
        self.assertAlmostEqual(scores['precision@3'], 2 / 3)
        self.assertAlmostEqual(scores['precision@5'], 2 / 5)
        self.assertAlmostEqual(scores['f1@3'], 0.8)
        self.assertEqual(scores['hit_rate@3'], 1)
        self.assertAlmostEqual(scores['mrr@3'], 0.5)
        self.assertAlmostEqual(scores['map@3'], (1 / 2 + 2 / 3) / 2)
        self.assertEqual(report['retrieval']['count'], 1)

    def test_unannotated_and_empty_sources_are_not_perfect_or_failed_retrieval(self):
        for sources, reason in [(None, 'unannotated_sources'), ([], 'no_positive_sources')]:
            with self.subTest(sources=sources):
                report = self.evaluate([item(source=sources)], groups=[group('배터리', ['a#0'])])
                self.assertEqual(report['coverage']['retrieval_skipped'], {reason: 1})
                self.assertIsNone(report['retrieval']['metrics']['recall@5'])
                self.assertEqual(report['classification']['overall']['accuracy'], 1)

    def test_missing_record_counts_as_failure_and_zero_search_when_gold_exists(self):
        report = self.evaluate([item(source=['a#0'])], write_record=False)
        self.assertEqual(report['coverage']['missing_records'], ['q'])
        self.assertEqual(report['classification']['overall']['accuracy'], 0)
        self.assertEqual(report['question_all_decisions_accuracy'], 0)
        self.assertEqual(report['retrieval']['metrics']['recall@5'], 0)
        self.assertEqual(report['text']['mean_sentence_bleu4'], 0)

    def test_error_record_is_not_accepted_even_with_diagnostic_answer(self):
        report = self.evaluate([item(source=['a#0'])], groups=[group('배터리', ['a#0'])], status='error')
        self.assertEqual(report['classification']['overall']['accuracy'], 0)
        self.assertEqual(report['retrieval']['metrics']['recall@5'], 1)

    def test_missing_trace_is_not_replaced_by_public_source(self):
        report = self.evaluate([item(source=['a#0'])])
        self.assertEqual(report['coverage']['retrieval_skipped'], {'missing_retrieval_trace': 1})
        self.assertIsNone(report['retrieval']['metrics']['recall@5'])

    def test_reordered_items_match_by_name_and_extra_items_are_penalized(self):
        gold = [item('배터리'), item('가위')]
        report = self.evaluate(gold, predicted=[gold[1], gold[0], item('없는 물품')], groups=[])
        self.assertEqual(report['coverage']['matched_items'], 2)
        self.assertEqual(report['coverage']['extra_items'], 1)
        self.assertAlmostEqual(report['classification']['overall']['accuracy'], 4 / 6)
        self.assertEqual(report['question_all_decisions_accuracy'], 0)

    def test_aliases_are_explicit_and_duplicate_names_are_never_position_matched(self):
        expected = [item('보조배터리', aliases=['power bank'])]
        matches, _, extra = match_items(expected, [item(' POWER   BANK ')])
        self.assertEqual(matches, {0: 0})
        self.assertEqual(extra, [])
        matches, reasons, extra = match_items([item(), item()], [item(), item()])
        self.assertEqual(matches, {})
        self.assertEqual(set(reasons.values()), {'ambiguous_item'})
        self.assertEqual(extra, [0, 1])

    def test_null_reference_is_separate_and_missing_is_not_abstention(self):
        gold = item()
        gold['carry_on'] = None
        report = self.evaluate([gold], predicted=[])
        self.assertEqual(report['classification']['overall']['count'], 1)
        self.assertEqual(report['classification']['overall']['accuracy'], 0)
        self.assertEqual(report['classification']['overall']['macro_f1'], 0)

    def test_empty_item_answer_is_scored_at_question_level(self):
        report = self.evaluate([], predicted=[])
        self.assertEqual(report['question_all_decisions_accuracy'], 1)
        self.assertIsNone(report['classification']['overall']['accuracy'])
        self.assertIsNone(report['text']['corpus_bleu4'])

    def test_korean_text_is_preserved_and_null_notes_are_excluded(self):
        gold = item()
        report = self.evaluate([gold])
        self.assertAlmostEqual(report['text']['mean_sentence_bleu4'], 1)
        for score in report['text']['rouge'].values():
            self.assertEqual(score['f1'], 1)
        gold['notes'] = None
        report = self.evaluate([gold])
        self.assertEqual(report['text']['count'], 0)
        self.assertIsNone(report['text']['rouge'])

    def test_wrong_korean_text_does_not_score_as_identical_empty_tokens(self):
        prediction = item()
        prediction['notes'] = '무관한 설명 문구'
        report = self.evaluate([item()], predicted=[prediction])
        self.assertEqual(report['text']['rouge']['rouge1']['f1'], 0)

    def test_plural_sources_and_document_granularity_deduplicate_first_rank(self):
        gold = item()
        del gold['source']
        gold['sources'] = [{'doc_id': 'document'}]
        docs = [{'id': 'a#0', 'metadata': {'doc_id': 'document'}},
                {'id': 'b#0', 'metadata': {'doc_id': 'document'}}]
        report = self.evaluate([gold], groups=[{'item': {'name': '배터리'}, 'documents': docs}],
                               source_key='doc_id', ks=(1,))
        self.assertEqual(report['retrieval']['metrics']['recall@1'], 1)
        self.assertEqual(report['details']['retrieval'][0]['retrieved_ids'], ['document'])

    def test_mixed_source_granularity_is_rejected_without_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, 'chunk_id'):
            self.evaluate([item(source=[{'source_url': 'https://example.org'}])])
        with self.assertRaisesRegex(ValueError, 'doc_id'):
            source_id({'id': 'chunk'}, 'doc_id', retrieved=True)

    def test_conflicting_source_fields_are_rejected(self):
        gold = item(source=['a#0'], sources=['b#0'])
        with self.assertRaisesRegex(ValueError, 'source와 sources'):
            self.evaluate([gold])

    def test_id_migration_maps_saved_retrieval_without_modifying_inputs(self):
        self.evaluate([item(source=['new.document'])], groups=[{
            'item': {'name': '배터리'}, 'documents': [{'id': 'old#0', 'metadata': {'doc_id': 'old.document'}}]}],
                      source_key='doc_id')
        mapping = self.root / 'mapping.json'
        mapping.write_text(json.dumps({'document_id_map': {'old.document': 'new.document'}}), encoding='utf-8')
        before = (self.run / 'records/q.json').read_bytes()
        report = evaluate_dataset(self.dataset, self.run, [1], 'doc_id', mapping)
        self.assertEqual(report['retrieval']['metrics']['recall@1'], 1)
        self.assertEqual(report['details']['retrieval'][0]['retrieved_ids'], ['new.document'])
        self.assertIsNotNone(report['provenance']['id_mapping']['sha256'])
        self.assertEqual(before, (self.run / 'records/q.json').read_bytes())

    def test_wrong_chunk_of_same_document_is_not_a_hit(self):
        report = self.evaluate([item(source=['kr.rule.0001#20'])], groups=[
            group('배터리', ['kr.rule.0001#0', 'kr.rule.0001#20'])], ks=(1, 2))
        self.assertEqual(report['retrieval']['metrics']['hit_rate@1'], 0)
        self.assertEqual(report['retrieval']['metrics']['recall@1'], 0)
        self.assertEqual(report['retrieval']['metrics']['recall@2'], 1)
        self.assertEqual(report['retrieval']['metrics']['precision@2'], 0.5)
        self.assertEqual(report['settings']['retrieval_unit'], 'chunk')
        self.assertEqual(len(report['details']['retrieval'][0]['retrieved_ids']), 2)

    def test_document_ids_and_malformed_chunk_ids_are_rejected(self):
        for identifier in ['kr.rule.0001', 'kr.rule.0001#', 'kr.rule.0001#-1', 'kr.rule.0001#x']:
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, 'chunk_id'):
                self.evaluate([item(source=[identifier])])
        with self.assertRaisesRegex(ValueError, 'chunk_id'):
            self.evaluate([item(source=['a#0'])], groups=[group('배터리', ['a'])])

    def test_chunk_migration_keeps_exact_chunk_and_rejects_document_mapping(self):
        old_chunk = 'law.항공보안법 시행규칙.본문#6'
        self.evaluate([item(source=['kr.rule.0001#6'])], groups=[group('배터리', [old_chunk])])
        mapping = self.root / 'chunk_mapping.json'
        mapping.write_text(json.dumps({'chunk_id_map': {old_chunk: 'kr.rule.0001#6'}}), encoding='utf-8')
        report = evaluate_dataset(self.dataset, self.run, [1], id_map_path=mapping)
        self.assertEqual(report['retrieval']['metrics']['recall@1'], 1)
        mapping.write_text(json.dumps({'chunk_id_map': {old_chunk: 'kr.rule.0001'}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'chunk_id'):
            evaluate_dataset(self.dataset, self.run, [1], id_map_path=mapping)

    def test_csv_export_matches_json_metrics_and_keeps_korean(self):
        report = self.evaluate([item(source=['a#0'])], groups=[group('배터리', ['a#0'])])
        folder = self.root / 'evaluate'
        export_report(report, folder)
        final = folder / 'final_answer'
        self.assertEqual(json.loads((final / 'report.json').read_text(encoding='utf-8')),
                         {key: value for key, value in report.items() if key != 'query_structure'})
        self.assertTrue((folder / 'summary.csv').read_bytes().startswith(b'\xef\xbb\xbf'))
        with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(float(next(row['값'] for row in rows if row['지표'] == 'recall@5')), 1)
        with (final / 'decisions.csv').open(encoding='utf-8-sig', newline='') as file:
            decisions = list(csv.DictReader(file))
        self.assertEqual(decisions[0]['name'], '배터리')
        self.assertEqual(decisions[0]['correct'], 'True')
        self.assertEqual((folder / 'summary.csv').read_bytes(), (final / 'summary.csv').read_bytes())
        query = json.loads((folder / 'query_structure/outputs.json').read_text(encoding='utf-8'))
        self.assertEqual(query['evaluation_status'], 'pending_reference')
        self.assertIsNone(query['metrics'])
        self.assertIsNone(query['outputs'][0]['parsed'])
        guide = (folder / 'README.md').read_text(encoding='utf-8')
        for name in ['decisions.csv', 'diagnostics.csv', 'Action Items', 'expected_parsed']:
            self.assertIn(name, guide)
        with self.assertRaisesRegex(ValueError, '이미 있습니다'):
            export_report(report, folder)

    def test_retrieval_is_scored_per_item_not_against_global_sources(self):
        gold = [item('샴푸', source=['shampoo#0']), item('선크림', source=['suncream#0'])]
        report = self.evaluate(gold, predicted=list(reversed(gold)), groups=[
            group('선크림', ['shampoo#0']), group('샴푸', ['suncream#0'])])
        self.assertEqual(report['classification']['overall']['accuracy'], 1)
        self.assertEqual(report['retrieval']['count'], 2)
        self.assertEqual(report['retrieval']['metrics']['recall@5'], 0)
        self.assertEqual([row['expected_ids'] for row in report['details']['retrieval']],
                         [['shampoo#0'], ['suncream#0']])

    def test_query_structure_output_is_preserved_without_being_scored(self):
        self.evaluate([item()])
        path = self.run / 'records/q.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        record['parsed'] = {'conditions': {'travel_date': None},
                            'items': [{'name': '배터리', 'details': '100Wh', 'item_id': 'item_1'}]}
        path.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
        report = evaluate_dataset(self.dataset, self.run)
        folder = self.root / 'separated'
        export_report(report, folder)
        outputs = json.loads((folder / 'query_structure/outputs.json').read_text(encoding='utf-8'))
        self.assertEqual(outputs['outputs'][0]['parsed'], record['parsed'])
        self.assertIsNone(outputs['metrics'])
        final = json.loads((folder / 'final_answer/report.json').read_text(encoding='utf-8'))
        self.assertNotIn('query_structure', final)

    def test_cli_default_output_uses_separated_evaluate_folders(self):
        from scripts.evaluate_dataset import main
        self.evaluate([item()])
        command = ['evaluate_dataset', '--dataset', str(self.dataset), '--run', str(self.run)]
        with patch('scripts.evaluate_dataset.ROOT', self.root), patch('sys.argv', command):
            self.assertEqual(main(), 0)
        folders = list((self.run.parent / 'evaluate').iterdir())
        self.assertEqual(len(folders), 1)
        self.assertTrue((folders[0] / 'final_answer/report.json').is_file())
        self.assertTrue((folders[0] / 'query_structure/outputs.json').is_file())
        self.assertTrue((folders[0] / 'summary.csv').is_file())

    def test_question_mismatch_and_invalid_k_are_rejected(self):
        self.evaluate([item()])
        path = self.run / 'records/q.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        record['question'] = '다른 질문'
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '질문이 다릅니다'):
            evaluate_dataset(self.dataset, self.run)
        with self.assertRaisesRegex(ValueError, 'K는'):
            evaluate_dataset(self.dataset, self.run, [0])

    def test_cli_works_without_api_keys_and_does_not_overwrite(self):
        import os
        self.evaluate([item()])
        output = self.root / 'report.json'
        command = [sys.executable, '-m', 'scripts.evaluate_dataset', '--dataset', str(self.dataset),
                   '--run', str(self.run), '--output', str(output)]
        env = dict(os.environ, OPENAI_API_KEY='', OPENAI_MODEL='', LAW_API_KEY='', PYTHONIOENCODING='utf-8')
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                                   capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        original = output.read_bytes()
        report = json.loads(original)
        self.assertEqual(report['settings']['llm_calls'], 0)
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                                   capture_output=True, text=True, encoding='utf-8')
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
