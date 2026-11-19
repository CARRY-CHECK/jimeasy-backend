"""실제 구조화 진단을 정답 구조화 검색 평가로 잘못 표시하지 않는다."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.settings import RETRIEVAL
from scripts.evaluation import evaluate_service_retrieval as evaluation


class ServiceRetrievalReportTests(unittest.TestCase):
    def test_model_queries_report_uses_its_own_support_and_no_gold_baselines(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chunks = root / 'chunks.jsonl'
            chunks.write_text(json.dumps({'id': 'x#0', 'metadata': {'source_type': 'government'}}), encoding='utf-8')
            summary = {
                'input': 'model_structure_queries', 'top_ks': [1, 3, 5, 8], 'k': 8,
                'retrieval': RETRIEVAL, 'item_statuses': {'scored': 1, 'unmatched_or_ambiguous_item': 1},
                'dataset': 'questions.json', 'embedding': {'model': 'test', 'parameters': {'device': 'cpu'}},
                'database': {'collection': 'test', 'chunks': 1},
                'timing_seconds': {'searches': 1, 'mean': 1, 'median': 1, 'max': 1, 'first': 1},
            }
            pairs = [{'chunk_id': 'x#0', 'eligible_by_filters': False,
                      'pool_rank': None, 'final_rank': None}]
            items = [{'pairs': pairs, 'set': 'test', 'qid': 'q', 'name': 'item'}]
            args = argparse.Namespace(runs=[root], out_dir=root, overwrite=False, title='실제 구조화')
            with patch.object(evaluation, 'load_run', return_value=(items, [summary])), \
                    patch.object(evaluation, 'CHUNK_SOURCES', {'KR': chunks}), \
                    patch.object(evaluation, 'baseline_values', side_effect=AssertionError('different support')), \
                    patch.object(evaluation, 'offline_rerank_values', side_effect=AssertionError('different input')):
                evaluation.report(args)
            report = (root / 'present.md').read_text(encoding='utf-8')
            self.assertIn('실제 LLM 구조화', report)
            self.assertIn('이름이 연결되지 않은 물품은 이 진단에서 제외', report)
            self.assertNotIn('같은 98개 물품', report)
            self.assertNotIn('ver8 오프라인 평가와의 일치', report)
            self.assertIn('| 필터에서 제외 | 1 |', report)
            self.assertIn('| 필터 통과했지만 후보 pool(30개) 밖 | 0 |', report)

    def test_mixed_input_kinds_are_rejected_before_writing(self):
        args = argparse.Namespace(runs=[])
        with patch.object(evaluation, 'load_run', return_value=([], [
                {'input': 'gold_structure_queries'}, {'input': 'model_structure_queries'}])):
            with self.assertRaisesRegex(ValueError, '서로 다른 종류'):
                evaluation.report(args)


if __name__ == '__main__':
    unittest.main()
