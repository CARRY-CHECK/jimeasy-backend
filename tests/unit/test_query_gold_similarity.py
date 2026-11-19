import unittest
from unittest.mock import patch

from app import query_gold_similarity as diagnostic
from app.settings import ROOT


class SimilarityRankingTests(unittest.TestCase):
    def setUp(self):
        self.case = {'id': 'x', 'question': 'question', 'answer': {'items': [
            {'name': 'item', 'source': ['gold', 'excluded']}]}}
        self.groups = [{'item': {'name': 'item'}, 'query': 'query'}]
        self.chunks = {
            'a': {'text': 'a', 'metadata': {'country': 'KR', 'source_type': 'government'}},
            'gold': {'text': 'gold', 'metadata': {'country': 'KR', 'source_type': 'government'}},
            'excluded': {'text': 'excluded', 'metadata': {'country': 'JP', 'source_type': 'airline'}},
        }
        self.vectors = {'query': [1, 0], 'a': [1, 0], 'gold': [1, 0], 'excluded': [1, 0]}

    def analyze(self, **kwargs):
        return diagnostic.analyze_case(self.case, self.groups, {}, self.chunks,
                                       self.vectors.__getitem__, **kwargs)['items'][0]

    def test_rank_ties_and_filtered_gold_recall(self):
        item = self.analyze(filters={'country': 'KR'}, top_ks=[1, 2, 5])
        gold, excluded = item['documents']
        self.assertEqual(item['candidate_count'], 2)
        self.assertEqual(gold['rank'], 2)
        self.assertFalse(gold['in_top_1'])
        self.assertTrue(gold['in_top_2'])
        self.assertIsNone(excluded['rank'])
        self.assertEqual(excluded['cosine_similarity'], 1)
        self.assertFalse(excluded['in_top_5'])
        self.assertEqual(item['top_k']['2']['recall'], 0.5)
        self.assertEqual(item['reciprocal_rank'], 0.5)
        self.assertEqual([c['chunk_id'] for c in item['top_candidates']], ['a', 'gold'])

    def test_ranking_uses_similarity_before_chunk_id(self):
        self.vectors['a'] = [0, 1]
        item = self.analyze(filters={'country': 'KR'})
        self.assertEqual(item['documents'][0]['rank'], 1)
        self.assertTrue(item['documents'][0]['in_top_1'])

    def test_empty_candidates_are_misses(self):
        item = self.analyze(filters={'country': 'US'})
        self.assertEqual(item['candidate_count'], 0)
        self.assertEqual(item['reciprocal_rank'], 0)
        self.assertEqual(item['top_k']['5']['recall'], 0)

    def test_no_gold_and_unmatched_items_do_not_embed(self):
        with patch.dict(self.case['answer']['items'][0], source=None):
            self.assertEqual(self.analyze()['status'], 'unannotated_sources')
        self.groups = []
        self.assertEqual(self.analyze()['status'], 'unmatched_or_ambiguous_item')

    def test_explicit_filters_intersect_place_countries(self):
        where = diagnostic.retrieval_where({'country': 'JP', 'source_type': 'government'},
                                          {'departure': {'place': '인천'}})
        self.assertFalse(any(diagnostic.matches_where(c['metadata'], where) for c in self.chunks.values()))
        where = diagnostic.retrieval_where({'source_type': 'government'},
                                          {'departure': {'place': '인천'}})
        self.assertTrue(diagnostic.matches_where(self.chunks['gold']['metadata'], where))
        self.assertFalse(diagnostic.matches_where(self.chunks['excluded']['metadata'], where))

    # 비공개 평가셋(test8)과 그 실행 기록이 있을 때만 도는 회귀 검사다. 공개 레포에는 해당 파일이 없다.
    @unittest.skipUnless((ROOT / 'data/test/test8_questions.json').exists()
                         and (ROOT / 'tests/ver2_test/attempt_1/records').is_dir(),
                         '비공개 평가셋(data/test/test8_questions.json)과 실행 기록이 없음')
    def test_stored_queries_follow_current_retrieval_builder_without_model(self):
        dataset = diagnostic.read_json(ROOT / 'data/test/test8_questions.json')
        cases = [case for group in dataset.values() for case in group]
        with patch.object(diagnostic, 'create_embedding_function', side_effect=AssertionError('model forbidden')):
            for case in cases:
                record = diagnostic.read_json(ROOT / f"tests/ver2_test/attempt_1/records/{case['id']}.json")
                diagnostic.validate_record(case, record)
        self.assertEqual(len(cases), 50)

    def test_query_drift_is_rejected(self):
        record = {'status': 'success', 'question': 'question', 'parsed': {'conditions': {}},
                  'groups': self.groups}
        with self.assertRaises(ValueError):
            diagnostic.validate_record(self.case, record)


if __name__ == '__main__':
    unittest.main()
