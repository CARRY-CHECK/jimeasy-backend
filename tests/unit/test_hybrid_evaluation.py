import unittest

from app.evaluate_hybrid_search import evaluate_case, summarize
from app.query_gold_similarity import attach_hybrid_result


class HybridEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {'id': '1', 'question': 'q', 'answer': {'items': [
            {'name': 'item', 'source': ['a', 'b', 'excluded']},
            {'name': 'unknown', 'source': None}]}}
        self.record = {'parsed': {'conditions': {}}, 'filters': {'country': 'KR'},
                       'groups': [{'item': {'name': 'item'}, 'query': 'query'}]}
        self.chunks = {key: {'metadata': {'country': country}}
                       for key, country in [('a', 'KR'), ('b', 'KR'), ('excluded', 'JP')]}

    def fake_search(self, query, filters, conditions):
        return {'where': {'country': {'$eq': 'KR'}}, 'candidate_count': 2, 'traces': [
            {'chunk_id': 'a', 'dense_rank': 1, 'bm25_rank': 2, 'hybrid_rank': 2,
             'bm25_score': 1.0, 'rrf_score': 0.01},
            {'chunk_id': 'b', 'dense_rank': None, 'bm25_rank': 1, 'hybrid_rank': 1,
             'bm25_score': 2.0, 'rrf_score': 0.02}]}

    def test_pair_gains_losses_and_filter_exclusion(self):
        result = evaluate_case(self.case, self.record, self.chunks, self.fake_search, [1, 2])
        summary = summarize([result], [1, 2])
        self.assertEqual(summary['evaluated_items'], 1)
        self.assertEqual(summary['filter_excluded_pairs'], 1)
        self.assertEqual(summary['methods']['hybrid']['top_k']['2']['macro_recall'], 2/3)
        self.assertEqual(summary['changes']['1']['gained_gold_pairs'], 1)
        self.assertEqual(summary['changes']['1']['lost_gold_pairs'], 1)
        self.assertEqual(summary['changes']['2']['gained_gold_pairs'], 1)
        self.assertEqual(summary['changes']['1']['recovered_items'], 0)

    def test_mrr_uses_same_cutoff_for_all_methods(self):
        def search(*args):
            result = self.fake_search(*args)
            for trace in result['traces']:
                trace['dense_rank'], trace['hybrid_rank'] = 15, 11
            return result
        result = evaluate_case(self.case, self.record, self.chunks, search, [1, 10])
        summary = summarize([result], [1, 10])
        self.assertEqual(summary['methods']['dense']['mrr_at_k'], 0)
        self.assertEqual(summary['methods']['hybrid']['mrr_at_k'], 0)

    def test_attachment_preserves_cosine_and_exact_rank(self):
        comparison = evaluate_case(self.case, self.record, self.chunks, self.fake_search, [1, 2])
        exact = {key: comparison[key] for key in ('id', 'question', 'conditions', 'filters', 'where')}
        exact['items'] = []
        for item in comparison['items']:
            copy = {key: value for key, value in item.items() if key not in ('metrics', 'documents')}
            copy['documents'] = [{'chunk_id': d['chunk_id'], 'rank': 8, 'cosine_similarity': 0.42,
                                  'eligible_by_filters': d['eligible_by_filters']} for d in item['documents']]
            exact['items'].append(copy)
        attach_hybrid_result(exact, comparison, [1, 2])
        doc = exact['items'][0]['documents'][0]
        self.assertEqual(doc['rank'], 8)
        self.assertEqual(doc['cosine_similarity'], 0.42)
        self.assertEqual(doc['hybrid_rank'], 2)


if __name__ == '__main__':
    unittest.main()
