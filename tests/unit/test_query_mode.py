import unittest

from app.evaluate_hybrid_search import evaluate_case
from app.query_gold_similarity import experiment_query, validate_record
from app.rag import build_retrieval_query

CONDITIONS = {'departure': {'place': '인천', 'airline': '대한항공'}, 'arrival': {'place': '도쿄', 'airline': None},
              'travel_date': '2026-12-25', 'baggage_type': '기내'}
ITEM = {'name': '보조배터리', 'details': '100Wh 3개', 'item_id': 'item_1'}


def record(query):
    return {'question': 'q', 'status': 'success', 'parsed': {'conditions': CONDITIONS, 'items': [ITEM]},
            'filters': {}, 'groups': [{'item': ITEM, 'query': query, 'documents': []}]}


class QueryModeTests(unittest.TestCase):
    def test_item_query_has_no_travel_conditions(self):
        self.assertEqual(build_retrieval_query(ITEM, None), '보조배터리 100Wh 3개')
        self.assertEqual(build_retrieval_query({'name': '샴푸', 'details': None}, None), '샴푸')
        self.assertIn('인천 대한항공 도쿄', build_retrieval_query(ITEM, CONDITIONS))

    def test_validation_follows_query_mode(self):
        case = {'id': 'q', 'question': 'q'}
        full, item = build_retrieval_query(ITEM, CONDITIONS), build_retrieval_query(ITEM, None)
        validate_record(case, record(full))
        validate_record(case, record(item), 'item')
        with self.assertRaises(ValueError):
            validate_record(case, record(item))
        with self.assertRaises(ValueError):
            validate_record(case, record(full), 'item')

    def test_item_airline_query_keeps_only_airlines(self):
        self.assertEqual(experiment_query(ITEM, CONDITIONS, 'item_airline'), '보조배터리 100Wh 3개 대한항공')
        round_trip = {**CONDITIONS, 'arrival': {'place': '도쿄', 'airline': 'ANA'}}
        self.assertEqual(experiment_query(ITEM, round_trip, 'item_airline'), '보조배터리 100Wh 3개 대한항공 ANA')
        self.assertEqual(experiment_query({'name': '샴푸', 'details': None}, {'departure': {'place': '인천', 'airline': None},
                                                                       'arrival': {'place': '도쿄', 'airline': None}}, 'item_airline'), '샴푸')
        with self.assertRaises(ValueError):
            experiment_query(ITEM, CONDITIONS, 'unknown')
        case = {'id': 'q', 'question': 'q'}
        validate_record(case, record('보조배터리 100Wh 3개 대한항공'), 'item_airline')

    def test_no_metadata_filter_searches_without_where_but_keeps_conditions(self):
        item = {'name': '보조배터리', 'details': None, 'item_id': 'item_1'}
        case = {'id': 'q', 'question': 'q', 'answer': {'items': [{'name': '보조배터리', 'source': ['kr.a#1']}]}}
        rec = {'question': 'q', 'status': 'success', 'filters': {}, 'parsed': {'conditions': CONDITIONS, 'items': [item]},
               'groups': [{'item': item, 'query': 'x', 'documents': []}]}
        chunks = {'kr.a#1': {'metadata': {'country': 'US', 'source_type': 'airline', 'org_name': '델타항공'}}}
        calls = []

        def search(query, filters, conditions):
            calls.append(conditions)
            trace = {'chunk_id': 'kr.a#1', 'dense_rank': 3, 'bm25_rank': None, 'bm25_score': None,
                     'rrf_score': 0.1, 'hybrid_rank': 3}
            return {'where': None, 'candidate_count': 1, 'traces': [trace]}

        result = evaluate_case(case, rec, chunks, search, (5,), metadata_filter=False)
        self.assertEqual(calls[-1], {})
        self.assertIsNone(result['where'])
        self.assertEqual(result['conditions'], CONDITIONS)
        self.assertTrue(result['items'][0]['documents'][0]['eligible_by_filters'])


if __name__ == '__main__':
    unittest.main()
