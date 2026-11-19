import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app import rerank, search, settings

DOCS = [{'id': f'c{i}', 'text': f't{i}', 'metadata': {}} for i in range(1, 7)]
CONDITIONS = {'departure': {'place': '인천', 'airline': '대한항공'}, 'arrival': {'place': '도쿄', 'airline': None}}


def hybrid_result(docs=DOCS):
    return {'hybrid': list(docs)}


class ConfigTests(unittest.TestCase):
    def test_shipped_config_is_hybrid_rerank_30_final_8(self):
        self.assertEqual(settings.DEFAULT_K, 8)
        self.assertEqual(settings.RETRIEVAL['method'], 'hybrid')
        self.assertEqual(settings.RETRIEVAL['rerank_pool'], 30)
        self.assertEqual(settings.RETRIEVAL['candidate_k'], 50)
        self.assertTrue(settings.RETRIEVAL['airline_filter'])
        self.assertEqual(settings.RETRIEVAL['reranker']['model'], 'BAAI/bge-reranker-v2-m3')

    def test_validation_rejects_bad_settings(self):
        base = {'method': 'hybrid', 'candidate_k': 50, 'rerank_pool': 30, 'reranker': {'model': 'm'}}
        settings._validate_retrieval(base, 8)
        for change in ({'method': 'bm25'}, {'candidate_k': 0}, {'rerank_pool': 51}, {'rerank_pool': -1},
                       {'reranker': {}}, {'reranker': {'model': 'm', 'dtype': 'bf16'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                settings._validate_retrieval(dict(base, **change), 8)
        with self.assertRaises(ValueError):
            settings._validate_retrieval(base, 0)
        settings._validate_retrieval({'method': 'dense', 'candidate_k': 50}, 5)  # 이전 방식은 reranker 설정 불필요


class SearchModeTests(unittest.TestCase):
    def test_hybrid_reranks_pool_and_returns_final_k(self):
        with patch('app.hybrid_retrieval.compare_retrieval', return_value=hybrid_result()) as compare, \
                patch('app.rerank.rerank', side_effect=lambda q, docs, k: docs[::-1][:k]) as reranker:
            result = search.retrieve_for_item('보조배터리', None, 3, CONDITIONS)
        args, kwargs = compare.call_args
        self.assertEqual(args[2], 30)  # 후보 pool = rerank_pool
        self.assertEqual((kwargs['candidate_k'], kwargs['airline_scope']), (50, True))
        self.assertEqual(kwargs['conditions'], CONDITIONS)
        reranker.assert_called_once()
        self.assertEqual([d['id'] for d in result], ['c6', 'c5', 'c4'])

    def test_trace_records_pool_before_rerank_without_changing_result(self):
        trace = {}
        with patch('app.hybrid_retrieval.compare_retrieval', return_value=hybrid_result()), \
                patch('app.rerank.rerank', side_effect=lambda q, docs, k: docs[::-1][:k]):
            result = search.retrieve_for_item('x', None, 2, CONDITIONS, trace=trace)
        self.assertEqual(trace['pool_ids'], ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
        self.assertEqual([d['id'] for d in result], ['c6', 'c5'])

    def test_final_k_larger_than_pool_enlarges_the_pool(self):
        with patch('app.hybrid_retrieval.compare_retrieval', return_value=hybrid_result()) as compare, \
                patch('app.rerank.rerank', return_value=[]):
            search.retrieve_for_item('x', None, 40, CONDITIONS)
        self.assertEqual(compare.call_args.args[2], 40)
        self.assertEqual(compare.call_args.kwargs['candidate_k'], 50)

    def test_k_larger_than_candidate_k_enlarges_candidates_instead_of_failing(self):
        with patch('app.hybrid_retrieval.compare_retrieval', return_value=hybrid_result()) as compare, \
                patch('app.rerank.rerank', return_value=[]):
            search.retrieve_for_item('x', None, 60, CONDITIONS)
        self.assertEqual((compare.call_args.args[2], compare.call_args.kwargs['candidate_k']), (60, 60))

    def test_concurrent_hybrid_searches_do_not_overlap(self):
        import threading
        import time
        active, overlaps = [0], []

        def slow_compare(*args, **kwargs):
            active[0] += 1
            overlaps.append(active[0])
            time.sleep(0.05)
            active[0] -= 1
            return hybrid_result()

        with patch('app.hybrid_retrieval.compare_retrieval', side_effect=slow_compare), \
                patch('app.rerank.rerank', return_value=[]):
            threads = [threading.Thread(target=search.retrieve_for_item, args=('x', None, 2, CONDITIONS)) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(max(overlaps), 1)

    def test_dense_override_uses_previous_country_only_path(self):
        with patch.object(search, 'retrieve', return_value=[DOCS[0]]) as dense, \
                patch('app.hybrid_retrieval.compare_retrieval') as compare:
            with search.use_retrieval_method('dense'):
                self.assertEqual(search.active_retrieval()['method'], 'dense')
                result = search.retrieve_for_item('x', {'country': 'KR'}, 5, CONDITIONS)
            self.assertEqual(search.active_retrieval()['method'], settings.RETRIEVAL['method'])  # 범위가 끝나면 복원
        dense.assert_called_once_with('x', {'country': 'KR'}, 5, conditions=CONDITIONS)
        compare.assert_not_called()
        self.assertEqual(result, [DOCS[0]])
        with self.assertRaises(ValueError):
            with search.use_retrieval_method('bm25'):
                pass

    def test_rerank_pool_zero_keeps_hybrid_order(self):
        config = dict(settings.RETRIEVAL, rerank_pool=0)
        with patch.object(search, 'RETRIEVAL', config), \
                patch('app.hybrid_retrieval.compare_retrieval', return_value=hybrid_result()), \
                patch('app.rerank.rerank') as reranker:
            result = search.retrieve_for_item('x', None, 2, CONDITIONS)
        reranker.assert_not_called()
        self.assertEqual([d['id'] for d in result], ['c1', 'c2'])

    def test_invalid_input_is_rejected(self):
        for query, k in (('  ', 8), ('x', 0)):
            with self.subTest(query=query, k=k), self.assertRaises(ValueError):
                search.retrieve_for_item(query, None, k, CONDITIONS)


class RerankTests(unittest.TestCase):
    def test_orders_by_score_stable_for_ties_and_does_not_mutate_input(self):
        model = MagicMock()
        model.predict.return_value = [0.1, 0.9, 0.9, 0.5]
        docs = [{'id': str(i), 'text': f't{i}', 'metadata': {}} for i in range(4)]
        with patch.object(rerank, 'get_reranker', return_value=model):
            result = rerank.rerank('q', docs, 3)
        self.assertEqual([d['id'] for d in result], ['1', '2', '3'])  # 동점은 입력 순서
        self.assertEqual(result[0]['rerank_score'], 0.9)
        self.assertNotIn('rerank_score', docs[1])
        self.assertEqual(model.predict.call_args.args[0][0], ('q', 't0'))
        self.assertEqual(rerank.rerank('q', [], 3), [])

    def test_load_failure_raises_service_error_not_silent_fallback(self):
        with patch.object(rerank, '_model', None), patch.dict('sys.modules', {'sentence_transformers': None}):
            with self.assertRaises(RuntimeError) as error:
                rerank.get_reranker()
        self.assertIn('rerank_pool', str(error.exception))


class DatabaseCheckTests(unittest.TestCase):
    def collection(self, ids, metadatas):
        collection = MagicMock()
        collection.get.return_value = {'ids': ids, 'metadatas': metadatas}
        return collection

    def verify(self, collection, expected):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'c.jsonl'
            source.write_text('\n'.join('{}' for _ in range(expected)) + '\n', encoding='utf-8')
            with patch.object(search, 'CHUNK_SOURCES', {'KR': source}), patch.object(search.chromadb, 'PersistentClient') as client:
                client.return_value.get_collection.return_value = collection
                return search.verify_database()

    def test_complete_database_passes(self):
        meta = {'country': 'KR', 'source_type': 'airline', 'org_name': '대한항공'}
        self.assertEqual(self.verify(self.collection(['a', 'b'], [meta, meta]), 2), [])

    def test_missing_chunks_and_metadata_are_reported(self):
        meta = {'country': 'KR', 'source_type': 'airline', 'org_name': '대한항공'}
        old = {'country': 'KR', 'source_type': 'airline'}  # org_name 없는 오래된 적재
        problems = self.verify(self.collection(['a', 'b'], [meta, old]), 5)
        self.assertEqual(len(problems), 2)
        self.assertIn('청크 수', problems[0])
        self.assertIn('org_name', problems[1])

    def test_warmup_fails_loudly_on_database_problem(self):
        with patch.object(search, 'retrieve'), patch.object(search, 'verify_database', return_value=['문제']):
            with self.assertRaises(RuntimeError):
                search.warmup()


class ApiWarmupTests(unittest.TestCase):
    def run_lifespan(self):
        from app.api import main

        async def go():
            async with main.lifespan(main.app):
                pass
        asyncio.run(go())
        return main

    def test_startup_warms_models_and_survives_failure(self):
        from app.api import main
        with patch.object(main, 'warmup') as warm:
            self.run_lifespan()
        warm.assert_called_once()
        with patch.object(main, 'warmup', side_effect=RuntimeError('no model')):
            self.run_lifespan()  # 사전 로딩 실패는 서버 시작을 막지 않는다
        with patch.object(main, 'RETRIEVAL', dict(main.RETRIEVAL, warmup=False)), patch.object(main, 'warmup') as warm:
            self.run_lifespan()
        warm.assert_not_called()


if __name__ == '__main__':
    unittest.main()
