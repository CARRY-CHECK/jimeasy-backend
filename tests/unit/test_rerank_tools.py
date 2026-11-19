import os
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation import evaluate_rerank
from scripts.evaluation.rerank_common import RunLocked, ScoreStore, gpu_lock, unit_key
from scripts.evaluation.score_rerank import plan


def unit(qid='1-1', item='샴푸', pool_size=6, gold=('c1',)):
    pool = [{'id': f'c{i}', 'text': f't{i}', 'hybrid_rank': i, 'dense_rank': i, 'bm25_rank': None}
            for i in range(1, pool_size + 1)]
    return {'set': 'test9', 'qid': qid, 'item': item, 'query': 'q', 'item_query': item, 'gold': list(gold), 'pool': pool}


class ScoreStoreTests(unittest.TestCase):
    def test_resume_gap_and_truncated_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 's.jsonl'
            store = ScoreStore(path)
            store.append('k', 1, [0.1, 0.2, 0.3])
            store.append('k', 6, [0.9])  # 4~5가 비어 있는 구간은 이어 붙이지 않는다
            self.assertEqual(store.scores('k'), [0.1, 0.2, 0.3])
            store.append('k', 4, [0.4, 0.5])
            self.assertEqual(store.scores('k'), [0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
            with path.open('a', encoding='utf-8') as file:
                file.write('{"key": "k", "rank_from": 7, "scor')  # 중단으로 잘린 줄
            reopened = ScoreStore(path)
            self.assertEqual(reopened.scored_until('k'), 6)
            self.assertEqual(reopened.scores('missing'), [])


class PlanTests(unittest.TestCase):
    def test_plan_skips_done_ranges_and_respects_pool_size(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoreStore(Path(directory) / 's.jsonl')
            a, b = unit('1-1', pool_size=6), unit('1-2', pool_size=3)
            store.append(unit_key(a), 1, [0.0] * 4)
            todo = plan([a, b], store, 'full', 1, 5)
            self.assertEqual([(u['qid'], s, e) for u, s, e in todo], [('1-1', 5, 5), ('1-2', 1, 3)])
            with self.assertRaises(ValueError):
                plan([unit('1-3')], store, 'full', 4, 6)  # 앞 구간을 건너뛰고 시작할 수 없다


class LockTests(unittest.TestCase):
    def test_second_run_is_rejected_and_lock_released(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.lock'
            with gpu_lock(path):
                with self.assertRaises(RunLocked):
                    with gpu_lock(path):
                        pass
            self.assertFalse(path.exists())
            with gpu_lock(path):
                pass

    @unittest.skipIf(os.name == 'nt', 'POSIX에서만 죽은 프로세스를 감지한다')
    def test_stale_lock_of_dead_process_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.lock'
            path.write_text('999999\n', encoding='utf-8')
            with gpu_lock(path):
                self.assertEqual(int(path.read_text().split()[0]), os.getpid())


class EvaluateTests(unittest.TestCase):
    def test_usable_sizes_only_when_every_unit_is_scored(self):
        a, b = unit('1-1', pool_size=6), unit('1-2', pool_size=3)
        scores = {unit_key(a): [0.0] * 4, unit_key(b): [0.0] * 3}
        self.assertEqual(evaluate_rerank.usable_sizes([a, b], scores, [3, 4, 5, 'all']), [3, 4])
        scores[unit_key(a)] = [0.0] * 6
        self.assertEqual(evaluate_rerank.usable_sizes([a, b], scores, [3, 4, 5, 'all']), [3, 4, 5, 'all'])

    def test_reranking_moves_gold_up_and_cannot_exceed_pool_ceiling(self):
        evaluate_rerank.CUTOFFS = [1, 3]
        u = unit(pool_size=6, gold=('c5', 'c9'))  # c9는 pool 밖
        key = unit_key(u)
        scores = {'full': {key: [0.1, 0.2, 0.3, 0.4, 0.9, 0.0]}}
        rows = {(r['N'], r['variant']): r for r in evaluate_rerank.evaluate([u], scores, [6])}
        hybrid, rerank = rows[(6, 'hybrid_order')]['items'][0], rows[(6, 'rerank_full')]['items'][0]
        self.assertEqual(hybrid['hits'][1], 0)
        self.assertEqual(rerank['hits'][1], 1)
        self.assertEqual(rerank['pool_hits'], 1)  # 천장: 정답 2개 중 1개만 pool 안
        self.assertEqual(rerank['hits'][3], 1)

    def test_dense_pool_reuses_scores_by_chunk_id_and_stops_at_first_gap(self):
        u = unit(pool_size=4, gold=('c3',))
        # Hybrid 순서 c1..c4, Dense 순위는 c3, c1, c2, (c4는 Dense 후보 아님)
        for d, rank in zip(u['pool'], [2, 3, 1, None]):
            d['dense_rank'] = rank
        key = unit_key(u)
        dense_units, dense_scores = evaluate_rerank.to_dense_pool([u], {'full': {key: [0.2, 0.1, 0.9]}})  # c4는 미채점
        self.assertEqual([d['id'] for d in dense_units[0]['pool']], ['c3', 'c1', 'c2'])
        self.assertEqual(dense_scores['full'][key], [0.9, 0.2, 0.1])
        # c2까지 점수가 있으나 한 문서라도 비면 거기서 멈춘다
        _, partial = evaluate_rerank.to_dense_pool([u], {'full': {key: [0.2]}})
        self.assertEqual(partial['full'][key], [])
        self.assertEqual(evaluate_rerank.usable_sizes(dense_units, dense_scores['full'], [2, 3, 'all']), [2, 3, 'all'])


if __name__ == '__main__':
    unittest.main()
