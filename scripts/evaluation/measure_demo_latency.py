# -*- coding: utf-8 -*-
"""물품 2개 질문의 검색 + reranking 시간을 순차 처리로 잰다. LLM·OpenAI 호출 없음(구조화·판정 LLM 시간은 포함하지 않는다).

python -m scripts.evaluation.measure_demo_latency --records data/test/test9_gold_query_records/test9 \\
    --output tests/ver8_test/latency/demo_two_items_mps.json

질문마다 물품 둘을 차례로 처리한다(서비스가 물품별로 검색한다는 가정, 병렬 처리는 확인하지 않음).
- Hybrid 후보(Dense 50 + BM25 50 + RRF, 국가+항공사 필터) 검색 후 앞 N개를 cross-encoder로 재정렬
- Dense 상위 N개 검색 후 재정렬
모델 로딩과 예열은 따로 기록하고 질문별 시간에서 뺀다. Hybrid 검색은 BM25 인덱스 캐시가 비었을 때(cold)와 같은 검색을 다시 했을 때(warm)를 모두 잰다.
질문 하나를 끝낼 때마다 출력 파일에 저장한다. GPU 작업이 겹치지 않도록 실행 잠금을 쓴다.
"""
import argparse
import json
import platform
import statistics
from datetime import datetime
from pathlib import Path
from time import perf_counter

from scripts.evaluation.rerank_common import RunLocked, gpu_lock

MODEL = 'BAAI/bge-reranker-v2-m3'


def pick_questions(records_dir, count):
    """물품이 둘 이상인 질문 중 항공사를 명시한 것과 아닌 것을 절반씩 고른다. 물품은 앞의 둘만 쓴다."""
    withair, without = [], []
    for path in sorted(Path(records_dir).glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if len(record['groups']) < 2:
            continue
        record = dict(record, groups=record['groups'][:2])
        c = record['parsed']['conditions']
        (withair if c['departure']['airline'] or c['arrival']['airline'] else without).append(record)
    half = count // 2
    return withair[:count - half] + without[:half]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--questions', type=int, default=6)
    parser.add_argument('--hybrid-pools', type=int, nargs='+', default=[20, 30, 50])
    parser.add_argument('--dense-pools', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'이미 있습니다: {args.output}')
    questions = pick_questions(args.records, args.questions)
    if not questions:
        parser.error('물품이 둘인 질문이 없습니다.')

    try:
        with gpu_lock():
            import torch
            from sentence_transformers import CrossEncoder
            from app.hybrid_retrieval import compare_retrieval
            from app.retrieval import retrieve
            from scripts.evaluation.score_rerank import pick_device

            device = pick_device(args.device)
            report = {'date': datetime.now().isoformat(timespec='seconds'), 'machine': platform.platform(), 'device': device,
                      'torch': torch.__version__, 'model': MODEL, 'dtype': 'fp32', 'batch_size': args.batch_size,
                      'items_per_question': 2, 'processing': 'sequential', 'llm_included': False, 'questions': []}

            def save():
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

            start = perf_counter()
            reranker = CrossEncoder(MODEL, device=device, max_length=512)
            report['reranker_load_s'] = perf_counter() - start
            first = questions[0]['groups'][0]
            c0 = questions[0]['parsed']['conditions']
            start = perf_counter()
            retrieve(first['query'], k=5, conditions=c0)  # 임베딩 모델 로딩
            report['embedding_first_call_s'] = perf_counter() - start
            reranker.predict([(first['query'], 'warmup')] * args.batch_size, batch_size=args.batch_size, show_progress_bar=False)

            def rerank(query, docs):
                start = perf_counter()
                reranker.predict([(query, d['text']) for d in docs], batch_size=args.batch_size, show_progress_bar=False)
                if device == 'mps':
                    torch.mps.synchronize()
                elif device == 'cuda':
                    torch.cuda.synchronize()
                return perf_counter() - start

            for record in questions:
                conditions = record['parsed']['conditions']
                entry = {'id': record['id'], 'airline': conditions['departure']['airline'] or conditions['arrival']['airline'],
                         'items': []}
                for group in record['groups']:
                    query, row = group['query'], {'item': group['item']['name']}
                    start = perf_counter()
                    result = compare_retrieval(query, None, max(args.hybrid_pools), conditions=conditions, candidate_k=50,
                                               airline_scope=True)
                    row['hybrid_retrieval_cold_s'] = perf_counter() - start
                    start = perf_counter()
                    compare_retrieval(query, None, max(args.hybrid_pools), conditions=conditions, candidate_k=50, airline_scope=True)
                    row['hybrid_retrieval_warm_s'] = perf_counter() - start
                    row['hybrid_rerank_s'] = {str(n): rerank(query, result['hybrid'][:n]) for n in args.hybrid_pools}
                    start = perf_counter()
                    dense = retrieve(query, None, max(args.dense_pools), conditions=conditions, airline_scope=True)
                    row['dense_retrieval_s'] = perf_counter() - start
                    row['dense_rerank_s'] = {str(n): rerank(query, dense[:n]) for n in args.dense_pools}
                    entry['items'].append(row)
                report['questions'].append(entry)
                save()
                print(f"{record['id']} 완료", flush=True)

            def total(entry, retrieval_key, rerank_key, n):
                return sum(i[retrieval_key] + i[rerank_key][str(n)] for i in entry['items'])

            summary = {}
            for n in args.hybrid_pools:
                for label, key in (('cold', 'hybrid_retrieval_cold_s'), ('warm', 'hybrid_retrieval_warm_s')):
                    values = [total(q, key, 'hybrid_rerank_s', n) for q in report['questions']]
                    summary[f'hybrid_N{n}_{label}'] = {'mean_s': statistics.mean(values), 'max_s': max(values)}
            for n in args.dense_pools:
                values = [total(q, 'dense_retrieval_s', 'dense_rerank_s', n) for q in report['questions']]
                summary[f'dense_N{n}'] = {'mean_s': statistics.mean(values), 'max_s': max(values)}
            report['summary_two_items_s'] = summary
            save()
            for name, v in summary.items():
                print(f'{name}: 평균 {v["mean_s"]:.1f}s, 최대 {v["max_s"]:.1f}s', flush=True)
    except RunLocked as error:
        print(error)
        return 2
    print('Saved:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
