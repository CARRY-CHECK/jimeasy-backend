"""저장된 실제 쿼리로 Chroma dense / Kiwi BM25 / RRF hybrid를 동일하게 평가한다.

python -m app.evaluate_hybrid_search
python -m app.evaluate_hybrid_search --id 1-1 --candidate-k 50 --top-k 1 3 5 10

LLM 호출·쿼리 변경 없음. query_gold_similarity의 정답 매칭/필터 규칙을 재사용한다.
현재 JSONL과 DB의 ID/본문/필터 metadata가 다르면 실행을 중단한다.
MRR은 세 방식 모두 최종 max(top_k) 내 첫 정답 기준(MRR@K)으로 비교한다.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

import chromadb

from app.hybrid_retrieval import compare_retrieval
from app.query_gold_similarity import (experiment_query, gold_ids, match_groups, matches_where, read_json, sha,
                                       validate_record)
from app.retrieval import retrieval_where
from app.settings import (CHROMA_COLLECTION, CHROMA_PATH, CHUNK_SOURCES, FILTER_FIELDS,
                          KST, MODEL_CONFIG, RESULTS_DIR, ROOT)


METHODS = ('dense', 'bm25', 'hybrid')


def load_chunks():
    chunks = {}
    for path in CHUNK_SOURCES.values():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in chunks:
                raise ValueError(f"중복 청크 ID: {row['id']}")
            chunks[row['id']] = row
    return chunks


def validate_database(chunks):
    """비교할 corpus를 확인하며 빈 DB의 자동 생성/적재는 허용하지 않는다."""
    database = ROOT / CHROMA_PATH
    if not (database / 'chroma.sqlite3').is_file():
        raise ValueError(f'기존 Chroma DB가 필요합니다: {database}')
    collection = chromadb.PersistentClient(path=str(database)).get_collection(name=CHROMA_COLLECTION)
    data = collection.get(include=['documents', 'metadatas'])
    stored = {key: (text, meta or {}) for key, text, meta in zip(
        data['ids'], data['documents'], data['metadatas'])}
    if set(stored) != set(chunks):
        raise ValueError(f'DB/JSONL ID 불일치: DB에 없음 {len(set(chunks)-set(stored))}, '
                         f'JSONL에 없음 {len(set(stored)-set(chunks))}')
    for key, chunk in chunks.items():
        text, metadata = stored[key]
        if text != chunk['text'] or any(metadata.get(field) != chunk['metadata'].get(field)
                                        for field in FILTER_FIELDS):
            raise ValueError(f'DB/JSONL 본문 또는 필터 metadata 불일치: {key}')
    payload = [(key, text, metadata) for key, (text, metadata) in sorted(stored.items())]
    import hashlib
    return {'path': str(database), 'collection': CHROMA_COLLECTION, 'chunks': len(stored),
            'sha256': hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}


def metrics(documents, method, top_ks):
    ranks = [d[f'{method}_rank'] for d in documents if d[f'{method}_rank'] is not None]
    limit = max(top_ks)
    return {
        'best_gold_candidate_rank': min(ranks) if ranks else None,
        'reciprocal_rank_at_k': 1/min(ranks) if ranks and min(ranks) <= limit else 0.0,
        'top_k': {str(k): {'gold_hits': sum(d[f'in_{method}_top_{k}'] for d in documents),
                           'hit': any(d[f'in_{method}_top_{k}'] for d in documents),
                           'recall': sum(d[f'in_{method}_top_{k}'] for d in documents)/len(documents)}
                  for k in top_ks},
    }


def evaluate_case(case, record, chunks, search, top_ks, airline_scope=False, metadata_filter=True):
    conditions, filters = record['parsed']['conditions'], record.get('filters') or {}
    # 메타데이터 필터 없음(POC 설정): 장소·항공사로 후보를 거르지 않는다. 결과의 conditions는 실제 조건을 유지한다.
    scope_conditions = conditions if metadata_filter else {}
    where = retrieval_where(filters, scope_conditions, airline_scope=airline_scope)
    items = []
    searches = {}
    for index, (item, match) in enumerate(zip(case['answer']['items'],
                                             match_groups(case['answer']['items'], record['groups']))):
        ids = gold_ids(item)
        entry = {'item_index': index, 'name': item['name'], 'gold_chunk_ids': ids,
                 'query': record['groups'][match]['query'] if match is not None else None,
                 'status': 'scored', 'documents': []}
        if ids is None or not ids:
            entry['status'] = 'unannotated_sources' if ids is None else 'no_positive_sources'
        elif match is None:
            entry['status'] = 'unmatched_or_ambiguous_item'
        else:
            query = entry['query']
            if query not in searches:
                searches[query] = search(query, filters, scope_conditions)
            result = searches[query]
            if result['where'] != where:
                raise ValueError('검색과 평가의 필터가 다릅니다.')
            entry['candidate_count'] = result['candidate_count']
            traces = {trace['chunk_id']: trace for trace in result['traces']}
            for key in ids:
                trace = traces.get(key, {})
                eligible = matches_where(chunks[key]['metadata'], where)
                row = {'chunk_id': key, 'eligible_by_filters': eligible,
                       'bm25_score': trace.get('bm25_score'), 'rrf_score': trace.get('rrf_score')}
                for method in METHODS:
                    rank = trace.get(f'{method}_rank')
                    if rank is not None and not eligible:
                        raise ValueError(f'필터 밖 문서가 검색되었습니다: {key}')
                    row[f'{method}_rank'] = rank
                    row.update({f'in_{method}_top_{k}': rank is not None and rank <= k for k in top_ks})
                row['in_candidate_union'] = key in traces
                entry['documents'].append(row)
            entry['metrics'] = {method: metrics(entry['documents'], method, top_ks) for method in METHODS}
        items.append(entry)
    return {'id': case['id'], 'question': case['question'], 'conditions': conditions,
            'filters': filters, 'where': where, 'items': items,
            'searches': [{'query': query, **result} for query, result in searches.items()]}


def summarize(results, top_ks):
    items = [item for result in results for item in result['items']]
    scored = [i for i in items if i['status'] == 'scored']
    pairs = [d for item in scored for d in item['documents']]
    mean = lambda values: sum(values)/len(values) if values else None
    summary = {'questions': len(results), 'evaluated_items': len(scored), 'gold_pairs': len(pairs),
               'item_statuses': dict(Counter(i['status'] for i in items)),
               'filter_excluded_pairs': sum(not d['eligible_by_filters'] for d in pairs),
               'candidate_union_gold_hits': sum(d['in_candidate_union'] for d in pairs),
               'methods': {}, 'changes': {}}
    for method in METHODS:
        summary['methods'][method] = {
            'mrr_at_k': mean([i['metrics'][method]['reciprocal_rank_at_k'] for i in scored]),
            'top_k': {str(k): {
                'item_hits': sum(i['metrics'][method]['top_k'][str(k)]['hit'] for i in scored),
                'hit_rate': mean([i['metrics'][method]['top_k'][str(k)]['hit'] for i in scored]),
                'macro_recall': mean([i['metrics'][method]['top_k'][str(k)]['recall'] for i in scored]),
                'gold_pair_hits': sum(d[f'in_{method}_top_{k}'] for d in pairs),
                'micro_recall': mean([d[f'in_{method}_top_{k}'] for d in pairs]),
            } for k in top_ks},
        }
    for k in top_ks:
        dense_key, hybrid_key = f'in_dense_top_{k}', f'in_hybrid_top_{k}'
        summary['changes'][str(k)] = {
            'gained_gold_pairs': sum(d[hybrid_key] and not d[dense_key] for d in pairs),
            'lost_gold_pairs': sum(d[dense_key] and not d[hybrid_key] for d in pairs),
            'recovered_items': sum(i['metrics']['hybrid']['top_k'][str(k)]['hit'] and
                                   not i['metrics']['dense']['top_k'][str(k)]['hit'] for i in scored),
            'regressed_items': sum(i['metrics']['dense']['top_k'][str(k)]['hit'] and
                                   not i['metrics']['hybrid']['top_k'][str(k)]['hit'] for i in scored),
        }
    return summary


def write_reports(output, results, summary, top_ks):
    pairs = [dict(question_id=r['id'], name=i['name'], query=i['query'], **d)
             for r in results for i in r['items'] for d in i['documents']]
    fields = ['question_id', 'name', 'query', 'chunk_id', 'eligible_by_filters', 'bm25_score', 'rrf_score',
              'in_candidate_union', *[f'{m}_rank' for m in METHODS],
              *[f'in_{m}_top_{k}' for m in METHODS for k in top_ks]]
    with (output/'pairs.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    percent = lambda value: f'{value:.2%}' if value is not None else 'N/A'
    md = ['# Dense / BM25 / Hybrid 비교', '',
          '동일한 records 쿼리·DB·후보 필터·정답 매칭으로 비교. 쿼리 확장·LLM·reranker 없음.',
          'Dense는 Chroma 검색 순위이며 기존 JSONL의 정확한 코사인 순위와 구분한다.',
          '순위 없음은 해당 검색 후보 목록 밖 또는 필터 제외다. RRF 점수는 코사인이 아니다.',
          f"MRR은 Top-{max(top_ks)} 안의 첫 정답만 반영한다. 필터 밖 정답도 Recall 분모에 포함한다.", '',
          f"평가 물품: {summary['evaluated_items']}, 정답 청크 쌍: {summary['gold_pairs']}", '',
          '| 방식 | K | 적중 물품 | Hit rate | Macro Recall | 정답 쌍 적중 | Micro Recall |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        for k in top_ks:
            s = summary['methods'][method]['top_k'][str(k)]
            md.append(f"| {method} | {k} | {s['item_hits']} | {percent(s['hit_rate'])} | "
                      f"{percent(s['macro_recall'])} | {s['gold_pair_hits']} | {percent(s['micro_recall'])} |")
    md += ['', '| K | 추가 회수 쌍 | 누락된 쌍 | 복구 물품 | 회귀 물품 |', '| ---: | ---: | ---: | ---: | ---: |']
    for k, c in summary['changes'].items():
        md.append(f"| {k} | {c['gained_gold_pairs']} | {c['lost_gold_pairs']} | "
                  f"{c['recovered_items']} | {c['regressed_items']} |")
    md += ['', '| 질문 | 물품 | 정답 청크 | Dense 순위 | BM25 순위 | Hybrid 순위 |',
           '| --- | --- | --- | ---: | ---: | ---: |']
    for p in pairs:
        cells = [p['question_id'], p['name'], p['chunk_id'], *[p[f'{m}_rank'] for m in METHODS]]
        md.append('| ' + ' | '.join(str(v).replace('|', '&#124;').replace('\n', ' ') for v in cells) + ' |')
    md += ['', '평가 상태: ' + json.dumps(summary['item_statuses'], ensure_ascii=False), '']
    (output/'report.md').write_text('\n'.join(md), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/sample/questions.json')
    parser.add_argument('--records', type=Path, default=ROOT/'outputs/runs/attempt_1/records')
    parser.add_argument('--id', action='append', dest='ids')
    parser.add_argument('--top-k', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--candidate-k', type=int, default=50)
    parser.add_argument('--weights', type=float, nargs=2, default=[0.5, 0.5], metavar=('BM25', 'DENSE'))
    parser.add_argument('--rrf-c', type=int, default=60)
    parser.add_argument('--airline-filter', action='store_true',
                        help='국가 필터에 더해 항공사 자료는 질문의 항공사 것만 남긴다(평가 실험용, 서비스 경로 미연결)')
    parser.add_argument('--no-metadata-filter', action='store_true',
                        help='국가·항공사 메타데이터 필터를 모두 끈다(필터 없는 POC 설정 평가용). 쿼리는 그대로다')
    parser.add_argument('--query-mode', choices=['full', 'item', 'item_airline'], default='full',
                        help='full: 서비스 쿼리(물품명+details+장소·항공사·날짜·수하물). item: 물품명+details만. item_airline: 물품명+details+출발·귀국 항공사(평가 실험용). 필터는 모든 경우 여행 조건에서 만든다')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if any(k < 1 for k in args.top_k) or args.candidate_k < max(args.top_k):
        parser.error('top-k는 양수이며 candidate-k 이하여야 합니다.')
    if args.no_metadata_filter and args.airline_filter:
        parser.error('--no-metadata-filter와 --airline-filter는 함께 쓸 수 없습니다.')
    if args.rrf_c < 0 or any(not math.isfinite(w) or w <= 0 for w in args.weights):
        parser.error('rrf-c는 음수가 아니어야 하고 weights는 유한한 양수여야 합니다.')
    top_ks = sorted(set(args.top_k))
    cases = [c for group in read_json(args.dataset).values() for c in group]
    if len({c['id'] for c in cases}) != len(cases):
        parser.error('중복 질문 ID')
    if args.ids:
        unknown = set(args.ids)-{c['id'] for c in cases}
        if unknown:
            parser.error(f'없는 질문 ID: {sorted(unknown)}')
        cases = [c for c in cases if c['id'] in args.ids]
    records, record_paths = {}, {}
    for path in args.records.glob('*.json'):
        record = read_json(path)
        if record['id'] in records:
            raise ValueError(f"중복 실행 ID: {record['id']}")
        records[record['id']], record_paths[record['id']] = record, path
    if args.query_mode != 'full':
        for record in records.values():
            for group in record['groups']:
                group['query'] = experiment_query(group['item'], record['parsed']['conditions'], args.query_mode)
    chunks = load_chunks()
    for case in cases:
        validate_record(case, records.get(case['id']), args.query_mode)
        missing = {key for item in case['answer']['items'] for key in gold_ids(item) or [] if key not in chunks}
        if missing:
            raise ValueError(f"{case['id']}: 없는 정답 청크 {sorted(missing)}")
    # 설치 여부를 DB 접근/모델 로딩 전에 확인한다.
    packages = {name: version(name) for name in [
        'chromadb', 'sentence-transformers', 'langchain-classic', 'langchain-community', 'rank-bm25', 'kiwipiepy']}
    database = validate_database(chunks)
    output = args.output or RESULTS_DIR/f'hybrid_search_{datetime.now(KST):%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    cache = {}

    def search(query, filters, conditions):
        key = (query, json.dumps(filters, sort_keys=True), json.dumps(conditions, sort_keys=True))
        if key not in cache:
            cache[key] = compare_retrieval(query, filters, max(top_ks), conditions=conditions,
                                           candidate_k=args.candidate_k, weights=args.weights, rrf_c=args.rrf_c,
                                           airline_scope=args.airline_filter)
        return cache[key]

    results = []
    started = perf_counter()
    for case in cases:
        result = evaluate_case(case, records[case['id']], chunks, search, top_ks, args.airline_filter,
                               not args.no_metadata_filter)
        results.append(result)
        (output/f"{case['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{case['id']}: completed ({len(results)}/{len(cases)})", flush=True)
    summary = summarize(results, top_ks)
    summary.update({
        'created_at': datetime.now(KST).isoformat(), 'dataset': str(args.dataset.resolve()),
        'dataset_sha256': sha(args.dataset), 'chunk_files_sha256': {str(p): sha(p) for p in CHUNK_SOURCES.values()},
        'record_files_sha256': {c['id']: sha(record_paths[c['id']]) for c in cases},
        'records': str(args.records.resolve()), 'database': database, 'embedding': MODEL_CONFIG['embedding'],
        'versions': packages, 'query_mode': 'stored_groups', 'airline_filter': args.airline_filter, 'metadata_filter': not args.no_metadata_filter, 'query_variant': args.query_mode,
        'candidate_k': args.candidate_k,
        'top_ks': top_ks, 'mrr_k': max(top_ks), 'rrf_c': args.rrf_c,
        'weights': {'bm25': args.weights[0]/sum(args.weights), 'dense': args.weights[1]/sum(args.weights)},
        'elapsed_seconds': perf_counter()-started, 'unique_searches': len(cache),
    })
    # DB가 실행 도중 바뀌면 최종 비교 보고서를 확정하지 않는다.
    if validate_database(chunks) != database:
        raise ValueError('평가 중 DB가 변경되었습니다.')
    write_reports(output, results, summary, top_ks)
    print(f'Saved: {output}', flush=True)


if __name__ == '__main__':
    main()
