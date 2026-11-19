"""실제 물품별 검색 쿼리와 정답 청크의 코사인 유사도 진단.

python -m app.query_gold_similarity --id 1-1
python -m app.query_gold_similarity

기존 attempt_1/records의 groups[].query를 그대로 사용한다. LLM 호출 없음.
retrieve와 동일한 metadata 조건으로 전체 후보를 코사인 유사도로 정렬한다.
동점은 chunk_id 오름차순. 현재 JSONL을 사용하며 Chroma는 변경하지 않는다.
--hybrid-results로 evaluate_hybrid_search 결과를 연결하여 실제 검색 순위를 나란히 본다.
--baseline-run은 입력 해시가 일치하는 기존 코사인 결과를 재사용한다(모델 재계산 없음).
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

from app.integrations.embedding import create_embedding_function
from app.rag import build_retrieval_query
from app.retrieval import countries_from_places, retrieval_where
from app.settings import CHUNK_SOURCES, KST, MODEL_CONFIG, RESULTS_DIR, ROOT


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(name):
    return ' '.join(unicodedata.normalize('NFC', name).casefold().split())


def cosine(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('임베딩 차원 또는 값이 올바르지 않습니다.')
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if not denominator:
        raise ValueError('영벡터의 코사인 유사도는 정의되지 않습니다.')
    return float(np.clip(np.dot(left, right) / denominator, -1, 1))


def gold_ids(item):
    sources = item.get('source', item.get('sources'))
    if sources is None:
        return None
    if not isinstance(sources, list):
        raise ValueError('source는 배열 또는 null이어야 합니다.')
    ids = [s if isinstance(s, str) else s.get('chunk_id') if isinstance(s, dict) else None for s in sources]
    if any(not isinstance(key, str) or not key.strip() for key in ids):
        raise ValueError('정답 source에 청크 ID가 필요합니다.')
    return list(dict.fromkeys(ids))


def match_groups(items, groups):
    """이름/명시적 aliases가 일대일인 경우만 연결하며 순서로 추측하지 않는다."""
    candidates = [{n for n, group in enumerate(groups)
                   if normalized(group['item']['name']) in
                   {normalized(name) for name in [item['name'], *item.get('aliases', [])]}}
                  for item in items]
    usage = Counter(n for matches in candidates for n in matches)
    return [next(iter(matches)) if len(matches) == 1 and usage[next(iter(matches))] == 1 else None
            for matches in candidates]


def matches_where(metadata, where):
    """retrieval_where가 생성하는 $and/$eq/$in 조건을 적용한다."""
    if where is None:
        return True
    if '$and' in where:
        return all(matches_where(metadata, clause) for clause in where['$and'])
    if '$or' in where:
        return any(matches_where(metadata, clause) for clause in where['$or'])
    for field, operators in where.items():
        for operator, value in operators.items():
            if operator not in ('$eq', '$ne', '$in'):
                raise ValueError(f'지원하지 않는 필터 연산자: {operator}')
            if field not in metadata:
                return False
            if operator == '$eq' and metadata[field] != value:
                return False
            if operator == '$ne' and metadata[field] == value:
                return False
            if operator == '$in' and metadata[field] not in value:
                return False
    return True


def experiment_query(item, conditions, mode='full'):
    """평가 실험용 검색 쿼리. 서비스의 build_retrieval_query는 바꾸지 않고 조건만 골라 넘긴다.

    full: 서비스 쿼리, item: 물품명+details, item_airline: 물품명+details+출발·귀국 항공사.
    """
    if mode == 'full':
        return build_retrieval_query(item, conditions)
    if mode == 'item':
        return build_retrieval_query(item, None)
    if mode == 'item_airline':
        airlines = {leg: {'airline': (conditions.get(leg) or {}).get('airline')} for leg in ('departure', 'arrival')}
        return build_retrieval_query(item, airlines)
    raise ValueError(f'지원하지 않는 쿼리 방식: {mode}')


def validate_record(case, record, query_mode='full'):
    """query_mode는 experiment_query와 같다. 저장된 쿼리가 그 방식으로 만든 것과 같아야 한다."""
    if not record or record.get('question') != case['question'] or record.get('status') != 'success':
        raise ValueError(f"{case['id']}: 질문이 일치하는 성공 실행 기록이 필요합니다.")
    conditions = record['parsed']['conditions']
    for group in record['groups']:
        expected = experiment_query(group['item'], conditions, query_mode)
        if group['query'] != expected:
            raise ValueError(f"{case['id']}: 저장 쿼리가 현재 build_retrieval_query와 다릅니다.")
    retrieval_where(record.get('filters'), conditions)


def analyze_case(case, groups, conditions, chunks, embed, filters=None, top_ks=(1, 3, 5)):
    results = []
    matches = match_groups(case['answer']['items'], groups)
    countries = countries_from_places(conditions)
    where = retrieval_where(filters, conditions)
    candidates = {key: chunk for key, chunk in chunks.items()
                  if matches_where(chunk['metadata'], where)}
    rankings = {}
    for index, (item, match) in enumerate(zip(case['answer']['items'], matches)):
        ids = gold_ids(item)
        entry = {'item_index': index, 'name': item['name'], 'gold_chunk_ids': ids,
                 'query': groups[match]['query'] if match is not None else None,
                 'candidate_count': len(candidates), 'status': 'scored', 'documents': []}
        if ids is None or not ids:
            entry['status'] = 'unannotated_sources' if ids is None else 'no_positive_sources'
        elif match is None:
            entry['status'] = 'unmatched_or_ambiguous_item'
        else:
            query = entry['query']
            if query not in rankings:
                vector = embed(query)
                ordered = sorted(((key, cosine(vector, embed(chunk['text'])))
                                  for key, chunk in candidates.items()), key=lambda pair: (-pair[1], pair[0]))
                rankings[query] = ({key: (rank, score) for rank, (key, score) in enumerate(ordered, 1)}, ordered)
            ranked, ordered = rankings[query]
            entry['top_candidates'] = [{'chunk_id': key, 'cosine_similarity': score, 'rank': rank,
                                        'is_gold': key in ids}
                                       for rank, (key, score) in enumerate(ordered[:max(top_ks)], 1)]
            for key in ids:
                chunk = chunks.get(key)
                eligible = key in candidates
                rank, score = ranked.get(key, (None, None))
                if chunk is not None and not eligible:
                    score = cosine(embed(query), embed(chunk['text']))
                entry['documents'].append({
                    'chunk_id': key, 'status': 'missing_chunk' if chunk is None else 'scored',
                    'cosine_similarity': score, 'rank': rank,
                    'country': chunk['metadata'].get('country') if chunk else None,
                    'eligible_by_country_filter': chunk is not None and (
                        countries is None or chunk['metadata'].get('country') in countries),
                    'eligible_by_filters': eligible,
                    **{f'in_top_{k}': rank is not None and rank <= k for k in top_ks},
                })
            scores = [doc['cosine_similarity'] for doc in entry['documents'] if doc['status'] == 'scored']
            entry['mean_cosine_similarity'] = float(np.mean(scores)) if scores else None
            ranks = [doc['rank'] for doc in entry['documents'] if doc['rank'] is not None]
            entry['best_gold_rank'] = min(ranks) if ranks else None
            entry['reciprocal_rank'] = 1 / min(ranks) if ranks else 0.0
            entry['top_k'] = {str(k): {
                'gold_hits': sum(doc[f'in_top_{k}'] for doc in entry['documents']),
                'hit': any(doc[f'in_top_{k}'] for doc in entry['documents']),
                'recall': sum(doc[f'in_top_{k}'] for doc in entry['documents']) / len(ids),
            } for k in top_ks}
            if len(scores) != len(ids):
                entry['status'] = 'missing_chunk' if not scores else 'partially_scored'
        results.append(entry)
    return {'id': case['id'], 'question': case['question'], 'conditions': conditions,
            'country_filter': countries, 'filters': filters or {}, 'where': where,
            'groups': groups, 'items': results}


def validate_saved_inputs(folder, dataset, record_paths, cases):
    summary = read_json(folder / 'summary.json')
    expected = {
        'dataset_sha256': sha(dataset),
        'chunk_files_sha256': {str(p): sha(p) for p in CHUNK_SOURCES.values()},
        'embedding': MODEL_CONFIG['embedding'],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f'{folder}: 현재 입력과 {key}가 다릅니다.')
    for case in cases:
        if summary.get('record_files_sha256', {}).get(case['id']) != sha(record_paths[case['id']]):
            raise ValueError(f"{folder}: {case['id']} 실행 기록이 다릅니다.")
    return summary


def validate_saved_case(result, case, record, top_ks):
    if (result['id'] != case['id'] or result['question'] != case['question']
            or result['conditions'] != record['parsed']['conditions']
            or result.get('filters', {}) != (record.get('filters') or {})
            or len(result['items']) != len(case['answer']['items'])):
        raise ValueError(f"{case['id']}: 저장된 코사인 결과의 질문/조건/물품이 다릅니다.")
    matches = match_groups(case['answer']['items'], record['groups'])
    for item, gold, match in zip(result['items'], case['answer']['items'], matches):
        query = record['groups'][match]['query'] if match is not None else None
        if item['name'] != gold['name'] or item['gold_chunk_ids'] != gold_ids(gold) or item['query'] != query:
            raise ValueError(f"{case['id']}: 저장된 코사인 결과의 정답/쿼리가 다릅니다.")
        for doc in item['documents']:
            for k in top_ks:
                if f'in_top_{k}' not in doc:
                    raise ValueError(f"{case['id']}: baseline에 Top-{k}가 없습니다.")


def attach_hybrid_result(result, comparison, top_ks):
    if any(result[key] != comparison[key] for key in ('id', 'question', 'conditions', 'filters', 'where')):
        raise ValueError('코사인 분석과 hybrid 평가의 질문/검색 조건이 다릅니다.')
    if len(result['items']) != len(comparison['items']):
        raise ValueError('코사인 분석과 hybrid 평가의 물품 수가 다릅니다.')
    for item, actual in zip(result['items'], comparison['items']):
        for key in ('item_index', 'name', 'gold_chunk_ids', 'query', 'status'):
            if item[key] != actual[key]:
                raise ValueError(f'코사인 분석과 hybrid 평가의 {key}가 다릅니다.')
        if item['status'] != 'scored':
            continue
        if item['candidate_count'] != actual['candidate_count']:
            raise ValueError('코사인 분석과 실제 검색의 후보 수가 다릅니다.')
        by_id = {doc['chunk_id']: doc for doc in actual['documents']}
        item['retrieval_metrics'] = actual['metrics']
        for doc in item['documents']:
            retrieved = by_id[doc['chunk_id']]
            if doc['eligible_by_filters'] != retrieved['eligible_by_filters']:
                raise ValueError('코사인 분석과 실제 검색의 후보 필터가 다릅니다.')
            for method in ('dense', 'bm25', 'hybrid'):
                doc[f'{method}_rank'] = retrieved[f'{method}_rank']
                for k in top_ks:
                    doc[f'in_{method}_top_{k}'] = retrieved[f'in_{method}_top_{k}']
            doc['bm25_score'] = retrieved['bm25_score']
            doc['rrf_score'] = retrieved['rrf_score']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/sample/questions.json')
    parser.add_argument('--id', action='append', dest='ids', help='질문 ID. 반복 지정 가능')
    parser.add_argument('--records', type=Path, default=ROOT / 'outputs/runs/attempt_1/records',
                        help='기존 실행 records/ 폴더. 저장된 실제 쿼리를 사용')
    parser.add_argument('--output', type=Path, help='새 결과 폴더. 기존 폴더 덮어쓰기 금지')
    parser.add_argument('--top-k', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--hybrid-results', type=Path, help='evaluate_hybrid_search 결과 폴더')
    parser.add_argument('--baseline-run', type=Path, help='해시가 일치하는 기존 코사인 결과 폴더')
    args = parser.parse_args()
    if any(k < 1 for k in args.top_k):
        parser.error('--top-k는 1 이상이어야 합니다.')
    top_ks = sorted(set(args.top_k))
    dataset = read_json(args.dataset)
    cases = [case for questions in dataset.values() for case in questions]
    if len({case['id'] for case in cases}) != len(cases):
        parser.error('질문 ID가 중복됩니다.')
    if args.ids:
        unknown = set(args.ids) - {case['id'] for case in cases}
        if unknown:
            parser.error(f'없는 질문 ID: {sorted(unknown)}')
        cases = [case for case in cases if case['id'] in args.ids]
    chunks = {}
    for path in CHUNK_SOURCES.values():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row = json.loads(line)
                if row['id'] in chunks:
                    raise ValueError(f"중복 청크 ID: {row['id']}")
                chunks[row['id']] = row
    # Validate references before any API call or model loading.
    missing = {key for case in cases for item in case['answer']['items']
               for key in gold_ids(item) or [] if key not in chunks}
    if missing:
        raise ValueError(f'현재 청크에 없는 정답 ID: {sorted(missing)}')
    records = {}
    record_paths = {}
    for path in args.records.glob('*.json'):
        record = read_json(path)
        if record.get('id') in records:
            raise ValueError(f"중복 실행 ID: {record['id']}")
        records[record.get('id')] = record
        record_paths[record.get('id')] = path
    for case in cases:
        validate_record(case, records.get(case['id']))
    if args.baseline_run:
        validate_saved_inputs(args.baseline_run, args.dataset, record_paths, cases)
        for case in cases:
            validate_saved_case(read_json(args.baseline_run / f"{case['id']}.json"),
                                case, records[case['id']], top_ks)
    hybrid_summary = None
    if args.hybrid_results:
        hybrid_summary = validate_saved_inputs(args.hybrid_results, args.dataset, record_paths, cases)
        if not set(top_ks).issubset(hybrid_summary['top_ks']):
            raise ValueError('hybrid 평가 결과에 요청한 Top-K가 없습니다.')
    output = args.output or RESULTS_DIR / f'query_gold_similarity_{datetime.now(KST):%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    function = None
    cache = {}

    def embed(text):
        nonlocal function
        if text not in cache:
            if function is None:
                function = create_embedding_function()
            cache[text] = function([text])[0]
        return cache[text]

    results = []
    for case in cases:
        record = records[case['id']]
        conditions = record['parsed']['conditions']
        groups = record['groups']
        if args.baseline_run:
            result = read_json(args.baseline_run / f"{case['id']}.json")
        else:
            result = analyze_case(case, groups, conditions, chunks, embed, record.get('filters'), top_ks)
        if args.hybrid_results:
            attach_hybrid_result(result, read_json(args.hybrid_results / f"{case['id']}.json"), top_ks)
        result['record_k'] = record['k']
        results.append(result)
        (output / f"{case['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{case['id']}: " + ', '.join(f"{i['name']}={i['status']}" for i in result['items']), flush=True)
    pairs = [dict(question_id=result['id'], name=item['name'], query=item['query'], **doc)
             for result in results for item in result['items'] for doc in item['documents']]
    with (output / 'pairs.csv').open('w', encoding='utf-8-sig', newline='') as file:
        fields = ['question_id', 'name', 'query', 'chunk_id', 'status', 'cosine_similarity',
                  'country', 'eligible_by_country_filter', 'eligible_by_filters', 'rank',
                  *[f'in_top_{k}' for k in top_ks]]
        if args.hybrid_results:
            fields += ['dense_rank', 'bm25_rank', 'hybrid_rank', 'bm25_score', 'rrf_score',
                       *[f'in_{m}_top_{k}' for m in ('dense', 'bm25', 'hybrid') for k in top_ks]]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(pairs)
    scores = [p['cosine_similarity'] for p in pairs if p['status'] == 'scored']
    evaluated = [item for result in results for item in result['items'] if item['status'] == 'scored']
    summary = {'dataset': str(args.dataset.resolve()), 'dataset_sha256': sha(args.dataset),
               'chunk_files_sha256': {str(p): sha(p) for p in CHUNK_SOURCES.values()},
               'embedding': MODEL_CONFIG['embedding'], 'query_mode': 'stored_groups',
               'questions': len(results), 'scored_pairs': len(scores),
               'item_statuses': dict(Counter(item['status'] for result in results for item in result['items'])),
               'pair_weighted_mean_cosine': float(np.mean(scores)) if scores else None,
               'records': str(args.records.resolve()),
               'record_files_sha256': {case['id']: sha(record_paths[case['id']]) for case in cases},
               'evaluated_items': len(evaluated),
               'excluded_gold_pairs': sum(not p['eligible_by_filters'] for p in pairs),
               'mean_reciprocal_rank': float(np.mean([i['reciprocal_rank'] for i in evaluated])) if evaluated else None,
               'top_k': {str(k): {
                   'hit_rate': float(np.mean([i['top_k'][str(k)]['hit'] for i in evaluated])) if evaluated else None,
                   'macro_recall': float(np.mean([i['top_k'][str(k)]['recall'] for i in evaluated])) if evaluated else None,
               } for k in top_ks},
               'note': 'Exact cosine ranking of current JSONL candidates using retrieve metadata filters. '
                       'Ties: chunk_id ascending. Not Chroma ANN rank. Filter-excluded gold has null rank, '
                       'false Top-K and counts as a miss in recall; source-null/unmatched items are excluded.'}
    if args.baseline_run:
        summary['baseline_run'] = str(args.baseline_run.resolve())
    if hybrid_summary is not None:
        # 재사용 폴더가 더 넓은 질문 집합일 수 있으므로 현재 선택 물품만 집계한다.
        from app.evaluate_hybrid_search import summarize
        selected_comparisons = [read_json(args.hybrid_results / f"{case['id']}.json") for case in cases]
        summary['retrieval_comparison'] = summarize(selected_comparisons, top_ks)
        summary['hybrid_results'] = str(args.hybrid_results.resolve())
        summary['note'] += ' Cosine is unchanged by fusion; dense_rank is Chroma, hybrid_rank is RRF.'
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# 정답 청크 의미 유사도 및 순위', '', summary['note'], '',
          f"평가 물품: {len(evaluated)}, MRR: {summary['mean_reciprocal_rank']}", '',
          '| K | Hit rate | Macro recall |', '| ---: | ---: | ---: |']
    for k, metrics in summary['top_k'].items():
        md.append(f"| {k} | {metrics['hit_rate']} | {metrics['macro_recall']} |")
    if hybrid_summary is not None:
        md += ['', '## 실제 검색 비교', '',
               '같은 코사인 점수를 유지하면서 BM25/RRF 결합으로 검색 순위가 어떻게 바뀌는지 비교한다.',
               f'MRR은 세 검색 방식 모두 Top-{max(top_ks)}에서 측정한다.', '',
               '| 방식 | K | Hit rate | Macro Recall | 정답 쌍 적중 |',
               '| --- | ---: | ---: | ---: | ---: |']
        for method, entry in summary['retrieval_comparison']['methods'].items():
            for k in top_ks:
                s = entry['top_k'][str(k)]
                hit = f"{s['hit_rate']:.2%}" if s['hit_rate'] is not None else 'N/A'
                recall = f"{s['macro_recall']:.2%}" if s['macro_recall'] is not None else 'N/A'
                md.append(f"| {method} | {k} | {hit} | {recall} | {s['gold_pair_hits']} |")
    md += ['', '| 질문 | 물품 | 후보 수 | 정답 청크 | 코사인 | 순위 | '
           + ' | '.join(f'Top-{k}' for k in top_ks) + ' |',
           '| --- | --- | ---: | --- | ---: | ---: | ' + ' | '.join('---' for _ in top_ks) + ' |']
    if hybrid_summary is not None:
        md[-2] += ' Chroma Dense 순위 | BM25 순위 | Hybrid 순위 |'
        md[-1] += ' ---: | ---: | ---: |'
    for result in results:
        for item in result['items']:
            for doc in item['documents']:
                cells = [result['id'], item['name'], item['candidate_count'], doc['chunk_id'],
                         doc['cosine_similarity'], doc['rank'], *[doc[f'in_top_{k}'] for k in top_ks]]
                if hybrid_summary is not None:
                    cells += [doc.get(f'{method}_rank') for method in ('dense', 'bm25', 'hybrid')]
                md.append('| ' + ' | '.join(str(c).replace('|', '\\|').replace('\n', ' ') for c in cells) + ' |')
    md += ['', '물품별 평가 상태: ' + json.dumps(summary['item_statuses'], ensure_ascii=False), '']
    (output / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'저장됨: {output}')


if __name__ == '__main__':
    main()
