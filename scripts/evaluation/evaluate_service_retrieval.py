# -*- coding: utf-8 -*-
"""현재 서비스 검색 경로(app.search.retrieve_for_item)의 정답 근거 검색 성능을 평가한다. LLM·OpenAI 호출 없음.

python -m scripts.evaluation.evaluate_service_retrieval run --dataset data/test/test9_questions.json \\
    --records data/test/test9_gold_query_records/test9 --output tests/ver9_test/retrieval/service_hybrid_rerank30_k8/test9
python -m scripts.evaluation.evaluate_service_retrieval report --out-dir tests/ver9_test/retrieval/service_hybrid_rerank30_k8 \\
    --runs <.../test9> <.../supplement>

run: 정답 구조화에서 만든 쿼리(records)로 서비스가 실제 호출하는 retrieve_for_item을 실행해 최종 문서 순위를 저장한다.
     설정(config/setting.json의 retrieval)과 default_k를 그대로 따르며 물품 단위 JSON·summary.json을 쓴다.
report: 저장된 결과만 읽어 present.md·metrics_by_k.csv·case_changes.csv를 만든다(검색 재실행 없음).
기존 출력 폴더·파일은 덮어쓰지 않는다(report는 --overwrite 필요).
"""
import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from app.evaluate_hybrid_search import load_chunks, validate_database
from app.query_gold_similarity import gold_ids, match_groups, matches_where, read_json, sha, validate_record
from app.retrieval import retrieval_where
from app.search import active_retrieval, retrieve_for_item
from app.settings import CHUNK_SOURCES, DEFAULT_K, KST, MODEL_CONFIG, ROOT

TYPE_NAMES = {'airline': '항공사', 'admin_rule': '행정규칙', 'airport': '공항', 'government': '정부 기관', 'press': '보도자료'}
BASELINES = [  # (표시 이름, 폴더, 순위 키) — 같은 정답 구조화 입력으로 평가한 기존 결과
    ('Dense, 국가 필터 (이전 서비스 방식, ver4)', 'tests/ver4_test/retrieval/dense_country_depth50', 'dense_rank'),
    ('Dense, 국가+항공사 필터 (ver6)', 'tests/ver6_test/retrieval/country_airline_depth50', 'dense_rank'),
    ('Hybrid 순서, 국가+항공사 필터 (ver6)', 'tests/ver6_test/retrieval/country_airline_depth50', 'hybrid_rank'),
]
OFFLINE_RERANK = 'tests/ver8_test/retrieval/rerank_full_n30/metrics.csv'


def pct(value):
    return f'{value * 100:.2f}%'


# ----------------------------------------------------------------------------- run
def evaluate_case(case, record, chunks, top_ks, k, search):
    conditions, filters = record['parsed']['conditions'], record.get('filters') or {}
    config = active_retrieval()
    where = retrieval_where(filters, conditions, airline_scope=config['method'] == 'hybrid' and config.get('airline_filter', False))
    items, searches = [], {}
    for index, (item, match) in enumerate(zip(case['answer']['items'], match_groups(case['answer']['items'], record['groups']))):
        ids = gold_ids(item)
        entry = {'item_index': index, 'name': item['name'], 'gold_chunk_ids': ids,
                 'query': record['groups'][match]['query'] if match is not None else None, 'status': 'scored', 'documents': []}
        if ids is None or not ids:
            entry['status'] = 'unannotated_sources' if ids is None else 'no_positive_sources'
        elif match is None:
            entry['status'] = 'unmatched_or_ambiguous_item'
        else:
            query = entry['query']
            if query not in searches:
                searches[query] = search(query, filters, k, conditions)
            result = searches[query]
            final = {row['id']: row for row in result['final']}
            pool = {chunk_id: n for n, chunk_id in enumerate(result['pool_ids'], 1)}
            for key in ids:
                row = final.get(key)
                entry['documents'].append({
                    'chunk_id': key, 'eligible_by_filters': matches_where(chunks[key]['metadata'], where),
                    'pool_rank': pool.get(key), 'final_rank': row['rank'] if row else None,
                    'rerank_score': row.get('rerank_score') if row else None,
                    **{f'in_final_top_{n}': bool(row) and row['rank'] <= n for n in top_ks}})
        items.append(entry)
    return {'id': case['id'], 'question': case['question'], 'conditions': conditions, 'filters': filters, 'where': where,
            'items': items, 'searches': [{'query': q, **r} for q, r in searches.items()]}


def run(args):
    top_ks = sorted(set(args.top_k))
    k = args.k or DEFAULT_K
    if max(top_ks) > k:
        raise SystemExit(f'--top-k는 최종 문서 수({k}) 이하여야 합니다.')
    cases = [c for group in read_json(args.dataset).values() for c in group]
    records = {r['id']: r for r in (read_json(p) for p in sorted(args.records.glob('*.json')))}
    chunks = load_chunks()
    for case in cases:
        validate_record(case, records.get(case['id']), 'full')
    database = validate_database(chunks)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    cache_timing = []

    def search(query, filters, final_k, conditions):
        trace = {}
        start = perf_counter()
        docs = retrieve_for_item(query, filters, final_k, conditions, trace=trace)
        elapsed = perf_counter() - start
        cache_timing.append(elapsed)
        return {'elapsed_seconds': elapsed, 'pool_ids': trace['pool_ids'],
                'final': [{'id': d['id'], 'rank': n, 'rerank_score': d.get('rerank_score'),
                           'source_type': d['metadata'].get('source_type'), 'org_name': d['metadata'].get('org_name')}
                          for n, d in enumerate(docs, 1)]}

    started, results = perf_counter(), []
    for case in cases:
        result = evaluate_case(case, records[case['id']], chunks, top_ks, k, search)
        results.append(result)
        (output / f"{case['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{case['id']}: completed ({len(results)}/{len(cases)})", flush=True)

    items = [i for r in results for i in r['items']]
    scored = [i for i in items if i['status'] == 'scored']
    pairs = [d for i in scored for d in i['documents']]
    summary = {
        'created_at': datetime.now(KST).isoformat(), 'dataset': str(args.dataset.resolve()), 'dataset_sha256': sha(args.dataset),
        'records': str(args.records.resolve()), 'input': args.input_kind, 'retrieval': active_retrieval(), 'k': k,
        'top_ks': top_ks, 'questions': len(results), 'evaluated_items': len(scored), 'gold_pairs': len(pairs),
        'item_statuses': dict(Counter(i['status'] for i in items)),
        'filter_excluded_pairs': sum(not d['eligible_by_filters'] for d in pairs),
        'pool_gold_pairs': sum(d['pool_rank'] is not None for d in pairs),
        'top_k': {str(n): {'gold_pair_hits': sum(d[f'in_final_top_{n}'] for d in pairs),
                           'macro_recall': statistics.mean(sum(d[f'in_final_top_{n}'] for d in i['documents']) / len(i['documents']) for i in scored),
                           'hit_items': sum(any(d[f'in_final_top_{n}'] for d in i['documents']) for i in scored)} for n in top_ks},
        'timing_seconds': {'searches': len(cache_timing), 'mean': statistics.mean(cache_timing), 'median': statistics.median(cache_timing),
                           'max': max(cache_timing), 'first': cache_timing[0], 'total': perf_counter() - started},
        'embedding': MODEL_CONFIG['embedding'], 'database': database,
        'chunk_files_sha256': {str(p): sha(p) for p in CHUNK_SOURCES.values()},
        'versions': {n: version(n) for n in ['chromadb', 'sentence-transformers', 'rank-bm25', 'kiwipiepy']}}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', output, flush=True)


# ----------------------------------------------------------------------------- report
def load_run(run_dirs):
    items, summaries = [], []
    for run in map(Path, run_dirs):
        summaries.append(json.loads((run / 'summary.json').read_text(encoding='utf-8')))
        for path in sorted(run.glob('*.json')):
            if path.name == 'summary.json':
                continue
            result = json.loads(path.read_text(encoding='utf-8'))
            for item in result['items']:
                if item['status'] == 'scored':
                    items.append({'set': run.name, 'qid': result['id'], 'name': item['name'], 'pairs': item['documents'],
                                  'conditions': result['conditions']})
    return items, summaries


def macro(items, rank_of, k):
    per_item = [sum(rank_of(p) is not None and rank_of(p) <= k for p in i['pairs']) / len(i['pairs']) for i in items]
    pair_hits = sum(rank_of(p) is not None and rank_of(p) <= k for i in items for p in i['pairs'])
    return statistics.mean(per_item), pair_hits


def baseline_values(folder, rank_key, ks):
    import glob
    items = []
    for dataset in ('test9', 'supplement'):
        for path in glob.glob(str(ROOT / folder / dataset / '*.json')):
            if path.endswith('summary.json'):
                continue
            for item in json.loads(Path(path).read_text(encoding='utf-8'))['items']:
                if item['status'] == 'scored':
                    items.append({'pairs': item['documents']})
    return {k: macro(items, lambda p: p.get(rank_key), k) for k in ks}


def offline_rerank_values(ks):
    path = ROOT / OFFLINE_RERANK
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open(encoding='utf-8-sig')):
        if row['pool_N'] == '30' and row['variant'] == 'rerank_full' and int(row['M']) in ks:
            out[int(row['M'])] = (float(row['macro_recall']), int(row['pair_hits']))
    return out


def report(args):
    items, summaries = load_run(args.runs)
    input_kinds = {s.get('input', 'gold_structure_queries') for s in summaries}
    if len(input_kinds) != 1:
        raise ValueError('서로 다른 종류의 입력을 하나의 검색 보고서로 합칠 수 없습니다.')
    gold_input = input_kinds == {'gold_structure_queries'}
    ks = summaries[0]['top_ks']
    final = lambda p: p['final_rank']
    pool = lambda p: p['pool_rank']
    targets = [args.out_dir / n for n in ('present.md', 'metrics_by_k.csv', 'case_changes.csv')]
    if any(t.exists() for t in targets) and not args.overwrite:
        raise SystemExit('이미 있습니다(--overwrite 필요): ' + ', '.join(str(t) for t in targets if t.exists()))
    pairs = [p for i in items for p in i['pairs']]
    cfg = summaries[0]['retrieval']
    now = {k: macro(items, final, k) for k in ks}
    hybrid_order = {k: macro(items, pool, k) for k in ks}
    meta = {}
    for path in CHUNK_SOURCES.values():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row = json.loads(line)
                meta[row['id']] = row['metadata']['source_type']

    with open(args.out_dir / 'metrics_by_k.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'K', 'macro_recall', 'micro_recall', 'pair_hits', 'evaluated_items', 'gold_pairs'])
        for name, table in (('service_final', now), ('hybrid_order_same_run', hybrid_order)):
            for k in ks:
                w.writerow([name, k, f'{table[k][0]:.6f}', f'{table[k][1] / len(pairs):.6f}', table[k][1], len(items), len(pairs)])
    changes = []
    for i in items:
        for k in (5, ks[-1]):
            a = {p['chunk_id'] for p in i['pairs'] if p['final_rank'] and p['final_rank'] <= k}
            b = {p['chunk_id'] for p in i['pairs'] if p['pool_rank'] and p['pool_rank'] <= k}
            if a != b:
                changes.append((k, i['set'], i['qid'], i['name'], len(i['pairs']), sorted(a - b), sorted(b - a)))
    with open(args.out_dir / 'case_changes.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['K', 'set', 'question_id', 'item', 'gold_pairs', 'gained_by_rerank', 'lost_by_rerank'])
        for row in changes:
            w.writerow([*row[:5], ' '.join(row[5]), ' '.join(row[6])])

    timing = [s['timing_seconds'] for s in summaries]
    searches = sum(t['searches'] for t in timing)
    mean_time = sum(t['mean'] * t['searches'] for t in timing) / searches
    in_pool = sum(p['pool_rank'] is not None for p in pairs)
    kmax = ks[-1]
    in_final = sum(p['final_rank'] is not None and p['final_rank'] <= kmax for p in pairs)
    n_items = len(items)
    reranking = ('`' + cfg['reranker']['model'] + '` ' + cfg['reranker']['dtype'] + f", 상위 {cfg['rerank_pool']}개 재정렬"
                 if cfg.get('rerank_pool') else '없음')
    medians = ', '.join(f"{t['median']:.2f}초" for t in timing)
    md = [f'# 검색 평가 — {args.title}', '', '## 실험 개요', '',
          '- **목적:** 현재 서비스 검색 경로(`app.search.retrieve_for_item`)를 실제로 호출해 정답 근거가 최종 문서에 얼마나 들어오는지 측정한다.',
          ('- **주의:** 정답 구조화 쿼리를 입력으로 사용한다. 실제 LLM 구조화 오류와 최종 판정 정확도는 포함하지 않는다.' if gold_input else
           '- **주의:** 실제 LLM 구조화 쿼리를 입력으로 사용한다. 이름이 연결되지 않은 물품은 이 진단에서 제외된다. 전체 물품의 누락을 0점으로 포함한 검색 지표는 evaluate_dataset 결과를 확인한다. 최종 판정 정확도는 측정하지 않는다.'),
          '', '| 항목 | 내용 |', '| --- | --- |',
          f"| 검색 방식 | `{cfg['method']}`, 방식별 후보 {cfg['candidate_k']}개, RRF 가중 BM25 0.5 : Dense 0.5, 후보 필터 국가{' + 항공사' if cfg.get('airline_filter') else ''} |",
          f"| reranking | {reranking} |",
          f"| 최종 문서 수 | 물품당 k={summaries[0]['k']} |",
          f"| 입력 쿼리 | {'정답 구조화' if gold_input else '실제 LLM 구조화'}에서 서비스와 같은 `build_retrieval_query`로 만든 전체 쿼리 |",
          f"| 물품 상태(제외 포함) | {dict(sum((Counter(s['item_statuses']) for s in summaries), Counter()))} |",
          f"| 평가 데이터 | {' + '.join('`' + Path(s['dataset']).name + '`' for s in summaries)} |",
          f"| 평가 대상 | {n_items}개 물품, 정답 청크 쌍 {len(pairs)}개 |",
          f"| 임베딩·DB | {summaries[0]['embedding']['model']} ({summaries[0]['embedding']['parameters'].get('device')}), Chroma `{summaries[0]['database']['collection']}` {summaries[0]['database']['chunks']}청크 |",
          '', '## 1. 종합 평가: 최종 문서 기준 Macro Recall', '',
          '| 지표 | ' + ' | '.join(f'@{k}' for k in ks) + ' |', '| --- | ' + ' | '.join('---:' for _ in ks) + ' |',
          '| Macro Recall (물품 평균) | ' + ' | '.join(pct(now[k][0]) for k in ks) + ' |',
          '| Micro Recall (정답 쌍 기준) | ' + ' | '.join(pct(now[k][1] / len(pairs)) for k in ks) + ' |',
          '| 정답 쌍 적중 | ' + ' | '.join(f'{now[k][1]}/{len(pairs)}' for k in ks) + ' |',
          '| Hit(1개 이상 회수한 물품) | ' + ' | '.join(
              f"{sum(any(p['final_rank'] and p['final_rank'] <= k for p in i['pairs']) for i in items)}/{n_items}" for k in ks) + ' |',
          '| 전체 회수 물품(정답을 모두 담음) | ' + ' | '.join(
              f"{sum(all(p['final_rank'] and p['final_rank'] <= k for p in i['pairs']) for i in items)}/{n_items}" for k in ks) + ' |',
          '', '## 2. 설정별 비교 (각 칸은 Macro / Micro)', '',
          '| 설정 | ' + ' | '.join(f'@{k}' for k in ks) + ' |', '| --- | ' + ' | '.join('---:' for _ in ks) + ' |']
    rows = [(name, baseline_values(folder, key, ks)) for name, folder, key in BASELINES] if gold_input else []
    rows.append(('Hybrid 순서, reranking 전 (이번 실행의 pool, 대조군)', hybrid_order))
    offline = offline_rerank_values(ks) if gold_input else {}
    if offline:
        rows.append(('오프라인 재정렬 N=30 (ver8, 저장된 점수)', {k: offline.get(k) for k in ks}))
    rows.append(('**현재 서비스 경로 (이번 실행)**', now))
    for name, table in rows:
        md.append(f'| {name} | ' + ' | '.join(
            f'{pct(table[k][0])} / {pct(table[k][1] / len(pairs))}' if table.get(k) else '-' for k in ks) + ' |')
    if offline:
        diffs = [f"@{k} {(now[k][0] - offline[k][0]) * 100:+.2f}%p" for k in ks if k in offline]
        md += ['', f"- **ver8 오프라인 평가와의 일치:** 서비스 경로의 최종 순위가 저장된 점수로 계산한 재정렬과 같아야 한다. 차이: {', '.join(diffs)}."]
    md += ['', '## 3. 손실 분해 (정답 청크 쌍이 어디서 빠지는가)', '',
           '| 구분 | 정답 쌍 |', '| --- | ---: |',
           f"| 필터에서 제외 | {sum(not p['eligible_by_filters'] for p in pairs)} |",
           f"| 필터 통과했지만 후보 pool({cfg.get('rerank_pool')}개) 밖 | {sum(p['eligible_by_filters'] and p['pool_rank'] is None for p in pairs)} |",
           f"| pool 안이지만 최종 {kmax}개 밖 | {in_pool - in_final} |",
           f"| 최종 {kmax}개 안(회수) | {in_final} |", '',
           '후보 pool 밖의 정답은 reranking으로 회수되지 않는다(검색 방식·필터·쿼리의 몫). pool 안이지만 최종 밖은 최종 개수나 reranker 성능의 몫이다.', '',
           f'## 4. 자료 유형별 정답 쌍 (최종 {kmax}개 vs reranking 전 Hybrid 순서 상위 {kmax}개)', '',
           '| 자료 유형 | 정답 쌍 | Hybrid 순서 회수 | 서비스 경로 회수 | 순변화 |', '| --- | ---: | ---: | ---: | ---: |']
    stats = {}
    for p in pairs:
        t = TYPE_NAMES.get(meta[p['chunk_id']], meta[p['chunk_id']])
        row = stats.setdefault(t, [0, 0, 0])
        row[0] += 1
        row[1] += p['pool_rank'] is not None and p['pool_rank'] <= kmax
        row[2] += p['final_rank'] is not None and p['final_rank'] <= kmax
    for t, (total, before, after) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        md.append(f'| {t} | {total} | {before} | {after} | {after - before:+d} |')
    md += ['', f'## 5. 시간 (검색 1건당 `retrieve_for_item`, {summaries[0]["embedding"]["parameters"].get("device")}, LLM 제외)', '',
           '| 항목 | 값 |', '| --- | ---: |', f'| 검색 횟수(물품 단위) | {searches} |',
           f'| 평균 | {mean_time:.2f}초 |', f"| 중앙값(데이터셋별) | {medians} |",
           f"| 최대 | {max(t['max'] for t in timing):.2f}초 |", f"| 첫 호출 | {timing[0]['first']:.2f}초 |",
           '', '평균에는 BM25 인덱스를 필터 조합별로 처음 만드는 시간(cold)이 섞여 있다. 같은 프로세스에서 반복되는 필터 조합은 더 빠르다(ver8 시간 측정 참고). 다른 GPU 작업과 겹치면 느려진다.',
           '', '## 핵심 지표 (장표용)', '']
    md += [f"- **최종 {k}개 기준 Macro Recall {pct(now[k][0])}, Micro Recall {pct(now[k][1] / len(pairs))}** (정답 쌍 {now[k][1]}/{len(pairs)})" for k in ks if k in (5, kmax)]
    md += [f"- 후보 pool({cfg.get('rerank_pool')}개)에 담긴 정답 쌍 {in_pool}/{len(pairs)} ({pct(in_pool / len(pairs))}) — 이 값이 reranking 후 Recall의 천장이다.",
           f"- 검색 1건 평균 {mean_time:.2f}초(물품 2개 질문이면 순차 처리 시 약 2배).",
           '', '## 지표 설명', '',
           '- **Macro Recall@K:** 물품마다 (정답 청크 중 최종 상위 K개에 든 수 ÷ 정답 청크 수)를 구해 평균한다. 정답이 많은 물품과 적은 물품을 같은 비중으로 본다.',
           '- **Micro Recall@K:** 모든 (물품, 정답 청크) 쌍 중 최종 상위 K개에 든 쌍의 비율이다(정답 쌍 적중 ÷ 전체 정답 쌍). 정답이 많은 물품의 비중이 더 크다.',
           '- **정답 쌍 적중:** 모든 (물품, 정답 청크) 쌍 중 최종 상위 K개에 든 수다.',
           '- 정답 밖 청크가 검색되어도 오답으로 세지 않는다(미검수 문서일 수 있다). 그래서 Precision은 쓰지 않는다.', '']
    (args.out_dir / 'present.md').write_text('\n'.join(md), encoding='utf-8')
    print('Saved:', args.out_dir / 'present.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--dataset', type=Path, required=True)
    r.add_argument('--records', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--input-kind', choices=['gold_structure_queries', 'model_structure_queries'],
                   default='gold_structure_queries', help='입력 출처를 명시한다. 실제 LLM records에는 model_structure_queries를 지정')
    r.add_argument('--k', type=int, help='최종 문서 수. 기본은 config의 default_k')
    r.add_argument('--top-k', type=int, nargs='+', default=[1, 3, 5, 8])
    p = sub.add_parser('report')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--title', default='서비스 경로(Hybrid → 국가+항공사 필터 → reranking N=30 → 최종 k=8)')
    p.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    run(args) if args.command == 'run' else report(args)


if __name__ == '__main__':
    main()
