# -*- coding: utf-8 -*-
"""검색 전용 평가(app.evaluate_hybrid_search 결과)에서 장표용 present.md와 CSV를 만든다.

python -m scripts.evaluation.make_retrieval_present --out-dir tests/ver4_test/retrieval/dense_country_depth50 \\
    --runs tests/ver4_test/retrieval/dense_country_depth50/test9 tests/ver4_test/retrieval/dense_country_depth50/supplement \\
    --title "검색 깊이 곡선" --purpose "..." --methods dense hybrid [--baseline-runs 폴더... --baseline-label 라벨]

LLM 호출·검색 재실행은 없다. 저장된 질문별 JSON(정답 청크별 순위)만 읽는다. 여러 --runs(예: test9와 supplement)는
물품 단위로 합쳐 하나의 표로 계산한다. 같은 파일이 있으면 --overwrite 없이는 덮어쓰지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path

METHOD_NAMES = {'dense': 'Dense', 'bm25': 'BM25', 'hybrid': 'Hybrid(RRF)'}


def load_runs(run_dirs):
    items, summaries = [], []
    for run in run_dirs:
        run = Path(run)
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        summaries.append(summary)
        for path in sorted(run.glob('*.json')):
            if path.name == 'summary.json' or path.name.startswith('db_'):
                continue
            result = json.loads(path.read_text(encoding='utf-8'))
            for item in result['items']:
                items.append({'set': run.name, 'qid': result['id'], 'name': item['name'], 'status': item['status'],
                              'pairs': item.get('documents', [])})
    return items, summaries


def pct(value):
    return f'{value * 100:.2f}%'


def delta(now, before, kind='pp'):
    if before is None:
        return '-'
    diff = now - before
    return f'{diff * 100:+.2f}%p' if kind == 'pp' else f'{diff:+d}'


def metrics(items, method, ks):
    scored = [i for i in items if i['status'] == 'scored']
    pairs = [p for i in scored for p in i['pairs']]
    out = {'items': len(scored), 'pairs': len(pairs), 'excluded': sum(not p['eligible_by_filters'] for p in pairs),
           'pool': sum(p[f'{method}_rank'] is not None for p in pairs), 'k': {}}
    for k in ks:
        hit = lambda p: p[f'{method}_rank'] is not None and p[f'{method}_rank'] <= k
        item_recalls = [sum(hit(p) for p in i['pairs']) / len(i['pairs']) for i in scored]
        out['k'][k] = {
            'macro': sum(item_recalls) / len(scored), 'micro': sum(hit(p) for p in pairs) / len(pairs),
            'hit_items': sum(r > 0 for r in item_recalls), 'full_items': sum(r == 1 for r in item_recalls),
            'pair_hits': sum(hit(p) for p in pairs)}
    return out


def loss_split(items, method, k):
    pairs = [p for i in items if i['status'] == 'scored' for p in i['pairs']]
    split = {'필터에서 제외': 0, f'후보 50개 밖(순위 없음)': 0, f'후보 안이지만 Top-{k} 밖': 0, f'Top-{k} 안(회수)': 0}
    for p in pairs:
        rank = p[f'{method}_rank']
        if not p['eligible_by_filters']:
            split['필터에서 제외'] += 1
        elif rank is None:
            split['후보 50개 밖(순위 없음)'] += 1
        elif rank > k:
            split[f'후보 안이지만 Top-{k} 밖'] += 1
        else:
            split[f'Top-{k} 안(회수)'] += 1
    return split


def case_changes(items, base_items, method, k, base_method=None):
    base_method = base_method or method
    base = {(i['set'], i['qid'], i['name']): i for i in base_items if i['status'] == 'scored'}
    rows = []
    for i in items:
        if i['status'] != 'scored':
            continue
        b = base.get((i['set'], i['qid'], i['name']))
        if b is None:
            continue
        now = {p['chunk_id'] for p in i['pairs'] if p[f'{method}_rank'] is not None and p[f'{method}_rank'] <= k}
        was = {p['chunk_id'] for p in b['pairs'] if p[f'{base_method}_rank'] is not None and p[f'{base_method}_rank'] <= k}
        if now != was:
            rows.append({'set': i['set'], 'qid': i['qid'], 'item': i['name'], 'gold_pairs': len(i['pairs']),
                         'gained': sorted(now - was), 'lost': sorted(was - now)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--purpose', required=True)
    parser.add_argument('--methods', nargs='+', default=['dense', 'hybrid'], choices=list(METHOD_NAMES))
    parser.add_argument('--baseline-runs', type=Path, nargs='+')
    parser.add_argument('--baseline-label', default='직전 기준')
    parser.add_argument('--baseline-method', choices=list(METHOD_NAMES),
                        help='기준 실행에서 비교할 방식. 생략하면 같은 방식끼리 비교(예: Hybrid를 같은 실행의 Dense와 비교)')
    parser.add_argument('--focus-k', type=int, nargs='+', default=[5, 10], help='손실 분해·개선/회귀 사례를 볼 K')
    parser.add_argument('--setting', action='append', default=[], metavar='항목=값', help='설정 표에 더할 줄')
    parser.add_argument('--note', action='append', default=[])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    items, summaries = load_runs(args.runs)
    base_items, base_summaries = load_runs(args.baseline_runs) if args.baseline_runs else ([], [])
    ks = summaries[0]['top_ks']
    targets = [args.out_dir / name for name in ('present.md', 'metrics_by_k.csv', 'case_changes.csv')]
    existing = [str(t) for t in targets if t.exists()]
    if existing and not args.overwrite:
        parser.error(f'이미 있습니다(--overwrite 필요): {existing}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    now = {m: metrics(items, m, ks) for m in args.methods}
    before = {m: metrics(base_items, args.baseline_method or m, ks) for m in args.methods} if base_items else {}
    scored = [i for i in items if i['status'] == 'scored']
    statuses = {}
    for i in items:
        statuses[i['status']] = statuses.get(i['status'], 0) + 1
    first = summaries[0]

    with open(args.out_dir / 'metrics_by_k.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'K', 'macro_recall', 'micro_recall', 'hit_items', 'full_recovery_items', 'pair_hits',
                    'evaluated_items', 'gold_pairs', 'filter_excluded_pairs', 'pairs_in_candidate_pool'])
        for m in args.methods:
            for k in ks:
                v = now[m]['k'][k]
                w.writerow([m, k, f"{v['macro']:.6f}", f"{v['micro']:.6f}", v['hit_items'], v['full_items'], v['pair_hits'],
                            now[m]['items'], now[m]['pairs'], now[m]['excluded'], now[m]['pool']])

    change_rows = []
    for m in args.methods:
        for k in args.focus_k:
            for row in case_changes(items, base_items, m, k, args.baseline_method) if base_items else []:
                change_rows.append({'method': m, 'K': k, **row})
    with open(args.out_dir / 'case_changes.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'K', 'set', 'question_id', 'item', 'gold_pairs', 'gained_chunks', 'lost_chunks'])
        for r in change_rows:
            w.writerow([r['method'], r['K'], r['set'], r['qid'], r['item'], r['gold_pairs'],
                        ' '.join(r['gained']), ' '.join(r['lost'])])

    filters = '국가(장소 별칭)' + (' + 항공사(질문의 항공사 자료만, 항공사 외 자료는 유지)' if first.get('airline_filter') else '')
    setting = [('평가 데이터', ' + '.join(f"`{Path(s['dataset']).name}`" for s in summaries)),
               ('입력 쿼리', '정답 구조화(`expected_parsed`)로 만든 물품별 검색 쿼리. LLM 호출 없음'),
               ('임베딩·DB', f"{first['embedding']['model']} ({first['embedding']['parameters'].get('device')}), "
                            f"Chroma `{first['database']['collection']}` {first['database']['chunks']}청크"),
               ('후보 필터', filters), ('후보 수 / 깊이', f"방식별 후보 {first['candidate_k']}개, K={', '.join(map(str, ks))}"),
               ('Hybrid', f"BM25 {first['weights']['bm25']:.2f} + Dense {first['weights']['dense']:.2f}, RRF c={first['rrf_c']}"),
               ('평가 대상', f"{len(scored)}개 물품, 정답 청크 쌍 {now[args.methods[0]]['pairs']}개 "
                            f"(근거 없음·대응 불가 제외: {json.dumps({k: v for k, v in statuses.items() if k != 'scored'}, ensure_ascii=False)})")]
    setting += [tuple(s.split('=', 1)) for s in args.setting]

    md = [f'# 검색 후보 평가 — {args.title}', '', '## 실험 개요', '', f'- **목적:** {args.purpose}']
    md += [f'- **주의:** {n}' for n in args.note]
    md += ['', '| 항목 | 내용 |', '| --- | --- |'] + [f'| {a} | {b} |' for a, b in setting]
    if base_items:
        md.append(f'| 직전 기준 | {args.baseline_label} |')

    md += ['', '## 1. 종합 평가: 깊이별 Macro Recall', '',
           '정답 청크 중 상위 K개 안에 든 비율을 물품마다 구해 평균한 값이다(물품 평균). 값이 높을수록 판정에 쓸 후보 pool이 정답 근거를 많이 담는다.', '']
    for m in args.methods:
        header = ['K'] + [str(k) for k in ks]
        md += [f'**{METHOD_NAMES[m]}**', '', '| 지표 | ' + ' | '.join(f'@{k}' for k in ks) + ' |', '| --- | ' + ' | '.join('---:' for _ in ks) + ' |']
        md.append('| Macro Recall | ' + ' | '.join(pct(now[m]['k'][k]['macro']) for k in ks) + ' |')
        if before:
            md.append(f'| 대비({args.baseline_label}) | ' + ' | '.join(delta(now[m]['k'][k]['macro'], before[m]['k'][k]['macro']) for k in ks) + ' |')
        md.append('| Micro Recall(정답 쌍) | ' + ' | '.join(pct(now[m]['k'][k]['micro']) for k in ks) + ' |')
        md.append('| 정답 쌍 적중 | ' + ' | '.join(f"{now[m]['k'][k]['pair_hits']}/{now[m]['pairs']}" for k in ks) + ' |')
        md.append('| Hit(1개 이상 회수한 물품) | ' + ' | '.join(f"{now[m]['k'][k]['hit_items']}/{now[m]['items']}" for k in ks) + ' |')
        md.append('| 전체 회수 물품(정답을 모두 담음) | ' + ' | '.join(f"{now[m]['k'][k]['full_items']}/{now[m]['items']}" for k in ks) + ' |')
        md.append('')

    md += ['## 2. 손실 분해 (정답 청크 쌍이 어디서 빠지는가)', '']
    for k in args.focus_k:
        md += [f'**K={k}**', '', '| 구분 | ' + ' | '.join(METHOD_NAMES[m] for m in args.methods) + ' |',
               '| --- | ' + ' | '.join('---:' for _ in args.methods) + ' |']
        splits = {m: loss_split(items, m, k) for m in args.methods}
        for label in splits[args.methods[0]]:
            md.append(f'| {label} | ' + ' | '.join(str(splits[m][label]) for m in args.methods) + ' |')
        md.append('')
    md += ['후보 50개 밖은 이 방식으로는 깊이를 늘려도 회수되지 않는 쌍이다. 후보 안이지만 Top-K 밖은 깊이를 늘리거나 재정렬(reranking)로 회수할 수 있다.', '']

    if base_items:
        md += [f'## 3. {args.baseline_label} 대비 질문별 변화', '']
        for m in args.methods:
            for k in args.focus_k:
                rows = [r for r in change_rows if r['method'] == m and r['K'] == k]
                gained = sum(len(r['gained']) for r in rows)
                lost = sum(len(r['lost']) for r in rows)
                md.append(f'- **{METHOD_NAMES[m]} Top-{k}:** 정답 쌍 {gained}개 추가 회수, {lost}개 손실, 변화가 있는 물품 {len(rows)}개'
                          + (f' (회귀 물품 {sum(1 for r in rows if r["lost"])}개: ' +
                             ', '.join(f"{r['qid']} {r['item']}" for r in rows if r['lost'])[:300] + ')' if lost else ''))
        md += ['', f'전체 목록은 `case_changes.csv`에 있다.', '']

    md += ['## 핵심 지표 (장표용)', '']
    for m in args.methods:
        for k in args.focus_k:
            v = now[m]['k'][k]
            line = f"- **{METHOD_NAMES[m]} Top-{k}:** Macro Recall {pct(v['macro'])}, 정답 쌍 {v['pair_hits']}/{now[m]['pairs']}, 전체 회수 물품 {v['full_items']}/{now[m]['items']}"
            if before:
                line += f" ({args.baseline_label} 대비 Macro Recall {delta(v['macro'], before[m]['k'][k]['macro'])})"
            md.append(line)
    md += ['', '## 지표 설명', '',
           '- **Macro Recall@K:** 물품마다 (정답 청크 중 상위 K개에 든 수 ÷ 정답 청크 수)를 구해 평균한다. 정답이 많은 물품과 적은 물품을 같은 비중으로 본다.',
           '- **Micro Recall@K:** 모든 (물품, 정답 청크) 쌍 중 상위 K개에 든 쌍의 비율이다.',
           '- **Hit / 전체 회수:** 정답을 1개 이상 / 전부 상위 K개에 담은 물품 수다.',
           '- 정답에 없는 청크가 검색되어도 오답으로 세지 않는다(정답 밖 문서는 미검수일 수 있다). 그래서 이 단계에서는 Precision을 쓰지 않는다.',
           '- 판정 정확도(Accuracy)·Macro-F1과 순위 지표(MRR, nDCG)는 이 단계의 주 지표가 아니다. 후보 확대 단계는 Recall 계열로 판단한다.', '']
    (args.out_dir / 'present.md').write_text('\n'.join(md), encoding='utf-8')
    print('Saved:', args.out_dir)


if __name__ == '__main__':
    main()
