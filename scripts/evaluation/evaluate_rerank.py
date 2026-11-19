# -*- coding: utf-8 -*-
"""저장된 reranker 점수로 pool을 재정렬해 Recall@M·MRR·nDCG@10을 비교한다. LLM·OpenAI 호출·모델 로딩 없음.

python -m scripts.evaluation.evaluate_rerank --runs <test9 검색 평가 폴더> <supplement 검색 평가 폴더> \\
    --score-dir tests/ver8_test/scores --out-dir tests/ver8_test/retrieval/rerank_full_n30 --pool-sizes 10 20 30

점수는 `scripts.evaluation.score_rerank`가 `<score-dir>/scores_<form>_<dtype>_<max_length>.jsonl`에 저장한 것을 읽는다. 모든 물품에서
채점이 끝난 순위 구간까지만 pool 크기 N으로 평가하며, 채점이 모자란 N은 건너뛰고 알린다(이어서 채점한 뒤 다시 실행).
pool은 Hybrid(RRF) 융합 순서의 앞 N개이고 N=all은 두 방식 후보 합집합 전체다.
기존 출력 파일은 --overwrite 없이는 덮어쓰지 않는다.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.evaluation.rerank_common import ROOT, ScoreStore, load_pools, unit_key

MODEL = 'BAAI/bge-reranker-v2-m3'
FORMS = {'full': ('전체 쿼리', 'query'), 'item': ('물품명+details', 'item_query')}
TYPE_NAMES = {'airline': '항공사', 'admin_rule': '행정규칙', 'airport': '공항', 'government': '정부 기관', 'press': '보도자료'}


def load_scores(units, form, args):
    """저장된 점수 파일에서 물품별 점수와 설정을 읽는다. 파일이 없으면 None."""
    path = Path(args.score_dir) / f'scores_{form}_{args.dtype}_{args.max_length}.jsonl'
    if not path.exists():
        return None, None
    store = ScoreStore(path)
    scores = {unit_key(u): store.scores(unit_key(u)) for u in units}
    first = next((json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()), {})
    return scores, {k: first.get(k) for k in ('model', 'dtype', 'max_length', 'batch_size', 'device')}


def to_dense_pool(units, scores_by_form):
    """pool을 Dense 순위 순서로 바꾸고 점수는 문서 ID로 옮긴다(추가 채점 없이 재사용).

    Hybrid 순서로 채점해 둔 문서만 점수가 있으므로 Dense 순서의 앞에서부터 점수가 이어진 만큼만 쓴다.
    """
    dense_units, dense_scores = [], {form: {} for form in scores_by_form}
    for u in units:
        ordered = sorted((d for d in u['pool'] if d['dense_rank'] is not None), key=lambda d: d['dense_rank'])
        dense_units.append(dict(u, pool=ordered))
        for form, scores in scores_by_form.items():
            by_id = {d['id']: sc for d, sc in zip(u['pool'], scores[unit_key(u)])}
            prefix = []
            for d in ordered:
                if d['id'] not in by_id:
                    break
                prefix.append(by_id[d['id']])
            dense_scores[form][unit_key(u)] = prefix
    return dense_units, dense_scores


def usable_sizes(units, scores, sizes):
    """모든 물품에서 앞 N개의 점수가 있는 N만 남긴다."""
    ok = []
    for n in sizes:
        need = lambda u: len(u['pool']) if n == 'all' else min(n, len(u['pool']))
        if all(len(scores[unit_key(u)]) >= need(u) for u in units):
            ok.append(n)
    return ok


def ranking_metrics(order_ids, gold):
    """order_ids: 순위대로의 청크 ID. gold: 정답 청크 집합. 반환: M별 회수 수, MRR, nDCG@10(이진)."""
    hits = {m: sum(1 for c in order_ids[:m] if c in gold) for m in CUTOFFS}
    first = next((i for i, c in enumerate(order_ids, 1) if c in gold), None)
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(order_ids[:10], 1) if c in gold)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    return hits, (1 / first if first else 0.0), (dcg / ideal if ideal else 0.0)


def evaluate(units, scores_by_form, sizes):
    rows = []
    for n_label in sizes:
        variants = {'hybrid_order': lambda u, s, n: [d['id'] for d in u['pool'][:n]]}
        variants['dense_order'] = lambda u, s, n: [d['id'] for d in sorted(
            (d for d in u['pool'] if d['dense_rank'] is not None), key=lambda d: d['dense_rank'])][:n]
        for form in scores_by_form:
            variants[f'rerank_{form}'] = (lambda form: lambda u, s, n: [
                u['pool'][i]['id'] for i in sorted(range(min(n, len(u['pool']))), key=lambda i: -s[form][i])])(form)
        for name, order in variants.items():
            per_item = []
            for u in units:
                n = len(u['pool']) if n_label == 'all' else min(n_label, len(u['pool']))
                key = f"{u['set']}|{u['qid']}|{u['item']}"
                s = {form: scores[key] for form, scores in scores_by_form.items()}
                if name == 'dense_order':
                    ids = order(u, s, 10 ** 6)  # Dense 순서는 pool 크기와 무관하게 상위 M개
                else:
                    ids = order(u, s, n)
                gold = set(u['gold'])
                hits, rr, ndcg = ranking_metrics(ids, gold)
                pool_ids = {d['id'] for d in u['pool'][:n]}
                per_item.append({'gold': len(gold), 'hits': hits, 'rr': rr, 'ndcg': ndcg, 'pool_hits': len(gold & pool_ids),
                                 'pool_n': n})
            rows.append({'N': n_label, 'variant': name, 'items': per_item})
    return rows


def aggregate(row, m):
    items = row['items']
    return {'macro': statistics.mean(i['hits'][m] / i['gold'] for i in items),
            'pairs': sum(i['hits'][m] for i in items), 'gold': sum(i['gold'] for i in items)}


def main():
    global CUTOFFS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--forms', nargs='+', default=['full', 'item'], choices=list(FORMS))
    parser.add_argument('--pool-sizes', nargs='+', default=['10', '20', '30', '50', 'all'])
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[3, 5, 8, 10])
    parser.add_argument('--score-dir', type=Path, required=True, help='scripts.evaluation.score_rerank가 점수를 저장한 폴더')
    parser.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32', help='읽을 점수 파일의 정밀도')
    parser.add_argument('--max-length', type=int, default=512, help='읽을 점수 파일의 최대 길이')
    parser.add_argument('--latency', type=Path, help='benchmark_reranker 결과 JSON(보고서에 지연 표를 넣는다)')
    parser.add_argument('--title', default='cross-encoder 재정렬')
    parser.add_argument('--pool-source', choices=['hybrid', 'dense'], default='hybrid',
                        help='hybrid: Hybrid 융합 순서의 앞 N개. dense: Dense 순위의 앞 N개(저장된 점수를 문서 ID로 재사용, 채점 안 된 문서가 있으면 그 N은 건너뜀)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    CUTOFFS = sorted(args.cutoffs)
    sizes = [s if s == 'all' else int(s) for s in args.pool_sizes]
    present = args.out_dir / 'present.md'
    if present.exists() and not args.overwrite:
        parser.error(f'이미 있습니다(--overwrite 필요): {present}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    units = load_pools(args.runs)
    scores_by_form, settings = {}, {}
    for form in args.forms:
        scores, setting = load_scores(units, form, args)
        if scores is None:
            print(f'점수 파일이 없어 쿼리 형태 {form}은 건너뜁니다.')
            continue
        scores_by_form[form], settings[form] = scores, setting
    if not scores_by_form:
        parser.error('읽을 점수 파일이 없습니다. 먼저 scripts.evaluation.score_rerank로 채점하세요.')
    args.forms = [f for f in args.forms if f in scores_by_form]
    if args.pool_source == 'dense':
        units, scores_by_form = to_dense_pool(units, scores_by_form)
    ready = usable_sizes(units, {k: v for k, v in list(scores_by_form.values())[0].items()}, sizes)
    for form in args.forms[1:]:
        ready = [n for n in ready if n in usable_sizes(units, scores_by_form[form], [n])]
    skipped = [n for n in sizes if n not in ready]
    if skipped:
        print(f'채점이 모자라 건너뛰는 pool 크기: {skipped} (이어서 채점한 뒤 다시 실행)')
    if not ready:
        parser.error('평가할 수 있는 pool 크기가 없습니다.')
    sizes = ready
    rows = evaluate(units, scores_by_form, sizes)
    dense = args.pool_source == 'dense'
    if dense:
        rows = [r for r in rows if r['variant'] != 'dense_order']  # Dense pool에서는 pool 순서가 곧 Dense 순서
    names = {'hybrid_order': 'Dense 순서(재정렬 없음)' if dense else 'Hybrid 순서(재정렬 없음)',
             **({} if dense else {'dense_order': 'Dense 순서(재정렬 없음)'}),
             **{f'rerank_{f}': f'재정렬: {FORMS[f][0]}' for f in args.forms}}

    import csv
    with open(args.out_dir / 'metrics.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pool_N', 'variant', 'M', 'macro_recall', 'micro_recall', 'pair_hits', 'gold_pairs', 'mrr', 'ndcg10', 'pool_macro_recall'])
        for r in rows:
            mrr = statistics.mean(i['rr'] for i in r['items'])
            ndcg = statistics.mean(i['ndcg'] for i in r['items'])
            pool_recall = statistics.mean(i['pool_hits'] / i['gold'] for i in r['items'])
            for m in CUTOFFS:
                a = aggregate(r, m)
                w.writerow([r['N'], r['variant'], m, f"{a['macro']:.6f}", f"{a['pairs'] / a['gold']:.6f}", a['pairs'], a['gold'], f'{mrr:.6f}', f'{ndcg:.6f}', f'{pool_recall:.6f}'])

    pct = lambda v: f'{v * 100:.2f}%'
    items_n = len(units)
    gold_n = sum(len(u['gold']) for u in units)
    md = [f'# 재정렬 평가 — {args.title}', '', '## 실험 개요', '',
          ('- **목적:** Dense(국가+항공사 필터, 전체 쿼리) 상위 N개 후보를 cross-encoder로 다시 정렬했을 때 정답 근거가 상위 M개에 더 들어오는지 확인한다. 검색이 빠른 Dense pool이 Hybrid pool의 대안이 되는지 보는 용도다.' if dense else
           '- **목적:** Hybrid(국가+항공사 필터, 전체 쿼리) 후보 pool의 앞 N개를 cross-encoder로 다시 정렬했을 때 정답 근거가 상위 M개에 더 들어오는지, 그리고 N을 키울 때의 이득을 확인한다.'),
          '- **주의:** 입력 pool은 정답 구조화에서 만든 쿼리로 검색한 결과다(구조화 오류 제외). 같은 문서 목록을 순서만 바꾸므로 재정렬 후 Recall@M은 pool의 Recall@N을 넘을 수 없다.',
          '', '| 항목 | 내용 |', '| --- | --- |',
          f'| 평가 대상 | {items_n}개 물품, 정답 청크 쌍 {gold_n}개 (`{"`, `".join(Path(r).name for r in args.runs)}`) |',
          (f'| 후보 pool | Dense 순위의 앞 N개(국가+항공사 필터). 점수는 Hybrid 순서로 채점해 둔 것을 문서 ID로 재사용 |' if dense else
           f'| 후보 pool | Hybrid(Kiwi BM25 + Dense, RRF) 융합 순서의 앞 N개. N=all은 Dense 50 ∪ BM25 50(평균 {statistics.mean(len(u["pool"]) for u in units):.0f}개) |'),
          f'| reranker | `{MODEL}` (cross-encoder), {args.dtype}, 최대 길이 {args.max_length}, 배치 {settings[args.forms[0]].get("batch_size")}, device {settings[args.forms[0]].get("device")} (채점은 `scripts.evaluation.score_rerank`로 따로 실행) |',
          f'| reranker 쿼리 | ' + ', '.join(f'{FORMS[f][0]}' for f in args.forms) + ' |',
          '| 기준 | 재정렬 없는 Hybrid 순서, Dense 순서 |', '']

    for m in CUTOFFS:
        md += [f'## Recall@{m} (Macro, 물품 평균)', '', '| pool N | pool 천장(Recall@N) | ' + ' | '.join(names[v] for v in names) + ' |',
               '| --- | ---: | ' + ' | '.join('---:' for _ in names) + ' |']
        for n_label in sizes:
            sel = {r['variant']: r for r in rows if r['N'] == n_label}
            ceiling = statistics.mean(i['pool_hits'] / i['gold'] for i in sel['hybrid_order']['items'])
            md.append(f'| {n_label} | {pct(ceiling)} | ' + ' | '.join(pct(aggregate(sel[v], m)['macro']) for v in names) + ' |')
        md.append('')
    md += ['## Micro Recall (정답 쌍 기준)', '',
           '모든 (물품, 정답 청크) 쌍 중 재정렬 후 상위 M개에 든 쌍의 비율이다. 위의 Macro Recall(물품 평균)과 함께 본다.', '',
           '| pool N | 방식 | ' + ' | '.join(f'@{m}' for m in CUTOFFS) + ' |', '| --- | --- | ' + ' | '.join('---:' for _ in CUTOFFS) + ' |']
    for n_label in sizes:
        for v in names:
            sel = next(r for r in rows if r['N'] == n_label and r['variant'] == v)
            md.append(f'| {n_label} | {names[v]} | ' + ' | '.join(
                f"{aggregate(sel, m)['pairs'] / aggregate(sel, m)['gold'] * 100:.2f}% ({aggregate(sel, m)['pairs']}/{aggregate(sel, m)['gold']})" for m in CUTOFFS) + ' |')
    md.append('')
    md += ['## 순위 지표: MRR / nDCG@10', '', '| pool N | ' + ' | '.join(f'{names[v]} MRR / nDCG@10' for v in names) + ' |',
           '| --- | ' + ' | '.join('---:' for _ in names) + ' |']
    for n_label in sizes:
        sel = {r['variant']: r for r in rows if r['N'] == n_label}
        md.append(f'| {n_label} | ' + ' | '.join(
            f"{statistics.mean(i['rr'] for i in sel[v]['items']):.3f} / {statistics.mean(i['ndcg'] for i in sel[v]['items']):.3f}" for v in names) + ' |')
    md.append('')

    meta = {}
    for rel in ['Korea/chunk/all_kr.jsonl', 'China/chunk/cn.jsonl', 'Japan/chunk/jp.jsonl', 'UK/chunk/uk.jsonl', 'US/chunk/us.jsonl']:
        for line in (ROOT / 'data' / rel).read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row = json.loads(line)
                meta[row['id']] = row['metadata']['source_type']
    best_form = args.forms[0]
    largest = max((n for n in sizes if n != 'all'), default=None)  # 평가한 가장 큰 숫자 N
    for n_label, m in ((largest, 5), (largest, 10)):
        if n_label is None or m not in CUTOFFS:
            continue
        stats = defaultdict(lambda: [0, 0, 0])
        for u in units:
            key = f"{u['set']}|{u['qid']}|{u['item']}"
            n = min(n_label, len(u['pool']))
            base = {d['id'] for d in u['pool'][:m]}
            order = sorted(range(n), key=lambda i: -scores_by_form[best_form][key][i])[:m]
            new = {u['pool'][i]['id'] for i in order}
            for c in u['gold']:
                t = TYPE_NAMES.get(meta[c], meta[c])
                stats[t][0] += 1
                stats[t][1] += c in base
                stats[t][2] += c in new
        md += [f'## 자료 유형별 정답 쌍 (pool N={n_label}, Top-{m}, 재정렬: {FORMS[best_form][0]})', '',
               '| 자료 유형 | 정답 쌍 | Hybrid 순서 회수 | 재정렬 회수 | 순변화 |', '| --- | ---: | ---: | ---: | ---: |']
        for t, (total, before, after) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            md.append(f'| {t} | {total} | {before} | {after} | {after - before:+d} |')
        md.append('')

    if args.latency and args.latency.exists():
        lat = json.loads(args.latency.read_text(encoding='utf-8'))
        md += [f'## 지연 (검색 1건당, {lat["device"]}, 표본 {lat["samples"]}건)', '',
               '| 설정(정밀도:최대길이:배치) | ' + ' | '.join(f'N={n}' for n in lat['pool_sizes']) + ' | 상위5 일치율(vs fp32:512) |',
               '| --- | ' + ' | '.join('---:' for _ in lat['pool_sizes']) + ' | ---: |']
        for spec, res in lat['configs'].items():
            cells = [f"{res[str(n)]['mean_s']:.2f}s" if str(n) in res else '-' for n in lat['pool_sizes']]
            same = statistics.mean(res[str(n)]['top5_same_as_reference'] for n in lat['pool_sizes'] if str(n) in res)
            md.append(f'| {spec} | ' + ' | '.join(cells) + f' | {pct(same)} |')
        md.append('')

    md += ['## 지표 설명', '',
           '- **Recall@M:** 정답 청크 중 재정렬 후 상위 M개에 든 비율을 물품마다 구해 평균한다(Macro).',
           '- **pool 천장:** 같은 N에서 pool 안에 있는 정답의 비율이다. 재정렬은 이 값을 넘을 수 없다.',
           '- **MRR:** 첫 정답의 순위 역수 평균, **nDCG@10:** 상위 10개 안의 정답 위치를 반영한 이진 이득 지표다.',
           '- 정답 밖 문서는 미검수일 수 있어 Precision은 쓰지 않는다.', '']
    present.write_text('\n'.join(md), encoding='utf-8')
    print('Saved:', present)


if __name__ == '__main__':
    main()
