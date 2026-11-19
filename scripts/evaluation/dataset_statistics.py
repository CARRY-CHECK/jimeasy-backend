# -*- coding: utf-8 -*-
"""평가셋의 정답 통계를 계산해 마크다운과 CSV로 저장한다.

python -m scripts.evaluation.dataset_statistics --questions data/test/test9_questions.json data/test/test9_supplement_questions.json \\
    --names test9 supplement --output-md data/test/test9_statistics.md --output-dir data/test/test9_audit/statistics

LLM·DB 호출은 없다. 질문 파일과 청크 JSONL(data/*/chunk)만 읽는다. 기존 출력 파일은 --overwrite 없이는 덮어쓰지 않는다.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/evaluation/ 기준 저장소 루트
CHUNK_FILES = ['Korea/chunk/all_kr.jsonl', 'China/chunk/cn.jsonl', 'Japan/chunk/jp.jsonl',
               'UK/chunk/uk.jsonl', 'US/chunk/us.jsonl']
LABELS = ['가능', '조건부 가능', '불가', '확인 필요']


def load_chunks():
    meta = {}
    for name in CHUNK_FILES:
        path = ROOT / 'data' / name
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row = json.loads(line)
                meta[row['id']] = row['metadata']
    return meta


def flatten(path, dataset):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = []
    for category, cases in data.items():
        for case in cases:
            items = case['answer']['items']
            if not items:
                rows.append({'dataset': dataset, 'category': category, 'qid': case['id'], 'question': case['question'],
                             'item': None, 'details': None, 'carry_on': None, 'checked': None, 'sources': []})
            for item in items:
                rows.append({'dataset': dataset, 'category': category, 'qid': case['id'], 'question': case['question'],
                             'item': item['name'], 'details': item['details'], 'carry_on': item['carry_on'],
                             'checked': item['checked_baggage'], 'sources': item['source'] or []})
    return rows


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |']
    out += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]
    return '\n'.join(out)


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def source_family(chunk_id):
    return '.'.join(chunk_id.split('.')[:2])


def build(datasets, meta):
    sections = []
    for name, rows in datasets.items():
        item_rows = [r for r in rows if r['item']]
        questions = {r['qid'] for r in rows}
        with_source = [r for r in item_rows if r['sources']]
        pairs = sum(len(r['sources']) for r in item_rows)
        unique_q_chunk = {(r['qid'], s) for r in item_rows for s in r['sources']}
        unique_chunks = {s for r in item_rows for s in r['sources']}
        sections.append(f'## {name}\n\n' + table(['항목', '값'], [
            ['질문', len(questions)], ['물품(정답 항목)', len(item_rows)], ['물품 중 근거가 있는 것', len(with_source)],
            ['물품 중 근거가 없는 것(확인 필요·범위 밖)', len(item_rows) - len(with_source)],
            ['물품×근거 쌍 (같은 청크를 여러 물품이 인용하면 각각 셈)', pairs],
            ['질문×근거 고유 쌍 (같은 질문에서 같은 청크는 한 번)', len(unique_q_chunk)],
            ['고유 청크', len(unique_chunks)],
            ['근거가 있는 물품의 평균 근거 수', f'{pairs / len(with_source):.2f}' if with_source else '-']]))

        by_cat = defaultdict(lambda: {'q': set(), 'items': 0, 'with': 0, 'pairs': 0})
        for r in rows:
            c = by_cat[r['category']]
            c['q'].add(r['qid'])
            if r['item']:
                c['items'] += 1
                c['with'] += bool(r['sources'])
                c['pairs'] += len(r['sources'])
        sections.append(f'### {name}: 분류별\n\n' + table(
            ['분류', '질문', '물품', '근거 있는 물품', '물품×근거 쌍', '평균 근거 수'],
            [[k, len(v['q']), v['items'], v['with'], v['pairs'], f"{v['pairs'] / v['with']:.2f}" if v['with'] else '-']
             for k, v in sorted(by_cat.items())]))

        dist = Counter(len(r['sources']) for r in item_rows)
        sections.append(f'### {name}: 물품당 근거 수 분포\n\n' + table(
            ['근거 수', '물품'], [[k, dist[k]] for k in sorted(dist)]))

        carry, checked = Counter(r['carry_on'] for r in item_rows), Counter(r['checked'] for r in item_rows)
        sections.append(f'### {name}: 라벨 분포\n\n' + table(
            ['라벨', '기내', '위탁'], [[label, carry[label], checked[label]] for label in LABELS]))

        fam = Counter(source_family(s) for r in item_rows for s in r['sources'])
        stype, country = Counter(), Counter()
        for r in item_rows:
            for s in r['sources']:
                m = meta.get(s, {})
                stype[m.get('source_type', '?')] += 1
                country[m.get('country', '?')] += 1
        sections.append(f'### {name}: 근거 출처 구성 (물품×근거 쌍 기준)\n\n'
                        + table(['국가', '쌍'], [[k, v] for k, v in country.most_common()]) + '\n\n'
                        + table(['자료 유형', '쌍'], [[k, v] for k, v in stype.most_common()]) + '\n\n'
                        + table(['문서', '쌍'], [[k, v] for k, v in fam.most_common()]))

        names = defaultdict(lambda: {'q': set(), 'pairs': 0})
        for r in item_rows:
            names[r['item']]['q'].add(r['qid'])
            names[r['item']]['pairs'] += len(r['sources'])
        top = sorted(names.items(), key=lambda kv: (-len(kv[1]['q']), kv[0]))[:15]
        sections.append(f'### {name}: 물품 이름별 (질문 수 상위 15)\n\n' + table(
            ['물품', '질문 수', '물품×근거 쌍'], [[k, len(v['q']), v['pairs']] for k, v in top]))

        usage = Counter(s for r in item_rows for s in r['sources'])
        sections.append(f'### {name}: 많이 인용된 청크 (상위 10)\n\n' + table(
            ['청크', '인용 물품 수', '자료'],
            [[cid, n, meta.get(cid, {}).get('org_name', '?')] for cid, n in usage.most_common(10)]))
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', nargs='+', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if len(args.questions) != len(args.names):
        parser.error('--questions와 --names의 개수가 같아야 합니다.')

    out_md, out_dir = ROOT / args.output_md, ROOT / args.output_dir
    targets = [out_md] + [out_dir / n for n in ('by_question.csv', 'by_item.csv', 'by_source.csv')]
    existing = [str(t) for t in targets if t.exists()]
    if existing and not args.overwrite:
        parser.error(f'이미 있습니다(--overwrite 필요): {existing}')
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_chunks()
    datasets = {name: flatten(ROOT / path, name) for name, path in zip(args.names, args.questions)}
    all_rows = [r for rows in datasets.values() for r in rows]

    by_q = defaultdict(lambda: {'items': 0, 'with': 0, 'pairs': 0})
    for r in all_rows:
        key = (r['dataset'], r['category'], r['qid'], r['question'])
        entry = by_q[key]
        if r['item']:
            entry['items'] += 1
            entry['with'] += bool(r['sources'])
            entry['pairs'] += len(r['sources'])
    write_csv(out_dir / 'by_question.csv', ['dataset', 'category', 'qid', 'question', 'items', 'items_with_source', 'item_source_pairs'],
              [[*k, v['items'], v['with'], v['pairs']] for k, v in by_q.items()])
    write_csv(out_dir / 'by_item.csv',
              ['dataset', 'category', 'qid', 'item', 'details', 'carry_on', 'checked_baggage', 'source_count', 'sources'],
              [[r['dataset'], r['category'], r['qid'], r['item'], r['details'] or '', r['carry_on'], r['checked'],
                len(r['sources']), ' '.join(r['sources'])] for r in all_rows if r['item']])
    cited = defaultdict(list)
    for r in all_rows:
        for s in r['sources']:
            cited[s].append(f"{r['dataset']}:{r['qid']}:{r['item']}")
    write_csv(out_dir / 'by_source.csv', ['chunk_id', 'org_name', 'source_type', 'country', 'cited_by_items', 'cited_by'],
              [[cid, meta.get(cid, {}).get('org_name', ''), meta.get(cid, {}).get('source_type', ''),
                meta.get(cid, {}).get('country', ''), len(v), ' | '.join(v)]
               for cid, v in sorted(cited.items(), key=lambda kv: (-len(kv[1]), kv[0]))])

    rel = Path(os.path.relpath(out_dir, out_md.parent)).as_posix()
    head = ('# 평가셋 정답 통계\n\n'
            f'질문 파일: {", ".join(f"`{p}`" for p in args.questions)}. `scripts/evaluation/dataset_statistics.py`로 생성했으며 LLM·DB 호출은 없다. '
            f'CSV는 [by_question.csv]({rel}/by_question.csv), [by_item.csv]({rel}/by_item.csv), '
            f'[by_source.csv]({rel}/by_source.csv)이다.\n\n'
            '**용어:** *물품×근거 쌍*은 정답 물품마다 적힌 근거 청크의 합이다(같은 청크를 여러 물품이 인용하면 각각 센다). '
            '검색 평가의 정답 쌍 수는 이 값과 다를 수 있다(저장된 쿼리가 있는 물품만 평가하고 필터로 제외되는 쌍이 있다).\n')
    out_md.write_text(head + '\n' + '\n\n'.join(build(datasets, meta)) + '\n', encoding='utf-8')
    print('wrote', out_md, 'and', out_dir)


if __name__ == '__main__':
    main()
