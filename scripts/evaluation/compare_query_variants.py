# -*- coding: utf-8 -*-
"""같은 필터에서 쿼리 방식만 다른 두 검색 평가의 정답 쌍 변화를 방식·자료 유형·질문 조건별로 비교한다.

python -m scripts.evaluation.compare_query_variants --baseline <전체 쿼리 run 폴더...> --variant <간소 쿼리 run 폴더...> \\
    --output tests/ver7_test/retrieval/query_variant_analysis.md

LLM 호출·검색 재실행은 없다. 저장된 질문별 JSON(정답 청크별 순위)과 청크 metadata만 읽는다. 기존 파일은 덮어쓰지 않는다.
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/evaluation/ 기준 저장소 루트
CHUNKS = ['Korea/chunk/all_kr.jsonl', 'China/chunk/cn.jsonl', 'Japan/chunk/jp.jsonl', 'UK/chunk/uk.jsonl', 'US/chunk/us.jsonl']
METHODS = ('dense', 'bm25', 'hybrid')
TYPE_NAMES = {'airline': '항공사', 'admin_rule': '행정규칙(국토교통부 고시)', 'airport': '공항', 'government': '정부 기관', 'press': '보도자료'}


def load_meta():
    meta = {}
    for name in CHUNKS:
        for line in (ROOT / 'data' / name).read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row = json.loads(line)
                meta[row['id']] = row['metadata']
    return meta


def load(dirs):
    out = {}
    for d in dirs:
        for path in glob.glob(str(Path(d) / '*.json')):
            if path.endswith('summary.json'):
                continue
            result = json.loads(Path(path).read_text(encoding='utf-8'))
            for item in result['items']:
                if item['status'] != 'scored':
                    continue
                for doc in item['documents']:
                    out[(Path(d).name, result['id'], item['name'], doc['chunk_id'])] = dict(doc, cond=result['conditions'])
    return out


def macro(data, method, k):
    by_item = defaultdict(list)
    for (s, q, n, _), v in data.items():
        by_item[(s, q, n)].append(v[f'{method}_rank'] is not None and v[f'{method}_rank'] <= k)
    return sum(sum(x) / len(x) for x in by_item.values()) / len(by_item)


def hit(v, method, k):
    return v[f'{method}_rank'] is not None and v[f'{method}_rank'] <= k


def split(base, variant, method, k, group):
    total, lost, gained = Counter(), Counter(), Counter()
    for key, v in base.items():
        g = group(key, v)
        total[g] += 1
        a, b = hit(v, method, k), hit(variant[key], method, k)
        lost[g] += a and not b
        gained[g] += b and not a
    return total, lost, gained


def table(title, header, rows):
    return [f'**{title}**', '', '| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |'] + \
           ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows] + ['']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', type=Path, nargs='+', required=True)
    parser.add_argument('--variant', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'이미 있습니다: {args.output}')
    meta, base, variant = load_meta(), load(args.baseline), load(args.variant)
    if set(base) != set(variant):
        parser.error('두 실행의 정답 쌍 집합이 다릅니다.')

    md = ['# 쿼리 방식 비교 분석 (전체 쿼리 → 간소 쿼리)', '',
          f'같은 필터·같은 정답 쌍 {len(base)}개에서 쿼리만 바꿨을 때의 변화다. 기준은 {", ".join(f"`{p}`" for p in args.baseline)}, '
          f'비교는 {", ".join(f"`{p}`" for p in args.variant)}이다.', '']
    md += table('방식별 Macro Recall (전체 쿼리 → 간소 쿼리)', ['방식', '@5', '@10', '@20', '@50'],
                [[m] + [f'{macro(base, m, k) * 100:.1f}% → {macro(variant, m, k) * 100:.1f}%' for k in (5, 10, 20, 50)] for m in METHODS])
    for title, group in [
        ('정답 근거의 자료 유형', lambda key, v: TYPE_NAMES.get(meta[key[3]]['source_type'], meta[key[3]]['source_type'])),
        ('질문의 항공사 명시 여부', lambda key, v: '항공사 명시' if (v['cond']['departure']['airline'] or v['cond']['arrival']['airline']) else '항공사 미명시'),
    ]:
        for method in ('dense', 'bm25'):
            total, lost, gained = split(base, variant, method, args.k, group)
            md += table(f'{method.upper()} Top-{args.k}: {title}별 정답 쌍 (간소 쿼리로 잃음 / 얻음)',
                        ['구분', '정답 쌍', '잃음', '얻음', '순변화'],
                        [[g, total[g], lost[g], gained[g], f'{gained[g] - lost[g]:+d}'] for g in sorted(total, key=lambda x: -total[x])])
    md += ['정답 쌍 단위의 집계이며 원인을 직접 검증한 것은 아니다. 자료 유형별 순변화는 쿼리에서 빠진 단어가 어떤 문서의 순위를 움직였는지에 대한 단서다.', '']
    args.output.write_text('\n'.join(md), encoding='utf-8')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
