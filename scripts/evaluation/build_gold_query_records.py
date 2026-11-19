# -*- coding: utf-8 -*-
"""정답 구조화(expected_parsed)로 물품별 검색 쿼리를 만들어 records 형식으로 저장한다.

python -m scripts.evaluation.build_gold_query_records --questions data/test/test9_questions.json \\
    --queries data/test/test9_queries.json --output data/test/test9_gold_query_records/test9

LLM 호출·DB 검색은 없다. 검색 쿼리는 서비스와 같은 build_retrieval_query로 만든다. 구조화 정답을 입력으로 쓰므로
"구조화가 정답과 같이 이루어졌다고 가정한" 검색 평가의 입력이며 실제 LLM 출력이 아니다(records의 input_kind로 표시).
검색 평가 도구(app.evaluate_hybrid_search --records)가 그대로 읽는다. 기존 폴더는 덮어쓰지 않는다.
"""
import argparse
import json
from pathlib import Path

from app.rag import build_retrieval_query


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'이미 있습니다: {args.output}')
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    queries = json.loads(args.queries.read_text(encoding='utf-8'))
    parsed_by_id = {c['id']: c['expected_parsed'] for rows in queries.values() for c in rows}
    args.output.mkdir(parents=True)
    count = 0
    for category, rows in questions.items():
        for case in rows:
            parsed = json.loads(json.dumps(parsed_by_id[case['id']], ensure_ascii=False))
            for index, item in enumerate(parsed['items'], 1):
                item['item_id'] = f'item_{index}'
            groups = [{'item': item, 'query': build_retrieval_query(item, parsed['conditions']), 'documents': []}
                      for item in parsed['items']]
            record = {'id': case['id'], 'category': category, 'question': case['question'], 'status': 'success',
                      'input_kind': 'gold_structure', 'parsed': parsed, 'filters': {}, 'groups': groups}
            (args.output / f"{case['id']}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            count += 1
    print(f'{count}개 질문의 records를 {args.output}에 저장했습니다.')


if __name__ == '__main__':
    main()
