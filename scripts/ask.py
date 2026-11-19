"""질문 분석 → 물품별 검색 → 최종 답변. 기본 결과는 outputs/results에 JSON 저장."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from app.rag import answer_question
from app.search import use_retrieval_method
from app.settings import FILTER_FIELDS, ENV_FILE, RESULTS_DIR, KST, DEFAULT_K, OVERRIDE_KEYS

def save_result(path, question, result):
    """답변을 JSON 파일로 저장한다. 폴더(또는 / \\ 로 끝나는 경로)를 주면 이름을 자동으로 만든다."""
    target = Path(path)
    if target.is_dir() or str(path).endswith(('/', '\\')):
        target = target / f'answer_{datetime.now(KST):%Y%m%d_%H%M%S_%f}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'question': question,
        'parsed': result['parsed'],
        'answer': result['answer'],
        'sources': result['sources'],
        'groups': result['groups'],
        'searched_document_sources': result['searched_document_sources'],
        'model_used_sources': result['model_used_sources'],
        'model_decisions': result['model_decisions'],
        'filters': result['filters'],
        'k': result['k'],
        'message': result['message'],
        'executed_at': datetime.now(KST).isoformat(),
    }
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return target


# ── 5. CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('question')
    parser.add_argument('--env-file', type=Path, default=ENV_FILE)
    parser.add_argument('--filter', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--k', type=int, default=DEFAULT_K, help='물품 1개당 판정에 넘기는 최종 문서 수')
    parser.add_argument('--retrieval', choices=['config', 'dense'], default='config',
                        help='config: config/setting.json의 retrieval(기본 Hybrid+reranking). dense: 이전 방식')
    parser.add_argument('--output', default=str(RESULTS_DIR) + '/', metavar='PATH', help='JSON 저장 위치. 기본: outputs/results/')
    parser.add_argument('--no-save', action='store_true', help='파일로 저장하지 않고 터미널에만 출력')
    parser.add_argument('--show-parsed', action='store_true', help='분리 결과와 물품별 문서 수 출력')
    for name in ('departure', 'arrival', 'departure-airline', 'arrival-airline', 'travel-date', 'baggage-type', 'airline'):
        parser.add_argument('--' + name, help='지정하면 질문에서 추출한 값보다 우선')
    args = parser.parse_args()
    filters = {}
    for item in args.filter:
        key, separator, value = item.partition('=')
        if not separator or key not in FILTER_FIELDS or not value or key in filters:
            parser.error('필터는 중복 없이 KEY=VALUE로 입력하세요. 지원 필드: ' + ', '.join(sorted(FILTER_FIELDS)))
        filters[key] = value
    overrides = {key: getattr(args, key) for key in (*OVERRIDE_KEYS, 'airline') if getattr(args, key)}
    load_dotenv(args.env_file, override=False)
    with use_retrieval_method(args.retrieval):
        result = answer_question(args.question, overrides, filters, args.k)
    if args.show_parsed:
        print(json.dumps(result['parsed'], ensure_ascii=False, indent=2))
        for group in result['groups']:
            print(f"- {group['item']['name']}: 문서 {len(group['documents'])}개")
        print('-' * 40)
    print(json.dumps(result['answer'], ensure_ascii=False, indent=2)
          if isinstance(result['answer'], dict) else result['answer'])
    if not args.no_save:
        print('\n저장됨:', save_result(args.output, args.question, result))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
