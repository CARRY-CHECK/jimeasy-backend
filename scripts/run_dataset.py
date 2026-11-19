"""예시 질문만 RAG에 전달하고 답변/검색 기록을 저장한다. 채점은 수행하지 않는다."""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from app.settings import MODEL_CONFIG
from app.rag import answer_question
from app.query_parser import active_parse_system_prompt, use_parse_system_prompt
from app.search import active_retrieval, use_retrieval_method
from app.settings import (ROOT, ENV_FILE, PROMPTS_DIR, KST, DEFAULT_K, CHROMA_PATH, CHROMA_COLLECTION, CHUNK_SOURCES,
                          PARSE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def create_attempt(output_root):
    """기존 실행을 덮어쓰지 않고 다음 번호의 폴더를 확보한다."""
    output_root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.removeprefix('attempt_')) for path in output_root.iterdir()
               if path.is_dir() and path.name.startswith('attempt_')
               and path.name.removeprefix('attempt_').isdigit()]
    number = max(numbers, default=0) + 1
    while True:
        output = output_root / f'attempt_{number}'
        try:
            output.mkdir()
            return output
        except FileExistsError:
            number += 1


def run_dataset(dataset_path, output, k=DEFAULT_K, parse_system_prompt=PARSE_SYSTEM_PROMPT, retrieval='config'):
    with use_parse_system_prompt(parse_system_prompt), use_retrieval_method(retrieval):
        return _run_dataset(dataset_path, output, k)


def _run_dataset(dataset_path, output, k=DEFAULT_K):
    dataset = json.loads(dataset_path.read_text(encoding='utf-8'))
    cases = [(category, case) for category, rows in dataset.items() for case in rows]
    ids = [case['id'] for _, case in cases]
    if len(ids) != len(set(ids)) or any(not case['question'].strip() for _, case in cases):
        raise ValueError('데이터셋 ID는 고유하고 질문은 비어 있지 않아야 합니다.')
    if any(Path(key).name != key or key in ('.', '..') or '/' in key or '\\' in key for key in ids):
        raise ValueError('데이터셋 ID는 파일명으로 사용할 수 있어야 합니다.')
    output_root = output
    output = create_attempt(output_root)
    print(f'저장 위치: {output}', flush=True)
    records = output / 'records'
    records.mkdir()
    prompts = {name: (PROMPTS_DIR / name).read_text(encoding='utf-8') for name in
               (active_parse_system_prompt(), 'parse_user.txt', JUDGE_SYSTEM_PROMPT, 'user_items.txt')}
    write_json(output / 'prompts.json', prompts)
    write_json(output / 'reference.json', dataset)
    summary = {
        'started_at': datetime.now(KST).isoformat(), 'model': MODEL_CONFIG['answer']['model'], 'models': MODEL_CONFIG,
        'output_dir': str(output.resolve()), 'attempt': output.name,
        'source_contract': 'answer.items[].source=searched; diagnostic model_used_sources=model-reported subset',
        'dataset': str(dataset_path), 'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        'k': k, 'filters': {}, 'chroma_path': str(ROOT / CHROMA_PATH), 'collection': CHROMA_COLLECTION,
        'parse_system_prompt': active_parse_system_prompt(), 'retrieval': active_retrieval(),
        'chunk_sha256': {country: hashlib.sha256(path.read_bytes()).hexdigest() for country, path in CHUNK_SOURCES.items()},
        'prompt_sha256': {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in prompts.items()},
        'total': len(cases), 'success': 0, 'error': 0, 'results': [],
        'note': 'success는 실행/응답 검증 성공이며 정답률이 아닙니다. reference 답변은 LLM에 전달하지 않습니다.',
    }
    answers = {category: [] for category in dataset}
    write_json(output / 'summary.json', summary)
    for category, case in cases:
        start = perf_counter()
        record = {'id': case['id'], 'category': category, 'question': case['question'],
                  'executed_at': datetime.now(KST).isoformat()}
        try:
            # 기준 답변은 이 호출에 전달하지 않는다.
            result = answer_question(case['question'], k=k)
            record.update(status='success', **result)
            summary['success'] += 1
        except Exception as error:
            # 인증 정보나 SDK 요청 본문을 결과에 쓰지 않는다.
            message = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
            for key in ('OPENAI_API_KEY', 'LAW_API_KEY'):
                if os.environ.get(key):
                    message = message.replace(os.environ[key], '[REDACTED]')
            record.update(status='error', answer=None, error=message)
            if hasattr(error, 'diagnostics'):
                record.update(error.diagnostics)
            summary['error'] += 1
        record['duration_seconds'] = round(perf_counter() - start, 3)
        write_json(records / f"{case['id']}.json", record)
        answers[category].append({key: record[key] for key in ('id', 'question', 'answer')})
        summary['results'].append({'id': case['id'], 'status': record['status'], 'file': f"records/{case['id']}.json"})
        write_json(output / 'answers.json', answers)
        write_json(output / 'summary.json', summary)
        print(f"{case['id']}: {record['status']} ({len(summary['results'])}/{len(cases)})", flush=True)
    summary['finished_at'] = datetime.now(KST).isoformat()
    write_json(output / 'summary.json', summary)
    write_json(output_root / 'latest.json', {
        'attempt': output.name, 'summary': f'{output.name}/summary.json',
        'answers': f'{output.name}/answers.json', 'success': summary['success'],
        'error': summary['error'], 'finished_at': summary['finished_at'],
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/sample/questions.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/runs',
                        help='실행 묶음의 상위 폴더. 내부에 attempt_N을 자동 생성합니다.')
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--retrieval', choices=['config', 'dense'], default='config',
                        help="config: config/setting.json의 retrieval(기본 Hybrid+reranking). dense: 이전 방식(국가 필터, Dense 상위 k개). 과거 기준선 재현용")
    parser.add_argument('--parse-system-prompt', default=PARSE_SYSTEM_PROMPT,
                        help='prompts/ 안의 구조화 시스템 프롬프트 파일명. 실행 범위에서만 적용')
    args = parser.parse_args()
    if args.k < 1:
        parser.error('k는 1 이상이어야 합니다.')
    load_dotenv(ENV_FILE, override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error('OPENAI_API_KEY를 설정하세요.')
    summary = run_dataset(args.dataset, args.output, args.k, args.parse_system_prompt, args.retrieval)
    print(f"완료: 성공 {summary['success']}, 오류 {summary['error']} (정확도 평가 아님)")
    return 1 if summary['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
