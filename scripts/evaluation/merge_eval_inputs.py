# -*- coding: utf-8 -*-
"""평가 질문·구조화 정답과 실행 기록을 합쳐 총합(test9 + supplement) 평가 입력을 만든다. LLM·DB 호출 없음.

python -m scripts.evaluation.merge_eval_inputs \\
    --questions data/test/test9_questions.json data/test/test9_supplement_questions.json \\
    --queries data/test/test9_queries.json data/test/test9_supplement_queries.json \\
    --out-questions data/test/test9_all_questions.json --out-queries data/test/test9_all_queries.json \\
    [--runs tests/ver2_test/attempt_1 tests/ver2_test/supplement/attempt_2 --out-run tests/ver2_test/combined/attempt_1]

- 분류별 질문 배열을 이어 붙인다. 질문 ID는 전체에서 고유해야 하고 questions·queries의 ID와 질문 원문이 일치해야 한다.
- --runs를 주면 각 실행의 records/를 한 폴더로 복사하고 summary.json에 합친 출처(merged_from)와 설정 차이를 기록한다.
  evaluate_dataset은 records/만 읽고 make_present는 summary.json을 읽으므로 둘 다 그대로 쓸 수 있다.
- 출력 데이터셋 파일이 이미 있으면 내용이 같을 때만 그대로 재사용하고 다르면 오류로 멈춘다(덮어쓰지 않는다).
  --out-run 폴더가 이미 있으면 오류다. 원본 데이터셋과 실행 기록은 수정하지 않는다.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

SETTING_KEYS = ('model', 'models', 'k', 'filters', 'parse_system_prompt', 'prompt_sha256', 'retrieval', 'collection',
                'chunk_sha256')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):  # noqa: D103
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_datasets(paths):
    """분류별 질문 배열을 이어 붙인다. 같은 분류 이름이면 뒤에 덧붙이고 질문 ID 중복은 오류다."""
    merged, seen = {}, {}
    for path in paths:
        for category, cases in read(path).items():
            for case in cases:
                if case['id'] in seen:
                    raise ValueError(f"질문 ID가 겹칩니다: {case['id']} ({seen[case['id']]} / {path})")
                seen[case['id']] = str(path)
                merged.setdefault(category, []).append(case)
    return merged


def questions_by_id(dataset):
    return {case['id']: case['question'] for cases in dataset.values() for case in cases}


def merge_runs(run_dirs, out_run, merged_questions_path, questions):
    out_run = Path(out_run)
    if out_run.exists():
        raise ValueError(f'이미 있습니다: {out_run}')
    records, sources, summaries = {}, [], []
    for run in map(Path, run_dirs):
        summary = read(run / 'summary.json')
        summaries.append(summary)
        count = 0
        for path in sorted((run / 'records').glob('*.json')):
            record = read(path)
            key = record['id']
            if key in records:
                raise ValueError(f'실행 기록 ID가 겹칩니다: {key} ({run})')
            if key not in questions or record.get('question') != questions[key]:
                raise ValueError(f'{key}: 합친 데이터셋에 없거나 질문 원문이 다릅니다 ({run})')
            records[key] = path
            count += 1
        sources.append({'run': run.as_posix(), 'records': count, 'summary_sha256': digest(run / 'summary.json'),
                        'success': summary.get('success'), 'error': summary.get('error')})
    missing = sorted(set(questions) - set(records))
    differing = [key for key in SETTING_KEYS if len({json.dumps(s.get(key), sort_keys=True) for s in summaries}) > 1]
    (out_run / 'records').mkdir(parents=True)
    results = []
    for key, path in sorted(records.items(), key=lambda kv: kv[0]):
        shutil.copyfile(path, out_run / 'records' / path.name)
        status = read(path).get('status')
        results.append({'id': key, 'status': status, 'file': f'records/{path.name}'})
    summary = {key: value for key, value in summaries[0].items() if key not in ('results', 'output_dir', 'attempt')}
    summary.update({
        'merged_from': sources, 'settings_differ': differing,
        'dataset': Path(merged_questions_path).as_posix(), 'dataset_sha256': digest(merged_questions_path),
        'started_at': min(s['started_at'] for s in summaries if s.get('started_at')),
        'finished_at': max(s['finished_at'] for s in summaries if s.get('finished_at')),
        'total': len(results), 'success': sum(r['status'] == 'success' for r in results),
        'error': sum(r['status'] != 'success' for r in results), 'missing_questions': missing, 'results': results,
        'note': '여러 실행의 records를 합친 폴더다. 점수는 합친 records 전체를 합친 정답으로 채점한 값이다.'})
    write(out_run / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', type=Path, nargs='+', required=True)
    parser.add_argument('--queries', type=Path, nargs='+', required=True)
    parser.add_argument('--out-questions', type=Path, required=True)
    parser.add_argument('--out-queries', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+')
    parser.add_argument('--out-run', type=Path)
    args = parser.parse_args()
    if bool(args.runs) != bool(args.out_run):
        parser.error('--runs와 --out-run은 함께 지정하세요.')
    questions, queries = merge_datasets(args.questions), merge_datasets(args.queries)
    q_text, p_text = questions_by_id(questions), questions_by_id(queries)
    if q_text != p_text:
        parser.error('questions와 queries의 질문 ID 또는 원문이 일치하지 않습니다.')
    for target, data in ((args.out_questions, questions), (args.out_queries, queries)):
        if not target.exists():
            write(target, data)
        elif read(target) != data:
            parser.error(f'이미 있고 내용이 다릅니다(덮어쓰지 않음): {target}')
    items = [i for cases in questions.values() for c in cases for i in c['answer']['items']]
    pairs = sum(len(i['source'] or []) for i in items)
    print(f"질문 {len(q_text)}개, 물품 {len(items)}개(근거 있음 {sum(bool(i['source']) for i in items)}), 정답 쌍 {pairs}개")
    if args.runs:
        summary = merge_runs(args.runs, args.out_run, args.out_questions, q_text)
        print(f"실행 기록 {summary['total']}개 합침(성공 {summary['success']}, 오류 {summary['error']}), "
              f"설정 차이: {summary['settings_differ'] or '없음'}, 기록 없는 질문: {summary['missing_questions'] or '없음'}")


if __name__ == '__main__':
    main()
