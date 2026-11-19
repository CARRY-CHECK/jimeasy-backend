# -*- coding: utf-8 -*-
"""저장된 검색 pool의 일부 순위 구간을 cross-encoder로 채점해 이어하기용 점수 파일에 덧붙인다. LLM·OpenAI 호출 없음.

python -m scripts.evaluation.score_rerank --runs tests/ver6_test/retrieval/country_airline_depth50/test9 \\
    --score-dir tests/ver8_test/scores --form full --rank-from 1 --rank-to 30

- pool은 Hybrid(RRF) 융합 순서이고 순위 구간(1부터, 양 끝 포함)만 채점한다. 구간을 나눠 실행해도 된다(1~30, 이어서 31~50).
- 점수는 물품 하나를 채점할 때마다 `<score-dir>/scores_<form>_<dtype>_<max_length>.jsonl`에 바로 저장한다.
  중단 후 같은 명령을 다시 실행하면 이미 채점한 물품·구간은 건너뛴다.
- GPU 작업이 겹치지 않도록 실행 잠금을 쓴다. 다른 작업이 실행 중이면 바로 종료한다.
- `--limit-items`로 앞의 몇 개 물품만 채점해 속도를 먼저 확인할 수 있다.
"""
import argparse
from pathlib import Path
from time import perf_counter

from scripts.evaluation.rerank_common import RunLocked, ScoreStore, gpu_lock, load_pools, unit_key

MODEL = 'BAAI/bge-reranker-v2-m3'
FORMS = {'full': ('전체 쿼리', 'query'), 'item': ('물품명+details', 'item_query')}


def pick_device(name):
    import torch
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def plan(units, store, form, rank_from, rank_to):
    """채점해야 할 (물품, 시작 순위, 끝 순위) 목록. 이미 이어진 구간과 pool 크기를 반영한다."""
    todo = []
    for unit in units:
        key = unit_key(unit)
        end = min(rank_to, len(unit['pool']))
        done = store.scored_until(key)
        start = max(rank_from, done + 1)
        if rank_from > done + 1:
            raise ValueError(f'{key}: 순위 {done + 1}~{rank_from - 1}이 아직 채점되지 않아 {rank_from}부터 시작할 수 없습니다. '
                             f'앞 구간부터 채점하세요.')
        if start <= end:
            todo.append((unit, start, end))
    return todo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--score-dir', required=True)
    parser.add_argument('--form', choices=list(FORMS), default='full')
    parser.add_argument('--rank-from', type=int, default=1)
    parser.add_argument('--rank-to', type=int, required=True)
    parser.add_argument('--limit-items', type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.rank_from < 1 or args.rank_to < args.rank_from:
        parser.error('순위 구간은 1 이상이고 --rank-to가 --rank-from 이상이어야 합니다.')

    units = load_pools(args.runs)
    if args.limit_items:
        units = units[:args.limit_items]
    store = ScoreStore(Path(args.score_dir) / f'scores_{args.form}_{args.dtype}_{args.max_length}.jsonl')
    todo = plan(units, store, args.form, args.rank_from, args.rank_to)
    pairs_total = sum(end - start + 1 for _, start, end in todo)
    print(f'물품 {len(units)}개 중 채점 필요 {len(todo)}개, 쌍 {pairs_total}개 (이미 채점한 물품·구간은 건너뜀)', flush=True)
    if not todo:
        return 0

    try:
        with gpu_lock():
            import torch
            from sentence_transformers import CrossEncoder
            device = pick_device(args.device)
            kwargs = {'torch_dtype': torch.float16} if args.dtype == 'fp16' else {}
            model = CrossEncoder(MODEL, device=device, max_length=args.max_length, model_kwargs=kwargs)
            started, done_pairs = perf_counter(), 0
            for number, (unit, start, end) in enumerate(todo, 1):
                pairs = [(unit[FORMS[args.form][1]], d['text']) for d in unit['pool'][start - 1:end]]
                scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
                store.append(unit_key(unit), start, scores, model=MODEL, form=args.form, dtype=args.dtype,
                             max_length=args.max_length, batch_size=args.batch_size, device=device)
                done_pairs += len(pairs)
                elapsed = perf_counter() - started
                rate = done_pairs / elapsed
                print(f'[{number}/{len(todo)}] {unit_key(unit)} 순위 {start}~{end} | {elapsed:.0f}s 경과, '
                      f'{rate:.1f}쌍/초, 남은 시간 약 {(pairs_total - done_pairs) / rate:.0f}s', flush=True)
    except RunLocked as error:
        print(error)
        return 2
    print('완료:', store.path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
