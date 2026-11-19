# -*- coding: utf-8 -*-
"""cross-encoder reranker의 검색 1건당 지연을 pool 크기·정밀도·최대 길이별로 잰다. LLM·OpenAI 호출 없음.

python -m scripts.evaluation.benchmark_reranker --runs tests/ver6_test/retrieval/country_airline_depth50/test9 \\
    --output tests/ver8_test/latency/mps_m1pro.json

각 설정을 한 번 예열한 뒤 같은 표본 검색들의 pool 앞 N개를 재정렬하는 시간을 잰다. 점수가 같은 설정과 달라졌는지
(fp16, 최대 길이 256)는 fp32·512 기준과 상위 5개 일치율로 확인한다.

설정 하나를 끝낼 때마다 출력 파일에 바로 저장한다. 중단 후 `--resume`으로 같은 명령을 다시 실행하면 저장된 설정은 건너뛴다
(상위 5개 일치율의 기준은 첫 설정이므로 첫 설정이 저장돼 있어야 이어할 수 있다). GPU 작업이 겹치지 않도록 실행 잠금을 쓴다.
"""
import argparse
import json
import platform
import statistics
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from sentence_transformers import CrossEncoder

from scripts.evaluation.rerank_common import RunLocked, gpu_lock, load_pools

MODEL = 'BAAI/bge-reranker-v2-m3'


def pick_device(name):
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--samples', type=int, default=4, help='표본 검색 수(질문 ID 순 균등 추출)')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=[10, 30, 50])
    parser.add_argument('--configs', nargs='+', default=['fp32:512:16', 'fp16:512:16'],
                        help='정밀도:최대길이:배치 크기. 앞의 fp32:512 설정이 점수 기준이다')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='출력 파일에 저장된 설정은 건너뛰고 이어서 측정')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error(f'이미 있습니다(이어하려면 --resume): {args.output}')
    device = pick_device(args.device)

    units = [u for u in load_pools(args.runs) if len(u['pool']) >= max(args.pool_sizes)]
    step = max(1, len(units) // args.samples)
    sample = units[::step][:args.samples]
    report = {'date': datetime.now().isoformat(timespec='seconds'), 'model': MODEL, 'device': device,
              'machine': platform.platform(), 'torch': torch.__version__, 'samples': len(sample),
              'pool_sizes': args.pool_sizes, 'configs': {}, 'reference_top5': {}}
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding='utf-8'))
        if saved.get('pool_sizes') != args.pool_sizes or saved.get('samples') != len(sample) or saved.get('device') != device:
            parser.error('저장된 측정과 pool 크기·표본 수·장치가 달라 이어할 수 없습니다.')
        report = saved
    pending = [spec for spec in args.configs if spec not in report['configs']]
    if pending and args.configs[0] not in report['configs'] and pending[0] != args.configs[0]:
        parser.error('기준 설정(첫 설정)이 저장돼 있어야 이어할 수 있습니다.')
    if not pending:
        print('측정할 설정이 없습니다(모두 저장돼 있음).')
        return 0

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(args.output)

    try:
        with gpu_lock():
            loaded = {}
            for spec in pending:
                dtype, max_length, batch = spec.split(':')
                max_length, batch = int(max_length), int(batch)
                if (dtype, max_length) not in loaded:
                    kwargs = {'torch_dtype': torch.float16} if dtype == 'fp16' else {}
                    loaded[(dtype, max_length)] = CrossEncoder(MODEL, device=device, max_length=max_length, model_kwargs=kwargs)
                model = loaded[(dtype, max_length)]
                warm = [(sample[0]['query'], d['text']) for d in sample[0]['pool'][:batch]]
                model.predict(warm, batch_size=batch, show_progress_bar=False)
                sync(device)
                result = {}
                for n in args.pool_sizes:
                    times, top5 = [], []
                    for unit in sample:
                        pairs = [(unit['query'], d['text']) for d in unit['pool'][:n]]
                        start = perf_counter()
                        scores = model.predict(pairs, batch_size=batch, show_progress_bar=False)
                        sync(device)
                        times.append(perf_counter() - start)
                        order = sorted(range(n), key=lambda i: -float(scores[i]))[:5]
                        top5.append([unit['pool'][i]['id'] for i in order])
                        if spec == args.configs[0]:
                            report['reference_top5'][f"{unit['qid']}|{unit['item']}|{n}"] = top5[-1]
                    entry = {'mean_s': statistics.mean(times), 'median_s': statistics.median(times), 'max_s': max(times),
                             'pairs_per_s': n / statistics.mean(times)}
                    entry['top5_same_as_reference'] = statistics.mean(
                        set(t) == set(report['reference_top5'][f"{u['qid']}|{u['item']}|{n}"]) for t, u in zip(top5, sample))
                    result[str(n)] = entry
                    print(f'{spec} N={n}: 평균 {entry["mean_s"]:.2f}s, 상위5 일치 {entry["top5_same_as_reference"]:.0%}', flush=True)
                report['configs'][spec] = result
                save()
                print(f'저장: {spec} -> {args.output}', flush=True)
    except RunLocked as error:
        print(error)
        return 2
    print('Saved:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
