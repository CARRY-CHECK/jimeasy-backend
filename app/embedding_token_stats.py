"""최종 국가별 청크의 임베딩 입력 토큰 통계. API/Chroma/벡터 생성 없음.

프로젝트 루트에서: python -m app.embedding_token_stats
embedding.py로 현재 SentenceTransformer를 로드하고 실제 전처리기를 사용한다.
"""
import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from app.integrations.embedding import create_embedding_function
from app.settings import CHUNK_SOURCES, KST, MODEL_CONFIG, RESULTS_DIR


def statistics(values):
    if not values:
        return {'count': 0, 'total': 0, 'mean': None, 'min': None, 'max': None,
                'median': None, 'std_population': None, 'p25': None, 'p75': None, 'p90': None, 'p95': None, 'p99': None}
    numbers = np.asarray(values, dtype=np.int64)
    result = {'count': len(values), 'total': int(numbers.sum()), 'mean': float(numbers.mean()),
              'min': int(numbers.min()), 'max': int(numbers.max()), 'median': float(np.median(numbers)),
              'std_population': float(numbers.std(ddof=0))}
    result.update({f'p{p}': float(np.percentile(numbers, p)) for p in [25, 75, 90, 95, 99]})
    return result


def summarize(rows):
    minimum = min((r['raw_input_tokens'] for r in rows), default=None)
    maximum = max((r['raw_input_tokens'] for r in rows), default=None)
    return {
        'chunks': len(rows), 'documents': len({r['doc_id'] for r in rows}),
        **{field: statistics([r[field] for r in rows]) for field in
           ['content_tokens', 'raw_input_tokens', 'effective_input_tokens', 'removed_tokens']},
        'truncated_chunks': sum(r['removed_tokens'] > 0 for r in rows),
        'empty_text_chunks': sum(r['characters'] == 0 for r in rows),
        'min_chunk_ids': [r['chunk_id'] for r in rows if r['raw_input_tokens'] == minimum],
        'max_chunk_ids': [r['chunk_id'] for r in rows if r['raw_input_tokens'] == maximum],
    }


def count_batch(model, texts):
    # Same text preprocessing as the embedding model, but no forward/encode call.
    raw = model.preprocess(inputs=texts, processing_kwargs={
        'text': {'padding': False, 'truncation': False}, 'common': {'return_tensors': None}})
    effective = model.preprocess(inputs=texts)
    raw_lengths = [sum(mask) for mask in raw['attention_mask']]
    effective_lengths = effective['attention_mask'].sum(dim=1).tolist()
    # Exclude model-added special tokens explicitly, instead of assuming their count.
    contents = model.preprocess(inputs=texts, processing_kwargs={
        'text': {'padding': False, 'truncation': False, 'add_special_tokens': False},
        'common': {'return_tensors': None}})
    content_lengths = [sum(mask) for mask in contents['attention_mask']]
    return content_lengths, raw_lengths, effective_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='새 결과 폴더 (기존 폴더 덮어쓰기 금지)')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size는 1 이상이어야 합니다.')
    if MODEL_CONFIG['embedding']['provider'] != 'sentence_transformer':
        parser.error('이 통계 도구는 현재 sentence_transformer 임베딩 설정을 지원합니다.')
    chunks, inputs, seen = [], {}, set()
    for country, path in CHUNK_SOURCES.items():
        raw_bytes = path.read_bytes()
        inputs[country] = {'path': str(path), 'sha256': hashlib.sha256(raw_bytes).hexdigest()}
        for line in raw_bytes.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in seen:
                raise ValueError(f"중복 청크 ID: {row['id']}")
            if row['metadata'].get('country') != country or not isinstance(row['text'], str):
                raise ValueError(f"청크 국가/본문 오류: {row['id']}")
            seen.add(row['id'])
            chunks.append((country, path, row))
    function = create_embedding_function()
    model = function._model
    if getattr(model, 'default_prompt_name', None):
        raise ValueError('기본 임베딩 프롬프트가 있는 모델은 별도 전처리 검증이 필요합니다.')
    rows = []
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start:start + args.batch_size]
        content, raw, effective = count_batch(model, [row['text'] for _, _, row in batch])
        for (country, path, chunk), c, r, e in zip(batch, content, raw, effective):
            if e > r:
                raise ValueError('실제 입력 길이가 원문 길이보다 큽니다. 전처리 설정 확인 필요.')
            rows.append({'country': country, 'chunk_id': chunk['id'], 'doc_id': chunk['metadata']['doc_id'],
                         'source_name': chunk['metadata'].get('source_name', ''), 'path': str(path),
                         'characters': len(chunk['text']), 'content_tokens': int(c), 'raw_input_tokens': int(r),
                         'effective_input_tokens': int(e), 'removed_tokens': int(r-e)})
        print(f'토큰 집계: {len(rows)}/{len(chunks)}', flush=True)
    report = {'created_at': datetime.now(KST).isoformat(), 'embedding': MODEL_CONFIG['embedding'],
              'tokenizer_class': type(model.tokenizer).__name__, 'max_seq_length': model.max_seq_length,
              'versions': {name: version(name) for name in ['sentence-transformers', 'transformers', 'chromadb']},
              'definitions': {'content_tokens': '특수 토큰 제외, 잘림 전 본문',
                              'raw_input_tokens': '특수 토큰 포함, 잘림 전 입력',
                              'effective_input_tokens': '모델의 실제 전처리 후 입력. 특수 토큰 포함, padding 제외',
                              'removed_tokens': 'raw_input_tokens - effective_input_tokens',
                              'std_population': '모집단 표준편차 ddof=0',
                              'percentiles': 'NumPy linear percentile; 청크별 길이 기준',
                              'scope': '설정된 최종 chunk JSONL의 text만 집계. 국가별 합계를 더한 총합이며 중복 제거하지 않음. OpenAI 토큰/과금과 무관.'},
              'inputs': inputs, 'countries': {country: summarize([r for r in rows if r['country'] == country])
                                            for country in CHUNK_SOURCES}, 'total': summarize(rows)}
    assert sum(v['raw_input_tokens']['total'] for v in report['countries'].values()) == report['total']['raw_input_tokens']['total']
    output = args.output or RESULTS_DIR / f'embedding_token_stats_{datetime.now(KST):%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'chunks.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['country', 'chunk_id', 'doc_id', 'source_name', 'path', 'characters',
                                                'content_tokens', 'raw_input_tokens', 'effective_input_tokens', 'removed_tokens'])
        writer.writeheader(); writer.writerows(rows)
    md = ['# 임베딩 토큰 통계', '',
          'BGE-M3 모델 입력 기준: 특수 토큰 포함, padding 제외. 원문은 길이 제한 적용 전이며 실제 입력은 전처리 후 길이입니다.', '',
          '| 국가 | 청크 수 | 원문 총 토큰 | 평균 | 최소 | 최대 | 중앙값 | P95 | 실제 입력 총 토큰 | 잘린 청크 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for country, entry in [*report['countries'].items(), ('전체', report['total'])]:
        s = entry['raw_input_tokens']
        md.append(f"| {country} | {entry['chunks']:,} | {s['total']:,} | {s['mean']:.2f} | {s['min']} | {s['max']} | {s['median']:.2f} | {s['p95']:.2f} | {entry['effective_input_tokens']['total']:,} | {entry['truncated_chunks']} |")
    md += ['', f"모델 최대 입력 길이: {model.max_seq_length}. 상세 청크 길이는 chunks.csv, 추가 분위수·표준편차·특수 토큰 제외 통계는 summary.json 참조.", '']
    (output / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md), flush=True)
    print(f'저장됨: {output}')


if __name__ == '__main__':
    main()
