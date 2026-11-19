"""청크 변환과 별도로 test2의 source 문서 ID만 변환하여 test3를 만든다.

python -m scripts.data_preprocessing.remap_test_sources --apply
python -m scripts.data_preprocessing.remap_test_sources --verify
원본 테스트셋/매핑/청크를 수정하거나 LLM·Chroma를 호출하지 않는다.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from scripts.data_preprocessing.normalize_chunk_ids import (
    ROOT, ACTIVE, ID_PATTERN, atomic_write, encoded, read_json, read_rows, sha,
)


def items(dataset):
    if not isinstance(dataset, dict):
        raise ValueError('분류별 질문 배열이 필요합니다.')
    for category, questions in dataset.items():
        if not isinstance(questions, list):
            raise ValueError(f'{category}: 질문 배열이 필요합니다.')
        for question in questions:
            for item in question['answer']['items']:
                yield item


def counts(dataset):
    rows = list(items(dataset))
    return {
        'categories': len(dataset),
        'questions': sum(len(questions) for questions in dataset.values()),
        'items': len(rows),
        'source_references': sum(len(item['source']) for item in rows if isinstance(item.get('source'), list)),
        'null_sources': sum(item.get('source') is None for item in rows),
    }


def transform(dataset, mapping):
    result = deepcopy(dataset)
    documents = mapping['document_id_map']
    for item in items(result):
        sources = item.get('source')
        if sources is None:
            continue
        if not isinstance(sources, list) or any(not isinstance(key, str) for key in sources):
            raise ValueError('test2 source는 문서 ID 문자열 배열 또는 null이어야 합니다.')
        unknown = set(sources) - documents.keys()
        if unknown:
            raise ValueError(f'매핑되지 않은 정답 문서 ID: {sorted(unknown)}')
        item['source'] = [documents[key] for key in sources]
    return result


def run(root=ROOT, source=None, output=None, verify=False):
    root = Path(root).resolve()
    source = Path(source) if source else root / 'data/test/test2_questions.json'
    output = Path(output) if output else root / 'data/test/test3_questions.json'
    if source.resolve() == output.resolve():
        raise ValueError('원본과 출력 경로는 달라야 합니다.')
    mapping_path = root / 'data/id_migration_v1/mapping.json'
    mapping_bytes = mapping_path.read_bytes()
    mapping = read_json(mapping_path)
    forward, reverse = mapping['document_id_map'], mapping['reverse_document_id_map']
    if len(set(forward.values())) != len(forward) or reverse != {new: old for old, new in forward.items()}:
        raise ValueError('문서 ID 매핑이 양방향 일대일이 아닙니다.')
    if any(not ID_PATTERN.fullmatch(new) for new in forward.values()):
        raise ValueError('매핑의 새 문서 ID 형식이 잘못되었습니다.')
    source_bytes = source.read_bytes()
    before = read_json(source)
    expected = transform(before, mapping)
    after = read_json(output) if output.exists() else expected
    if verify and not output.exists():
        raise ValueError('검증할 출력 데이터셋이 없습니다.')
    if after != expected:
        raise ValueError('기존 출력이 매핑 결과와 다릅니다. 덮어쓰지 않습니다.')
    restored = transform(after, {'document_id_map': reverse})
    if restored != before or counts(before) != counts(after):
        raise ValueError('source ID 외 데이터·순서 또는 데이터 건수가 변경됐습니다.')
    active_docs = {row['metadata']['doc_id'] for folder, filename in ACTIVE.values()
                   for row in read_rows(root / 'data' / folder / 'chunk' / filename)}
    references = [key for item in items(after) for key in (item.get('source') or [])]
    if set(references) - active_docs:
        raise ValueError('test3의 문서 ID가 현재 청크에 없습니다.')
    if source.read_bytes() != source_bytes or mapping_path.read_bytes() != mapping_bytes:
        raise ValueError('처리 중 원본/매핑이 변경됐습니다.')
    if not verify and not output.exists():
        atomic_write(output, encoded(after))
    if read_json(output) != expected:
        raise ValueError('저장된 출력이 변환 결과와 다릅니다.')
    report = {
        'source': str(source.resolve()), 'output': str(output.resolve()),
        'source_sha256': sha(source_bytes), 'output_sha256': sha(output.read_bytes()),
        'mapping_sha256': sha(mapping_bytes), 'source_bytes': len(source_bytes),
        'output_bytes': output.stat().st_size, 'before': counts(before), 'after': counts(after),
        'unique_source_documents': len(set(references)),
        'changed_fields': ['answer.items[].source[]'],
        'roundtrip_matches_original': True, 'all_sources_exist_in_chunks': True,
        'original_bytes_preserved': source.read_bytes() == source_bytes, 'all_checks_passed': True,
    }
    atomic_write(root / 'data/id_migration_v1/test_dataset_validation.json', encoded(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--verify', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = run(source=args.input, output=args.output, verify=args.verify)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(encoded(report).decode('utf-8'))


if __name__ == '__main__':
    main()
