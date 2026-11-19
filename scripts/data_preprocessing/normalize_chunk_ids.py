"""원본 청크를 ver1_로 보존하고 통일된 ID의 국가별 청크와 역추적 매핑을 만든다.

python -m scripts.data_preprocessing.normalize_chunk_ids --plan
python -m scripts.data_preprocessing.normalize_chunk_ids --apply
python -m scripts.data_preprocessing.normalize_chunk_ids --verify
LLM, 임베딩, Chroma를 호출하지 않는다.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE = Path(__file__).with_name('id_sources.json')
SOURCES = json.loads(SOURCES_FILE.read_text(encoding='utf-8'))
ACTIVE = {'KR': ('Korea', 'all_kr.jsonl'), 'CN': ('China', 'cn.jsonl'),
          'JP': ('Japan', 'jp.jsonl'), 'UK': ('UK', 'uk.jsonl'), 'US': ('US', 'us.jsonl')}
ID_PATTERN = re.compile(r'^[a-z]{2}\.[a-z][a-z0-9_]*\.\d{4,}$')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise ValueError(f'임시 파일이 이미 있습니다: {temporary}')
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def confined(path, root):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'작업 범위 밖 경로: {path}')
    return path


def namespace(country, document_id):
    matches = [(prefix, name) for prefix, name in SOURCES[country].items()
               if document_id.startswith(prefix)]
    if not matches:
        raise ValueError(f'출처 규칙이 없습니다. id_sources.json에 명시적으로 등록하세요: {document_id}')
    return max(matches, key=lambda pair: len(pair[0]))[1]


def content_without_ids(rows):
    result = deepcopy(rows)
    for row in result:
        del row['id']
        del row['metadata']['doc_id']
    return sha(json.dumps(result, ensure_ascii=False, sort_keys=True).encode('utf-8'))


def inventory(root):
    """한국 kr/law도 보존하되 변환 대상은 all_kr 하나로 한정한다."""
    files, inputs = [], {}
    for country, (folder, filename) in ACTIVE.items():
        directory = root / 'data' / folder
        names = [filename] + (['kr.jsonl', 'law.jsonl'] if country == 'KR' else [])
        unexpected = {p.name for p in (directory / 'chunk').iterdir() if p.is_file()} - set(names)
        if unexpected:
            raise ValueError(f'알 수 없는 chunk 파일을 먼저 확인하세요: {folder}: {sorted(unexpected)}')
        for name in names:
            target = confined(directory / 'chunk' / name, root)
            archive = confined(directory / 'preprocessed' / f'ver1_{name}', root)
            source = archive if archive.exists() else target
            if not source.exists():
                if name == filename:
                    raise ValueError(f'입력 청크가 없습니다: {source}')
                continue
            data = source.read_bytes()
            rows = read_rows(source)
            entry = {'country': country, 'original': str(target.relative_to(root)),
                     'archive': str(archive.relative_to(root)), 'sha256': sha(data),
                     'bytes': len(data), 'rows': len(rows), 'active': name == filename}
            files.append(entry)
            if name == filename:
                inputs[country] = rows
    return files, inputs


def build_mapping(inputs, existing=None):
    docs = {}
    chunk_ids = set()
    for country, rows in inputs.items():
        for row in rows:
            if set(row) != {'id', 'text', 'metadata'} or not isinstance(row['text'], str):
                raise ValueError('id/text/metadata의 원본 청크 형식이 필요합니다.')
            meta = row['metadata']
            old_doc = meta['doc_id']
            index = meta['chunk_index']
            if meta['country'] != country or type(index) is not int or index < 0:
                raise ValueError(f'잘못된 국가/청크 번호: {row["id"]}')
            if row['id'] != f'{old_doc}#{index}' or row['id'] in chunk_ids:
                raise ValueError(f'불일치하거나 중복된 청크 ID: {row["id"]}')
            chunk_ids.add(row['id'])
            docs.setdefault(old_doc, {'country': country, 'namespace': namespace(country, old_doc),
                                     'title': meta.get('title'), 'source_name': meta.get('source_name'),
                                     'org_name': meta.get('org_name'), 'source_url': meta.get('source_url'),
                                     'section_type': meta.get('section_type')})
    document_map = dict((existing or {}).get('document_id_map', {}))
    if set(document_map) - docs.keys():
        raise ValueError('매핑에만 있는 문서가 있습니다. 원본 세트 변경은 별도 버전으로 처리하세요.')
    used = defaultdict(set)
    for old, new in document_map.items():
        info = docs[old]
        prefix = f"{info['country'].lower()}.{info['namespace']}"
        if not ID_PATTERN.fullmatch(new) or not new.startswith(prefix + '.'):
            raise ValueError(f'기존 매핑이 출처 규칙과 다릅니다: {old}')
        number = int(new.rsplit('.', 1)[1])
        if number in used[prefix]:
            raise ValueError('문서 시리얼 충돌')
        used[prefix].add(number)
    grouped = defaultdict(list)
    for old, info in docs.items():
        if old not in document_map:
            grouped[f"{info['country'].lower()}.{info['namespace']}"].append(old)
    for prefix, ids in sorted(grouped.items()):
        # 기존 숫자 시리얼(CN/UK)은 유지한다. 새 문서는 저장 순서와 무관한 순서로 부여한다.
        for old in sorted(ids):
            suffix = re.search(r'\.(\d{4,})$', old)
            if suffix:
                number = int(suffix[1])
                if number in used[prefix]:
                    raise ValueError(f'기존 숫자 시리얼 충돌: {old}')
                document_map[old] = f'{prefix}.{number:04d}'
                used[prefix].add(number)
        for old in sorted(ids, key=lambda key: (docs[key]['section_type'] != 'body',
                                                re.sub(r'\d+', lambda m: m[0].zfill(8), key))):
            if old not in document_map:
                number = max(used[prefix], default=0) + 1
                document_map[old] = f'{prefix}.{number:04d}'
                used[prefix].add(number)
    chunk_map = {row['id']: f"{document_map[row['metadata']['doc_id']]}#{row['metadata']['chunk_index']}"
                 for rows in inputs.values() for row in rows}
    if len(set(chunk_map.values())) != len(chunk_map):
        raise ValueError('통일된 청크 ID 충돌')
    if existing and (existing['document_id_map'] != document_map or existing['chunk_id_map'] != chunk_map):
        raise ValueError('원본/매핑이 이전 실행과 다릅니다. 기존 매핑을 덮어쓰지 않습니다.')
    if any(not ID_PATTERN.fullmatch(new) for new in document_map.values()):
        raise ValueError('새 문서 ID는 country.source.0001 형식의 ASCII 코드여야 합니다.')
    return {'schema_version': 1, 'pattern': 'country.source.document_serial#chunk_index',
            'document_id_map': document_map,
            'reverse_document_id_map': {new: old for old, new in document_map.items()},
            'chunk_id_map': chunk_map, 'reverse_chunk_id_map': {new: old for old, new in chunk_map.items()},
            'documents': [dict(docs[old], old_doc_id=old, new_doc_id=new) for old, new in sorted(document_map.items())]}


def transform(rows, mapping):
    result = deepcopy(rows)
    for row in result:
        row['id'] = mapping['chunk_id_map'][row['id']]
        row['metadata']['doc_id'] = mapping['document_id_map'][row['metadata']['doc_id']]
    return result


def validate(inputs, outputs, mapping):
    countries = {}
    for country, before in inputs.items():
        after = outputs[country]
        if len(before) != len(after) or transform(before, mapping) != after:
            raise ValueError(f'{country}: ID 이외의 데이터 또는 순서가 변경됐습니다.')
        before_hash, after_hash = content_without_ids(before), content_without_ids(after)
        if before_hash != after_hash:
            raise ValueError(f'{country}: 본문/metadata 내용이 달라졌습니다.')
        countries[country] = {'before_rows': len(before), 'after_rows': len(after),
                              'before_documents': len({r['metadata']['doc_id'] for r in before}),
                              'after_documents': len({r['metadata']['doc_id'] for r in after}),
                              'content_without_ids_sha256': before_hash,
                              'preserved': True}
    return {'countries': countries, 'total_before': sum(map(len, inputs.values())),
            'total_after': sum(map(len, outputs.values())), 'duplicate_new_ids': 0,
            'changed_fields': ['id', 'metadata.doc_id'], 'all_checks_passed': True}


def run(root=ROOT, apply=False, verify=False):
    root = root.resolve()
    directory = root / 'data' / 'id_migration_v1'
    mapping_path = directory / 'mapping.json'
    files, inputs = inventory(root)
    existing = read_json(mapping_path) if mapping_path.exists() else None
    mapping = build_mapping(inputs, existing)
    if existing is not None and existing != mapping:
        raise ValueError('저장된 양방향 매핑/문서 정보가 원본에서 계산한 매핑과 다릅니다.')
    outputs = {country: transform(rows, mapping) for country, rows in inputs.items()}
    report = validate(inputs, outputs, mapping)
    if verify:
        if not existing:
            raise ValueError('검증할 mapping.json이 없습니다.')
        manifest = read_json(directory / 'manifest.json')
        for entry in manifest['archives']:
            path = root / entry['archive']
            if sha(path.read_bytes()) != entry['sha256']:
                raise ValueError(f'원본 보존 해시 불일치: {path}')
        actual = {country: read_rows(root / 'data' / folder / 'chunk' / filename)
                  for country, (folder, filename) in ACTIVE.items()}
        report = validate(inputs, actual, mapping)
    elif apply:
        # 이미 완료됐다면 내용을 검증하고 종료한다. 원본을 다시 이동하거나 번호를 바꾸지 않는다.
        if existing:
            return run(root, verify=True)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('불완전한 이전 마이그레이션 기록이 있습니다. 먼저 확인하세요.')
        for entry in files:
            archive = root / entry['archive']
            if archive.exists():
                raise ValueError(f'기존 보존 파일을 덮어쓰지 않습니다: {archive}')
        moved, created = [], []
        try:
            for entry in files:
                source, archive = root / entry['original'], root / entry['archive']
                archive.parent.mkdir(parents=True, exist_ok=True)
                # 계획을 읽은 이후 입력이 바뀌었는지 검사한다.
                if sha(source.read_bytes()) != entry['sha256']:
                    raise ValueError(f'작업 중 입력 변경: {source}')
                source.replace(archive)
                moved.append((source, archive))
            for country, rows in outputs.items():
                folder, filename = ACTIVE[country]
                target = root / 'data' / folder / 'chunk' / filename
                atomic_write(target, jsonl(rows))
                created.append(target)
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in [('mapping.json', mapping), ('manifest.json', {'archives': files}),
                                ('source_registry.json', SOURCES)]:
                target = directory / name
                if target.exists():
                    raise ValueError(f'기존 기록을 덮어쓰지 않습니다: {target}')
                atomic_write(target, encoded(value))
                created.append(target)
        except Exception:
            for target in reversed(created):
                confined(target, root).unlink()
            for source, archive in reversed(moved):
                confined(archive, root).replace(confined(source, root))
            raise
        report = run(root, verify=True)
    else:
        print(json.dumps({'mode': 'plan', 'archive_files': files,
                          'source_prefixes': sorted({new.rsplit('.', 1)[0] for new in mapping['document_id_map'].values()}),
                          'example_mappings': list(mapping['document_id_map'].items())[:8]}, ensure_ascii=False, indent=2))
    if apply or verify:
        report['files'] = [{'country': country,
                            'archive_bytes': (root / 'data' / folder / 'preprocessed' / f'ver1_{filename}').stat().st_size,
                            'output_bytes': (root / 'data' / folder / 'chunk' / filename).stat().st_size}
                           for country, (folder, filename) in ACTIVE.items()]
        report['archive_files_verified'] = len(files)
        report['document_ids'] = len(mapping['document_id_map'])
        report['chunk_ids'] = len(mapping['chunk_id_map'])
        atomic_write(directory / 'validation.json', encoded(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    report = run(apply=args.apply, verify=args.verify)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
