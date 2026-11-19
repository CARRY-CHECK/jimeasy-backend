import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.data_preprocessing import normalize_chunk_ids as normalizer
from scripts.data_preprocessing import remap_test_sources as dataset_migration


def chunk(doc_id, index=0, country='KR'):
    return {'id': f'{doc_id}#{index}', 'text': '원문 본문과 공백  보존',
            'metadata': {'doc_id': doc_id, 'chunk_index': index, 'country': country,
                         'section_type': 'body', 'title': '문서 이름', 'source_url': 'https://example.org'}}


class IdMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        active = {'KR': ('Korea', 'all_kr.jsonl'), 'CN': ('China', 'cn.jsonl')}
        self.addCleanup(patch.stopall)
        patch.object(normalizer, 'ACTIVE', active).start()
        patch.object(dataset_migration, 'ACTIVE', active).start()
        self.rows = [chunk('kr.koreanair_restricted', 0), chunk('kr.koreanair_restricted', 1),
                     chunk('law.항공위험물운송기술기준.본문')]
        self.originals = {}
        for country, (folder, filename) in active.items():
            rows = self.rows if country == 'KR' else [chunk('cn.gov.0002', country='CN')]
            for name in [filename] + (['kr.jsonl', 'law.jsonl'] if country == 'KR' else []):
                path = self.root / 'data' / folder / 'chunk' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(normalizer.jsonl(rows))
                self.originals[path] = path.read_bytes()
            sentinel = self.root / 'data' / folder / 'preprocessed' / 'existing.jsonl'
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_bytes(b'existing preprocessing\r\n')
        self.dataset = {'category': [{'id': 'q1', 'question': '질문 원문', 'answer': {'items': [
            {'name': '배터리', 'carry_on': '불가', 'notes': None,
             'source': ['kr.koreanair_restricted', 'law.항공위험물운송기술기준.본문', 'cn.gov.0002']},
            {'name': '가위', 'source': None}]}}]}
        self.source = self.root / 'data/test/test2_questions.json'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(normalizer.encoded(self.dataset))

    def test_plan_is_read_only_and_project_root_is_correct(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = normalizer.run(self.root)
        self.assertEqual(result['total_before'], 4)
        self.assertFalse((self.root / 'data/id_migration_v1').exists())
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertTrue((normalizer.ROOT / 'config/setting.json').exists())

    def test_apply_archives_all_kr_inputs_but_only_writes_all_kr(self):
        result = normalizer.run(self.root, apply=True)
        self.assertEqual((result['total_before'], result['total_after']), (4, 4))
        for path, original in self.originals.items():
            archive = path.parent.parent / 'preprocessed' / f'ver1_{path.name}'
            self.assertEqual(archive.read_bytes(), original)
        korea = self.root / 'data/Korea'
        self.assertEqual([p.name for p in (korea / 'chunk').iterdir()], ['all_kr.jsonl'])
        self.assertEqual((korea / 'preprocessed/existing.jsonl').read_bytes(), b'existing preprocessing\r\n')
        mapping = normalizer.read_json(self.root / 'data/id_migration_v1/mapping.json')
        self.assertEqual(mapping['document_id_map']['cn.gov.0002'], 'cn.shanghai_gov.0002')
        self.assertEqual(mapping['document_id_map']['law.항공위험물운송기술기준.본문'], 'kr.dangerous_goods.0001')
        for old, new in mapping['chunk_id_map'].items():
            self.assertEqual(mapping['reverse_chunk_id_map'][new], old)
        after = normalizer.read_rows(korea / 'chunk/all_kr.jsonl')
        self.assertEqual(normalizer.content_without_ids(after), normalizer.content_without_ids(self.rows))

    def test_mapping_stable_for_reordered_input_and_repeated_apply(self):
        inputs = {'KR': self.rows, 'CN': [chunk('cn.gov.0002', country='CN')]}
        mapping = normalizer.build_mapping(inputs)
        reordered = {key: list(reversed(value)) for key, value in reversed(list(inputs.items()))}
        self.assertEqual(normalizer.build_mapping(reordered), mapping)
        first = normalizer.run(self.root, apply=True)
        saved = (self.root / 'data/id_migration_v1/mapping.json').read_bytes()
        self.assertEqual(normalizer.run(self.root, apply=True), first)
        self.assertEqual((self.root / 'data/id_migration_v1/mapping.json').read_bytes(), saved)

    def test_unknown_source_fails_before_any_move(self):
        path = self.root / 'data/China/chunk/cn.jsonl'
        path.write_bytes(normalizer.jsonl([chunk('cn.unknown.0001', country='CN')]))
        with self.assertRaisesRegex(ValueError, '출처 규칙'):
            normalizer.run(self.root, apply=True)
        self.assertTrue((self.root / 'data/Korea/chunk/kr.jsonl').exists())
        self.assertFalse((self.root / 'data/Korea/preprocessed/ver1_all_kr.jsonl').exists())

    def test_write_failure_restores_original_files(self):
        original_write = normalizer.atomic_write
        def fail_china(path, data):
            if path == self.root / 'data/China/chunk/cn.jsonl':
                raise OSError('simulated write failure')
            original_write(path, data)
        with patch.object(normalizer, 'atomic_write', side_effect=fail_china), self.assertRaises(OSError):
            normalizer.run(self.root, apply=True)
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((path.parent.parent / 'preprocessed' / f'ver1_{path.name}').exists())

    def test_verification_rejects_changed_body_and_reverse_mapping(self):
        normalizer.run(self.root, apply=True)
        path = self.root / 'data/Korea/chunk/all_kr.jsonl'
        original = path.read_bytes()
        rows = normalizer.read_rows(path)
        rows[0]['text'] = 'tampered'
        path.write_bytes(normalizer.jsonl(rows))
        with self.assertRaisesRegex(ValueError, 'ID 이외'):
            normalizer.run(self.root, verify=True)
        path.write_bytes(original)
        mapping_path = self.root / 'data/id_migration_v1/mapping.json'
        mapping = normalizer.read_json(mapping_path)
        mapping['reverse_document_id_map'] = {}
        mapping_path.write_bytes(normalizer.encoded(mapping))
        with self.assertRaisesRegex(ValueError, '양방향'):
            normalizer.run(self.root, verify=True)

    def test_separate_dataset_conversion_preserves_original_and_roundtrip(self):
        normalizer.run(self.root, apply=True)
        original = self.source.read_bytes()
        report = dataset_migration.run(self.root)
        self.assertEqual(report['before'], report['after'])
        self.assertEqual(report['before']['source_references'], 3)
        self.assertEqual(self.source.read_bytes(), original)
        output = self.source.with_name('test3_questions.json')
        converted = normalizer.read_json(output)
        self.assertEqual(converted['category'][0]['answer']['items'][0]['source'][0], 'kr.koreanair.0001')
        self.assertIsNone(converted['category'][0]['answer']['items'][1]['source'])
        saved = output.read_bytes()
        dataset_migration.run(self.root, verify=True)
        dataset_migration.run(self.root)
        self.assertEqual(output.read_bytes(), saved)

    def test_unmapped_or_changed_test_output_is_never_silently_overwritten(self):
        normalizer.run(self.root, apply=True)
        source = deepcopy(self.dataset)
        source['category'][0]['answer']['items'][0]['source'] = ['missing.doc']
        self.source.write_bytes(normalizer.encoded(source))
        with self.assertRaisesRegex(ValueError, '매핑되지 않은'):
            dataset_migration.run(self.root)
        output = self.source.with_name('test3_questions.json')
        self.assertFalse(output.exists())
        self.source.write_bytes(normalizer.encoded(self.dataset))
        output.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, '덮어쓰지'):
            dataset_migration.run(self.root)
        self.assertEqual(output.read_bytes(), b'{}')


if __name__ == '__main__':
    unittest.main()
