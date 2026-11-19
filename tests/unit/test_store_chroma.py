import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from scripts import store_chroma as loader


class LoaderTests(unittest.TestCase):
    def setUp(self):
        factory = patch.object(loader, 'create_embedding_function')
        factory.start()
        self.addCleanup(factory.stop)

    def test_existing_chunks_are_read_without_reprocessing(self):
        rows = [
            {'id': 'kr#0', 'text': '물품 | 기내 | 위탁\n배터리 | 조건부 | 금지',
             'metadata': {'country': 'KR', 'source_url': 'https://example.com/rule', 'effective_date': ''}},
            {'id': 'kr#1', 'text': '용량과 포장 조건',
             'metadata': {'country': 'KR', 'chunk_index': 1}},
        ]
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'all_kr.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
            original = path.read_bytes()
            with patch.object(loader, 'CHUNK_SOURCES', {'KR': path}):
                self.assertEqual(loader.load_chunks(), rows)
            self.assertEqual(path.read_bytes(), original)

    def test_upsert_preserves_ids_text_and_metadata_across_batches(self):
        rows = [{'id': f'kr#{i}', 'text': f'규정 {i}', 'metadata': {'country': 'KR', 'chunk_index': i}} for i in range(65)]
        with patch.object(loader.chromadb, 'PersistentClient') as client:
            collection = MagicMock()
            client.return_value.get_or_create_collection.return_value = collection
            collection.count.return_value = len(rows)
            self.assertEqual(loader.store_chunks(rows), len(rows))
        self.assertEqual(len(collection.upsert.call_args_list), 2)
        for field, key in [('ids', 'id'), ('documents', 'text'), ('metadatas', 'metadata')]:
            actual = [value for call in collection.upsert.call_args_list for value in call.kwargs[field]]
            self.assertEqual(actual, [row[key] for row in rows])

    def test_small_batch_keeps_tail_and_source_fields(self):
        rows = [{'id': f'kr#{i}', 'text': f'규정 {i}', 'metadata': {'country': 'KR'}} for i in range(19)]
        collection = MagicMock()
        collection.count.return_value = len(rows)
        self.assertEqual(loader.store_chunks(rows, collection, batch_size=8), len(rows))
        calls = collection.upsert.call_args_list
        self.assertEqual([len(call.kwargs['ids']) for call in calls], [8, 8, 3])
        for field, key in [('ids', 'id'), ('documents', 'text'), ('metadatas', 'metadata')]:
            self.assertEqual([v for call in calls for v in call.kwargs[field]], [r[key] for r in rows])

    def test_invalid_batch_fails_before_opening_database(self):
        with patch.object(loader.chromadb, 'PersistentClient') as client:
            for size in (0, -1):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    loader.store_chunks([], batch_size=size)
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
