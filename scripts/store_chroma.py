"""국가별 전처리 완료 JSONL 청크를 그대로 로컬 Chroma에 저장한다."""
import argparse
import json

import chromadb
from app.integrations.embedding import create_embedding_function

from app.settings import ROOT, CHROMA_PATH, CHROMA_COLLECTION, CHUNK_SOURCES


def load_chunks():
    """설정된 청크 파일의 ID·본문·metadata를 변경하지 않고 읽는다."""
    rows = []
    for country, path in CHUNK_SOURCES.items():
        with path.open(encoding='utf-8-sig') as file:
            chunks = [json.loads(line) for line in file if line.strip()]
        rows.extend(chunks)
        print(f'{country}: {path} / {len(chunks)} chunks', flush=True)
    return rows


def store_chunks(rows, collection=None, *, batch_size=64):
    """model.json의 임베딩으로 적재하며 같은 ID는 upsert한다."""
    if batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if collection is None:
        client = chromadb.PersistentClient(path=str(ROOT / CHROMA_PATH))
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION, embedding_function=create_embedding_function(),
        )
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        collection.upsert(
            ids=[row['id'] for row in batch],
            documents=[row['text'] for row in batch],
            metadatas=[row['metadata'] for row in batch],
        )
        print(f'Chroma 적재: {min(start + batch_size, len(rows))}/{len(rows)}', flush=True)
    return collection.count()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Chunks per upsert/embedding call (default: 64; Mac MPS: 8).')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be a positive integer')
    rows = load_chunks()
    count = store_chunks(rows, batch_size=args.batch_size)
    print(f'저장 완료: {ROOT / CHROMA_PATH} / {CHROMA_COLLECTION} / {count}개', flush=True)


if __name__ == '__main__':
    main()
