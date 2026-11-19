"""기존 Chroma dense 검색 + Kiwi BM25 검색을 가중 RRF로 결합한다.

기존 retrieve()/rag의 기본 검색 경로는 변경하지 않는다. 사용 예:

    from app.hybrid_retrieval import retrieve_hybrid, compare_retrieval

    docs = retrieve_hybrid(query, filters, k=10, conditions=conditions)
    comparison = compare_retrieval(query, filters, k=10, conditions=conditions)

query는 records의 groups[].query 등 기존 retrieve에 전달하던 문자열이다.
질의 구조화, 쿼리 확장, reranking, 답변 생성은 수행하지 않는다.
compare_retrieval은 dense를 한 번만 호출하고 각 검색 방식의 결과와 순위를 반환한다.
모듈 import만으로 모델 로딩·검색·DB 접근은 하지 않는다.

BM25 후보는 현재 Chroma 컬렉션에서 동일한 where 조건으로 읽는다.
IDF는 필터를 통과한 문서 집합 기준이다. BM25 점수 임계값은 적용하지 않는다.
DB가 비어 있으면 기존 retrieve의 초기 청크 적재 동작을 그대로 따른다.
"""
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import math

import chromadb

from app.retrieval import retrieve, retrieval_where
from app.settings import CHROMA_COLLECTION, CHROMA_PATH, DEFAULT_K, ROOT


_BM25_CACHE = OrderedDict()
_BM25_CACHE_SIZE = 8


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def kiwi_tokenize(text):
    """품사 제거 없이 형태소 token.form을 그대로 사용한다."""
    return [token.form for token in _kiwi().tokenize(text)]


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name}는 1 이상의 정수여야 합니다.')


def _bm25_index(documents, retriever_class):
    """현재 후보 ID·본문·metadata가 같을 때만 인메모리 인덱스를 재사용한다."""
    signature = hashlib.sha256(json.dumps(
        [(doc.metadata['chunk_id'], doc.page_content, doc.metadata) for doc in documents],
        ensure_ascii=False, sort_keys=True,
    ).encode('utf-8')).hexdigest()
    if signature not in _BM25_CACHE:
        _BM25_CACHE[signature] = retriever_class.from_documents(
            documents, preprocess_func=kiwi_tokenize,
            bm25_params={'k1': 1.5, 'b': 0.75},
        )
    _BM25_CACHE.move_to_end(signature)
    while len(_BM25_CACHE) > _BM25_CACHE_SIZE:
        _BM25_CACHE.popitem(last=False)
    return _BM25_CACHE[signature]


def _rows(documents):
    return [{'id': doc.metadata['chunk_id'], 'text': doc.page_content,
             'metadata': dict(doc.metadata)} for doc in documents]


def compare_retrieval(question, filters=None, k=DEFAULT_K, *, conditions=None,
                      candidate_k=50, weights=(0.5, 0.5), rrf_c=60, airline_scope=False):
    """같은 쿼리로 dense/BM25/hybrid를 비교할 결과를 반환한다.

    weights는 (BM25, dense) 순서다. candidate_k는 각 방식의 후보 수, k는 최종 출력 수다.
    후보가 k보다 적으면 있는 만큼 반환한다. Dense는 기존 retrieve를 그대로 사용한다.
    traces의 순위 None은 해당 후보 목록 밖이라는 뜻이며 전체 후보의 정확한 순위는 아니다.
    RRF 점수는 코사인 유사도가 아니다. 동점 순서는 EnsembleRetriever를 따른다.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError('질문을 입력하세요.')
    _positive_integer('k', k)
    _positive_integer('candidate_k', candidate_k)
    if candidate_k < k:
        raise ValueError('candidate_k는 최종 k 이상이어야 합니다.')
    if isinstance(rrf_c, bool) or not isinstance(rrf_c, int) or rrf_c < 0:
        raise ValueError('rrf_c는 0 이상의 정수여야 합니다.')
    if len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('weights는 (BM25, dense) 순서의 양수 두 개여야 합니다.')
    total_weight = sum(weights)
    if not math.isfinite(total_weight):
        raise ValueError('weights의 합은 유한해야 합니다.')
    weights = [w / total_weight for w in weights]
    where = retrieval_where(filters, conditions, airline_scope=airline_scope)

    # 의존성이 준비되지 않았으면 dense 실행/DB 초기화 전에 실패한다.
    from kiwipiepy import Kiwi  # noqa: F401
    import rank_bm25  # noqa: F401
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda

    dense_rows = retrieve(question, filters, candidate_k, conditions=conditions, airline_scope=airline_scope)
    collection = chromadb.PersistentClient(path=str(ROOT / CHROMA_PATH)).get_collection(
        name=CHROMA_COLLECTION,
    )
    get_options = {'include': ['documents', 'metadatas']}
    if where is not None:
        get_options['where'] = where
    corpus = collection.get(**get_options)
    documents = []
    for key, text, metadata in sorted(zip(
            corpus['ids'], corpus['documents'], corpus['metadatas']), key=lambda row: row[0]):
        if not isinstance(text, str):
            raise ValueError(f'BM25 후보에 본문이 없습니다: {key}')
        documents.append(Document(page_content=text, metadata={**(metadata or {}), 'chunk_id': key}))
    by_id = {doc.metadata['chunk_id']: doc for doc in documents}
    dense_docs = []
    for row in dense_rows:
        doc = by_id.get(row['id'])
        if doc is None or doc.page_content != row['text']:
            raise ValueError(f"Dense와 BM25 후보가 다릅니다. DB 변경 여부를 확인하세요: {row['id']}")
        dense_docs.append(doc)

    bm25_docs = []
    bm25_scores = {}
    # 빈 corpus 또는 전체 토큰이 없는 corpus에서 BM25의 0 나눗셈을 피한다.
    if documents and any(kiwi_tokenize(doc.page_content) for doc in documents):
        index = _bm25_index(documents, BM25Retriever)
        retriever = index.model_copy(update={'k': min(candidate_k, len(documents))})
        bm25_docs = retriever.invoke(question)
        bm25_scores = dict(zip(
            [doc.metadata['chunk_id'] for doc in documents],
            index.vectorizer.get_scores(kiwi_tokenize(question)).tolist(),
        ))

    # 이미 얻은 결과를 연결하여 dense 검색을 중복 실행하지 않는다.
    ensemble = EnsembleRetriever(
        retrievers=[RunnableLambda(lambda _: bm25_docs), RunnableLambda(lambda _: dense_docs)],
        weights=weights, c=rrf_c, id_key='chunk_id',
    )
    fused_docs = ensemble.invoke(question)
    dense_ranks = {doc.metadata['chunk_id']: rank for rank, doc in enumerate(dense_docs, 1)}
    bm25_ranks = {doc.metadata['chunk_id']: rank for rank, doc in enumerate(bm25_docs, 1)}
    traces = []
    for rank, doc in enumerate(fused_docs, 1):
        key = doc.metadata['chunk_id']
        dense_rank, bm25_rank = dense_ranks.get(key), bm25_ranks.get(key)
        traces.append({
            'chunk_id': key, 'dense_rank': dense_rank, 'bm25_rank': bm25_rank,
            'bm25_score': float(bm25_scores[key]) if key in bm25_scores else None,
            'rrf_score': sum(weight / (rrf_c + source_rank)
                             for weight, source_rank in zip(weights, (bm25_rank, dense_rank))
                             if source_rank is not None),
            'hybrid_rank': rank, 'in_hybrid_top_k': rank <= k,
        })
    return {
        'query': question, 'where': where, 'k': k, 'candidate_k': candidate_k,
        'weights': {'bm25': weights[0], 'dense': weights[1]}, 'rrf_c': rrf_c,
        'candidate_count': len(documents),
        'dense': _rows(dense_docs[:k]), 'bm25': _rows(bm25_docs[:k]),
        'hybrid': _rows(fused_docs[:k]),
        'dense_candidates': _rows(dense_docs), 'bm25_candidates': _rows(bm25_docs),
        'traces': traces,
    }


def retrieve_hybrid(question, filters=None, k=DEFAULT_K, *, conditions=None,
                    candidate_k=50, weights=(0.5, 0.5), rrf_c=60, airline_scope=False):
    """retrieve와 동일한 id/text/metadata 형식으로 hybrid 상위 k개를 반환한다."""
    return compare_retrieval(
        question, filters, k, conditions=conditions,
        candidate_k=candidate_k, weights=weights, rrf_c=rrf_c, airline_scope=airline_scope,
    )['hybrid']
