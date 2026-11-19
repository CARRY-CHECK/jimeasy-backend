"""서비스의 물품별 문서 검색. 설정(config/setting.json의 retrieval)에 따라 방식을 고른다.

- hybrid(기본): Dense 50 + Kiwi BM25 50을 RRF로 합치고(국가+항공사 필터) 상위 rerank_pool개를 cross-encoder로 재정렬해
  최종 k개를 반환한다. rerank_pool이 0이면 Hybrid 순서의 상위 k개를 그대로 쓴다.
- dense: 이전 방식(국가 필터, Dense 상위 k개). 과거 평가 기준선 재현용이다.
"""
import threading
from contextlib import contextmanager
from contextvars import ContextVar

import chromadb

from app.retrieval import retrieve
from app.settings import CHROMA_COLLECTION, CHROMA_PATH, CHUNK_SOURCES, DEFAULT_K, RETRIEVAL, ROOT

_OVERRIDE = ContextVar('retrieval_method_override', default=None)
# Hybrid 검색은 프로세스 안의 BM25 인덱스 캐시와 Kiwi 인스턴스를 공유한다. API 요청이 스레드풀에서 동시에 실행될 수 있어
# 후보 검색은 한 번에 하나만 실행한다(reranker도 별도 락으로 직렬화되므로 처리량 손실은 작다).
_RETRIEVAL_LOCK = threading.Lock()


def active_retrieval():
    """현재 적용 중인 검색 설정(실험 실행의 override 반영). 결과 기록에 남기는 용도로도 쓴다."""
    config = dict(RETRIEVAL)
    if _OVERRIDE.get() == 'dense':
        config['method'] = 'dense'
    return config


@contextmanager
def use_retrieval_method(name):
    """실험 실행 범위에서만 검색 방식을 바꾼다('config'는 설정 그대로, 'dense'는 이전 방식)."""
    if name not in ('config', 'dense'):
        raise ValueError("검색 방식은 'config' 또는 'dense'여야 합니다.")
    token = _OVERRIDE.set(None if name == 'config' else name)
    try:
        yield
    finally:
        _OVERRIDE.reset(token)


def retrieve_for_item(query, filters=None, k=DEFAULT_K, conditions=None, *, trace=None):
    """물품 하나의 검색 쿼리에 대한 최종 문서 k개를 반환한다. 문서는 {'id','text','metadata'} 형태다.

    trace에 dict를 주면 reranking 전 후보 순서를 trace['pool_ids']에 기록한다(평가·진단용, 반환값은 같다).
    """
    if not query.strip() or k < 1:
        raise ValueError('질문을 입력하고 k를 1 이상으로 설정하세요.')
    config = active_retrieval()
    if config['method'] == 'dense':
        documents = retrieve(query, filters, k, conditions=conditions)
        if trace is not None:
            trace['pool_ids'] = [doc['id'] for doc in documents]
        return documents
    from app.hybrid_retrieval import compare_retrieval
    pool = max(config.get('rerank_pool') or 0, k)
    # compare_retrieval은 최종 개수가 방식별 후보 수 이하여야 한다. 설정보다 큰 k(CLI --k)는 후보 수를 함께 늘린다.
    candidate_k = max(config['candidate_k'], pool)
    with _RETRIEVAL_LOCK:
        result = compare_retrieval(query, filters, pool, conditions=conditions, candidate_k=candidate_k,
                                   airline_scope=config.get('airline_filter', False))
    documents = result['hybrid']
    if trace is not None:
        trace['pool_ids'] = [doc['id'] for doc in documents[:pool]]
    if not config.get('rerank_pool'):
        return documents[:k]
    from app.rerank import rerank
    return rerank(query, documents, k)


# 필터가 의존하는 metadata. 없으면 국가·항공사 필터가 문서를 조용히 지운다.
REQUIRED_METADATA = ('country', 'source_type', 'org_name')


def verify_database():
    """DB가 현재 청크 파일과 맞는지 점검해 문제 목록을 반환한다(문제가 없으면 빈 목록).

    다른 장치(예: Windows)에서 따로 적재한 DB가 오래됐거나 일부만 적재됐을 때를 드러내기 위한 점검이다.
    청크 개수와 필터용 metadata 필드를 확인하며 임베딩 값 자체는 비교하지 않는다.
    """
    collection = chromadb.PersistentClient(path=str(ROOT / CHROMA_PATH)).get_collection(CHROMA_COLLECTION)
    data = collection.get(include=['metadatas'])
    expected = sum(1 for path in CHUNK_SOURCES.values()
                   for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip())
    problems = []
    if len(data['ids']) != expected:
        problems.append(f'DB 청크 수 {len(data["ids"])}개가 청크 파일 {expected}개와 다릅니다. 다시 적재하세요.')
    missing = {field: sum(1 for meta in data['metadatas'] if not (meta or {}).get(field)) for field in REQUIRED_METADATA}
    for field, count in missing.items():
        if count:
            problems.append(f'DB 청크 {count}개에 필터용 metadata `{field}`가 없습니다. 최신 청크 파일로 다시 적재하세요.')
    return problems


def warmup():
    """임베딩·reranker 모델을 미리 불러오고 DB를 점검한다. 설정이 dense면 임베딩만 불러온다."""
    retrieve('warmup', k=1)
    problems = verify_database()
    if problems:
        raise RuntimeError(' '.join(problems))
    config = active_retrieval()
    if config['method'] == 'hybrid' and config.get('rerank_pool'):
        from app.rerank import get_reranker
        get_reranker()
