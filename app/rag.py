"""질의 구조화 → 물품별 검색 → 공통 판정. 단발성 CLI 계약을 유지한다."""
from app.integrations.openai_client import generate_structured_langchain
from app.query_parser import parse_query
from app.settings import DEFAULT_K, FILTER_FIELDS
from app.retrieval import retrieve  # noqa: F401  (기존 import 경로 호환)
from app.search import active_retrieval, retrieve_for_item
from app.judgment import (AnswerValidationError, assemble_answer, build_items_prompt,
                          source_metadata, source_traces, judge_groups, NO_EVIDENCE)

NO_ITEMS = '확인 필요: 질문에서 반입하려는 물품을 찾지 못했습니다. 물품을 알려 주세요.'


def build_retrieval_query(item, conditions=None):
    """실제 검색과 임베딩 진단에서 동일한 검색 문자열을 사용한다."""
    conditions = conditions or {}
    travel = [conditions.get(leg, {}).get(field)
              for leg in ('departure', 'arrival') for field in ('place', 'airline')]
    travel += [conditions.get('travel_date'), conditions.get('baggage_type')]
    return ' '.join(p for p in (item['name'], item.get('details'), *travel) if p)


def search_by_items(items, filters=None, k=DEFAULT_K, conditions=None):
    """여행 조건은 검색어에 포함하고 출발/도착 place는 국가 필터에도 전달한다."""
    conditions = conditions or {}
    groups = []
    for item in items:
        query = build_retrieval_query(item, conditions)
        documents = [dict(doc, evidence_id=f"{item['item_id']}_doc_{index}")
                     for index, doc in enumerate(retrieve_for_item(query, filters, k, conditions=conditions), 1)]
        groups.append({'item': item, 'query': query, 'documents': documents})
    return groups


def answer_question(question, conditions=None, filters=None, k=DEFAULT_K):
    if not question.strip():
        raise ValueError('질문을 입력하세요.')
    if k < 1:
        raise ValueError('k must be at least 1.')
    if set(filters or {}) - FILTER_FIELDS:
        raise ValueError('지원 필터: ' + ', '.join(sorted(FILTER_FIELDS)))
    parsed = parse_query(question, conditions)
    for index, item in enumerate(parsed['items'], 1):
        item['item_id'] = f'item_{index}'
    groups = search_by_items(parsed['items'], filters, k, parsed['conditions'])
    try:
        result = judge_groups(question, parsed['conditions'], groups, filters, k,
                              generate=generate_structured_langchain)
    except AnswerValidationError as error:
        error.diagnostics['parsed'] = parsed
        raise
    return dict(result, parsed=parsed, retrieval=active_retrieval(), message=NO_ITEMS if not parsed['items'] else None)
