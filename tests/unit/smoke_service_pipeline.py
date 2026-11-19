# -*- coding: utf-8 -*-
"""서비스 경로를 LLM 없이 끝까지 점검하는 통합 점검(단위 테스트가 아니라 실제 모델·DB를 쓰므로 `tests/unit` 실행에는 포함되지 않는다). OpenAI 호출 없음(구조화·판정·문장 생성만 가짜로 대체).

python -m tests.unit.smoke_service_pipeline

실제로 도는 것: 설정 로딩, DB 점검, 임베딩, Hybrid 검색(국가+항공사 필터), reranker(장치 포함), 근거 ID 매핑·검증,
대화 도구 루프(structure_query → search_regulations 두 번 → finalize_answer), 응답 직렬화.
가짜인 것: 질의 구조화(parse_query), 판정 LLM(generate_structured_langchain), 문장 만들기(present), 에이전트 모델.
Windows 시연 PC에서 같은 명령으로 reranker 장치(cuda/cpu)·DB 점검·물품당 지연을 확인한다. 모델은 처음 실행에서 내려받는다.
실패한 점검이 있으면 종료 코드 1이다.
"""
import json
import sys
from time import perf_counter
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

CONDITIONS = {'departure': {'place': '인천', 'airline': '대한항공'}, 'arrival': {'place': '도쿄', 'airline': None},
              'travel_date': None, 'baggage_type': '위탁'}
ITEMS = [{'name': '보조배터리', 'details': None}, {'name': '전자담배', 'details': None}]
checks = []


def check(name, ok, detail=''):
    checks.append(ok)
    print(f"[{'통과' if ok else '실패'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


def fake_judge(system, user, schema, **kwargs):
    """제공된 문서 중 첫 번째 evidence_id를 근거로 고르는 가짜 판정. 사용자 프롬프트의 item_id를 읽어 응답한다."""
    from app.schemas.baggage import DecisionResponse, ItemDecision
    item_ids = []
    for part in user.split('"item_id"')[1:]:
        item_id = part.split('"')[1]
        if item_id not in item_ids:
            item_ids.append(item_id)
    return DecisionResponse(items=[ItemDecision(item_id=item_id, carry_on='조건부 가능', checked_baggage='불가', notes='스모크 점검용 설명',
                                                evidence_ids=[f'{item_id}_doc_1']) for item_id in item_ids])


def main():
    from app import rerank, search, settings
    from app.settings import DEFAULT_K, RETRIEVAL
    print('설정:', json.dumps({'default_k': DEFAULT_K, **RETRIEVAL}, ensure_ascii=False), flush=True)
    check('설정 값', (RETRIEVAL['method'], RETRIEVAL['candidate_k'], RETRIEVAL['rerank_pool'], DEFAULT_K, RETRIEVAL['airline_filter']) ==
          ('hybrid', 50, 30, 8, True), 'hybrid / 후보 50 / reranking 30 / 최종 8 / 항공사 필터')
    check('기본 구조화 프롬프트', settings.PARSE_SYSTEM_PROMPT == 'parse_system_en_ver2.txt', settings.PARSE_SYSTEM_PROMPT)

    start = perf_counter()
    search.warmup()
    check('사전 로딩과 DB 점검', True, f'{perf_counter() - start:.1f}초')
    problems = search.verify_database()
    check('DB 청크 수·필터용 metadata', not problems, problems or '문제 없음')
    device = str(rerank.get_reranker().device)
    check('reranker 장치', True, f'{device}' + (' (GPU가 있다면 CUDA 빌드 PyTorch를 확인하세요)' if device == 'cpu' else ''))

    # 1) 단발성 경로
    from app import rag
    parsed = {'conditions': CONDITIONS, 'items': ITEMS}
    start = perf_counter()
    with patch.object(rag, 'parse_query', return_value=json.loads(json.dumps(parsed))), \
            patch.object(rag, 'generate_structured_langchain', side_effect=fake_judge):
        result = rag.answer_question('대한항공으로 인천에서 도쿄 가는데 보조배터리와 전자담배를 위탁수하물에 넣어도 되나요?')
    elapsed = perf_counter() - start
    groups = result['groups']
    check('단발성: 물품 2개, 문서 8개씩', [len(g['documents']) for g in groups] == [8, 8], f'{elapsed:.1f}초(검색+reranking, 물품 2개)')
    scores = [[d['rerank_score'] for d in g['documents']] for g in groups]
    check('단발성: reranker 점수 내림차순', all(s == sorted(s, reverse=True) for s in scores))
    check('단발성: 근거 ID가 실제 청크로 매핑',
          all(len(item['source']) == 8 and all(s.get('chunk_id') for s in item['source']) for item in result['answer']['items']))
    airline_docs = {d['metadata']['org_name'] for g in groups for d in g['documents'] if d['metadata']['source_type'] == 'airline'}
    check('단발성: 항공사 필터(대한항공 외 항공사 문서 없음)', airline_docs <= {'대한항공'}, sorted(airline_docs))
    countries = {d['metadata']['country'] for g in groups for d in g['documents']}
    check('단발성: 국가 필터(KR/JP만)', countries <= {'KR', 'JP'}, sorted(countries))
    check('단발성: 결과에 검색 설정 기록과 JSON 직렬화', result['retrieval']['method'] == 'hybrid' and bool(json.dumps(result, ensure_ascii=False)))

    # 2) 대화 경로(도구 루프, 에이전트가 search_regulations를 두 번 호출)
    from app.chat import service, tools
    from app.chat.schemas import ChatRequest
    from app.chat.sessions import SessionStore
    calls = []
    real_search = tools.search_by_items

    def counted(*args, **kwargs):
        calls.append(1)
        return real_search(*args, **kwargs)

    def call(name, arguments=None):
        return AIMessage(content='', tool_calls=[dict(name=name, args=arguments or {}, id='call_' + name + str(len(calls)))])

    structure = dict(changes=[dict(field='departure.place', value='인천'), dict(field='departure.airline', value='대한항공'),
                              dict(field='arrival.place', value='도쿄')], mode='replace',
                     items=[dict(item_id=None, name=i['name'], details=None) for i in ITEMS],
                     question='보조배터리와 전자담배를 위탁수하물에 넣어도 되나요?', unknown_fields=[])
    replies = [call('structure_query', structure), call('search_regulations'), call('search_regulations'), call('finalize_answer')]
    model = Mock()
    model.bind_tools.return_value.invoke.side_effect = replies
    store = SessionStore()
    start = perf_counter()
    with patch.object(tools, 'search_by_items', side_effect=counted), patch.object(tools, 'parse_query', return_value=json.loads(json.dumps(parsed))), \
            patch('app.judgment.generate_structured_langchain', side_effect=fake_judge), \
            patch('app.chat.presenter.present', side_effect=lambda q, c, cards, g, **kw: (cards, '스모크 요약', None)):
        response = service.handle_tools(ChatRequest(message='대한항공으로 인천에서 도쿄 가는데 보조배터리와 전자담배 위탁 가능해?'),
                                        store, model=model)
    elapsed = perf_counter() - start
    check('대화: 카드 2개', [c.name for c in response.cards] == ['보조배터리', '전자담배'], f'{elapsed:.1f}초(에이전트·LLM 제외)')
    check('대화: 같은 상태의 재검색 생략(검색 1회)', len(calls) == 1, f'search_by_items 호출 {len(calls)}회')
    check('대화: 카드 근거(evidence) 포함', all(c.evidence for c in response.cards))
    dumped = response.model_dump_json()
    check('대화: 응답 직렬화(FE가 받는 JSON)', 'rerank_score' not in dumped and bool(dumped), f'{len(dumped)}자')
    print('통과 %d / 전체 %d' % (sum(checks), len(checks)), flush=True)
    return 0 if all(checks) else 1


if __name__ == '__main__':
    sys.exit(main())
