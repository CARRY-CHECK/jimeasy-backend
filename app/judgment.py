"""단발성·대화형이 공유하는 판정과 근거 검증. 검색·세션·출력은 호출자가 관리한다."""
import json
from string import Template

from app.integrations.openai_client import generate_structured_langchain
from app.query_parser import read_prompt
from app.schemas.baggage import BaggageItem, BaggageResponse, DecisionResponse
from app.settings import DEFAULT_K, JUDGE_SYSTEM_PROMPT

NO_EVIDENCE = '확인 필요: 해당 물품을 판단할 검색 근거가 없습니다.'


class AnswerValidationError(ValueError):
    """잘못된 판정의 원인 분석을 위한 검색·모델 출력 기록."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def build_items_prompt(question, conditions, filters, groups):
    system, template = read_prompt(JUDGE_SYSTEM_PROMPT, 'user_items.txt')
    context = [{'item': group['item'], 'documents': [
        {'evidence_id': doc['evidence_id'], 'text': doc['text'], 'metadata': doc['metadata']}
        for doc in group['documents']]} for group in groups]
    try:
        user = Template(template).substitute(
            question=question, conditions=json.dumps(conditions, ensure_ascii=False),
            filters=json.dumps(filters, ensure_ascii=False),
            context=json.dumps(context, ensure_ascii=False, indent=2))
    except (KeyError, ValueError):
        raise ValueError('user_items.txt의 변수는 $question, $conditions, $filters, $context만 사용할 수 있습니다.') from None
    return system.strip(), user


def assemble_answer(groups, response):
    """모델이 반환한 ID를 검증하고 원본 이름·출처를 코드에서 연결한다."""
    expected = {group['item']['item_id'] for group in groups}
    ids = [item.item_id for item in response.items]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('최종 판정의 물품 ID가 입력과 일치하지 않습니다 (누락·중복·추가).')
    decisions = {item.item_id: item for item in response.items}
    items = []
    for group in groups:
        item = group['item']
        decision = decisions[item['item_id']]
        documents = {doc['evidence_id']: doc for doc in group['documents']}
        if set(decision.evidence_ids) - documents.keys():
            raise ValueError(f"{item['item_id']}: 해당 물품에 제공되지 않은 근거 ID입니다.")
        evidence_ids = list(dict.fromkeys(decision.evidence_ids))
        if not evidence_ids:
            carry_on = checked = '확인 필요'
            notes = decision.notes or NO_EVIDENCE
        else:
            carry_on, checked = decision.carry_on, decision.checked_baggage
            notes = decision.notes
        items.append(BaggageItem(
            name=item['name'], carry_on=carry_on, checked_baggage=checked, notes=notes,
            source=[source_metadata(doc) for doc in group['documents']]))
    return BaggageResponse(items=items).model_dump()


def source_metadata(document):
    """문서 식별자와 metadata는 항상 검색 결과에서 가져온다."""
    return dict(document['metadata'], chunk_id=document['id'])


def source_traces(groups, response):
    """물품별 검색 후보와 모델이 사용했다고 보고한 출처를 분리한다.

    response는 assemble_answer에서 ID 검증을 마친 판정만 전달한다.
    """
    decisions = {item.item_id: item for item in response.items}
    searched, used = [], []
    for group in groups:
        item = group['item']
        identity = {'item_id': item['item_id'], 'name': item['name']}
        documents = {doc['evidence_id']: doc for doc in group['documents']}
        decision = decisions.get(item['item_id'])
        references = list(dict.fromkeys(decision.evidence_ids)) if decision else []
        searched.append(dict(identity, sources=[source_metadata(doc) for doc in group['documents']]))
        used.append(dict(identity, sources=[source_metadata(documents[key]) for key in references]))
    return searched, used


def judge_groups(question, conditions, groups, filters=None, k=DEFAULT_K, *, generate=None):
    """검증된 판정과 출처 진단을 반환한다. 문서 없는 물품은 모델 호출 없이 확인 필요."""
    generate = generate or generate_structured_langchain
    sources = [doc for group in groups for doc in group['documents']]
    searchable = [group for group in groups if group['documents']]
    response = DecisionResponse(items=[])
    if searchable:
        system, user = build_items_prompt(question, conditions, filters or {}, searchable)
        response = generate(system, user, DecisionResponse)
        try:
            answer = assemble_answer(searchable, response)
        except ValueError as error:
            raise AnswerValidationError(str(error), {
                'groups': groups, 'sources': sources,
                'searched_document_sources': source_traces(groups, DecisionResponse(items=[]))[0],
                'model_decisions': response.model_dump(), 'filters': filters or {}, 'k': k,
            }) from None
    else:
        answer = BaggageResponse(items=[]).model_dump()
    by_id = {group['item']['item_id']: result
             for group, result in zip(searchable, answer['items'])}
    for group in groups:
        if not group['documents']:
            by_id[group['item']['item_id']] = BaggageItem(
                name=group['item']['name'], carry_on='확인 필요', checked_baggage='확인 필요',
                notes=NO_EVIDENCE, source=[]).model_dump()
    answer = BaggageResponse.model_validate({'items': [by_id[group['item']['item_id']] for group in groups]}).model_dump()
    searched, used = source_traces(groups, response)
    return {'answer': answer, 'sources': sources, 'groups': groups,
            'searched_document_sources': searched, 'model_used_sources': used,
            'model_decisions': response.model_dump(),
            'filters': filters or {}, 'k': k}
