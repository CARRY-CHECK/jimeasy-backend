"""대화 상태와 bind_tools 실행 루프. legacy 경로는 전환 중 비교/롤백용이다."""
import json
from copy import deepcopy
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import APIConnectionError, APIStatusError

from app.chat.conditions import merge_conditions
from app.chat.presenter import UNKNOWN_OPTION, present
from app.chat.schemas import Card, ChatResponse, Verdict
from app.chat.sessions import SessionStore
from app.integrations.openai_client import generate_structured_langchain, create_chat_model, collect_usage
from app.query_parser import parse_query, read_prompt
from app.rag import search_by_items
from app.judgment import judge_groups
from app.settings import AGENT_SYSTEM_PROMPT, DEFAULT_K, MODEL_CONFIG

ASK_ITEM = '어떤 물품을 가져가실지 알려 주세요. 예: 보조배터리 기내에 가져가도 돼?'
ASK_ANSWER = '위 질문에 답해 주시거나, 궁금한 물품을 알려 주세요.'
NO_EVIDENCE = '자료에서 찾지 못했어요. 이용하실 항공사에 확인해 주세요.'
STALE_CHOICE = '선택지가 현재 대화와 맞지 않습니다. 화면을 새로 고쳐 주세요.'
YES_NO = ('네', '예', '응', '아니', '아뇨')
UNKNOWN_WORD = '모르'

store = SessionStore()


class ChatInputError(ValueError):
    """사용자 입력 문제(빈 질문, 지난 선택지). 모델·검색 오류와 구분한다."""


def handle(request, sessions=store):
    if MODEL_CONFIG['chat']['engine'] == 'tools':
        return handle_tools(request, sessions)
    if MODEL_CONFIG['chat']['engine'] != 'legacy':
        raise RuntimeError('chat.engine은 tools 또는 legacy여야 합니다.')
    return handle_legacy(request, sessions)


def handle_legacy(request, sessions=store):
    """이전 카드 대화 경로. 전환 기간의 회귀 비교/롤백용."""
    session_id, session = sessions.get(request.session_id)
    if request.choice:
        return answer_choice(session_id, session, request.choice)
    return answer_message(session_id, session, request.message.strip())


def handle_tools(request, sessions=store, *, model=None):
    """요청별 도구 루프. 사용자 대기/최종 응답에서 종료하며 성공한 상태만 반영한다."""
    from app.chat.tools import ToolRunner
    session_id = request.session_id or uuid4().hex
    with sessions.lock_for(session_id):
        _, original = sessions.get(session_id)
        session = deepcopy(original)
        session.turn_id = uuid4().hex
        if request.choice:
            choice = request.choice
            if not session.pending or choice.question_id != session.pending['id']:
                raise ChatInputError(STALE_CHOICE)
            try:
                row = session.pending['questions'][choice.card]
                label = row['options'][choice.option]
            except IndexError:
                raise ChatInputError(STALE_CHOICE) from None
            message = f"{row['question']} {label}"
            if label == UNKNOWN_OPTION:
                session.unknown_fields.add(row['field'])
        else:
            message = request.message.strip()
        if not message:
            raise ChatInputError('질문을 입력하세요.')
        runner = ToolRunner(session_id, session, message=message)
        prompt, = read_prompt(AGENT_SYSTEM_PROMPT)
        messages = [SystemMessage(content=prompt), SystemMessage(content='현재 상태(JSON 데이터):\n' +
                    json.dumps(session.snapshot(), ensure_ascii=False))]
        settings = MODEL_CONFIG['chat']
        for row in session.history[-settings['history_messages']:]:
            cls = HumanMessage if row['role'] == 'user' else AIMessage
            messages.append(cls(content=row['content'][:4000]))
        messages.append(HumanMessage(content=message))
        start, events = perf_counter(), []
        with collect_usage() as usage:
            bound = (model or create_chat_model('agent')).bind_tools(
                runner.tools, strict=True, parallel_tool_calls=False, tool_choice='required')
            for step in range(settings['max_steps']):
                try:
                    reply = bound.invoke(messages, store=False)
                except APIStatusError as error:
                    raise RuntimeError(f'대화 모델 요청 실패 (HTTP {error.status_code}).') from None
                except APIConnectionError:
                    raise RuntimeError('대화 모델 연결에 실패했습니다.') from None
                if not isinstance(reply, AIMessage):
                    raise RuntimeError('대화 모델에서 유효한 메시지를 받지 못했습니다.')
                messages.append(reply)
                calls = reply.tool_calls
                if not calls:
                    # 모델의 자유 텍스트를 규정 판정으로 노출하지 않는다.
                    messages.append(HumanMessage(content='도구를 하나 호출하세요. 직접 판정하지 마세요.'))
                    events.append(dict(status='error', reason='missing_tool_call'))
                    continue
                for call in calls:
                    if len(calls) != 1:
                        result = dict(error='의존성이 있는 도구는 한 번에 하나씩 호출하세요.')
                    else:
                        try:
                            result = runner.execute(call['name'], call['args'])
                        except ValueError as error:
                            result = dict(error=str(error)[:1000])
                    event = dict(tool=call['name'], status='error' if 'error' in result else 'success')
                    if 'error' in result:
                        event['reason'] = result['error']
                    events.append(event)
                    messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call['id']))
                if runner.response is not None:
                    response = runner.response
                    diagnostics = dict(engine='tools', revision=session.revision,
                                       search_revision=session.search_revision, agent_calls=step + 1,
                                       model_calls=list(usage), tool_events=events,
                                       seconds=round(perf_counter() - start, 3))
                    diagnostics['parsed_query'] = runner.parsed
                    response.turn_id, response.diagnostics = session.turn_id, diagnostics
                    # 카드 판정과 질문을 보존하되 검색 본문을 매 턴 기록에 재주입하지 않는다.
                    visible = response.model_dump(exclude={'diagnostics', 'conditions', 'turn_id', 'session_id'})
                    for card in visible['cards']:
                        card.pop('evidence', None)
                    session.history.extend([dict(role='user', content=message),
                                            dict(role='assistant', content=json.dumps(visible, ensure_ascii=False))])
                    session.last_diagnostics = diagnostics
                    sessions.put(session_id, session)
                    return response
        # 실패한 요청의 부분 상태를 저장하지 않는다. 자동 재시도는 다음 요청에서 가능하다.
        raise RuntimeError('대화 처리 호출 상한에 도달했습니다. 조건을 구체적으로 적어 다시 질문해 주세요.')


def answer_message(session_id, session, message):
    """새 물품 → 검색부터, 조건만 변경 → 직전 물품 재검색, 되묻기 답변 → 판정만 다시."""
    if not message:
        raise ChatInputError('질문을 입력하세요.')
    parsed = parse_query(message)
    conditions = merge_conditions(session.conditions, parsed['conditions'])
    pending = [card for card in session.cards if card.follow_up]
    if parsed['items']:
        question, items = message, parsed['items']
    elif conditions != session.conditions and session.groups:
        question, items = session.question, [group['item'] for group in session.groups]
    elif pending and UNKNOWN_WORD in message:
        return close_follow_ups(session_id, session, [card.id for card in pending])
    elif answered := answered_cards(message, pending):
        return rejudge(session_id, session, {card.id: f'{card.follow_up.question} {message}' for card in answered})
    else:
        session.conditions = conditions
        return ChatResponse(session_id=session_id, conditions=conditions,
                            message=ASK_ANSWER if pending else ASK_ITEM)
    items = [dict(item, item_id=f'item_{index}') for index, item in enumerate(items, 1)]
    groups = search_by_items(items, k=DEFAULT_K, conditions=conditions)
    cards, summary, suggested = present(question, conditions, judge(question, conditions, groups), groups)
    session.question, session.conditions, session.groups, session.cards = question, conditions, groups, cards
    return ChatResponse(session_id=session_id, conditions=conditions, summary=summary,
                        cards=cards, suggested_next=suggested)


def answered_cards(message, cards):
    """직접 입력이 어느 카드의 되묻기 답인지 고른다. 선택지 단어, 숫자 기준, 예/아니요로 판단한다."""
    text = message.replace(' ', '')
    has_digit = any(char.isdigit() for char in text)
    answered = []
    for card in cards:
        options = [option for option in card.follow_up.options if option != UNKNOWN_OPTION]
        words = {word for option in options for word in option.split() if len(word) >= 2}
        words |= {option.replace(' ', '') for option in options}
        numeric = any(char.isdigit() for option in options for char in option)
        binary = any(option.startswith(YES_NO) for option in options)
        if (any(word in text for word in words) or (has_digit and numeric)
                or (binary and text.startswith(YES_NO))):
            answered.append(card)
    return answered


def answer_choice(session_id, session, choice):
    try:
        card = session.cards[choice.card]
        label = card.follow_up.options[choice.option]
    except (IndexError, AttributeError):
        raise ChatInputError(STALE_CHOICE) from None
    if label == UNKNOWN_OPTION:
        return close_follow_ups(session_id, session, [card.id])
    return rejudge(session_id, session, {card.id: f'{card.follow_up.question} {label}'})


def close_follow_ups(session_id, session, card_ids):
    """모른다는 답은 판정을 유지하고 되묻기만 닫는다."""
    for card_id in card_ids:
        session.cards[card_id] = session.cards[card_id].model_copy(update={'follow_up': None})
    return ChatResponse(session_id=session_id, conditions=session.conditions,
                        cards=[session.cards[card_id] for card_id in card_ids])


def rejudge(session_id, session, answers):
    """재검색 없이 저장된 문서로 판정만 다시 하고 해당 카드만 교체한다. answers: {카드 id: 답변}."""
    groups = {card_id: with_detail(session.groups[card_id], answer) for card_id, answer in answers.items()}
    question, conditions = session.question, session.conditions
    cards = judge(question, conditions, list(groups.values()), list(groups))
    cards, summary, suggested = present(question, conditions, cards, list(groups.values()))
    for card_id, card in zip(groups, cards):
        session.groups[card_id], session.cards[card_id] = groups[card_id], card
    return ChatResponse(session_id=session_id, conditions=conditions, summary=summary,
                        cards=cards, suggested_next=suggested)


def with_detail(group, detail):
    item = group['item']
    details = ', '.join(part for part in (item.get('details'), detail) if part)
    return dict(group, item=dict(item, details=details))


def judge(question, conditions, groups, card_ids=None):
    """rag.answer_question의 판정 단계와 같은 순서. 사용한 근거가 없는 카드는 고정 안내로 바꾼다."""
    judged = judge_groups(question, conditions, groups, generate=generate_structured_langchain)
    results = {group['item']['item_id']: item for group, item in zip(groups, judged['answer']['items'])}
    used = judged['model_used_sources']
    cards = []
    for card_id, group, trace in zip(card_ids or range(len(groups)), groups, used):
        result = results.get(group['item']['item_id']) or {
            'name': group['item']['name'], 'carry_on': '확인 필요', 'checked_baggage': '확인 필요', 'notes': None}
        cards.append(Card(
            id=card_id, name=result['name'], carry_on=Verdict(status=result['carry_on']),
            checked=Verdict(status=result['checked_baggage']),
            notes=result['notes'] if trace['sources'] else NO_EVIDENCE, evidence=trace['sources']))
    return cards
