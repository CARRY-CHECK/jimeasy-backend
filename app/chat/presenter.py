"""③ 문장 만들기. 확정된 판정을 결론 한 줄, 짧은 조건, 되묻기 질문으로 바꾼다.
판정은 바꾸지 않으며, 실패하면 문장 없이 판정만 반환한다.
"""
import json
import logging
from string import Template
from typing import get_args

from app.chat.schemas import FollowUp, Presentation, Verdict
from app.integrations.openai_client import generate_structured_langchain
from app.query_parser import read_prompt
from app.schemas.baggage import Status
from app.settings import PRESENT_SYSTEM_PROMPT

UNKNOWN_OPTION = '잘 모르겠어요'
OPEN_STATUSES = ('조건부 가능', '확인 필요')
STATUS_WORDS = set(get_args(Status))
VAGUE_CONDITIONS = ('확인 필요', '조건부 가능')
YES_NO = ('네', '예', '아니')
PARTICLES = ('', '은', '는', '이', '가', '을', '를', '의', '도', '만', '로', '으로', '에', '에서', '와', '과', '까지')

logger = logging.getLogger(__name__)
MAX_OPTIONS = 3


def present(question, conditions, cards, groups, *, allow_follow_up=True):
    """cards와 groups는 같은 순서. 반환: (cards, summary, suggested_next)."""
    if not cards:
        return cards, None, None
    try:
        result = generate_structured_langchain(*build_prompt(question, conditions, cards, groups),
                                               Presentation, role='present')
        texts = match_texts(cards, result.cards)
    except (RuntimeError, ValueError) as error:
        logger.warning('문장 만들기 실패, 판정만 반환: %s', error)
        return cards, None, None
    known = known_conditions(conditions)
    cards = [apply_text(card, texts[card.id], group['item'].get('details'), known)
             for card, group in zip(cards, groups)]
    if not allow_follow_up:
        cards = [card.model_copy(update={'follow_up': None}) for card in cards]
    return cards, result.summary, result.suggested_next


def match_texts(cards, texts):
    """모델이 같은 카드를 중복 반환하면 첫 번째만 쓰고, 번호만 틀리면 입력 순서로 맞춘다."""
    by_id = {}
    for text in texts:
        by_id.setdefault(text.card_id, text)
    if set(by_id) == {card.id for card in cards}:
        return by_id
    if len(texts) == len(cards):
        return {card.id: text for card, text in zip(cards, texts)}
    raise ValueError('문장 결과의 카드가 입력과 일치하지 않습니다.')


def build_prompt(question, conditions, cards, groups):
    system, template = read_prompt(PRESENT_SYSTEM_PROMPT, 'chat/present_user.txt')
    rows = [{'card_id': card.id, 'name': card.name, 'details': group['item'].get('details'),
             'carry_on': card.carry_on.status, 'checked_baggage': card.checked.status, 'notes': card.notes}
            for card, group in zip(cards, groups)]
    user = Template(template).substitute(
        question=question, conditions=json.dumps(conditions, ensure_ascii=False),
        cards=json.dumps(rows, ensure_ascii=False, indent=2))
    return system.strip(), user


def known_conditions(conditions):
    """이미 아는 장소·항공사 값. 되묻기가 이 값을 다시 물으면 버린다."""
    values = [conditions[leg][key] for leg in ('departure', 'arrival') for key in ('place', 'airline')]
    return {value for value in values if value}


def apply_text(card, text, details=None, known=()):
    """되묻기는 판정이 열려 있고, 근거 구절이 notes에 실제로 있고, 아직 답하지 않았고,
    이미 아는 여행 조건을 묻지 않을 때만 붙인다.
    모름 선택지는 코드가 추가한다.
    """
    options = [option.strip() for option in text.follow_up_options if option.strip()]
    options = [option for option in dict.fromkeys(options) if option != UNKNOWN_OPTION][:MAX_OPTIONS]
    question, basis = text.follow_up_question, (text.follow_up_basis or '').strip()
    is_open = card.carry_on.status in OPEN_STATUSES or card.checked.status in OPEN_STATUSES
    grounded = bool(card.evidence and basis and card.notes and basis in card.notes)
    answered = bool(question and details and question in details)
    asks_known = any(value in part for value in known for part in [question or '', *options])
    follow_up = None
    if (is_open and grounded and question and options and not answered and not asks_known
            and cites_notes(card, question, options)):
        follow_up = FollowUp(question=question, options=options + [UNKNOWN_OPTION])
    return card.model_copy(update={
        'carry_on': Verdict(status=card.carry_on.status, condition=condition(text.carry_on_condition)),
        'checked': Verdict(status=card.checked.status, condition=condition(text.checked_condition)),
        'follow_up': follow_up})


def cites_notes(card, question, options):
    """선택지(네/아니요뿐이면 질문)의 단어 중 물품명이 아닌 것이 notes에 있어야 한다.
    예: '6cm 이하'의 '6cm'는 통과, '미용 가위'처럼 notes에 없는 분류는 탈락.
    """
    choices = [option for option in options if not option.startswith(YES_NO)]
    words = {word.strip('?.,!') for part in (choices or [question]) for word in part.split()}
    notes = [word.strip('?.,!()') for word in card.notes.split()]
    return any(len(word) >= 2 and word not in card.name and any(
        note.startswith(word) and note[len(word):] in PARTICLES for note in notes) for word in words)


def condition(value):
    """판정 단어를 반복하거나 '확인 필요'처럼 내용 없는 조건은 배지와 중복되므로 버린다."""
    value = (value or '').strip()
    if value.lower() in ('', 'null', 'none') or value in STATUS_WORDS or any(word in value for word in VAGUE_CONDITIONS):
        return None
    return value
