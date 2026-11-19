"""터미널 챗봇. 서버 없이 대화 계층(app.chat.service)을 직접 호출한다.
번호 입력 = 선택지 클릭, /new = 새로 확인하기, /q = 종료.
"""
from dotenv import load_dotenv

from app.chat import service
from app.chat.schemas import ChatRequest, Choice
from app.settings import ENV_FILE

HELP = '번호 입력 = 선택지, /new = 새로 확인하기, /q = 종료'
CHIPS = (('출발지', lambda c: c.departure.place), ('도착', lambda c: c.arrival.place),
         ('항공사', lambda c: c.departure.airline), ('날짜', lambda c: c.travel_date),
         ('수하물', lambda c: c.baggage_type))


def render(response):
    """응답을 출력하고 {번호: Choice} 를 반환한다."""
    print('\n[' + '] ['.join(value(response.conditions) or f'{name} 미입력' for name, value in CHIPS) + ']')
    print('\n' + (response.summary or response.message or ''))
    choices = {}
    if response.clarification:
        for question_index, row in enumerate(response.clarification.questions):
            print(f'\n  {row.question}')
            for option_index, option in enumerate(row.options):
                number = len(choices) + 1
                choices[number] = Choice(card=question_index, option=option_index,
                                         question_id=response.clarification.id)
                print(f'    {number}) {option}')
    for card in response.cards:
        print(f'\n  ■ {card.name}')
        for label, verdict in (('기내', card.carry_on), ('위탁', card.checked)):
            print(f'    {label}  [{verdict.status or "-"}] {verdict.condition or ""}')
        if card.notes:
            print(f'    설명: {card.notes}')
        if card.follow_up:
            print(f'    {card.follow_up.question}')
            for index, option in enumerate(card.follow_up.options):
                number = len(choices) + 1
                choices[number] = Choice(card=card.id, option=index)
                print(f'      {number}) {option}')
        print(f'    근거 {len(card.evidence)}개: ' + ', '.join(
            source.get('source_name') or source.get('chunk_id', '') for source in card.evidence))
    if response.suggested_next:
        print(f'\n  (예: {response.suggested_next})')
    return choices


def main():
    load_dotenv(ENV_FILE, override=False)
    print('JimEasy  |  ' + HELP)
    session_id, choices = None, {}
    while True:
        try:
            text = input('\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == '/q':
            break
        if text == '/new':
            if session_id:
                service.store.reset(session_id)
            choices = {}
            print('새로 시작합니다.')
            continue
        if text.startswith('/'):
            print(f'모르는 명령입니다. {HELP}')
            continue
        if not text:
            continue
        if text.isdigit() and int(text) in choices:
            request = ChatRequest(session_id=session_id, choice=choices[int(text)])
        else:
            request = ChatRequest(session_id=session_id, message=text)
        try:
            response = service.handle(request)
        except (RuntimeError, ValueError) as error:
            print(f'오류: {error}')
            continue
        session_id = response.session_id
        choices = render(response)


if __name__ == '__main__':
    main()
