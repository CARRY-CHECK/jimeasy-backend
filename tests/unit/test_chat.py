import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.chat import presenter, service
from app.chat.conditions import empty_conditions, merge_conditions
from app.chat.schemas import Card, CardText, ChatRequest, FollowUp, Presentation, Verdict
from app.chat.sessions import SessionStore
from app.schemas.baggage import DecisionResponse, ItemDecision


def conditions(**values):
    result = empty_conditions()
    for key, value in values.items():
        if '_' in key and key.split('_')[0] in ('departure', 'arrival'):
            leg, field = key.split('_')
            result[leg][field] = value
        else:
            result[key] = value
    return result


class ConditionTests(unittest.TestCase):
    def test_unmentioned_values_are_kept(self):
        previous = conditions(arrival_place='대만', departure_airline='아시아나항공', baggage_type='기내')
        merged = merge_conditions(previous, empty_conditions())
        self.assertEqual(merged, previous)

    def test_new_values_replace_previous(self):
        previous = conditions(arrival_place='대만', departure_airline='아시아나항공')
        merged = merge_conditions(previous, conditions(arrival_place='일본'))
        self.assertEqual(merged['arrival']['place'], '일본')
        self.assertEqual(merged['departure']['airline'], '아시아나항공')


class SessionTests(unittest.TestCase):
    def test_new_session_gets_id_and_empty_state(self):
        store = SessionStore()
        session_id, session = store.get()
        self.assertTrue(session_id)
        self.assertEqual(session.conditions, empty_conditions())
        self.assertIs(store.get(session_id)[1], session)

    def test_unknown_id_is_created_and_reset_clears_state(self):
        store = SessionStore()
        session_id, session = store.get('stale')
        self.assertEqual(session_id, 'stale')
        session.conditions['arrival']['place'] = '대만'
        store.reset(session_id)
        self.assertEqual(store.get(session_id)[1].conditions, empty_conditions())


class RequestTests(unittest.TestCase):
    def test_exactly_one_of_message_or_choice(self):
        ChatRequest(message='가위 돼?')
        ChatRequest(session_id='a', choice={'card': 0, 'option': 1})
        with self.assertRaises(ValidationError):
            ChatRequest()
        with self.assertRaises(ValidationError):
            ChatRequest(message='가위', choice={'card': 0, 'option': 0})
        with self.assertRaises(ValidationError):
            ChatRequest(choice={'card': -1, 'option': 0})


def parsed(items=(), **values):
    return {'conditions': conditions(**values),
            'items': [{'name': name, 'details': None} for name in items]}


def fake_search(items, filters=None, k=None, conditions=None):
    return [{'item': item, 'query': item['name'], 'documents': [
        {'id': f"{item['name']}-{index}", 'text': '규정', 'metadata': {'country': 'TW'},
         'evidence_id': f"{item['item_id']}_doc_{index}"} for index in (1, 2)]}
        for item in items]


def decide(carry_on, checked='가능', evidence=(1,)):
    """프롬프트에 포함된 item_id마다 같은 판정을 반환하는 가짜 판정 LLM."""
    def generate(system, user, schema):
        ids = sorted({line.split('"')[3] for line in user.splitlines() if '"item_id"' in line})
        return DecisionResponse(items=[ItemDecision(
            item_id=item_id, carry_on=carry_on, checked_baggage=checked, notes='날 6cm 이하',
            evidence_ids=[f'{item_id}_doc_{index}' for index in evidence]) for item_id in ids])
    return generate


def fake_present(system, user, schema, role):
    """프롬프트의 card_id마다 같은 조건·되묻기를 반환하는 가짜 문장 LLM."""
    ids = [int(line.split(':')[1].strip(' ,')) for line in user.splitlines() if '"card_id"' in line]
    return Presentation(summary='날 길이가 6cm 이하면 기내에 가져갈 수 있어요.', suggested_next='라이터도 돼?', cards=[
        CardText(card_id=card_id, carry_on_condition='날 6cm 이하', checked_condition='날 부분 감싸서 포장', follow_up_basis='날 6cm 이하',
                 follow_up_question='날 길이가 어느 정도인가요?', follow_up_options=['6cm 이하', '6cm 초과'])
        for card_id in ids])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.start(patch.dict(service.MODEL_CONFIG['chat'], engine='legacy'))
        self.store = SessionStore()
        self.parse = self.start(patch.object(service, 'parse_query'))
        self.search = self.start(patch.object(service, 'search_by_items', side_effect=fake_search))
        self.generate = self.start(patch.object(service, 'generate_structured_langchain',
                                                side_effect=decide('조건부 가능')))
        self.present = self.start(patch.object(presenter, 'generate_structured_langchain',
                                               side_effect=fake_present))

    def start(self, patcher):
        self.addCleanup(patcher.stop)
        return patcher.start()

    def send(self, session_id=None, **request):
        return service.handle(ChatRequest(session_id=session_id, **request), self.store)

    def ask_scissors(self):
        self.parse.return_value = parsed(['가위'], arrival_place='대만', departure_airline='아시아나항공')
        self.response = self.send(message='아시아나로 대만 가는데 가위 돼?')
        return self.response.session_id, self.store.get(self.response.session_id)[1]

    def test_new_question_returns_presented_card_with_used_evidence_only(self):
        session_id, session = self.ask_scissors()
        card = session.cards[0]
        self.assertEqual((card.id, card.name, card.carry_on.status, card.checked.status),
                         (0, '가위', '조건부 가능', '가능'))
        self.assertEqual(card.carry_on.condition, '날 6cm 이하')
        self.assertEqual(card.follow_up.options, ['6cm 이하', '6cm 초과', presenter.UNKNOWN_OPTION])
        self.assertEqual([source['chunk_id'] for source in card.evidence], ['가위-1'])
        self.assertEqual(session.conditions['arrival']['place'], '대만')
        self.assertEqual((self.response.summary, self.response.suggested_next),
                         ('날 길이가 6cm 이하면 기내에 가져갈 수 있어요.', '라이터도 돼?'))

    def test_choice_rejudges_same_documents_without_search_or_parse(self):
        session_id, session = self.ask_scissors()
        documents = session.groups[0]['documents']
        self.generate.side_effect = decide('가능')
        response = self.send(session_id, choice={'card': 0, 'option': 0})
        self.assertEqual(self.search.call_count, 1)
        self.assertEqual(self.parse.call_count, 1)
        self.assertEqual(response.cards[0].carry_on.status, '가능')
        self.assertIsNone(response.cards[0].follow_up)
        self.assertIs(session.groups[0]['documents'], documents)
        self.assertEqual(session.groups[0]['item']['details'], '날 길이가 어느 정도인가요? 6cm 이하')
        self.assertIn('6cm 이하', self.generate.call_args.args[1])

    def test_unknown_option_keeps_verdict_without_llm(self):
        session_id, _ = self.ask_scissors()
        response = self.send(session_id, choice={'card': 0, 'option': 2})
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(response.cards[0].carry_on.status, '조건부 가능')
        self.assertIsNone(response.cards[0].follow_up)

    def test_choice_replaces_only_that_card(self):
        self.parse.return_value = parsed(['가위', '보조배터리'])
        session_id = self.send(message='가위랑 보조배터리 돼?').session_id
        session = self.store.get(session_id)[1]
        self.generate.side_effect = decide('가능')
        response = self.send(session_id, choice={'card': 1, 'option': 0})
        self.assertEqual([card.id for card in response.cards], [1])
        self.assertIsNotNone(session.cards[0].follow_up)
        self.assertEqual(session.cards[1].carry_on.status, '가능')
        self.assertIsNone(session.cards[1].follow_up)

    def test_free_text_answers_pending_follow_up(self):
        session_id, session = self.ask_scissors()
        self.parse.return_value = parsed()
        self.send(session_id, message='5cm 정도야')
        self.assertEqual(self.search.call_count, 1)
        self.assertEqual(session.groups[0]['item']['details'], '날 길이가 어느 정도인가요? 5cm 정도야')

    def test_unrelated_text_with_pending_follow_up_asks_for_answer(self):
        session_id, session = self.ask_scissors()
        self.parse.return_value = parsed()
        response = self.send(session_id, message='안녕')
        self.assertEqual(response.message, service.ASK_ANSWER)
        self.assertEqual(self.generate.call_count, 1)
        self.assertIsNone(session.groups[0]['item']['details'])

    def test_free_text_unknown_closes_follow_up(self):
        session_id, session = self.ask_scissors()
        self.parse.return_value = parsed()
        response = self.send(session_id, message='잘 모르겠어')
        self.assertEqual(self.generate.call_count, 1)
        self.assertIsNone(response.cards[0].follow_up)
        self.assertIsNone(session.cards[0].follow_up)

    def test_answer_matching_picks_cards_by_options(self):
        def pending(card_id, options):
            return card(card_id=card_id).model_copy(update={'follow_up': FollowUp(question='?', options=options)})
        cards = [pending(0, ['6cm 이하', '6cm 초과', presenter.UNKNOWN_OPTION]),
                 pending(1, ['충전식', '일반 라이터', presenter.UNKNOWN_OPTION]),
                 pending(2, ['네', '아니요', presenter.UNKNOWN_OPTION])]
        ids = lambda message: [card.id for card in service.answered_cards(message, cards)]
        self.assertEqual(ids('5cm 정도야'), [0])
        self.assertEqual(ids('충전식이야'), [1])
        self.assertEqual(ids('아니요 그런 거 없어'), [2])
        self.assertEqual(ids('안녕'), [])

    def test_condition_change_searches_previous_items_again(self):
        session_id, session = self.ask_scissors()
        self.parse.return_value = parsed(arrival_place='일본')
        self.send(session_id, message='아 일본으로 바꿀게')
        self.assertEqual(self.search.call_count, 2)
        conditions_used = self.search.call_args.kwargs['conditions']
        self.assertEqual(conditions_used['arrival']['place'], '일본')
        self.assertEqual(conditions_used['departure']['airline'], '아시아나항공')
        self.assertEqual(self.search.call_args.args[0][0]['name'], '가위')

    def test_no_item_and_nothing_pending_asks_for_item(self):
        self.parse.return_value = parsed(arrival_place='대만')
        response = self.send(message='대만 가요')
        self.assertEqual(response.message, service.ASK_ITEM)
        self.assertEqual(response.conditions.arrival.place, '대만')
        self.generate.assert_not_called()

    def test_item_without_documents_is_check_needed(self):
        self.search.side_effect = lambda items, **_: [{'item': items[0], 'query': '', 'documents': []}]
        self.parse.return_value = parsed(['마법봉'])
        card = self.send(message='마법봉 돼?').cards[0]
        self.assertEqual((card.carry_on.status, card.notes), ('확인 필요', service.NO_EVIDENCE))
        self.generate.assert_not_called()

    def test_verdict_without_used_evidence_gets_fixed_notice_and_no_follow_up(self):
        self.generate.side_effect = decide('확인 필요', '확인 필요', evidence=())
        self.parse.return_value = parsed(['마법봉'])
        card = self.send(message='마법봉 돼?').cards[0]
        self.assertEqual((card.notes, card.evidence, card.follow_up), (service.NO_EVIDENCE, [], None))

    def test_failed_judgement_leaves_session_unchanged(self):
        session_id, session = self.ask_scissors()
        self.generate.side_effect = RuntimeError('OpenAI 실패')
        self.parse.return_value = parsed(['라이터'], arrival_place='일본')
        with self.assertRaises(RuntimeError):
            self.send(session_id, message='일본 가는데 라이터는?')
        self.assertEqual(session.cards[0].name, '가위')
        self.assertEqual(session.conditions['arrival']['place'], '대만')

    def test_stale_choice_is_rejected(self):
        session_id, _ = self.ask_scissors()
        with self.assertRaises(ValueError):
            self.send(session_id, choice={'card': 5, 'option': 0})


def card(carry_on='조건부 가능', checked='가능', card_id=0):
    return Card(id=card_id, name='가위', carry_on=Verdict(status=carry_on),
                checked=Verdict(status=checked), notes='날 6cm 이하', evidence=[{'chunk_id': 'kr.a#1'}])


def text(card_id=0, options=('6cm 이하', '6cm 초과')):
    return CardText(card_id=card_id, carry_on_condition='날 6cm 이하', checked_condition=None, follow_up_basis='6cm',
                    follow_up_question='날 길이가 어느 정도인가요?', follow_up_options=list(options))


class PresenterTests(unittest.TestCase):
    groups = [{'item': {'name': '가위', 'details': None}}]

    def run_present(self, cards, result):
        with patch.object(presenter, 'generate_structured_langchain', return_value=result) as generate:
            output = presenter.present('가위 돼?', empty_conditions(), cards, self.groups * len(cards))
        self.assertEqual(generate.call_args.kwargs['role'], 'present')
        return output

    def test_conditions_and_unknown_option_are_added(self):
        cards, summary, _ = self.run_present([card()], Presentation(
            summary='결론', suggested_next=None, cards=[text(options=[' 6cm 이하', '6cm 이하', '잘 모르겠어요', '6cm 초과'])]))
        self.assertEqual(summary, '결론')
        self.assertEqual(cards[0].carry_on, Verdict(status='조건부 가능', condition='날 6cm 이하'))
        self.assertEqual(cards[0].follow_up.options, ['6cm 이하', '6cm 초과', presenter.UNKNOWN_OPTION])

    def test_confirmed_verdict_gets_no_follow_up(self):
        cards, _, _ = self.run_present([card('가능', '불가')], Presentation(
            summary='결론', suggested_next=None, cards=[text()]))
        self.assertIsNone(cards[0].follow_up)
        self.assertEqual(cards[0].carry_on.status, '가능')

    def test_status_word_condition_is_dropped(self):
        result = text()
        result.checked_condition = '추가 확인 필요'
        cards, _, _ = self.run_present([card()], Presentation(summary='결론', suggested_next=None, cards=[result]))
        self.assertIsNone(cards[0].checked.condition)
        self.assertEqual(presenter.condition('1개까지 가능'), '1개까지 가능')
        self.assertIsNone(presenter.condition('null'))

    def test_options_must_come_from_notes(self):
        scissors = card().model_copy(update={'notes': '가위는 일반적으로 항공사에 따라 다를 수 있습니다.'})
        self.assertFalse(presenter.cites_notes(scissors, '어떤 종류의 가위인가요?', ['일반 가위', '미용 가위']))
        lighter = card().model_copy(update={'name': '라이터', 'notes': '충전식 전기라이터는 반입 불가'})
        self.assertTrue(presenter.cites_notes(lighter, '충전식 라이터인가요?', ['네', '아니요']))
        self.assertTrue(presenter.cites_notes(lighter, '어떤 라이터인가요?', ['충전식', '일반 라이터']))
        self.assertTrue(presenter.cites_notes(card(), '날 길이는?', ['6cm 이하', '6cm 초과']))

    def test_follow_up_asking_known_condition_is_dropped(self):
        result = text(options=['아시아나항공', '다른 항공사'])
        known = empty_conditions()
        known['departure']['airline'] = '아시아나항공'
        with patch.object(presenter, 'generate_structured_langchain',
                          return_value=Presentation(summary='결론', suggested_next=None, cards=[result])):
            cards, _, _ = presenter.present('가위 돼?', known, [card()], self.groups)
        self.assertIsNone(cards[0].follow_up)

    def test_duplicated_card_uses_first(self):
        cards, summary, _ = self.run_present([card()], Presentation(
            summary='결론', suggested_next=None, cards=[text(), text(options=['다른 선택지'])]))
        self.assertEqual(summary, '결론')
        self.assertEqual(cards[0].follow_up.options[0], '6cm 이하')

    def test_wrong_card_id_is_aligned_by_order(self):
        cards, _, _ = self.run_present([card(card_id=3)], Presentation(
            summary='결론', suggested_next=None, cards=[text(card_id=0)]))
        self.assertEqual((cards[0].id, cards[0].carry_on.condition), (3, '날 6cm 이하'))

    def test_follow_up_without_basis_in_notes_is_dropped(self):
        result = text(options=['일반 가위', '미용 가위'])
        result.follow_up_basis = '가위 종류에 따라 다름'
        cards, _, _ = self.run_present([card()], Presentation(summary='결론', suggested_next=None, cards=[result]))
        self.assertIsNone(cards[0].follow_up)

    def test_already_answered_question_is_not_asked_again(self):
        self.groups = [{'item': {'name': '가위', 'details': '날 길이가 어느 정도인가요? 6cm 이하'}}]
        cards, _, _ = self.run_present([card()], Presentation(summary='결론', suggested_next=None, cards=[text()]))
        self.assertIsNone(cards[0].follow_up)

    def test_mismatched_card_count_falls_back_to_verdicts(self):
        original = [card()]
        output = self.run_present(original, Presentation(summary='결론', suggested_next=None, cards=[text(), text(card_id=1)]))
        self.assertEqual(output, (original, None, None))

    def test_llm_failure_falls_back_to_verdicts(self):
        original = [card()]
        with patch.object(presenter, 'generate_structured_langchain', side_effect=RuntimeError('실패')):
            self.assertEqual(presenter.present('가위 돼?', empty_conditions(), original, self.groups),
                             (original, None, None))

    def test_prompt_contains_details_and_verdicts(self):
        groups = [{'item': {'name': '가위', 'details': '6cm 이하'}}]
        _, user = presenter.build_prompt('가위 돼?', empty_conditions(), [card()], groups)
        self.assertIn('"details": "6cm 이하"', user)
        self.assertIn('"carry_on": "조건부 가능"', user)


class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app.api import main
        self.client = TestClient(main.app)

    def post_chat(self, error):
        with patch.object(service, 'handle', side_effect=error):
            return self.client.post('/chat', json={'message': '가위 돼?'})

    def test_chat_returns_service_response(self):
        response = service.ChatResponse(session_id='abc', conditions=empty_conditions(), message='안내')
        with patch.object(service, 'handle', return_value=response):
            body = self.client.post('/chat', json={'message': '가위 돼?'}).json()
        self.assertEqual((body['session_id'], body['message']), ('abc', '안내'))

    def test_errors_are_mapped_to_status_codes(self):
        self.assertEqual(self.post_chat(service.ChatInputError('질문을 입력하세요.')).status_code, 400)
        self.assertEqual(self.post_chat(RuntimeError('OpenAI 실패')).status_code, 502)
        self.assertEqual(self.client.post('/chat', json={}).status_code, 422)

    def test_reset_clears_session(self):
        _, session = service.store.get('api-reset')
        session.conditions['arrival']['place'] = '대만'
        body = self.client.post('/chat/reset', json={'session_id': 'api-reset'}).json()
        self.assertIsNone(body['conditions']['arrival']['place'])
        self.assertEqual(service.store.get('api-reset')[1].conditions, empty_conditions())


if __name__ == '__main__':
    unittest.main()
