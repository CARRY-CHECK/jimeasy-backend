"""상태 전이와 도구 프로토콜 회귀. 외부 모델·Chroma를 호출하지 않는다."""
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.chat import service, tools
from app.chat.conditions import empty_conditions
from app.chat.schemas import ChatRequest, StructureInput
from app.chat.sessions import Session, SessionStore


def structure(name='보조배터리', details='100Wh 2개', changes=None, **extra):
    value = dict(changes=changes or [], mode='replace', items=[dict(item_id=None, name=name, details=details)],
                 question=f'{name} 기내 반입 가능한가요?', unknown_fields=[])
    value.update(extra)
    return value


def parsed_structure(arguments):
    conditions = empty_conditions()
    for row in arguments['changes']:
        parts = row['field'].split('.')
        if len(parts) == 2:
            conditions[parts[0]][parts[1]] = row['value']
        else:
            conditions[parts[0]] = row['value']
    return dict(conditions=conditions, items=[dict(name=item['name'], details=item['details'])
                                             for item in arguments['items']])


def call(name, arguments=None):
    return AIMessage(content='', tool_calls=[dict(name=name, args=arguments or {}, id='call_' + name)])


def search(items, **kwargs):
    return [dict(item=item, query=item['name'], documents=[dict(
        id='kr.rule#0', evidence_id=item['item_id'] + '_doc_1', text='규정 본문', metadata={'country': 'KR'})])
        for item in items]


def judgment(question, conditions, groups, **kwargs):
    return dict(answer={'items': [dict(name=g['item']['name'], carry_on='조건부 가능',
                                     checked_baggage='확인 필요', notes='용량 확인 필요', source=[])
                                 for g in groups]}, model_used_sources=[dict(sources=[{'chunk_id': 'kr.rule#0'}])
                                                                         for g in groups])


class StateTests(unittest.TestCase):
    def test_new_item_keeps_travel_but_does_not_inherit_item_details(self):
        session = Session()
        session.structure(StructureInput(**structure(changes=[{'field': 'arrival.place', 'value': '일본'}])))
        old_id = session.items[0]['item_id']
        session.structure(StructureInput(**structure('텀블러', None)))
        self.assertEqual(session.conditions['departure']['place'], '한국')
        self.assertEqual(session.conditions['arrival']['place'], '일본')
        self.assertEqual([x['name'] for x in session.items], ['텀블러'])
        self.assertIsNone(session.items[0]['details'])
        self.assertNotEqual(session.items[0]['item_id'], old_id)

    def test_explicit_clear_keeps_other_conditions(self):
        session = Session()
        session.structure(StructureInput(**structure(changes=[{'field': 'departure.airline', 'value': '대한항공'}])))
        session.structure(StructureInput(**structure(mode='keep', items=[], changes=[
            {'field': 'departure.airline', 'value': None}])))
        self.assertIsNone(session.conditions['departure']['airline'])
        self.assertEqual(session.conditions['departure']['place'], '한국')

    def test_quantity_update_invalidates_search_and_preserves_identity(self):
        session = Session()
        session.structure(StructureInput(**structure()))
        identity = session.items[0]['item_id']
        session.search_revision = session.revision
        revision = session.revision
        session.structure(StructureInput(**structure(mode='update', items=[dict(
            item_id=identity, name='보조배터리', details='100Wh 3개')])))
        self.assertEqual(session.items[0]['item_id'], identity)
        self.assertEqual(session.items[0]['details'], '100Wh 3개')
        self.assertGreater(session.revision, revision)
        self.assertIsNone(session.search_revision)

    def test_invalid_update_is_atomic(self):
        session = Session()
        before = deepcopy(session)
        with self.assertRaises(ValueError):
            session.structure(StructureInput(**structure(changes=[{'field': 'arrival.place', 'value': '일본'}],
                mode='update', items=[dict(item_id='missing', name='가위', details=None)])))
        self.assertEqual(session, before)

    def test_partial_answer_preserves_only_unanswered_pending_fields(self):
        session = Session()
        session.structure(StructureInput(**structure()))
        runner = tools.ToolRunner('test', session)
        runner.structured = True
        runner.execute('request_clarification', {'questions': [
            dict(field='arrival.place', question='도착 국가는 어디인가요?', options=[]),
            dict(field='departure.airline', question='항공사는 어디인가요?', options=[])]})
        session.structure(StructureInput(**structure(mode='keep', items=[], changes=[
            dict(field='arrival.place', value='일본')])))
        self.assertEqual([q['field'] for q in session.pending['questions']], ['departure.airline'])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.runner = tools.ToolRunner('test', self.session)

    def test_no_search_before_structure_or_final_before_search(self):
        with self.assertRaises(ValueError):
            self.runner.execute('search_regulations', {})
        self.runner.execute('structure_query', structure())
        with self.assertRaises(ValueError):
            self.runner.execute('finalize_answer', {})

    def test_clarification_is_terminal_and_unknown_is_not_reasked(self):
        self.runner.execute('structure_query', structure())
        args = dict(questions=[dict(field='departure.airline', question='항공사는 어디인가요?', options=[])])
        self.runner.execute('request_clarification', args)
        self.assertIsNotNone(self.runner.response.clarification)
        with self.assertRaises(ValueError):
            self.runner.execute('search_regulations', {})
        other = tools.ToolRunner('test', self.session)
        other.execute('structure_query', structure(mode='keep', items=[], unknown_fields=['departure.airline']))
        with self.assertRaises(ValueError):
            other.execute('request_clarification', args)
        self.assertIsNone(self.session.pending)

    def test_max_three_questions_and_already_known_conditions(self):
        self.runner.execute('structure_query', structure())
        with self.assertRaises(ValidationError):
            self.runner.execute('request_clarification', dict(questions=[dict(
                field='arrival.place', question='어디인가요?', options=[])] * 4))
        with self.assertRaises(ValueError):
            self.runner.execute('request_clarification', dict(questions=[dict(
                field='departure.place', question='어디인가요?', options=[])]))

    def test_repeated_question_stops_with_uncertain_cards(self):
        self.runner.execute('structure_query', structure())
        self.session.asked['arrival.place:도착지는?'] = 2
        result = self.runner.execute('request_clarification', dict(questions=[dict(
            field='arrival.place', question='도착지는?', options=[])]))
        self.assertEqual(result['reason'], 'clarification_limit')
        self.assertEqual(self.runner.response.cards[0].carry_on.status, '확인 필요')
        self.assertIsNone(self.session.pending)

    def test_departure_airline_is_not_reasked_as_return_airline(self):
        self.runner.execute('structure_query', structure(changes=[dict(
            field='departure.airline', value='대한항공')]))
        with self.assertRaises(ValueError):
            self.runner.execute('request_clarification', dict(questions=[dict(
                field='arrival.airline', question='어떤 항공사인가요?', options=[])]))

    def test_search_is_reused_for_the_same_state_and_rerun_after_a_change(self):
        self.runner.execute('structure_query', structure())
        with patch.object(tools, 'search_by_items', side_effect=search) as retrieve:
            first = self.runner.execute('search_regulations', {})
            second = self.runner.execute('search_regulations', {})
        retrieve.assert_called_once()  # 같은 상태의 두 번째 호출은 검색·reranking을 반복하지 않는다
        self.assertNotIn('reused', first)
        self.assertTrue(second['reused'])
        self.assertEqual(first['groups'], second['groups'])
        self.session.revision += 1  # 상태 변경으로 무효화되면 다시 검색한다
        self.session.search_revision = None
        self.session.groups = []
        with patch.object(tools, 'search_by_items', side_effect=search) as retrieve:
            self.runner.execute('search_regulations', {})
        retrieve.assert_called_once()

    def test_unsupported_destination_never_falls_back_to_global_search(self):
        self.runner.execute('structure_query', structure(changes=[dict(field='arrival.place', value='캐나다')]))
        with patch.object(tools, 'search_by_items') as retrieve:
            result = self.runner.execute('search_regulations', {})
        retrieve.assert_not_called()
        self.assertFalse(result['supported'])
        self.runner.execute('finalize_answer', {})
        self.assertIn('지원', self.runner.response.cards[0].notes)
        self.assertEqual(self.runner.response.cards[0].carry_on.status, '확인 필요')


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.sessions = SessionStore()
        self.search = patch.object(tools, 'search_by_items', side_effect=search)
        self.judge = patch.object(tools, 'judge_groups', side_effect=judgment)
        self.present = patch('app.chat.presenter.present', side_effect=lambda q,c,cards,g,**kw: (cards, '안내', None))
        for patcher in (self.search, self.judge, self.present):
            patcher.start()
            self.addCleanup(patcher.stop)

    def send(self, replies, **request):
        model = Mock()
        model.bind_tools.return_value.invoke.side_effect = replies
        structures = [tool['args'] for reply in replies for tool in reply.tool_calls if tool['name'] == 'structure_query']
        parsed = parsed_structure(structures[0]) if structures else dict(conditions=empty_conditions(), items=[])
        with patch.object(tools, 'parse_query', return_value=parsed):
            return service.handle_tools(ChatRequest(**request), self.sessions, model=model), model

    def test_final_answer_and_follow_up_new_item_only(self):
        response, model = self.send([call('structure_query', structure(changes=[dict(field='arrival.place', value='일본')])), call('search_regulations'),
                                     call('finalize_answer')], message='보조배터리 100Wh 2개 돼?')
        self.assertEqual([c.name for c in response.cards], ['보조배터리'])
        self.assertEqual(response.diagnostics['agent_calls'], 3)
        self.assertFalse(model.bind_tools.call_args.kwargs['parallel_tool_calls'])
        follow, _ = self.send([call('structure_query', structure('텀블러', None)), call('search_regulations'),
                               call('finalize_answer')], session_id=response.session_id, message='같은 조건으로 텀블러는?')
        self.assertEqual([c.name for c in follow.cards], ['텀블러'])
        self.assertEqual(len(self.sessions.get(response.session_id)[1].history), 4)

    def test_tool_results_are_matched_and_direct_text_is_not_exposed(self):
        response, model = self.send([AIMessage(content='무조건 가능합니다'), call('search_regulations'),
            call('structure_query', structure(changes=[dict(field='arrival.place', value='일본')])), call('search_regulations'), call('finalize_answer')], message='배터리 돼?')
        self.assertEqual(response.diagnostics['agent_calls'], 5)
        messages = model.bind_tools.return_value.invoke.call_args.args[0]
        results = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual(results[0].tool_call_id, 'call_search_regulations')
        self.assertIn('error', results[0].content)
        self.assertNotIn('무조건', str(response.model_dump()))

    def test_failed_request_does_not_commit_partial_changes(self):
        _, original = self.sessions.get('test')
        before = deepcopy(original)
        with patch.dict(service.MODEL_CONFIG['chat'], max_steps=2), self.assertRaises(RuntimeError):
            self.send([call('structure_query', structure()), call('search_regulations')],
                      session_id='test', message='배터리 돼?')
        self.assertEqual(self.sessions.get('test')[1], before)

    def test_wait_resume_and_stale_choice(self):
        response, model = self.send([call('structure_query', structure()), call('request_clarification', {
            'questions': [dict(field='departure.airline', question='항공사는?', options=['대한항공'])]})], message='배터리 돼?')
        self.assertEqual(model.bind_tools.return_value.invoke.call_count, 2)
        with self.assertRaises(service.ChatInputError):
            self.send([], session_id=response.session_id, choice=dict(card=0, option=0, question_id='old'))
        final, _ = self.send([call('structure_query', structure(mode='keep', items=[], changes=[
            dict(field='departure.airline', value='대한항공'), dict(field='arrival.place', value='일본')])), call('search_regulations'), call('finalize_answer')],
            session_id=response.session_id, choice=dict(card=0, option=0, question_id=response.clarification.id))
        self.assertEqual(final.conditions.departure.airline, '대한항공')

    def test_missing_destination_requires_clarification(self):
        response, _ = self.send([call('structure_query', structure()), call('search_regulations'),
            call('finalize_answer'), call('request_clarification', {'questions': [dict(
                field='arrival.place', question='도착 국가는 어디인가요?', options=[])]})], message='배터리 돼?')
        self.assertIsNotNone(response.clarification)
        self.assertEqual(response.cards, [])


class ToolApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app.api.main import app
        self.client = TestClient(app)
        self.sessions = SessionStore()
        # handle의 기본 인자는 모듈 초기화 시 캡처되므로 실제 도구 핸들러에 저장소를 명시한다.
        real_handle = service.handle_tools
        self.handle = patch.object(service, 'handle', side_effect=lambda request: real_handle(request, self.sessions))
        self.handle.start()
        self.addCleanup(self.handle.stop)

    def test_http_clarification_and_stale_choice(self):
        model = Mock()
        model.bind_tools.return_value.invoke.side_effect = [call('structure_query', structure()),
            call('request_clarification', {'questions': [dict(field='arrival.place', question='어디로 가나요?', options=['일본'])]})]
        with patch.object(service, 'create_chat_model', return_value=model), patch.object(
                tools, 'parse_query', return_value=parsed_structure(structure())):
            response = self.client.post('/chat', json={'message': '배터리 돼?'})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['clarification']['questions'][0]['field'], 'arrival.place')
        stale = self.client.post('/chat', json=dict(session_id=body['session_id'],
            choice=dict(card=0, option=0, question_id='old')))
        self.assertEqual(stale.status_code, 400)

    def test_http_reset_clears_tool_state(self):
        _, session = self.sessions.get('reset-tool')
        session.history.append(dict(role='user', content='이전 질문'))
        # reset 엔드포인트는 service.store를 직접 사용한다.
        with patch.object(service, 'store', self.sessions):
            response = self.client.post('/chat/reset', json=dict(session_id='reset-tool'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.sessions.get('reset-tool')[1].history, [])


if __name__ == '__main__':
    unittest.main()
