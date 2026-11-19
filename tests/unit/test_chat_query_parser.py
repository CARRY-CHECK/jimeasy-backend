import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from app.chat import service
from app.chat.conditions import empty_conditions
from app.chat.schemas import ChatRequest, StructureInput
from app.chat.sessions import Session, SessionStore
from app.chat.tools import ToolRunner
from app.schemas.query import ParsedQuery
from app.query_parser import parse_query


def update(mode='replace', items=None, changes=None):
    return dict(mode=mode, items=items or [], changes=changes or [], question='현재 질문', unknown_fields=[])


class ChatQueryParserTests(unittest.TestCase):
    def test_failed_mode_can_retry_without_reparsing_or_mutating_state(self):
        session = Session()
        before = deepcopy(session)
        runner = ToolRunner('test', session, message='텀블러는?')
        parsed = dict(conditions=empty_conditions(), items=[dict(name='텀블러', details=None)])
        with patch('app.chat.tools.parse_query', return_value=parsed) as parse:
            with self.assertRaisesRegex(ValueError, '물품 변경'):
                runner.execute('structure_query', update(mode='keep'))
            self.assertEqual(session, before)
            runner.execute('structure_query', update())
        parse.assert_called_once()
        self.assertEqual(session.items[0]['name'], '텀블러')

    def test_unparsed_nonnull_agent_conditions_are_not_added(self):
        session = Session()
        runner = ToolRunner('test', session, message='텀블러 돼?')
        parsed = dict(conditions=empty_conditions(), items=[dict(name='텀블러', details=None)])
        with patch('app.chat.tools.parse_query', return_value=parsed):
            runner.execute('structure_query', update(changes=[dict(field='arrival.place', value='미국')]))
        self.assertIsNone(session.conditions['arrival']['place'])

    def test_parser_items_and_conditions_drive_search(self):
        session = Session()
        parsed = dict(conditions=empty_conditions(), items=[dict(name='알코올', details='소독용 100ml')])
        parsed['conditions']['arrival']['place'] = '일본'
        runner = ToolRunner('test', session, message='일본 갈 때 소독용 알코올 100ml 돼?')
        with patch('app.chat.tools.parse_query', return_value=parsed) as parse:
            result = runner.execute('structure_query', update(items=[dict(item_id=None, name='잘못된 물품', details=None)]))
        parse.assert_called_once()
        self.assertEqual(parse.call_args.args[0], '일본 갈 때 소독용 알코올 100ml 돼?')
        self.assertFalse(parse.call_args.kwargs['context']['has_previous_turn'])
        self.assertEqual(result['parsed_query'], parsed)
        self.assertEqual(session.items[0]['name'], '알코올')
        self.assertEqual(session.items[0]['details'], '소독용 100ml')
        with patch('app.chat.tools.search_by_items', return_value=[]) as search:
            runner.execute('search_regulations', {})
        self.assertEqual(search.call_args.kwargs['conditions']['arrival']['place'], '일본')

    def test_followup_quantity_uses_context_and_keeps_country(self):
        session = Session()
        session.structure(StructureInput(**update(items=[dict(item_id=None, name='보조배터리', details='100Wh 2개')],
                                                changes=[dict(field='departure.place', value='미국')])))
        identity = session.items[0]['item_id']
        parsed = dict(conditions=empty_conditions(), items=[dict(name='보조배터리', details='100Wh 3개')])
        runner = ToolRunner('test', session, message='아니 3개야')
        with patch('app.chat.tools.parse_query', return_value=parsed) as parse:
            runner.execute('structure_query', update(mode='update', items=[dict(item_id=identity, name='보조배터리', details='3개')]))
        self.assertTrue(parse.call_args.kwargs['context']['has_previous_turn'])
        self.assertEqual(session.items[0]['details'], '100Wh 3개')
        self.assertEqual(session.items[0]['item_id'], identity)
        self.assertEqual(session.conditions['departure']['place'], '미국')

    def test_explicit_clear_and_pending_answer_remain_supported(self):
        session = Session()
        session.conditions['departure']['place'] = '미국'
        parsed = dict(conditions=empty_conditions(), items=[])
        parsed['conditions']['arrival']['place'] = '일본'
        runner = ToolRunner('test', session, message='도착은 일본이고 출발은 미정')
        with patch('app.chat.tools.parse_query', return_value=parsed):
            runner.execute('structure_query', update(mode='keep', changes=[dict(field='departure.place', value=None)]))
        self.assertIsNone(session.conditions['departure']['place'])
        self.assertEqual(session.conditions['arrival']['place'], '일본')

    def test_parser_failure_does_not_commit_session(self):
        sessions = SessionStore()
        _, before = sessions.get('test')
        original = deepcopy(before)
        model = Mock()
        model.bind_tools.return_value.invoke.return_value = AIMessage(content='', tool_calls=[
            dict(id='c1', name='structure_query', args=update())])
        with patch('app.chat.tools.parse_query', side_effect=RuntimeError('parser failed')):
            with self.assertRaisesRegex(RuntimeError, 'parser failed'):
                service.handle_tools(ChatRequest(session_id='test', message='물품 질문'), sessions, model=model)
        self.assertEqual(sessions.get('test')[1], original)

    def test_context_is_passed_as_data_without_changing_single_turn_contract(self):
        value = ParsedQuery(conditions=empty_conditions(), items=[])
        with patch('app.query_parser.generate_structured_langchain', return_value=value) as llm:
            parse_query('일본이요', context={'has_previous_turn': True, 'pending': {'field': 'arrival.place'}})
        system, user, _ = llm.call_args.args
        self.assertIn('has_previous_turn', system)
        self.assertIn('일본이요', user)
        self.assertIn('arrival.place', user)


if __name__ == '__main__':
    unittest.main()
