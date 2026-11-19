"""대화 도구의 얇은 연결 계층. 검색·판정은 공통 모듈에 위임한다."""
from uuid import uuid4

from langchain_core.tools import StructuredTool

from app.chat.conditions import CONDITION_FIELDS
from app.chat.schemas import (Card, ChatResponse, Clarification, ClarificationInput,
                              EmptyInput, StructureInput, Verdict)
from app.judgment import judge_groups
from app.query_parser import parse_query
from app.rag import search_by_items
from app.retrieval import countries_from_places
from app.settings import DEFAULT_K

UNSUPPORTED = '본 시스템은 한국, 일본, 중국, 미국, 영국의 수하물 규정만 지원합니다. 장소를 확인할 수 없거나 지원 범위 밖인 구간은 반입 여부를 안내할 수 없습니다.'


class ToolRunner:
    def __init__(self, session_id, session, *, message=None):
        self.session_id, self.session = session_id, session
        self.message = message
        self.parsed = None
        self.structured = False
        self.response = None
        self.trace = []
        self.tools = [StructuredTool.from_function(
            func=method, name=name, description=description, args_schema=schema)
            for name, method, schema, description in (
                ('structure_query', self.structure_query, StructureInput,
                 '매 사용자 턴의 첫 도구. query_parser로 실제 사용자 발화를 구조화하고 대화 상태 변경을 반영한다.'),
                ('request_clarification', self.request_clarification, ClarificationInput,
                 '판정을 바꿀 부족한 정보 최대 3개를 묶어 묻고 이번 실행을 끝낸다.'),
                ('search_regulations', self.search_regulations, EmptyInput,
                 '검증된 현재 상태의 물품별 규정을 검색한다. 새 속성이나 조건 변경 후 다시 실행한다.'),
                ('finalize_answer', self.finalize_answer, EmptyInput,
                 '현재 버전의 검색 근거로 판정·검증 후 최종 답변을 반환하고 실행을 끝낸다.'),
            )]
        self.by_name = {tool.name: tool for tool in self.tools}

    def execute(self, name, arguments):
        if name not in self.by_name:
            raise ValueError('등록되지 않은 도구입니다.')
        if self.response is not None:
            raise ValueError('종료된 턴에서는 도구를 실행할 수 없습니다.')
        if name != 'structure_query' and not self.structured:
            raise ValueError('먼저 structure_query로 이번 발화를 반영하세요.')
        # 검증 실패 시에도 해당 도구가 상태를 부분 변경하지 않도록 각 함수가 검증 후 반영한다.
        result = self.by_name[name].invoke(arguments)
        self.trace.append(dict(tool=name, revision=self.session.revision))
        return result

    def structure_query(self, **arguments):
        if self.structured:
            raise ValueError('같은 턴의 structure_query는 한 번만 실행하세요.')
        update = StructureInput.model_validate(arguments)
        if self.message is not None:
            if self.parsed is None:
                context = self.session.snapshot()
                context['has_previous_turn'] = bool(self.session.history or self.session.focus)
                self.parsed = parse_query(self.message, context=context)
            update = self.with_parsed_query(update)
        self.session.structure(update)
        self.structured = True
        result = self.session.snapshot()
        if self.parsed is not None:
            result['parsed_query'] = self.parsed
        return result

    def with_parsed_query(self, update):
        """파서는 조건/물품 추출을, 에이전트는 대상 추가/수정/해제 등 대화 동작을 맡는다."""
        payload = update.model_dump()
        # 명시적 해제는 파서의 null(미언급과 같은 표현)과 구분해 유지한다.
        changes = {row['field']: row['value'] for row in payload['changes'] if row['value'] is None}
        conditions = self.parsed['conditions']
        for field in sorted(CONDITION_FIELDS):
            parts = field.split('.')
            value = conditions[parts[0]][parts[1]] if len(parts) == 2 else conditions[field]
            if value is not None and field not in changes:
                changes[field] = value
        payload['changes'] = [dict(field=field, value=value) for field, value in changes.items()]
        parsed_items = self.parsed['items']
        if update.mode in ('replace', 'add'):
            payload['items'] = [dict(item_id=None, name=item['name'], details=item.get('details'))
                                for item in parsed_items]
            if update.mode == 'add' and not parsed_items:
                raise ValueError('query_parser가 추가 물품을 찾지 못했습니다. 조건 답변이면 keep을 사용하세요.')
        elif update.mode == 'update':
            for item in payload['items']:
                matches = [parsed for parsed in parsed_items if parsed['name'] == item['name']]
                if len(matches) != 1:
                    raise ValueError('query_parser의 물품명과 수정 대상이 일치하지 않습니다: ' +
                                     str([(p['name'], p.get('details')) for p in parsed_items]))
                item['details'] = matches[0].get('details')
        elif parsed_items:
            # 파서가 찾은 새 물품이나 변경된 속성을 keep으로 버리지 않는다.
            for parsed in parsed_items:
                matches = [item for item in self.session.items if item['name'] == parsed['name']]
                if len(matches) != 1 or (parsed.get('details') is not None and parsed['details'] != matches[0].get('details')):
                    raise ValueError('query_parser가 물품 변경을 찾았습니다. replace/add/update로 반영하세요: ' +
                                     str([(p['name'], p.get('details')) for p in parsed_items]))
        return StructureInput.model_validate(payload)

    def request_clarification(self, **arguments):
        payload = ClarificationInput.model_validate(arguments)
        session = self.session
        allowed = CONDITION_FIELDS | {item['item_id'] + '.details' for item in session.items}
        questions, fields = [], set()
        batch_id = uuid4().hex
        for index, row in enumerate(payload.questions):
            field = row.field
            if field not in allowed or field in fields:
                raise ValueError('질문 대상 필드가 잘못되거나 중복되었습니다.')
            if field in session.unknown_fields:
                raise ValueError('사용자가 모른다고 답한 정보는 다시 묻지 마세요. 확인 가능한 범위에서 종료하세요.')
            if field in CONDITION_FIELDS:
                parts = field.split('.')
                known = session.conditions[parts[0]][parts[1]] if len(parts) == 2 else session.conditions[field]
                if known is not None:
                    raise ValueError('이미 확인된 여행 조건을 다시 물을 수 없습니다.')
                if field == 'arrival.airline' and session.conditions['departure']['airline']:
                    raise ValueError('현재 가는 편 항공사는 이미 확인됐습니다. arrival.airline은 귀국편 항공사이며 현재 구간 판정에 필수 정보가 아닙니다.')
            question = row.question.strip()
            key = field + ':' + ''.join(question.split())
            if not question:
                raise ValueError('추가 질문을 비울 수 없습니다.')
            if session.asked.get(key, 0) >= 2 or sum(
                    count for old, count in session.asked.items() if old.startswith(field + ':')) >= 2:
                return self.finish_unresolved('추가 정보를 확인하지 못해 판정을 확정할 수 없습니다. 물품 속성과 여행 조건을 확인한 뒤 다시 질문해 주세요.')
            options = list(dict.fromkeys(x.strip() for x in row.options if x.strip() and x.strip() != '잘 모르겠어요'))[:3]
            questions.append(dict(id=f'{batch_id}_{index}', field=field, question=question,
                                  options=options + ['잘 모르겠어요']))
            fields.add(field)
        for row in questions:
            key = row['field'] + ':' + ''.join(row['question'].split())
            session.asked[key] = session.asked.get(key, 0) + 1
        session.pending = dict(id=batch_id, questions=questions)
        clarification = Clarification.model_validate(session.pending)
        self.response = ChatResponse(session_id=self.session_id, conditions=session.conditions,
                                     message='판정에 필요한 정보를 알려 주세요.', clarification=clarification)
        return dict(status='waiting_for_user', clarification=session.pending)

    def finish_unresolved(self, message):
        """질문 반복 시 사용자 답변을 만들어 내지 않고 확인 필요로 종료한다."""
        self.session.pending = None
        self.session.cards = [Card(id=index, name=item['name'], carry_on=Verdict(status='확인 필요'),
                                   checked=Verdict(status='확인 필요'), notes=message)
                              for index, item in enumerate(self.session.items)]
        self.response = ChatResponse(session_id=self.session_id, conditions=self.session.conditions,
                                     message=message, cards=self.session.cards)
        return dict(status='completed', reason='clarification_limit')

    def scope_supported(self):
        # 미지원 또는 해석 불가능한 장소가 하나라도 있으면 전체 검색으로 우회하지 않는다.
        return all(not (self.session.conditions[leg]['place']) or countries_from_places(
            {leg: self.session.conditions[leg]}) is not None for leg in ('departure', 'arrival'))

    def search_regulations(self):
        session = self.session
        if session.items and session.groups and session.search_revision == session.revision:
            # 같은 상태에서 다시 검색해도 결과가 같다(검색·reranking은 결정적). 중복 호출은 수 초~수십 초를 더한다.
            return dict(supported=self.scope_supported(), revision=session.revision, groups=session.groups, reused=True)
        if not self.scope_supported():
            session.groups = [dict(item=item, query=item['name'], documents=[]) for item in session.items]
        else:
            session.groups = search_by_items(session.items, k=DEFAULT_K, conditions=session.conditions)
        session.search_revision = session.revision
        return dict(supported=self.scope_supported(), revision=session.revision,
                    groups=session.groups)

    def finalize_answer(self):
        session = self.session
        if (session.items and self.scope_supported() and not session.conditions['arrival']['place']
                and 'arrival.place' not in session.unknown_fields):
            raise ValueError('국내선/국제선과 도착국 적용 범위가 확인되지 않았습니다. arrival.place를 먼저 질문하세요. 사용자가 모르면 그 상태를 기록하세요.')
        if session.items and session.search_revision != session.revision:
            raise ValueError('현재 조건으로 검색한 근거가 없습니다. search_regulations를 먼저 실행하세요.')
        if session.pending:
            raise ValueError('아직 답하지 않은 질문이 있습니다. 필요한 남은 정보를 묻거나 모름 상태를 반영하세요.')
        if not session.items:
            self.response = ChatResponse(session_id=self.session_id, conditions=session.conditions,
                                         message='반입하려는 물품을 알려 주세요. 수하물 반입 질문을 도와드릴 수 있습니다.')
            return dict(status='completed')
        question = session.focus + '\n판정은 사용자가 제시한 수량·용량 그대로 수행하세요. 조건을 바꾼 대안은 notes에만 적으세요.'
        if session.unknown_fields:
            question += '\n사용자가 확인할 수 없다고 답한 정보: ' + ', '.join(sorted(session.unknown_fields))
        result = judge_groups(question, session.conditions, session.groups, k=DEFAULT_K)
        cards = []
        for index, (item, used) in enumerate(zip(result['answer']['items'], result['model_used_sources'])):
            cards.append(Card(id=index, name=item['name'], carry_on=Verdict(status=item['carry_on']),
                              checked=Verdict(status=item['checked_baggage']),
                              notes=item['notes'] if used['sources'] else '자료에서 확인하지 못했습니다. 이용 항공사에 확인해 주세요.',
                              evidence=used['sources']))
        summary, suggested = None, None
        if not self.scope_supported():
            cards = [card.model_copy(update={'notes': UNSUPPORTED}) for card in cards]
            summary = UNSUPPORTED
        else:
            from app.chat.presenter import present
            cards, summary, suggested = present(session.focus, session.conditions, cards, session.groups,
                                                 allow_follow_up=False)
        session.cards = cards
        self.response = ChatResponse(session_id=self.session_id, conditions=session.conditions,
                                     summary=summary, cards=cards, suggested_next=suggested)
        return dict(status='completed', items=len(cards))
