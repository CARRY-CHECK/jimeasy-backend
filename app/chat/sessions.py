"""세션별 대화 상태를 프로세스 메모리(dict)에 보관한다.
서버 재시작 시 사라지며, 외부 저장소로 바꿀 때는 SessionStore만 교체한다.
"""
from dataclasses import dataclass, field
from uuid import uuid4
from threading import RLock

from app.chat.conditions import empty_conditions


@dataclass
class Session:
    question: str = ''                           # 물품을 처음 물은 질문
    conditions: dict = field(default_factory=empty_conditions)
    groups: list = field(default_factory=list)   # 물품별 검색 문서 (판정만 다시 할 때 재사용)
    cards: list = field(default_factory=list)    # groups와 같은 순서의 카드, 인덱스 = Card.id
    history: list = field(default_factory=list)
    items: list = field(default_factory=list)    # 현재 답변 대상. 과거 결과는 history에 보존
    pending: dict | None = None
    unknown_fields: set = field(default_factory=set)
    asked: dict = field(default_factory=dict)    # 상태/필드별 질문 반복 방지
    revision: int = 0
    search_revision: int | None = None
    next_item_id: int = 1
    focus: str = ''
    turn_id: str = ''
    last_diagnostics: dict = field(default_factory=dict)

    def snapshot(self):
        """모델에 제공하는 현재 상태. 문서 본문은 검색 도구에서만 제공한다."""
        return dict(conditions=self.conditions, items=self.items, question=self.focus,
                    pending=self.pending, unknown_fields=sorted(self.unknown_fields),
                    revision=self.revision, search_revision=self.search_revision)

    def structure(self, update):
        from copy import deepcopy
        from app.chat.conditions import apply_changes, CONDITION_FIELDS
        conditions = apply_changes(self.conditions, update.changes)
        # 합의한 서비스 기본값. 명시적으로 출발지를 해제하면 다시 보충하지 않는다.
        if not self.history and not self.focus and not conditions['departure']['place'] and not any(
                change.field == 'departure.place' for change in update.changes):
            conditions['departure']['place'] = '한국'
        existing = {item['item_id']: item for item in self.items}
        incoming, next_id, seen = [], self.next_item_id, set()
        for item in update.items:
            identity = item.item_id
            if identity is None:
                if update.mode in ('update', 'keep'):
                    raise ValueError('기존 물품 수정에는 저장된 item_id가 필요합니다.')
                identity, next_id = f'item_{next_id}', next_id + 1
            elif identity not in existing:
                raise ValueError('저장되지 않은 물품 ID입니다.')
            if identity in seen or not item.name.strip():
                raise ValueError('물품 ID 중복 또는 빈 물품명입니다.')
            seen.add(identity)
            incoming.append(dict(item_id=identity, name=item.name.strip(), details=item.details))
        if update.mode == 'keep':
            if incoming:
                raise ValueError('keep에는 items를 비워 주세요.')
            items = deepcopy(self.items)
        elif update.mode == 'replace':
            items = incoming
        elif update.mode == 'add':
            if any(item['item_id'] in existing for item in incoming):
                raise ValueError('추가 물품은 새 ID를 사용해야 합니다.')
            items = deepcopy(self.items) + incoming
        else:
            changed = {item['item_id']: item for item in incoming}
            items = [changed.get(item['item_id'], deepcopy(item)) for item in self.items]
        valid_fields = CONDITION_FIELDS | {item['item_id'] + '.details' for item in items}
        if set(update.unknown_fields) - valid_fields:
            raise ValueError('모름 상태의 필드가 현재 질의에 없습니다.')
        if update.mode == 'update' and (not incoming or all(
                item == existing[item['item_id']] for item in incoming)):
            raise ValueError('update는 물품 속성의 실제 변경을 포함해야 합니다. 수량 정정을 details에도 반영하세요.')
        changed_fields = {change.field for change in update.changes}
        changed_fields |= {item['item_id'] + '.details' for item in incoming
                           if item['item_id'] not in existing or item != existing[item['item_id']]}
        unknown = (self.unknown_fields - changed_fields) | set(update.unknown_fields)
        unknown &= valid_fields
        # 질문 초점도 검색에 영향을 줄 수 있으므로 변경 시 재검색한다.
        question = update.question.strip()
        if not question:
            raise ValueError('현재 질문을 비울 수 없습니다.')
        if (conditions, items, question) != (self.conditions, self.items, self.focus):
            self.revision += 1
            self.search_revision = None
            self.groups, self.cards = [], []
        self.conditions, self.items, self.focus = conditions, items, question
        self.next_item_id, self.unknown_fields = next_id, unknown
        if update.mode == 'replace':
            self.pending = None
        if self.pending:
            remaining = [q for q in self.pending['questions']
                         if q['field'] not in changed_fields | unknown and q['field'] in valid_fields]
            self.pending = dict(self.pending, questions=remaining) if remaining else None


class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._locks = {}
        self._guard = RLock()

    def lock_for(self, session_id):
        with self._guard:
            return self._locks.setdefault(session_id, RLock())

    def put(self, session_id, session):
        self._sessions[session_id] = session

    def get(self, session_id=None):
        """없는 ID(서버 재시작 등)는 같은 ID로 새로 만든다."""
        session_id = session_id or uuid4().hex
        return session_id, self._sessions.setdefault(session_id, Session())

    def reset(self, session_id):
        with self.lock_for(session_id):
            self._sessions[session_id] = Session()
            return self._sessions[session_id]
