"""챗봇 요청·응답 구조. 판정 값과 여행 조건은 기존 RAG 스키마를 재사용한다."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.baggage import Status
from app.schemas.query import TravelConditions


class ChatModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Choice(ChatModel):
    """되묻기 선택지 클릭. 카드 id와 선택지 순번만 보낸다."""
    card: int = Field(ge=0)
    option: int = Field(ge=0)
    question_id: str | None = None


class ChatRequest(ChatModel):
    session_id: str | None = None
    message: str | None = None
    choice: Choice | None = None

    @model_validator(mode='after')
    def one_input(self):
        if (self.message is None) == (self.choice is None):
            raise ValueError('message와 choice 중 하나만 보내세요.')
        return self


class Verdict(ChatModel):
    status: Status | None
    condition: str | None = None


class FollowUp(ChatModel):
    question: str
    options: list[str]


class Card(ChatModel):
    id: int                    # 새 질문 전까지 세션 안에서 유지되는 카드 번호
    name: str
    carry_on: Verdict
    checked: Verdict
    notes: str | None = None   # RAG 판정의 설명 원문
    follow_up: FollowUp | None = None
    evidence: list[dict[str, Any]] = []


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class CardText(StrictModel):
    card_id: int = Field(description='입력의 card_id를 그대로 반환.')
    carry_on_condition: str | None = Field(description='기내 판정 옆 짧은 조건. notes에 근거가 없으면 null.')
    checked_condition: str | None = Field(description='위탁 판정 옆 짧은 조건. notes에 근거가 없으면 null.')
    follow_up_basis: str | None = Field(description='질문의 기준이 적힌 notes 원문 구절을 글자 그대로 복사. 질문이 없으면 null.')
    follow_up_question: str | None = Field(description='판정을 확정할 정보 하나를 묻는 질문. 해당 없으면 null.')
    follow_up_options: list[str] = Field(description='질문의 짧은 선택지 2~3개. 질문이 없으면 빈 배열.')


class Presentation(StrictModel):
    """③ 문장 만들기 LLM 출력. 판정 값은 포함하지 않는다."""
    summary: str = Field(description='질문에 대한 결론 한 문장.')
    cards: list[CardText] = Field(description='입력된 모든 카드의 문장. card_id를 누락·중복·추가하지 않음.')
    suggested_next: str | None = Field(description='이어서 물어볼 만한 다른 물품 질문 하나.')


class ChatResponse(ChatModel):
    session_id: str
    conditions: TravelConditions
    summary: str | None = None
    message: str | None = None
    cards: list[Card] = []
    suggested_next: str | None = None
    clarification: 'Clarification | None' = None
    turn_id: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ClarificationQuestion(ChatModel):
    id: str
    field: str
    question: str
    options: list[str] = Field(default_factory=list)


class Clarification(ChatModel):
    id: str
    questions: list[ClarificationQuestion]


class ConditionChange(StrictModel):
    field: Literal['departure.place', 'departure.airline', 'arrival.place', 'arrival.airline',
                   'travel_date', 'baggage_type']
    value: str | None = Field(description='사용자가 조건을 해제하거나 미정으로 바꿀 때만 null. 값이 있는 조건은 서버가 구조화 결과로 적용하므로 새 값을 넣지 않음.')


class ToolItem(StrictModel):
    item_id: str | None = Field(description='기존 물품은 저장된 ID, 새 물품은 null.')
    name: str = Field(description='수정 시 저장된 이름 그대로(구조화 결과와 이름으로 대조됨). 새 물품 이름은 구조화 결과가 적용됨.')
    details: str | None = Field(description='용량·수량 등 현재 명시 속성 전체. 수정 시 기존 속성 중 바뀌지 않은 값도 보존하며, 최종 값은 구조화 결과가 적용됨. 예: 1개→4개.')


class StructureInput(StrictModel):
    changes: list[ConditionChange] = Field(description='조건 해제(value:null)만 포함. 값이 있는 조건은 서버가 구조화 결과로 적용하며 미언급은 포함하지 않음.')
    mode: Literal['replace', 'add', 'update', 'keep'] = Field(description='새 대상 replace, 추가 add, 기존 물품 속성 수정 update, 물품 변경 없음 keep.')
    items: list[ToolItem] = Field(description='replace/add/update에 필요한 물품. 수량 정정은 update로 해당 ID와 수정한 details를 포함. keep이면 빈 배열.')
    question: str = Field(description='현재 대상 물품에 대한 질문. 사용자 원문 표현을 유지하고 생략된 대상·수량만 보충. "최대 몇 개", "충분한지" 같은 질문 유형을 바꾸지 않으며 규정·수치·판정을 추가하지 않음.')
    unknown_fields: list[str] = Field(description='사용자가 이번 발화에서 모름/거절을 명시한 필드만. 미언급 정보를 넣지 않으며 보통 빈 배열.')


class ClarificationInputQuestion(StrictModel):
    field: str = Field(description='여행 조건 경로 또는 item_id.details.')
    question: str
    options: list[str] = Field(description='짧은 답 2~3개. 확인한 근거의 수치·구분으로 나눔. "잘 모르겠어요"는 서버가 추가하므로 넣지 않음.')


class ClarificationInput(StrictModel):
    questions: list[ClarificationInputQuestion] = Field(min_length=1, max_length=3)


class EmptyInput(StrictModel):
    pass


ChatResponse.model_rebuild()
