"""LLM 판정과 metadata 조립 후 공개 응답을 분리한다."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal['가능', '불가', '조건부 가능', '확인 필요']


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class ItemDecision(StrictModel):
    item_id: str = Field(description='입력의 item_id를 그대로 반환. 물품마다 정확히 한 번 반환.')
    carry_on: Status | None = Field(description='검색 본문에 근거한 기내 판정. 근거/정보 부족은 확인 필요. 수하물 반입 판정 자체가 해당하지 않는 질문만 null.')
    checked_baggage: Status | None = Field(description='기내와 독립적으로 판단한 위탁 판정. 근거/정보 부족은 확인 필요. 반입 판정 자체가 해당하지 않을 때만 null.')
    notes: str | None = Field(description='검색 본문에 근거한 조건·이유 및 부족한 정보. 없으면 null. 출처 URL이나 문서명은 작성하지 않음.')
    evidence_ids: list[str] = Field(description='해당 물품에 제공된 문서 중 실제 판정/설명에 사용한 evidence_id만 반환. 예: item_1_doc_1. 사용한 근거가 없으면 빈 배열.')


class DecisionResponse(StrictModel):
    items: list[ItemDecision] = Field(description='제공된 모든 물품의 판정. 입력 item_id를 누락·중복·추가하지 않음.')


class BaggageItem(StrictModel):
    name: str
    carry_on: Status | None
    checked_baggage: Status | None
    notes: str | None
    source: list[dict[str, Any]]


class BaggageResponse(StrictModel):
    items: list[BaggageItem]
