"""사용자 질의의 여행 조건과 물품 목록 구조."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TravelLeg(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    place: str | None = Field(description="질문에 명시된 국가 또는 도시. 없으면 null.")
    airline: str | None = Field(description="해당 항공편의 항공사. 알 수 없으면 null.")


class TravelConditions(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    departure: TravelLeg = Field(description="출발 장소와 가는 편 항공사.")
    arrival: TravelLeg = Field(description="도착 장소와 오는 편 항공사. 편도이거나 오는 편 정보가 없으면 항공사는 null.")
    travel_date: str | None = Field(description="질문에 적힌 여행일 표현 그대로. 상대 날짜를 계산하지 않으며 없으면 null.")
    baggage_type: Literal['기내', '위탁', '둘 다'] | None = Field(description="질문의 수하물 구분. 기내와 위탁을 모두 묻거나 물품마다 다르면 둘 다. 없으면 null.")


class QueryItem(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    name: str = Field(description="질문에 언급된 물품의 일반적인 이름.")
    details: str | None = Field(description="질문에 명시된 용량·수량·종류·크기·재질 등 물품 속성. 임의로 보충하지 않으며 없으면 null.")


class ParsedQuery(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    conditions: TravelConditions = Field(description="질문에서 추출한 여행 조건.")
    items: list[QueryItem] = Field(description="질문에 언급된 물품별 목록. 물품이 없으면 빈 배열.")
