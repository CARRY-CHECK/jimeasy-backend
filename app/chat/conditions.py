"""대화 중 여행 조건 누적. 새로 말한 값이 이전 값을 대체하고 말하지 않은 값은 유지한다."""
from app.schemas.query import TravelConditions

LEGS = ('departure', 'arrival')


def empty_conditions():
    leg = {'place': None, 'airline': None}
    return TravelConditions.model_validate(
        {'departure': dict(leg), 'arrival': dict(leg), 'travel_date': None, 'baggage_type': None}).model_dump()


def merge_conditions(previous, new):
    merged = {leg: {key: new[leg][key] or previous[leg][key] for key in previous[leg]} for leg in LEGS}
    for key in ('travel_date', 'baggage_type'):
        merged[key] = new[key] or previous[key]
    return TravelConditions.model_validate(merged).model_dump()


CONDITION_FIELDS = frozenset(('departure.place', 'departure.airline', 'arrival.place',
                              'arrival.airline', 'travel_date', 'baggage_type'))


def apply_changes(previous, changes):
    """미언급은 유지하고 명시적 null은 해제한다. 이전 상태를 변경하지 않는다."""
    from copy import deepcopy
    merged = deepcopy(previous)
    seen = set()
    for change in changes:
        field, value = change.field, change.value
        if field not in CONDITION_FIELDS or field in seen:
            raise ValueError('여행 조건 변경 필드가 잘못되거나 중복되었습니다.')
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError('빈 조건 대신 해제는 null을 사용하세요.')
        seen.add(field)
        if '.' in field:
            leg, key = field.split('.')
            merged[leg][key] = value
        else:
            merged[field] = value
    return TravelConditions.model_validate(merged).model_dump()
