"""문서 검색. 대상 DB·컬렉션이 비어 있으면 준비된 청크를 먼저 적재한다."""
import re

import chromadb
from app.integrations.embedding import create_embedding_function

from app.settings import ROOT, CHROMA_PATH, CHROMA_COLLECTION, DEFAULT_K, FILTER_FIELDS
from scripts.store_chroma import load_chunks, store_chunks


COUNTRY_ALIASES = {
    'KR': 'KR', 'KOREA': 'KR', 'SOUTH KOREA': 'KR', '한국': 'KR', '대한민국': 'KR',
    'CN': 'CN', 'CHINA': 'CN', '중국': 'CN',
    'JP': 'JP', 'JAPAN': 'JP', '일본': 'JP',
    'UK': 'UK', 'GB': 'UK', 'UNITED KINGDOM': 'UK', '영국': 'UK',
    'US': 'US', 'USA': 'US', 'UNITED STATES': 'US', '미국': 'US',
}

PLACE_COUNTRIES = {
    **COUNTRY_ALIASES,
    **dict.fromkeys(['인천', '인천공항', '인천국제공항', '서울', '김포', '김포공항', '부산', '김해',
                     '제주', 'INCHEON', 'SEOUL', 'ICN', 'GMP', 'PUS', 'CJU'], 'KR'),
    **dict.fromkeys(['베이징', '북경', '상하이', '상해', '광저우', '선전',
                     'BEIJING', 'SHANGHAI', 'GUANGZHOU', 'SHENZHEN', 'PEK', 'PKX', 'PVG', 'SHA', 'CAN', 'SZX'], 'CN'),
    **dict.fromkeys(['도쿄', '동경', '오사카', '나리타', '하네다', '간사이', '후쿠오카', '삿포로', '나고야',
                     'TOKYO', 'OSAKA', 'NARITA', 'HANEDA', 'KANSAI', 'FUKUOKA', 'NRT', 'HND', 'KIX', 'FUK', 'CTS', 'NGO'], 'JP'),
    **dict.fromkeys(['런던', '히스로', '맨체스터', '에든버러', 'LONDON', 'HEATHROW', 'MANCHESTER', 'EDINBURGH',
                     'LHR', 'LGW', 'MAN', 'EDI'], 'UK'),
    **dict.fromkeys(['뉴욕', '로스앤젤레스', '로스엔젤레스', '샌프란시스코', '시애틀', '하와이', '호놀룰루',
                     'NEW YORK', 'LOS ANGELES', 'SAN FRANCISCO', 'SEATTLE', 'HONOLULU',
                     'JFK', 'EWR', 'LAX', 'SFO', 'SEA', 'HNL'], 'US'),
}


# 항공사 이름 → 말뭉치 metadata의 org_name. 말뭉치에 없는 항공사는 매핑하지 않는다.
AIRLINE_ORGS = {
    **dict.fromkeys(['대한항공', 'KOREAN AIR', 'KE'], ['대한항공']),
    **dict.fromkeys(['아시아나', '아시아나항공', 'ASIANA', 'ASIANA AIRLINES', 'OZ'], ['아시아나항공']),
    **dict.fromkeys(['진에어', 'JIN AIR', 'JINAIR', 'LJ'], ['진에어']),
    **dict.fromkeys(['트리니티항공', '티웨이', '티웨이항공', "T'WAY", 'TWAY'], ['트리니티항공(구 티웨이항공)']),
    **dict.fromkeys(['ANA', '전일본공수', '전일본공수(ANA)'], ['ANA']),
    **dict.fromkeys(['JAL', '일본항공'], ['JAL']),
    **dict.fromkeys(['PEACH', '피치', '피치항공'], ['Peach']),
    **dict.fromkeys(['델타', '델타항공', 'DELTA', 'DELTA AIR LINES', 'DL'], ['델타항공']),
    **dict.fromkeys(['아메리칸항공', 'AMERICAN AIRLINES', 'AA'], ['아메리칸항공']),
    **dict.fromkeys(['유나이티드항공', 'UNITED AIRLINES', 'UA'], ['유나이티드항공']),
    **dict.fromkeys(['중국동방항공', 'CHINA EASTERN', 'CHINA EASTERN AIRLINES', 'MU'], ['China Eastern Airlines']),
    **dict.fromkeys(['중국남방항공', 'CHINA SOUTHERN', 'CHINA SOUTHERN AIRLINES', 'CZ'], ['China Southern Airlines']),
}


def airline_orgs(conditions):
    """출발·도착(귀국편) 항공사 중 말뭉치에 있는 항공사의 org_name 목록. 없으면 빈 목록.

    항공사가 지정되지 않았거나 매핑되지 않는 항공사만 있으면 빈 목록이며, 이때는 항공사 자료를 거르지 않는다.
    """
    orgs = []
    for leg in ('departure', 'arrival'):
        airline = (conditions.get(leg) or {}).get('airline')
        if not isinstance(airline, str):
            continue
        for org in AIRLINE_ORGS.get(' '.join(airline.upper().split()), []):
            if org not in orgs:
                orgs.append(org)
    return orgs


def airline_where(conditions):
    """항공사가 아닌 자료는 모두 남기고, 항공사 자료는 질문의 항공사 것만 남기는 조건. 해당 없으면 None."""
    orgs = airline_orgs(conditions)
    if not orgs:
        return None
    return {'$or': [{'source_type': {'$ne': 'airline'}},
                    {'org_name': {'$eq': orgs[0]} if len(orgs) == 1 else {'$in': orgs}}]}


def countries_from_places(conditions):
    """place만 사용한다. 장소 미지정/미지원/모호한 경우 None으로 일반 검색한다.

    도시 목록은 명시적 별칭이다. 단순 부분문자열로 국가를 추측하지 않으며
    홍콩·마카오 등을 중국 본토로 자동 분류하지 않는다.
    """
    countries = []
    for leg in ('departure', 'arrival'):
        place = (conditions.get(leg) or {}).get('place')
        if place is None or (isinstance(place, str) and not place.strip()):
            continue
        if not isinstance(place, str):
            return None
        name = ' '.join(place.upper().split())
        country = PLACE_COUNTRIES.get(name)
        if country is None:
            # Accept explicit combinations such as "일본 도쿄" or "Tokyo, Japan".
            # Every component must be known and agree on the same country.
            parts = [p.strip() for p in re.split(r'[,()/]+', name) if p.strip()]
            if len(parts) == 1:
                parts = name.split()
            resolved = [PLACE_COUNTRIES.get(p) for p in parts]
            if not resolved or None in resolved or len(set(resolved)) != 1:
                return None
            country = resolved[0]
        if country not in countries:
            countries.append(country)
    return countries or None


def build_where(filters):
    """국가는 코드/국가명 및 쉼표 목록을 허용하고 나머지는 정확 일치한다.

    예: {'country': '한국,일본', 'source_type': 'airline'}는
    (KR 또는 JP) 그리고 airline 문서만 선택한다. 국가 미지정 시 전체 검색.
    """
    if set(filters) - FILTER_FIELDS:
        raise ValueError('지원 필터: ' + ', '.join(sorted(FILTER_FIELDS)))
    clauses = []
    for key, value in filters.items():
        if key != 'country':
            clauses.append({key: {'$eq': value}})
            continue
        if not isinstance(value, str):
            raise ValueError('country는 국가 코드/이름을 쉼표로 구분한 문자열이어야 합니다.')
        names = [name.strip().upper() for name in value.split(',')]
        if any(name not in COUNTRY_ALIASES for name in names):
            raise ValueError('지원 국가: KR(한국), CN(중국), JP(일본), UK(영국), US(미국). 복수 국가는 쉼표로 구분하세요.')
        countries = list(dict.fromkeys(COUNTRY_ALIASES[name] for name in names))
        clauses.append({'country': {'$eq': countries[0]} if len(countries) == 1 else {'$in': countries}})
    return ({'$and': clauses} if len(clauses) > 1 else clauses[0]) if clauses else None


def retrieval_where(filters=None, conditions=None, *, airline_scope=False):
    """실제 검색과 진단에서 공유하는 후보 범위 조건.

    airline_scope는 평가 실험용 옵션이며 기본값은 꺼짐이다. 서비스 경로는 이 옵션을 쓰지 않는다.
    """
    where = build_where(filters or {})
    countries = countries_from_places(conditions or {})
    if countries is not None:
        place_where = {'country': {'$eq': countries[0]} if len(countries) == 1 else {'$in': countries}}
        where = {'$and': [where, place_where]} if where else place_where
    if airline_scope:
        scope = airline_where(conditions or {})
        if scope is not None:
            where = {'$and': [where, scope]} if where else scope
    return where


def retrieve(question, filters=None, k=DEFAULT_K, *, conditions=None, airline_scope=False):
    """place의 국가를 알면 해당 국가 내 검색하고, 모르면 일반 검색한다.

    사용자가 별도로 지정한 filters는 두 경우 모두 유지한다.
    """
    if not question.strip() or k < 1:
        raise ValueError('질문을 입력하고 k를 1 이상으로 설정하세요.')
    where = retrieval_where(filters, conditions, airline_scope=airline_scope)
    collection = chromadb.PersistentClient(path=str(ROOT / CHROMA_PATH)).get_or_create_collection(
        name=CHROMA_COLLECTION, embedding_function=create_embedding_function(),
    )
    if not collection.count():
        store_chunks(load_chunks(), collection=collection)
    if not collection.count():
        return []
    result = collection.query(query_texts=[question], n_results=min(k, collection.count()),
                              where=where, include=['documents', 'metadatas'])
    return [{'id': doc_id, 'text': text, 'metadata': metadata}
            for doc_id, text, metadata in zip(result['ids'][0], result['documents'][0], result['metadatas'][0])]


