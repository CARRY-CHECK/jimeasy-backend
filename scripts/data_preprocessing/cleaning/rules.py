"""
# rules.py
정제 규칙 모음: 모든 문서에 쓰는 공통 규칙과 출처(source_id)별 규칙.
사이트가 늘면 SITE_RULES에 한 항목만 추가한다.
"""

# 모든 문서에서 지울 줄 (정규식, 제목 줄은 지우지 않음)
COMMON_DROP_LINES = [
    r"pdf 새창 열기",
    r"^\[동영상:",
]

# 출처별 규칙
#   drop_lines:      지울 줄 (정규식)
#   drop_head_lines: 첫 제목이 나오기 전 부분에서만 지울 줄 (정확히 같은 글자)
#   drop_sections:   지울 제목 (그 제목부터 같은·상위 단계 제목 전까지 통째로)
#   drop_blocks:     [시작 줄 정규식, 끝 줄 정규식] 시작 줄부터 끝 줄 직전까지 지움 (끝 줄은 남김)
#   line_to_heading: [줄 정규식, 제목 글자, 제목 단계] 평문 구분 줄을 제목으로 바꿈 (청킹 때 맥락이 남도록)
SITE_RULES = {
    "icn_restricted": {
        "drop_sections": ["여객 포기물품 처리"],
    },
    "koreanair_restricted": {
        # 맨 위 탭 이름
        "drop_head_lines": ["리튬배터리", "항공기 반입금지 물품", "제한적 기내 반입 물품", "위탁 수하물 제한 물품", "기타"],
        "drop_lines": [r"^한국교통안전공단 홈페이지$"],
        "drop_sections": ["MacBook Pro 배터리 리콜 대상 제품 운송 제한 사항 안내"],
    },
    "asiana_restricted": {
        # 나라 선택 메뉴는 지우고, 나라별 액체류 표는 나라마다 제목을 달아 남긴다
        "drop_lines": [r"^대한민국 대만 독일"],
        "line_to_heading": [[r"^(\S+) 의 조회 결과입니다\.$", r"\1 출발·환승 시", 5]],
    },
    "trinity_restricted": {
        # 탭 버튼 글자와 탭 목록
        "drop_lines": [
            r"^- (객실|위탁수하물|주요 운송제한물품|리튬배터리 물품)$",
            r"^- (폭발성, 인화성, 유독성 물질|무기로 사용 될 수 있는 물품|일반 생활 용품 및 의료용품)$",
            r"^- (액체류 / 위생용품 / 욕실용품 / 의약품류|전자담배 운송 규정|분말류 \(파우더류\))$",
            r"^- (리튬 배터리 운송 규정\((휴대|위탁) 수하물\)|리튬 배터리 방식 전동 휠체어|스마트가방\(Smart Luggage\) 운송규정)$",
        ],
    },
}

# 보도자료(정책브리핑 PDF 본문)에서 지울 줄
PRESS_DROP_LINES = [
    r"^보도자료$",
    r"^보도시점\s*:",
    r"^담당 부서",
    r"^(책임자|담당자)\s",
    r"^\S+과 (책임자|담당자)",
    r"^[➊➋➌➍➎➏]$",
]


# ---------------------------------------------------------------
# 미국
#   replace:         [[정규식, 바꿀 글자]] 줄 안의 일부를 바꿈 (예: 링크 꼬리말 지우기)
#   updated_date:    [정규식, 날짜 형식] 문서에 적힌 수정일을 찾아 updated_date로 (찾은 줄은 drop_lines로 지움)
#   table_to_sections: True면 표 한 줄을 "## 첫 칸 + 나머지 칸 목록"으로 바꿈 (물품별 표)
#   converter:       원본 HTML에서 직접 만드는 출처 (TSA)
#   dedupe:          False면 중복 정리를 안 함. 물품별 문서는 같은 설명이 물품마다 있어야 해서
# ---------------------------------------------------------------
COMMON_DROP_LINES += [
    r"^\[삽입 콘텐츠",
    r"^Back to top$",
]

_DELTA_CATEGORIES = (r"^- (Ammunition, Explosives & Firearms|Battery or Fuel-Powered|Food & Alcohol|Personal Care & Medical Items|"
                     r"Automotive, Household & Personal Electronic Items|Robotic, Machine, Household Items & Other|"
                     r"Sporting & Leisure Goods|Tobacco, E-Cigarettes, Marijuana|Other Items)$")
for _sid in ("delta_overview", "delta_firearms", "delta_battery", "delta_food_alcohol", "delta_other",
             "delta_personal_medical", "delta_electronics", "delta_sporting", "delta_tobacco"):
    SITE_RULES[_sid] = {"drop_lines": [_DELTA_CATEGORIES, r"^Next ", r"^\[이미지: Exclamation point\]", r"^Review Prohibited Items$", r"^\.$"]}

_AA_LINK = r" Opens another site in a new window that may not meet accessibility guidelines"
SITE_RULES.update({
    "tsa_whatcanibring": {"converter": "tsa_items", "dedupe": False},
    "faa_packsafe_chart": {
        "updated_date": [r"Updated:\s*(\d{1,2}/\d{1,2}/\d{2})", "%m/%d/%y"],
        "drop_lines": [r"^Updated:\s*\d{1,2}/\d{1,2}/\d{2}"],
        "table_to_sections": True,
        "dedupe": False,
    },
    "faa_packsafe": {
        "drop_head_lines": ["- Interactive PackSafe Chart", "- Printable PackSafe Chart",
                            "- Not All Batteries Are Safe to Travel With",
                            "- Damaged or Recalled Batteries and Battery-Powered Devices"],
        "updated_date": [r"Last updated:\s*\w+,\s*(\w+ \d{1,2}, \d{4})", "%B %d, %Y"],
        "drop_lines": [r"^Last updated:"],
    },
    "faa_lithium": {
        "updated_date": [r"Last updated:\s*\w+,\s*(\w+ \d{1,2}, \d{4})", "%B %d, %Y"],
        "drop_lines": [r"^Last updated:"],
    },
    "faa_ped": {
        "updated_date": [r"Last updated:\s*\w+,\s*(\w+ \d{1,2}, \d{4})", "%B %d, %Y"],
        "drop_lines": [r"^Last updated:"],
    },
    "aa_restricted": {
        "drop_head_lines": ["1. Home", "2. At the airport", "3. Bags", "4. Restricted items"],
        "replace": [[_AA_LINK, ""]],
        "drop_lines": [r"^- (Bags|Checked bag policy|Carry-on bags|Delayed or damaged bags|Track your bags|Batteries|Gas-powered tools and generators)$",
                       r"^Learn more about traveling with:$", r"^- (TSA prohibited items|FAA Pack Safe Hazmat restrictions)$"],
        "line_to_heading": [[r"^(.+) This content can be expanded$", r"\1", 3]],
    },
    "aa_hazmat": {
        "drop_head_lines": ["1. Home", "2. Hazardous materials policy"],
        "replace": [[_AA_LINK, ""]],
    },
    "united_dangerous": {
        "drop_lines": [r"^Select an item$", r"^Categories "],
        "drop_sections": ["Select item", "Related links"],
    },
})


# ---------------------------------------------------------------
# 일본 (키는 jp.{출처})
#   drop_breadcrumb: True면 문서 맨 앞의 번호 메뉴 경로(1. … 2. …)를 지움
#   heading_strip:   제목에서 지울 글자 (정규식, 예: 접힌 메뉴 버튼의 "Open")
# ---------------------------------------------------------------
COMMON_DROP_LINES += [r"^Adobe Acrobat Reader is required to view PDF format files\.$"]

SITE_RULES.update({
    "jp.ana": {"drop_breadcrumb": True},
    "jp.jal": {"heading_strip": [r"\s+Open$"]},
    "jp.peach": {"drop_lines": [r"^- (Prohibited carry-on/checked items|Carry-on/checked items requiring special attention|Plant Quarantine|Animal Quarantine)$"]},
    "jp.jnto": {"drop_lines": [r"^JNTO - Japan National Tourism Organization$"]},
})
