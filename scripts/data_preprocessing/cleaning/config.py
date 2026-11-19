"""
# config.py
정제 설정: 원본 위치, 결과 위치, 메타데이터 값 변환표.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "collect"))  # collection_paths
from collection_paths import PIPELINE_DIR, ROOT, STATE_DIR, country_file, raw_dir, resolve_record_path  # noqa: E402

BASE_DIR = ROOT  # 결과: data/{나라}/preprocessed/{kr,law,us,jp}.jsonl
DATA_DIR = ROOT / "data"
WEB_DIR = PIPELINE_DIR / "web"  # 웹 수집기 (일본 HTML·PDF 변환 함수를 빌려 쓴다)

# 웹 수집 결과
WEB_MANIFEST = STATE_DIR / "manifest.jsonl"
WEB_SIGNALS = STATE_DIR / "signals.jsonl"
WEB_SOURCE_FILES = [raw_dir("kr") / "kr.csv", raw_dir("us") / "us.csv"]

# 법령 수집 결과 (scripts/collect/Korea가 만든 것)
LAW_DATA_DIR = DATA_DIR / "Korea" / "korean_law"
SKIP_LAW_FORMS = True  # 별지 서식·빈칸 양식·점검표는 승객 질문과 무관해서 뺀다
LAW_FORM_TITLE = r"(양식|신청서|지정서|점검표)"  # 별표 제목에 이 말이 있으면 양식으로 본다

# 정제 전후 글자 수가 이 비율 아래로 줄면 경고
MIN_KEEP_RATIO = 0.5

# 메타데이터 값 변환 (한글 → 영어 코드)
SOURCE_TYPE = {"공항": "airport", "항공사": "airline", "정부기관": "government", "보도자료": "press"}
ROUTE = {"공통": "all", "국내선": "domestic", "국제선": "international", "미표기": "all", "": "all"}

# 근거 우선순위: 1 법령·고시, 2 항공사·해외 정부기관, 3 공항·국내 정부기관, 9 보도자료·공지
PRIORITY = {"law": 1, "admin_rule": 1, "airline": 2, "government": 2, "airport": 3, "press": 9}
PRIORITY_DOMESTIC_GOVERNMENT = 3  # 국내(KR) 정부기관 웹 문서

# 정제 결과에 이 단어들이 남아 있는지 확인 (나라별)
CHECK_TERMS = {
    "kr": ["보조배터리", "160Wh", "라이터", "100ml"],
    "us": ["power bank", "160", "lighter", "3.4"],
    "law": ["위해물품", "보조배터리", "리튬", "액체"],
    "jp": ["power bank", "lithium", "lighter", "100"],
}
