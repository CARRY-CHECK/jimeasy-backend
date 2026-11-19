"""
# adapters/law.py
법령(data/Korea/korean_law의 all_laws_final.json)을 읽어 정제한다.
API 값이 섞인 줄과 개정 이력을 지우고, 장·절·조를 제목으로 바꾸고, 박스 문자 표를 정리한다.
"""

import json
import re

import config
import steps

# 법령 수집 코드(collect_laws.py)의 순서와 같다: 01~03 법령, 04~06 고시
LAW_FILES = [
    ("law", "항공보안법"),
    ("law", "항공보안법 시행령"),
    ("law", "항공보안법 시행규칙"),
    ("admin_rule", "항공기 내 반입금지 위해물품"),
    ("admin_rule", "액체·분무·겔류 등 항공기내 휴대 반입 금지물질 운영기준"),
    ("admin_rule", "항공위험물운송기술기준"),
]

# 본문에 섞인 API 값 줄 (숫자, 날짜, Y/N, 조문 코드, 구분 단어, 개정일 목록)
_NOISE_LINE = re.compile(
    r"^(\d{1,7}|\d{8}|[YN]|조문|전문|개정|신설|삭제|"
    r"\d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.?(,\s*\d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.?)*)$"
)
# 번호만 있는 줄 ("1.", "①", "가.") - 다음 줄에 같은 번호로 내용이 또 나온다
_MARKER_LINE = re.compile(r"^(\d+\.|[①-⑳]|[가-하]\.)$")
# 개정 이력 표시
_HISTORY = re.compile(r"<(개정|신설|삭제|전문개정|본조신설|제목개정|타법개정|종전)[^>]*>|\[(전문개정|본조신설|제목개정|종전|시행일|조번호변경)[^\]]*\]")
_ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?\s*\(([^)]*)\))\s*(.*)$")
_BOX = re.compile(r"[┌┐└┘├┤┬┴┼─━│┃═║╔╗╚╝╠╣╦╩╬]")
_BOX_ALL = re.compile(r"[─-╿]")
_CHECKBOX_ONLY = re.compile(r"^[\s□☐■○◯]+$")  # 체크박스·서식 기호만 있는 칸·줄  # 표 테두리 문자 전체 (가는 선·굵은 선·이중 선)


def _iso(yyyymmdd: str) -> str:
    s = re.sub(r"\D", "", str(yyyymmdd or ""))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else ""


def _info(i: int, kind: str, name: str) -> dict:
    """body/*.json(기본정보)과 search/*.json(일련번호)에서 메타데이터 재료를 꺼낸다."""
    body = json.loads((config.LAW_DATA_DIR / "body" / f"{i:02d}.json").read_text(encoding="utf-8"))
    root = body.get("법령") or body.get("AdmRulService") or {}
    info = root.get("기본정보") or root.get("행정규칙기본정보") or {}
    org = info.get("소관부처") or info.get("소관부처명") or ""
    org = org.get("content", "") if isinstance(org, dict) else org
    serial = info.get("행정규칙일련번호", "")
    if not serial:  # 법령은 일련번호가 검색 결과에 있다 (시행일이 가장 최근인 것)
        search = json.loads((config.LAW_DATA_DIR / "search" / f"{i:02d}.json").read_text(encoding="utf-8"))
        items = search.get("LawSearch", {}).get("law", [])
        items = [items] if isinstance(items, dict) else items
        same = [x for x in items if x.get("법령명한글") == name]
        serial = max(same, key=lambda x: x.get("시행일자", ""))["법령일련번호"] if same else ""
    url = (f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={serial}" if kind == "law"
           else f"https://www.law.go.kr/LSW/admRulLsInfoP.do?admRulSeq={serial}")
    return {
        "org": org, "serial": str(serial), "url": url,
        "effective": _iso(info.get("시행일자", "")),
        "updated": _iso(info.get("공포일자") or info.get("발령일자", "")),
        "raw_file": f"data/Korea/korean_law/body/{i:02d}.json",
    }


def _flat(v) -> list[str]:
    """API 값(문자열·목록·dict가 섞임)을 글 줄 목록으로 편다."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [s for x in v for s in _flat(x)]
    if isinstance(v, dict):
        return _flat(v.get("content", ""))
    return [str(v)]


def render_law_body(i: int) -> str:
    """법령(01~03)은 원본 JSON의 조문 구조(조 → 항 → 호 → 목)를 순서대로 읽어 글로 만든다.
    펼쳐진 글(all_laws_final)은 조문 제목이 항목 뒤에 오는 경우가 있어 쓰지 않는다."""
    body = json.loads((config.LAW_DATA_DIR / "body" / f"{i:02d}.json").read_text(encoding="utf-8"))
    units = body["법령"]["조문"]["조문단위"]
    units = [units] if isinstance(units, dict) else units
    out = []
    for u in units:
        head = _HISTORY.sub("", " ".join(_flat(u.get("조문내용")))).strip()
        if u.get("조문여부") == "전문":  # 장·절·관 제목
            level = 2 if re.match(r"^제\d+장", head) else 3 if re.match(r"^제\d+절", head) else 4
            if head:
                out.append("#" * level + " " + re.sub(r"\s+", " ", head))
            continue
        m = _ARTICLE.match(head)
        if m:
            if re.fullmatch(r"삭제\s*\.?", m.group(3).strip()):
                continue  # 삭제된 조문
            out.append(f"##### {m.group(1)}")
            if m.group(3).strip():
                out.append(m.group(3).strip())
        elif head:
            out.append(head)
        hangs = u.get("항") or []
        for h in (hangs if isinstance(hangs, list) else [hangs]):
            if not isinstance(h, dict):
                continue
            out += [_HISTORY.sub("", s).strip() for s in _flat(h.get("항내용"))]
            hos = h.get("호") or []
            for ho in (hos if isinstance(hos, list) else [hos]):
                if not isinstance(ho, dict):
                    continue
                out += [_HISTORY.sub("", s).strip() for s in _flat(ho.get("호내용"))]
                moks = ho.get("목") or []
                for mk in (moks if isinstance(moks, list) else [moks]):
                    if isinstance(mk, dict):
                        out += [_HISTORY.sub("", s).strip() for s in _flat(mk.get("목내용"))]
    return "\n".join(line for line in out if line)


def clean_body(text: str) -> str:
    """법령·고시 본문: API 값 줄·번호만 있는 줄·개정 이력을 지우고 장·절·조를 제목으로 바꾼다."""
    text = text.replace("　", "\n")
    # 한 줄에 여러 조문이 붙어 있으면 조문마다 줄을 나눈다 (「…」 제14조제5항 같은 인용은 괄호가 없어서 안 나뉨)
    text = re.sub(r"(?<=[.다])\s*(제\d+조(?:의\d+)?\()", r"\n\1", text)
    # 문장 끝에 바로 붙은 번호 항목도 줄을 나눈다 ("…같다.1. ‘액체…’2. …", "…한다.나. …")
    text = re.sub(r"(?<=[가-힣]\.)\s*(\d{1,2}\.)(?=\s?[‘'\"「가-힣])", r"\n\1", text)
    text = re.sub(r"(?<=[가-힣]\.)\s*([가-하]\.)(?=\s)", r"\n\1", text)
    out, last_title = [], ""
    for raw in text.split("\n"):
        line = _HISTORY.sub("", raw).strip()
        if not line or _NOISE_LINE.match(line) or _MARKER_LINE.match(line):
            continue
        if line == last_title:  # "제1조(목적)" 다음 줄에 다시 나오는 "목적"
            continue
        if re.match(r"^제\d+장\s", line):
            out.append(f"## {line}")
            continue
        if re.match(r"^제\d+절\s", line):
            out.append(f"### {line}")
            continue
        m = _ARTICLE.match(line)
        if m:
            if re.fullmatch(r"삭제\s*\.?", m.group(3).strip()):
                continue  # 삭제된 조문
            last_title = m.group(2).strip()
            out.append(f"#### {m.group(1)}")
            if m.group(3).strip():
                out.append(m.group(3).strip())
            continue
        out.append(line)
    return "\n".join(out)


def _merge_row(lines: list[str]) -> str:
    """표 한 행(가로 구분선 사이의 여러 줄)을 칸별로 이어 붙여 '칸 | 칸' 한 줄로 만든다."""
    rows = []
    for line in lines:
        parts = re.split(r"[│┃║]", line)
        if parts and not parts[0].strip():
            parts = parts[1:]  # 왼쪽 테두리 바깥
        if parts and not parts[-1].strip():
            parts = parts[:-1]  # 오른쪽 테두리 바깥
        rows.append([_CHECKBOX_ONLY.sub("", _BOX_ALL.sub("", p)).strip() for p in parts])  # 체크박스만 있는 칸은 빈칸으로
    width = max((len(r) for r in rows), default=0)
    cols = [" ".join(r[i] for r in rows if i < len(r) and r[i]) for i in range(width)]
    cols = [re.sub(r"\s{2,}", " ", c).strip() for c in cols]
    return " | ".join(c for c in cols if c)


def clean_appendix(text: str) -> str:
    """별표: 박스 문자 표를 행 단위 '칸 | 칸' 줄로 바꾸고, 테두리 문자·개정 이력을 지운다.
    가로 구분선(├─┼─┤ 등) 사이의 여러 줄은 한 행이라서 칸별로 이어 붙인다."""
    out, row = [], []

    def flush() -> None:
        if row:
            merged = _merge_row(row)
            if merged:
                out.append(merged)
            row.clear()

    for raw in text.split("\n"):
        line = _HISTORY.sub("", raw).rstrip()
        if re.search(r"[│┃║]", line):
            row.append(line)  # 표 안의 줄: 구분선이 나올 때까지 모은다
            continue
        if _BOX_ALL.search(line) and not _BOX_ALL.sub("", line).strip():
            flush()  # 테두리·구분선만 있는 줄 = 행의 끝
            continue
        flush()
        line = re.sub(r"\s{2,}", " ", _BOX_ALL.sub("", line)).strip()
        if re.match(r"^■.*\[별(표|지)", line):
            continue  # "■ 항공보안법 시행규칙 [별표] <개정 …>" 머리말 (제목에 이미 있음)
        line = _CHECKBOX_ONLY.sub("", line).strip()
        if line:
            out.append(line)
    flush()
    return "\n".join(out)


def documents():
    """정제할 법령 문서를 하나씩 (원본 글, 정제된 글, 메타데이터)로 돌려준다. 내용 없는 별표는 건너뛴다."""
    recs = json.loads((config.LAW_DATA_DIR / "processed" / "all_laws_final.json").read_text(encoding="utf-8"))
    infos = {name: (kind, _info(i, kind, name)) for i, (kind, name) in enumerate(LAW_FILES, 1)}
    count: dict[str, int] = {}
    for r in recs:
        name = r["document_name"].strip()
        kind, info = infos[name]
        raw = r.get("content", "")
        if r["section_type"] == "본문":
            section, doc_key, title = "body", "본문", name
            idx = [n for _, n in LAW_FILES].index(name) + 1
            text = render_law_body(idx) if kind == "law" else clean_body(raw)
        else:
            n = count[name] = count.get(name, 0) + 1
            ap_title = (r.get("appendix_title") or "").strip()
            section, doc_key, title = "appendix", f"별표{n}", f"{name} [별표 {n}] {ap_title}".strip()
            text = clean_appendix(raw)
            if len(text) < 80 and re.search(r"(첨부파일|pdf|PDF)", text):
                print(f"  건너뜀 (내용 없는 별표, 첨부파일 안내뿐): {title}")
                continue
            if config.SKIP_LAW_FORMS and ("[별지" in text[:300] or re.search(config.LAW_FORM_TITLE, ap_title)):
                print(f"  건너뜀 (서식·양식·점검표): {title}")
                continue
            if len(re.sub(r"\(별표\s*\d+\)|\s", "", text)) < 40:
                print(f"  건너뜀 (그림만 있는 별표, 글 없음): {title}")
                continue
        meta = {
            "doc_id": f"law.{name}.{doc_key}",
            "title": title,
            "source_name": title,
            "org_name": info["org"],
            "source_type": kind,
            "priority": config.PRIORITY[kind],
            "role": "evidence",
            "country": "KR",
            "language": "ko",
            "route": "all",
            "section_type": section,
            "version": info["serial"],
            "effective_date": info["effective"],
            "updated_date": info["updated"],
            "collected_at": (r.get("collected_at") or "")[:10],
            "source_url": info["url"],
            "raw_file": info["raw_file"],
        }
        text = steps.normalize(steps.dedupe_blocks(text), title)
        yield raw, text, meta
