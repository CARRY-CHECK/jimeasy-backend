"""
# adapters/japan.py
일본 수집 결과(data/Japan/raw)를 읽어 정제할 문서를 만든다. 저장된 HTML은 웹 수집기의 변환 함수로 Markdown으로,
MLIT PDF는 글자를 뽑아 Markdown으로 바꾼다. JNTO 공지는 관련 있는 것(candidate)만 참고용으로 넣는다.
"""

import json
import re
import sys

import config
import rules
import steps

JP_DIR = config.raw_dir("jp")

# 출처별 본문 위치 (앞에서부터 찾음)
SELECTORS = {
    "ana": [".asw-sidenavi__content", ".contents-asw"],
    "jal": ["div.contentsInr"],
    "peach": ["main#main"],
    "jnto": ["article"],
}
ORG = {"ana": "ANA", "jal": "JAL", "peach": "Peach", "mlit": "일본 국토교통성(MLIT)", "jnto": "일본정부관광국(JNTO)"}
SOURCE_TYPE = {"ana": "airline", "jal": "airline", "peach": "airline", "mlit": "government", "jnto": "press"}


def _load_web_tools():
    """웹 수집기(scripts/collect/web)의 HTML·PDF 변환 함수를 불러온다.
    웹 수집기에도 config 모듈이 있어서, 불러오는 동안만 정제 쪽 모듈을 잠시 비켜 둔다."""
    web = str(config.WEB_DIR)
    names = ("config", "storage", "notice", "pdf_md", "crawl")
    saved = {n: sys.modules.pop(n, None) for n in names}
    sys.path.insert(0, web)
    try:
        import crawl as web_crawl
        import pdf_md as web_pdf
    finally:
        sys.path.remove(web)
        for n in names:
            sys.modules.pop(n, None)
            if saved[n] is not None:
                sys.modules[n] = saved[n]
    return web_crawl, web_pdf


def _items(state: dict):
    """현재 문서(규정)와 관련 있는 공지만 고른다."""
    docs = state["documents"]
    for d in (docs.values() if isinstance(docs, dict) else docs):
        yield d, "evidence"
    notices = state.get("notices", {})
    for n in (notices.values() if isinstance(notices, dict) else notices):
        if isinstance(n, dict) and n.get("relevance_status") == "candidate":
            yield n, "notice"


def _slug(path: str) -> str:
    """원본 파일명에서 시각·출처를 뗀 이름 (doc_id용)."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"^\d{8}_\d{4}_[A-Za-z]+_", "", stem)


def documents():
    """정제할 일본 문서를 하나씩 (원본 글, 정제된 글, 메타데이터)로 돌려준다."""
    web_crawl, web_pdf = _load_web_tools()
    state = json.loads((JP_DIR / "japan-state.json").read_text(encoding="utf-8"))
    for item, role in _items(state):
        sid = item.get("source_id", "jnto")
        raw_rel = item.get("raw_content_path", "")
        raw_path = JP_DIR / raw_rel
        if not raw_path.exists():
            print(f"  건너뜀 (원본 파일 없음): {raw_rel}")
            continue
        if raw_path.suffix.lower() == ".pdf":
            md = web_pdf.pdf_to_markdown(str(raw_path))
        else:
            html = raw_path.read_text(encoding="utf-8")
            node = next((n for n in (web_crawl.find_content(html, s) for s in SELECTORS[sid]) if n is not None), None)
            if node is None:
                print(f"  건너뜀 (본문 위치를 못 찾음): {raw_rel}")
                continue
            md = web_crawl.html_to_markdown(node)

        # 원본 옆의 .metadata.json(수집 시각·주소·발행일)을 우선 쓴다. 공지는 state 칸 이름이 달라서
        side_path = raw_path.with_name(raw_path.stem + ".metadata.json")
        side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.exists() else {}
        item = {**item, **{k: v for k, v in {
            "document_url": side.get("document_url"), "fetched_at": side.get("collected_at"),
            "published_at": side.get("published_at"), "document_title": side.get("document_title"),
        }.items() if v}}
        title = re.sub(r"[「」]", "", item.get("document_title") or item.get("title") or "").strip()
        org = ORG.get(sid, sid)
        stype = SOURCE_TYPE.get(sid, "airline")
        meta = {
            "doc_id": f"jp.{sid}.{_slug(raw_rel)}",
            "title": title,
            "source_name": f"{org} {title}".strip(),
            "org_name": org,
            "source_type": stype,
            "priority": config.PRIORITY.get(stype, 9),
            "role": role,
            "country": "JP",
            "language": item.get("language", "en") or "en",
            "route": "international" if re.search(r"international", title, re.I) else "all",
            "section_type": "page",
            "version": str(item.get("version_id", ""))[:16],
            "effective_date": str(item.get("effective_date") or "")[:10],
            "updated_date": str(item.get("published_at") or "")[:10],
            "collected_at": str(item.get("fetched_at") or item.get("first_seen_at") or "")[:10],
            "source_url": item.get("document_url") or item.get("url") or item.get("source_url") or "",
            "raw_file": f"data/Japan/raw/{raw_rel}",
        }
        text = steps.drop_lines(md, rules.COMMON_DROP_LINES)
        rule = rules.SITE_RULES.get(f"jp.{sid}", {})
        if rule.get("drop_breadcrumb"):
            text = steps.drop_breadcrumb(text)
        if rule.get("heading_strip"):
            text = steps.strip_headings(text, rule["heading_strip"])
        if rule.get("drop_head_lines"):
            text = steps.drop_head_lines(text, rule["drop_head_lines"])
        if rule.get("drop_lines"):
            text = steps.drop_lines(text, rule["drop_lines"])
        if rule.get("drop_sections"):
            text = steps.drop_sections(text, rule["drop_sections"])
        if rule.get("replace"):
            text = steps.replace_text(text, rule["replace"])
        if rule.get("dedupe", True):
            text = steps.dedupe_blocks(text)
        text = steps.drop_empty_headings(text)
        text = steps.normalize(text, meta["source_name"])
        yield md, text, meta
