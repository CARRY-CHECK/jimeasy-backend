"""
# adapters/web.py
웹 수집 결과(manifest + raw/*.md)를 읽어 정제할 문서 목록을 만든다. 국내·미국 페이지와 정책브리핑 보도자료.
"""

import csv
import json
import re
from datetime import datetime

from bs4 import BeautifulSoup

import config
import rules
import steps


def _load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _sources() -> dict[str, dict]:
    rows = {}
    for path in config.WEB_SOURCE_FILES:
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            rows[r["source_id"]] = r
    return rows


def _iso(value: str) -> str:
    """날짜를 YYYY-MM-DD로 맞춘다 (시각이 붙어 있으면 뗀다)."""
    return (value or "")[:10]


def latest_records(country: str) -> list[dict]:
    """manifest에서 나라에 맞는 문서를, 출처마다 가장 최근 버전 하나씩 고른다."""
    sources = _sources()
    latest = {}
    for r in _load_jsonl(config.WEB_MANIFEST):
        if r.get("role") == "notice":  # 보도자료
            key, c = r["doc_id"], "kr"
        else:
            key = r["doc_id"].rsplit("_", 1)[0]
            c = sources.get(key, {}).get("country", "").lower()
        if c == country and (key not in latest or r["doc_id"] > latest[key]["doc_id"]):
            latest[key] = r
    return list(latest.values())


def _metadata(r: dict, sources: dict, signals: dict) -> tuple[str, dict]:
    """manifest 한 줄 → (출처 키, 통일 메타데이터)."""
    if r.get("role") == "notice":
        sig = signals.get(r["doc_id"], {})
        key, src = r["doc_id"], {}
        source_type, org, lang, route = "press", r.get("source_org", ""), "ko", "all"
        effective = sig.get("effective_date", "")
    else:
        key = r["doc_id"].rsplit("_", 1)[0]
        src = sources.get(key, {})
        source_type = config.SOURCE_TYPE.get(src.get("org_type", r.get("source_type", "")), "")
        org, lang = src.get("org_name", r.get("source_org", "")), src.get("lang", "ko")
        route = config.ROUTE.get(src.get("route", ""), "all")
        effective = ""
    title = r.get("title", "")
    meta = {
        "doc_id": f"{'kr' if source_type == 'press' else src.get('country', '').lower()}.{key}",
        "title": title,
        "source_name": f"{org} {title}".strip(),
        "org_name": org,
        "source_type": source_type,
        "priority": (config.PRIORITY_DOMESTIC_GOVERNMENT if source_type == "government" and src.get("country") == "KR"
                     else config.PRIORITY.get(source_type, 9)),
        "role": r.get("role", "evidence"),
        "country": "KR" if source_type == "press" else src.get("country", ""),
        "language": lang,
        "route": route,
        "section_type": "page",
        "version": r["doc_id"],
        "effective_date": _iso(effective),
        "updated_date": _iso(r.get("page_updated", "")),
        "collected_at": _iso(r.get("collected_at", "")),
        "source_url": r.get("url", ""),
        "raw_file": r.get("raw_path", ""),
    }
    return key, meta


def _strip_press_header(md: str) -> str:
    """보도자료 .md의 머리말(# 제목, 발표일·부처·원문 줄)을 뗀다. 메타데이터에 이미 있다."""
    return re.sub(r"\A# .*\n+(- (발표일|부처|원문):.*\n)+", "", md)


def _tsa_items(html_path) -> str:
    """TSA 'What Can I Bring?' 표를 원본 HTML에서 물품별로 바꾼다: 물품 이름, 기내·위탁 결과, 설명."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    parts = []
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        name = tds[0].find("strong") or tds[0].find("a")
        desc = " ".join(p.get_text(" ", strip=True) for p in tds[0].find_all("p"))
        if not name:
            continue
        parts.append(f"## {name.get_text(' ', strip=True)}\n"
                     f"- Carry-on bags: {tds[1].get_text(' ', strip=True)}\n"
                     f"- Checked bags: {tds[2].get_text(' ', strip=True)}\n\n{desc}".strip())
    return "\n\n".join(parts)


def _find_updated(md: str, rule: list[str]) -> str:
    """문서에 적힌 수정일을 찾아 YYYY-MM-DD로 돌려준다. 못 찾으면 빈칸."""
    m = re.search(rule[0], md)
    if not m:
        return ""
    try:
        return datetime.strptime(m.group(1), rule[1]).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def documents(country: str):
    """정제할 문서를 하나씩 (manifest 줄, 원본 글, 정제된 글, 메타데이터)로 돌려준다."""
    sources = _sources()
    signals = {s["signal_id"]: s for s in _load_jsonl(config.WEB_SIGNALS)}
    for r in latest_records(country):
        md = config.resolve_record_path(r["md_path"]).read_text(encoding="utf-8")
        key, meta = _metadata(r, sources, signals)
        rule = rules.SITE_RULES.get(key, {})
        if rule.get("converter") == "tsa_items":
            md = _tsa_items(config.resolve_record_path(r["raw_path"]))
        if rule.get("updated_date"):
            meta["updated_date"] = _find_updated(md, rule["updated_date"]) or meta["updated_date"]
        text = md
        if meta["source_type"] == "press":
            text = steps.drop_lines(_strip_press_header(text), rules.PRESS_DROP_LINES)
        text = steps.drop_lines(text, rules.COMMON_DROP_LINES)
        if rule.get("drop_head_lines"):
            text = steps.drop_head_lines(text, rule["drop_head_lines"])
        if rule.get("drop_lines"):
            text = steps.drop_lines(text, rule["drop_lines"])
        if rule.get("drop_sections"):
            text = steps.drop_sections(text, rule["drop_sections"])
        if rule.get("drop_blocks"):
            text = steps.drop_blocks(text, rule["drop_blocks"])
        if rule.get("line_to_heading"):
            text = steps.line_to_heading(text, rule["line_to_heading"])
        if rule.get("replace"):
            text = steps.replace_text(text, rule["replace"])
        if rule.get("table_to_sections"):
            text = steps.table_to_sections(text)
        if rule.get("dedupe", True):
            text = steps.dedupe_blocks(text)
        text = steps.drop_empty_headings(text)
        text = steps.normalize(text, meta["source_name"])
        yield r, md, text, meta
