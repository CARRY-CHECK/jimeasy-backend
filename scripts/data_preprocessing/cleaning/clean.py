"""
# clean.py
정제 실행: 출처별로 읽어 공통 단계로 정제하고 data/{나라}/preprocessed/{kr,law,us,jp}.jsonl에 저장한다.
사용: python clean.py --country kr   (kr / us / jp / law)
"""

import argparse
import json
import re
import sys

import config
from adapters import japan, law, web


def _write_atomic(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _mark_cleaned(doc_ids: set[str]) -> None:
    """정제한 웹 문서를 manifest에서 '정제완료'로 바꾼다."""
    rows = [json.loads(l) for l in open(config.WEB_MANIFEST, encoding="utf-8") if l.strip()]
    for r in rows:
        if r["doc_id"] in doc_ids:
            r["status"] = "정제완료"
    _write_atomic(config.WEB_MANIFEST, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _real_len(text: str) -> int:
    """공백·박스 문자·표 기호·제목 기호를 뺀 실제 글자 수 (칸 맞춤 공백 때문에 비율이 잘못 나오지 않게)."""
    return len(re.sub(r"[\s┌┐└┘├┤┬┴┼─━│┃═║╔╗╚╝╠╣╦╩╬|#\-]", "", text))


def run(country: str) -> None:
    out_rows, done, report = [], set(), []
    if country in ("law", "jp"):
        for before, text, meta in (law.documents() if country == "law" else japan.documents()):
            out_rows.append({"doc_id": meta["doc_id"], "metadata": meta, "text": text})
            report.append((meta["doc_id"], _real_len(before), _real_len(text)))
    else:
        for r, before, text, meta in web.documents(country):
            out_rows.append({"doc_id": meta["doc_id"], "metadata": meta, "text": text})
            done.add(r["doc_id"])
            report.append((meta["doc_id"], _real_len(before), _real_len(text)))

    out_path = config.country_file(country, "preprocessed")
    _write_atomic(out_path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows))
    if done:
        _mark_cleaned(done)

    # 검증: 글자 수 변화, 핵심 단어
    print(f"정제 완료: {len(out_rows)}건 → {out_path.relative_to(config.BASE_DIR).as_posix()}\n")
    print(f"{'문서':<46} {'전':>7} {'후':>7} {'비율':>6}  (공백 뺀 글자 수)")
    for doc_id, a, b in report:
        warn = "  ⚠ 많이 줄어듦" if a and b / a < config.MIN_KEEP_RATIO else ""
        print(f"{doc_id:<46} {a:>7} {b:>7} {b / a if a else 0:>6.0%}{warn}")
    all_text = "\n".join(x["text"] for x in out_rows)
    terms = config.CHECK_TERMS.get(country, [])
    if terms:
        print("\n핵심 단어:", ", ".join(f"{t} {all_text.count(t)}" for t in terms))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="수집 자료 정제")
    parser.add_argument("--country", choices=["kr", "us", "jp", "law"], required=True)
    run(parser.parse_args().country)


if __name__ == "__main__":
    main()
