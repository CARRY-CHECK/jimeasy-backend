"""
# chunk.py
정제 결과(data/{나라}/preprocessed/*.jsonl)를 제목 단위로 잘라 조각(data/{나라}/chunk/*.jsonl)으로 저장한다.
사용: python chunk.py kr law us jp
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "collect"))  # collection_paths
from collection_paths import ROOT, country_file  # noqa: E402

TARGET = 800  # 짧은 항목을 합칠 때 목표 길이(자)
MAX = 1000  # 한 조각 본문 최대 길이(자). 약 512토큰 임베딩 모델 기준
MIN = 150  # 이보다 짧은 항목은 같은 부모 아래 형제와 합친다

_HEADING = re.compile(r"^(#{2,6})\s+(.*)$")


def split_sections(text: str) -> list[tuple[tuple, str]]:
    """'# 문서 이름' 아래를 제목 단위로 나눈다. (제목 경로, 본문) 목록. 제목 전 머리 글은 경로가 ()."""
    out, stack, buf = [], [], []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append((tuple(t for _, t in stack), body))
        buf.clear()

    for line in text.split("\n"):
        if line.startswith("# "):
            continue  # 문서 이름 (메타데이터 source_name과 같음)
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return out


def split_long(body: str) -> list[str]:
    """MAX보다 긴 본문을 나눈다. 글은 줄 경계에서, 표는 행 단위로 자르고 조각마다 머리글 줄을 다시 붙인다."""
    lines, pieces, cur = body.split("\n"), [], []
    header: list[str] = []  # 지금 읽고 있는 표의 머리글 (머리 줄 + 구분 줄)

    def size(xs: list[str]) -> int:
        return sum(len(x) + 1 for x in xs)

    def flush() -> None:
        if any(x.strip() for x in cur):
            pieces.append("\n".join(cur).strip())
        cur.clear()

    for i, line in enumerate(lines):
        is_row = line.strip().startswith("|")
        if is_row and not header and i + 1 < len(lines) and re.match(r"^\|(\s*-+\s*\|)+\s*$", lines[i + 1].strip()):
            header = [line, lines[i + 1]]  # 새 표 시작
        if not is_row:
            header = []
        if size(cur) + len(line) + 1 > MAX and cur:
            flush()
            if is_row and header and line not in header:
                cur.extend(header)  # 표가 이어지면 머리글을 다시 붙인다
        cur.append(line)
    flush()
    # 줄 하나가 MAX보다 긴 경우(아주 긴 문단)는 문장 단위로 한 번 더 나눈다
    final = []
    for p in pieces:
        while len(p) > MAX:
            cut = max(p.rfind(". ", 0, MAX), p.rfind("다.", 0, MAX), p.rfind(" ", 0, MAX))
            cut = cut + 1 if cut > MAX // 2 else MAX
            final.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            final.append(p)
    # 표 줄을 자르다 남은 빈 칸 조각(예: "|  |  |")처럼 글자가 없는 조각은 버린다
    return [p for p in final if re.sub(r"[|\-\s]", "", p)]


def make_units(sections: list[tuple[tuple, str]]) -> list[tuple[tuple, str]]:
    """긴 항목은 나누고, 짧은 항목은 같은 부모 아래 형제끼리 TARGET까지 합친다."""
    units = []
    for path, body in sections:
        if len(body) > MAX:
            units += [(path, p) for p in split_long(body)]
        else:
            units.append((path, body))

    merged, acc_path, acc_parts = [], None, []

    def flush() -> None:
        nonlocal acc_path
        if acc_parts:
            if len(acc_parts) == 1:
                merged.append((acc_parts[0][0], acc_parts[0][1]))
            else:  # 여러 항목을 합친 조각: 각 항목 제목을 본문 안에 남긴다
                parent = acc_parts[0][0][:-1]
                text = "\n\n".join(f"[{p[-1]}]\n{b}" if p else b for p, b in acc_parts)
                merged.append((parent, text))
        acc_parts.clear()
        acc_path = None

    for path, body in units:
        parent = path[:-1]
        acc_len = sum(len(b) + len(p[-1] if p else "") + 4 for p, b in acc_parts)
        small = len(body) < MIN
        if acc_parts and small and parent == acc_path and acc_len + len(body) <= TARGET:
            acc_parts.append((path, body))
            continue
        flush()
        if small and path:
            acc_path, acc_parts[:] = parent, [(path, body)]
        else:
            merged.append((path, body))
    flush()
    return merged


def chunk_document(doc: dict) -> list[dict]:
    meta = doc["metadata"]
    units = make_units(split_sections(doc["text"]))
    rows = []
    for i, (path, body) in enumerate(units):
        section_path = " > ".join(path)
        prefix = " > ".join([meta["source_name"]] + list(path))
        rows.append({
            "id": f"{doc['doc_id']}#{i}",
            "text": f"{prefix}\n{body}",
            "metadata": dict(meta, section_path=section_path, unit=section_path, chunk_index=i, chunk_count=len(units)),
        })
    return rows


def run(name: str) -> None:
    src = country_file(name, "preprocessed")
    docs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows = [r for d in docs for r in chunk_document(d)]
    out = country_file(name, "chunk")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(out)

    lens = sorted(len(r["text"]) for r in rows)
    over = sum(len(r["text"].split("\n", 1)[-1]) > MAX for r in rows)
    print(f"[{name}] 문서 {len(docs)}개 → 조각 {len(rows)}개 → {out.relative_to(ROOT).as_posix()}")
    print(f"   조각 길이(맥락 포함): 최소 {lens[0]}, 중앙값 {lens[len(lens) // 2]}, 최대 {lens[-1]} | 본문 {MAX}자 초과 {over}개")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for name in sys.argv[1:] or ["kr", "law"]:
        run(name)


if __name__ == "__main__":
    main()
