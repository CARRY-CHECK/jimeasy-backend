"""
# steps.py
정제 공통 단계: 줄 지우기, 구간 지우기, 중복 정리, 형식 통일. Markdown 글을 받아 Markdown 글을 돌려준다.
"""

import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _is_heading(line: str) -> bool:
    return bool(_HEADING.match(line))


def drop_lines(md: str, patterns: list[str]) -> str:
    """패턴에 맞는 줄을 지운다. 제목 줄은 지우지 않는다."""
    regs = [re.compile(p) for p in patterns]
    return "\n".join(
        line for line in md.split("\n")
        if _is_heading(line) or not any(r.search(line.strip()) for r in regs)
    )


def drop_head_lines(md: str, exact: list[str]) -> str:
    """첫 제목이 나오기 전 부분에서, 글자가 정확히 같은 줄을 지운다 (탭 이름 등)."""
    out, before_heading = [], True
    for line in md.split("\n"):
        if _is_heading(line):
            before_heading = False
        if before_heading and line.strip() in exact:
            continue
        out.append(line)
    return "\n".join(out)


def drop_sections(md: str, titles: list[str]) -> str:
    """지정한 제목부터 같은 단계 이상의 다음 제목 전까지 통째로 지운다."""
    out, skip_level = [], None
    for line in md.split("\n"):
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and m.group(2).strip() in titles:
                skip_level = level
                continue
        if skip_level is None:
            out.append(line)
    return "\n".join(out)


def drop_blocks(md: str, ranges: list[list[str]]) -> str:
    """[시작 정규식, 끝 정규식]: 시작 줄부터 끝 줄 직전까지 지운다. 끝 줄은 남기고 다시 검사한다."""
    for start, end in ranges:
        rs, re_end = re.compile(start), re.compile(end)
        out, skipping = [], False
        for line in md.split("\n"):
            if skipping and re_end.search(line.strip()) and not rs.search(line.strip()):
                skipping = False
            if not skipping and rs.search(line.strip()):
                skipping = True
            if not skipping:
                out.append(line)
        md = "\n".join(out)
    return md


def line_to_heading(md: str, rules: list[list]) -> str:
    """[줄 정규식, 제목 글자, 제목 단계]: 평문 구분 줄(예: '대만 의 조회 결과입니다.')을 제목으로 바꾼다."""
    for pattern, title, level in rules:
        reg = re.compile(pattern)
        md = "\n".join(
            "#" * level + " " + reg.sub(title, line.strip()) if not _is_heading(line) and reg.fullmatch(line.strip()) else line
            for line in md.split("\n")
        )
    return md


def drop_breadcrumb(md: str) -> str:
    """문서 맨 앞에 이어진 번호 메뉴 경로(예: '1. ANA Website', '3. Guide')를 지운다."""
    lines = md.split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or re.match(r"^\d+\.\s", lines[i].strip())):
        i += 1
    return "\n".join(lines[i:])


def strip_headings(md: str, patterns: list[str]) -> str:
    """제목 글자에서 패턴을 지운다 (예: 접힌 메뉴 버튼의 'Open')."""
    regs = [re.compile(p) for p in patterns]
    out = []
    for line in md.split("\n"):
        m = _HEADING.match(line)
        if m:
            title = m.group(2)
            for r in regs:
                title = r.sub("", title)
            line = f"{m.group(1)} {title.strip()}"
        out.append(line)
    return "\n".join(out)


def replace_text(md: str, rules: list[list[str]]) -> str:
    """[정규식, 바꿀 글자]: 제목이 아닌 줄 안에서 일부를 바꾼다 (링크 꼬리말 지우기 등)."""
    for pattern, repl in rules:
        reg = re.compile(pattern)
        md = "\n".join(line if _is_heading(line) else reg.sub(repl, line) for line in md.split("\n"))
    return md


def table_to_sections(md: str) -> str:
    """물품별 표를 제목으로 바꾼다: 표 한 줄 → '## 첫 칸' + '- 머리글: 칸 값' 목록. 물품마다 따로 검색되게 한다."""
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        is_table = (lines[i].strip().startswith("|") and i + 1 < len(lines)
                    and re.match(r"^\|(\s*-+\s*\|)+\s*$", lines[i + 1].strip()))
        if not is_table:
            out.append(lines[i]); i += 1
            continue
        header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        i += 2
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if cells and cells[0]:
                out.append(f"\n## {cells[0]}")
                out += [f"- {h}: {v}" for h, v in zip(header[1:], cells[1:]) if v]
            i += 1
    return "\n".join(out)


def dedupe_blocks(md: str, min_len: int = 30) -> str:
    """한 문서 안에서 똑같이 반복되는 문단(빈 줄로 나뉜 덩어리)과 긴 줄은 첫 번째만 남긴다.
    제목·짧은 줄·표 줄은 그대로 둔다. 표 줄은 같아도 어느 표(나라·항목)에 있느냐에 따라 뜻이 달라서다."""
    seen_blocks, out = set(), []
    for block in re.split(r"\n\s*\n", md):
        key = re.sub(r"\s+", " ", block).strip()
        if len(key) >= min_len and not _is_heading(block.strip()):
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
        out.append(block)
    seen_lines, lines = set(), []
    for line in "\n\n".join(out).split("\n"):
        key = re.sub(r"^[-*]\s+", "", re.sub(r"\s+", " ", line).strip())  # 목록 기호는 빼고 비교
        if len(key) >= min_len and not _is_heading(line) and not line.strip().startswith("|"):
            if key in seen_lines:
                continue
            seen_lines.add(key)
        lines.append(line)
    return "\n".join(lines)


def drop_empty_headings(md: str) -> str:
    """내용 없이 바로 같은 단계 이상의 제목이 이어지는 제목은 지운다 (탭 목록을 지운 뒤 남은 빈 제목 등)."""
    lines = md.split("\n")
    out = []
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            nxt = next((l for l in lines[i + 1:] if l.strip()), None)
            nm = _HEADING.match(nxt) if nxt else None
            if nxt is None or (nm and len(nm.group(1)) <= level):
                continue
        out.append(line)
    return "\n".join(out)


def normalize(md: str, title: str) -> str:
    """형식 통일: 맨 위 '# 문서 이름' 하나, 본문 제목은 ##부터, 단계 건너뛰기 없음, 빈 줄은 최대 1줄."""
    lines = [l for l in md.split("\n") if not re.match(r"^#\s", l)]  # 기존 # 제목은 문서 이름으로 대체
    lines = [l for l in lines if not re.fullmatch(r"\s*\|(\s*\|)+\s*", l)]  # 칸이 모두 빈 표 줄
    # 문서에 있는 제목 단계를 작은 것부터 ##, ###, #### … 로 차례로 매긴다 (예: ###·####·##### → ##·###·####)
    levels = sorted({len(m.group(1)) for m in map(_HEADING.match, lines) if m})
    rank = {lv: min(i + 2, 6) for i, lv in enumerate(levels)}
    out = []
    for line in lines:
        m = _HEADING.match(line)
        if m:
            line = "#" * rank[len(m.group(1))] + " " + m.group(2).strip()
        out.append(line.rstrip())
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return f"# {title}\n\n{body}\n"
