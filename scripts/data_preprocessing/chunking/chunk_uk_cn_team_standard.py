"""Team-compatible, non-overlapping baggage RAG chunker. Run: python chunk_uk_cn_team_standard.py"""
from pathlib import Path
import json, re, statistics

BASE = Path(__file__).resolve().parent
MAX_LEN, TARGET, MIN_LEN = 1000, 800, 150


def safe_meta(row):
    out = {}
    for k, v in row.items():
        if k in ('content', 'text', 'metadata'): continue
        if v is None: v = ''
        if isinstance(v, (list, dict)): v = json.dumps(v, ensure_ascii=False)
        if not isinstance(v, (str, int, float, bool)): v = str(v)
        out[k] = v
    return out


def sections(body):
    """Split Markdown headings, preserve text before first heading."""
    lines = body.splitlines(keepends=True)
    path, parts, current = [], [], []
    for line in lines:
        match = re.match(r'^\s*(#{2,6})\s+(.+?)\s*$', line)
        if match:
            if current:
                parts.append((' > '.join(path), path[-1] if path else '', ''.join(current).strip()))
            level, name = len(match.group(1)), match.group(2)
            path = path[:level-2] + [name]
            current = []
        elif re.match(r'^\s*#\s+.+$', line):
            # Top-level document title is omitted as agreed.
            continue
        else:
            current.append(line)
    if current: parts.append((' > '.join(path), path[-1] if path else '', ''.join(current).strip()))
    return [(p, u, s) for p, u, s in parts if s]


def boundary(s, cap):
    """Prefer newline, then sentence end, then whitespace; never exceed cap."""
    if len(s) <= cap: return len(s)
    for pattern in (r'\n', r'(?<=[.!?。])\s+', r'\s+'):
        matches = list(re.finditer(pattern, s[:cap+1]))
        if matches:
            cut = matches[-1].end()
            if cut >= max(40, cap//3): return cut
    return cap


def table_rows(s):
    return [line for line in s.splitlines() if line.lstrip().startswith('|')]


def split_table(text, cap):
    lines = text.splitlines()
    header = lines[:2]
    prefix = '\n'.join(header) + '\n'
    out, cur = [], prefix
    for line in lines[2:]:
        if len(cur) + len(line) + 1 > cap and cur != prefix:
            out.append(cur.rstrip()); cur = prefix
        if len(prefix) + len(line) > cap:
            # pathological oversized row: retain header and split its text safely
            if cur != prefix: out.append(cur.rstrip()); cur = prefix
            rest = line
            while rest:
                size = cap-len(prefix)
                cut = boundary(rest,size)
                out.append((prefix+rest[:cut]).rstrip());rest=rest[cut:]
        else: cur += line + '\n'
    if cur.strip() and cur != prefix: out.append(cur.rstrip())
    return out or [text]


def split_long(text, cap):
    if len(text) <= cap: return [text]
    # Recognize standalone Markdown tables, repeating their two-line header.
    lines = text.splitlines(keepends=True)
    groups=[]; i=0
    while i<len(lines):
        if lines[i].lstrip().startswith('|') and i+1<len(lines) and re.match(r'^\s*\|?[\s:\-|]+\|\s*$',lines[i+1]):
            j=i+2
            while j<len(lines) and lines[j].lstrip().startswith('|'):j+=1
            groups.append(('table',''.join(lines[i:j]).strip()));i=j
        else:
            j=i+1
            while j<len(lines) and not (lines[j].lstrip().startswith('|') and j+1<len(lines) and re.match(r'^\s*\|?[\s:\-|]+\|\s*$',lines[j+1])):j+=1
            groups.append(('text',''.join(lines[i:j]).strip()));i=j
    out=[]
    for kind,segment in groups:
        if not segment:continue
        if kind=='table': out.extend(split_table(segment,cap))
        else:
            remaining=segment
            while remaining:
                cut=boundary(remaining,cap)
                out.append(remaining[:cut].strip())
                remaining=remaining[cut:].lstrip()
    return [x for x in out if x]


def prefix(source,path):
    return (source + (' > '+path if path else '')).strip()


def chunks_for(row):
    meta=safe_meta(row)
    source=meta.get('source_name') or meta.get('title') or meta['doc_id']
    pieces=[]
    for path,unit,body in sections(row.get('content','')):
        head=prefix(source,path)
        cap=MAX_LEN-len(head)-1
        if cap<100: raise ValueError('Too-long source/title context')
        for piece in split_long(body,cap): pieces.append({'path':path,'unit':unit,'body':piece})
    # Only merge short sibling units under the same parent; never cross unrelated parents.
    merged=[]
    for item in pieces:
        if merged and len(item['body'])<MIN_LEN:
            prev=merged[-1]
            parent=lambda p: ' > '.join(p.split(' > ')[:-1])
            if parent(prev['path'])==parent(item['path']) and prev['path'] and item['path'] and prev['path']!=item['path']:
                body=prev['body']+'\n['+item['unit']+']\n'+item['body']
                if len(prefix(source,prev['path']))+1+len(body)<=min(TARGET,MAX_LEN):
                    prev['body']=body;continue
        merged.append(item)
    result=[]
    for idx,item in enumerate(merged):
        text=prefix(source,item['path'])+'\n'+item['body']
        assert len(text)<=MAX_LEN,(meta['doc_id'],len(text))
        m=dict(meta,section_path=item['path'],unit=item['path'],chunk_index=idx,chunk_count=len(merged))
        result.append({'id':f"{meta['doc_id']}#{idx}",'text':text,'metadata':m})
    return result


def run(country, filename):
    path=BASE/filename
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    output=[]
    for row in rows:output.extend(chunks_for(row))
    outpath=BASE/'data'/'chunks'/f'{country}.jsonl';outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output),encoding='utf-8')
    lengths=[len(x['text']) for x in output]
    ids=[x['id'] for x in output]
    assert len(ids)==len(set(ids))
    assert all(set(x)=={'id','text','metadata'} for x in output)
    assert all(all(not isinstance(v,(list,dict)) and v is not None for v in x['metadata'].values()) for x in output)
    print(f'{country}: {len(rows)} documents -> {len(output)} chunks; max={max(lengths)}, median={statistics.median(lengths)}; {outpath}')
    return output

if __name__=='__main__':
    uk=run('uk','all_uk_baggage_filtered.jsonl')
    cn=run('cn','all_china_baggage_filtered.jsonl')
    for c in (uk,cn):
        print('EXAMPLE:',json.dumps(c[0],ensure_ascii=False)[:450])
