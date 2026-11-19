import json
from pathlib import Path
from collections import Counter
BASE = Path(__file__).resolve().parent
# Conservative, page/section-level screening. Retains full text without inventing or changing rules.
UK_LAW_KEEP = {'4', '12', '13', '15', '21A'}
CHINA_SOUTHERN_PAGES = set(range(32,46))

def decision(r):
    doc = r.get('doc_id','')
    if doc.startswith('uk.asa1982.'):
        section = str(r.get('section_number') or '').strip()
        if r.get('schedule_number'):
            return False, '법령 부칙: 물품 반입 제한과 직접 관련성이 낮음'
        if section in UK_LAW_KEEP:
            return True, '법령: 위험물·수하물 검사·검색 및 관련 보안규정'
        return False, '법령: 반입 물품 판정과 직접 관련성이 낮은 조항 또는 부칙'
    if doc.startswith('uk.caa.'):
        return True, 'CAA: 물품별 위탁/기내 반입 및 위험물 제한'
    if doc.startswith('uk.gov.hand.'):
        return True, 'GOV.UK: 기내 수하물 제한 항목'
    if doc.startswith('cn.cs.'):
        page = r.get('page_number')
        if page in CHINA_SOUTHERN_PAGES:
            return True, '중국남방항공: Article 8 Baggage (수하물 운송 규정)'
        return False, '중국남방항공: 수하물 규정 외 항공권·환불·운임·일반 운송약관 등'
    if doc.startswith('cn.ce.'):
        return True, '중국동방항공: 리튬배터리·위험물 안전 규정'
    if doc.startswith('cn.gov.'):
        return True, '중국 지방정부: 보조배터리 3C 인증 안내'
    return False, '알 수 없는 문서 유형: 수동 확인 필요'

def run(country):
    src = BASE / f'all_{country}_preprocessed.jsonl'
    dst = BASE / f'all_{country}_baggage_filtered.jsonl'
    excluded = BASE / f'all_{country}_excluded_review.jsonl'
    kept, dropped = [], []
    for line in src.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        ok, reason = decision(r)
        if ok: kept.append(r)
        else: dropped.append({**r, 'exclusion_reason': reason})
    for path, records in [(dst, kept), (excluded, dropped)]:
        with path.open('w',encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
    assert len(kept)+len(dropped)==sum(bool(x.strip()) for x in src.read_text(encoding='utf-8').splitlines())
    assert len({r['doc_id'] for r in kept}) == len(kept)
    print(country.upper(), '원본',len(kept)+len(dropped),'유지',len(kept),'제외',len(dropped))
    print('유지 출처:', dict(Counter(r['org_name'] for r in kept)))
    return kept,dropped
uk,uk_ex=run('uk'); cn,cn_ex=run('china')
combined=BASE/'all_uk_china_baggage_filtered.jsonl'
with combined.open('w',encoding='utf-8') as f:
    for r in uk+cn:
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('COMBINED',len(uk)+len(cn))
