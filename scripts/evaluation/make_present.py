# -*- coding: utf-8 -*-
"""평가 폴더의 report.json에서 장표용 present.md를 만든다.

python -m scripts.evaluation.make_present --eval-dir tests/ver2_test/evaluate/attempt_1 --run tests/ver2_test/attempt_1 \\
    --label ver2 --purpose "..." [--baseline-eval 이전/평가/폴더 --baseline-label ver2] [--note "..."]

LLM 호출·재채점·기존 평가 파일 수정은 없다. 저장된 report.json과 실행 summary.json만 읽어
<eval-dir>/query_structure/present.md, <eval-dir>/final_answer/present.md를 쓴다.
같은 이름의 파일이 있으면 --overwrite 없이는 덮어쓰지 않는다.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

FIELD_NAMES = {
    'departure.place': '출발지 (departure.place)', 'departure.airline': '출발 항공사 (departure.airline)',
    'arrival.place': '도착지 (arrival.place)', 'arrival.airline': '귀국 항공사 (arrival.airline)',
    'travel_date': '여행 날짜 (travel_date)', 'baggage_type': '수하물 유형 (baggage_type)',
}
LABELS = ['가능', '불가', '조건부 가능', '확인 필요']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pct(value):
    return f'{value * 100:.2f}%'


def short(sha):
    return sha[:8] if sha else '-'


def base_name(path):
    """다른 OS에서 저장한 경로(역슬래시)도 파일명만 남긴다."""
    return str(path or '').replace('\\', '/').rsplit('/', 1)[-1]


def stamp(value):
    return (value or '')[:19].replace('T', ' ')


def delta(kind, current, base):
    if current is None or base is None:
        return '-'
    if kind == 'pct':
        return f'{(current - base) * 100:+.2f}%p'
    return f'{current - base:+.4f}'


def render_value(kind, value):
    return pct(value) if kind == 'pct' else f'{value:.4f}'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    out += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join(out)


def metric_table(label, rows, base_rows, base_label):
    """rows: [(지표, 종류, 값, 해석)]. base_rows가 있으면 이전 대비 열을 추가한다."""
    base = {r[0]: r[2] for r in base_rows or []}
    header = ['평가 지표', f'{label} 결과'] + ([f'{base_label} 대비'] if base_rows else []) + ['해석']
    body = []
    for name, kind, value, note in rows:
        row = [name, render_value(kind, value)]
        if base_rows:
            row.append(delta(kind, value, base.get(name)))
        body.append(row + [note])
    return table(header, body)


def settings_section(run, purpose, notes, extra_rows):
    summary = load(Path(run) / 'summary.json')
    models = summary.get('models', {})
    prompt = summary.get('parse_system_prompt', 'parse_system.txt')
    hashes = summary.get('prompt_sha256', {})
    rows = [
        ('실행 폴더', f"`{Path(run).as_posix()}` ({summary.get('total')}문항 중 성공 {summary.get('success')}, 오류 {summary.get('error')})"),
        ('질문·정답 데이터셋', f"`{base_name(summary.get('dataset'))}` (sha256 {short(summary.get('dataset_sha256'))})"),
        ('구조화·답변 모델', f"{models.get('query', {}).get('model')} / {models.get('answer', {}).get('model')} "
                       f"(temperature {models.get('query', {}).get('parameters', {}).get('temperature')})"),
        ('구조화 프롬프트', f"`{prompt}` (sha256 {short(hashes.get(prompt))})"),
        ('임베딩·검색', f"{models.get('embedding', {}).get('model')} ({models.get('embedding', {}).get('parameters', {}).get('device')}), "
                    f"Chroma `{summary.get('collection')}`, 물품당 k={summary.get('k')}, 명시 필터 {summary.get('filters') or '없음'}, 국가 필터는 장소에서 자동 구성"),
        ('실행 일시', f"{stamp(summary.get('started_at'))} ~ {stamp(summary.get('finished_at'))}"),
    ] + extra_rows
    lines = ['## 실험 개요', '', f'- **목적:** {purpose}']
    lines += [f'- **주의:** {n}' for n in notes]
    lines += ['', table(['항목', '내용'], rows), '']
    return '\n'.join(lines)


# ---------------------------------------------------------------- 질의 구조화
def query_rows(report):
    m, items, questions = report['metrics'], report['items'], report['questions']
    n = m['annotated_questions']
    expected_rows = [i for i in items if i['reason'] != 'extra_item']
    nonnull = [i for i in expected_rows if i.get('expected_details') is not None]
    nonnull_ok = sum(1 for i in nonnull if i['details_correct'])
    details_ok = sum(1 for i in expected_rows if i['details_correct'])
    exact = lambda key: sum(1 for q in questions if q[key])
    rows = [
        ('물품 Precision', 'pct', m['item_precision'], f"추출한 물품 {m['predicted_items']}개 중 {m['matched_items']}개가 정답"),
        ('물품 Recall', 'pct', m['item_recall'], f"정답 물품 {m['expected_items']}개 중 {m['matched_items']}개 추출"),
        ('물품 목록 전체 일치율', 'pct', m['items_exact_match'], f"{n}문항 중 {exact('items_exact')}문항 일치"),
        ('Nonnull Details 정확도', 'pct', m['nonnull_details_accuracy'], f"값이 존재하는 세부 속성 {len(nonnull)}개 중 {nonnull_ok}개 일치"),
        ('Details 전체 정확도', 'pct', m['details_accuracy'], f"null을 포함한 세부 속성 {len(expected_rows)}개 중 {details_ok}개 일치"),
        ('여행 조건 전체 일치율', 'pct', m['conditions_exact_match'], f"{n}문항 중 {exact('conditions_exact')}문항 일치"),
        ('질의 구조화 전체 일치율', 'pct', m['query_exact_match'], f"{n}문항 중 {exact('all_exact')}문항 완전 일치"),
    ]
    return rows


def query_present(eval_dir, run, args):
    report = load(eval_dir / 'query_structure/report.json')
    base_report = load(Path(args.baseline_eval) / 'query_structure/report.json') if args.baseline_eval else None
    m, items, fields, questions = report['metrics'], report['items'], report['fields'], report['questions']
    n = m['annotated_questions']
    rows = query_rows(report)
    base_rows = query_rows(base_report) if base_report else None

    field_rows = []
    for key, name in FIELD_NAMES.items():
        pf = report['per_field'][key]
        field_rows.append((name, pct(pf['accuracy']), f"{round(pf['accuracy'] * pf['count'])}/{pf['count']}"))
    ok_fields = sum(1 for f in fields if f['correct'])
    field_rows.append(('전체 필드 정확도', pct(m['condition_field_accuracy']), f'{ok_fields}/{len(fields)}'))

    # 오류 원인 (자동 분류 규칙은 지표 설명에 적었다)
    dep_errors = [f for f in fields if f['field'] == 'departure.place' and not f['correct']]
    policy = [f for f in dep_errors if f['expected'] == '한국' and not f.get('predicted')]
    other_errors = Counter(f['field'] for f in fields if not f['correct'] and f['field'] != 'departure.place')
    missing = [i for i in items if i['reason'] == 'missing_item']
    extras = [i for i in items if i['reason'] == 'extra_item']

    def is_split(item):
        return any(e['question_id'] == item['question_id'] and (item['name'] in e['name'] or e['name'] in item['name']) for e in extras)

    split = [i for i in missing if is_split(i)]
    nonnull_wrong = [i for i in items if i['reason'] != 'extra_item' and i.get('expected_details') is not None and not i['details_correct']]
    nn_unlinked_split = [i for i in nonnull_wrong if not i['matched'] and is_split(i)]
    nn_unlinked_real = [i for i in nonnull_wrong if not i['matched'] and not is_split(i)]
    nn_string = [i for i in nonnull_wrong if i['matched']]
    failed = n - sum(1 for q in questions if q['all_exact'])
    cause_rows = [
        ('출발지 오류', f'{len(dep_errors)}건', f"이 중 {len(policy)}건은 정답 한국·출력 null. 실행 프롬프트의 기본값 정책과 대조 필요" if policy else '정답 한국·출력 null 사례 없음'),
        ('물품 미연결', f'{len(missing)}건', f"이름 포함 관계 있음 {len(split)}건, 포함 관계 없음 {len(missing) - len(split)}건. 실제 누락·분리 여부는 원문 검수 필요"),
        ('Nonnull Details 오류', f'{len(nonnull_wrong)}건',
         f"이름 미연결 중 포함 관계 있음 {len(nn_unlinked_split)}건·없음 {len(nn_unlinked_real)}건, 연결된 물품의 문자열 불일치 {len(nn_string)}건. 의미 오류 여부는 별도 검수 필요"),
        ('전체 구조화 불일치', f'{failed}문항',
         f"여행 조건 불일치 {n - sum(1 for q in questions if q['conditions_exact'])}문항, 물품 목록 불일치 {n - sum(1 for q in questions if q['items_exact'])}문항, "
         f"세부 속성 불일치 {n - sum(1 for q in questions if q['details_exact'])}문항이 복합적으로 발생"),
    ]
    if other_errors:
        cause_rows.insert(1, ('기타 여행 조건 오류', f'{sum(other_errors.values())}건',
                              ', '.join(f"{FIELD_NAMES[k].split(' ')[0]} {v}건" for k, v in other_errors.items())))

    notes = list(args.note or [])
    if policy:
        notes.append(f"출발지 오류 중 {len(policy)}건은 정답 한국·출력 null이다. 실행 프롬프트도 한국 기본값을 요구한다면 추출 불일치이며, 정책 차이 여부는 프롬프트 스냅샷으로 확인해야 한다.")
    key_lines = [
        f"- **물품 Recall {pct(m['item_recall'])}** ({m['matched_items']}/{m['expected_items']}): 정답 물품을 얼마나 빠뜨리지 않고 추출했는가",
        f"- **Nonnull Details 정확도 {pct(m['nonnull_details_accuracy'])}**: 용량·수량 등 규정을 가르는 속성을 얼마나 정확히 담았는가",
        f"- **질의 구조화 전체 일치율 {pct(m['query_exact_match'])}** ({sum(1 for q in questions if q['all_exact'])}/{n}문항): 조건·물품·속성을 모두 맞힌 질문 비율",
        f"- **출발지 정확도 {pct(report['per_field']['departure.place']['accuracy'])}**: 기본값 정책 영향을 분리해서 해석해야 하는 필드",
    ]
    if base_report:
        key_lines.append(f"- {args.baseline_label} 대비 변화는 1번 표의 '{args.baseline_label} 대비' 열 참고")

    md = [f'# 질의 구조화 평가 — {args.label}', '',
          settings_section(run, args.purpose, notes, [('정답 구조화 파일', f"`{base_name(report['reference']['path'])}` (sha256 {short(report['reference']['sha256'])})"),
                                                       ('평가 방식', 'NFC·대소문자·공백만 정규화한 정확 일치. 의미 동등성·기본값은 추론하지 않으며 평가용 LLM 호출 없음')]),
          '## 1. 종합 평가', '', metric_table(args.label, rows, base_rows, args.baseline_label), '',
          '## 2. 여행 조건 필드별 Accuracy', '', table(['필드', 'Accuracy', '정답 수 / 전체'], field_rows), '',
          '## 3. 오류 원인 분석', '', table(['오류 유형', '건수', '주요 원인'], cause_rows), '',
          '## 핵심 지표 (장표용)', '', *key_lines, '',
          '## 지표 설명', '',
          '- **물품 Precision / Recall:** 같은 질문 안에서 물품 이름을 일대일로 연결해 센다. Precision은 추출한 물품 중 정답 비율, Recall은 정답 물품 중 추출한 비율이다.',
          '- **물품 목록 전체 일치율:** 정답 물품이 하나도 빠지거나 더해지지 않은 질문의 비율.',
          '- **Nonnull Details 정확도:** 정답에 세부 속성(details) 값이 있는 물품만의 일치율. 이름이 연결되지 않으면 오답이다.',
          '- **Details 전체 정확도:** 정답 물품 전체(null 포함)의 세부 속성 일치율. null 정답이 많아 Nonnull보다 높게 나온다.',
          '- **여행 조건 전체 일치율:** 6개 조건 필드가 모두 맞은 질문의 비율. **질의 구조화 전체 일치율:** 조건·물품·속성이 모두 맞은 질문의 비율.',
          '- **진단 규칙:** 정답 출발지가 한국이고 출력이 null이면 기본값 정책과 대조할 후보로 표시한다. 이름 포함 관계는 미연결 정답 이름과 같은 질문의 추가 출력 이름 사이의 부분문자열 관계일 뿐이며, 실제 누락·물품 분리·의미 오류를 확정하지 않는다.', '']
    return '\n'.join(md)


# ---------------------------------------------------------------- 최종 답변
def final_rows(report):
    cl, ret, cov = report['classification'], report['retrieval'], report['coverage']
    per = cl['overall']['per_class']
    n_q = cov['questions']
    ok = lambda block: round(block['accuracy'] * block['count'])
    unknown = per['확인 필요']
    rows = [
        ('최종 판정 Accuracy', 'pct', cl['overall']['accuracy'], f"기내·위탁 판정 {cl['overall']['count']}개 중 {ok(cl['overall'])}개 일치"),
        ('최종 판정 Macro-F1', 'f4', cl['overall']['macro_f1'], '가능·불가·조건부 가능·확인 필요 4개 라벨 F1의 평균'),
        ('기내 판정 Accuracy', 'pct', cl['carry_on']['accuracy'], f"기내 판정 {cl['carry_on']['count']}개 중 {ok(cl['carry_on'])}개 일치"),
        ('위탁 판정 Accuracy', 'pct', cl['checked_baggage']['accuracy'], f"위탁 판정 {cl['checked_baggage']['count']}개 중 {ok(cl['checked_baggage'])}개 일치"),
        ('질문 전체 판정 일치율', 'pct', report['question_all_decisions_accuracy'],
         f"{n_q}문항 중 {round(report['question_all_decisions_accuracy'] * n_q)}문항에서 모든 물품의 기내·위탁 판정이 일치"),
        ('확인 필요 Recall', 'pct', unknown['recall'], f"정답 확인 필요 {unknown['support']}판정 중 {round(unknown['recall'] * unknown['support'])}판정 일치"),
    ]
    metrics = ret['metrics']
    for k in report['settings']['k']:
        hit = metrics[f'hit_rate@{k}']
        rows += [
            (f'근거 Recall@{k}', 'pct', metrics[f'recall@{k}'], f"정답 근거 청크 중 상위 {k}개 검색 결과에 든 비율(물품 평균)"),
            (f'근거 Hit Rate@{k}', 'pct', hit, f"정답 근거를 1개 이상 찾은 물품 {round(hit * ret['count'])}/{ret['count']}"),
            (f'근거 MRR@{k}', 'f4', metrics[f'mrr@{k}'], '첫 정답 근거의 순위 역수 평균(높을수록 상위에 위치)'),
        ]
    return rows


def final_present(eval_dir, run, args):
    report = load(eval_dir / 'final_answer/report.json')
    base_report = load(Path(args.baseline_eval) / 'final_answer/report.json') if args.baseline_eval else None
    cl, ret, cov, details = report['classification'], report['retrieval'], report['coverage'], report['details']
    rows = final_rows(report)
    base_rows = final_rows(base_report) if base_report else None
    per = cl['overall']['per_class']
    class_rows = [(label, pct(per[label]['precision']), pct(per[label]['recall']), pct(per[label]['f1']), per[label]['support']) for label in LABELS]

    decisions = details['decisions']
    confusion = Counter((d['expected'], d['predicted']) for d in decisions if d['expected'] != d['predicted'])
    top = confusion.most_common(4)
    risky = sum(c for (e, p), c in confusion.items() if e == '불가' and p in ('가능', '조건부 가능'))
    k_max = max(report['settings']['k'])
    hit_max = round(ret['metrics'][f'hit_rate@{k_max}'] * ret['count'])
    skipped = sum((cov.get('retrieval_skipped') or {}).values())
    cause_rows = [
        ('판정 불일치', f"{sum(confusion.values())}건 / {len(decisions)}판정",
         '가장 많은 혼동(정답→모델): ' + ', '.join(f'{e}→{p} {c}건' for (e, p), c in top)),
        ('확인 필요 판정 불일치', f"{per['확인 필요']['support'] - round(per['확인 필요']['recall'] * per['확인 필요']['support'])}건 / {per['확인 필요']['support']}판정",
         '근거 또는 속성이 부족한 사례에서 모델이 확인 필요 대신 다른 라벨을 선택'),
        ('위험 허용 오답', f'{risky}건', '정답은 불가인데 가능·조건부 가능으로 판정한 경우(안전 관점의 위험 오답)'),
        ('근거 미적중 물품', f"{ret['count'] - hit_max}개 / {ret['count']}개", f'상위 {k_max}개 안에 정답 근거가 없는 물품. 구조화 미연결·검색 손실을 상세 기록으로 구분해야 함'),
        ('물품 미연결', f"{cov['reference_items'] - cov['matched_items']}개 / {cov['reference_items']}개",
         f"이름으로 연결되지 않은 정답 물품(유효 정답 라벨은 누락 오답으로 포함). 추가 출력 물품 {cov['extra_items']}개도 분모에 포함"),
    ]
    notes = list(args.note or [])
    key_lines = [
        f"- **최종 판정 Accuracy {pct(cl['overall']['accuracy'])} / Macro-F1 {cl['overall']['macro_f1']:.4f}**: 기내·위탁 판정과 라벨 균형",
        f"- **확인 필요 Recall {pct(per['확인 필요']['recall'])}**: 근거·속성 부족을 알아채는 능력",
        f"- **근거 Recall@{k_max} {pct(ret['metrics'][f'recall@{k_max}'])} / Hit Rate@{k_max} {pct(ret['metrics'][f'hit_rate@{k_max}'])}**: 검색이 정답 근거를 후보에 담는 능력",
        f"- **질문 전체 판정 일치율 {pct(report['question_all_decisions_accuracy'])}**: 질문 하나의 모든 판정이 맞은 비율",
    ]
    if base_report:
        key_lines.append(f"- {args.baseline_label} 대비 변화는 1번 표의 '{args.baseline_label} 대비' 열 참고")
    text = report.get('text', {})
    reference = [('평가 대상 질문', f"{cov['questions']}문항 (유효 응답 {cov['valid_answers']})"),
                 ('정답 물품 / 연결 물품', f"{cov['reference_items']}개 / {cov['matched_items']}개"),
                 ('근거 평가 물품', f"{ret['count']}개 (정답 근거 미작성 {skipped}개 제외)"),
                 ('정답 파일', f"`{base_name(report['provenance']['dataset'])}` (sha256 {short(report['provenance'].get('dataset_sha256'))})")]
    md = [f'# 최종 답변 평가 — {args.label}', '',
          settings_section(run, args.purpose, notes, reference[3:] + [('평가 방식', '물품 이름 연결 후 라벨 일치·검색 순위 지표를 계산. 평가용 LLM 호출 없음')]),
          '## 1. 종합 평가', '', metric_table(args.label, rows, base_rows, args.baseline_label), '',
          '## 2. 라벨별 성능', '', table(['정답 라벨', 'Precision', 'Recall', 'F1', '정답 판정 수'], class_rows), '',
          '## 3. 오류 원인 분석', '', table(['오류 유형', '건수', '주요 원인'], cause_rows), '',
          '## 평가 범위', '', table(['항목', '값'], reference[:3]), '',
          '## 핵심 지표 (장표용)', '', *key_lines, '',
          '## 지표 설명', '',
          '- **최종 판정 Accuracy:** 정답 물품의 기내·위탁 판정(가능·불가·조건부 가능·확인 필요) 일치율. 물품 이름 연결에 실패한 정답은 오답으로 센다.',
          '- **Macro-F1:** 4개 라벨의 F1을 같은 비중으로 평균한 값. 정답 수가 적은 라벨(확인 필요 등)의 성능이 반영된다.',
          '- **확인 필요 Recall:** 정답이 확인 필요인 판정 중 모델도 확인 필요로 답한 비율. 근거 부족을 인지하는 능력이다.',
          '- **근거 Recall@K / Hit Rate@K / MRR@K:** 물품별 정답 근거 청크가 상위 K개 검색 결과에 포함되는 비율 / 하나라도 포함한 물품의 비율 / 첫 정답 근거 순위의 역수 평균. 정답 근거가 없는 물품은 0점이 아니라 평가에서 제외한다.',
          '- **위험 허용 오답:** 정답이 불가인데 가능·조건부 가능으로 판정한 판정 수.']
    if text:
        md += [f"- **참고 지표:** notes의 BLEU-4 {text.get('corpus_bleu4', 0):.4f}, ROUGE-L {text.get('rouge', {}).get('rougeL', {}).get('f1', 0):.4f}는 표현 유사도이며 규정·수치의 정확도가 아니다."]
    return '\n'.join(md) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-dir', type=Path, required=True, help='query_structure/·final_answer/가 있는 evaluate/attempt_N 폴더')
    parser.add_argument('--run', type=Path, required=True, help='평가한 실행 폴더(summary.json의 설정을 표에 사용)')
    parser.add_argument('--label', required=True, help='결과 열 이름(예: ver2, ver3/attempt_2)')
    parser.add_argument('--purpose', required=True, help='실험 목적 한 문장')
    parser.add_argument('--baseline-eval', type=Path, help='비교할 이전 evaluate 폴더')
    parser.add_argument('--baseline-label', default='이전')
    parser.add_argument('--note', action='append', help='장표에 함께 적을 주의 사항(여러 번 지정 가능)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    outputs = {args.eval_dir / 'query_structure/present.md': query_present, args.eval_dir / 'final_answer/present.md': final_present}
    for path in outputs:
        if path.exists() and not args.overwrite:
            parser.error(f'이미 있습니다(덮어쓰려면 --overwrite): {path}')
    for path, build in outputs.items():
        path.write_text(build(args.eval_dir, args.run, args), encoding='utf-8')
        print('Saved:', path)


if __name__ == '__main__':
    main()
