"""저장된 records를 채점한다. LLM·Chroma·네트워크를 호출하지 않는다."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import unicodedata

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from ranx import Qrels, Run, evaluate as evaluate_ranking
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from scripts.query_evaluation import evaluate_queries


ROOT = Path(__file__).resolve().parents[1]
STATUSES = ['가능', '불가', '조건부 가능', '확인 필요']
SIDES = ['carry_on', 'checked_baggage']
MISSING, INVALID, NULL, EXTRA = '__MISSING__', '__INVALID__', '__NULL__', '__EXTRA__'
RANK_METRICS = ['recall', 'precision', 'f1', 'hit_rate', 'mrr', 'map']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


class KoreanTokenizer:
    """Unicode 단어 토큰. 다운로드·형태소 분석·영어 stemming을 사용하지 않는다."""
    def tokenize(self, text):
        return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold())


def flatten_dataset(dataset):
    if not isinstance(dataset, dict):
        raise ValueError('데이터셋은 분류별 질문 배열이어야 합니다.')
    cases = {}
    for category, rows in dataset.items():
        if not isinstance(rows, list):
            raise ValueError(f'{category}: 질문 배열이 필요합니다.')
        for case in rows:
            key = case['id']
            if not isinstance(key, str) or not key or key in cases:
                raise ValueError('질문 ID는 비어 있지 않은 고유 문자열이어야 합니다.')
            if not isinstance(case.get('question'), str) or not case['question'].strip():
                raise ValueError(f'{key}: 질문이 비어 있습니다.')
            items = case.get('answer', {}).get('items')
            if not isinstance(items, list):
                raise ValueError(f'{key}: answer.items 배열이 필요합니다.')
            for item in items:
                if not isinstance(item.get('name'), str) or not item['name'].strip():
                    raise ValueError(f'{key}: 물품명이 필요합니다.')
                if any(side not in item or item[side] not in [*STATUSES, None] for side in SIDES):
                    raise ValueError(f'{key}: 올바른 기내·위탁 정답 상태가 필요합니다.')
                if item.get('notes') is not None and not isinstance(item['notes'], str):
                    raise ValueError(f'{key}: notes는 문자열 또는 null이어야 합니다.')
                aliases = item.get('aliases', [])
                if not isinstance(aliases, list) or any(not isinstance(x, str) or not x.strip() for x in aliases):
                    raise ValueError(f'{key}: aliases는 물품명 문자열 배열이어야 합니다.')
            cases[key] = dict(case, category=category)
    return cases


def load_records(run_dir, cases):
    folder = run_dir / 'records'
    if not folder.is_dir():
        raise ValueError('--run에는 records/가 있는 실행 폴더를 지정하세요.')
    records, hashes, unknown = {}, {}, []
    for path in sorted(folder.glob('*.json')):
        record = read_json(path)
        key = record.get('id')
        if not isinstance(key, str) or key in records:
            raise ValueError(f'{path.name}: 누락되거나 중복된 실행 ID입니다.')
        if key not in cases:
            unknown.append(key)
            continue
        if record.get('question') != cases[key]['question']:
            raise ValueError(f'{key}: 정답과 실행 기록의 질문이 다릅니다.')
        records[key] = record
        hashes[path.name] = digest(path)
    return records, hashes, unknown


def match_items(expected, actual):
    """순서로 추측하지 않는다. 이름/명시적 별칭이 일대일일 때만 연결한다."""
    candidate_sets = []
    for item in expected:
        names = {normalized(x) for x in [item['name'], *item.get('aliases', [])]}
        candidate_sets.append({i for i, other in enumerate(actual)
                               if isinstance(other, dict) and isinstance(other.get('name'), str)
                               and normalized(other['name']) in names})
    counts = Counter(i for candidates in candidate_sets for i in candidates)
    matches, reasons = {}, {}
    for i, candidates in enumerate(candidate_sets):
        if len(candidates) == 1 and counts[next(iter(candidates))] == 1:
            matches[i] = next(iter(candidates))
        else:
            reasons[i] = 'ambiguous_item' if candidates else 'missing_item'
    return matches, reasons, sorted(set(range(len(actual))) - set(matches.values()))


def source_id(source, source_key, retrieved=False):
    if isinstance(source, str) and source.strip():
        return validate_source_id(source, source_key)
    if not isinstance(source, dict):
        raise ValueError('근거는 ID 문자열 또는 metadata 객체여야 합니다.')
    value = source.get(source_key)
    if value is None and retrieved:
        value = source.get('metadata', {}).get(source_key)
        if value is None and source_key == 'chunk_id':
            value = source.get('id')
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'근거에 {source_key} 문자열이 필요합니다. 자동으로 다른 식별자를 대신 쓰지 않습니다.')
    return validate_source_id(value, source_key)


def validate_source_id(value, source_key):
    """이 프로젝트의 청크 ID 형식: 문서ID#0 이상의 정수. 문서 ID 혼입 방지."""
    if source_key == 'chunk_id' and (not re.fullmatch(r'[^#\r\n]+#[0-9]+', value)
                                     or not value.rpartition('#')[0].strip()):
        raise ValueError(f'청크 평가에는 문서ID#청크번호 형식의 chunk_id가 필요합니다: {value!r}. '
                         '문서 ID를 자동으로 청크로 확장하지 않습니다.')
    return value


def gold_sources(item, source_key):
    if 'source' in item and 'sources' in item and item['source'] != item['sources']:
        raise ValueError('source와 sources가 서로 다릅니다. 한 필드만 사용하세요.')
    sources = item.get('sources', item.get('source'))
    if sources is None:
        return None
    if not isinstance(sources, list):
        raise ValueError('정답 source/sources는 배열 또는 null이어야 합니다.')
    return list(dict.fromkeys(source_id(s, source_key) for s in sources))


def prediction_label(item, side):
    if item is None:
        return MISSING
    if side not in item:
        return INVALID
    value = item[side]
    if value is None:
        return NULL
    return value if value in STATUSES else INVALID


def classification(rows):
    if not rows:
        return {'count': 0, 'accuracy': None, 'macro_f1': None,
                'per_class': {}, 'confusion_matrix': None}
    truth, predicted = [r['expected'] for r in rows], [r['predicted'] for r in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=STATUSES, zero_division=0)
    labels = [*STATUSES, MISSING, INVALID, NULL, EXTRA]
    return {
        'count': len(rows), 'accuracy': float(accuracy_score(truth, predicted)),
        'macro_f1': float(f1.mean()),
        'per_class': {label: {'precision': float(p), 'recall': float(r),
                             'f1': float(f), 'support': int(n)}
                      for label, p, r, f, n in zip(STATUSES, precision, recall, f1, support)},
        'confusion_matrix': {'labels': labels, 'rows_true_columns_predicted':
                             confusion_matrix(truth, predicted, labels=labels).tolist()},
    }


def text_metrics(pairs):
    if not pairs:
        return {'count': 0, 'mean_sentence_bleu4': None, 'corpus_bleu4': None, 'rouge': None}
    tokenizer = KoreanTokenizer()
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=tokenizer)
    smoothing = SmoothingFunction().method1
    references, hypotheses, bleu_scores, rouge_scores = [], [], [], []
    for pair in pairs:
        ref, pred = tokenizer.tokenize(pair['reference']), tokenizer.tokenize(pair['prediction'])
        references.append([ref])
        hypotheses.append(pred)
        pair['bleu4'] = float(sentence_bleu([ref], pred, smoothing_function=smoothing)) if pred else 0.0
        scores = scorer.score(pair['reference'], pair['prediction'])
        pair['rouge'] = {name: dict(zip(['precision', 'recall', 'f1'], map(float, score)))
                         for name, score in scores.items()}
        bleu_scores.append(pair['bleu4'])
        rouge_scores.append(pair['rouge'])
    return {
        'count': len(pairs), 'mean_sentence_bleu4': sum(bleu_scores) / len(pairs),
        'corpus_bleu4': float(corpus_bleu(references, hypotheses, smoothing_function=smoothing)),
        'rouge': {name: {field: sum(row[name][field] for row in rouge_scores) / len(pairs)
                         for field in ['precision', 'recall', 'f1']}
                  for name in ['rouge1', 'rouge2', 'rougeL']},
    }


def retrieval_metrics(entries, ks):
    metrics = [f'{metric}@{k}' for k in ks for metric in RANK_METRICS]
    if not entries:
        return {'count': 0, 'metrics': {name: None for name in metrics}}
    # ranx cannot construct a Run containing only empty result lists. Empty searches
    # have zero for all six metrics; keep them in the macro denominator explicitly.
    nonempty = [entry for entry in entries if entry['retrieved_ids']]
    for entry in entries:
        entry['metrics'] = {name: 0.0 for name in metrics}
    if not nonempty:
        return {'count': len(entries), 'metrics': {name: 0.0 for name in metrics}}
    qrels = {entry['key']: {doc: 1 for doc in entry['expected_ids']} for entry in nonempty}
    # Unique decreasing scores preserve stored rank, even for lexicographically reversed IDs.
    runs = {entry['key']: {doc: float(len(entry['retrieved_ids']) - i)
                          for i, doc in enumerate(entry['retrieved_ids'])} for entry in nonempty}
    run = Run(runs)
    evaluate_ranking(Qrels(qrels), run, metrics, threads=1, make_comparable=True)
    for entry in nonempty:
        entry['metrics'] = {name: float(run.scores[name][entry['key']]) for name in metrics}
    return {'count': len(entries), 'metrics': {
        name: sum(entry['metrics'][name] for entry in entries) / len(entries) for name in metrics}}


def evaluate_dataset(dataset_path, run_dir, ks=(5,), source_key='chunk_id', id_map_path=None,
                     query_dataset_path=None):
    ks = sorted(set(ks))
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError('K는 1 이상의 정수여야 합니다.')
    if source_key not in ['chunk_id', 'doc_id', 'source_url']:
        raise ValueError('source_key는 chunk_id, doc_id, source_url 중 하나여야 합니다.')
    id_map = {}
    if id_map_path is not None:
        if source_key == 'source_url':
            raise ValueError('ID 매핑은 chunk_id 또는 doc_id 평가에서만 사용합니다.')
        mapping = read_json(id_map_path)
        map_key = 'document_id_map' if source_key == 'doc_id' else 'chunk_id_map'
        id_map = mapping.get(map_key)
        if not isinstance(id_map, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               or not k or not v for k, v in id_map.items()):
            raise ValueError(f'ID 매핑에 올바른 {map_key} 객체가 필요합니다.')
    cases = flatten_dataset(read_json(dataset_path))
    if not cases:
        raise ValueError('평가 질문이 없습니다.')
    records, record_hashes, unknown = load_records(run_dir, cases)
    rows, retrieval, text_pairs, questions, skipped, diagnostics = [], [], [], [], [], []
    tokenizer = KoreanTokenizer()
    for key, case in cases.items():
        record = records.get(key)
        valid_answer = (record is not None and record.get('status') == 'success'
                        and isinstance(record.get('answer'), dict)
                        and isinstance(record['answer'].get('items'), list)
                        and all(isinstance(item, dict) for item in record['answer']['items']))
        actual = record['answer']['items'] if valid_answer else []
        expected = case['answer']['items']
        matches, reasons, extra = match_items(expected, actual)
        groups = record.get('groups') if record else []
        if groups is not None and (not isinstance(groups, list) or
                                  any(not isinstance(g, dict) or not isinstance(g.get('item'), dict) or
                                      not isinstance(g.get('documents'), list) for g in groups)):
            raise ValueError(f'{key}: 잘못된 groups 검색 기록입니다.')
        group_matches, group_reasons, _ = match_items(expected, [g['item'] for g in groups or []])
        exact = valid_answer and not extra and len(matches) == len(expected)
        for index, item in enumerate(expected):
            identity = {'question_id': key, 'item_index': index, 'name': item['name']}
            pred = actual[matches[index]] if index in matches else None
            if index in reasons:
                diagnostics.append(dict(identity, reason=reasons[index]))
            for side in SIDES:
                label = prediction_label(pred, side)
                if item[side] is None:
                    diagnostics.append(dict(identity, side=side, reason='non_applicable_reference',
                                            predicted=label, correct=label == NULL))
                    exact = exact and label == NULL
                    continue
                rows.append(dict(identity, side=side, expected=item[side], predicted=label))
                exact = exact and item[side] == label
            reference_notes = item.get('notes') or ''
            if tokenizer.tokenize(reference_notes):
                predicted_notes = pred.get('notes') if pred else None
                text_pairs.append(dict(identity, reference=reference_notes,
                                       prediction=predicted_notes if isinstance(predicted_notes, str) else ''))
            try:
                gold = gold_sources(item, source_key)
            except ValueError as error:
                raise ValueError(f"{key} / {item['name']}: {error}") from None
            if gold:
                gold = list(dict.fromkeys(validate_source_id(id_map.get(identifier, identifier), source_key)
                                          for identifier in gold))
            if not gold:
                skipped.append(dict(identity, reason='unannotated_sources' if gold is None else 'no_positive_sources'))
                continue
            if record is not None and groups is None:
                skipped.append(dict(identity, reason='missing_retrieval_trace'))
                continue
            if index in group_reasons and group_reasons[index] == 'ambiguous_item':
                skipped.append(dict(identity, reason='ambiguous_retrieval_item'))
                continue
            if index not in group_matches:
                diagnostics.append(dict(identity, reason='missing_retrieval_item'))
            documents = groups[group_matches[index]]['documents'] if index in group_matches else []
            # Deduplicate at the chosen granularity, retaining the first rank.
            retrieved = [source_id(doc, source_key, retrieved=True) for doc in documents]
            retrieved = list(dict.fromkeys(validate_source_id(id_map.get(identifier, identifier), source_key)
                                           for identifier in retrieved))
            retrieval.append(dict(identity, key=f'{key}:{index}', expected_ids=gold, retrieved_ids=retrieved))
            if max(ks) > len(retrieved):
                diagnostics.append(dict(identity, reason='fewer_results_than_k', retrieved=len(retrieved)))
        for index in extra:
            diagnostics.append({'question_id': key, 'predicted_item_index': index,
                                'name': actual[index].get('name'), 'reason': 'unmatched_prediction'})
            for side in SIDES:
                rows.append({'question_id': key, 'name': actual[index].get('name'), 'side': side,
                             'expected': EXTRA, 'predicted': prediction_label(actual[index], side)})
        questions.append({'id': key, 'category': case['category'],
                          'record_status': record.get('status') if record else 'missing',
                          'valid_answer': valid_answer, 'all_decisions_correct': bool(exact),
                          'expected_items': len(expected), 'matched_items': len(matches),
                          'extra_items': len(extra)})
    retrieval_summary = retrieval_metrics(retrieval, ks)
    text_summary = text_metrics(text_pairs)
    query_report = evaluate_queries(query_dataset_path, cases, records) if query_dataset_path else {
        'evaluation_status': 'pending_reference', 'metrics': None,
        'outputs': [{'question_id': key, 'question': case['question'],
                     'record_status': records.get(key, {}).get('status', 'missing'),
                     'parsed': records.get(key, {}).get('parsed')}
                    for key, case in cases.items()]}
    return {
        'schema_version': 4,
        'provenance': {'evaluated_at': datetime.now(timezone.utc).isoformat(),
                       'dataset': str(dataset_path.resolve()), 'dataset_sha256': digest(dataset_path),
                       'run': str(run_dir.resolve()), 'record_sha256': record_hashes,
                       'evaluator_sha256': digest(Path(__file__)),
                       'id_mapping': {'path': str(id_map_path.resolve()), 'sha256': digest(id_map_path)}
                                     if id_map_path else None,
                       'libraries': {name: version(name) for name in ['ranx', 'scikit-learn', 'nltk', 'rouge-score']}},
        'settings': {'k': ks, 'source_key': source_key, 'retrieval_input': 'groups[].documents (stored rank)',
                     'evaluation_scope': 'final_answer',
                     'retrieval_unit': 'chunk' if source_key == 'chunk_id' else 'document',
                     'item_matching': 'NFC/casefold/whitespace-normalized unique name or reference aliases; no positional matching',
                     'macro_f1_labels': STATUSES, 'precision_denominator': 'K (ranx precision@K)',
                     'text_field': 'notes', 'tokenizer': 'Unicode words, casefold, NFC, no stemming',
                     'bleu': 'uniform 1-4 grams, NLTK smoothing method1; corpus and mean sentence reported',
                     'llm_calls': 0},
        'coverage': {'questions': len(cases), 'records_found': len(records),
                     'missing_records': sorted(set(cases) - set(records)), 'unknown_record_ids': unknown,
                     'valid_answers': sum(q['valid_answer'] for q in questions),
                     'reference_items': sum(q['expected_items'] for q in questions),
                     'matched_items': sum(q['matched_items'] for q in questions),
                     'extra_items': sum(q['extra_items'] for q in questions),
                     'retrieval_evaluated': len(retrieval),
                     'retrieval_skipped': dict(Counter(s['reason'] for s in skipped))},
        'classification': {'overall': classification(rows),
                           **{side: classification([r for r in rows if r['side'] == side]) for side in SIDES}},
        'question_all_decisions_accuracy': sum(q['all_decisions_correct'] for q in questions) / len(questions),
        'retrieval': retrieval_summary, 'text': text_summary,
        'query_structure': query_report,
        'details': {'questions': questions, 'decisions': rows, 'retrieval': retrieval,
                    'retrieval_skipped': skipped, 'text': text_pairs, 'diagnostics': diagnostics},
    }


def write_csv(path, fields, rows):
    # BOM makes Korean text readable when the CSV is opened directly in Excel.
    with path.open('x', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def export_report(report, folder):
    """최종 답변 평가와 구조화 원본을 구분해 저장한다. 구조화 점수는 생성하지 않는다."""
    root = folder
    files = ['report.json', 'summary.csv', 'questions.csv', 'decisions.csv', 'retrieval.csv',
             'text.csv', 'diagnostics.csv', 'per_class.csv', 'confusion_matrix.csv', 'README.md']
    if (any((root / name).exists() for name in files) or
            (root / 'final_answer').exists() or (root / 'query_structure').exists()):
        raise ValueError('결과 파일이 이미 있습니다. 새 --output-dir 폴더를 지정하세요.')
    folder = root / 'final_answer'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'report.json').open('x', encoding='utf-8') as file:
        json.dump({key: value for key, value in report.items() if key != 'query_structure'},
                  file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    summary = []
    for name, value in report['coverage'].items():
        summary.append({'영역': '평가 범위', '지표': name, '값': value, '표본수': ''})
    for name, value in report['retrieval']['metrics'].items():
        summary.append({'영역': '검색', '지표': name, '값': value, '표본수': report['retrieval']['count']})
    per_class, matrices = [], []
    for scope, metrics in report['classification'].items():
        for name in ['accuracy', 'macro_f1']:
            summary.append({'영역': scope, '지표': name, '값': metrics[name], '표본수': metrics['count']})
        for label, scores in metrics['per_class'].items():
            per_class.append(dict(scope=scope, label=label, **scores))
        matrix = metrics['confusion_matrix']
        if matrix:
            for i, true in enumerate(matrix['labels']):
                for j, predicted in enumerate(matrix['labels']):
                    matrices.append({'scope': scope, 'expected': true, 'predicted': predicted,
                                     'count': matrix['rows_true_columns_predicted'][i][j]})
    summary.append({'영역': '질문', '지표': 'all_decisions_accuracy',
                    '값': report['question_all_decisions_accuracy'], '표본수': report['coverage']['questions']})
    for name in ['mean_sentence_bleu4', 'corpus_bleu4']:
        summary.append({'영역': '표현', '지표': name, '값': report['text'][name],
                        '표본수': report['text']['count']})
    for name, scores in (report['text']['rouge'] or {}).items():
        for metric, value in scores.items():
            summary.append({'영역': '표현', '지표': f'{name}.{metric}', '값': value,
                            '표본수': report['text']['count']})
    for row in summary:
        row['검색단위'] = report['settings']['source_key'] if row['영역'] == '검색' else ''
    write_csv(folder / 'summary.csv', ['영역', '지표', '값', '표본수', '검색단위'], summary)
    query_metrics = report.get('query_structure', {}).get('metrics') or {}
    combined = summary + [{'영역': '질의 구조화', '지표': name, '값': value,
                           '표본수': query_metrics.get('annotated_questions'), '검색단위': ''}
                          for name, value in query_metrics.items()]
    write_csv(root / 'summary.csv', ['영역', '지표', '값', '표본수', '검색단위'], combined)
    details = report['details']
    write_csv(folder / 'questions.csv', ['id', 'category', 'record_status', 'valid_answer',
              'all_decisions_correct', 'expected_items', 'matched_items', 'extra_items'], details['questions'])
    write_csv(folder / 'decisions.csv', ['question_id', 'item_index', 'name', 'side', 'expected', 'predicted', 'correct'],
              [dict(row, correct=row['expected'] == row['predicted']) for row in details['decisions']])
    write_csv(folder / 'retrieval.csv', ['question_id', 'item_index', 'name', 'source_key', 'expected_ids', 'retrieved_ids',
              *report['retrieval']['metrics']],
              [dict(row, source_key=report['settings']['source_key'], **row['metrics']) for row in details['retrieval']])
    text_fields = [f'{name}.{metric}' for name in ['rouge1', 'rouge2', 'rougeL']
                   for metric in ['precision', 'recall', 'f1']]
    write_csv(folder / 'text.csv', ['question_id', 'item_index', 'name', 'reference', 'prediction', 'bleu4', *text_fields],
              [dict(row, **{f'{name}.{metric}': value for name, scores in row['rouge'].items()
                            for metric, value in scores.items()}) for row in details['text']])
    write_csv(folder / 'diagnostics.csv', ['question_id', 'item_index', 'name', 'reason', 'side',
              'predicted', 'correct', 'predicted_item_index', 'retrieved'],
              details['diagnostics'] + details['retrieval_skipped'])
    write_csv(folder / 'per_class.csv', ['scope', 'label', 'precision', 'recall', 'f1', 'support'], per_class)
    write_csv(folder / 'confusion_matrix.csv', ['scope', 'expected', 'predicted', 'count'], matrices)
    lines = ['# 오프라인 평가 결과', '',
             f"- 정답: `{report['provenance']['dataset']}`",
             f"- 실행: `{report['provenance']['run']}`",
             f"- 비교 단위: `{report['settings']['source_key']}` / K={report['settings']['k']}",
             '- 평가용 LLM 호출: 0회', '',
             '`summary.csv`에서 전체 점수를 확인하고 `report.json`에서 설정·해시·상세 결과를 확인하세요.',
             'CSV는 Excel에서 한글을 읽을 수 있도록 UTF-8 BOM으로 저장했습니다. 빈 점수는 미평가이며 0점과 다릅니다.', '',
             '| 영역 | 지표 | 값 | 표본수 |', '| --- | --- | ---: | ---: |']
    for row in summary:
        if isinstance(row['값'], (int, float)) and row['영역'] != '평가 범위':
            lines.append(f"| {row['영역']} | {row['지표']} | {row['값']:.6f} | {row['표본수']} |")
    lines += ['', '## 해석 시 확인할 점', '',
              '- 검색은 저장된 결과를 평가합니다. 최신 모델·임베딩을 새로 실행한 성능이 아닙니다.',
              ('- 청크 단위 평가입니다. 같은 문서의 다른 청크는 각각 별도로 채점하며 K는 상위 청크 수입니다.'
               if report['settings']['source_key'] == 'chunk_id' else
               '- 문서 수준 평가는 동일 문서의 청크를 첫 등장 순서로 축약합니다. K는 축약된 문서 순위입니다.'),
              '- 이름/별칭이 정확히 연결되지 않는 물품은 누락·추가로 반영됩니다. diagnostics.csv에서 확인하세요.',
              '- 정답 sources가 없으면 검색 평가에서 제외합니다. coverage와 제외 사유를 함께 확인하세요.',
              '- BLEU/ROUGE는 표현 유사도이며 조건·규정의 의미적 정확성을 보장하지 않습니다.', '']
    (folder / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    write_evaluation_guide(report, root)


def write_evaluation_guide(report, root):
    """저장된 점수의 읽는 순서와 후속 작업을 실행마다 함께 보존한다."""
    query = root / 'query_structure'
    query.mkdir(parents=True, exist_ok=True)
    structure = report.get('query_structure', {
        'evaluation_status': 'pending_reference', 'metrics': None, 'outputs': []})
    with (query / 'outputs.json').open('x', encoding='utf-8') as file:
        json.dump({key: value for key, value in structure.items() if key not in ['fields', 'items', 'questions']},
                  file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    if structure.get('metrics') is not None:
        with (query / 'report.json').open('x', encoding='utf-8') as file:
            json.dump(structure, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write('\n')
        write_csv(query / 'summary.csv', ['metric', 'value'],
                  [{'metric': name, 'value': value} for name, value in structure['metrics'].items()])
        write_csv(query / 'conditions.csv', ['question_id', 'field', 'expected', 'predicted',
                  'present', 'correct', 'expected_is_null'], structure['fields'])
        write_csv(query / 'items.csv', ['question_id', 'item_index', 'name', 'predicted_item_index',
                  'expected_details', 'predicted_details', 'matched', 'details_correct', 'reason'], structure['items'])
        write_csv(query / 'questions.csv', ['question_id', 'valid_parsed', 'conditions_exact', 'items_exact',
                  'details_exact', 'all_exact', 'expected_items', 'predicted_items', 'matched_items'], structure['questions'])
        write_csv(query / 'per_field.csv', ['field', 'count', 'accuracy'],
                  [dict(field=field, **values) for field, values in structure['per_field'].items()])
        (query / 'README.md').write_text(
            '# 질의 구조화 평가\n\n'
            f"정답: `{structure['reference']['path']}`\n\n"
            '`summary.csv`는 전체 지표, `per_field.csv`는 여행 조건 필드별 정확도입니다.\n'
            '`conditions.csv`는 질문×조건 필드 비교, `items.csv`는 물품명 대응과 details 비교, '
            '`questions.csv`는 질문별 완전 일치 여부입니다.\n'
            '`report.json`에 설정·정답 해시·상세 비교, `outputs.json`에 기대/실제 parsed를 저장합니다.\n\n'
            '문자열은 Unicode·대소문자·연속 공백만 정규화합니다. 의미 동등성이나 단위 변환은 추론하지 않습니다. '
            'details_accuracy는 정답 물품 전체(누락 포함)가 분모입니다. name 연결에 실패하면 details도 오답입니다.\n'
            'query_exact_match는 모든 조건·물품·속성이 일치해야 성공이며, 요청 내부 item_id는 비교하지 않습니다.\n'
            '조건의 expected_is_null·present를 함께 확인하세요. CSV 빈 셀만으로 null과 필드 누락을 구별하지 마세요.\n'
            '정답의 한국 기본값 정책과 실행 프롬프트의 미명시 null 정책이 다르면 해당 차이도 오답에 반영됩니다. '
            '높은 null 필드 비율의 영향을 피하려면 nonnull_condition_accuracy도 함께 봅니다.\n'
            '정답이나 최종 판정은 모델 입력에 전달하지 않으며, 평가용 LLM 호출은 없습니다.\n', encoding='utf-8')
    else:
        (query / 'README.md').write_text(
        '# 질의 구조화 출력 (평가 대기)\n\n'
        '`outputs.json`은 records에 저장된 parsed 원본과 질문 ID를 보존합니다. 새 LLM 호출은 없습니다.\n'
        '질문별 구조화 정답이 준비되지 않아 평가 상태는 pending_reference, metrics는 null입니다.\n'
        '이는 0점도 성공도 아닙니다. 최종 답변 점수와 합산하지 않습니다.\n\n'
        '추후 질문에 명시된 물품·속성·여행 조건과 미명시 필드의 null을 정답으로 구축한 뒤 평가합니다.\n'
            '최종 답변 notes의 규정 제한을 사용자 입력 속성의 정답으로 사용하지 않습니다.\n', encoding='utf-8')
    coverage = report['coverage']
    unit = report['settings']['source_key']
    guide = f'''# 평가 결과 읽는 방법

최종 답변의 물품별 검색·판정·설명 표현을 평가합니다.
질의 구조화 평가 상태: **{structure['evaluation_status']}**. 정답을 지정한 경우 별도로 채점하며 최종 답변 점수와 합산하지 않습니다.

- 정답 파일: `{report['provenance']['dataset']}`
- 평가한 실행: `{report['provenance']['run']}`
- 현재 저장된 검색 평가 단위: **{unit}**, K={report['settings']['k']}
- 질문 {coverage['questions']}개 / 정답 물품 {coverage['reference_items']}개 / 연결 물품 {coverage['matched_items']}개
- 검색 평가 {coverage['retrieval_evaluated']}개 / 제외 사유: `{json.dumps(coverage['retrieval_skipped'], ensure_ascii=False)}`
- doc_id로 저장된 과거 점수는 문서 단위입니다. 청크 ID 정답이 준비되면 새 폴더에서 chunk_id로 실행합니다.

## 무엇부터 보나요?

1. **summary.csv의 평가 범위**: missing_records, matched_items, extra_items, retrieval_skipped부터 확인합니다.
   평가 제외·물품 대응 실패를 확인하지 않고 전체 점수를 해석하지 마세요.
2. **최종 판정**: overall Accuracy와 Macro-F1, 이어 final_answer/per_class.csv의 불가·확인 필요 Recall을 확인합니다.
   final_answer/confusion_matrix.csv에서 불가→가능 등의 오분류 건수를 확인합니다.
3. **검색**: Recall@K를 우선 보고 Precision@K·mAP@K를 함께 봅니다. 물품별 실패는 final_answer/retrieval.csv로 내려갑니다.
4. **설명 표현**: BLEU·ROUGE는 참고값입니다. 수치·조건·출처가 실제로 맞는지 자동 검증한 점수가 아닙니다.
5. **원인 확인**: final_answer/decisions.csv·text.csv·diagnostics.csv에서 해당 question_id와 물품을 찾습니다.
6. **질의 구조화**: 정답을 지정했다면 query_structure/summary.csv의 물품 Precision/Recall, 조건·속성 정확도와
   query_exact_match를 확인합니다. conditions.csv·items.csv로 누락·추가·표현 차이를 구별합니다.

## 폴더와 CSV 설명

```text
evaluate/
├─ README.md
├─ summary.csv                 # 최종 답변 + 정답이 있는 경우 구조화 요약 (합산하지 않음)
├─ query_structure/
│  ├─ README.md
│  ├─ outputs.json             # parsed 원본과 기대 출력 (정답이 없으면 평가 대기)
│  └─ report.json / *.csv      # --query-dataset 지정 시 구조화 평가
└─ final_answer/
   ├─ report.json              # 전체 최종 답변 지표·설정·해시·상세 데이터
   ├─ summary.csv              # 루트 summary.csv와 동일한 최종 답변 요약
   ├─ questions.csv
   ├─ decisions.csv
   ├─ per_class.csv
   ├─ confusion_matrix.csv
   ├─ retrieval.csv
   ├─ text.csv
   └─ diagnostics.csv
```

| 파일 | 한 행의 단위 | 주요 열 / 읽는 방법 |
| --- | --- | --- |
| summary.csv | 평가 범위 또는 지표 하나 | 영역·지표·값·표본수. 새 출력에는 검색단위도 표시합니다. 빈 점수는 미평가이며 0과 다릅니다. |
| questions.csv | 질문 하나 | expected_items·matched_items·extra_items는 물품 대응 수. all_decisions_correct는 모든 물품의 기내/위탁 상태가 일치했는지입니다. 설명까지 모두 맞았다는 뜻은 아닙니다. |
| decisions.csv | 물품 하나 × 기내/위탁 | question_id·item_index(0부터 시작)·name·side로 식별. expected=정답, predicted=출력, correct=라벨 일치입니다. 추가 출력 물품에는 정답 item_index가 없습니다. |
| per_class.csv | 범위 × 판정 클래스 | scope는 overall/carry_on/checked_baggage. precision은 해당 판정 출력의 정확도, recall은 해당 정답의 회수율, support는 정답 표본수입니다. |
| confusion_matrix.csv | 범위 × 정답 × 출력 | expected가 실제 정답, predicted가 모델 출력, count가 발생 건수입니다. 0건 조합도 포함합니다. |
| retrieval.csv | 정답 물품 하나 | expected_ids=정답 근거, retrieved_ids=해당 물품의 검색 순위. Recall/Precision/F1/Hit Rate/MRR/mAP@K를 물품별로 기록합니다. 새 출력의 source_key 또는 report.json의 settings.source_key로 문서/청크 단위를 확인하세요. |
| text.csv | 정답 notes가 있는 물품 하나 | reference와 prediction은 설명 원문. bleu4, rouge1/2/L의 precision/recall/f1은 표현 유사도입니다. |
| diagnostics.csv | 진단 사건 하나 | 같은 물품에 여러 행이 생길 수 있습니다. reason과 question_id·name으로 미연결·추가·검색 제외 원인을 확인합니다. |

CSV는 UTF-8 BOM으로 저장해 Excel에서 한글을 읽을 수 있습니다.

## items별로 어떻게 연결하고 평가하나요?

- question_id 안에서 정답 items와 최종 answer.items를 이름/명시적 aliases로 일대일 연결합니다.
  Unicode·대소문자·연속 공백만 정규화하며 순서·임의 의미 유사도로 연결하지 않습니다.
- 검색은 같은 정답 물품과 groups[].item을 별도로 연결한 뒤 **그 그룹의 documents만** 비교합니다.
  질문 전체의 sources나 다른 물품 검색 결과를 합쳐 정답 처리하지 않습니다.
- 기내 carry_on과 위탁 checked_baggage는 서로 독립된 정답 라벨로 비교합니다.
- names가 달라 연결되지 않으면 missing_item, 출력에만 있으면 unmatched_prediction입니다.
  같은 이름의 여러 물품이 모호하면 ambiguous_item으로 기록합니다. 이 경우 사람의 별칭/대응 검수가 필요합니다.
- 따라서 최종 점수에는 구조화 물품 누락의 영향도 포함됩니다. 지금은 최종 파이프라인 평가이며
  별도의 구조화 정확도나 순수 검색기 성능으로 해석하지 않습니다.

## 분모·누락·제외 기준

- 판정 Accuracy는 정답 물품의 유효 상태와 추가 출력 물품의 상태를 포함합니다. 물품 누락도 오답입니다.
  최종 답변 실패를 정상적인 확인 필요로 대체하지 않습니다.
- __MISSING__=출력 물품 누락, __INVALID__=상태 누락/잘못된 값,
  __NULL__=출력 null, __EXTRA__=정답에 없는 추가 물품입니다.
- 정답 null은 네 클래스 분류에서 제외하지만 질문 단위 전체 일치 여부와 진단에 반영합니다.
- Macro-F1은 가능·불가·조건부 가능·확인 필요 네 클래스의 고정 평균입니다. support=0인 클래스는 0으로 처리합니다.
- 검색 지표는 양성 정답 근거가 있는 평가 가능 물품의 평균입니다. source null은 unannotated_sources,
  빈 배열은 no_positive_sources로 제외합니다. 문서 ID를 청크로 자동 확장하지 않습니다.
- Precision@K의 분모는 K입니다. AP@K의 분모는 전체 정답 근거 수입니다.
  같은 청크의 반복만 중복 제거하며, 청크 평가에서 같은 문서의 다른 청크는 각각 셉니다.
- 정답 notes가 null/빈 값이면 표현 평가에서 제외합니다. 정답 설명이 있는데 답변이 없으면 표현 점수는 0입니다.

## Action Items: 질의 구조화 평가

- 현재 구조화 평가 상태는 `{structure['evaluation_status']}`입니다. `--query-dataset`을 지정하면 정답을 검증하고 평가합니다.
- [ ] 새 평가셋의 기대 물품명·사용자 명시 속성과 실행 프롬프트의 기본값 정책을 대조·검수합니다.
- [ ] 출발/도착 장소·항공사·여행일·수하물 구분의 정답과 미명시 필드 null을 정의합니다.
- [ ] 현재 parsed 스키마에 맞춘 expected_parsed 또는 별도 구조화 정답 파일 형식을 확정합니다.
- [ ] 동의어·동일 이름/다른 속성 물품의 일대일 대응과 details의 허용 표현/정규화 규칙을 정합니다.
- [ ] 정답 준비 후 query_structure에서 추출 누락·잘못된 값·질문에 없는 정보 추가를 평가합니다.
- [ ] 구조화 정답이 없는 항목은 미평가(null)로 유지하고 최종 답변 점수와 분리합니다.

평가용 LLM 호출은 없습니다. 원본 정답과 실행 records는 수정하지 않습니다.
'''
    (root / 'README.md').write_text(guide, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/sample/questions.json')
    parser.add_argument('--query-dataset', type=Path, help='expected_parsed 구조화 정답. 질문 ID·원문이 일치해야 함')
    parser.add_argument('--run', type=Path, required=True, help='records/가 있는 attempt_N 폴더')
    parser.add_argument('--k', nargs='+', type=int, default=[5])
    parser.add_argument('--source-key', choices=['chunk_id', 'doc_id', 'source_url'], default='chunk_id')
    parser.add_argument('--id-map', type=Path, help='명시적 ID 변경 mapping.json. 원본은 수정하지 않음')
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument('--output', type=Path, help='새 JSON 파일. 기존 파일은 덮어쓰지 않음')
    outputs.add_argument('--output-dir', type=Path,
                         help='단계별 CSV/JSON 폴더. 기본: 실행 상위 폴더/evaluate/<실행명_UTC시각>')
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise ValueError('평가 결과가 이미 있습니다. 새 --output 파일을 지정하세요.')
        report = evaluate_dataset(args.dataset, args.run, args.k, args.source_key, args.id_map, args.query_dataset)
        if not args.output:
            output_dir = args.output_dir or args.run.parent / 'evaluate' / (
                f"{args.run.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
            export_report(report, output_dir)
            output = output_dir / 'final_answer' / 'report.json'
        else:
            output = args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as file:
                json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
                file.write('\n')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f'평가 결과: {output}')
    print(json.dumps({key: report[key] for key in ['coverage', 'retrieval', 'question_all_decisions_accuracy']},
                     ensure_ascii=False, indent=2))
    print(f"판정 Accuracy: {report['classification']['overall']['accuracy']}")
    print(f"판정 Macro-F1: {report['classification']['overall']['macro_f1']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
