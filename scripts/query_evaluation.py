"""질의 구조화 정답과 저장된 parsed의 필드·물품을 오프라인으로 비교한다."""
from collections import Counter
import hashlib
import json
import unicodedata

from app.schemas.query import ParsedQuery


CONDITION_FIELDS = ['departure.place', 'departure.airline', 'arrival.place',
                    'arrival.airline', 'travel_date', 'baggage_type']


def normalize(value):
    return ' '.join(unicodedata.normalize('NFC', value).casefold().split())


def same(expected, actual, present):
    if not present:
        return False
    if expected is None:
        return actual is None
    return isinstance(actual, str) and normalize(expected) == normalize(actual)


def get_field(data, path):
    for part in path.split('.'):
        if not isinstance(data, dict) or part not in data:
            return None, False
        data = data[part]
    return data, True


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_queries(path, final_cases, records):
    reference = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(reference, dict) or any(not isinstance(rows, list) for rows in reference.values()):
        raise ValueError('구조화 정답은 분류별 질문 배열이어야 합니다.')
    gold = {}
    for rows in reference.values():
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in gold:
                raise ValueError('구조화 정답 ID가 없거나 중복되었습니다.')
            key = row['id']
            if key not in final_cases or row.get('question') != final_cases[key]['question']:
                raise ValueError(f'{key}: 구조화 정답과 최종 답변 데이터셋의 질문이 다릅니다.')
            if 'expected_parsed' not in row:
                raise ValueError(f'{key}: expected_parsed가 필요합니다.')
            try:
                ParsedQuery.model_validate(row['expected_parsed'])
            except ValueError:
                raise ValueError(f'{key}: expected_parsed가 ParsedQuery 스키마와 일치하지 않습니다.') from None
            gold[key] = row['expected_parsed']
    if not gold:
        raise ValueError('구조화 정답이 비어 있습니다.')
    fields, items, questions, outputs = [], [], [], []
    totals = Counter()
    for key, case in final_cases.items():
        record = records.get(key, {})
        parsed = record.get('parsed')
        expected = gold.get(key)
        outputs.append({'question_id': key, 'question': case['question'],
                        'record_status': record.get('status', 'missing'),
                        'parsed': parsed, 'expected_parsed': expected})
        if expected is None:
            continue
        actual_items = parsed.get('items', []) if isinstance(parsed, dict) else []
        if not isinstance(actual_items, list):
            actual_items = []
        try:
            # item_id is assigned by the pipeline after parsing; it is not a gold field.
            cleaned = dict(parsed, items=[{k: v for k, v in item.items() if k != 'item_id'}
                                          for item in actual_items])
            ParsedQuery.model_validate(cleaned)
            valid = True
        except (ValueError, TypeError, AttributeError):
            valid = False
        conditions = parsed.get('conditions', {}) if isinstance(parsed, dict) else {}
        field_rows = []
        for name in CONDITION_FIELDS:
            truth, _ = get_field(expected['conditions'], name)
            value, present = get_field(conditions, name)
            correct = same(truth, value, present)
            field_rows.append({'question_id': key, 'field': name, 'expected': truth,
                               'predicted': value, 'present': present, 'correct': correct,
                               'expected_is_null': truth is None})
        fields.extend(field_rows)
        expected_items = expected['items']
        names = Counter(normalize(i['name']) for i in expected_items)
        matched = set()
        details_correct = 0
        for index, item in enumerate(expected_items):
            candidates = [j for j, pred in enumerate(actual_items)
                          if isinstance(pred, dict) and isinstance(pred.get('name'), str)
                          and normalize(pred['name']) == normalize(item['name'])]
            ambiguous = len(candidates) > 1 or names[normalize(item['name'])] > 1
            match = candidates[0] if len(candidates) == 1 and not ambiguous else None
            predicted_item = actual_items[match] if match is not None else {}
            if match is not None:
                matched.add(match)
            correct = same(item['details'], predicted_item.get('details'), 'details' in predicted_item)
            details_correct += int(correct)
            items.append({'question_id': key, 'item_index': index, 'name': item['name'],
                          'predicted_item_index': match, 'expected_details': item['details'],
                          'predicted_details': predicted_item.get('details'), 'details_correct': correct,
                          'matched': match is not None,
                          'reason': 'ambiguous_item' if ambiguous else 'matched' if match is not None else 'missing_item'})
        for index, item in enumerate(actual_items):
            if index not in matched:
                items.append({'question_id': key, 'predicted_item_index': index,
                              'name': item.get('name') if isinstance(item, dict) else None,
                              'predicted_details': item.get('details') if isinstance(item, dict) else None,
                              'matched': False, 'details_correct': False, 'reason': 'extra_item'})
        exact_items = valid and len(matched) == len(expected_items) == len(actual_items)
        exact_conditions = all(row['correct'] for row in field_rows)
        questions.append({'question_id': key, 'valid_parsed': valid,
                          'conditions_exact': valid and exact_conditions,
                          'items_exact': exact_items,
                          'details_exact': exact_items and details_correct == len(expected_items),
                          'all_exact': valid and exact_conditions and exact_items and details_correct == len(expected_items),
                          'expected_items': len(expected_items), 'predicted_items': len(actual_items),
                          'matched_items': len(matched)})
        totals.update(expected_items=len(expected_items), predicted_items=len(actual_items),
                      matched_items=len(matched), details_correct=details_correct)
    nonnull = [r for r in fields if not r['expected_is_null']]
    nulls = [r for r in fields if r['expected_is_null']]
    details_rows = [row for row in items if 'item_index' in row]
    nonnull_details = [row for row in details_rows if row['expected_details'] is not None]
    null_details = [row for row in details_rows if row['expected_details'] is None]
    precision = rate(totals['matched_items'], totals['predicted_items'])
    recall = rate(totals['matched_items'], totals['expected_items'])
    metrics = {
        'annotated_questions': len(questions), 'valid_parsed_questions': sum(q['valid_parsed'] for q in questions),
        'query_exact_match': rate(sum(q['all_exact'] for q in questions), len(questions)),
        'conditions_exact_match': rate(sum(q['conditions_exact'] for q in questions), len(questions)),
        'condition_field_accuracy': rate(sum(r['correct'] for r in fields), len(fields)),
        'nonnull_condition_accuracy': rate(sum(r['correct'] for r in nonnull), len(nonnull)),
        'null_condition_accuracy': rate(sum(r['correct'] for r in nulls), len(nulls)),
        'condition_addition_count': sum(r['present'] and r['predicted'] is not None for r in nulls),
        'condition_omission_count': sum(not r['present'] or r['predicted'] is None for r in nonnull),
        'expected_items': totals['expected_items'], 'predicted_items': totals['predicted_items'],
        'matched_items': totals['matched_items'], 'item_precision': precision, 'item_recall': recall,
        'item_f1': 2 * precision * recall / (precision + recall) if precision and recall else 0.0,
        'items_exact_match': rate(sum(q['items_exact'] for q in questions), len(questions)),
        'details_accuracy': rate(totals['details_correct'], totals['expected_items']),
        'nonnull_details_accuracy': rate(sum(r['details_correct'] for r in nonnull_details), len(nonnull_details)),
        'null_details_accuracy': rate(sum(r['details_correct'] for r in null_details), len(null_details)),
    }
    return {'evaluation_status': 'evaluated' if len(gold) == len(final_cases) else 'partially_evaluated',
            'reference': {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'policy': 'NFC/casefold/whitespace-normalized exact match; no inference of defaults or semantic equivalence; item_id ignored',
            'metrics': metrics, 'outputs': outputs, 'fields': fields, 'items': items, 'questions': questions,
            'per_field': {name: {'count': len([r for r in fields if r['field'] == name]),
                                  'accuracy': rate(sum(r['correct'] for r in fields if r['field'] == name),
                                                   len([r for r in fields if r['field'] == name]))}
                          for name in CONDITION_FIELDS}}
