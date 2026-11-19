import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation import merge_eval_inputs as merge


def case(qid, question, items=()):
    return {'id': qid, 'question': question, 'answer': {'items': [
        {'name': n, 'details': None, 'carry_on': '가능', 'checked_baggage': '가능', 'notes': 'n', 'source': s} for n, s in items]}}


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def make_run(folder, ids, status='success', **extra):
    (folder / 'records').mkdir(parents=True)
    for qid in ids:
        write(folder / 'records' / f'{qid}.json', {'id': qid, 'question': f'q{qid}', 'status': status})
    write(folder / 'summary.json', {'started_at': '2026-11-01T10:00:00', 'finished_at': '2026-11-01T11:00:00', 'k': 5,
                                    'success': len(ids), 'error': 0, 'results': [], **extra})


class MergeTests(unittest.TestCase):
    def test_datasets_are_concatenated_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = Path(d, 'a.json'), Path(d, 'b.json'), Path(d, 'c.json')
            write(a, {'1_기본': [case('1-1', 'q1-1', [('샴푸', ['x#1'])])]})
            write(b, {'1_기본': [case('1-2', 'q1-2')], '9_날짜': [case('9-1', 'q9-1')]})
            write(c, {'x': [case('1-1', 'again')]})
            merged = merge.merge_datasets([a, b])
            self.assertEqual([x['id'] for x in merged['1_기본']], ['1-1', '1-2'])
            self.assertEqual(list(merged), ['1_기본', '9_날짜'])
            with self.assertRaises(ValueError):
                merge.merge_datasets([a, c])

    def test_runs_are_merged_with_settings_difference_reported_and_originals_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_run(root / 'r1', ['1', '2'], retrieval='dense')
            make_run(root / 'r2', ['3'], status='error', retrieval='hybrid')
            questions = {'1': 'q1', '2': 'q2', '3': 'q3'}
            write(root / 'all_q.json', {'c': [case(k, v) for k, v in questions.items()]})
            summary = merge.merge_runs([root / 'r1', root / 'r2'], root / 'out', root / 'all_q.json', questions)
            self.assertEqual(sorted(p.name for p in (root / 'out' / 'records').glob('*.json')), ['1.json', '2.json', '3.json'])
            self.assertEqual((summary['total'], summary['success'], summary['error']), (3, 2, 1))
            self.assertEqual(summary['settings_differ'], ['retrieval'])
            self.assertEqual(summary['missing_questions'], [])
            self.assertEqual(len(summary['merged_from']), 2)
            self.assertEqual(sorted(p.name for p in (root / 'r1' / 'records').glob('*.json')), ['1.json', '2.json'])
            with self.assertRaises(ValueError):  # 이미 있는 출력은 덮어쓰지 않는다
                merge.merge_runs([root / 'r1'], root / 'out', root / 'all_q.json', questions)

    def test_record_with_different_question_or_unknown_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_run(root / 'r1', ['1'])
            write(root / 'q.json', {'c': []})
            with self.assertRaises(ValueError):
                merge.merge_runs([root / 'r1'], root / 'o1', root / 'q.json', {'1': 'different'})
            with self.assertRaises(ValueError):
                merge.merge_runs([root / 'r1'], root / 'o2', root / 'q.json', {'2': 'q2'})

    def test_main_reuses_identical_output_and_rejects_different(self):
        import sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / 'q.json', {'c': [case('1', 'q1', [('샴푸', ['x#1'])])]})
            write(root / 'p.json', {'c': [case('1', 'q1')]})
            argv = ['merge', '--questions', str(root / 'q.json'), '--queries', str(root / 'p.json'),
                    '--out-questions', str(root / 'oq.json'), '--out-queries', str(root / 'op.json')]
            with patch.object(sys, 'argv', argv):
                merge.main()
                merge.main()  # 같은 내용이면 다시 실행해도 된다
            write(root / 'oq.json', {'c': []})
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                merge.main()


if __name__ == '__main__':
    unittest.main()
