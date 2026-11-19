import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import rag
from app.schemas.baggage import DecisionResponse, ItemDecision, BaggageResponse
from scripts import run_dataset


def decision(item_id, evidence_ids):
    return ItemDecision(item_id=item_id, carry_on='가능', checked_baggage='불가',
                        notes=None, evidence_ids=evidence_ids)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.groups = [
            {'item': {'item_id': 'item_1', 'name': '배터리', 'details': '100Wh'},
             'documents': [{'id': 'a', 'evidence_id': 'a', 'text': 'A', 'metadata': {'version': '1'}}]},
            {'item': {'item_id': 'item_2', 'name': '배터리', 'details': '160Wh'},
             'documents': [{'id': 'b', 'evidence_id': 'b', 'text': 'B', 'metadata': {'version': '2'}}]},
        ]

    def test_same_name_reordered_decisions_preserve_item_and_source(self):
        answer = rag.assemble_answer(self.groups, DecisionResponse(items=[decision('item_2', ['b']), decision('item_1', ['a'])]))
        self.assertEqual([x['source'][0]['chunk_id'] for x in answer['items']], ['a', 'b'])
        self.assertEqual(set(answer['items'][0]), {'name', 'carry_on', 'checked_baggage', 'notes', 'source'})
        BaggageResponse.model_validate(answer)

    def test_missing_duplicate_extra_and_cross_item_ids_are_rejected(self):
        for items in [[], [decision('item_1', ['a'])],
                      [decision('item_1', ['a']), decision('item_1', ['a'])],
                      [decision('other', []), decision('item_2', ['b'])],
                      [decision('item_1', ['b']), decision('item_2', ['b'])]]:
            with self.subTest(items=items), self.assertRaises(ValueError):
                rag.assemble_answer(self.groups, DecisionResponse(items=items))

    def test_uncited_decision_cannot_be_definitive(self):
        answer = rag.assemble_answer(self.groups[:1], DecisionResponse(items=[decision('item_1', [])]))
        self.assertEqual(answer['items'][0]['carry_on'], '확인 필요')
        self.assertEqual(answer['items'][0]['source'][0]['chunk_id'], 'a')
        searched, used = rag.source_traces(self.groups[:1], DecisionResponse(items=[decision('item_1', [])]))
        self.assertEqual(searched[0]['sources'], answer['items'][0]['source'])
        self.assertEqual(used[0]['sources'], [])

    def test_search_candidates_and_model_selection_are_separate(self):
        self.groups[0]['documents'].append({'id': 'other', 'evidence_id': 'other', 'text': 'extra', 'metadata': {}})
        response = DecisionResponse(items=[decision('item_2', ['b']), decision('item_1', ['a', 'a'])])
        answer = rag.assemble_answer(self.groups, response)
        searched, used = rag.source_traces(self.groups, response)
        self.assertEqual([source['chunk_id'] for source in answer['items'][0]['source']], ['a', 'other'])
        self.assertEqual(searched[0]['sources'], answer['items'][0]['source'])
        self.assertEqual([source['chunk_id'] for source in used[0]['sources']], ['a'])
        self.assertEqual([source['chunk_id'] for source in used[1]['sources']], ['b'])

    def test_no_items_still_structures_and_returns_json(self):
        with patch.object(rag, 'parse_query', return_value={'conditions': {}, 'items': []}) as parse, patch.object(rag, 'generate_structured_langchain') as llm:
            result = rag.answer_question('안녕하세요')
        parse.assert_called_once()
        llm.assert_not_called()
        self.assertEqual(result['answer'], {'items': []})
        self.assertTrue(result['message'])

    def test_explicit_conditions_are_in_semantic_query(self):
        with patch.object(rag, 'retrieve_for_item', return_value=[]) as search:
            groups = rag.search_by_items([{'item_id': 'item_1', 'name': '가위', 'details': '5cm'}], conditions={
                'departure': {'place': '인천', 'airline': '아시아나'}, 'arrival': {'place': '중국', 'airline': None}, 'baggage_type': '기내'})
        self.assertEqual(groups[0]['query'], '가위 5cm 인천 아시아나 중국 기내')
        self.assertIsNone(search.call_args.args[1])

    def test_short_evidence_reference_maps_back_to_original_chunk(self):
        original = {'id': 'very.long.original.chunk#15', 'text': 'rule', 'metadata': {}}
        parsed = {'conditions': {}, 'items': [{'name': 'battery', 'details': None}]}
        with patch.object(rag, 'parse_query', return_value=parsed), patch.object(rag, 'retrieve_for_item', return_value=[original]), patch.object(rag, 'generate_structured_langchain', return_value=DecisionResponse(items=[decision('item_1', ['item_1_doc_1'])])) as llm:
            result = rag.answer_question('battery')
        self.assertEqual(result['answer']['items'][0]['source'][0]['chunk_id'], original['id'])
        context = llm.call_args.args[1]
        self.assertIn('item_1_doc_1', context)
        self.assertNotIn(original['id'], context)

    def test_invalid_citation_keeps_model_and_retrieval_diagnostics(self):
        parsed = {'conditions': {}, 'items': [{'name': 'battery', 'details': None}]}
        original = {'id': 'a', 'text': 'rule', 'metadata': {}}
        with patch.object(rag, 'parse_query', return_value=parsed), patch.object(rag, 'retrieve_for_item', return_value=[original]), patch.object(rag, 'generate_structured_langchain', return_value=DecisionResponse(items=[decision('item_1', ['wrong'])])):
            with self.assertRaises(rag.AnswerValidationError) as error:
                rag.answer_question('battery')
        self.assertEqual(error.exception.diagnostics['model_decisions']['items'][0]['evidence_ids'], ['wrong'])
        self.assertEqual(error.exception.diagnostics['groups'][0]['documents'][0]['id'], 'a')

    def test_dataset_runner_never_passes_reference_and_keeps_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / 'dataset.json'
            dataset.write_text(json.dumps({'category': [
                {'id': '1', 'question': 'question one', 'answer': {'secret_reference': True}},
                {'id': '2', 'question': 'question two', 'answer': {}}]}), encoding='utf-8')
            with patch.object(run_dataset, 'answer_question', side_effect=[{'answer': {'items': []}, 'sources': [], 'groups': [], 'parsed': {}}, RuntimeError('failed')]) as ask:
                summary = run_dataset.run_dataset(dataset, root / 'output', 2)
            self.assertEqual(ask.call_args_list[0].args, ('question one',))
            self.assertEqual(ask.call_args_list[0].kwargs, {'k': 2})
            self.assertEqual((summary['success'], summary['error']), (1, 1))
            record_path = Path(summary['output_dir']) / 'records/2.json'
            record = json.loads(record_path.read_text(encoding='utf-8'))
            self.assertIsNone(record['answer'])
            self.assertEqual(summary['attempt'], 'attempt_1')
            self.assertEqual(run_dataset.create_attempt(root / 'output').name, 'attempt_2')
            self.assertEqual(json.loads(record_path.read_text(encoding='utf-8')), record)
            latest = json.loads((root / 'output/latest.json').read_text(encoding='utf-8'))
            self.assertEqual(latest['attempt'], 'attempt_1')


if __name__ == '__main__':
    unittest.main()
