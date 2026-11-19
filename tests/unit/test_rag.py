import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from app import rag, query_parser, retrieval
from app.integrations import openai_client
from app.schemas.query import ParsedQuery
from app.schemas.baggage import ItemDecision, DecisionResponse
from pydantic import ValidationError
from langchain_core.messages import AIMessage


class RagTests(unittest.TestCase):
    def setUp(self):
        factory = patch.object(retrieval, 'create_embedding_function')
        factory.start()
        self.addCleanup(factory.stop)

    def test_filters_are_passed_to_semantic_query(self):
        collection = MagicMock()
        collection.count.return_value = 10
        collection.query.return_value = {'ids': [['a']], 'documents': [['根拠']],
                                         'metadatas': [[{'country': 'KR'}]]}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / retrieval.CHROMA_PATH).mkdir(parents=True)
            with patch.object(retrieval, 'ROOT', Path(folder)), patch.object(retrieval.chromadb, 'PersistentClient') as client:
                client.return_value.get_or_create_collection.return_value = collection
                result = rag.retrieve('배터리', {'country': 'KR', 'source_name': '고시'}, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(collection.query.call_args.kwargs['query_texts'], ['배터리'])
        self.assertEqual(collection.query.call_args.kwargs['where'], {'$and': [
            {'country': {'$eq': 'KR'}}, {'source_name': {'$eq': '고시'}}]})

    def test_empty_collection_calls_store_before_search(self):
        collection = MagicMock()
        collection.count.side_effect = [0, 2, 2]
        collection.query.return_value = {'ids': [['a']], 'documents': [['규정']], 'metadatas': [[{'country': 'KR'}]]}
        rows = [{'id': 'a', 'text': '규정', 'metadata': {'country': 'KR'}}]
        with patch.object(retrieval.chromadb, 'PersistentClient') as client, patch.object(retrieval, 'load_chunks', return_value=rows), patch.object(retrieval, 'store_chunks') as store:
            client.return_value.get_or_create_collection.return_value = collection
            result = retrieval.retrieve('배터리')
        store.assert_called_once_with(rows, collection=collection)
        self.assertEqual(result[0]['id'], 'a')

    def test_existing_collection_is_not_reloaded(self):
        collection = MagicMock()
        collection.count.return_value = 2
        collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]]}
        with patch.object(retrieval.chromadb, 'PersistentClient') as client, patch.object(retrieval, 'load_chunks') as load, patch.object(retrieval, 'store_chunks') as store:
            client.return_value.get_or_create_collection.return_value = collection
            retrieval.retrieve('배터리')
        load.assert_not_called()
        store.assert_not_called()

    def test_missing_metadata_field_is_not_assumed(self):
        with self.assertRaises(ValueError):
            rag.retrieve('가위', {'airline': '대한항공'})

    def test_prompts_are_read_from_files(self):
        with tempfile.TemporaryDirectory() as folder:
            prompts = Path(folder) / 'prompts'
            prompts.mkdir()
            (prompts / 'system.txt').write_text('근거만 사용', encoding='utf-8')
            (prompts / 'user_items.txt').write_text('$question\n$conditions\n$filters\n$context', encoding='utf-8')
            with patch.object(query_parser, 'PROMPTS_DIR', prompts):
                system, user = rag.build_items_prompt('질문', {'airline': '항공사'}, {'country': 'KR'},
                    [{'item': {'name': '배터리'}, 'documents': [{'evidence_id': 'item_1_doc_1', 'text': '자료 $value', 'metadata': {}}]}])
        self.assertEqual(system, '근거만 사용')
        self.assertIn('documents', user)
        self.assertIn('자료 $value', user)

    def test_empty_retrieval_does_not_call_llm(self):
        with patch.object(rag, 'parse_query', return_value={'conditions': {}, 'items': [{'name': '배터리'}]}), patch.object(rag, 'retrieve_for_item', return_value=[]), patch.object(rag, 'generate_structured_langchain') as generate:
            result = rag.answer_question('질문')
        generate.assert_not_called()
        self.assertEqual(result['answer']['items'][0]['carry_on'], '확인 필요')
        self.assertEqual(result['answer']['items'][0]['source'], [])

    def test_complete_workflow_returns_answer_and_sources(self):
        documents = [{'id': 'a', 'text': '규정', 'metadata': {'country': 'KR'}}]
        with patch.object(rag, 'parse_query', return_value={'conditions': {}, 'items': [{'name': '배터리'}]}), patch.object(rag, 'retrieve_for_item', return_value=documents), patch.object(rag, 'build_items_prompt', return_value=('system', 'user')), patch.object(rag, 'generate_structured_langchain', return_value=DecisionResponse(items=[ItemDecision(item_id='item_1', carry_on='확인 필요', checked_baggage='불가', notes=None, evidence_ids=['item_1_doc_1'])])):
            result = rag.answer_question('질문', {'travel_date': '2026-10-23'}, {'country': 'KR'})
        self.assertEqual(result['answer']['items'][0]['checked_baggage'], '불가')
        self.assertEqual(result['answer']['items'][0]['source'], [{'country': 'KR', 'chunk_id': 'a'}])
        self.assertEqual(result['sources'][0]['id'], documents[0]['id'])
        self.assertEqual(result['parsed']['items'], [{'name': '배터리', 'item_id': 'item_1'}])

    def test_query_extraction_and_explicit_conditions(self):
        raw = {'conditions': {'departure': {'place': '한국', 'airline': '항공사'}, 'arrival': {'place': None, 'airline': None}, 'travel_date': None, 'baggage_type': None},
               'items': [{'name': '배터리', 'details': '2개'}, {'name': '배터리', 'details': '2개'}, {'name': '가위', 'details': None}]}
        with patch.object(query_parser, 'generate_structured_langchain', return_value=ParsedQuery.model_validate(raw)):
            result = query_parser.parse_query('배터리와 가위', {'departure': '인천', 'airline': '직접 입력'})
        self.assertEqual(result['conditions']['departure'], {'place': '인천', 'airline': '직접 입력'})
        self.assertEqual(result['items'], raw['items'])

    def test_structured_query_passes_schema_to_model(self):
        raw = {'conditions': {'departure': {'place': None, 'airline': None}, 'arrival': {'place': None, 'airline': None}, 'travel_date': None, 'baggage_type': '기내'}, 'items': [{'name': '보조배터리', 'details': None}]}
        expected = ParsedQuery.model_validate(raw)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'OPENAI_MODEL': 'gpt-4o-mini'}), patch.object(openai_client, 'ChatOpenAI') as chat:
            structured = chat.return_value.with_structured_output.return_value
            structured.invoke.return_value = expected
            self.assertIs(openai_client.generate_structured_langchain('규칙', '질문', ParsedQuery), expected)
        chat.return_value.with_structured_output.assert_called_once_with(ParsedQuery, method='json_schema', strict=True)
        structured.invoke.assert_called_once_with([('system', '규칙'), ('human', '질문')], store=False)

    def test_structured_call_accepts_another_schema(self):
        from pydantic import BaseModel

        class Summary(BaseModel):
            text: str

        expected = Summary(text='요약')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'OPENAI_MODEL': 'gpt-4o-mini'}), patch.object(openai_client, 'ChatOpenAI') as chat:
            chat.return_value.with_structured_output.return_value.invoke.return_value = expected
            self.assertIs(openai_client.generate_structured_langchain('규칙', '질문', Summary), expected)
        chat.return_value.with_structured_output.assert_called_once_with(Summary, method='json_schema', strict=True)

    def test_query_schema_rejects_invalid_shape_and_values(self):
        valid = {'conditions': {'departure': {'place': None, 'airline': None}, 'arrival': {'place': None, 'airline': None}, 'travel_date': None, 'baggage_type': None}, 'items': []}
        self.assertEqual(ParsedQuery.model_validate(valid).model_dump(), valid)
        for conditions in [dict(valid['conditions'], baggage_type='캐리어'), dict(valid['conditions'], route='국제선'), {'departure': {'place': None, 'airline': None}}]:
            with self.assertRaises(ValidationError):
                ParsedQuery.model_validate(dict(valid, conditions=conditions))
        with self.assertRaises(ValidationError):
            ParsedQuery.model_validate(dict(valid, items=[{'name': 123, 'details': None}]))

    def test_no_items_skips_search_and_final_generation(self):
        with patch.object(rag, 'parse_query', return_value={'conditions': {}, 'items': []}), patch.object(rag, 'retrieve_for_item') as search, patch.object(rag, 'generate_structured_langchain') as generate:
            result = rag.answer_question('안녕하세요')
        search.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(result['groups'], [])

    def test_item_search_and_missing_evidence_stay_separate(self):
        parsed = {'conditions': {}, 'items': [{'name': '배터리', 'details': '2개'}, {'name': '가위'}]}
        docs = [{'id': 'a', 'text': '배터리 규정', 'metadata': {}}]
        with patch.object(rag, 'parse_query', return_value=parsed), patch.object(rag, 'retrieve_for_item', side_effect=[docs, []]) as search, patch.object(rag, 'generate_structured_langchain', return_value=DecisionResponse(items=[ItemDecision(item_id='item_1', carry_on='조건부 가능', checked_baggage='불가', notes=None, evidence_ids=['item_1_doc_1'])])) as generate:
            result = rag.answer_question('배터리와 가위', filters={'country': 'KR'}, k=2)
        self.assertEqual(search.call_args_list[0].args, ('배터리 2개', {'country': 'KR'}, 2))
        self.assertEqual(search.call_args_list[1].args, ('가위', {'country': 'KR'}, 2))
        self.assertEqual(result['groups'][1]['documents'], [])
        self.assertEqual(result['answer']['items'][1]['source'], [])
        self.assertIs(generate.call_args.args[2], DecisionResponse)
        self.assertNotIn('item_2', generate.call_args.args[1])

    def test_invalid_options_fail_before_query_model(self):
        with patch.object(rag, 'parse_query') as parse:
            with self.assertRaises(ValueError):
                rag.answer_question('질문', k=0)
            with self.assertRaises(ValueError):
                rag.answer_question('질문', filters={'airline': '항공사'})
        parse.assert_not_called()

    def test_langchain_receives_messages_and_returns_text(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'OPENAI_MODEL': 'gpt-4o-mini'}), patch.object(openai_client, 'ChatOpenAI') as chat:
            chat.return_value.invoke.return_value = AIMessage(content=' 답변 ', response_metadata={'finish_reason': 'stop'})
            self.assertEqual(openai_client.generate_answer_langchain('규칙', '질문'), '답변')
        chat.assert_called_once_with(api_key='test-key', model='gpt-4o-mini', temperature=0, timeout=120, max_retries=0)
        chat.return_value.invoke.assert_called_once_with([('system', '규칙'), ('human', '질문')], store=False)

    def test_langchain_handles_blocks_and_rejects_empty_or_partial_text(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'OPENAI_MODEL': 'gpt-4o-mini'}), patch.object(openai_client, 'ChatOpenAI') as chat:
            chat.return_value.invoke.return_value = AIMessage(content=[{'type': 'text', 'text': '답변'}], response_metadata={})
            self.assertEqual(openai_client.generate_answer_langchain('규칙', '질문'), '답변')
            for content, metadata in [('', {}), ('부분 답변', {'finish_reason': 'length'})]:
                chat.return_value.invoke.return_value = AIMessage(content=content, response_metadata=metadata)
                with self.assertRaises(RuntimeError):
                    openai_client.generate_answer_langchain('규칙', '질문')

    def test_openai_sdk_receives_prompts_and_returns_text(self):
        config = {'answer': {'model': 'configured-model', 'timeout': 120,
                             'max_retries': 0, 'parameters': {'temperature': 0.3}}}
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only-key', 'OPENAI_MODEL': 'ignored'}), patch.object(openai_client, 'MODEL_CONFIG', config), patch.object(openai_client, 'OpenAI') as sdk:
            llm = sdk.return_value.__enter__.return_value
            llm.responses.create.return_value = MagicMock(status='completed', output_text='  최종 답변 [1]  ')
            self.assertEqual(openai_client.generate_answer('instructions', 'context'), '최종 답변 [1]')
        llm.responses.create.assert_called_once_with(model='configured-model', instructions='instructions', input='context', store=False, temperature=0.3)
        sdk.return_value.__exit__.assert_called_once()

    def test_openai_sdk_rejects_incomplete_or_empty_answers(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only-key', 'OPENAI_MODEL': 'configured-model'}), patch.object(openai_client, 'OpenAI') as sdk:
            llm = sdk.return_value.__enter__.return_value
            for status, text in [('incomplete', 'partial'), ('completed', '  ')]:
                llm.responses.create.return_value = MagicMock(status=status, output_text=text)
                with self.assertRaises(RuntimeError):
                    openai_client.generate_answer('rules', 'question')


if __name__ == '__main__':
    unittest.main()
