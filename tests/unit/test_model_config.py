import os
import json
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from app.integrations import embedding, openai_client
from app.settings import MODEL_CONFIG


class Result(BaseModel):
    value: str


class ModelConfigTests(unittest.TestCase):
    def selected_config(self, selector):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, '-B', '-c',
             'import json; from app.settings import MODEL_CONFIG; print(json.dumps(MODEL_CONFIG))'],
            cwd=root, env={**os.environ, 'CARRY_CHECK_MODEL_CONFIG': selector},
            capture_output=True, text=True,
        )
        return result

    def test_shared_config_can_be_selected_without_mac_overrides(self):
        result = self.selected_config('')
        self.assertEqual(result.returncode, 0, result.stderr)
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(json.loads(result.stdout), json.loads((root / 'config/model.json').read_text()))

    def test_mac_config_changes_only_embedding_device(self):
        result = self.selected_config('config/mac_model.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        self.assertEqual(actual['embedding']['parameters']['device'], 'mps')
        root = Path(__file__).resolve().parents[2]
        expected = json.loads((root / 'config/model.json').read_text())
        expected['embedding']['parameters']['device'] = 'mps'
        self.assertEqual(actual, expected)

    def test_missing_selected_config_does_not_silently_use_default(self):
        result = self.selected_config('config/not_existing_model_for_test.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileNotFoundError', result.stderr)

    def test_roles_use_separate_models_and_parameters(self):
        config = deepcopy(MODEL_CONFIG)
        config['query']['model'] = 'query-model'
        config['query']['parameters'] = {'temperature': 0.2}
        config['answer']['model'] = 'answer-model'
        config['answer']['parameters'] = {'temperature': 0.7}
        with patch.dict(os.environ, OPENAI_API_KEY='test', OPENAI_MODEL='ignored'), \
                patch.object(openai_client, 'MODEL_CONFIG', config), \
                patch.object(openai_client, 'ChatOpenAI') as client:
            client.return_value.with_structured_output.return_value.invoke.return_value = Result(value='ok')
            for role in ('query', 'answer'):
                openai_client.generate_structured_langchain('system', 'user', Result, role=role)
                self.assertEqual(client.call_args.kwargs['model'], config[role]['model'])
                self.assertEqual(client.call_args.kwargs['temperature'], config[role]['parameters']['temperature'])
                self.assertEqual(client.call_args.kwargs['timeout'], config[role]['timeout'])
                client.return_value.with_structured_output.assert_called_with(
                    Result, **config[role]['structured_output'])

    def test_existing_embedding_identity_is_preserved(self):
        config = {'embedding': {'provider': 'chroma_default', 'model': 'all-MiniLM-L6-v2',
                                'parameters': {}}}
        with patch.object(embedding, 'MODEL_CONFIG', config):
            function = embedding.create_embedding_function()
        self.assertEqual(function.name(), 'default')

    def test_alternative_embedding_receives_model_and_parameters(self):
        config = {'embedding': {'provider': 'sentence_transformer', 'model': 'other-model',
                                'parameters': {'device': 'cpu', 'normalize_embeddings': True}}}
        with patch.object(embedding, 'MODEL_CONFIG', config), \
                patch.object(embedding, 'SentenceTransformerEmbeddingFunction') as factory:
            embedding.create_embedding_function()
            factory.assert_called_once_with(model_name='other-model', device='cpu', normalize_embeddings=True)


if __name__ == '__main__':
    unittest.main()
