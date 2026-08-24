from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import requests

from local_rag.adapters import OllamaChatBackend
from local_rag.config import AppConfig
from local_rag.models import CorpusProfile


class Response:
    def __init__(self, content=''):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {
            'response': self.content,
            'total_duration': 10,
            'load_duration': 2,
            'prompt_eval_count': 20,
            'prompt_eval_duration': 3,
            'eval_count': 5,
            'eval_duration': 4,
        }


class PsResponse(Response):
    def __init__(self, models):
        self.models = models

    def json(self):
        return {'models': self.models}


class OllamaServeTests(unittest.TestCase):
    def test_warm_two_stage_qa_and_unload_payloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                AppConfig.for_test(index_root=Path(temp_dir) / 'index'),
                llm_model='gemma-fixture', llm_context_tokens=32768,
            )
            backend = OllamaChatBackend(
                config,
                splitter_prompt='Profile={corpus_profile} History={history} Input={latest_input}',
                admission_prompt='unused', query_builder_prompt='unused',
                evidence_draft_prompt='unused', final_answer_prompt='Items={items}',
            )
            profile = CorpusProfile(
                document_type='manual', summary='summary', in_scope_topics=['reset'],
                document_id='doc', document_name='manual.pdf', profile_source='fixture',
                generation_model='fixture', prompt_version='fixture',
            )
            responses = [
                Response(),
                Response(json.dumps({'items': [{
                    'question': '如何重設？', 'route': 'document_question',
                    'queries': ['重設'], 'needs_visual_context': False,
                }]})),
                Response(json.dumps({'items': [{
                    'item_index': 0,
                    'blocks': [{'text': '按住 RESET。', 'source_numbers': [1]}],
                    'attachment_names': [],
                }]})),
                Response(),
            ]
            with patch('local_rag.adapters.requests.get',
                       return_value=PsResponse([{'name': 'gemma-fixture:latest'}])) as get:
                self.assertEqual(backend.model_status(), (True, True))
            with patch('local_rag.adapters.requests.post', side_effect=responses) as post:
                backend.warm()
                backend.plan('如何重設？', [], profile)
                backend.answer_batch([{
                    'item_index': 0, 'question': '如何重設？',
                    'sources': [{'source_number': 1, 'content': '按住 RESET。'}],
                    'attachment_names': [],
                }], [], None)
                backend.unload()

            self.assertEqual(post.call_count, 4)
            self.assertEqual(get.call_count, 1)
            payloads = [call.kwargs['json'] for call in post.call_args_list]
            self.assertEqual(payloads[0]['keep_alive'], '15m')
            self.assertEqual(post.call_args_list[0].kwargs['timeout'], 300)
            self.assertEqual(payloads[0]['options']['num_ctx'], 32768)
            self.assertEqual(payloads[1]['keep_alive'], '15m')
            self.assertEqual(payloads[2]['keep_alive'], '15m')
            self.assertFalse(payloads[1]['think'])
            self.assertFalse(payloads[2]['think'])
            self.assertNotIn('num_predict', payloads[2]['options'])
            self.assertEqual(payloads[3]['keep_alive'], 0)

            with patch('local_rag.adapters.requests.get',
                       return_value=PsResponse([])):
                self.assertEqual(backend.model_status(), (True, False))
            with patch('local_rag.adapters.requests.get',
                       side_effect=requests.ConnectionError):
                self.assertEqual(backend.model_status(), (False, False))

    def test_answer_retries_validation_error_but_not_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                AppConfig.for_test(index_root=Path(temp_dir) / 'index'),
                llm_model='gemma-fixture',
            )
            backend = OllamaChatBackend(
                config, splitter_prompt='{latest_input}{history}{corpus_profile}',
                admission_prompt='unused', query_builder_prompt='unused',
                evidence_draft_prompt='unused', final_answer_prompt='{items}',
            )
            invalid = Response(json.dumps({'items': [{
                'item_index': 0,
                'blocks': [{'text': '錯誤引用', 'source_numbers': [2]}],
            }]}))
            valid = Response(json.dumps({'items': [{
                'item_index': 0,
                'blocks': [{'text': '正確引用', 'source_numbers': [1]}],
            }]}))
            items = [{'item_index': 0, 'question': 'Q',
                      'sources': [{'source_number': 1, 'content': 'A'}],
                      'attachment_names': []}]
            with patch('local_rag.adapters.requests.post', side_effect=[invalid, valid]) as post:
                result = backend.answer_batch(items, [], None)
            self.assertEqual(result.items[0].blocks[0].text, '正確引用')
            self.assertEqual(post.call_count, 2)

            with patch('local_rag.adapters.requests.post', side_effect=requests.Timeout) as post:
                with self.assertRaises(requests.Timeout):
                    backend.answer_batch(items, [], None)
            self.assertEqual(post.call_count, 1)


if __name__ == '__main__':
    unittest.main()
