from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from local_rag.config import AppConfig
from local_rag.models import (
    CorpusProfile,
    GroundedAnswerBatch,
    GroundedAnswerItem,
    FinalAnswerBlock,
    QueryPlan,
    QueryPlanItem,
)
from local_rag.serve_web import create_app


class FakeEmbedding:
    dimension = 2

    def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeChatBackend:
    def __init__(self):
        self.plan_calls = 0
        self.answer_calls = 0

    def plan(self, latest_input, history, corpus_profile, trace=None):
        self.plan_calls += 1
        return QueryPlan(items=[QueryPlanItem(
            question=latest_input,
            route='document_question',
            queries=[latest_input],
            needs_visual_context=False,
        )])

    def answer_batch(self, items, image_paths, trace=None):
        self.answer_calls += 1
        return GroundedAnswerBatch(items=[GroundedAnswerItem(
            item_index=item['item_index'],
            blocks=[FinalAnswerBlock(
                text='請按住 RESET 三秒。',
                source_numbers=[1] if item['sources'] else [],
            )],
            attachment_names=item['attachment_names'],
        ) for item in items])


class MultiChatBackend(FakeChatBackend):
    def plan(self, latest_input, history, corpus_profile, trace=None):
        self.plan_calls += 1
        return QueryPlan(items=[
            QueryPlanItem(question='如何重設？', route='document_question', queries=['重設']),
            QueryPlanItem(question='等待多久？', route='document_question', queries=['等待時間']),
        ])


class FakeIndex:
    def __init__(self, root):
        source = SimpleNamespace(document_name='manual.pdf', page_start=4,
                                 page_end=4, artifact_path=None)
        record = SimpleNamespace(record_id='record-1', modality='text',
                                 content='按住 RESET 三秒。', source=source)
        self.hit = SimpleNamespace(record=record, score=0.9)
        self.records = [record]
        self.output_root = root
        self.corpus_profile = CorpusProfile(
            document_type='manual', summary='設備手冊',
            in_scope_topics=['重設'], document_id='doc-1',
            document_name='manual.pdf', profile_source='fixture',
            generation_model='fixture', prompt_version='fixture',
        )

    def search(self, vector, top_k):
        return [self.hit]


class HttpChatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        config = AppConfig.for_test(index_root=root / 'index')
        self.config = replace(
            config,
            base_dir=Path(__file__).resolve().parents[1],
            session_secret='test-secret-at-least-local',
            session_log_enabled=True,
            session_log_root=root / 'logs',
            log_root=root / 'runtime-logs',
            retrieval_min_score=0.5,
        )
        self.backend = FakeChatBackend()
        with patch('local_rag.serve_web.FileVectorIndex.load', return_value=FakeIndex(root)):
            self.app = create_app(
                config=self.config,
                embedding_backend=FakeEmbedding(),
                chat_backend=self.backend,
                build_root=root / 'build',
                output_root=root,
            )
        self.client = self.app.test_client()

    def tearDown(self):
        self.app.extensions['chat_job_manager'].shutdown()
        self.temp_dir.cleanup()

    def _wait(self, job_id):
        manager = self.app.extensions['chat_job_manager']
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = manager._jobs[job_id]
            if job.status in {'completed', 'failed', 'cancelled'}:
                return job
            time.sleep(0.01)
        self.fail('job did not finish')

    def test_single_question_uses_one_plan_and_one_answer_call(self):
        response = self.client.post('/api/chat/jobs', data={'question': '如何重設？'})
        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()['job_id']
        job = self._wait(job_id)
        self.assertEqual(job.status, 'completed')
        self.assertEqual((self.backend.plan_calls, self.backend.answer_calls), (1, 1))

        events = self.client.get(f'/api/chat/jobs/{job_id}/events')
        self.assertIn(b'event: planning', events.data)
        self.assertIn(b'event: completed', events.data)
        history = self.client.get('/api/history').get_json()['messages']
        self.assertEqual([item['role'] for item in history], ['user', 'assistant'])
        answer = history[1]['items'][0]
        self.assertEqual(answer['feedback'], 0)
        self.assertEqual(answer['actions'], [])
        self.assertEqual(len(answer['qa_id']), 32)
        self.assertTrue(list(self.config.session_log_root.glob('*.pretty.json')))

    def test_job_is_owned_by_signed_cookie_session(self):
        response = self.client.post('/api/chat/jobs', data={'question': '如何重設？'})
        job_id = response.get_json()['job_id']
        other = self.app.test_client()
        self.assertEqual(other.get(f'/api/chat/jobs/{job_id}/events').status_code, 404)

    def test_multi_question_uses_one_plan_and_one_answer_batch(self):
        backend = MultiChatBackend()
        manager = self.app.extensions['chat_job_manager']
        manager.engine.chat_backend = backend
        response = self.client.post('/api/chat/jobs', data={
            'question': '如何重設？等待多久？',
        })
        job = self._wait(response.get_json()['job_id'])
        self.assertEqual(job.status, 'completed')
        self.assertEqual((backend.plan_calls, backend.answer_calls), (1, 1))
        self.assertEqual(len(job.events[-1]['result']['items']), 2)

    def test_no_evidence_without_attachment_skips_answer(self):
        manager = self.app.extensions['chat_job_manager']
        manager.engine.index.search = lambda vector, top_k: []
        before = self.backend.answer_calls
        response = self.client.post('/api/chat/jobs', data={'question': '未知內容？'})
        job = self._wait(response.get_json()['job_id'])
        item = job.events[-1]['result']['items'][0]
        self.assertTrue(item['insufficient_context'])
        self.assertEqual(self.backend.answer_calls, before)

    def test_attachment_is_per_job_and_removed_after_completion(self):
        buffer = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(buffer, format='PNG')
        buffer.seek(0)
        response = self.client.post('/api/chat/jobs', data={
            'question': '附件內容是什麼？',
            'attachments': (buffer, 'sample.png'),
        }, content_type='multipart/form-data')
        job = self._wait(response.get_json()['job_id'])
        path = job.attachments[0].path
        self.assertEqual(job.status, 'completed')
        self.assertFalse(path.exists())
        completed = job.events[-1]['result']['items'][0]
        self.assertEqual(completed['citations'][-1]['source_type'], 'attachment')

    def test_invalid_attachment_is_rejected_before_queueing(self):
        response = self.client.post('/api/chat/jobs', data={
            'question': '附件內容？',
            'attachments': (io.BytesIO(b'not-an-image'), 'fake.png'),
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['error_code'], 'invalid_attachment')

    def test_suggestion_and_qa_actions_are_stored_in_qa_block(self):
        suggestion = 'CMP 機台開機前需要確認哪些項目？'
        selected = self.client.post('/api/chat/suggestions', json={'question': suggestion})
        self.assertEqual(selected.status_code, 204)
        response = self.client.post('/api/chat/jobs', data={'question': suggestion})
        self._wait(response.get_json()['job_id'])
        item = self.client.get('/api/history').get_json()['messages'][1]['items'][0]
        self.assertEqual(item['actions'][0]['action'], 'suggestion_selected')

        updated = self.client.post(
            f'/api/chat/qa/{item["qa_id"]}/actions',
            json={'action': 'feedback', 'feedback': 1},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.get_json()['feedback'], 1)
        copied = self.client.post(
            f'/api/chat/qa/{item["qa_id"]}/actions',
            json={'action': 'copy'},
        )
        self.assertEqual(copied.status_code, 200)
        history_item = self.client.get('/api/history').get_json()['messages'][1]['items'][0]
        self.assertEqual(history_item['feedback'], 1)
        self.assertEqual([entry['action'] for entry in history_item['actions']],
                         ['suggestion_selected', 'feedback', 'copy'])

        other = self.app.test_client()
        denied = other.post(
            f'/api/chat/qa/{item["qa_id"]}/actions',
            json={'action': 'feedback', 'feedback': -1},
        )
        self.assertEqual(denied.status_code, 404)

    def test_markdown_report_is_downloaded_without_disk_copy(self):
        empty = self.client.post('/api/history/report')
        self.assertEqual(empty.status_code, 409)
        response = self.client.post('/api/chat/jobs', data={'question': '如何重設？'})
        self._wait(response.get_json()['job_id'])
        report = self.client.post('/api/history/report')
        self.assertEqual(report.status_code, 200)
        self.assertIn('attachment; filename="cmp_chat_', report.headers['Content-Disposition'])
        text = report.data.decode('utf-8')
        self.assertIn('report_version: 1', text)
        self.assertIn('## 對話 1', text)
        self.assertIn('manual.pdf，第 4 頁', text)
        self.assertFalse(list(self.config.session_log_root.glob('*.md')))

    def test_end_session_cancels_jobs_rotates_cookie_and_finishes_log(self):
        self.client.get('/')
        before = self.client.get_cookie('local_rag_session').value
        ended = self.client.post('/api/session/end')
        self.assertEqual(ended.status_code, 204)
        after = self.client.get_cookie('local_rag_session').value
        self.assertNotEqual(before, after)
        pretty = json.loads(next(self.config.session_log_root.glob('*.pretty.json')).read_text(encoding='utf-8'))
        self.assertIn('qa_blocks', pretty)
        self.assertEqual(pretty['session_actions'][-1]['action'], 'end_chat')
        self.assertEqual(pretty['diagnostics'][-1]['event'], 'session_end')


if __name__ == '__main__':
    unittest.main()
