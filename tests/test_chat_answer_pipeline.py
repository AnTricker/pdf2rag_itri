from __future__ import annotations

import unittest
from types import SimpleNamespace

from local_rag.serve_chat import ServeChatEngine


class ChatAnswerPipelineTests(unittest.TestCase):
    def test_answer_batches_preserve_items_and_use_minimum_batches(self):
        engine = object.__new__(ServeChatEngine)
        engine.config = SimpleNamespace(answer_input_budget_tokens=1400)
        items = [
            {'item_index': 0, 'question': 'A', 'sources': [], 'attachment_names': []},
            {'item_index': 1, 'question': 'B', 'sources': [], 'attachment_names': []},
        ]
        batches = engine._batches(items)
        self.assertEqual([[item['item_index'] for item in batch] for batch in batches], [[0, 1]])

    def test_answer_batches_split_without_truncating_oversized_items(self):
        engine = object.__new__(ServeChatEngine)
        engine.config = SimpleNamespace(answer_input_budget_tokens=1400)
        items = [
            {'item_index': 0, 'question': 'A' * 300, 'sources': [], 'attachment_names': []},
            {'item_index': 1, 'question': 'B' * 300, 'sources': [], 'attachment_names': []},
        ]
        batches = engine._batches(items)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0]['question'], 'A' * 300)
        self.assertEqual(batches[1][0]['question'], 'B' * 300)


if __name__ == '__main__':
    unittest.main()
