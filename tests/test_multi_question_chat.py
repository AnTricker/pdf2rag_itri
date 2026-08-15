from __future__ import annotations

import unittest

from local_rag.adapters import OllamaChatBackend
from local_rag.models import QueryPlan, QueryPlanItem


class MultiQuestionChatTests(unittest.TestCase):
    def test_planner_preserves_all_questions_in_order(self):
        latest = '如何重設？等待多久？'
        plan = QueryPlan(items=[
            QueryPlanItem(question='如何重設？', route='document_question', queries=['重設']),
            QueryPlanItem(question='等待多久？', route='document_question', queries=['等待']),
        ])
        OllamaChatBackend._validate_query_plan(latest, plan)

    def test_planner_rejects_missing_document_queries(self):
        plan = QueryPlan(items=[
            QueryPlanItem(question='如何重設？', route='document_question', queries=[]),
        ])
        with self.assertRaises(ValueError):
            OllamaChatBackend._validate_query_plan('如何重設？', plan)


if __name__ == '__main__':
    unittest.main()
