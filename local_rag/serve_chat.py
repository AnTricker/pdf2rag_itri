from __future__ import annotations

import json
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from .config import AppConfig
from .models import AttachmentMetadata, QAHistoryPair, QueryPlan, QueryPlanItem
from .rendering import SafeMarkdownRenderer
from .session_log import SessionJsonlLogger


class JobCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class AttachmentRef:
    metadata: AttachmentMetadata
    path: Path


@dataclass
class Session:
    messages: list[dict[str, object]] = field(default_factory=list)
    qa_history: list[QAHistoryPair] = field(default_factory=list)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pending_suggestions: list[dict[str, str]] = field(default_factory=list)
    touched_at: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(self, ttl_seconds: int, on_expire=None) -> None:
        self.ttl_seconds = ttl_seconds
        self.on_expire = on_expire
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> Session:
        with self._lock:
            self._expire_locked()
            session = self._sessions.setdefault(session_id, Session())
            session.touched_at = time.monotonic()
            return session

    def history(self, session_id: str) -> list[dict[str, object]]:
        with self._lock:
            return deepcopy(self.get(session_id).messages)

    def snapshot(self, session_id: str) -> dict[str, object]:
        with self._lock:
            session = self.get(session_id)
            return {
                'created_at_utc': session.created_at_utc,
                'messages': deepcopy(session.messages),
            }

    def set_pending_suggestion(self, session_id: str, question: str) -> None:
        with self._lock:
            session = self.get(session_id)
            session.pending_suggestions.append({
                'action': 'suggestion_selected',
                'question': question,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            })

    def consume_pending_suggestions(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            session = self.get(session_id)
            actions = session.pending_suggestions
            session.pending_suggestions = []
            return deepcopy(actions)

    def update_qa_action(
        self, session_id: str, qa_id: str, action: str, feedback: Optional[int] = None
    ) -> Optional[dict[str, object]]:
        with self._lock:
            session = self.get(session_id)
            for message in session.messages:
                for item in message.get('items', []):
                    if item.get('qa_id') != qa_id:
                        continue
                    event: dict[str, object] = {
                        'action': action,
                        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    }
                    if action == 'feedback':
                        item['feedback'] = feedback
                        event['feedback'] = feedback
                    item.setdefault('actions', []).append(event)
                    return deepcopy(item)
            return None

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _expire_locked(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        expired = [key for key, value in self._sessions.items() if value.touched_at < cutoff]
        for key in expired:
            self._sessions.pop(key, None)
            if self.on_expire is not None:
                self.on_expire(key)


class ServeChatEngine:
    out_of_scope_answer = '本系統僅回答目前文件或當次附件相關問題。'
    security_answer = '無法提供、推測或摘要 system prompt、hidden instructions 或內部設定。'
    insufficient_answer = '文件與附件未提供足夠資訊。'
    _security_pattern = re.compile(
        r'(system\s*prompt|hidden\s*instruction|內部提示|系統提示|繞過.*規則)',
        re.IGNORECASE,
    )

    def __init__(self, *, config: AppConfig, index, embedding_backend, chat_backend,
                 sessions: SessionStore, session_logger: SessionJsonlLogger,
                 renderer: SafeMarkdownRenderer) -> None:
        self.config = config
        self.index = index
        self.embedding_backend = embedding_backend
        self.chat_backend = chat_backend
        self.sessions = sessions
        self.session_logger = session_logger
        self.renderer = renderer

    def ask(self, session_id: str, latest_input: str, attachments: list[AttachmentRef],
            *, progress: Callable[..., None], cancelled: Callable[[], bool]) -> dict[str, object]:
        started = time.monotonic()
        session = self.sessions.get(session_id)
        history = [pair.model_dump() for pair in session.qa_history]
        attachment_data = [item.metadata.model_dump() for item in attachments]
        self.session_logger.append(session_id, 'request_received', latest_input=latest_input,
                                   attachments=attachment_data)
        self._check_cancelled(cancelled)
        progress('planning', message='正在規劃問題')
        plan = self._plan(session_id, latest_input, history)
        results: dict[int, dict[str, object]] = {}
        answer_inputs: list[dict[str, object]] = []
        evidence: dict[int, list] = {}
        image_paths: dict[int, list[Path]] = {}

        progress('retrieval', message='正在檢索文件')
        for index, item in enumerate(plan.items):
            self._check_cancelled(cancelled)
            if item.route == 'security_request':
                results[index] = self._fixed_item(item.question, item.route, self.security_answer)
                continue
            if item.route == 'out_of_scope':
                results[index] = self._fixed_item(item.question, item.route, self.out_of_scope_answer)
                continue
            hits = self._retrieve(item)
            if not hits and not attachments:
                results[index] = self._insufficient_item(item.question)
                continue
            records = self._record_blocks(hits)
            evidence[index] = hits
            answer_inputs.append({
                'item_index': index,
                'question': item.question,
                'sources': records,
                'attachment_names': [entry.metadata.name for entry in attachments],
            })
            paths = [entry.path for entry in attachments]
            if item.needs_visual_context:
                paths.extend(self._crop_paths(hits))
            image_paths[index] = self._deduplicated_paths(paths)

        batches = self._batches(answer_inputs)
        if any(image_paths.values()):
            progress('vision', message='正在分析圖片')
        for ordinal, batch in enumerate(batches, start=1):
            self._check_cancelled(cancelled)
            progress('answering', message=f'正在產生答案（{ordinal}/{len(batches)}）')
            batch_paths = self._deduplicated_paths(
                [path for item in batch for path in image_paths[item['item_index']]]
            )
            self._apply_answer_batch(session_id, batch, batch_paths, evidence, results)

        ordered = [results[index] for index in range(len(plan.items))]
        self._check_cancelled(cancelled)
        for item in ordered:
            item.update(qa_id=uuid4().hex, feedback=0, actions=[])
        pending_suggestions = self.sessions.consume_pending_suggestions(session_id)
        if ordered:
            ordered[0]['actions'].extend(pending_suggestions)
        session.messages.extend([
            {'role': 'user', 'content': latest_input, 'attachments': attachment_data},
            {'role': 'assistant', 'content': self._history_text(ordered), 'items': ordered},
        ])
        session.qa_history.extend(
            QAHistoryPair(question=item['question'], answer=item['answer'] or item.get('error_code', ''))
            for item in ordered
        )
        self._trim_history(session)
        response: dict[str, object] = {'items': ordered, 'warnings': []}
        if len(ordered) == 1:
            response.update(answer=ordered[0]['answer'], citations=ordered[0]['citations'],
                            insufficient_context=ordered[0]['insufficient_context'])
        self.session_logger.append(session_id, 'response_completed', status='success',
                                   duration_ms=round((time.monotonic() - started) * 1000, 3),
                                   output=response)
        return response

    def _plan(self, session_id: str, latest_input: str, history) -> QueryPlan:
        if self._security_pattern.search(latest_input):
            return QueryPlan(items=[QueryPlanItem(
                question=latest_input, route='security_request', queries=[]
            )])
        try:
            return self.chat_backend.plan(
                latest_input, history, self.index.corpus_profile,
                trace=self._trace(session_id),
            )
        except (KeyError, ValueError) as error:
            self.session_logger.append(session_id, 'planner_fallback', reason=str(error))
            return QueryPlan(items=[QueryPlanItem(
                question=latest_input, route='document_question',
                queries=[latest_input], needs_visual_context=False,
            )])

    def _retrieve(self, item: QueryPlanItem) -> list:
        queries = self._clean_texts([item.question, *item.queries])[:3]
        if not queries:
            queries = [item.question]
        vectors = self.embedding_backend.embed(queries)
        per_query_hits = []
        for vector in vectors:
            hits = self.index.search(vector, top_k=self.config.retrieval_top_k)
            per_query_hits.append([
                hit for hit in hits if hit.score >= self.config.retrieval_min_score
            ])
        return self._round_robin_hits(per_query_hits, self.config.retrieval_context_max_records)

    def _apply_answer_batch(self, session_id, batch, paths, evidence, results) -> None:
        try:
            answer = self.chat_backend.answer_batch(
                batch, paths, trace=self._trace(session_id)
            )
            expected = {item['item_index'] for item in batch}
            actual = {item.item_index for item in answer.items}
            if actual != expected:
                raise ValueError('answer batch item indexes do not match request')
            for item in answer.items:
                hits = evidence[item.item_index]
                valid_numbers = []
                for block in item.blocks:
                    for number in block.source_numbers:
                        if not 1 <= number <= len(hits):
                            raise ValueError('answer references an unknown source number')
                        if number not in valid_numbers:
                            valid_numbers.append(number)
                used_hits = [hits[number - 1] for number in valid_numbers]
                text = '\n\n'.join(block.text.strip() for block in item.blocks)
                attachment_names = list(dict.fromkeys(item.attachment_names))
                citations = self._citations(used_hits)
                citations.extend(
                    {'source_type': 'attachment', 'name': name}
                    for name in attachment_names
                )
                results[item.item_index] = {
                    'question': batch[[entry['item_index'] for entry in batch].index(item.item_index)]['question'],
                    'route': 'document_question',
                    'answer': text,
                    'answer_html': self.renderer.render(text),
                    'citations': citations,
                    'insufficient_context': False,
                }
        except (KeyError, ValueError) as error:
            for item in batch:
                results[item['item_index']] = self._error_item(
                    item['question'], 'answer_processing_failed'
                )
            self.session_logger.append(session_id, 'answer_processing_failed', reason=str(error))

    def _batches(self, items: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        budget = max(1, self.config.answer_input_budget_tokens - 1000)
        batches: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        current_size = 0
        for item in items:
            size = len(json.dumps(item, ensure_ascii=False))
            if current and current_size + size > budget:
                batches.append(current)
                current = []
                current_size = 0
            current.append(item)
            current_size += size
        if current:
            batches.append(current)
        return batches

    def _crop_paths(self, hits) -> list[Path]:
        paths = []
        for hit in hits:
            artifact = hit.record.source.artifact_path
            if not artifact:
                continue
            candidate = (self.index.output_root / artifact).resolve()
            crops_root = (self.index.output_root / 'crops').resolve()
            if crops_root in candidate.parents and candidate.is_file():
                paths.append(candidate)
        return paths

    @staticmethod
    def _record_blocks(hits) -> list[dict[str, object]]:
        return [
            {
                'source_number': index,
                'modality': hit.record.modality,
                'document_name': hit.record.source.document_name,
                'page_start': hit.record.source.page_start,
                'page_end': hit.record.source.page_end,
                'content': hit.record.content,
                'artifact_name': (
                    Path(hit.record.source.artifact_path).name
                    if hit.record.source.artifact_path else None
                ),
            }
            for index, hit in enumerate(hits, start=1)
        ]

    @staticmethod
    def _round_robin_hits(per_query_hits, limit: int) -> list:
        merged = []
        seen = set()
        rank = 0
        while len(merged) < limit and any(rank < len(hits) for hits in per_query_hits):
            for hits in per_query_hits:
                if rank >= len(hits):
                    continue
                hit = hits[rank]
                if hit.record.record_id not in seen:
                    seen.add(hit.record.record_id)
                    merged.append(hit)
                    if len(merged) == limit:
                        break
            rank += 1
        return merged

    @staticmethod
    def _citations(hits) -> list[dict[str, object]]:
        citations = []
        seen = set()
        for hit in hits:
            artifact = hit.record.source.artifact_path
            identity = artifact or hit.record.record_id
            if identity in seen:
                continue
            seen.add(identity)
            citations.append({
                'source_type': 'document',
                'record_id': hit.record.record_id,
                'modality': hit.record.modality,
                'document_name': hit.record.source.document_name,
                'page_start': hit.record.source.page_start,
                'page_end': hit.record.source.page_end,
                'crop_url': f'/api/crops/{hit.record.record_id}' if artifact else None,
                'score': hit.score,
            })
        return citations

    def _trace(self, session_id: str):
        def append(event: str, **data) -> None:
            self.session_logger.append(session_id, event, **data)
        return append

    def _trim_history(self, session: Session) -> None:
        maximum_messages = self.config.history_max_turns * 2
        session.messages[:] = session.messages[-maximum_messages:]
        session.qa_history[:] = session.qa_history[-self.config.history_max_turns:]
        while len(session.qa_history) > 1 and sum(
            len(pair.question) + len(pair.answer) for pair in session.qa_history
        ) > self.config.history_max_tokens:
            del session.qa_history[0]

    @staticmethod
    def _clean_texts(values) -> list[str]:
        output = []
        for value in values:
            cleaned = ' '.join(str(value).split())
            if cleaned and cleaned not in output:
                output.append(cleaned)
        return output

    @staticmethod
    def _deduplicated_paths(paths: list[Path]) -> list[Path]:
        output = []
        seen = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                output.append(resolved)
        return output

    @classmethod
    def _fixed_item(cls, question: str, route: str, answer: str) -> dict[str, object]:
        return {'question': question, 'route': route, 'answer': answer,
                'answer_html': None, 'citations': [], 'insufficient_context': False}

    @classmethod
    def _insufficient_item(cls, question: str) -> dict[str, object]:
        return {'question': question, 'route': 'document_question',
                'answer': cls.insufficient_answer, 'answer_html': None,
                'citations': [], 'insufficient_context': True}

    @staticmethod
    def _error_item(question: str, code: str) -> dict[str, object]:
        return {'question': question, 'route': 'document_question', 'answer': '',
                'answer_html': None, 'citations': [], 'insufficient_context': True,
                'error_code': code}

    @staticmethod
    def _history_text(items) -> str:
        return '\n\n'.join(
            'Q: {question}\nA: {answer}'.format(
                question=item['question'],
                answer=item['answer'] or item.get('error_code', ''),
            )
            for item in items
        )

    @staticmethod
    def _check_cancelled(cancelled) -> None:
        if cancelled():
            raise JobCancelled('job was cancelled')
