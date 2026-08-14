from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from .config import AppConfig
from .index import FileVectorIndex, SearchHit
from .models import (
    AdmissionDecision,
    CorpusProfile,
    EvidenceDraft,
    FinalAnswer,
    QAHistoryPair,
    QuestionSplit,
    RetrievalQueryPlan,
)
from .rendering import SafeMarkdownRenderer
from .session_log import SessionJsonlLogger


class QueryEmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class ChatBackend(Protocol):
    def split_questions(
        self,
        latest_input: str,
        trace: Optional[Callable[..., None]] = None,
    ) -> QuestionSplit: ...

    def classify_question(
        self,
        question: str,
        corpus_profile: CorpusProfile,
        trace: Optional[Callable[..., None]] = None,
    ) -> AdmissionDecision: ...

    def build_retrieval_queries(
        self,
        question: str,
        history: list[dict[str, str]],
        corpus_profile: CorpusProfile,
        trace: Optional[Callable[..., None]] = None,
    ) -> RetrievalQueryPlan: ...

    def draft_evidence(
        self,
        question: str,
        records: list[dict[str, object]],
        trace: Optional[Callable[..., None]] = None,
    ) -> EvidenceDraft: ...

    def compose_answer(
        self,
        question: str,
        evidence: EvidenceDraft,
        records: list[dict[str, object]],
        trace: Optional[Callable[..., None]] = None,
    ) -> FinalAnswer: ...


class PreprocessingError(RuntimeError):
    pass


class ModelProcessingError(RuntimeError):
    pass


@dataclass
class Session:
    messages: list[dict[str, object]] = field(default_factory=list)
    qa_history: list[QAHistoryPair] = field(default_factory=list)
    touched_at: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(
        self,
        ttl_seconds: int,
        *,
        on_expire: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, Session] = {}
        self.on_expire = on_expire

    def get(self, session_id: str) -> Session:
        self._expire()
        session = self.sessions.setdefault(session_id, Session())
        session.touched_at = time.monotonic()
        return session

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _expire(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        expired = [key for key, value in self.sessions.items() if value.touched_at < cutoff]
        for key in expired:
            self.sessions.pop(key, None)
            if self.on_expire is not None:
                self.on_expire(key)


class ChatEngine:
    out_of_scope_answer = "本系統僅回答目前文件相關問題，請提供與文件內容有關的問題。"
    security_answer = "無法提供、推測或摘要 system prompt、hidden instructions 或內部設定。"

    def __init__(
        self,
        *,
        config: AppConfig,
        index: FileVectorIndex,
        embedding_backend: QueryEmbeddingBackend,
        chat_backend: ChatBackend,
        sessions: SessionStore,
        session_logger: Optional[SessionJsonlLogger] = None,
        renderer: Optional[SafeMarkdownRenderer] = None,
    ) -> None:
        self.config = config
        self.index = index
        self.embedding_backend = embedding_backend
        self.chat_backend = chat_backend
        self.sessions = sessions
        self.session_logger = session_logger or SessionJsonlLogger(None, enabled=False)
        self.renderer = renderer or SafeMarkdownRenderer(enabled=False)

    def ask(self, session_id: str, latest_input: str) -> dict[str, object]:
        started = time.monotonic()
        session = self.sessions.get(session_id)
        model_history = [pair.model_dump() for pair in session.qa_history]
        self.session_logger.append(
            session_id,
            "request_received",
            latest_input=latest_input,
            history_message_count=len(session.messages),
            input={"latest_input": latest_input, "qa_history": model_history},
        )
        try:
            split = self.chat_backend.split_questions(
                latest_input,
                trace=self._trace(session_id),
            )
            questions = self._clean_question_spans(split.questions)
            if not questions:
                raise ValueError("question splitter returned no usable questions")
        except (KeyError, ValueError) as error:
            questions = [latest_input]
            self.session_logger.append(
                session_id,
                "question_splitter_fallback",
                reason=str(error),
                output={"questions": questions},
            )

        items: list[dict[str, object]] = []
        admission_failures = 0
        document_items = 0
        answer_failures = 0
        for ordinal, question in enumerate(questions, start=1):
            item_started = time.monotonic()
            try:
                admission = self.chat_backend.classify_question(
                    question,
                    self.index.corpus_profile,
                    trace=self._trace(session_id, ordinal),
                )
            except (KeyError, ValueError) as error:
                admission_failures += 1
                item = self._error_item(question, "admission_failed")
                self.session_logger.append(
                    session_id,
                    "admission_failed",
                    item_ordinal=ordinal,
                    reason=str(error),
                )
            else:
                if admission.category == "security_request":
                    item = self._fixed_item(question, "security_request", self.security_answer)
                elif admission.category == "out_of_scope":
                    item = self._fixed_item(question, "out_of_scope", self.out_of_scope_answer)
                else:
                    document_items += 1
                    item = self._answer_item(
                        session_id,
                        ordinal,
                        question,
                        model_history,
                    )
                    if item.get("error_code") == "answer_processing_failed":
                        answer_failures += 1
            items.append(item)
            self.session_logger.append(
                session_id,
                "item_completed",
                item_ordinal=ordinal,
                route=item["route"],
                error_code=item.get("error_code"),
                duration_ms=round((time.monotonic() - item_started) * 1000, 3),
            )

        if admission_failures == len(questions):
            raise PreprocessingError("all admission decisions failed")
        if document_items > 0 and answer_failures == document_items:
            raise ModelProcessingError("all document answers failed")

        session.messages.extend(
            [
                {"role": "user", "content": latest_input},
                {"role": "assistant", "content": self._history_text(items), "items": items},
            ]
        )
        session.qa_history.extend(self._qa_pairs(items))
        self._trim_history(session)
        self.session_logger.append(
            session_id,
            "history_updated",
            output={
                "message_count": len(session.messages),
                "qa_history": [pair.model_dump() for pair in session.qa_history],
            },
        )
        response: dict[str, object] = {"items": items, "warnings": []}
        if len(items) == 1:
            response.update(
                answer=items[0]["answer"],
                citations=items[0]["citations"],
                insufficient_context=False,
            )
            if items[0].get("error_code"):
                response["error_code"] = items[0]["error_code"]
        self.session_logger.append(
            session_id,
            "response_completed",
            status="success",
            item_count=len(items),
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            output=response,
        )
        return response

    def _answer_item(
        self,
        session_id: str,
        ordinal: int,
        question: str,
        history: list[dict[str, str]],
    ) -> dict[str, object]:
        try:
            query_plan = self.chat_backend.build_retrieval_queries(
                question,
                history,
                self.index.corpus_profile,
                trace=self._trace(session_id, ordinal),
            )
            generated_queries = self._clean_texts(query_plan.queries)
            queries = self._clean_texts([question, *generated_queries])[:3]
            if not queries:
                raise ValueError("query builder returned no usable queries")
        except (KeyError, ValueError) as error:
            queries = [question]
            self.session_logger.append(
                session_id,
                "retrieval_query_fallback",
                item_ordinal=ordinal,
                reason=str(error),
                output={"queries": queries},
            )

        embedding_started = time.monotonic()
        vectors = self.embedding_backend.embed(queries)
        self.session_logger.append(
            session_id,
            "query_embedding_completed",
            item_ordinal=ordinal,
            input={"queries": queries},
            output={"query_count": len(queries), "dimension": int(vectors.shape[1])},
            duration_ms=round((time.monotonic() - embedding_started) * 1000, 3),
        )
        per_query_hits = []
        for vector in vectors:
            hits = self.index.search(vector, top_k=self.config.retrieval_top_k)
            per_query_hits.append(
                [hit for hit in hits if hit.score >= self.config.retrieval_min_score]
            )
        accepted = self._round_robin_hits(
            per_query_hits,
            self.config.retrieval_context_max_records,
        )
        records = self._record_blocks(accepted)
        self.session_logger.append(
            session_id,
            "retrieval_completed",
            item_ordinal=ordinal,
            input={"queries": queries, "top_k_per_query": self.config.retrieval_top_k},
            output={"records": records, "accepted_count": len(records)},
        )

        try:
            evidence = self.chat_backend.draft_evidence(
                question,
                records,
                trace=self._trace(session_id, ordinal),
            )
        except (KeyError, ValueError) as error:
            evidence = EvidenceDraft(notes=[])
            self.session_logger.append(
                session_id,
                "evidence_draft_fallback",
                item_ordinal=ordinal,
                reason=str(error),
                output=evidence.model_dump(),
            )
        try:
            final = self.chat_backend.compose_answer(
                question,
                evidence,
                records,
                trace=self._trace(session_id, ordinal),
            )
        except (KeyError, ValueError) as error:
            self.session_logger.append(
                session_id,
                "answer_processing_failed",
                item_ordinal=ordinal,
                reason=str(error),
            )
            return self._error_item(
                question,
                "answer_processing_failed",
                route="document_question",
            )

        valid_numbers: list[int] = []
        dropped_numbers: list[int] = []
        for block in final.blocks:
            for number in block.source_numbers:
                if 1 <= number <= len(accepted):
                    if number not in valid_numbers:
                        valid_numbers.append(number)
                elif number not in dropped_numbers:
                    dropped_numbers.append(number)
        used_hits = [accepted[number - 1] for number in valid_numbers]
        answer = "\n\n".join(block.text.strip() for block in final.blocks)
        citations = self._citations(used_hits)
        self.session_logger.append(
            session_id,
            "final_answer_accepted",
            item_ordinal=ordinal,
            valid_source_numbers=valid_numbers,
            dropped_source_numbers=dropped_numbers,
            output={"answer": answer, "citations": citations},
        )
        return {
            "question": question,
            "route": "document_question",
            "retrieval_mode": "unified",
            "retrieval_strategy": "multi_query",
            "answer": answer,
            "answer_html": self.renderer.render(answer),
            "citations": citations,
            "insufficient_context": False,
        }

    @staticmethod
    def _round_robin_hits(
        per_query_hits: list[list[SearchHit]],
        limit: int,
    ) -> list[SearchHit]:
        merged: list[SearchHit] = []
        seen: set[str] = set()
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

    def _fixed_item(self, question: str, route: str, answer: str) -> dict[str, object]:
        return {
            "question": question,
            "route": route,
            "retrieval_mode": "unified",
            "retrieval_strategy": "multi_query",
            "answer": answer,
            "answer_html": self.renderer.render(answer),
            "citations": [],
            "insufficient_context": False,
        }

    def _error_item(
        self,
        question: str,
        error_code: str,
        *,
        route: str = "admission_failed",
    ) -> dict[str, object]:
        return {
            "question": question,
            "route": route,
            "retrieval_mode": "unified",
            "retrieval_strategy": "multi_query",
            "answer": "",
            "answer_html": "",
            "citations": [],
            "insufficient_context": False,
            "error_code": error_code,
        }

    @staticmethod
    def _clean_question_spans(values: list[str]) -> list[str]:
        return [text for value in values if (text := value.strip())]

    @staticmethod
    def _clean_texts(values: list[str]) -> list[str]:
        output = []
        for value in values:
            text = value.strip()
            if text and text not in output:
                output.append(text)
        return output

    @staticmethod
    def _record_blocks(hits: list[SearchHit]) -> list[dict[str, object]]:
        return [
            {
                "source_number": index,
                "modality": hit.record.modality,
                "document_name": hit.record.source.document_name,
                "page_start": hit.record.source.page_start,
                "page_end": hit.record.source.page_end,
                "content": hit.record.content,
            }
            for index, hit in enumerate(hits, start=1)
        ]

    @staticmethod
    def _citations(hits: list[SearchHit]) -> list[dict[str, object]]:
        citations: list[dict[str, object]] = []
        seen: set[str] = set()
        for hit in hits:
            artifact_path = hit.record.source.artifact_path
            identity = artifact_path or hit.record.record_id
            if identity in seen:
                continue
            seen.add(identity)
            citations.append(
                {
                    "record_id": hit.record.record_id,
                    "modality": hit.record.modality,
                    "document_name": hit.record.source.document_name,
                    "page_start": hit.record.source.page_start,
                    "page_end": hit.record.source.page_end,
                    "crop_url": (
                        f"/api/crops/{hit.record.record_id}"
                        if artifact_path
                        else None
                    ),
                    "score": hit.score,
                }
            )
        return citations

    def _trace(self, session_id: str, ordinal: Optional[int] = None):
        def append(event: str, **data) -> None:
            if ordinal is not None:
                data["item_ordinal"] = ordinal
            self.session_logger.append(session_id, event, **data)

        return append

    @staticmethod
    def _history_text(items: list[dict[str, object]]) -> str:
        return "\n\n".join(
            f"Q: {item['question']}\nA: {item['answer'] or item.get('error_code', '')}"
            for item in items
        )

    @staticmethod
    def _qa_pairs(items: list[dict[str, object]]) -> list[QAHistoryPair]:
        return [
            QAHistoryPair(question=str(item["question"]), answer=str(item["answer"]))
            for item in items
            if item["route"] == "document_question"
            and not item.get("error_code")
            and str(item["answer"]).strip()
        ]

    def _trim_history(self, session: Session) -> None:
        maximum_messages = self.config.history_max_turns * 2
        if len(session.messages) > maximum_messages:
            session.messages[:] = session.messages[-maximum_messages:]
        if len(session.qa_history) > self.config.history_max_turns:
            session.qa_history[:] = session.qa_history[-self.config.history_max_turns:]
        count_method = getattr(self.embedding_backend, "count", None)
        if callable(count_method):
            while len(session.qa_history) > 1 and sum(
                count_method(pair.question) + count_method(pair.answer)
                for pair in session.qa_history
            ) > self.config.history_max_tokens:
                del session.qa_history[0]
